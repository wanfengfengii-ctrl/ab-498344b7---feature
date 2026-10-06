import json
import threading
import unittest
import urllib.error
import urllib.request

from app.server import create_server

VALID_BODY = {
    "inputs": [
        {"id": "x1", "value": "3/2", "sensitivity": "2"},
        {"id": "x2", "value": "-1/4", "sensitivity": "1/3"},
    ],
    "covariance": [["1/4", "1/8"], ["1/8", "1/2"]],
    "intercept": "1/2",
    "variance_budget": "2",
}


class ApiTestCase(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.server = create_server(0, "127.0.0.1")
        cls.port = cls.server.server_address[1]
        cls.thread = threading.Thread(target=cls.server.serve_forever, daemon=True)
        cls.thread.start()

    @classmethod
    def tearDownClass(cls):
        cls.server.shutdown()
        cls.server.server_close()

    def url(self, path):
        return "http://127.0.0.1:%d%s" % (self.port, path)

    def post(self, path, payload=None, raw=None):
        data = raw if raw is not None else json.dumps(payload).encode("utf-8")
        request = urllib.request.Request(
            self.url(path), data=data,
            headers={"Content-Type": "application/json"}, method="POST")
        try:
            with urllib.request.urlopen(request, timeout=5) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def get(self, path):
        try:
            with urllib.request.urlopen(self.url(path), timeout=5) as response:
                return response.status, json.loads(response.read())
        except urllib.error.HTTPError as exc:
            return exc.code, json.loads(exc.read())

    def assert_error(self, status, body, code, field):
        self.assertEqual(status, 400)
        error = body["error"]
        self.assertEqual(error["code"], code)
        self.assertEqual(error["field"], field)
        self.assertIsInstance(error["message"], str)
        # An error must never look like a release decision.
        for verdict_key in ("estimate", "variance", "exceeds_budget"):
            self.assertNotIn(verdict_key, body)


class SuccessTests(ApiTestCase):
    def test_valid_request_exact_result(self):
        status, body = self.post("/api/uncertainty/evaluate", VALID_BODY)
        self.assertEqual(status, 200)
        self.assertEqual(body, {
            "estimate": "41/12",
            "variance": "11/9",
            "exceeds_budget": False,
        })

    def test_budget_exceeded(self):
        payload = dict(VALID_BODY, variance_budget="1")
        status, body = self.post("/api/uncertainty/evaluate", payload)
        self.assertEqual(status, 200)
        self.assertEqual(body["variance"], "11/9")
        self.assertIs(body["exceeds_budget"], True)

    def test_budget_equal_to_variance_is_not_exceeded(self):
        payload = dict(VALID_BODY, variance_budget="11/9")
        status, body = self.post("/api/uncertainty/evaluate", payload)
        self.assertEqual(status, 200)
        self.assertIs(body["exceeds_budget"], False)

    def test_single_integer_input(self):
        status, body = self.post("/api/uncertainty/evaluate", {
            "inputs": [{"id": "a", "value": 5, "sensitivity": 1}],
            "covariance": [[4]],
            "intercept": 0,
            "variance_budget": 4,
        })
        self.assertEqual(status, 200)
        self.assertEqual(body, {
            "estimate": "5", "variance": "4", "exceeds_budget": False})

    def test_zero_budget_with_zero_variance(self):
        status, body = self.post("/api/uncertainty/evaluate", {
            "inputs": [{"id": "a", "value": "1/3", "sensitivity": "0"}],
            "covariance": [["7/2"]],
            "intercept": "0",
            "variance_budget": "0",
        })
        self.assertEqual(status, 200)
        self.assertEqual(body["variance"], "0")
        self.assertIs(body["exceeds_budget"], False)

    def test_max_inputs_accepted(self):
        n = 24
        payload = {
            "inputs": [
                {"id": "x%d" % i, "value": "1/%d" % (i + 2), "sensitivity": "1"}
                for i in range(n)
            ],
            "covariance": [["1" if i == j else "0" for j in range(n)]
                           for i in range(n)],
            "intercept": "0",
            "variance_budget": "100",
        }
        status, body = self.post("/api/uncertainty/evaluate", payload)
        self.assertEqual(status, 200)
        self.assertEqual(body["variance"], "24")
        self.assertIs(body["exceeds_budget"], False)


class HealthAndRoutingTests(ApiTestCase):
    def test_health(self):
        status, body = self.get("/healthz")
        self.assertEqual(status, 200)
        self.assertEqual(body, {"status": "ok"})

    def test_get_on_evaluate_is_405(self):
        status, body = self.get("/api/uncertainty/evaluate")
        self.assertEqual(status, 405)
        self.assertEqual(body["error"]["code"], "METHOD_NOT_ALLOWED")

    def test_unknown_path_is_404(self):
        status, body = self.get("/nope")
        self.assertEqual(status, 404)
        self.assertEqual(body["error"]["code"], "NOT_FOUND")
        status, body = self.post("/nope", VALID_BODY)
        self.assertEqual(status, 404)
        self.assertEqual(body["error"]["code"], "NOT_FOUND")


class MalformedBodyTests(ApiTestCase):
    def test_invalid_json(self):
        status, body = self.post("/api/uncertainty/evaluate", raw=b"{not json")
        self.assert_error(status, body, "MALFORMED_JSON", None)

    def test_json_nan_rejected(self):
        status, body = self.post("/api/uncertainty/evaluate", raw=b"NaN")
        self.assert_error(status, body, "MALFORMED_JSON", None)

    def test_body_not_object(self):
        status, body = self.post("/api/uncertainty/evaluate", [1, 2, 3])
        self.assert_error(status, body, "INVALID_SCHEMA", None)

    def test_missing_field(self):
        payload = dict(VALID_BODY)
        del payload["variance_budget"]
        status, body = self.post("/api/uncertainty/evaluate", payload)
        self.assert_error(status, body, "INVALID_SCHEMA", "variance_budget")

    def test_oversized_body(self):
        status, body = self.post(
            "/api/uncertainty/evaluate", raw=b" " * (1 << 20 + 1))
        self.assertEqual(status, 413)
        self.assertEqual(body["error"]["code"], "PAYLOAD_TOO_LARGE")


class InputValidationTests(ApiTestCase):
    def test_zero_inputs(self):
        payload = dict(VALID_BODY, inputs=[], covariance=[])
        status, body = self.post("/api/uncertainty/evaluate", payload)
        self.assert_error(status, body, "INVALID_SCHEMA", "inputs")

    def test_too_many_inputs(self):
        payload = dict(VALID_BODY, inputs=[
            {"id": "x%d" % i, "value": "1", "sensitivity": "1"}
            for i in range(25)])
        status, body = self.post("/api/uncertainty/evaluate", payload)
        self.assert_error(status, body, "INVALID_SCHEMA", "inputs")

    def test_duplicate_ids(self):
        payload = dict(VALID_BODY, inputs=[
            {"id": "x", "value": "1", "sensitivity": "1"},
            {"id": "x", "value": "2", "sensitivity": "1"},
        ])
        status, body = self.post("/api/uncertainty/evaluate", payload)
        self.assert_error(status, body, "INVALID_SCHEMA", "inputs[1].id")

    def test_empty_id(self):
        payload = dict(VALID_BODY, inputs=[
            {"id": "", "value": "1", "sensitivity": "1"}])
        status, body = self.post("/api/uncertainty/evaluate", payload)
        self.assert_error(status, body, "INVALID_SCHEMA", "inputs[0].id")

    def test_missing_input_key(self):
        payload = dict(VALID_BODY, inputs=[{"id": "x", "value": "1"}])
        status, body = self.post("/api/uncertainty/evaluate", payload)
        self.assert_error(status, body, "INVALID_SCHEMA", "inputs[0]")


class RationalValidationTests(ApiTestCase):
    def test_unreduced_fraction(self):
        payload = dict(VALID_BODY)
        payload["inputs"] = [dict(VALID_BODY["inputs"][0], value="2/4"),
                             VALID_BODY["inputs"][1]]
        status, body = self.post("/api/uncertainty/evaluate", payload)
        self.assert_error(status, body, "INVALID_RATIONAL", "inputs[0].value")

    def test_decimal_rejected(self):
        payload = dict(VALID_BODY)
        payload["inputs"] = [dict(VALID_BODY["inputs"][0], value=0.5),
                             VALID_BODY["inputs"][1]]
        status, body = self.post("/api/uncertainty/evaluate", payload)
        self.assert_error(status, body, "INVALID_RATIONAL", "inputs[0].value")

    def test_integral_float_rejected(self):
        payload = dict(VALID_BODY, intercept=2.0)
        status, body = self.post("/api/uncertainty/evaluate", payload)
        self.assert_error(status, body, "INVALID_RATIONAL", "intercept")

    def test_negative_denominator_rejected(self):
        payload = dict(VALID_BODY)
        payload["inputs"] = [dict(VALID_BODY["inputs"][0], sensitivity="1/-2"),
                             VALID_BODY["inputs"][1]]
        status, body = self.post("/api/uncertainty/evaluate", payload)
        self.assert_error(status, body, "INVALID_RATIONAL",
                          "inputs[0].sensitivity")

    def test_bool_rejected(self):
        payload = dict(VALID_BODY, intercept=True)
        status, body = self.post("/api/uncertainty/evaluate", payload)
        self.assert_error(status, body, "INVALID_RATIONAL", "intercept")

    def test_covariance_cell_validated(self):
        payload = dict(VALID_BODY,
                       covariance=[["1/4", "3/9"], ["1/8", "1/2"]])
        status, body = self.post("/api/uncertainty/evaluate", payload)
        self.assert_error(status, body, "INVALID_RATIONAL", "covariance[0][1]")


class BudgetValidationTests(ApiTestCase):
    def test_negative_budget(self):
        payload = dict(VALID_BODY, variance_budget="-1/2")
        status, body = self.post("/api/uncertainty/evaluate", payload)
        self.assert_error(status, body, "INVALID_VARIANCE_BUDGET",
                          "variance_budget")

    def test_malformed_budget(self):
        payload = dict(VALID_BODY, variance_budget="1.5")
        status, body = self.post("/api/uncertainty/evaluate", payload)
        self.assert_error(status, body, "INVALID_RATIONAL", "variance_budget")


class CovarianceValidationTests(ApiTestCase):
    def test_row_count_mismatch(self):
        payload = dict(VALID_BODY, covariance=[["1/4"]])
        status, body = self.post("/api/uncertainty/evaluate", payload)
        self.assert_error(status, body, "COVARIANCE_DIMENSION_MISMATCH",
                          "covariance")

    def test_row_length_mismatch(self):
        payload = dict(VALID_BODY, covariance=[["1/4", "1/8"], ["1/8"]])
        status, body = self.post("/api/uncertainty/evaluate", payload)
        self.assert_error(status, body, "COVARIANCE_DIMENSION_MISMATCH",
                          "covariance[1]")

    def test_not_symmetric(self):
        payload = dict(VALID_BODY,
                       covariance=[["1/4", "1/8"], ["1/6", "1/2"]])
        status, body = self.post("/api/uncertainty/evaluate", payload)
        self.assert_error(status, body, "COVARIANCE_NOT_SYMMETRIC",
                          "covariance[0][1]")

    def test_not_positive_semidefinite(self):
        payload = dict(VALID_BODY, covariance=[["1", "2"], ["2", "1"]])
        status, body = self.post("/api/uncertainty/evaluate", payload)
        self.assert_error(status, body, "COVARIANCE_NOT_POSITIVE_SEMIDEFINITE",
                          "covariance[1][1]")

    def test_indefinite_zero_pivot(self):
        payload = dict(VALID_BODY, covariance=[["0", "1"], ["1", "0"]])
        status, body = self.post("/api/uncertainty/evaluate", payload)
        self.assert_error(status, body, "COVARIANCE_NOT_POSITIVE_SEMIDEFINITE",
                          "covariance[0][0]")


if __name__ == "__main__":
    unittest.main()
