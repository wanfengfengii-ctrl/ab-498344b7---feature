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


class ConditioningSuccessTests(ApiTestCase):
    CONDITIONING = {
        "observations": [
            {"id": "r1", "value": "2", "coefficients": {"x1": "1"}},
            {"id": "r2", "value": "0", "coefficients": {"x2": "1"}},
        ],
        "noise_covariance": [["1", "0"], ["0", "1"]],
    }

    def test_conditional_result(self):
        status, body = self.post(
            "/api/uncertainty/evaluate",
            dict(VALID_BODY, conditioning=self.CONDITIONING))
        self.assertEqual(status, 200)
        self.assertEqual(body, {
            "estimate": "2629/714",
            "variance": "107/119",
            "exceeds_budget": False,
        })

    def test_conditional_budget_verdict_uses_conditional_variance(self):
        payload = dict(VALID_BODY, conditioning=self.CONDITIONING,
                       variance_budget="1/2")
        status, body = self.post("/api/uncertainty/evaluate", payload)
        self.assertEqual(status, 200)
        self.assertEqual(body["variance"], "107/119")
        self.assertIs(body["exceeds_budget"], True)

    def test_zero_noise_pins_observation_exactly(self):
        conditioning = {
            "observations": [
                {"id": "r1", "value": "2", "coefficients": {"x1": "1"}}],
            "noise_covariance": [["0"]],
        }
        status, body = self.post(
            "/api/uncertainty/evaluate",
            dict(VALID_BODY, conditioning=conditioning))
        self.assertEqual(status, 200)
        # A Sigma = [1/4, 1/8]; S = 1/4; gain on x1 = 1 (zero noise).
        # E[x2|r1] = -1/4 + (1/8)/(1/4)*(2 - 3/2) = -1/4 + (1/2)*(1/2) = 0
        # estimate = 1/2 + 2*2 + 1/3*0 = 9/2
        self.assertEqual(body["estimate"], "9/2")
        # variance: prior 11/9; reduction (A Sigma c)^2/S with
        # A Sigma c = 2*1/4 + 1/3*1/8 = 13/24, squared / (1/4) = 169/144
        # 11/9 - 169/144 = 7/144
        self.assertEqual(body["variance"], "7/144")
        self.assertIs(body["exceeds_budget"], False)

    def test_null_conditioning_is_ignored(self):
        status, body = self.post(
            "/api/uncertainty/evaluate",
            dict(VALID_BODY, conditioning=None))
        self.assertEqual(status, 200)
        self.assertEqual(body["estimate"], "41/12")
        self.assertEqual(body["variance"], "11/9")

    def test_four_observations_accepted(self):
        conditioning = {
            "observations": [
                {"id": "r%d" % r, "value": "1",
                 "coefficients": {"x1": "1", "x2": "1/2"}}
                for r in range(4)],
            "noise_covariance": [
                ["1" if i == j else "0" for j in range(4)]
                for i in range(4)],
        }
        status, body = self.post(
            "/api/uncertainty/evaluate",
            dict(VALID_BODY, conditioning=conditioning))
        self.assertEqual(status, 200)
        for key in ("estimate", "variance", "exceeds_budget"):
            self.assertIn(key, body)


class ConditioningValidationTests(ApiTestCase):
    def test_conditioning_not_object(self):
        status, body = self.post(
            "/api/uncertainty/evaluate",
            dict(VALID_BODY, conditioning=[]))
        self.assert_error(status, body, "INVALID_SCHEMA", "conditioning")

    def test_missing_observations(self):
        status, body = self.post(
            "/api/uncertainty/evaluate",
            dict(VALID_BODY,
                 conditioning={"noise_covariance": [["1"]]}))
        self.assert_error(status, body, "INVALID_SCHEMA",
                          "conditioning.observations")

    def test_missing_noise_covariance(self):
        status, body = self.post(
            "/api/uncertainty/evaluate",
            dict(VALID_BODY, conditioning={"observations": [
                {"id": "r1", "value": "1", "coefficients": {"x1": "1"}}]}))
        self.assert_error(status, body, "INVALID_SCHEMA",
                          "conditioning.noise_covariance")

    def test_zero_observations(self):
        status, body = self.post(
            "/api/uncertainty/evaluate",
            dict(VALID_BODY, conditioning={
                "observations": [], "noise_covariance": []}))
        self.assert_error(status, body, "INVALID_SCHEMA",
                          "conditioning.observations")

    def test_five_observations(self):
        obs = [{"id": "r%d" % i, "value": "1",
                "coefficients": {"x1": "1"}} for i in range(5)]
        status, body = self.post(
            "/api/uncertainty/evaluate",
            dict(VALID_BODY, conditioning={
                "observations": obs,
                "noise_covariance": [["1"] * 5] * 5}))
        self.assert_error(status, body, "INVALID_SCHEMA",
                          "conditioning.observations")

    def test_observations_not_array(self):
        status, body = self.post(
            "/api/uncertainty/evaluate",
            dict(VALID_BODY, conditioning={
                "observations": {}, "noise_covariance": []}))
        self.assert_error(status, body, "INVALID_SCHEMA",
                          "conditioning.observations")

    def test_observation_not_object(self):
        status, body = self.post(
            "/api/uncertainty/evaluate",
            dict(VALID_BODY, conditioning={
                "observations": [5], "noise_covariance": [["1"]]}))
        self.assert_error(status, body, "INVALID_SCHEMA",
                          "conditioning.observations[0]")

    def test_observation_missing_key(self):
        status, body = self.post(
            "/api/uncertainty/evaluate",
            dict(VALID_BODY, conditioning={
                "observations": [{"id": "r1", "value": "1"}],
                "noise_covariance": [["1"]]}))
        self.assert_error(status, body, "INVALID_SCHEMA",
                          "conditioning.observations[0]")

    def test_duplicate_observation_ids(self):
        status, body = self.post(
            "/api/uncertainty/evaluate",
            dict(VALID_BODY, conditioning={
                "observations": [
                    {"id": "r", "value": "1", "coefficients": {"x1": "1"}},
                    {"id": "r", "value": "2", "coefficients": {"x2": "1"}}],
                "noise_covariance": [["1", "0"], ["0", "1"]]}))
        self.assert_error(status, body, "INVALID_SCHEMA",
                          "conditioning.observations[1].id")

    def test_empty_observation_id(self):
        status, body = self.post(
            "/api/uncertainty/evaluate",
            dict(VALID_BODY, conditioning={
                "observations": [
                    {"id": "", "value": "1", "coefficients": {"x1": "1"}}],
                "noise_covariance": [["1"]]}))
        self.assert_error(status, body, "INVALID_SCHEMA",
                          "conditioning.observations[0].id")

    def test_invalid_observation_value(self):
        status, body = self.post(
            "/api/uncertainty/evaluate",
            dict(VALID_BODY, conditioning={
                "observations": [
                    {"id": "r1", "value": "1.5",
                     "coefficients": {"x1": "1"}}],
                "noise_covariance": [["1"]]}))
        self.assert_error(status, body, "INVALID_RATIONAL",
                          "conditioning.observations[0].value")

    def test_coefficients_not_object(self):
        status, body = self.post(
            "/api/uncertainty/evaluate",
            dict(VALID_BODY, conditioning={
                "observations": [
                    {"id": "r1", "value": "1", "coefficients": [1]}],
                "noise_covariance": [["1"]]}))
        self.assert_error(status, body, "INVALID_SCHEMA",
                          "conditioning.observations[0].coefficients")

    def test_coefficients_empty(self):
        status, body = self.post(
            "/api/uncertainty/evaluate",
            dict(VALID_BODY, conditioning={
                "observations": [
                    {"id": "r1", "value": "1", "coefficients": {}}],
                "noise_covariance": [["1"]]}))
        self.assert_error(status, body, "INVALID_SCHEMA",
                          "conditioning.observations[0].coefficients")

    def test_coefficient_unknown_input(self):
        status, body = self.post(
            "/api/uncertainty/evaluate",
            dict(VALID_BODY, conditioning={
                "observations": [
                    {"id": "r1", "value": "1",
                     "coefficients": {"nope": "1"}}],
                "noise_covariance": [["1"]]}))
        self.assert_error(status, body, "UNKNOWN_INPUT_REFERENCE",
                          "conditioning.observations[0].coefficients.nope")

    def test_coefficient_invalid_rational(self):
        status, body = self.post(
            "/api/uncertainty/evaluate",
            dict(VALID_BODY, conditioning={
                "observations": [
                    {"id": "r1", "value": "1",
                     "coefficients": {"x1": "2/4"}}],
                "noise_covariance": [["1"]]}))
        self.assert_error(status, body, "INVALID_RATIONAL",
                          "conditioning.observations[0].coefficients.x1")

    def test_noise_dimension_mismatch_rows(self):
        status, body = self.post(
            "/api/uncertainty/evaluate",
            dict(VALID_BODY, conditioning={
                "observations": [
                    {"id": "r1", "value": "1", "coefficients": {"x1": "1"}},
                    {"id": "r2", "value": "2", "coefficients": {"x2": "1"}}],
                "noise_covariance": [["1"]]}))
        self.assert_error(
            status, body, "NOISE_COVARIANCE_DIMENSION_MISMATCH",
            "conditioning.noise_covariance")

    def test_noise_dimension_mismatch_row_length(self):
        status, body = self.post(
            "/api/uncertainty/evaluate",
            dict(VALID_BODY, conditioning={
                "observations": [
                    {"id": "r1", "value": "1", "coefficients": {"x1": "1"}},
                    {"id": "r2", "value": "2", "coefficients": {"x2": "1"}}],
                "noise_covariance": [["1", "0"], ["0"]]}))
        self.assert_error(
            status, body, "NOISE_COVARIANCE_DIMENSION_MISMATCH",
            "conditioning.noise_covariance[1]")

    def test_noise_not_symmetric(self):
        status, body = self.post(
            "/api/uncertainty/evaluate",
            dict(VALID_BODY, conditioning={
                "observations": [
                    {"id": "r1", "value": "1", "coefficients": {"x1": "1"}},
                    {"id": "r2", "value": "2", "coefficients": {"x2": "1"}}],
                "noise_covariance": [["1", "1/2"], ["1/3", "1"]]}))
        self.assert_error(
            status, body, "NOISE_COVARIANCE_NOT_SYMMETRIC",
            "conditioning.noise_covariance[0][1]")

    def test_noise_not_psd(self):
        status, body = self.post(
            "/api/uncertainty/evaluate",
            dict(VALID_BODY, conditioning={
                "observations": [
                    {"id": "r1", "value": "1", "coefficients": {"x1": "1"}},
                    {"id": "r2", "value": "2", "coefficients": {"x2": "1"}}],
                "noise_covariance": [["1", "2"], ["2", "1"]]}))
        self.assert_error(
            status, body, "NOISE_COVARIANCE_NOT_POSITIVE_SEMIDEFINITE",
            "conditioning.noise_covariance[1][1]")

    def test_noise_bad_rational(self):
        status, body = self.post(
            "/api/uncertainty/evaluate",
            dict(VALID_BODY, conditioning={
                "observations": [
                    {"id": "r1", "value": "1", "coefficients": {"x1": "1"}}],
                "noise_covariance": [["1/0"]]}))
        self.assert_error(
            status, body, "INVALID_RATIONAL",
            "conditioning.noise_covariance[0][0]")

    def test_degenerate_zero_noise_on_unobserved_direction_rejected(self):
        # Sigma = [[0,0],[0,1/2]] (valid PSD); observing x1 with zero noise
        # gives S = [0] -> the reference carries no unique information.
        payload = dict(VALID_BODY, covariance=[["0", "0"], ["0", "1/2"]],
                       conditioning={
            "observations": [
                {"id": "r1", "value": "1", "coefficients": {"x1": "1"}}],
            "noise_covariance": [["0"]]})
        status, body = self.post("/api/uncertainty/evaluate", payload)
        self.assert_error(status, body, "REFERENCE_INFORMATION_NOT_UNIQUE",
                          "conditioning.observations")

    def test_degenerate_duplicate_noiseless_observation_rejected(self):
        # Two observations of the exact same linear form, both noiseless:
        # S = A Sigma A^T * [[1,1],[1,1]] is singular.
        payload = dict(VALID_BODY, conditioning={
            "observations": [
                {"id": "r1", "value": "2", "coefficients": {"x1": "1"}},
                {"id": "r2", "value": "3", "coefficients": {"x1": "1"}}],
            "noise_covariance": [["0", "0"], ["0", "0"]]})
        status, body = self.post("/api/uncertainty/evaluate", payload)
        self.assert_error(status, body, "REFERENCE_INFORMATION_NOT_UNIQUE",
                          "conditioning.observations")

    def test_degenerate_conflicting_but_distinguishable_by_noise_ok(self):
        # Same form twice but independent positive noise makes S definite,
        # even though the readings conflict.
        payload = dict(VALID_BODY, conditioning={
            "observations": [
                {"id": "r1", "value": "2", "coefficients": {"x1": "1"}},
                {"id": "r2", "value": "3", "coefficients": {"x1": "1"}}],
            "noise_covariance": [["1", "0"], ["0", "1"]]})
        status, body = self.post("/api/uncertainty/evaluate", payload)
        self.assertEqual(status, 200)
        self.assertNotIn("error", body)


if __name__ == "__main__":
    unittest.main()
