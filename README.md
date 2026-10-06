# Uncertainty Evaluation Service

Exact, rational-arithmetic uncertainty propagation for metrology workflows.
Given 1–24 input quantities (measurement value + sensitivity), a covariance
matrix of the same order, an intercept, and a non-negative variance budget,
the service computes

```
estimate = intercept + Σᵢ cᵢ·xᵢ
variance = cᵀ Σ c
```

**All arithmetic is exact** (`fractions.Fraction`). No floating point value is
ever produced on the decision path, so the budget verdict cannot be shifted
by rounding or tolerance tricks.

## API

### `POST /api/uncertainty/evaluate`

```json
{
  "inputs": [
    {"id": "x1", "value": "3/2",  "sensitivity": "2"},
    {"id": "x2", "value": "-1/4", "sensitivity": "1/3"}
  ],
  "covariance": [["1/4", "1/8"], ["1/8", "1/2"]],
  "intercept": "1/2",
  "variance_budget": "2"
}
```

* `inputs` — array of **1 to 24** entries, evaluated in order. Each entry has
  a unique non-empty string `id`, a rational `value` and a rational
  `sensitivity`.
* `covariance` — `n × n` array matching the input order. Must be exactly
  symmetric and positive semidefinite (checked exactly via LDLᵀ).
* `intercept` — rational.
* `variance_budget` — rational, must be **≥ 0**.

**Rational format** (applies everywhere): a JSON integer (`3`, `"-7"`) or a
string `"p/q"` in **lowest terms** with a **positive** denominator
(`"3/4"`, `"5/1"`). Decimals, floats, booleans, unreduced fractions
(`"2/4"`) and non-positive denominators (`"1/0"`, `"1/-2"`) are rejected.

### Success — `200`

```json
{"estimate": "41/12", "variance": "11/9", "exceeds_budget": false}
```

`estimate` and `variance` are rendered in canonical reduced form;
`exceeds_budget` is `true` iff `variance > variance_budget` (exact
comparison; equality is *within* budget).

### Errors — `400` (also `404`/`405`/`413`)

```json
{"error": {"code": "COVARIANCE_NOT_SYMMETRIC",
           "message": "covariance[0][1] = 1/8 differs from covariance[1][0] = 1/6",
           "field": "covariance[0][1]"}}
```

Error responses never contain `estimate`/`variance`/`exceeds_budget`, so a
failed request can never be mistaken for a release decision.

| Code | Meaning |
| --- | --- |
| `MALFORMED_JSON` | Body is not valid JSON (incl. `NaN`/`Infinity`) |
| `INVALID_SCHEMA` | Structural problem (missing field, wrong type, input count outside 1–24, duplicate/empty id) |
| `INVALID_RATIONAL` | Not an integer or reduced `p/q` with positive denominator |
| `INVALID_VARIANCE_BUDGET` | Budget is negative |
| `COVARIANCE_DIMENSION_MISMATCH` | Matrix is not `n × n` for `n` inputs |
| `COVARIANCE_NOT_SYMMETRIC` | `covariance[i][j] ≠ covariance[j][i]` |
| `COVARIANCE_NOT_POSITIVE_SEMIDEFINITE` | Exact LDLᵀ check failed at the reported pivot |
| `PAYLOAD_TOO_LARGE` | Body exceeds 1 MiB (HTTP 413) |
| `METHOD_NOT_ALLOWED` | Wrong HTTP method (HTTP 405) |
| `NOT_FOUND` | Unknown path (HTTP 404) |
| `INTERNAL_ERROR` | Unexpected server fault (HTTP 500) |

### `GET /healthz`

Returns `200 {"status": "ok"}` once the server accepts requests; used by the
Compose health check.

## Run with Docker Compose

```sh
docker compose up --build api          # serves on http://localhost:8000
API_PORT=9000 docker compose up --build api   # host port is configurable
```

The `api` service health check polls `/healthz` until the container can
receive requests.

## One-shot verification

The `verify` service waits for `api` to be healthy, then runs the unit
tests, the application build (byte-compile + import check) and HTTP smoke
tests covering valid and invalid covariance matrices. It prints a summary
and exits with code 0 on success, 1 on failure:

```sh
docker compose up --build --exit-code-from verify verify
docker compose down
```

## Local development (no dependencies, Python ≥ 3.10)

```sh
python3 -m unittest discover -v        # tests
API_PORT=8000 python3 -m app.server    # run
```

## Layout

```
app/
  rational.py   strict rational parsing (integer or reduced p/q, q > 0)
  matrix.py     exact symmetry check + exact PSD test (LDLᵀ over Fraction)
  service.py    estimate / variance propagation (exact)
  api.py        request validation -> stable error codes + field paths
  server.py     stdlib HTTP server (threaded), /healthz + evaluate
tests/          unit + in-process API tests
verify/         one-shot Compose verification (tests, build, HTTP smoke)
```
