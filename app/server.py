"""HTTP server for the uncertainty evaluation service (standard library only).

Endpoints
---------
GET  /healthz                    -> 200 {"status": "ok"} once requests are accepted
POST /api/uncertainty/evaluate   -> exact rational uncertainty evaluation
"""
from __future__ import annotations

import json
import os
import sys
import traceback
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from typing import Any, Optional

from .api import evaluate_request
from .errors import ApiError

EVALUATE_PATH = "/api/uncertainty/evaluate"
HEALTH_PATH = "/healthz"
MAX_BODY_BYTES = 1 << 20  # 1 MiB is far more than 24 inputs ever need.
# Bodies only marginally over the cap are drained before the 413 response so
# the verdict reliably reaches clients that are still sending; absurdly large
# declared bodies are rejected and the connection is closed instead.
MAX_DRAIN_BYTES = 4 << 20

_JSON = "application/json"


def _reject_json_constant(value: str) -> Any:
    # json.loads accepts NaN/Infinity/-Infinity by default; forbid them so no
    # non-rational float can slip into a decision.
    raise ValueError("invalid JSON constant %s" % value)


class RequestHandler(BaseHTTPRequestHandler):
    server_version = "UncertaintyService/1.0"
    protocol_version = "HTTP/1.1"

    # -- helpers ---------------------------------------------------------

    def log_message(self, fmt: str, *args: Any) -> None:  # noqa: A003 - stdlib name
        sys.stderr.write("%s - %s\n" % (self.address_string(), fmt % args))

    def _send_json(self, status: int, payload: dict) -> None:
        body = json.dumps(payload).encode("utf-8")
        self.send_response(status)
        self.send_header("Content-Type", _JSON)
        self.send_header("Content-Length", str(len(body)))
        self.end_headers()
        self.wfile.write(body)

    def _send_api_error(self, error: ApiError) -> None:
        self._send_json(error.status, error.payload())

    def _discard_body(self, length: int) -> None:
        """Read and discard ``length`` body bytes so the connection stays
        usable and the error response is not lost under a still-sending
        client."""
        remaining = length
        while remaining > 0:
            chunk = self.rfile.read(min(remaining, 1 << 16))
            if not chunk:
                break
            remaining -= len(chunk)

    def _read_body(self) -> bytes:
        """Read the request body, enforcing the size cap."""
        length_header = self.headers.get("Content-Length")
        if length_header is None:
            raise ApiError("MALFORMED_JSON", "missing Content-Length header")
        try:
            length = int(length_header)
        except ValueError:
            raise ApiError("MALFORMED_JSON", "invalid Content-Length header")
        if length < 0:
            raise ApiError("MALFORMED_JSON", "invalid Content-Length header")
        if length > MAX_BODY_BYTES:
            if length <= MAX_DRAIN_BYTES:
                self._discard_body(length)
            else:
                self.close_connection = True
            raise ApiError("PAYLOAD_TOO_LARGE",
                           "request body exceeds %d bytes" % MAX_BODY_BYTES,
                           status=413)
        return self.rfile.read(length)

    def _decode_json(self, raw: bytes) -> Any:
        try:
            text = raw.decode("utf-8")
        except UnicodeDecodeError as exc:
            raise ApiError("MALFORMED_JSON",
                           "request body is not valid UTF-8") from exc
        try:
            return json.loads(text, parse_constant=_reject_json_constant)
        except ValueError as exc:
            raise ApiError("MALFORMED_JSON",
                           "request body is not valid JSON: %s" % exc) from exc

    # -- routing ---------------------------------------------------------

    def do_GET(self) -> None:
        if self.path == HEALTH_PATH:
            self._send_json(200, {"status": "ok"})
        elif self.path == EVALUATE_PATH:
            self._send_api_error(ApiError(
                "METHOD_NOT_ALLOWED", "use POST for evaluation",
                EVALUATE_PATH, status=405))
        else:
            self._send_api_error(ApiError(
                "NOT_FOUND", "unknown path", self.path, status=404))

    def do_POST(self) -> None:
        try:
            raw = self._read_body()
            if self.path != EVALUATE_PATH:
                raise ApiError("NOT_FOUND", "unknown path", self.path, status=404)
            result = evaluate_request(self._decode_json(raw))
        except ApiError as error:
            self._send_api_error(error)
            return
        except Exception:  # pragma: no cover - defensive; must never mask bugs
            traceback.print_exc()
            self._send_api_error(ApiError(
                "INTERNAL_ERROR", "unexpected server error", status=500))
            return
        self._send_json(200, result)

    def _method_not_allowed(self) -> None:
        self._send_api_error(ApiError(
            "METHOD_NOT_ALLOWED", "method not allowed", self.path, status=405))

    do_PUT = _method_not_allowed
    do_PATCH = _method_not_allowed
    do_DELETE = _method_not_allowed
    do_HEAD = _method_not_allowed


def create_server(port: int, host: str = "0.0.0.0") -> ThreadingHTTPServer:
    return ThreadingHTTPServer((host, port), RequestHandler)


def main() -> None:
    port = int(os.environ.get("API_PORT", "8000"))
    server = create_server(port)
    print("uncertainty service listening on 0.0.0.0:%d" % port, flush=True)
    try:
        server.serve_forever()
    except KeyboardInterrupt:
        pass
    finally:
        server.server_close()


if __name__ == "__main__":
    main()
