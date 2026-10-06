#!/usr/bin/env python3
"""One-shot verification service.

Runs, in order:
  1. code tests      -- the unit test suite
  2. application build -- byte-compile + import check of the application
  3. HTTP smoke      -- compatible requests, valid conditioning, degenerate
                        reference combinations, and valid/invalid covariance
                        requests against the already-healthy API container

Every step is recorded; the process exits 0 only if all steps pass.
"""
from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
import sys
import tempfile
import time
import urllib.error
import urllib.request

APP_ROOT = os.environ.get("APP_ROOT", "/app")
API_BASE_URL = os.environ.get("API_BASE_URL", "http://api:8000").rstrip("/")
EVALUATE_URL = API_BASE_URL + "/api/uncertainty/evaluate"
HEALTH_URL = API_BASE_URL + "/healthz"
HEALTH_TIMEOUT_SECONDS = float(os.environ.get("VERIFY_HEALTH_TIMEOUT", "60"))

RESULTS = []  # list of (name, ok, detail)


def record(name: str, ok: bool, detail: str = "") -> None:
    RESULTS.append((name, ok, detail))
    line = "[%s] %s" % ("PASS" if ok else "FAIL", name)
    if detail and not ok:
        line += "\n       " + detail.replace("\n", "\n       ")
    print(line, flush=True)


def run_step_tests(source_dir: str) -> bool:
    proc = subprocess.run(
        [sys.executable, "-m", "unittest", "discover", "-v"],
        cwd=source_dir, capture_output=True, text=True)
    if proc.returncode != 0:
        record("unit tests", False,
               (proc.stdout + proc.stderr).strip()[-3000:])
        return False
    match = re.search(r"^Ran (\d+) tests", proc.stderr, re.MULTILINE)
    count = match.group(1) if match else "all"
    record("unit tests", True)
    print("       %s tests passed" % count, flush=True)
    return True


def run_step_build(source_dir: str) -> bool:
    compile_proc = subprocess.run(
        [sys.executable, "-m", "compileall", "-q", "app", "tests", "verify"],
        cwd=source_dir, capture_output=True, text=True)
    if compile_proc.returncode != 0:
        record("application build (compileall)", False,
               compile_proc.stderr.strip())
        return False
    record("application build (compileall)", True)

    import_proc = subprocess.run(
        [sys.executable, "-c",
         "import app.api, app.errors, app.matrix, app.rational, "
         "app.server, app.service"],
        cwd=source_dir, capture_output=True, text=True)
    if import_proc.returncode != 0:
        record("application build (import check)", False,
               import_proc.stderr.strip())
        return False
    record("application build (import check)", True)
    return True


def wait_for_health() -> bool:
    deadline = time.monotonic() + HEALTH_TIMEOUT_SECONDS
    while time.monotonic() < deadline:
        try:
            with urllib.request.urlopen(HEALTH_URL, timeout=3) as response:
                if response.status == 200:
                    record("api health check", True)
                    return True
        except (urllib.error.URLError, OSError):
            pass
        time.sleep(1)
    record("api health check", False,
           "api did not become healthy within %ds" % HEALTH_TIMEOUT_SECONDS)
    return False


def http_post(payload=None, raw=None):
    data = raw if raw is not None else json.dumps(payload).encode("utf-8")
    request = urllib.request.Request(
        EVALUATE_URL, data=data,
        headers={"Content-Type": "application/json"}, method="POST")
    try:
        with urllib.request.urlopen(request, timeout=10) as response:
            return response.status, json.loads(response.read())
    except urllib.error.HTTPError as exc:
        return exc.code, json.loads(exc.read())


VALID_PAYLOAD = {
    "inputs": [
        {"id": "x1", "value": "3/2", "sensitivity": "2"},
        {"id": "x2", "value": "-1/4", "sensitivity": "1/3"},
    ],
    "covariance": [["1/4", "1/8"], ["1/8", "1/2"]],
    "intercept": "1/2",
    "variance_budget": "2",
}

VERDICT_KEYS = ("estimate", "variance", "exceeds_budget")


def check_success(name, payload, expected):
    status, body = http_post(payload)
    if status != 200:
        record(name, False, "expected 200, got %d: %s" % (status, body))
        return False
    if body != expected:
        record(name, False, "expected %s, got %s" % (expected, body))
        return False
    record(name, True)
    return True


def check_error(name, payload, expected_code, expected_field, raw=None):
    status, body = http_post(payload, raw=raw)
    if status != 400:
        record(name, False, "expected 400, got %d: %s" % (status, body))
        return False
    error = body.get("error")
    if not isinstance(error, dict):
        record(name, False, "missing 'error' object in %s" % body)
        return False
    if error.get("code") != expected_code:
        record(name, False, "expected code %s, got %s"
               % (expected_code, error.get("code")))
        return False
    if error.get("field") != expected_field:
        record(name, False, "expected field %s, got %s"
               % (expected_field, error.get("field")))
        return False
    leaked = [key for key in VERDICT_KEYS if key in body]
    if leaked:
        record(name, False, "error response leaks verdict keys %s" % leaked)
        return False
    record(name, True)
    return True


def run_step_smoke() -> bool:
    ok = True

    # -- valid covariance: exact rational verdicts -------------------------
    ok &= check_success(
        "smoke: valid covariance, within budget",
        VALID_PAYLOAD,
        {"estimate": "41/12", "variance": "11/9", "exceeds_budget": False})
    ok &= check_success(
        "smoke: valid covariance, budget exceeded",
        dict(VALID_PAYLOAD, variance_budget="1"),
        {"estimate": "41/12", "variance": "11/9", "exceeds_budget": True})
    ok &= check_success(
        "smoke: valid covariance, budget exactly equal (not exceeded)",
        dict(VALID_PAYLOAD, variance_budget="11/9"),
        {"estimate": "41/12", "variance": "11/9", "exceeds_budget": False})

    # -- invalid covariance -------------------------------------------------
    ok &= check_error(
        "smoke: non-symmetric covariance rejected",
        dict(VALID_PAYLOAD, covariance=[["1/4", "1/8"], ["1/6", "1/2"]]),
        "COVARIANCE_NOT_SYMMETRIC", "covariance[0][1]")
    ok &= check_error(
        "smoke: non-PSD covariance rejected",
        dict(VALID_PAYLOAD, covariance=[["1", "2"], ["2", "1"]]),
        "COVARIANCE_NOT_POSITIVE_SEMIDEFINITE", "covariance[1][1]")
    ok &= check_error(
        "smoke: covariance dimension mismatch rejected",
        dict(VALID_PAYLOAD, covariance=[["1/4"]]),
        "COVARIANCE_DIMENSION_MISMATCH", "covariance")

    # -- invalid rationals / budget -----------------------------------------
    ok &= check_error(
        "smoke: unreduced rational rejected",
        dict(VALID_PAYLOAD, inputs=[
            {"id": "x1", "value": "2/4", "sensitivity": "2"},
            VALID_PAYLOAD["inputs"][1]]),
        "INVALID_RATIONAL", "inputs[0].value")
    ok &= check_error(
        "smoke: decimal rational rejected",
        dict(VALID_PAYLOAD, inputs=[
            {"id": "x1", "value": 0.5, "sensitivity": "2"},
            VALID_PAYLOAD["inputs"][1]]),
        "INVALID_RATIONAL", "inputs[0].value")
    ok &= check_error(
        "smoke: negative variance budget rejected",
        dict(VALID_PAYLOAD, variance_budget="-1/2"),
        "INVALID_VARIANCE_BUDGET", "variance_budget")
    ok &= check_error(
        "smoke: malformed JSON rejected",
        None, "MALFORMED_JSON", None, raw=b"{not json")

    # -- conditioning -------------------------------------------------------
    conditioned = dict(VALID_PAYLOAD, conditioning={
        "observations": [
            {"id": "r1", "value": "2", "coefficients": {"x1": "1"}},
            {"id": "r2", "value": "0", "coefficients": {"x2": "1"}},
        ],
        "noise_covariance": [["1", "0"], ["0", "1"]]})
    ok &= check_success(
        "smoke: compatible request (no conditioning) unchanged",
        VALID_PAYLOAD,
        {"estimate": "41/12", "variance": "11/9", "exceeds_budget": False})
    ok &= check_success(
        "smoke: valid conditioning updates estimate and variance",
        conditioned,
        {"estimate": "2629/714", "variance": "107/119", "exceeds_budget": False})
    ok &= check_success(
        "smoke: conditional verdict uses conditional variance",
        dict(conditioned, variance_budget="1/2"),
        {"estimate": "2629/714", "variance": "107/119", "exceeds_budget": True})
    ok &= check_error(
        "smoke: degenerate reference combination rejected",
        dict(VALID_PAYLOAD, conditioning={
            "observations": [
                {"id": "r1", "value": "2", "coefficients": {"x1": "1"}},
                {"id": "r2", "value": "3", "coefficients": {"x1": "1"}}],
            "noise_covariance": [["0", "0"], ["0", "0"]]}),
        "REFERENCE_INFORMATION_NOT_UNIQUE", "conditioning.observations")
    ok &= check_error(
        "smoke: conditioning coefficient referencing unknown input rejected",
        dict(VALID_PAYLOAD, conditioning={
            "observations": [
                {"id": "r1", "value": "1", "coefficients": {"nope": "1"}}],
            "noise_covariance": [["1"]]}),
        "UNKNOWN_INPUT_REFERENCE",
        "conditioning.observations[0].coefficients.nope")
    ok &= check_error(
        "smoke: non-PSD reference noise covariance rejected",
        dict(VALID_PAYLOAD, conditioning={
            "observations": [
                {"id": "r1", "value": "1", "coefficients": {"x1": "1"}},
                {"id": "r2", "value": "2", "coefficients": {"x2": "1"}}],
            "noise_covariance": [["1", "2"], ["2", "1"]]}),
        "NOISE_COVARIANCE_NOT_POSITIVE_SEMIDEFINITE",
        "conditioning.noise_covariance[1][1]")

    return ok


def main() -> int:
    print("== verify: waiting for api health at %s ==" % HEALTH_URL, flush=True)
    healthy = wait_for_health()

    print("== verify: code tests and application build ==", flush=True)
    build_dir = tempfile.mkdtemp(prefix="verify-src-")
    try:
        for entry in ("app", "tests", "verify"):
            shutil.copytree(os.path.join(APP_ROOT, entry),
                            os.path.join(build_dir, entry))
        tests_ok = run_step_tests(build_dir)
        build_ok = run_step_build(build_dir)
    finally:
        shutil.rmtree(build_dir, ignore_errors=True)

    print("== verify: HTTP smoke against %s ==" % EVALUATE_URL, flush=True)
    smoke_ok = run_step_smoke() if healthy else False
    if not healthy:
        record("HTTP smoke", False, "skipped: api never became healthy")

    passed = sum(1 for _, ok, _ in RESULTS if ok)
    failed = len(RESULTS) - passed
    print("== verify summary: %d passed, %d failed ==" % (passed, failed),
          flush=True)
    return 0 if failed == 0 else 1


if __name__ == "__main__":
    sys.exit(main())
