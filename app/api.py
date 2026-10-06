"""Request validation and evaluation orchestration for the evaluate endpoint.

Validation is deliberately strict and fully deterministic: the first problem
found is reported with a stable error code and a locatable field path, and no
error response ever contains an estimate or a budget verdict.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Any, List

from .errors import ApiError
from .matrix import check_symmetric, psd_failure_index
from .rational import RationalFormatError, format_rational, parse_rational
from .service import propagate

MIN_INPUTS = 1
MAX_INPUTS = 24

_REQUIRED_TOP_LEVEL = ("inputs", "covariance", "intercept", "variance_budget")
_REQUIRED_INPUT_KEYS = ("id", "value", "sensitivity")


def _parse_field(node: Any, field: str) -> Fraction:
    try:
        return parse_rational(node)
    except RationalFormatError as exc:
        raise ApiError("INVALID_RATIONAL",
                       "invalid rational: %s" % exc.reason, field) from exc


def _parse_inputs(node: Any) -> tuple:
    if not isinstance(node, list):
        raise ApiError("INVALID_SCHEMA", "'inputs' must be an array", "inputs")
    if not MIN_INPUTS <= len(node) <= MAX_INPUTS:
        raise ApiError(
            "INVALID_SCHEMA",
            "'inputs' must contain between %d and %d entries, got %d"
            % (MIN_INPUTS, MAX_INPUTS, len(node)),
            "inputs",
        )
    seen_ids = set()
    values: List[Fraction] = []
    sensitivities: List[Fraction] = []
    for i, item in enumerate(node):
        base = "inputs[%d]" % i
        if not isinstance(item, dict):
            raise ApiError("INVALID_SCHEMA", "input entry must be an object", base)
        for key in _REQUIRED_INPUT_KEYS:
            if key not in item:
                raise ApiError("INVALID_SCHEMA",
                               "input entry is missing '%s'" % key, base)
        input_id = item["id"]
        if not isinstance(input_id, str) or not input_id:
            raise ApiError("INVALID_SCHEMA",
                           "input id must be a non-empty string", base + ".id")
        if input_id in seen_ids:
            raise ApiError("INVALID_SCHEMA",
                           "duplicate input id '%s'" % input_id, base + ".id")
        seen_ids.add(input_id)
        values.append(_parse_field(item["value"], base + ".value"))
        sensitivities.append(_parse_field(item["sensitivity"], base + ".sensitivity"))
    return values, sensitivities


def _parse_covariance(node: Any, n: int) -> List[List[Fraction]]:
    if not isinstance(node, list):
        raise ApiError("INVALID_SCHEMA",
                       "'covariance' must be an array of arrays", "covariance")
    if len(node) != n:
        raise ApiError(
            "COVARIANCE_DIMENSION_MISMATCH",
            "covariance has %d rows but there are %d inputs" % (len(node), n),
            "covariance",
        )
    matrix: List[List[Fraction]] = []
    for i, row in enumerate(node):
        row_field = "covariance[%d]" % i
        if not isinstance(row, list):
            raise ApiError("INVALID_SCHEMA",
                           "covariance row must be an array", row_field)
        if len(row) != n:
            raise ApiError(
                "COVARIANCE_DIMENSION_MISMATCH",
                "covariance row %d has %d entries, expected %d" % (i, len(row), n),
                row_field,
            )
        matrix.append([
            _parse_field(cell, "covariance[%d][%d]" % (i, j))
            for j, cell in enumerate(row)
        ])
    return matrix


def evaluate_request(obj: Any) -> dict:
    """Validate the decoded JSON body and return the success payload.

    Raises :class:`ApiError` on any validation failure.
    """
    if not isinstance(obj, dict):
        raise ApiError("INVALID_SCHEMA", "request body must be a JSON object", None)
    for key in _REQUIRED_TOP_LEVEL:
        if key not in obj:
            raise ApiError("INVALID_SCHEMA",
                           "missing required field '%s'" % key, key)

    values, sensitivities = _parse_inputs(obj["inputs"])
    n = len(values)

    intercept = _parse_field(obj["intercept"], "intercept")

    budget = _parse_field(obj["variance_budget"], "variance_budget")
    if budget < 0:
        raise ApiError("INVALID_VARIANCE_BUDGET",
                       "variance budget must be non-negative",
                       "variance_budget")

    matrix = _parse_covariance(obj["covariance"], n)

    asymmetric = check_symmetric(matrix)
    if asymmetric is not None:
        i, j = asymmetric
        raise ApiError(
            "COVARIANCE_NOT_SYMMETRIC",
            "covariance[%d][%d] = %s differs from covariance[%d][%d] = %s"
            % (i, j, matrix[i][j], j, i, matrix[j][i]),
            "covariance[%d][%d]" % (i, j),
        )

    pivot = psd_failure_index(matrix)
    if pivot is not None:
        raise ApiError(
            "COVARIANCE_NOT_POSITIVE_SEMIDEFINITE",
            "covariance matrix is not positive semidefinite "
            "(LDL^T decomposition fails at pivot %d)" % pivot,
            "covariance[%d][%d]" % (pivot, pivot),
        )

    estimate, variance = propagate(values, sensitivities, matrix, intercept)
    return {
        "estimate": format_rational(estimate),
        "variance": format_rational(variance),
        "exceeds_budget": variance > budget,
    }
