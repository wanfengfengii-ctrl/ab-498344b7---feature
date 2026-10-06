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
from .service import condition, propagate

MIN_INPUTS = 1
MAX_INPUTS = 24
MIN_OBSERVATIONS = 1
MAX_OBSERVATIONS = 4

_REQUIRED_TOP_LEVEL = ("inputs", "covariance", "intercept", "variance_budget")
_REQUIRED_INPUT_KEYS = ("id", "value", "sensitivity")
_REQUIRED_CONDITIONING_KEYS = ("observations", "covariance")
_REQUIRED_OBSERVATION_KEYS = ("id", "value", "coefficients")


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
    input_ids: List[str] = []
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
        input_ids.append(input_id)
        values.append(_parse_field(item["value"], base + ".value"))
        sensitivities.append(_parse_field(item["sensitivity"], base + ".sensitivity"))
    return input_ids, values, sensitivities


def _parse_covariance(node: Any, n: int, field: str = "covariance",
                      mismatch_code: str = "COVARIANCE_DIMENSION_MISMATCH",
                      expected_label: str = "inputs"
                      ) -> List[List[Fraction]]:
    if not isinstance(node, list):
        raise ApiError("INVALID_SCHEMA",
                       "'%s' must be an array of arrays" % field, field)
    if len(node) != n:
        raise ApiError(
            mismatch_code,
            "%s has %d rows but there are %d %s"
            % (field, len(node), n, expected_label),
            field,
        )
    matrix: List[List[Fraction]] = []
    for i, row in enumerate(node):
        row_field = "%s[%d]" % (field, i)
        if not isinstance(row, list):
            raise ApiError("INVALID_SCHEMA",
                           "%s row must be an array" % field, row_field)
        if len(row) != n:
            raise ApiError(
                mismatch_code,
                "%s row %d has %d entries, expected %d"
                % (field, i, len(row), n),
                row_field,
            )
        matrix.append([
            _parse_field(cell, "%s[%d][%d]" % (field, i, j))
            for j, cell in enumerate(row)
        ])
    return matrix


def _check_exact_psd(matrix: List[List[Fraction]], field: str, label: str,
                     not_symmetric_code: str, not_psd_code: str) -> None:
    """Enforce exact symmetry and positive semidefiniteness of a matrix."""
    asymmetric = check_symmetric(matrix)
    if asymmetric is not None:
        i, j = asymmetric
        raise ApiError(
            not_symmetric_code,
            "%s[%d][%d] = %s differs from %s[%d][%d] = %s"
            % (field, i, j, matrix[i][j], field, j, i, matrix[j][i]),
            "%s[%d][%d]" % (field, i, j),
        )
    pivot = psd_failure_index(matrix)
    if pivot is not None:
        raise ApiError(
            not_psd_code,
            "%s is not positive semidefinite "
            "(LDL^T decomposition fails at pivot %d)" % (label, pivot),
            "%s[%d][%d]" % (field, pivot, pivot),
        )


def _parse_conditioning(node: Any, input_index: dict) -> tuple:
    """Validate the optional ``conditioning`` object.

    Returns ``(ref_values, ref_rows, ref_covariance)`` where ``ref_rows`` is
    the m x n coefficient matrix aligned with the input order.
    """
    if not isinstance(node, dict):
        raise ApiError("INVALID_SCHEMA",
                       "'conditioning' must be an object", "conditioning")
    for key in _REQUIRED_CONDITIONING_KEYS:
        if key not in node:
            raise ApiError("INVALID_SCHEMA",
                           "missing required field 'conditioning.%s'" % key,
                           "conditioning." + key)

    observations = node["observations"]
    if not isinstance(observations, list):
        raise ApiError("INVALID_SCHEMA",
                       "'conditioning.observations' must be an array",
                       "conditioning.observations")
    if not MIN_OBSERVATIONS <= len(observations) <= MAX_OBSERVATIONS:
        raise ApiError(
            "INVALID_SCHEMA",
            "'conditioning.observations' must contain between %d and %d "
            "entries, got %d"
            % (MIN_OBSERVATIONS, MAX_OBSERVATIONS, len(observations)),
            "conditioning.observations",
        )

    n = len(input_index)
    seen_ids = set()
    ref_values: List[Fraction] = []
    ref_rows: List[List[Fraction]] = []
    for k, item in enumerate(observations):
        base = "conditioning.observations[%d]" % k
        if not isinstance(item, dict):
            raise ApiError("INVALID_SCHEMA",
                           "observation entry must be an object", base)
        for key in _REQUIRED_OBSERVATION_KEYS:
            if key not in item:
                raise ApiError("INVALID_SCHEMA",
                               "observation entry is missing '%s'" % key, base)
        obs_id = item["id"]
        if not isinstance(obs_id, str) or not obs_id:
            raise ApiError("INVALID_SCHEMA",
                           "observation id must be a non-empty string",
                           base + ".id")
        if obs_id in seen_ids:
            raise ApiError("INVALID_SCHEMA",
                           "duplicate observation id '%s'" % obs_id,
                           base + ".id")
        seen_ids.add(obs_id)
        ref_values.append(_parse_field(item["value"], base + ".value"))

        coefficients = item["coefficients"]
        if not isinstance(coefficients, dict):
            raise ApiError(
                "INVALID_SCHEMA",
                "observation coefficients must be an object mapping input "
                "ids to rationals",
                base + ".coefficients",
            )
        if not coefficients:
            raise ApiError("INVALID_SCHEMA",
                           "observation coefficients must not be empty",
                           base + ".coefficients")
        row = [Fraction(0)] * n
        for input_id, coefficient_node in coefficients.items():
            coefficient_field = "%s.coefficients.%s" % (base, input_id)
            if input_id not in input_index:
                raise ApiError(
                    "UNKNOWN_INPUT_ID",
                    "coefficient references unknown input id '%s'" % input_id,
                    coefficient_field,
                )
            row[input_index[input_id]] = _parse_field(coefficient_node,
                                                      coefficient_field)
        ref_rows.append(row)

    m = len(ref_values)
    ref_covariance = _parse_covariance(
        node["covariance"], m,
        field="conditioning.covariance",
        mismatch_code="CONDITIONING_DIMENSION_MISMATCH",
        expected_label="observations")
    _check_exact_psd(ref_covariance,
                     field="conditioning.covariance",
                     label="conditioning covariance matrix",
                     not_symmetric_code="CONDITIONING_COVARIANCE_NOT_SYMMETRIC",
                     not_psd_code="CONDITIONING_COVARIANCE_NOT_POSITIVE_SEMIDEFINITE")
    return ref_values, ref_rows, ref_covariance


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

    input_ids, values, sensitivities = _parse_inputs(obj["inputs"])
    n = len(values)

    intercept = _parse_field(obj["intercept"], "intercept")

    budget = _parse_field(obj["variance_budget"], "variance_budget")
    if budget < 0:
        raise ApiError("INVALID_VARIANCE_BUDGET",
                       "variance budget must be non-negative",
                       "variance_budget")

    matrix = _parse_covariance(obj["covariance"], n)
    _check_exact_psd(matrix,
                     field="covariance",
                     label="covariance matrix",
                     not_symmetric_code="COVARIANCE_NOT_SYMMETRIC",
                     not_psd_code="COVARIANCE_NOT_POSITIVE_SEMIDEFINITE")

    if "conditioning" in obj:
        input_index = {input_id: i for i, input_id in enumerate(input_ids)}
        ref_values, ref_rows, ref_covariance = _parse_conditioning(
            obj["conditioning"], input_index)
        result = condition(values, sensitivities, matrix, intercept,
                           ref_values, ref_rows, ref_covariance)
        if result is None:
            raise ApiError(
                "CONDITIONING_SINGULAR",
                "reference observations do not determine a unique "
                "conditional result (joint observation covariance "
                "H*S*H^T + R is singular)",
                "conditioning",
            )
        estimate, variance = result
    else:
        estimate, variance = propagate(values, sensitivities, matrix, intercept)
    return {
        "estimate": format_rational(estimate),
        "variance": format_rational(variance),
        "exceeds_budget": variance > budget,
    }
