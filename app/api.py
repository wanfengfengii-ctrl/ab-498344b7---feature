"""Request validation and evaluation orchestration for the evaluate endpoint.

Validation is deliberately strict and fully deterministic: the first problem
found is reported with a stable error code and a locatable field path, and no
error response ever contains an estimate or a budget verdict.
"""
from __future__ import annotations

from fractions import Fraction
from typing import Any, List

from .errors import ApiError
from .matrix import (check_symmetric, pd_failure_index, psd_failure_index)
from .rational import RationalFormatError, format_rational, parse_rational
from .service import condition, observation_covariance, propagate

MIN_INPUTS = 1
MAX_INPUTS = 24
MIN_OBSERVATIONS = 1
MAX_OBSERVATIONS = 4

_REQUIRED_TOP_LEVEL = ("inputs", "covariance", "intercept", "variance_budget")
_REQUIRED_INPUT_KEYS = ("id", "value", "sensitivity")
_REQUIRED_OBSERVATION_KEYS = ("id", "value", "coefficients")
_REQUIRED_CONDITIONING_KEYS = ("observations", "noise_covariance")


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
    ids: List[str] = []
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
        if input_id in ids:
            raise ApiError("INVALID_SCHEMA",
                           "duplicate input id '%s'" % input_id, base + ".id")
        ids.append(input_id)
        values.append(_parse_field(item["value"], base + ".value"))
        sensitivities.append(_parse_field(item["sensitivity"], base + ".sensitivity"))
    return ids, values, sensitivities


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


def _parse_noise_covariance(node: Any, k: int) -> List[List[Fraction]]:
    base = "conditioning.noise_covariance"
    if not isinstance(node, list):
        raise ApiError("INVALID_SCHEMA",
                       "'noise_covariance' must be an array of arrays", base)
    if len(node) != k:
        raise ApiError(
            "NOISE_COVARIANCE_DIMENSION_MISMATCH",
            "noise covariance has %d rows but there are %d observations"
            % (len(node), k),
            base,
        )
    matrix: List[List[Fraction]] = []
    for i, row in enumerate(node):
        row_field = "%s[%d]" % (base, i)
        if not isinstance(row, list):
            raise ApiError("INVALID_SCHEMA",
                           "noise covariance row must be an array", row_field)
        if len(row) != k:
            raise ApiError(
                "NOISE_COVARIANCE_DIMENSION_MISMATCH",
                "noise covariance row %d has %d entries, expected %d"
                % (i, len(row), k),
                row_field,
            )
        matrix.append([
            _parse_field(cell, "%s[%d][%d]" % (base, i, j))
            for j, cell in enumerate(row)
        ])
    return matrix


def _parse_conditioning(node: Any, input_ids: List[str]):
    """Validate the optional ``conditioning`` block.

    Returns ``(obs_rows, obs_values, noise_covariance)`` where ``obs_rows``
    is the coefficient matrix A indexed by input order.
    """
    base = "conditioning"
    if not isinstance(node, dict):
        raise ApiError("INVALID_SCHEMA",
                       "'conditioning' must be an object", base)
    for key in _REQUIRED_CONDITIONING_KEYS:
        if key not in node:
            raise ApiError("INVALID_SCHEMA",
                           "conditioning is missing '%s'" % key,
                           "%s.%s" % (base, key))

    observations = node["observations"]
    obs_field = base + ".observations"
    if not isinstance(observations, list):
        raise ApiError("INVALID_SCHEMA",
                       "'observations' must be an array", obs_field)
    if not MIN_OBSERVATIONS <= len(observations) <= MAX_OBSERVATIONS:
        raise ApiError(
            "INVALID_SCHEMA",
            "'observations' must contain between %d and %d entries, got %d"
            % (MIN_OBSERVATIONS, MAX_OBSERVATIONS, len(observations)),
            obs_field,
        )

    n = len(input_ids)
    id_index = {input_id: i for i, input_id in enumerate(input_ids)}
    obs_values: List[Fraction] = []
    obs_rows: List[List[Fraction]] = []
    seen_obs_ids = set()
    for r, item in enumerate(observations):
        entry_base = "%s[%d]" % (obs_field, r)
        if not isinstance(item, dict):
            raise ApiError("INVALID_SCHEMA",
                           "observation entry must be an object", entry_base)
        for key in _REQUIRED_OBSERVATION_KEYS:
            if key not in item:
                raise ApiError("INVALID_SCHEMA",
                               "observation entry is missing '%s'" % key,
                               entry_base)
        obs_id = item["id"]
        if not isinstance(obs_id, str) or not obs_id:
            raise ApiError("INVALID_SCHEMA",
                           "observation id must be a non-empty string",
                           entry_base + ".id")
        if obs_id in seen_obs_ids:
            raise ApiError("INVALID_SCHEMA",
                           "duplicate observation id '%s'" % obs_id,
                           entry_base + ".id")
        seen_obs_ids.add(obs_id)

        obs_values.append(_parse_field(item["value"], entry_base + ".value"))

        coefficients = item["coefficients"]
        coeff_field = entry_base + ".coefficients"
        if not isinstance(coefficients, dict) or not coefficients:
            raise ApiError(
                "INVALID_SCHEMA",
                "coefficients must be a non-empty object keyed by input id",
                coeff_field,
            )
        row = [Fraction(0)] * n
        for key, cell in coefficients.items():
            if not isinstance(key, str) or not key:
                raise ApiError("INVALID_SCHEMA",
                               "coefficient key must be a non-empty string",
                               coeff_field)
            if key not in id_index:
                raise ApiError(
                    "UNKNOWN_INPUT_REFERENCE",
                    "coefficient references unknown input id '%s'" % key,
                    "%s.%s" % (coeff_field, key))
            row[id_index[key]] = _parse_field(
                cell, "%s.%s" % (coeff_field, key))
        obs_rows.append(row)

    k = len(obs_values)
    noise = _parse_noise_covariance(node["noise_covariance"], k)

    asymmetric = check_symmetric(noise)
    if asymmetric is not None:
        i, j = asymmetric
        raise ApiError(
            "NOISE_COVARIANCE_NOT_SYMMETRIC",
            "noise_covariance[%d][%d] = %s differs from [%d][%d] = %s"
            % (i, j, noise[i][j], j, i, noise[j][i]),
            "conditioning.noise_covariance[%d][%d]" % (i, j),
        )
    pivot = psd_failure_index(noise)
    if pivot is not None:
        raise ApiError(
            "NOISE_COVARIANCE_NOT_POSITIVE_SEMIDEFINITE",
            "noise covariance matrix is not positive semidefinite "
            "(LDL^T decomposition fails at pivot %d)" % pivot,
            "conditioning.noise_covariance[%d][%d]" % (pivot, pivot),
        )
    return obs_rows, obs_values, noise


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

    if "conditioning" in obj and obj["conditioning"] is not None:
        obs_rows, obs_values, noise = _parse_conditioning(
            obj["conditioning"], input_ids)
        # The reference information yields a unique result iff the joint
        # observation covariance S = A Sigma A^T + R is positive definite.
        joint = observation_covariance(matrix, obs_rows, noise)
        joint_pivot = pd_failure_index(joint)
        if joint_pivot is not None:
            raise ApiError(
                "REFERENCE_INFORMATION_NOT_UNIQUE",
                "reference observations do not determine a unique result: "
                "joint observation covariance is singular at pivot %d"
                % joint_pivot,
                "conditioning.observations",
            )
        estimate, variance = condition(
            values, sensitivities, matrix, intercept,
            obs_rows, obs_values, noise, joint_covariance=joint)
    else:
        estimate, variance = propagate(values, sensitivities, matrix, intercept)

    return {
        "estimate": format_rational(estimate),
        "variance": format_rational(variance),
        "exceeds_budget": variance > budget,
    }
