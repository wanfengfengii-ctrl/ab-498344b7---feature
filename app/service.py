"""Core exact computation: estimate and propagated variance.

Given inputs x_i with sensitivities c_i, intercept b and covariance matrix S:

    estimate = b + sum_i c_i * x_i
    variance = c^T S c = sum_i sum_j c_i * S[i][j] * c_j

Everything is computed with :class:`fractions.Fraction`; no floating point
value is ever produced, so the budget verdict is exact.
"""
from __future__ import annotations

from fractions import Fraction
from typing import List, Sequence, Tuple

from .matrix import Matrix


def propagate(values: Sequence[Fraction],
              sensitivities: Sequence[Fraction],
              covariance: Matrix,
              intercept: Fraction) -> Tuple[Fraction, Fraction]:
    """Return ``(estimate, variance)`` computed with exact rational arithmetic."""
    estimate = intercept
    for sensitivity, value in zip(sensitivities, values):
        estimate += sensitivity * value

    n = len(values)
    variance = Fraction(0)
    for i in range(n):
        s_i = sensitivities[i]
        if not s_i:
            continue
        row = covariance[i]
        inner = Fraction(0)
        for j in range(n):
            s_j = sensitivities[j]
            if s_j:
                inner += row[j] * s_j
        variance += s_i * inner
    return estimate, variance
