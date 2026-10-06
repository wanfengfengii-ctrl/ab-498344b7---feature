"""Core exact computation: estimate and propagated variance.

Given inputs x_i with sensitivities c_i, intercept b and covariance matrix S:

    estimate = b + sum_i c_i * x_i
    variance = c^T S c = sum_i sum_j c_i * S[i][j] * c_j

Optional reference observations z = H x + e (e with covariance R) update the
result by exact linear conditioning on the joint model:

    estimate' = estimate + B^T M^-1 (z - H x)
    variance' = variance - B^T M^-1 B
    M = H S H^T + R,  B = H S c

Everything is computed with :class:`fractions.Fraction`; no floating point
value is ever produced, so the budget verdict is exact.
"""
from __future__ import annotations

from fractions import Fraction
from typing import List, Optional, Sequence, Tuple

from .matrix import Matrix, invert


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


def condition(values: Sequence[Fraction],
              sensitivities: Sequence[Fraction],
              covariance: Matrix,
              intercept: Fraction,
              ref_values: Sequence[Fraction],
              ref_rows: Sequence[Sequence[Fraction]],
              ref_covariance: Matrix
              ) -> Optional[Tuple[Fraction, Fraction]]:
    """Return ``(estimate, variance)`` conditioned on reference observations.

    ``ref_rows`` is the m x n coefficient matrix H of the reference
    observations (row k maps the n inputs to observation k), ``ref_values``
    the m observed values z and ``ref_covariance`` the m x m reference noise
    covariance R.  Conditioning is exact::

        estimate' = estimate + B^T M^-1 (z - H x)
        variance' = variance - B^T M^-1 B
        M = H S H^T + R,  B = H S c

    Returns ``None`` when M is singular: the reference information then does
    not determine a unique conditional result.
    """
    estimate, variance = propagate(values, sensitivities, covariance, intercept)

    n = len(values)
    m = len(ref_values)

    # a = H S (m x n); only nonzero H entries contribute.
    a: Matrix = [[Fraction(0)] * n for _ in range(m)]
    for k in range(m):
        row_h = ref_rows[k]
        row_a = a[k]
        for i in range(n):
            h = row_h[i]
            if h:
                col_s = covariance[i]
                for j in range(n):
                    if col_s[j]:
                        row_a[j] += h * col_s[j]

    # M = H S H^T + R (m x m), beta = H S c = a c (m-vector).
    m_matrix: Matrix = [list(row) for row in ref_covariance]
    beta: List[Fraction] = [Fraction(0)] * m
    for k in range(m):
        row_a = a[k]
        row_m = m_matrix[k]
        for l in range(m):
            acc = row_m[l]
            row_h = ref_rows[l]
            for j in range(n):
                if row_a[j] and row_h[j]:
                    acc += row_a[j] * row_h[j]
            row_m[l] = acc
        for j in range(n):
            if row_a[j] and sensitivities[j]:
                beta[k] += row_a[j] * sensitivities[j]

    m_inverse = invert(m_matrix)
    if m_inverse is None:
        return None

    # residual r = z - H x (m-vector).
    residual: List[Fraction] = []
    for k in range(m):
        r = ref_values[k]
        row_h = ref_rows[k]
        for j in range(n):
            if row_h[j] and values[j]:
                r -= row_h[j] * values[j]
        residual.append(r)

    # estimate' = estimate + beta^T M^-1 r; variance' = variance - beta^T M^-1 beta.
    for k in range(m):
        if not beta[k]:
            continue
        alpha = Fraction(0)   # (M^-1 r)_k
        gamma = Fraction(0)   # (M^-1 beta)_k
        row_inv = m_inverse[k]
        for l in range(m):
            if row_inv[l]:
                if residual[l]:
                    alpha += row_inv[l] * residual[l]
                if beta[l]:
                    gamma += row_inv[l] * beta[l]
        estimate += beta[k] * alpha
        variance -= beta[k] * gamma
    return estimate, variance
