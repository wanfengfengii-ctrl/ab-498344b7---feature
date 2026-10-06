"""Core exact computation: estimate and propagated variance.

Given inputs x_i with sensitivities c_i, intercept b and covariance matrix S:

    estimate = b + sum_i c_i * x_i
    variance = c^T S c = sum_i sum_j c_i * S[i][j] * c_j

Everything is computed with :class:`fractions.Fraction`; no floating point
value is ever produced, so the budget verdict is exact.

Conditioning
------------

When reference observations y = A x + e are supplied, with e independent of
x and E[e] = 0, Cov(e) = R, the joint linear model is

    [x] ~ N([mu], [Sigma,   Sigma A^T])
    [y]    ([A mu]  [A Sigma, A Sigma A^T + R])

so the exact conditional moments of the output z = b + c^T x given y are

    estimate = b + c^T mu + c^T Sigma A^T S^{-1} (y - A mu)
    variance = c^T Sigma c - c^T Sigma A^T S^{-1} A Sigma c

with S = A Sigma A^T + R.  The result is unique iff S is invertible; since
R is PSD, S is PSD and invertibility is exactly positive definiteness.
"""
from __future__ import annotations

from fractions import Fraction
from typing import List, Optional, Sequence, Tuple

from .matrix import Matrix, solve_symmetric


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


def _mat_vec(matrix: Matrix, vector: Sequence[Fraction]) -> List[Fraction]:
    return [sum((matrix[i][j] * vector[j] for j in range(len(vector))),
                Fraction(0))
            for i in range(len(matrix))]


def observation_covariance(covariance: Matrix,
                           obs_rows: Sequence[Sequence[Fraction]],
                           noise_covariance: Matrix) -> Matrix:
    """Return the joint observation covariance S = A Sigma A^T + R."""
    n = len(covariance)
    k = len(obs_rows)
    # Precompute Sigma A_r^T for every observation row r.
    sigma_a = [_mat_vec(covariance, row) for row in obs_rows]
    s: Matrix = [[Fraction(0)] * k for _ in range(k)]
    for r in range(k):
        for t in range(r, k):
            total = sum((obs_rows[r][i] * sigma_a[t][i] for i in range(n)),
                        Fraction(0))
            s[r][t] = total + noise_covariance[r][t]
            s[t][r] = total + noise_covariance[t][r]
    return s


def condition(values: Sequence[Fraction],
              sensitivities: Sequence[Fraction],
              covariance: Matrix,
              intercept: Fraction,
              obs_rows: Sequence[Sequence[Fraction]],
              obs_values: Sequence[Fraction],
              noise_covariance: Matrix,
              joint_covariance: Optional[Matrix] = None
              ) -> Tuple[Fraction, Fraction]:
    """Return the exact conditional ``(estimate, variance)`` of the output.

    ``obs_rows`` is the coefficient matrix A (one row per reference
    observation, one column per input, indexed in input order);
    ``obs_values`` is y and ``noise_covariance`` is R.  The caller guarantees
    that S = A Sigma A^T + R is symmetric positive definite.
    """
    n = len(values)
    k = len(obs_values)

    # Sigma c  (n-vector)
    sigma_c = _mat_vec(covariance, sensitivities)
    # A Sigma c  (k-vector)
    a_sigma_c = [
        sum((obs_rows[r][i] * sigma_c[i] for i in range(n)), Fraction(0))
        for r in range(k)
    ]
    # S = A Sigma A^T + R  (k x k)
    s = (joint_covariance if joint_covariance is not None
         else observation_covariance(covariance, obs_rows, noise_covariance))
    # S^{-1} (A Sigma c)
    solved = solve_symmetric(s, a_sigma_c)

    # Innovation y - A mu.
    predicted = [
        sum((obs_rows[r][i] * values[i] for i in range(n)), Fraction(0))
        for r in range(k)
    ]
    innovation = [obs_values[r] - predicted[r] for r in range(k)]

    prior_estimate, prior_variance = propagate(
        values, sensitivities, covariance, intercept)
    # sigma_c already equals Sigma c, so c^T Sigma A^T S^{-1} A Sigma c is
    # (A Sigma c)^T S^{-1} (A Sigma c); the mean correction reuses solved.
    estimate = prior_estimate + sum(
        (solved[r] * innovation[r] for r in range(k)), Fraction(0))
    variance = prior_variance - sum(
        (a_sigma_c[r] * solved[r] for r in range(k)), Fraction(0))
    return estimate, variance
