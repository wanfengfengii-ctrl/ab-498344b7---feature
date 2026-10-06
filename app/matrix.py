"""Exact symmetric-matrix checks over rationals.

All computations use :class:`fractions.Fraction`, so the positive
semidefinite (PSD) / positive definite (PD) verdicts are exact -- no
floating point tolerance is ever involved.
"""
from __future__ import annotations

from fractions import Fraction
from typing import List, Optional, Sequence, Tuple

Matrix = List[List[Fraction]]


def check_symmetric(matrix: Sequence[Sequence[Fraction]]) -> Optional[Tuple[int, int]]:
    """Return the first ``(i, j)`` with ``i < j`` where A[i][j] != A[j][i].

    Returns ``None`` when the matrix is exactly symmetric.
    """
    n = len(matrix)
    for i in range(n):
        row = matrix[i]
        for j in range(i + 1, n):
            if row[j] != matrix[j][i]:
                return (i, j)
    return None


def psd_failure_index(matrix: Sequence[Sequence[Fraction]]) -> Optional[int]:
    """Exact PSD test via LDL^T (symmetric Gaussian elimination).

    Returns the index of the pivot where the decomposition proves the matrix
    is **not** positive semidefinite, or ``None`` when it is PSD.

    Correctness: the Schur complement of a PSD matrix is PSD, so every pivot
    of a PSD matrix is >= 0, and a zero pivot forces its whole remaining
    row/column to be zero.  Conversely, if the elimination completes with
    non-negative pivots, then A = L D L^T with D >= 0, hence PSD.  All
    arithmetic is exact, so the verdict is exact.
    """
    n = len(matrix)
    a: Matrix = [list(row) for row in matrix]
    for k in range(n):
        pivot = a[k][k]
        if pivot < 0:
            return k
        if pivot == 0:
            # A zero pivot in a PSD matrix forces the rest of its row to be 0.
            for j in range(k + 1, n):
                if a[k][j] != 0:
                    return k
            continue
        for i in range(k + 1, n):
            factor = a[i][k] / pivot
            if factor:
                row_i = a[i]
                row_k = a[k]
                for j in range(k + 1, n):
                    row_i[j] -= factor * row_k[j]
            a[i][k] = Fraction(0)
    return None


def pd_failure_index(matrix: Sequence[Sequence[Fraction]]) -> Optional[int]:
    """Exact positive-definite test via LDL^T (symmetric Gaussian elimination).

    Returns the index of a pivot that is not strictly positive, or ``None``
    when the matrix is positive definite.  The caller must first verify
    symmetry.

    This is the uniqueness test for conditioning: the joint linear model has
    a unique conditional result exactly when the observation covariance
    ``S = A Sigma A^T + R`` is positive definite.
    """
    n = len(matrix)
    a: Matrix = [list(row) for row in matrix]
    for k in range(n):
        pivot = a[k][k]
        if pivot <= 0:
            return k
        for i in range(k + 1, n):
            factor = a[i][k] / pivot
            if factor:
                row_i = a[i]
                row_k = a[k]
                for j in range(k + 1, n):
                    row_i[j] -= factor * row_k[j]
            a[i][k] = Fraction(0)
    return None


def solve_symmetric(matrix: Sequence[Sequence[Fraction]],
                    rhs: Sequence[Fraction]) -> List[Fraction]:
    """Exactly solve ``matrix x = rhs`` for a symmetric positive-definite matrix.

    Uses Gauss elimination followed by back substitution, all in
    :class:`fractions.Fraction`.  No pivoting is required: every leading
    principal minor of a positive-definite matrix is strictly positive, so
    each elimination pivot is nonzero.  A singular matrix nevertheless
    raises :class:`ValueError`.
    """
    n = len(matrix)
    a: Matrix = [list(row) + [rhs[i]] for i, row in enumerate(matrix)]
    # Forward elimination (only the lower triangle is read; symmetry lets us
    # skip the upper-triangle updates that Gaussian elimination would need).
    for k in range(n):
        pivot = a[k][k]
        if pivot == 0:
            raise ValueError("matrix is singular")
        for i in range(k + 1, n):
            factor = a[i][k] / pivot
            if factor:
                row_i = a[i]
                row_k = a[k]
                for j in range(k + 1, n + 1):
                    row_i[j] -= factor * row_k[j]
            a[i][k] = Fraction(0)
    # Back substitution.
    x: List[Fraction] = [Fraction(0)] * n
    for i in range(n - 1, -1, -1):
        total = a[i][n]
        row_i = a[i]
        for j in range(i + 1, n):
            total -= row_i[j] * x[j]
        x[i] = total / a[i][i]
    return x
