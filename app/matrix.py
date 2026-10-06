"""Exact symmetric-matrix checks over rationals.

All computations use :class:`fractions.Fraction`, so the positive
semidefinite (PSD) verdict is exact -- no floating point tolerance is ever
involved.
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


def invert(matrix: Sequence[Sequence[Fraction]]) -> Optional[Matrix]:
    """Exact inverse of a square matrix over the rationals.

    Uses Gauss-Jordan elimination with partial pivoting; every operation is
    exact, so the result is exact.  Returns ``None`` when the matrix is
    singular (no pivot in some column), in which case no unique solution
    exists.
    """
    n = len(matrix)
    aug = [
        list(matrix[i]) + [Fraction(1) if i == j else Fraction(0)
                           for j in range(n)]
        for i in range(n)
    ]
    for col in range(n):
        pivot_row = None
        for r in range(col, n):
            if aug[r][col] != 0:
                pivot_row = r
                break
        if pivot_row is None:
            return None
        if pivot_row != col:
            aug[col], aug[pivot_row] = aug[pivot_row], aug[col]
        inv_pivot = 1 / aug[col][col]
        aug[col] = [value * inv_pivot for value in aug[col]]
        for r in range(n):
            if r != col and aug[r][col] != 0:
                factor = aug[r][col]
                aug[r] = [a - factor * b for a, b in zip(aug[r], aug[col])]
    return [row[n:] for row in aug]


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
