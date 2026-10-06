import unittest
from fractions import Fraction as F

from app.matrix import (check_symmetric, pd_failure_index,
                        psd_failure_index, solve_symmetric)


def m(rows):
    return [[F(v) for v in row] for row in rows]


class CheckSymmetricTests(unittest.TestCase):
    def test_symmetric(self):
        self.assertIsNone(check_symmetric(m([[1, 2], [2, 3]])))
        self.assertIsNone(check_symmetric(m([[0]])))
        self.assertIsNone(check_symmetric(
            [[F(1, 2), F(1, 3)], [F(1, 3), F(1, 4)]]))

    def test_first_mismatch_reported(self):
        self.assertEqual(check_symmetric(m([[1, 5], [2, 3]])), (0, 1))
        self.assertEqual(
            check_symmetric(m([[1, 2, 3], [2, 1, 4], [3, 5, 1]])), (1, 2))


class PsdFailureIndexTests(unittest.TestCase):
    def assert_psd(self, rows):
        self.assertIsNone(psd_failure_index(m(rows)), msg=repr(rows))

    def assert_not_psd(self, rows, pivot):
        self.assertEqual(psd_failure_index(m(rows)), pivot, msg=repr(rows))

    def test_positive_definite(self):
        self.assert_psd([[1]])
        self.assert_psd([[2, 1], [1, 2]])
        self.assert_psd([[2, 1, 1], [1, 2, 1], [1, 1, 2]])
        self.assert_psd([[F(1, 2), F(1, 4)], [F(1, 4), F(1, 2)]])

    def test_positive_semidefinite_singular(self):
        self.assert_psd([[0]])
        self.assert_psd([[0, 0], [0, 0]])
        self.assert_psd([[0, 0], [0, 5]])
        self.assert_psd([[1, 2], [2, 4]])               # rank one
        self.assert_psd([[2, 0, 0], [0, 0, 0], [0, 0, 3]])
        self.assert_psd([[F(1, 2), F(1, 4)], [F(1, 4), F(1, 8)]])  # rank one
        self.assert_psd([[1, 1, 1], [1, 1, 1], [1, 1, 1]])

    def test_indefinite(self):
        self.assert_not_psd([[-1]], 0)
        self.assert_not_psd([[1, 2], [2, 1]], 1)
        self.assert_not_psd([[0, 1], [1, 0]], 0)   # zero pivot, nonzero row
        self.assert_not_psd([[0, 0], [0, -1]], 1)
        self.assert_not_psd([[1, 0, 0], [0, -2, 0], [0, 0, 1]], 1)

    def test_exact_fraction_pivot(self):
        # Second pivot is exactly 1/16 - (1/4)^2/(1/2) = -1/16 < 0.
        self.assert_not_psd(
            [[F(1, 2), F(1, 4)], [F(1, 4), F(1, 16)]], 1)


class PdFailureIndexTests(unittest.TestCase):
    def test_positive_definite(self):
        self.assertIsNone(pd_failure_index(m([[1]])))
        self.assertIsNone(pd_failure_index(m([[2, 1], [1, 2]])))
        self.assertIsNone(pd_failure_index(
            m([[F(1, 2), F(1, 4)], [F(1, 4), F(1, 2)]])))

    def test_singular_reported(self):
        # Zero and rank-one PSD matrices are not positive definite.
        self.assertEqual(pd_failure_index(m([[0]])), 0)
        self.assertEqual(pd_failure_index(m([[0, 0], [0, 5]])), 0)
        self.assertEqual(pd_failure_index(m([[1, 2], [2, 4]])), 1)
        self.assertEqual(
            pd_failure_index(m([[1, 1, 1], [1, 1, 1], [1, 1, 1]])), 1)
        # A negative pivot must be reported too.
        self.assertEqual(pd_failure_index(m([[1, 2], [2, 1]])), 1)


class SolveSymmetricTests(unittest.TestCase):
    def test_simple_system(self):
        x = solve_symmetric(m([[2, 1], [1, 3]]), [F(5), F(6)])
        self.assertEqual(x, [F(9, 5), F(7, 5)])

    def test_single_variable(self):
        self.assertEqual(solve_symmetric(m([[F(1, 3)]]), [F(2, 5)]),
                         [F(6, 5)])

    def test_solution_is_exact_fraction(self):
        # No finite binary float solves this exactly; Fraction must.
        x = solve_symmetric(m([[3, 1], [1, 3]]), [F(1, 3), F(0)])
        self.assertEqual(x, [F(1, 8), F(-1, 24)])
        for xi in x:
            self.assertIsInstance(xi, F)

    def test_singular_raises(self):
        with self.assertRaises(ValueError):
            solve_symmetric(m([[1, 2], [2, 4]]), [F(1), F(1)])


if __name__ == "__main__":
    unittest.main()
