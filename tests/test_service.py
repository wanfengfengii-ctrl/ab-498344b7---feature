import unittest
from fractions import Fraction as F

from app.service import propagate


class PropagateTests(unittest.TestCase):
    def test_single_input(self):
        estimate, variance = propagate(
            values=[F(3, 2)],
            sensitivities=[F(2)],
            covariance=[[F(1, 4)]],
            intercept=F(1, 2),
        )
        self.assertEqual(estimate, F(7, 2))
        self.assertEqual(variance, F(1))

    def test_two_inputs_with_cross_terms(self):
        # estimate = 1/2 + 2*(3/2) + (1/3)*(-1/4) = 41/12
        # variance = [2, 1/3] @ [[1/4, 1/8], [1/8, 1/2]] @ [2, 1/3]^T = 11/9
        estimate, variance = propagate(
            values=[F(3, 2), F(-1, 4)],
            sensitivities=[F(2), F(1, 3)],
            covariance=[[F(1, 4), F(1, 8)], [F(1, 8), F(1, 2)]],
            intercept=F(1, 2),
        )
        self.assertEqual(estimate, F(41, 12))
        self.assertEqual(variance, F(11, 9))

    def test_zero_sensitivity_contributes_nothing(self):
        estimate, variance = propagate(
            values=[F(5), F(99)],
            sensitivities=[F(1), F(0)],
            covariance=[[F(1), F(7)], [F(7), F(49)]],
            intercept=F(0),
        )
        self.assertEqual(estimate, F(5))
        self.assertEqual(variance, F(1))

    def test_negative_intercept_and_values(self):
        estimate, variance = propagate(
            values=[F(-3, 4)],
            sensitivities=[F(-2)],
            covariance=[[F(3, 8)]],
            intercept=F(-1, 4),
        )
        self.assertEqual(estimate, F(5, 4))
        self.assertEqual(variance, F(3, 2))  # (-2)^2 * 3/8

    def test_variance_is_exact_not_float(self):
        # 1/3 has no finite binary representation; the result must stay exact.
        _, variance = propagate(
            values=[F(0)],
            sensitivities=[F(1, 3)],
            covariance=[[F(1)]],
            intercept=F(0),
        )
        self.assertEqual(variance, F(1, 9))
        self.assertIsInstance(variance, F)


if __name__ == "__main__":
    unittest.main()
