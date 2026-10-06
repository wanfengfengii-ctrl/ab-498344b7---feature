import unittest
from fractions import Fraction as F

from app.service import condition, propagate


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


class ConditionTests(unittest.TestCase):
    VALUES = [F(3, 2), F(-1, 4)]
    SENSITIVITIES = [F(2), F(1, 3)]
    COVARIANCE = [[F(1, 4), F(1, 8)], [F(1, 8), F(1, 2)]]
    INTERCEPT = F(1, 2)
    # Unconditioned reference point: estimate 41/12, variance 11/9.

    def condition(self, ref_values, ref_rows, ref_covariance):
        return condition(self.VALUES, self.SENSITIVITIES, self.COVARIANCE,
                         self.INTERCEPT, ref_values, ref_rows, ref_covariance)

    def test_single_noisy_observation(self):
        # z = x1 + noise, R = 1/4, observed z = 2.
        # M = 1/4 + 1/4 = 1/2, beta = 13/24, residual = 1/2.
        estimate, variance = self.condition(
            ref_values=[F(2)],
            ref_rows=[[F(1), F(0)]],
            ref_covariance=[[F(1, 4)]],
        )
        self.assertEqual(estimate, F(95, 24))
        self.assertEqual(variance, F(61, 96))

    def test_observation_at_prior_mean_keeps_estimate(self):
        # Observing exactly H x leaves the estimate unchanged but still
        # reduces the variance.
        estimate, variance = self.condition(
            ref_values=[F(3, 2)],   # E[z] = x1 = 3/2
            ref_rows=[[F(1), F(0)]],
            ref_covariance=[[F(1, 4)]],
        )
        self.assertEqual(estimate, F(41, 12))
        self.assertEqual(variance, F(61, 96))
        self.assertLess(variance, F(11, 9))

    def test_noiseless_observation_of_output_zeroes_variance(self):
        # z = 2*x1 + (1/3)*x2 observed without noise determines y exactly.
        estimate, variance = self.condition(
            ref_values=[F(35, 12)],
            ref_rows=[[F(2), F(1, 3)]],
            ref_covariance=[[F(0)]],
        )
        self.assertEqual(estimate, F(41, 12))
        self.assertEqual(variance, F(0))

    def test_two_observations(self):
        estimate, variance = self.condition(
            ref_values=[F(2), F(-1)],
            ref_rows=[[F(1), F(0)], [F(1, 2), F(1)]],
            ref_covariance=[[F(1, 4), F(1, 8)], [F(1, 8), F(1, 3)]],
        )
        self.assertEqual(estimate, F(2765, 852))
        self.assertEqual(variance, F(1801, 3408))

    def test_uninformative_observation_is_a_no_op(self):
        # A coefficient row of zeros observes nothing; with noise R the
        # system stays invertible and nothing changes.
        estimate, variance = self.condition(
            ref_values=[F(99)],
            ref_rows=[[F(0), F(0)]],
            ref_covariance=[[F(1)]],
        )
        self.assertEqual((estimate, variance), (F(41, 12), F(11, 9)))

    def test_duplicate_noiseless_observations_are_singular(self):
        # Two identical observations with zero noise: M is rank one.
        self.assertIsNone(self.condition(
            ref_values=[F(2), F(2)],
            ref_rows=[[F(1), F(0)], [F(1), F(0)]],
            ref_covariance=[[F(0), F(0)], [F(0), F(0)]],
        ))

    def test_noiseless_observation_without_variance_is_singular(self):
        # x2 has no variance in the direction observed (zero row of S) and
        # the observation is noiseless, so M = 0.
        self.assertIsNone(condition(
            values=[F(1)],
            sensitivities=[F(1)],
            covariance=[[F(0)]],
            intercept=F(0),
            ref_values=[F(1)],
            ref_rows=[[F(1)]],
            ref_covariance=[[F(0)]],
        ))

    def test_results_are_exact_fractions(self):
        estimate, variance = self.condition(
            ref_values=[F(2)],
            ref_rows=[[F(1), F(0)]],
            ref_covariance=[[F(1, 4)]],
        )
        self.assertIsInstance(estimate, F)
        self.assertIsInstance(variance, F)


if __name__ == "__main__":
    unittest.main()
