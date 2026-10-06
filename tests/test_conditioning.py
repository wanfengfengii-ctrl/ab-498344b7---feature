import unittest
from fractions import Fraction as F

from app.service import condition, observation_covariance


class ObservationCovarianceTests(unittest.TestCase):
    def test_formula(self):
        # S = A Sigma A^T + R for a tiny case, computed by hand.
        Sigma = [[F(2), F(1)], [F(1), F(3)]]
        A = [[F(1), F(0)], [F(1), F(1)]]
        R = [[F(1, 2), F(0)], [F(0), F(1)]]
        # A Sigma A^T:
        #   [1 0] [[2 1],[1 3]] [1 1]^T ...
        s = observation_covariance(Sigma, A, R)
        # A Sigma = [[2,1],[3,4]]; times A^T = [[2, 3],[3, 7]]
        self.assertEqual(s, [[F(5, 2), F(3)], [F(3), F(8)]])

    def test_is_symmetric(self):
        Sigma = [[F(1), F(1, 2)], [F(1, 2), F(1)]]
        A = [[F(1), F(1)], [F(1), F(0)]]
        R = [[F(0), F(0)], [F(0), F(1)]]
        s = observation_covariance(Sigma, A, R)
        self.assertEqual(s[0][1], s[1][0])
        # (A Sigma A^T)[0][0] = 1 + 1/2 + 1/2 + 1 = 3; cross = 3/2;
        # bottom-right = 1 + R = 2.
        self.assertEqual(s, [[F(3), F(3, 2)], [F(3, 2), F(2)]])


class ConditionTests(unittest.TestCase):
    def test_exact_observation_pins_input(self):
        # x ~ N(mu, diag(4,9)); observing x1 = 10 with zero noise pins it.
        mu = [F(3), F(-1)]
        c = [F(2), F(1)]
        Sigma = [[F(4), F(0)], [F(0), F(9)]]
        A = [[F(1), F(0)]]
        y = [F(10)]
        R = [[F(0)]]
        s = observation_covariance(Sigma, A, R)
        estimate, variance = condition(mu, c, Sigma, F(0), A, y, R, s)
        # estimate = 2*10 + 1*(-1) = 19; variance collapses to that of x2.
        self.assertEqual(estimate, F(19))
        self.assertEqual(variance, F(9))

    def test_noisy_observation_kalman_update(self):
        # Gain on x1 = 4/(4+1) = 4/5; innovation 10-3 = 7.
        mu = [F(3), F(-1)]
        c = [F(2), F(1)]
        Sigma = [[F(4), F(0)], [F(0), F(9)]]
        A = [[F(1), F(0)]]
        y = [F(10)]
        R = [[F(1)]]
        s = observation_covariance(Sigma, A, R)
        estimate, variance = condition(mu, c, Sigma, F(0), A, y, R, s)
        # x1 | y = 3 + 4/5 * 7 = 43/5
        self.assertEqual(estimate, F(81, 5))
        # 25 - (A Sigma c)^2 / S = 25 - 64/5 = 61/5
        self.assertEqual(variance, F(61, 5))

    def test_matching_observation_leaves_prior(self):
        # y = A mu gives zero innovation; estimate is the prior value and
        # variance strictly decreases.
        mu = [F(3), F(-1)]
        c = [F(2), F(1)]
        Sigma = [[F(4), F(0)], [F(0), F(9)]]
        A = [[F(1), F(0)]]
        y = [F(3)]
        R = [[F(1)]]
        s = observation_covariance(Sigma, A, R)
        estimate, variance = condition(mu, c, Sigma, F(1, 2), A, y, R, s)
        self.assertEqual(estimate, F(11, 2))      # prior: 1/2 + 6 - 1
        self.assertEqual(variance, F(61, 5))      # information still sharpens

    def test_variance_never_increases(self):
        mu = [F(0), F(0), F(0)]
        c = [F(1), F(-1), F(2)]
        Sigma = [
            [F(2), F(1, 2), F(0)],
            [F(1, 2), F(1), F(1, 4)],
            [F(0), F(1, 4), F(1)],
        ]
        A = [[F(1), F(1), F(0)], [F(0), F(1), F(1)]]
        y = [F(1), F(-1)]
        R = [[F(1, 2), F(0)], [F(0), F(1)]]
        s = observation_covariance(Sigma, A, R)
        prior_var = sum(
            c[i] * sum(Sigma[i][j] * c[j] for j in range(3))
            for i in range(3))
        _, variance = condition(mu, c, Sigma, F(0), A, y, R, s)
        self.assertLess(variance, prior_var)
        self.assertGreater(variance, 0)

    def test_observation_orthogonal_to_output_changes_only_estimate(self):
        # Observing x2 with zero noise while c = [1, 0] must not change the
        # output's variance (c only depends on x1, which is independent).
        mu = [F(1), F(1)]
        c = [F(1), F(0)]
        Sigma = [[F(4), F(0)], [F(0), F(9)]]
        A = [[F(0), F(1)]]
        y = [F(5)]
        R = [[F(0)]]
        s = observation_covariance(Sigma, A, R)
        estimate, variance = condition(mu, c, Sigma, F(0), A, y, R, s)
        self.assertEqual(estimate, F(1))
        self.assertEqual(variance, F(4))


if __name__ == "__main__":
    unittest.main()
