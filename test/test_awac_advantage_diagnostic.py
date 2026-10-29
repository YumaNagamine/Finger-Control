import unittest

import numpy as np

from controller.rl.awac_advantage_diagnostic import (
    AdvantageRecorder,
    action_error,
    distribution_summary,
    effective_sample_size,
)


class AWACAdvantageDiagnosticTest(unittest.TestCase):
    def test_uniform_weights_have_full_effective_sample_size(self):
        weights = np.ones(64)
        self.assertAlmostEqual(effective_sample_size(weights), 64.0)

    def test_concentrated_weights_reduce_effective_sample_size(self):
        weights = np.zeros(64)
        weights[0] = 64.0
        self.assertAlmostEqual(effective_sample_size(weights), 1.0)

    def test_distribution_and_action_error_are_finite(self):
        values = distribution_summary(np.asarray([-1.0, 0.0, 1.0]))
        self.assertTrue(values["all_finite"])
        error = action_error(np.ones((4, 6)), np.zeros((4, 6)))
        self.assertAlmostEqual(error["rmse_mm"], 1.0)
        self.assertEqual(len(error["per_tendon_rmse_mm"]), 6)


if __name__ == "__main__":
    unittest.main()
