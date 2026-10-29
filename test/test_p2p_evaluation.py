from __future__ import annotations

import unittest

import numpy as np

from controller.rl.p2p_evaluation import (
    P2PEvaluationConfig,
    evaluate_p2p_trial,
    summarize_p2p_trials,
)


class P2PEvaluationTest(unittest.TestCase):
    def test_final_tracking_and_holding_metrics_are_separate(self):
        times = np.asarray([0.0, 0.5, 1.0, 1.5])
        measured = np.asarray(
            [[0.0, 0.0, 0.0], [0.4, 0.4, 0.4], [0.49, 0.5, 0.5], [0.5, 0.5, 0.5]]
        )
        target = np.linspace(np.zeros(3), np.full(3, 0.5), len(times))
        result = evaluate_p2p_trial(
            experiment_id="trial-1",
            controller_type="ff_fb",
            timestamps_s=times,
            measured_joint_rad=measured,
            goal_joint_rad=[0.5, 0.5, 0.5],
            target_trajectory_rad=target,
            tendon_tension_n=np.ones((4, 6)),
            feedback_correction_mm=np.full((4, 6), 0.1),
            safety_violation_count=0,
            config=P2PEvaluationConfig(0.02, 0.02, 0.5, 0.5),
        )
        self.assertEqual(result["evaluation_quality"], "valid")
        self.assertTrue(result["goal_reached"])
        self.assertEqual(result["settling_time_s"], 1.0)
        self.assertIsNotNone(result["trajectory_tracking_rmse_rad"])
        self.assertAlmostEqual(result["max_tendon_tension_n"], 1.0)

    def test_time_misalignment_marks_trial_invalid(self):
        result = evaluate_p2p_trial(
            experiment_id="bad",
            controller_type="ff",
            timestamps_s=[0.0, 0.0],
            measured_joint_rad=[[0.0] * 3] * 2,
            goal_joint_rad=[0.0] * 3,
            target_trajectory_rad=None,
            tendon_tension_n=None,
            feedback_correction_mm=None,
            safety_violation_count=0,
            config=P2PEvaluationConfig(0.1, 0.1, 0.1, 0.1),
        )
        self.assertEqual(result["evaluation_quality"], "invalid")

    def test_summary_excludes_invalid_trials(self):
        summary = summarize_p2p_trials(
            [
                {
                    "controller_type": "ff",
                    "evaluation_quality": "valid",
                    "goal_reached": True,
                    "integrated_final_posture_error_rad": 0.1,
                    "safety_violation_count": 1,
                },
                {"controller_type": "ff", "evaluation_quality": "invalid"},
            ]
        )
        self.assertEqual(summary["controllers"]["ff"]["trial_count"], 1)
        self.assertEqual(summary["invalid_trial_count"], 1)


if __name__ == "__main__":
    unittest.main()

