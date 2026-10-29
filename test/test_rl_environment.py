from __future__ import annotations

import unittest

from controller.rl.control_interface import IncrementalExcursionController
from controller.rl.environment import GoalConditionedMockEnv, RewardConfig


class GoalConditionedMockEnvTest(unittest.TestCase):
    def test_reset_and_step_use_24d_observation_and_incremental_action(self):
        events = []
        env = GoalConditionedMockEnv(
            controller=IncrementalExcursionController(safety_filter=None),
            reward_config=RewardConfig(horizon_steps=2, success_dwell_steps=10),
            event_sink=lambda name, payload: events.append((name, payload)),
        )
        observation, info = env.reset(
            initial_joint_rad=[0.0, 0.0, 0.0],
            goal_joint_rad=[0.5, 0.5, 0.5],
        )
        self.assertEqual(observation.shape, (24,))
        self.assertTrue(info["mock_physics_not_real"])
        next_observation, reward, terminated, truncated, step_info = env.step([0.1] * 6)
        self.assertEqual(next_observation.shape, (24,))
        self.assertLess(reward, 0.0)
        self.assertFalse(terminated)
        self.assertFalse(truncated)
        self.assertEqual(step_info["action_semantics"], "incremental_excursion_mm")
        _, _, _, truncated, _ = env.step([0.0] * 6)
        self.assertTrue(truncated)
        self.assertEqual([name for name, _ in events], ["reset", "step", "step"])

    def test_missing_action_is_rejected_without_plant_update(self):
        env = GoalConditionedMockEnv(
            controller=IncrementalExcursionController(safety_filter=None)
        )
        before, _ = env.reset(
            initial_joint_rad=[0.0, 0.0, 0.0],
            goal_joint_rad=[1.0, 1.0, 1.0],
        )
        after, _, _, _, info = env.step(None)
        self.assertTrue(info["action_rejected"])
        self.assertEqual(before[6:18].tolist(), after[6:18].tolist())


if __name__ == "__main__":
    unittest.main()

