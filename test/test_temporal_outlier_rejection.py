from __future__ import annotations

import math
import unittest

from observation.vision.realtime_joint_angle_source import TemporalAngleGate


class TemporalAngleGateTest(unittest.TestCase):
    def test_accepts_small_changes_and_holds_one_outlier(self) -> None:
        gate = TemporalAngleGate(jump_threshold_deg=10.0, max_hold_frames=1)

        first = gate.apply((0.0, 0.0, 0.0))
        self.assertTrue(first.valid)
        self.assertFalse(first.held_last)

        accepted = gate.apply((math.radians(5.0), 0.0, 0.0))
        self.assertTrue(accepted.valid)
        self.assertFalse(accepted.held_last)

        outlier = gate.apply((math.radians(40.0), 0.0, 0.0))
        self.assertTrue(outlier.valid)
        self.assertTrue(outlier.held_last)
        self.assertEqual(outlier.angles_rad, accepted.angles_rad)
        self.assertIn("DIP=35.0deg", outlier.reason or "")

    def test_rejects_a_second_consecutive_outlier(self) -> None:
        gate = TemporalAngleGate(jump_threshold_deg=10.0, max_hold_frames=1)
        gate.apply((0.0, 0.0, 0.0))

        first = gate.apply((math.radians(30.0), 0.0, 0.0))
        second = gate.apply((math.radians(30.0), 0.0, 0.0))

        self.assertTrue(first.valid)
        self.assertTrue(first.held_last)
        self.assertFalse(second.valid)
        self.assertFalse(second.held_last)

    def test_accepts_after_returning_to_the_last_valid_value(self) -> None:
        gate = TemporalAngleGate(jump_threshold_deg=10.0, max_hold_frames=1)
        gate.apply((0.0, 0.0, 0.0))
        gate.apply((math.radians(30.0), 0.0, 0.0))

        recovered = gate.apply((math.radians(5.0), 0.0, 0.0))

        self.assertTrue(recovered.valid)
        self.assertFalse(recovered.held_last)
        self.assertEqual(recovered.angles_rad[0], math.radians(5.0))


if __name__ == "__main__":
    unittest.main()
