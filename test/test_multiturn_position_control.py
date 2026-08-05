from __future__ import annotations

import unittest
from unittest.mock import patch

from controller.verify_multiturn_position_control import (
    build_boundary_test_plan,
    shortest_delta_to_raw,
    validate_crossings,
    wrapped_position,
)
from servo.servo_APIs import ServoAPI


class MultiturnPositionModelTest(unittest.TestCase):
    def test_wrapped_position_uses_zero_based_4096_counts(self) -> None:
        self.assertEqual(wrapped_position(5000), 904)
        self.assertEqual(wrapped_position(-1), 4095)

    def test_plan_crosses_both_directions(self) -> None:
        plan = build_boundary_test_plan(4000, start_raw=4000, delta=1000)

        self.assertEqual(plan.setup_position, 4000)
        self.assertEqual(plan.forward_target, 5000)
        self.assertEqual(wrapped_position(plan.forward_target), 904)
        self.assertEqual(plan.reverse_target, 4000)
        validate_crossings(plan, 5000, 4000, tolerance=0)

    def test_setup_uses_shortest_path_to_requested_raw_position(self) -> None:
        self.assertEqual(shortest_delta_to_raw(100, 4000), -196)
        plan = build_boundary_test_plan(100, start_raw=4000, delta=1000)
        self.assertEqual(plan.setup_position, -96)
        self.assertEqual(wrapped_position(plan.setup_position), 4000)

    def test_plan_requires_forward_boundary_crossing(self) -> None:
        with self.assertRaises(ValueError):
            build_boundary_test_plan(1000, start_raw=1000, delta=1000)


class ServoAPIMultiturnCommandTest(unittest.TestCase):
    def setUp(self) -> None:
        self.api = ServoAPI(auto_open=False)

    def test_sends_absolute_multiturn_command(self) -> None:
        with patch.object(self.api, "send_raw") as send_raw:
            self.api.set_multiturn_position(2, 5000, 200)

        send_raw.assert_called_once_with("ma,2,5000,200")

    def test_sends_negative_relative_multiturn_command(self) -> None:
        with patch.object(self.api, "send_raw") as send_raw:
            self.api.move_relative(2, -1000, 0)

        send_raw.assert_called_once_with("mr,2,-1000,0")

    def test_rejects_unrepresentable_relative_step(self) -> None:
        with self.assertRaises(ValueError):
            self.api.move_relative(2, -32768, 0)


if __name__ == "__main__":
    unittest.main()
