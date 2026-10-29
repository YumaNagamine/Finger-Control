from __future__ import annotations

import math
import unittest

from controller.rl.action_safety_filter import (
    ActionSafetyConfig,
    ActionSafetyContext,
    ActionSafetyFilter,
)
from controller.rl.control_interface import (
    ControlMode,
    IncrementalExcursionController,
)
from controller.servo_mapping import TENDONS


def _mapping(value: float) -> dict[str, float]:
    return {tendon: value for tendon in TENDONS}


def _filter(*, reject_projection: bool = False) -> ActionSafetyFilter:
    return ActionSafetyFilter(
        ActionSafetyConfig(
            min_action_mm_by_tendon=_mapping(-1.0),
            max_action_mm_by_tendon=_mapping(1.0),
            max_action_delta_mm_by_tendon=_mapping(0.25),
            require_force=False,
            reject_on_action_projection=reject_projection,
        )
    )


class IncrementalExcursionControllerTest(unittest.TestCase):
    def test_increment_accumulates_from_previous_pid_target(self):
        controller = IncrementalExcursionController(safety_filter=_filter())
        first = controller.step(
            [0.1] * 6,
            safety_context=ActionSafetyContext(),
            communication_ok=True,
        )
        second = controller.step(
            [0.1] * 6,
            safety_context=ActionSafetyContext(),
            communication_ok=True,
        )
        self.assertEqual(first.pid_reflected_delta_mm, (0.1,) * 6)
        self.assertEqual(second.previous_pid_target_mm, (0.1,) * 6)
        self.assertTrue(all(abs(value - 0.2) < 1e-12 for value in second.filtered_target_mm))

    def test_increment_and_absolute_limits_are_both_applied(self):
        controller = IncrementalExcursionController(safety_filter=_filter())
        limited = controller.step(
            [2.0] * 6,
            safety_context=ActionSafetyContext(),
            communication_ok=True,
        )
        self.assertFalse(limited.rejected)
        self.assertTrue(limited.intervened)
        self.assertEqual(limited.pid_reflected_delta_mm, (0.25,) * 6)
        for _ in range(4):
            limited = controller.step(
                [0.25] * 6,
                safety_context=ActionSafetyContext(),
                communication_ok=True,
            )
        self.assertEqual(controller.pid_target_mm, (1.0,) * 6)
        self.assertEqual(limited.pid_reflected_delta_mm, (0.0,) * 6)

    def test_projection_can_fail_closed_without_advancing_state(self):
        controller = IncrementalExcursionController(
            safety_filter=_filter(reject_projection=True)
        )
        result = controller.step(
            [0.5] * 6,
            safety_context=ActionSafetyContext(),
            communication_ok=True,
        )
        self.assertTrue(result.rejected)
        self.assertEqual(controller.pid_target_mm, (0.0,) * 6)
        self.assertEqual(result.pid_reflected_delta_mm, (0.0,) * 6)

    def test_missing_nonfinite_action_and_communication_fail_safe(self):
        controller = IncrementalExcursionController(safety_filter=_filter())
        for action, communication in (
            (None, True),
            ([math.nan] * 6, True),
            ([0.0] * 6, False),
            ([0.0] * 6, None),
        ):
            result = controller.step(
                action,
                safety_context=ActionSafetyContext(),
                communication_ok=communication,
            )
            self.assertTrue(result.rejected)
            self.assertEqual(result.filtered_target_mm, (0.0,) * 6)

    def test_residual_mode_is_available_without_changing_direct_contract(self):
        controller = IncrementalExcursionController(
            safety_filter=_filter(),
            mode=ControlMode.RESIDUAL,
        )
        result = controller.step(
            [0.1] * 6,
            base_target_mm=[0.2] * 6,
            safety_context=ActionSafetyContext(),
            communication_ok=True,
        )
        self.assertTrue(all(abs(value - 0.25) < 1e-12 for value in result.filtered_target_mm))


if __name__ == "__main__":
    unittest.main()

