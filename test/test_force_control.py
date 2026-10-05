from __future__ import annotations

import unittest

from controller.servo.force_control import (
    ForceControlConfig,
    ForceFeedback,
    ForcePIDController,
    ForceSafetyError,
    PID,
    PIDConfig,
)


class PIDTest(unittest.TestCase):
    def test_proportional_output(self):
        pid = PID(
            PIDConfig(
                kp=2.0,
                ki=0.0,
                kd=0.0,
                output_min=-100.0,
                output_max=100.0,
            )
        )

        self.assertEqual(pid.update(10.0, 5.0, now=0.0), 10.0)

    def test_output_saturation_does_not_wind_up(self):
        pid = PID(
            PIDConfig(
                kp=0.2,
                ki=1.0,
                kd=0.0,
                output_min=-1.0,
                output_max=1.0,
                integral_min=-10.0,
                integral_max=10.0,
            )
        )

        self.assertEqual(pid.update(10.0, 0.0, now=0.0), 1.0)
        self.assertEqual(pid.update(10.0, 0.0, now=1.0), 1.0)
        self.assertEqual(pid.update(0.0, 0.0, now=2.0), 0.0)


class ForcePIDControllerTest(unittest.TestCase):
    def _controller(self, **overrides):
        settings = {
            "target_force_n": 2.0,
            "max_speed": 100.0,
            "hard_force_limit_n": 10.0,
        }
        settings.update(overrides)
        return ForcePIDController(
            PID(
                PIDConfig(
                    kp=10.0,
                    ki=0.0,
                    kd=0.0,
                    output_min=-100.0,
                    output_max=100.0,
                )
            ),
            ForceControlConfig(**settings),
        )

    def test_force_error_becomes_speed_command(self):
        controller = self._controller()
        output = controller.step(
            ForceFeedback(
                force_n=1.0,
                servo_position=100.0,
                servo_age_s=0.0,
                encoder_age_s=0.0,
            ),
            now=0.0,
        )

        self.assertEqual(output.force_error_n, 1.0)
        self.assertEqual(output.speed_command, 10)

    def test_target_force_cannot_be_below_configured_minimum(self):
        with self.assertRaisesRegex(ValueError, "min_target_force_n"):
            self._controller(target_force_n=0.4)

    def test_position_limit_blocks_outward_speed(self):
        controller = self._controller(max_position=100.0)
        output = controller.step(
            ForceFeedback(
                force_n=0.0,
                servo_position=100.0,
                servo_age_s=0.0,
                encoder_age_s=0.0,
            ),
            now=0.0,
        )

        self.assertEqual(output.speed_command, 0)
        self.assertTrue(output.limited_by_position)

    def test_stale_feedback_stops_control(self):
        controller = self._controller(max_feedback_age_s=0.1)
        with self.assertRaises(ForceSafetyError):
            controller.step(
                ForceFeedback(
                    force_n=1.0,
                    servo_position=100.0,
                    servo_age_s=0.2,
                    encoder_age_s=0.0,
                ),
                now=0.0,
            )

    def test_force_limit_stops_control(self):
        controller = self._controller(hard_force_limit_n=5.0)
        with self.assertRaises(ForceSafetyError):
            controller.step(
                ForceFeedback(
                    force_n=5.0,
                    servo_position=100.0,
                    servo_age_s=0.0,
                    encoder_age_s=0.0,
                ),
                now=0.0,
            )


if __name__ == "__main__":
    unittest.main()
