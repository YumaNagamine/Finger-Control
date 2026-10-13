from __future__ import annotations

import unittest

from controller.servo.magneticEncoderPos_control import (
    MagneticEncoderPositionControlConfig,
    MagneticEncoderPositionFeedback,
    MagneticEncoderPositionPIDController,
    MagneticEncoderPositionSafetyError,
    PID,
    PIDConfig,
    encoder_position_from_sample,
)


class EncoderPositionControllerTest(unittest.TestCase):
    def _controller(self, **overrides):
        settings = {
            "target_position_deg": 10.0,
            "max_speed": 100.0,
            "hard_position_error_deg": 30.0,
        }
        settings.update(overrides)
        return MagneticEncoderPositionPIDController(
            PID(
                PIDConfig(
                    kp=2.0,
                    ki=0.0,
                    kd=0.0,
                    output_min=-100.0,
                    output_max=100.0,
                )
            ),
            MagneticEncoderPositionControlConfig(**settings),
        )

    @staticmethod
    def _feedback(position=5.0, **overrides):
        values = {
            "position_deg": position,
            "servo_age_s": 0.0,
            "encoder_age_s": 0.0,
            "status": "OK",
        }
        values.update(overrides)
        return MagneticEncoderPositionFeedback(**values)

    def test_position_error_becomes_speed_command(self):
        output = self._controller().step(self._feedback(), now=0.0)

        self.assertEqual(output.position_error_deg, 5.0)
        self.assertEqual(output.speed_command, 10)

    def test_encoder_position_uses_continuous_multi_turn_value(self):
        self.assertEqual(
            encoder_position_from_sample(
                {"status": "OK", "relative_multi_turn_deg": 721.5}
            ),
            721.5,
        )
        self.assertEqual(
            encoder_position_from_sample(
                {"status": "OK", "relative_multi_turn_deg": 721.5},
                encoder_direction=-1.0,
            ),
            -721.5,
        )

    def test_single_turn_value_is_not_used_as_position_fallback(self):
        with self.assertRaises(MagneticEncoderPositionSafetyError):
            encoder_position_from_sample({"status": "OK", "relative_deg": 5.0})

    def test_uncertain_multi_turn_status_stops_control(self):
        with self.assertRaises(MagneticEncoderPositionSafetyError):
            self._controller().step(
                self._feedback(status="MULTITURN_UNCERTAIN"),
                now=0.0,
            )

    def test_hard_position_error_stops_control(self):
        with self.assertRaises(MagneticEncoderPositionSafetyError):
            self._controller(hard_position_error_deg=5.0).step(
                self._feedback(position=0.0),
                now=0.0,
            )

    def test_position_limit_blocks_outward_speed(self):
        controller = self._controller(max_position_deg=5.0)
        output = controller.step(self._feedback(position=5.0), now=0.0)

        self.assertEqual(output.speed_command, 0)
        self.assertTrue(output.limited_by_position)

    def test_stale_encoder_feedback_stops_control(self):
        with self.assertRaises(MagneticEncoderPositionSafetyError):
            self._controller(max_feedback_age_s=0.1).step(
                self._feedback(encoder_age_s=0.2),
                now=0.0,
            )


if __name__ == "__main__":
    unittest.main()
