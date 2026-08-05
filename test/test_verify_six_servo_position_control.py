import unittest

from controller.verify_six_servo_position_control import (
    build_targets,
    first_command_misses,
)


class VerifySixServoPositionControlTest(unittest.TestCase):
    def test_builds_one_target_per_servo(self) -> None:
        self.assertEqual(
            build_targets((100, 200, 300, 400, 500, 600), 100),
            (200, 300, 400, 500, 600, 700),
        )

    def test_rejects_target_outside_multiturn_range(self) -> None:
        with self.assertRaises(ValueError):
            build_targets((28600, 0, 0, 0, 0, 0), 100)

    def test_identifies_only_servos_that_did_not_start(self) -> None:
        self.assertEqual(
            first_command_misses(
                (100, 100, 100, 100, 100, 100),
                (200, 200, 200, 200, 200, 200),
                (103, 102, 150, 200, 110, 99),
                start_min_delta=3,
                tolerance=10,
            ),
            (1, 5),
        )

    def test_start_detection_handles_negative_direction(self) -> None:
        self.assertEqual(
            first_command_misses(
                (200, 200, 200, 200, 200, 200),
                (100, 100, 100, 100, 100, 100),
                (197, 198, 150, 100, 190, 201),
                start_min_delta=3,
                tolerance=10,
            ),
            (1, 5),
        )


if __name__ == "__main__":
    unittest.main()
