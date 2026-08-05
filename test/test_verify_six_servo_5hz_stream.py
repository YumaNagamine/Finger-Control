import unittest

from controller.verify_six_servo_5hz_stream import build_sine_targets


class VerifySixServo5HzStreamTest(unittest.TestCase):
    def test_sine_targets_start_at_baseline(self) -> None:
        starts = (100, 200, 300, 400, 500, 600)
        self.assertEqual(build_sine_targets(starts, 100, 0.0, 4.0), starts)

    def test_sine_targets_reach_positive_and_negative_amplitude(self) -> None:
        starts = (100, 200, 300, 400, 500, 600)
        self.assertEqual(
            build_sine_targets(starts, 100, 1.0, 4.0),
            (200, 300, 400, 500, 600, 700),
        )
        self.assertEqual(
            build_sine_targets(starts, 100, 3.0, 4.0),
            (0, 100, 200, 300, 400, 500),
        )

    def test_sine_targets_reject_multiturn_overflow(self) -> None:
        with self.assertRaises(ValueError):
            build_sine_targets((28650, 0, 0, 0, 0, 0), 100, 1.0, 4.0)


if __name__ == "__main__":
    unittest.main()
