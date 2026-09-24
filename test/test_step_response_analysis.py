from __future__ import annotations

import math
import argparse
import sys
from pathlib import Path
from types import SimpleNamespace
from unittest import TestCase

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from controller.step_response_analysis import (
    TimedScalar,
    TimedServoSample,
    detect_joint_settling,
    detect_onset,
    detect_servo_settling,
    robust_noise,
    threshold_from_noise,
)
from controller.measure_servo_dlc_step_response import (
    analyse_step_response,
    resolve_analysis_args,
)


class StepResponseAnalysisTest(TestCase):
    def test_analysis_defaults_are_loaded_from_config_and_cli_wins(self):
        args = argparse.Namespace(
            observation_timeout_s=None,
            baseline_sample_count=None,
            servo_onset_threshold_count=7.0,
            joint_onset_threshold_deg=None,
            consecutive_detection_count=None,
            noise_scale=None,
            servo_settling_band_count=None,
            servo_speed_threshold=None,
            joint_settling_band_deg=None,
            joint_settling_slope_deg=None,
            settling_consecutive_count=None,
            settling_terminal_window=None,
        )
        config = {
            "step_response": {
                "observation_timeout_s": 6.0,
                "baseline_sample_count": 12,
                "detection": {
                    "servo_onset_threshold_count": 3.0,
                    "joint_onset_threshold_deg": 1.5,
                    "consecutive_count": 4,
                    "noise_scale": 2.0,
                },
                "settling": {
                    "servo_position_band_count": 9.0,
                    "servo_speed_threshold": 4.0,
                    "joint_position_band_deg": 1.0,
                    "joint_slope_threshold_deg": 0.25,
                    "consecutive_count": 6,
                    "terminal_window": 8,
                },
            }
        }
        resolve_analysis_args(args, config)
        self.assertEqual(args.observation_timeout_s, 6.0)
        self.assertEqual(args.servo_onset_threshold_count, 7.0)
        self.assertEqual(args.settling_terminal_window, 8)

    def test_onset_requires_consecutive_samples(self):
        detection = detect_onset(
            [
                TimedScalar(0.0, 100.0),
                TimedScalar(0.1, 101.0),
                TimedScalar(0.2, 105.0),
                TimedScalar(0.3, 106.0),
                TimedScalar(0.4, 107.0),
            ],
            baseline=100.0,
            threshold=3.0,
            consecutive_count=3,
            direction=1,
        )
        self.assertIsNotNone(detection)
        assert detection is not None
        self.assertEqual((detection.first_at, detection.confirmed_at), (0.2, 0.4))

    def test_opposite_direction_does_not_trigger(self):
        self.assertIsNone(
            detect_onset(
                [TimedScalar(0.1, 99.0), TimedScalar(0.2, 98.0)],
                baseline=100.0,
                threshold=3.0,
                consecutive_count=2,
                direction=1,
            )
        )

    def test_zero_threshold_does_not_treat_stationary_samples_as_motion(self):
        self.assertIsNone(
            detect_onset(
                [TimedScalar(0.1, 100.0), TimedScalar(0.2, 100.0)],
                baseline=100.0,
                threshold=0.0,
                consecutive_count=1,
            )
        )

    def test_noise_threshold_uses_configured_minimum(self):
        noise = robust_noise([10.0, 10.0, 10.0, 11.0, 9.0])
        self.assertGreaterEqual(noise, 0.0)
        self.assertEqual(threshold_from_noise(2.0, noise, 3.0), 2.0)

    def test_servo_settling_requires_target_and_low_speed(self):
        detection = detect_servo_settling(
            [
                TimedServoSample(0.1, 101.0, 20.0),
                TimedServoSample(0.2, 110.0, 4.0),
                TimedServoSample(0.3, 110.0, 2.0),
                TimedServoSample(0.4, 111.0, 1.0),
            ],
            onset_at=0.1,
            target_position=110.0,
            position_band=2.0,
            speed_threshold=5.0,
            consecutive_count=3,
        )
        self.assertIsNotNone(detection)
        assert detection is not None
        self.assertEqual(detection.first_at, 0.2)

    def test_joint_settling_uses_terminal_plateau(self):
        detection = detect_joint_settling(
            [
                TimedScalar(0.1, math.radians(0.0)),
                TimedScalar(0.2, math.radians(5.0)),
                TimedScalar(0.3, math.radians(10.0)),
                TimedScalar(0.4, math.radians(10.1)),
                TimedScalar(0.5, math.radians(10.0)),
                TimedScalar(0.6, math.radians(10.1)),
            ],
            onset_at=0.2,
            terminal_window=3,
            position_band=math.radians(0.2),
            slope_threshold=math.radians(0.2),
            consecutive_count=3,
        )
        self.assertIsNotNone(detection)
        assert detection is not None
        self.assertEqual(detection.first_at, 0.3)

    def test_overall_joint_settling_uses_last_joint_to_settle(self):
        args = SimpleNamespace(
            movement_direction="increase",
            servo_onset_threshold_count=1.0,
            noise_scale=0.0,
            consecutive_detection_count=1,
            servo_settling_band_count=0.0,
            servo_speed_threshold=0.0,
            settling_consecutive_count=1,
            joint_onset_threshold_deg=1.0,
            settling_terminal_window=2,
            joint_settling_band_deg=0.1,
            joint_settling_slope_deg=0.1,
        )
        measurements = [
            SimpleNamespace(
                captured_at=-1.0,
                inference_finished_at=-0.9,
                flexion_angles_rad=(0.0, 0.0, 0.0),
                valid=True,
                held_last=False,
            ),
            SimpleNamespace(
                captured_at=1.0,
                inference_finished_at=1.1,
                flexion_angles_rad=(0.2, 0.0, 0.0),
                valid=True,
                held_last=False,
            ),
            SimpleNamespace(
                captured_at=2.0,
                inference_finished_at=2.1,
                flexion_angles_rad=(0.2, 0.3, 0.0),
                valid=True,
                held_last=False,
            ),
            SimpleNamespace(
                captured_at=3.0,
                inference_finished_at=3.1,
                flexion_angles_rad=(0.2, 0.3, 0.4),
                valid=True,
                held_last=False,
            ),
            SimpleNamespace(
                captured_at=4.0,
                inference_finished_at=4.1,
                flexion_angles_rad=(0.2, 0.3, 0.4),
                valid=True,
                held_last=False,
            ),
        ]
        telemetry = [
            SimpleNamespace(received_at=-1.0, positions=(0,), speeds=(0,)),
            SimpleNamespace(received_at=1.0, positions=(1,), speeds=(0,)),
        ]

        result = analyse_step_response(
            args=args,
            measurements=measurements,
            telemetry_records=telemetry,
            commanded_at=0.0,
            command_requested_at=0.0,
            return_requested_at=None,
            servo_id=0,
            start_position=0,
            target_position=1,
        )

        self.assertEqual(result["joint_settled_at"], 3.0)
        self.assertEqual(result["joint_settling_time_ms"], 3000.0)
