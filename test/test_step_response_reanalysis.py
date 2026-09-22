from __future__ import annotations

import argparse
import csv
import json
import sys
import tempfile
from pathlib import Path
from unittest import TestCase

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from controller.reanalyze_servo_dlc_step_response import reanalyse_trial


class StepResponseReanalysisTest(TestCase):
    def test_saved_raw_trial_can_be_reanalysed(self):
        with tempfile.TemporaryDirectory() as temporary_directory:
            trial_directory = Path(temporary_directory)
            analysis_args = {
                "servo_onset_threshold_count": 3.0,
                "joint_onset_threshold_deg": 2.0,
                "consecutive_detection_count": 2,
                "noise_scale": 0.0,
                "servo_settling_band_count": 1.0,
                "servo_speed_threshold": 5.0,
                "joint_settling_band_deg": 1.0,
                "joint_settling_slope_deg": 1.0,
                "settling_consecutive_count": 2,
                "settling_terminal_window": 2,
            }
            metadata = {
                "servo_id": 0,
                "movement_direction": "increase",
                "start_position": 0,
                "target_position": 5,
                "command_requested_at": 0.1,
                "commanded_at": 0.1,
                "return_requested_at": None,
                "measurement_args": analysis_args,
            }
            (trial_directory / "trial.json").write_text(
                json.dumps(metadata), encoding="utf-8"
            )

            with (trial_directory / "telemetry.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                writer = csv.DictWriter(
                    handle,
                    fieldnames=("received_at", "positions_json", "speeds_json"),
                )
                writer.writeheader()
                for at, position, speed in (
                    (0.0, 0, 0),
                    (0.05, 0, 0),
                    (0.2, 1, 2),
                    (0.3, 4, 1),
                    (0.4, 5, 0),
                ):
                    writer.writerow(
                        {
                            "received_at": at,
                            "positions_json": json.dumps([position]),
                            "speeds_json": json.dumps([speed]),
                        }
                    )

            with (trial_directory / "vision.csv").open(
                "w", newline="", encoding="utf-8"
            ) as handle:
                fieldnames = (
                    "captured_at",
                    "inference_finished_at",
                    "DIP_angle_rad",
                    "PIP_angle_rad",
                    "MCP_angle_rad",
                    "valid",
                    "held_last",
                )
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                for captured_at, available_at, angle in (
                    (0.0, 0.02, 0.0),
                    (0.05, 0.07, 0.0),
                    (0.2, 0.22, 0.0),
                    (0.3, 0.32, 0.1),
                    (0.4, 0.42, 0.11),
                ):
                    writer.writerow(
                        {
                            "captured_at": captured_at,
                            "inference_finished_at": available_at,
                            "DIP_angle_rad": angle,
                            "PIP_angle_rad": angle,
                            "MCP_angle_rad": angle,
                            "valid": 1,
                            "held_last": 0,
                        }
                    )

            overrides = argparse.Namespace(
                **{name: None for name in analysis_args}
            )
            result = reanalyse_trial(trial_directory, overrides)

            self.assertAlmostEqual(
                result["analysis"]["servo_onset_latency_ms"], 200.0
            )
            self.assertAlmostEqual(
                result["analysis"]["joint_onset_latency_ms"], 200.0
            )
            self.assertIn("not fingertip or grip force", result["quantity_note"])
