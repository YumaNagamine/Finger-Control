from __future__ import annotations

import csv
import io
import json
import tempfile
import unittest
from contextlib import redirect_stdout
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

import numpy as np
from controller.csv_player import play_excursion_csv_with_dlc_feedback as feedback_main

from controller.feedback import (
    FeedbackCommandBuilder,
    FeedbackTrajectory,
    JointFeedbackController,
    MomentArmRuntime,
)
from controller.feedback.moment_arm_runtime import JOINTS, MOTIONS
from controller.servo_mapping import TENDONS


class SynchronizedFeedbackTest(unittest.TestCase):
    def test_trajectory_interpolates_reference_and_nominal_excursion(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            path = Path(raw_directory) / "prediction.csv"
            fieldnames = [
                "elapsed_s",
                *(f"{joint}_flexion_smoothed_angle_rad" for joint in JOINTS),
                *(f"{tendon}_predicted_excursion_mm" for tendon in TENDONS),
            ]
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.DictWriter(handle, fieldnames=fieldnames)
                writer.writeheader()
                writer.writerow(
                    {
                        "elapsed_s": 10.0,
                        **{
                            f"{joint}_flexion_smoothed_angle_rad": 0.0
                            for joint in JOINTS
                        },
                        **{
                            f"{tendon}_predicted_excursion_mm": 0.0
                            for tendon in TENDONS
                        },
                    }
                )
                writer.writerow(
                    {
                        "elapsed_s": 12.0,
                        **{
                            f"{joint}_flexion_smoothed_angle_rad": 0.2
                            for joint in JOINTS
                        },
                        **{
                            f"{tendon}_predicted_excursion_mm": 2.0
                            for tendon in TENDONS
                        },
                    }
                )

            sample = FeedbackTrajectory.from_csv(path).sample(1.0)

        np.testing.assert_allclose(sample.joint_angles_rad, (0.1, 0.1, 0.1))
        np.testing.assert_allclose(
            sample.nominal_excursions_mm,
            np.ones(len(TENDONS)),
        )
        self.assertEqual(sample.motion_directions, ("flexion",) * len(JOINTS))

    def test_moment_arm_maps_joint_correction_to_each_tendon(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            root = Path(raw_directory)
            for joint in JOINTS:
                for motion in MOTIONS:
                    payload = {
                        "moment_arm_coefficients": {
                            tendon: [1.0] for tendon in TENDONS
                        }
                    }
                    (root / f"moment_arm_{joint}_{motion}_average.json").write_text(
                        json.dumps(payload),
                        encoding="utf-8",
                    )
            runtime = MomentArmRuntime.from_directory(root)

        result = runtime.correction_excursion(
            (0.1, 0.2, 0.3),
            (0.01, 0.02, 0.03),
            ("flexion", "extension", "flexion"),
        )
        np.testing.assert_allclose(result, np.full(len(TENDONS), 0.06))

    def test_zero_joint_error_produces_zero_feedback(self) -> None:
        coefficients = {
            (joint, motion): {
                tendon: np.asarray([1.0]) for tendon in TENDONS
            }
            for joint in JOINTS
            for motion in MOTIONS
        }
        controller = JointFeedbackController(
            MomentArmRuntime(coefficients),
            kp=(1.0, 1.0, 1.0),
            max_joint_correction_rad=(0.1, 0.1, 0.1),
            max_excursion_correction_mm=(1.0,) * len(TENDONS),
        )

        result = controller.compute(
            (0.2, 0.3, 0.4),
            (0.2, 0.3, 0.4),
            ("flexion",) * len(JOINTS),
        )

        np.testing.assert_allclose(result.excursion_correction_mm, np.zeros(6))

    def test_feedback_excursion_is_not_accumulated_between_cycles(self) -> None:
        builder = FeedbackCommandBuilder(
            start_positions=(100,) * len(TENDONS),
            initial_excursions_mm=(0.0,) * len(TENDONS),
            position_units_per_mm=(10.0,) * len(TENDONS),
            position_limits=((0, 4095),) * len(TENDONS),
            max_position_step=(100,) * len(TENDONS),
        )

        first = builder.build((0.0,) * len(TENDONS), (0.1,) * len(TENDONS))
        second = builder.build((0.0,) * len(TENDONS), (0.1,) * len(TENDONS))

        self.assertEqual(first.target_positions, (101,) * len(TENDONS))
        self.assertEqual(second.target_positions, (101,) * len(TENDONS))

    def test_loop_sends_command_at_next_period_boundary(self) -> None:
        class FakeClock:
            def __init__(self) -> None:
                self.current = 0.0

            def monotonic(self) -> float:
                return self.current

            def sleep(self, duration_s: float) -> None:
                if duration_s < 0.0:
                    raise AssertionError("sleep duration must not be negative")
                self.current += duration_s

        clock = FakeClock()
        sample = SimpleNamespace(
            joint_angles_rad=(0.0, 0.0, 0.0),
            nominal_excursions_mm=(0.0,) * len(TENDONS),
            motion_directions=("flexion",) * len(JOINTS),
        )
        trajectory = SimpleNamespace(
            duration_s=0.0,
            sample=lambda _elapsed_s: sample,
        )

        def read_measurement(frame_index: int):
            clock.current += 0.05
            return SimpleNamespace(
                frame_index=frame_index,
                captured_at=0.0,
                inference_finished_at=clock.current,
                flexion_angles_rad=(0.0, 0.0, 0.0),
                min_likelihood=1.0,
                inference_ms=50.0,
                valid=True,
                reason=None,
            )

        angle_source = SimpleNamespace(read=read_measurement)
        feedback = SimpleNamespace(
            error_rad=(0.0, 0.0, 0.0),
            excursion_correction_mm=(0.0,) * len(TENDONS),
        )
        feedback_controller = SimpleNamespace(compute=lambda *_args: feedback)
        command = SimpleNamespace(
            nominal_excursions_mm=(0.0,) * len(TENDONS),
            feedback_excursions_mm=(0.0,) * len(TENDONS),
            total_excursions_mm=(0.0,) * len(TENDONS),
            target_positions=(100,) * len(TENDONS),
        )
        command_builder = SimpleNamespace(build=lambda *_args: command)

        class MemoryLogger:
            def __init__(self) -> None:
                self.rows: list[dict[str, object]] = []

            def write(self, row: dict[str, object]) -> None:
                self.rows.append(row)

        logger = MemoryLogger()
        config = {
            "control": {
                "frequency_hz": 5.0,
                "max_cycle_overrun_s": 0.05,
                "stream_time_ms": 0,
            },
            "safety": {
                "tracking_error_limit": 500,
                "tracking_error_cycles": 3,
                "telemetry_stale_s": 0.5,
            },
        }

        with (
            patch.object(feedback_main.time, "monotonic", clock.monotonic),
            patch.object(feedback_main.time, "sleep", clock.sleep),
            redirect_stdout(io.StringIO()),
        ):
            feedback_main._run_loop(
                config=config,
                trajectory=trajectory,
                feedback_controller=feedback_controller,
                angle_source=angle_source,
                command_builder=command_builder,
                logger=logger,
            )

        self.assertEqual(len(logger.rows), 1)
        self.assertAlmostEqual(logger.rows[0]["command_deadline_s"], 0.2)
        self.assertAlmostEqual(logger.rows[0]["commanded_s"], 0.2)

if __name__ == "__main__":
    unittest.main()
