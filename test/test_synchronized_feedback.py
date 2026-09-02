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

    def test_none_joint_correction_limit_leaves_proportional_correction_unclipped(self) -> None:
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
            max_joint_correction_rad=None,
            max_excursion_correction_mm=(100.0,) * len(TENDONS),
        )

        result = controller.compute(
            (1.0, 1.0, 1.0),
            (0.0, 0.0, 0.0),
            ("flexion",) * len(JOINTS),
        )

        np.testing.assert_allclose(result.joint_correction_rad, (1.0, 1.0, 1.0))

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

        self.assertEqual(first.nominal_target_positions, (100,) * len(TENDONS))
        self.assertEqual(first.target_positions, (101,) * len(TENDONS))
        self.assertEqual(second.target_positions, (101,) * len(TENDONS))

    def test_builder_reports_separate_nominal_and_feedback_targets(self) -> None:
        builder = FeedbackCommandBuilder(
            start_positions=(100,) * len(TENDONS),
            initial_excursions_mm=(0.0,) * len(TENDONS),
            position_units_per_mm=(10.0,) * len(TENDONS),
            position_limits=((0, 4095),) * len(TENDONS),
            max_position_step=(100,) * len(TENDONS),
        )

        command = builder.build(
            (0.2,) * len(TENDONS),
            (0.3,) * len(TENDONS),
        )

        self.assertEqual(command.nominal_target_positions, (102,) * len(TENDONS))
        self.assertEqual(command.target_positions, (105,) * len(TENDONS))

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
            nominal_target_positions=(100,) * len(TENDONS),
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

    def test_loop_reuses_previous_feedback_for_held_measurement(self) -> None:
        class FakeClock:
            def __init__(self) -> None:
                self.current = 0.0

            def monotonic(self) -> float:
                return self.current

            def sleep(self, duration_s: float) -> None:
                self.current += max(0.0, duration_s)

        clock = FakeClock()
        sample = SimpleNamespace(
            joint_angles_rad=(0.0, 0.0, 0.0),
            nominal_excursions_mm=(0.0,) * len(TENDONS),
            motion_directions=("flexion",) * len(JOINTS),
        )
        trajectory = SimpleNamespace(
            duration_s=0.2,
            sample=lambda _elapsed_s: sample,
        )
        measurements = [
            SimpleNamespace(
                frame_index=0,
                captured_at=0.0,
                inference_finished_at=0.05,
                flexion_angles_rad=(0.0, 0.0, 0.0),
                min_likelihood=1.0,
                inference_ms=50.0,
                valid=True,
                reason=None,
                held_last=False,
            ),
            SimpleNamespace(
                frame_index=1,
                captured_at=0.2,
                inference_finished_at=0.25,
                flexion_angles_rad=(0.0, 0.0, 0.0),
                min_likelihood=1.0,
                inference_ms=50.0,
                valid=True,
                reason="temporal DLC outlier",
                held_last=True,
            ),
        ]

        def read_measurement(frame_index: int):
            return measurements[frame_index]

        angle_source = SimpleNamespace(read=read_measurement)
        feedback_value = SimpleNamespace(
            error_rad=(0.1, 0.0, 0.0),
            excursion_correction_mm=(0.25,) * len(TENDONS),
        )
        compute_calls: list[object] = []

        def compute_feedback(*_args):
            compute_calls.append(True)
            return feedback_value

        feedback_controller = SimpleNamespace(compute=compute_feedback)
        commands: list[tuple[float, ...]] = []

        def build_command(_nominal, feedback_excursions):
            commands.append(tuple(feedback_excursions))
            return SimpleNamespace(
                nominal_excursions_mm=(0.0,) * len(TENDONS),
                feedback_excursions_mm=tuple(feedback_excursions),
                total_excursions_mm=tuple(feedback_excursions),
                nominal_target_positions=(100,) * len(TENDONS),
                target_positions=(100,) * len(TENDONS),
            )

        command_builder = SimpleNamespace(build=build_command)

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

        self.assertEqual(len(compute_calls), 1)
        self.assertEqual(len(commands), 2)
        self.assertEqual(commands[0], commands[1])
        self.assertEqual(logger.rows[1]["measurement_held_last"], 1)

    def test_feedback_excursion_plot_is_saved_from_logged_positions(self) -> None:
        with tempfile.TemporaryDirectory() as raw_directory:
            directory = Path(raw_directory)
            metadata = feedback_main._feedback_log_metadata(
                (100,) * len(TENDONS),
                (0.0,) * len(TENDONS),
                (10.0,) * len(TENDONS),
            )
            logger = feedback_main.CycleLogger(
                directory,
                session_id="plot_test",
                fixed_fields=metadata,
            )
            row: dict[str, object] = {
                "scheduled_s": 0.0,
                "captured_s": 0.05,
                "commanded_s": 0.2,
                "telemetry_received_s": 0.19,
                "measurement_held_last": 1,
            }
            for joint in JOINTS:
                row[f"{joint}_reference_rad"] = 0.2
                row[f"{joint}_measured_rad"] = 0.1
            for tendon in TENDONS:
                row[f"{tendon}_nominal_excursion_mm"] = 0.2
                row[f"{tendon}_nominal_target_position"] = 102
                row[f"{tendon}_target_position"] = 105
                row[f"{tendon}_actual_position"] = 104
            logger.write(row)
            logger.close()

            output_path = feedback_main.plot_feedback_excursion_log(logger.path)
            joint_output_path = feedback_main.plot_feedback_joint_angle_log(
                logger.path
            )

            self.assertTrue(output_path.is_file())
            self.assertGreater(output_path.stat().st_size, 0)
            self.assertTrue(joint_output_path.is_file())
            self.assertGreater(joint_output_path.stat().st_size, 0)

if __name__ == "__main__":
    unittest.main()
