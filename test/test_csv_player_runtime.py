from __future__ import annotations

import csv
import math
import tempfile
import unittest
from pathlib import Path

from controller.feedback.joint_feedback import JointFeedbackController
from controller.feedback.moment_arm_runtime import MomentArmRuntime
from controller.csv_player.gui_support import LatestEventMailbox
from controller.csv_player.recorder import ControlTraceWriter
from controller.csv_player.runner import PlaybackConfig, SpeedTrajectoryRunner
from controller.csv_player.speed_controller import SpeedControlConfig, SpeedController
from controller.csv_player.trajectory import (
    ExcursionTrajectory,
    validate_point_to_point_trajectory,
)
from controller.servo.hardware_session import HardwareFeedback
from controller.servo_mapping import SERVO_IDS_BY_TENDON, TENDONS


class _ConstantMomentArm:
    evaluation_ranges_rad = {
        joint: (0.0, 1.0) for joint in ("DIP", "PIP", "MCP")
    }

    @staticmethod
    def matrix(_angles):
        return [
            [2.0, 0.0, 0.0],
            [0.0, 3.0, 0.0],
            [-4.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
            [0.0, 0.0, 0.0],
        ]


def _constant_extrapolation_model() -> MomentArmRuntime:
    coefficients = {
        joint: {tendon: [2.0] for tendon in TENDONS}
        for joint in ("DIP", "PIP", "MCP")
    }
    coefficients["DIP"]["FDP"] = [1.0, 1.0]
    coefficients["PIP"]["FDP"] = [1.0, 1.0]
    coefficients["MCP"]["FDP"] = [2.0, 1.0]
    return MomentArmRuntime(
        coefficients,
        evaluation_ranges_rad={
            joint: (0.0, 1.0) for joint in ("DIP", "PIP", "MCP")
        },
    )


class CsvPlayerRuntimeTest(unittest.TestCase):
    def test_latest_event_mailbox_coalesces_each_live_kind(self):
        mailbox = LatestEventMailbox()
        mailbox.publish({"kind": "force", "sample": 1})
        mailbox.publish({"kind": "angle", "frame": 10})
        mailbox.publish({"kind": "force", "sample": 2})

        events = {event["kind"]: event for event in mailbox.drain()}

        self.assertEqual(events["force"]["sample"], 2)
        self.assertEqual(events["angle"]["frame"], 10)
        self.assertEqual(mailbox.drain(), [])

    def test_trajectory_interpolates_and_applies_signed_calibration(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "trajectory.csv"
            with path.open("w", newline="", encoding="utf-8") as handle:
                writer = csv.writer(handle)
                writer.writerow(["elapsed_s", *[f"{tendon}_predicted_excursion_mm" for tendon in TENDONS]])
                writer.writerow([0.0, 0, 0, 0, 0, 0, 0])
                writer.writerow([1.0, 1, 2, 3, 4, 5, 6])

            trajectory = ExcursionTrajectory.from_csv(
                path,
                (10, 10, 10, 10, 10, 10),
            )

            self.assertEqual(trajectory.duration_s, 1.0)
            self.assertEqual(trajectory.positions_at(0.5, (100, 100, 100, 100, 100, 100)), (105.0, 110.0, 115.0, 120.0, 125.0, 130.0))

    def test_point_to_point_equal_postures_keep_zero_relative_ff(self):
        model = _ConstantMomentArm()
        trajectory = ExcursionTrajectory.from_joint_point_to_point(
            (0.2, 0.3, 0.4),
            (0.2, 0.3, 0.4),
            2.0,
            model,
            sample_hz=10.0,
        )

        self.assertGreaterEqual(len(trajectory.elapsed_times_s), 2)
        self.assertTrue(
            all(row == (0.0,) * len(TENDONS) for row in trajectory.excursions_mm)
        )
        self.assertEqual(trajectory.target_angles_deg[0], trajectory.target_angles_deg[-1])

    def test_point_to_point_ff_sign_and_endpoint_integral(self):
        model = _ConstantMomentArm()
        q0 = (0.2, 0.3, 0.4)
        goal = (0.7, 0.1, 0.6)
        trajectory = ExcursionTrajectory.from_joint_point_to_point(
            q0, goal, 2.0, model, sample_hz=50.0
        )
        endpoint = trajectory.excursions_mm[-1]
        expected = (1.0, -0.6, -2.0, 0.0, 0.0, 0.0)

        for actual, target in zip(endpoint, expected):
            self.assertAlmostEqual(actual, target, places=10)
        self.assertAlmostEqual(trajectory.target_angles_deg[-1][0], math.degrees(goal[0]))
        self.assertAlmostEqual(trajectory.target_angles_deg[0][0], math.degrees(q0[0]))
        first_step = abs(
            trajectory.target_angles_deg[1][0] - trajectory.target_angles_deg[0][0]
        )
        middle_step = abs(
            trajectory.target_angles_deg[len(trajectory.target_angles_deg) // 2][0]
            - trajectory.target_angles_deg[len(trajectory.target_angles_deg) // 2 - 1][0]
        )
        last_step = abs(
            trajectory.target_angles_deg[-1][0]
            - trajectory.target_angles_deg[-2][0]
        )
        self.assertLess(first_step, middle_step)
        self.assertLess(last_step, middle_step)
        reverse = ExcursionTrajectory.from_joint_point_to_point(
            goal, q0, 2.0, model, sample_hz=50.0
        )
        for forward, backward in zip(endpoint, reverse.excursions_mm[-1]):
            self.assertAlmostEqual(forward, -backward, places=10)

    def test_point_to_point_same_goal_uses_each_measured_initial_posture(self):
        model = _ConstantMomentArm()
        goal = (0.8, 0.8, 0.8)
        first = ExcursionTrajectory.from_joint_point_to_point(
            (0.2, 0.3, 0.4), goal, 2.0, model, sample_hz=20.0
        )
        second = ExcursionTrajectory.from_joint_point_to_point(
            (0.4, 0.5, 0.6), goal, 2.0, model, sample_hz=20.0
        )

        self.assertNotEqual(first.excursions_mm[-1], second.excursions_mm[-1])
        self.assertEqual(first.target_angles_deg[-1], second.target_angles_deg[-1])

    def test_point_to_point_safety_preflight_checks_joint_and_decision_limits(self):
        model = _ConstantMomentArm()
        trajectory = ExcursionTrajectory.from_joint_point_to_point(
            (0.2, 0.3, 0.4),
            (0.7, 0.3, 0.4),
            2.0,
            model,
            sample_hz=50.0,
        )
        approved = {joint: (0.0, 1.0) for joint in ("DIP", "PIP", "MCP")}
        safety_range = {tendon: 5.0 for tendon in TENDONS}
        max_delta = {tendon: 0.5 for tendon in TENDONS}
        maxima = validate_point_to_point_trajectory(
            trajectory,
            approved_joint_limits_rad=approved,
            min_action_mm_by_tendon={tendon: -5.0 for tendon in TENDONS},
            max_action_mm_by_tendon=safety_range,
            max_action_delta_mm_by_tendon=max_delta,
            decision_hz=5.0,
        )
        self.assertLessEqual(maxima["FDP"], max_delta["FDP"])

        with self.assertRaisesRegex(ValueError, "decision delta"):
            validate_point_to_point_trajectory(
                trajectory,
                approved_joint_limits_rad=approved,
                min_action_mm_by_tendon={tendon: -5.0 for tendon in TENDONS},
                max_action_mm_by_tendon=safety_range,
                max_action_delta_mm_by_tendon={tendon: 0.01 for tendon in TENDONS},
                decision_hz=5.0,
            )
        with self.assertRaisesRegex(ValueError, "approved DIP"):
            validate_point_to_point_trajectory(
                trajectory,
                approved_joint_limits_rad={
                    "DIP": (0.0, 0.5), "PIP": (0.0, 1.0), "MCP": (0.0, 1.0)
                },
                min_action_mm_by_tendon={tendon: -5.0 for tendon in TENDONS},
                max_action_mm_by_tendon=safety_range,
                max_action_delta_mm_by_tendon=max_delta,
                decision_hz=5.0,
            )

    def test_point_to_point_safety_preflight_checks_intermediate_reference_rows(self):
        trajectory = ExcursionTrajectory(
            elapsed_times_s=(0.0, 0.5, 1.0),
            excursions_mm=((0.0,) * len(TENDONS),) * 3,
            target_angles_deg=(
                (0.0, 0.0, 0.0),
                (0.0, math.degrees(0.5), 0.0),
                (0.0, 0.0, 0.0),
            ),
        )
        broad_action_limits = {tendon: 1.0 for tendon in TENDONS}

        with self.assertRaisesRegex(ValueError, "reference row 1 exceeds approved PIP"):
            validate_point_to_point_trajectory(
                trajectory,
                approved_joint_limits_rad={
                    "DIP": (0.0, 1.0),
                    "PIP": (-0.2, 0.3),
                    "MCP": (0.0, 1.0),
                },
                min_action_mm_by_tendon={tendon: -1.0 for tendon in TENDONS},
                max_action_mm_by_tendon=broad_action_limits,
                max_action_delta_mm_by_tendon=broad_action_limits,
                decision_hz=1.0,
            )

    def test_point_to_point_constant_extrapolation_preserves_reference_angles(self):
        model = _constant_extrapolation_model()
        trajectory = ExcursionTrajectory.from_joint_point_to_point(
            (0.0, -0.1, 0.0),
            (0.0, 0.2, 0.0),
            1.0,
            model,
            sample_hz=1.0,
        )

        self.assertAlmostEqual(trajectory.excursions_mm[-1][0], 0.32)
        self.assertAlmostEqual(trajectory.target_angles_deg[0][1], math.degrees(-0.1))
        self.assertAlmostEqual(trajectory.target_angles_deg[-1][1], math.degrees(0.2))

        with self.assertRaisesRegex(ValueError, "finite"):
            ExcursionTrajectory.from_joint_point_to_point(
                (0.2, 0.3, 0.4),
                (float("nan"), 0.3, 0.4),
                2.0,
                model,
                sample_hz=50.0,
            )

        with self.assertRaisesRegex(ValueError, "finite"):
            ExcursionTrajectory.from_joint_point_to_point(
                (0.2, 0.3, 0.4),
                (0.7, 0.3, 0.4),
                float("nan"),
                model,
                sample_hz=50.0,
            )

        with self.assertRaisesRegex(ValueError, "Decision boundary"):
            validate_point_to_point_trajectory(
                ExcursionTrajectory.from_joint_point_to_point(
                    (0.0, -0.1, 0.0),
                    (0.0, 0.2, 0.0),
                    2.1,
                    model,
                    sample_hz=50.0,
                ),
                approved_joint_limits_rad={
                    "DIP": (0.0, 1.0),
                    "PIP": (-0.2, 0.3),
                    "MCP": (0.0, 1.0),
                },
                min_action_mm_by_tendon={tendon: -10.0 for tendon in TENDONS},
                max_action_mm_by_tendon={tendon: 10.0 for tendon in TENDONS},
                max_action_delta_mm_by_tendon={tendon: 10.0 for tendon in TENDONS},
                decision_hz=5.0,
            )

    def test_moment_arm_constant_extrapolation_clamps_only_evaluation_angle(self):
        model = _constant_extrapolation_model()
        events = []
        model.set_extrapolation_event_callback(events.append)

        matrix = model.matrix((-0.5, 0.25, 1.5))

        self.assertEqual(tuple(matrix[0]), (1.0, 1.25, 3.0))
        self.assertEqual(len(events), 2)
        self.assertEqual(events[0]["joint"], "DIP")
        self.assertEqual(events[0]["requested_angle_rad"], -0.5)
        self.assertEqual(events[0]["used_endpoint"], "q_min")
        self.assertTrue(events[0]["constant_extrapolation_applied"])
        self.assertEqual(events[1]["joint"], "MCP")
        self.assertEqual(events[1]["used_endpoint"], "q_max")
        model.validate_domain((-0.5, 0.25, 1.5))

    def test_moment_arm_integration_splits_boundary_and_reverses_sign(self):
        model = _constant_extrapolation_model()
        events = []
        model.set_extrapolation_event_callback(events.append)

        forward = model.integrate_excursion((0.0, -0.1, 0.0), (0.0, 0.2, 0.0))
        reverse = model.integrate_excursion((0.0, 0.2, 0.0), (0.0, -0.1, 0.0))

        self.assertAlmostEqual(forward[0], 0.32)
        self.assertAlmostEqual(forward[1], 0.6)
        self.assertAlmostEqual(reverse[0], -forward[0])
        self.assertAlmostEqual(reverse[1], -forward[1])
        self.assertEqual(events[0]["joint"], "PIP")
        self.assertEqual(events[0]["used_endpoint"], "q_min")
        self.assertAlmostEqual(events[0]["requested_angle_rad"], -0.05)

    def test_joint_feedback_uses_constant_extrapolation_runtime(self):
        model = _constant_extrapolation_model()
        events = []
        model.set_extrapolation_event_callback(events.append)
        controller = JointFeedbackController(
            model,
            kp=(1.0, 1.0, 1.0),
            max_joint_correction_rad=None,
            max_excursion_correction_mm=(10.0,) * len(TENDONS),
        )

        result = controller.compute(
            reference_angles_rad=(0.0, -0.1, 0.0),
            measured_angles_rad=(0.0, -0.2, 0.0),
        )

        self.assertAlmostEqual(result.excursion_correction_mm[0], 0.1)
        self.assertEqual(events[0]["joint"], "PIP")
        self.assertEqual(events[0]["used_endpoint"], "q_min")

    def test_physical_joint_limit_still_rejects_extrapolated_reference(self):
        model = _constant_extrapolation_model()
        trajectory = ExcursionTrajectory.from_joint_point_to_point(
            (0.0, -0.1, 0.0),
            (0.0, 0.2, 0.0),
            1.0,
            model,
            sample_hz=1.0,
        )
        common = {
            "min_action_mm_by_tendon": {
                tendon: -10.0 for tendon in TENDONS
            },
            "max_action_mm_by_tendon": {
                tendon: 10.0 for tendon in TENDONS
            },
            "max_action_delta_mm_by_tendon": {
                tendon: 10.0 for tendon in TENDONS
            },
            "decision_hz": 1.0,
        }
        validate_point_to_point_trajectory(
            trajectory,
            model_ranges_rad=model.evaluation_ranges_rad,
            approved_joint_limits_rad={
                "DIP": (0.0, 1.0),
                "PIP": (-0.2, 0.3),
                "MCP": (0.0, 1.0),
            },
            **common,
        )
        with self.assertRaisesRegex(ValueError, "approved PIP limits"):
            validate_point_to_point_trajectory(
                trajectory,
                approved_joint_limits_rad={
                    "DIP": (0.0, 1.0),
                    "PIP": (0.0, 0.3),
                    "MCP": (0.0, 1.0),
                },
                **common,
            )

    def test_point_to_point_reference_csv_round_trips_for_dataset_schema(self):
        trajectory = ExcursionTrajectory.from_joint_point_to_point(
            (0.2, 0.3, 0.4),
            (0.4, 0.5, 0.6),
            2.0,
            _ConstantMomentArm(),
            sample_hz=10.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            path = trajectory.to_csv(Path(directory) / "reference_trajectory.csv")
            loaded = ExcursionTrajectory.from_csv(path)

        self.assertEqual(loaded.target_angles_deg, trajectory.target_angles_deg)
        self.assertEqual(loaded.excursions_mm, trajectory.excursions_mm)
        self.assertEqual(loaded.duration_s, trajectory.duration_s)

    def test_speed_controller_limits_acceleration_and_speed(self):
        controller = SpeedController(
            SpeedControlConfig(
                kp=10.0,
                max_speed=100,
                max_acceleration=50.0,
                position_deadband=0.0,
            )
        )

        first = controller.update({4: 1000.0}, {4: 0.0}, 0.1)
        second = controller.update({4: 1000.0}, {4: 0.0}, 0.1)

        self.assertEqual(first, {4: 5})
        self.assertEqual(second, {4: 10})

    def test_trace_contains_force_columns(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "control_trace.csv"
            writer = ControlTraceWriter(path, SERVO_IDS_BY_TENDON)
            writer.write(
                phase="playback",
                elapsed_s=0.1,
                dt_s=0.02,
                servo_telemetry_age_s=0.01,
                target_excursions_mm=(0, 0, 0, 0, 0, 0),
                target_positions={servo_id: 100 for servo_id in SERVO_IDS_BY_TENDON},
                actual_positions={servo_id: 90 for servo_id in SERVO_IDS_BY_TENDON},
                speeds={servo_id: 10 for servo_id in SERVO_IDS_BY_TENDON},
                force_readings={
                    SERVO_IDS_BY_TENDON[0]: {
                        "force_n": 1.5,
                        "torque_nmm": 2.5,
                        "angle_delta_deg": 3.5,
                        "servo_age_s": 0.01,
                        "encoder_age_s": 0.02,
                        "sync_age_s": 0.01,
                        "status": "OK",
                    }
                },
            )
            writer.close()

            with path.open("r", newline="", encoding="utf-8") as handle:
                rows = list(csv.DictReader(handle))

        self.assertEqual(len(rows), 1)
        self.assertEqual(rows[0]["FDP_force_n"], "1.5")
        self.assertEqual(rows[0]["FDP_force_status"], "OK")
        self.assertIn("PI_force_n", rows[0])

    def test_runner_sends_speed_commands_without_position_api(self):
        class FakeSystem:
            servo_ids = (1, 2, 3, 4, 5, 6)

            @staticmethod
            def index_for_servo_id(servo_id):
                return FakeSystem.servo_ids.index(servo_id)

        class FakeSender:
            def __init__(self):
                self.commands = []

            def send_speed_to_servos(self, commands):
                self.commands.append(dict(commands))

            def stop_servos(self):
                self.commands.append({servo_id: 0 for servo_id in FakeSystem.servo_ids})

        class FakeHardware:
            def __init__(self):
                self.system = FakeSystem()
                self.sender = FakeSender()

            def check_health(self):
                return None

            def latest_feedback(self, max_age_s):
                return HardwareFeedback(
                    servo={
                        "positions": [0, 0, 0, 0, 0, 0],
                        "valid": [True, True, True, True, True, True],
                    },
                    servo_received_at=0.0,
                    encoder_by_sensor={},
                    encoder_received_at={},
                    sampled_at=0.0,
                )

            def position_by_servo_id(self, feedback):
                return {servo_id: 0.0 for servo_id in self.system.servo_ids}

            def stop_servos(self):
                self.sender.stop_servos()

        trajectory = ExcursionTrajectory(
            elapsed_times_s=(0.0, 0.02),
            excursions_mm=((0.0,) * 6, (0.0,) * 6),
            position_units_per_mm=(1.0,) * 6,
        )
        hardware = FakeHardware()
        runner = SpeedTrajectoryRunner(
            trajectory=trajectory,
            servo_ids_by_tendon=SERVO_IDS_BY_TENDON,
            hardware=hardware,
            speed_controller=SpeedController(),
            config=PlaybackConfig(
                control_hz=100.0,
                return_to_start=False,
                display_interval_s=10.0,
            ),
        )

        runner.run({servo_id: 0.0 for servo_id in SERVO_IDS_BY_TENDON})

        self.assertTrue(hardware.sender.commands)
        self.assertEqual(set(hardware.sender.commands[-1]), set(SERVO_IDS_BY_TENDON))


if __name__ == "__main__":
    unittest.main()
