from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

from controller.csv_player.gui_support import LatestEventMailbox
from controller.csv_player.recorder import ControlTraceWriter
from controller.csv_player.runner import PlaybackConfig, SpeedTrajectoryRunner
from controller.csv_player.speed_controller import SpeedControlConfig, SpeedController
from controller.csv_player.trajectory import ExcursionTrajectory
from controller.servo.hardware_session import HardwareFeedback
from controller.servo_mapping import SERVO_IDS_BY_TENDON, TENDONS


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
