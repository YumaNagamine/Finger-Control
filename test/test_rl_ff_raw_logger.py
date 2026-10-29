from __future__ import annotations

import json
import math
import queue
import tempfile
import threading
import time
import unittest
from pathlib import Path
from unittest.mock import patch

from controller.csv_player.encoder_calibration import EncoderExcursionCalibration
from controller.csv_player.encoder_runner import EncoderPositionTrajectoryRunner
from controller.csv_player.runner import PlaybackConfig
from controller.csv_player.trajectory import ExcursionTrajectory
from controller.rl.action_schedule import (
    INTERPOLATED_PLAYBACK_MODE,
    RL_FF_ZOH_PLAYBACK_MODE,
    RLFFZOHActionSchedule,
)
from controller.rl.action_safety_filter import (
    ActionSafetyConfig,
    ActionSafetyContextError,
    ActionSafetyFilter,
)
from controller.rl.raw_event_logger import AsyncRawEventLogger
from controller.servo.hardware_session import HardwareFeedback
from controller.servo_mapping import (
    ENCODER_SENSOR_IDS_BY_SERVO_ID,
    SERVO_IDS,
    SERVO_IDS_BY_TENDON,
    TENDONS,
)


class _FakeSystem:
    servo_ids = SERVO_IDS

    @staticmethod
    def index_for_servo_id(servo_id):
        return SERVO_IDS.index(servo_id)


class _FakeSender:
    def __init__(self) -> None:
        self.commands: list[dict[int, int]] = []

    def send_speed_to_servos(self, commands) -> None:
        self.commands.append(dict(commands))

    def stop_servos(self) -> None:
        self.commands.append({servo_id: 0 for servo_id in SERVO_IDS})


class _FakeEncoderHardware:
    def __init__(self) -> None:
        self.system = _FakeSystem()
        self.sender = _FakeSender()
        self.stop_latched = False

    @staticmethod
    def _feedback() -> HardwareFeedback:
        now = time.monotonic()
        encoder_by_sensor = {
            sensor_id: {
                "status": "OK",
                "relative_deg": 100.0 + sensor_id,
                "raw": round((100.0 + sensor_id) * 16384 / 360) % 16384,
            }
            for sensor_id in ENCODER_SENSOR_IDS_BY_SERVO_ID.values()
        }
        return HardwareFeedback(
            servo={
                "positions": [0] * len(SERVO_IDS),
                "valid": [True] * len(SERVO_IDS),
            },
            servo_received_at=now,
            encoder_by_sensor=encoder_by_sensor,
            encoder_received_at={
                sensor_id: now for sensor_id in encoder_by_sensor
            },
            sampled_at=now,
        )

    def wait_for_feedback(self, timeout_s, require_encoder=False):
        return self._feedback()

    def latest_feedback(self, max_age_s):
        return self._feedback()

    @staticmethod
    def check_health() -> None:
        return None

    def stop_servos(self) -> None:
        self.sender.stop_servos()

    def request_latched_stop(self):
        self.stop_latched = True
        self.sender.stop_servos()
        return None

    def stop_state_snapshot(self):
        return {
            "latched": self.stop_latched,
            "physical_stop_confirmed": False,
            "tension_safe_confirmed": False,
        }


def _calibration() -> EncoderExcursionCalibration:
    return EncoderExcursionCalibration(
        degrees_per_mm=(1.0,) * 6,
        encoder_directions=(1.0,) * 6,
        kp=2.0,
        ki=0.0,
        kd=0.0,
        max_speed=100.0,
        max_acceleration=10000.0,
        hard_error_margin_deg=10.0,
        minimum_hard_error_deg=10.0,
        return_tolerance_deg=1.0,
        calibration_status="test",
    )


class RLFFAndRawLoggerTest(unittest.TestCase):
    @staticmethod
    def _safety_filter(**overrides):
        settings = {
            "min_action_mm_by_tendon": {tendon: -10.0 for tendon in TENDONS},
            "max_action_mm_by_tendon": {tendon: 10.0 for tendon in TENDONS},
            "max_action_delta_mm_by_tendon": {tendon: 1.0 for tendon in TENDONS},
            "hard_force_limits_n_by_tendon": None,
            "joint_angle_limits_rad": None,
            "max_servo_age_s": 0.1,
            "max_encoder_age_s": 0.1,
            "max_joint_age_s": 0.1,
            "max_deadline_lateness_s": 0.02,
        }
        settings.update(overrides)
        return ActionSafetyFilter(ActionSafetyConfig(**settings))

    def test_existing_interpolated_mode_remains_default(self):
        config = PlaybackConfig()
        self.assertEqual(config.playback_mode, INTERPOLATED_PLAYBACK_MODE)
        trajectory = ExcursionTrajectory(
            elapsed_times_s=(0.0, 1.0),
            excursions_mm=((0.0,) * 6, (10.0,) * 6),
        )
        self.assertEqual(trajectory.excursions_at(0.25), (2.5,) * 6)

    def test_zoh_schedule_samples_at_configured_decision_rate(self):
        trajectory = ExcursionTrajectory(
            elapsed_times_s=(0.0, 1.0),
            excursions_mm=((3.0,) * 6, (13.0,) * 6),
        )
        schedule = RLFFZOHActionSchedule(trajectory, decision_hz=5.0)

        first = schedule.advance(0.0)
        self.assertIsNotNone(first)
        assert first is not None
        self.assertEqual(first.action_excursion_mm, (0.0,) * 6)
        self.assertIsNone(schedule.advance(0.199))

        second = schedule.advance(0.2)
        self.assertIsNotNone(second)
        assert second is not None
        self.assertEqual(second.decision_id, 1)
        self.assertEqual(second.action_excursion_mm, (2.0,) * 6)
        self.assertIsNone(schedule.advance(0.399))

    def test_async_logger_records_completion_and_time_separation(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw_events.jsonl"
            logger = AsyncRawEventLogger(path, run_id="fake")
            represented_at = time.monotonic() - 0.01
            self.assertTrue(
                logger.log(
                    "test_event",
                    event_monotonic_s=represented_at,
                    payload={"value": 1},
                )
            )
            stats = logger.close()
            event = json.loads(path.read_text(encoding="utf-8").strip())

        self.assertTrue(stats.complete)
        self.assertEqual(stats.dropped_events, 0)
        self.assertEqual(event["event_monotonic_s"], represented_at)
        self.assertGreaterEqual(
            event["written_monotonic_s"],
            event["enqueued_monotonic_s"],
        )

    def test_async_logger_detects_queue_overflow_without_blocking(self):
        with tempfile.TemporaryDirectory() as directory:
            logger = AsyncRawEventLogger(
                Path(directory) / "raw_events.jsonl",
                run_id="overflow",
                queue_capacity=1,
            )
            with patch.object(logger._queue, "put_nowait", side_effect=queue.Full):
                accepted = logger.log("will_drop", payload={})
            stats = logger.close()

        self.assertFalse(accepted)
        self.assertEqual(stats.dropped_events, 1)
        self.assertFalse(stats.complete)

    def test_async_logger_preserves_sequence_with_concurrent_producers(self):
        producer_count = 4
        events_per_producer = 100
        barrier = threading.Barrier(producer_count)
        accepted: list[bool] = []
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw_events.jsonl"
            logger = AsyncRawEventLogger(
                path,
                run_id="concurrent",
                queue_capacity=producer_count * events_per_producer,
            )

            def produce(producer_id):
                barrier.wait()
                for item_id in range(events_per_producer):
                    accepted.append(
                        logger.log(
                            "concurrent_event",
                            payload={
                                "producer_id": producer_id,
                                "item_id": item_id,
                            },
                        )
                    )

            threads = [
                threading.Thread(target=produce, args=(producer_id,))
                for producer_id in range(producer_count)
            ]
            for thread in threads:
                thread.start()
            for thread in threads:
                thread.join()
            stats = logger.close()
            sequences = [
                json.loads(line)["event_sequence"]
                for line in path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertTrue(stats.complete)
        self.assertTrue(all(accepted))
        self.assertEqual(
            sequences,
            list(range(producer_count * events_per_producer)),
        )

    def test_fake_encoder_default_mode_still_interpolates_every_tick(self):
        trajectory = ExcursionTrajectory(
            elapsed_times_s=(0.0, 0.08),
            excursions_mm=((0.0,) * 6, (8.0,) * 6),
        )
        hardware = _FakeEncoderHardware()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw_events.jsonl"
            logger = AsyncRawEventLogger(path, run_id="fake-interpolated")
            runner = EncoderPositionTrajectoryRunner(
                trajectory=trajectory,
                servo_ids_by_tendon=SERVO_IDS_BY_TENDON,
                encoder_sensor_ids_by_servo_id=ENCODER_SENSOR_IDS_BY_SERVO_ID,
                calibration=_calibration(),
                hardware=hardware,
                config=PlaybackConfig(
                    control_hz=50.0,
                    return_to_start=False,
                    display_interval_s=10.0,
                ),
                raw_event_logger=logger,
            )
            runner.run()
            logger.close()
            events = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]

        ticks = [event for event in events if event["event_type"] == "pid_tick"]
        actions = [
            round(event["payload"]["tendons"]["FDP"]["action_excursion_mm"], 4)
            for event in ticks
        ]
        self.assertFalse(
            any(event["event_type"] == "decision" for event in events)
        )
        self.assertGreaterEqual(len(set(actions)), 3)
        self.assertTrue(all(event["payload"]["decision_id"] is None for event in ticks))

    def test_fake_encoder_playback_logs_zoh_decisions_and_every_pid_tick(self):
        trajectory = ExcursionTrajectory(
            elapsed_times_s=(0.0, 0.45),
            excursions_mm=((0.0,) * 6, (4.5,) * 6),
        )
        hardware = _FakeEncoderHardware()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw_events.jsonl"
            logger = AsyncRawEventLogger(path, run_id="fake-zoh")
            runner = EncoderPositionTrajectoryRunner(
                trajectory=trajectory,
                servo_ids_by_tendon=SERVO_IDS_BY_TENDON,
                encoder_sensor_ids_by_servo_id=ENCODER_SENSOR_IDS_BY_SERVO_ID,
                calibration=_calibration(),
                hardware=hardware,
                config=PlaybackConfig(
                    control_hz=50.0,
                    return_to_start=False,
                    display_interval_s=10.0,
                    playback_mode=RL_FF_ZOH_PLAYBACK_MODE,
                    decision_hz=5.0,
                ),
                raw_event_logger=logger,
            )
            runner.run()
            stats = logger.close()
            events = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]

        decisions = [event for event in events if event["event_type"] == "decision"]
        ticks = [event for event in events if event["event_type"] == "pid_tick"]
        transports = [
            event for event in events if event["event_type"] == "transport_write"
        ]
        self.assertTrue(stats.complete)
        self.assertEqual([event["payload"]["decision_id"] for event in decisions], [0, 1, 2])
        self.assertTrue(
            all(
                event["payload"]["rl_dataset_action_is_incremental"]
                for event in decisions
            )
        )
        self.assertEqual(
            [round(event["payload"]["pid_reflected_delta_mm"]["FDP"], 6) for event in decisions],
            [0.0, 2.0, 2.0],
        )
        self.assertTrue(
            all(
                event["payload"]["physical_servo_applied_time_status"]
                == "unobserved"
                for event in decisions
            )
        )
        self.assertGreaterEqual(len(ticks), 20)
        self.assertEqual(len(ticks), len(transports))
        self.assertTrue(
            all(
                event["payload"]["physical_servo_applied_monotonic_s"] is None
                for event in transports
            )
        )
        decision_ids = {
            event["payload"]["decision_id"] for event in ticks
        }
        self.assertEqual(decision_ids, {0, 1, 2})
        self.assertTrue(
            all("period_violation" in event["payload"] for event in ticks)
        )
        self.assertTrue(hardware.sender.commands)

    def test_start_state_validation_is_logged_before_any_playback_command(self):
        trajectory = ExcursionTrajectory(
            elapsed_times_s=(0.0, 0.08),
            excursions_mm=((0.0,) * 6, (1.0,) * 6),
        )
        hardware = _FakeEncoderHardware()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "raw_events.jsonl"
            logger = AsyncRawEventLogger(path, run_id="start-state-guard")
            runner = EncoderPositionTrajectoryRunner(
                trajectory=trajectory,
                servo_ids_by_tendon=SERVO_IDS_BY_TENDON,
                encoder_sensor_ids_by_servo_id=ENCODER_SENSOR_IDS_BY_SERVO_ID,
                calibration=_calibration(),
                hardware=hardware,
                config=PlaybackConfig(return_to_start=False),
                raw_event_logger=logger,
                start_state_validator=lambda _feedback, _origin: {
                    "status": "passed",
                    "joint_delta_deg": {"DIP": 0.0, "PIP": 0.0, "MCP": 0.0},
                },
            )
            runner.run()
            stats = logger.close()
            events = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]

        session_start = next(
            event for event in events if event["event_type"] == "control_session_start"
        )
        self.assertTrue(stats.complete)
        self.assertEqual(
            session_start["payload"]["p2p_start_state_validation"]["status"],
            "passed",
        )
        self.assertTrue(hardware.sender.commands)

    def test_failed_start_state_validation_sends_no_servo_command(self):
        trajectory = ExcursionTrajectory(
            elapsed_times_s=(0.0, 0.08),
            excursions_mm=((0.0,) * 6, (1.0,) * 6),
        )
        hardware = _FakeEncoderHardware()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "rejected_start.jsonl"
            logger = AsyncRawEventLogger(path, run_id="rejected-start")

            def reject_start(_feedback, _origin):
                raise RuntimeError("P2P start state changed")

            runner = EncoderPositionTrajectoryRunner(
                trajectory=trajectory,
                servo_ids_by_tendon=SERVO_IDS_BY_TENDON,
                encoder_sensor_ids_by_servo_id=ENCODER_SENSOR_IDS_BY_SERVO_ID,
                calibration=_calibration(),
                hardware=hardware,
                config=PlaybackConfig(return_to_start=False),
                raw_event_logger=logger,
                start_state_validator=reject_start,
            )
            with self.assertRaisesRegex(RuntimeError, "start state changed"):
                runner.run()
            stats = logger.close()
            events = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]

        self.assertTrue(stats.complete)
        self.assertEqual(
            events[0]["event_type"], "point_to_point_start_state_rejected"
        )
        self.assertEqual(hardware.sender.commands, [])

    def test_lower_tick_safety_latches_on_stale_dlc_deadline_and_nonfinite_force(self):
        trajectory = ExcursionTrajectory(
            elapsed_times_s=(0.0, 1.0),
            excursions_mm=((0.0,) * 6, (1.0,) * 6),
        )
        hardware = _FakeEncoderHardware()
        runner = EncoderPositionTrajectoryRunner(
            trajectory=trajectory,
            servo_ids_by_tendon=SERVO_IDS_BY_TENDON,
            encoder_sensor_ids_by_servo_id=ENCODER_SENSOR_IDS_BY_SERVO_ID,
            calibration=_calibration(),
            hardware=hardware,
            config=PlaybackConfig(return_to_start=False),
            action_safety_filter=self._safety_filter(),
            joint_safety_state_provider=lambda: None,
        )
        feedback = hardware.latest_feedback(0.1)
        assert feedback is not None

        with self.assertRaises(ActionSafetyContextError) as context:
            runner._check_fast_safety_context(
                feedback=feedback,
                force_readings={},
                scheduled_tick_monotonic_s=time.monotonic() - 0.1,
                evaluated_at_monotonic_s=time.monotonic(),
            )

        self.assertIn("joint_stale_missing", context.exception.reasons)
        self.assertIn("control_deadline", context.exception.reasons)
        self.assertTrue(hardware.stop_latched)
        self.assertEqual(
            hardware.sender.commands[-1],
            {servo_id: 0 for servo_id in SERVO_IDS},
        )

        missing_encoder_feedback = HardwareFeedback(
            servo=feedback.servo,
            servo_received_at=feedback.servo_received_at,
            encoder_by_sensor=feedback.encoder_by_sensor,
            encoder_received_at={},
            sampled_at=time.monotonic(),
        )
        with self.assertRaises(ActionSafetyContextError) as context:
            runner._check_fast_safety_context(
                feedback=missing_encoder_feedback,
                force_readings={},
                scheduled_tick_monotonic_s=time.monotonic(),
                evaluated_at_monotonic_s=time.monotonic(),
            )
        self.assertIn("encoder_stale:FDP_missing", context.exception.reasons)

        nonfinite_hardware = _FakeEncoderHardware()
        nonfinite_runner = EncoderPositionTrajectoryRunner(
            trajectory=trajectory,
            servo_ids_by_tendon=SERVO_IDS_BY_TENDON,
            encoder_sensor_ids_by_servo_id=ENCODER_SENSOR_IDS_BY_SERVO_ID,
            calibration=_calibration(),
            hardware=nonfinite_hardware,
            config=PlaybackConfig(return_to_start=False),
            action_safety_filter=self._safety_filter(
                hard_force_limits_n_by_tendon={
                    tendon: 5.0 for tendon in TENDONS
                },
                max_joint_age_s=None,
            ),
        )
        force_readings = {
            servo_id: {"force_n": 0.0}
            for servo_id in SERVO_IDS
        }
        force_readings[SERVO_IDS_BY_TENDON[0]]["force_n"] = float("nan")
        with self.assertRaises(ActionSafetyContextError) as context:
            nonfinite_runner._check_fast_safety_context(
                feedback=nonfinite_hardware.latest_feedback(0.1),
                force_readings=force_readings,
                scheduled_tick_monotonic_s=time.monotonic(),
                evaluated_at_monotonic_s=time.monotonic(),
            )
        self.assertIn("force_non_finite:FDP", context.exception.reasons)
        self.assertTrue(nonfinite_hardware.stop_latched)

    def test_lower_tick_safety_latches_on_physical_joint_limit_violation(self):
        trajectory = ExcursionTrajectory(
            elapsed_times_s=(0.0, 1.0),
            excursions_mm=((0.0,) * len(TENDONS), (1.0,) * len(TENDONS)),
        )
        hardware = _FakeEncoderHardware()
        limits_rad = {
            "DIP": (math.radians(-3.0), math.radians(15.0)),
            "PIP": (math.radians(-10.0), math.radians(15.0)),
            "MCP": (math.radians(-10.0), math.radians(15.0)),
        }
        runner = EncoderPositionTrajectoryRunner(
            trajectory=trajectory,
            servo_ids_by_tendon=SERVO_IDS_BY_TENDON,
            encoder_sensor_ids_by_servo_id=ENCODER_SENSOR_IDS_BY_SERVO_ID,
            calibration=_calibration(),
            hardware=hardware,
            config=PlaybackConfig(return_to_start=False),
            action_safety_filter=self._safety_filter(
                joint_angle_limits_rad=limits_rad,
            ),
            joint_safety_state_provider=lambda: (
                {"DIP": 0.0, "PIP": math.radians(15.1), "MCP": 0.0},
                0.01,
            ),
        )

        with self.assertRaises(ActionSafetyContextError) as context:
            runner._check_fast_safety_context(
                feedback=hardware.latest_feedback(0.1),
                force_readings={},
                scheduled_tick_monotonic_s=time.monotonic(),
                evaluated_at_monotonic_s=time.monotonic(),
            )

        self.assertIn("joint_angle_limit:PIP", context.exception.reasons)
        self.assertTrue(hardware.stop_latched)
        self.assertEqual(
            hardware.sender.commands[-1],
            {servo_id: 0 for servo_id in SERVO_IDS},
        )


if __name__ == "__main__":
    unittest.main()

