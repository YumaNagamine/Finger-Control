from __future__ import annotations

import tempfile
import sys
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock, call

from controller.servo.force_control.gui.session import (
    API_ROOT,
    DEFAULT_CALIBRATION_DIR,
    DEFAULT_FORCE_CONFIG_PATH,
    DiagnosticSample,
    ForceControlSession,
    MANUAL_SPEED_PRESETS,
    _CompleteLineTransport,
    _RecentStreamStats,
    load_gui_force_config,
)
from controller.servo.force_control.runner import LatestSampleStore
from controller.servo.torque_calibration.storage import load_torque_intercept_calibration


class ForceControlGuiConfigTest(unittest.TestCase):
    def test_servo_callback_publishes_valid_raw_diagnostics(self):
        session = ForceControlSession(load_gui_force_config())
        session._servo_index = {servo_id: servo_id - 1 for servo_id in range(1, 7)}

        session._accept_servo(
            {
                "kind": "telemetry",
                "valid": [True, False, True, True, True, True],
                "positions": [101, 202, 303, 404, 505, 606],
            }
        )

        samples = session.drain_servo_diagnostic_samples()
        self.assertTrue(all(isinstance(sample, DiagnosticSample) for sample in samples))
        self.assertEqual(
            [(sample.channel_id, sample.value) for sample in samples],
            [(1, 101.0), (3, 303.0), (4, 404.0), (5, 505.0), (6, 606.0)],
        )

    def test_encoder_callback_publishes_only_valid_raw_diagnostics(self):
        session = ForceControlSession(load_gui_force_config())
        received_at = time.monotonic()

        session._record_encoder_sample(
            2, {"status": "OK", "relative_multi_turn_deg": 12.5}, received_at
        )
        session._record_encoder_sample(
            3, {"status": "ERROR", "relative_multi_turn_deg": 99.0}, received_at
        )
        session._record_encoder_sample(4, {"status": "OK"}, received_at)

        samples = session.drain_encoder_diagnostic_samples()
        self.assertEqual(
            [(sample.channel_id, sample.value) for sample in samples],
            [(2, 12.5)],
        )

    def test_serial_transport_reassembles_a_line_split_across_timeouts(self):
        base_transport = Mock()
        port = Mock()
        port.is_open = True
        port.read_until.side_effect = [b"AS5048A,SENSOR_DATA,1,1,10", b",0,0,0,OK,10,0,0\n"]
        base_transport.port = port
        transport = _CompleteLineTransport(base_transport)

        self.assertEqual(transport.read_raw_line(), b"")
        self.assertEqual(
            transport.read_raw_line(),
            b"AS5048A,SENSOR_DATA,1,1,10,0,0,0,OK,10,0,0\n",
        )

    def test_default_gui_config_has_fixed_target_levels(self):
        config = load_gui_force_config(Path(DEFAULT_FORCE_CONFIG_PATH))
        self.assertEqual(config.target_force_levels_n, (0.5, 1.0, 5.0, 10.0, 15.0, 20.0, 30.0))
        self.assertEqual(config.min_target_force_n, 0.5)
        self.assertEqual(config.max_target_force_n, 30.0)
        self.assertEqual(
            config.target_force_levels_by_tendon,
            {
                "ED": (0.5, 1.0, 5.0, 10.0, 20.0, 30.0),
                "DI": (0.5, 1.0, 5.0, 10.0, 15.0),
                "FDS": (0.5, 1.0, 5.0, 10.0, 20.0, 30.0),
                "FDP": (0.5, 1.0, 5.0, 10.0, 20.0, 30.0),
                "LU": (0.5, 1.0, 5.0),
                "PI": (0.5, 1.0, 5.0, 10.0, 15.0),
            },
        )
        self.assertEqual(
            config.hard_force_limits_n_by_tendon,
            {
                "ED": 35.0,
                "DI": 20.0,
                "FDS": 35.0,
                "FDP": 35.0,
                "LU": 20.0,
                "PI": 20.0,
            },
        )
        self.assertEqual(config.ramp_duration_s, 1.0)
        self.assertAlmostEqual(config.feedback_timeout_s, 0.2)
        self.assertAlmostEqual(config.feedback_recovery_grace_s, 0.3)

    def test_active_control_zeros_a_missing_servo_until_feedback_recovers(self):
        session = ForceControlSession(load_gui_force_config())
        stop_event = threading.Event()
        controller = Mock()
        controller.config = SimpleNamespace(target_force_n=0.5)
        controller.step.return_value = SimpleNamespace(
            speed_command=125,
            force_error_n=0.25,
        )
        state = SimpleNamespace(
            servo_id=1,
            tendon="ED",
            target_force_n=0.5,
            ramp_started_at=None,
            controller=controller,
            active=True,
            feedback_missing_since=None,
        )
        feedback_item = {
            "force_n": 0.75,
            "position": 2048.0,
            "encoder_position_deg": 10.0,
            "angle_delta_deg": 1.0,
            "servo_age_s": 0.0,
            "encoder_age_s": 0.0,
        }
        feedback_calls = 0

        def feedback_sequence():
            nonlocal feedback_calls
            feedback_calls += 1
            if feedback_calls == 1:
                return {}
            stop_event.set()
            return {1: feedback_item}

        session._connected = True
        session._armed = True
        session._running = True
        session._servo = Mock()
        session._states = {1: state}
        session._stop_event = stop_event
        session._control_thread = threading.current_thread()
        session._feedback = Mock(side_effect=feedback_sequence)

        session._loop(stop_event)

        self.assertEqual(
            [call.args for call in session._servo.send_speed_by_id.call_args_list],
            [(1, 0), (1, 125)],
        )
        self.assertIsNone(state.feedback_missing_since)
        self.assertFalse(session.faulted)
        self.assertIn(
            "Feedback recovered for ID 1; resuming control.",
            session.drain_events(),
        )

    def test_diagnostics_reports_mapping_and_zero_metadata_before_connect(self):
        session = ForceControlSession(load_gui_force_config())
        diagnostics = session.diagnostics_snapshot()
        mapping = diagnostics["mapping"]

        self.assertEqual(
            [(row["servo_id"], row["tendon"], row["sensor_id"]) for row in mapping],
            [(1, "ED", 1), (2, "DI", 2), (3, "FDS", 3),
             (4, "FDP", 4), (5, "LU", 5), (6, "PI", 6)],
        )
        self.assertTrue(all(row["calibration_path"] for row in mapping))
        self.assertTrue(all(row["calibration_updated"] for row in mapping))
        self.assertTrue(all(row["zero_offset_updated"] for row in mapping))
        self.assertTrue(all(row["zero_sample_count"] == 78 for row in mapping))
        self.assertTrue(all(row["zero_std_deg"] is not None for row in mapping))
        self.assertTrue(all(row["servo_stream"]["sample_count"] == 0 for row in mapping))
        self.assertTrue(all(row["encoder_stream"]["sample_count"] == 0 for row in mapping))
        self.assertEqual(Path(mapping[0]["calibration_path"]).parent, DEFAULT_CALIBRATION_DIR)

    def test_recent_stream_stats_reports_rate_gap_and_valid_percent(self):
        stats = _RecentStreamStats()
        stats.record(1, True, received_at=10.0)
        stats.record(1, False, received_at=10.1)
        stats.record(1, True, received_at=10.3)

        result = stats.snapshot(1, now=10.3)

        self.assertEqual(result["sample_count"], 3)
        self.assertAlmostEqual(result["effective_hz"], 2.0 / 0.3)
        self.assertAlmostEqual(result["max_interval_s"], 0.2)
        self.assertAlmostEqual(result["valid_percent"], 200.0 / 3.0)

    def test_drain_samples_returns_empty_list_when_queue_is_empty(self):
        session = ForceControlSession(load_gui_force_config())

        self.assertEqual(session.drain_samples(), [])
        self.assertEqual(session.baseline_snapshot(), {})

    def test_manual_speed_uses_api_units_and_rejects_non_presets(self):
        session = ForceControlSession(load_gui_force_config())
        session._connected = True
        session._servo = Mock()
        session._states = {1: SimpleNamespace(active=False)}

        session.set_manual_speed(1, MANUAL_SPEED_PRESETS[-1])

        session._servo.send_speed_by_id.assert_called_once_with(1, 500)
        self.assertEqual(session._manual_speed_commands[1], 500)
        with self.assertRaises(ValueError):
            session.set_manual_speed(1, 250)

    def test_manual_speed_is_blocked_while_force_control_runs(self):
        session = ForceControlSession(load_gui_force_config())
        session._connected = True
        session._running = True
        session._servo = Mock()
        session._states = {1: SimpleNamespace(active=True)}

        with self.assertRaisesRegex(RuntimeError, "Stop force control"):
            session.set_manual_speed(1, 100)

        session._servo.send_speed_by_id.assert_not_called()

    def test_manual_speed_is_allowed_while_feedback_is_unavailable(self):
        session = ForceControlSession(load_gui_force_config())
        session._connected = True
        session._servo = Mock()
        session._states = {1: SimpleNamespace(active=False)}
        session._manual_feedback_recovering = True
        session._feedback_errors = {1: "temporary dropout"}

        session.set_manual_speed(1, 500)

        session._servo.send_speed_by_id.assert_called_once_with(1, 500)
        self.assertEqual(session.manual_speed_snapshot()[1], 500)

    def test_arm_is_blocked_when_already_armed_or_running(self):
        session = ForceControlSession(load_gui_force_config())
        session._connected = True
        session._armed = True

        with self.assertRaisesRegex(
            RuntimeError, "Stop and disarm force control before arming"
        ):
            session.arm(
                kp=5.0,
                ki=0.0,
                kd=0.0,
                max_speed=20.0,
                hard_force_limits_n=session.config.hard_force_limits_n_by_tendon,
                control_sign=1.0,
            )

    def test_baseline_is_blocked_while_armed_or_during_calibration(self):
        session = ForceControlSession(load_gui_force_config())
        session._connected = True
        session._armed = True

        with self.assertRaisesRegex(
            RuntimeError, "Stop and disarm force control before capturing a new baseline"
        ):
            session.capture_baseline()

        session._armed = False
        session._calibration_lock.acquire()
        try:
            with self.assertRaisesRegex(
                RuntimeError, "Torque calibration is already in progress"
            ):
                session.capture_baseline()
        finally:
            session._calibration_lock.release()

    def test_baseline_failure_does_not_partially_update_servo_states(self):
        session = ForceControlSession(load_gui_force_config())
        session._connected = True
        session._states = {
            servo_id: SimpleNamespace(
                baseline_force_n=float(servo_id),
                target_force_n=0.5,
                ramp_start_n=0.5,
                ramp_end_n=0.5,
                ramp_started_at=None,
                pending_monitoring_jump=None,
                pending_monitoring_jump_received_at=None,
            )
            for servo_id in range(1, 7)
        }
        session._feedback = Mock(
            return_value={
                servo_id: {"force_n": 10.0 + servo_id}
                for servo_id in range(1, 6)
            }
        )

        with self.assertRaisesRegex(RuntimeError, "No baseline samples for servo 6"):
            session.capture_baseline(duration_s=0.01)

        self.assertEqual(
            session.baseline_snapshot(),
            {servo_id: float(servo_id) for servo_id in range(1, 7)},
        )

    def test_manual_feedback_error_keeps_speed_without_latching_fault(self):
        session = ForceControlSession(load_gui_force_config())
        session._connected = True
        session._servo = Mock()
        session._manual_speed_commands[1] = 500

        handled = session._handle_manual_feedback_error(RuntimeError("invalid telemetry"))
        handled_again = session._handle_manual_feedback_error(
            RuntimeError("invalid telemetry")
        )
        events = session.drain_events()

        self.assertTrue(handled)
        self.assertTrue(handled_again)
        self.assertFalse(session.faulted)
        self.assertTrue(session.manual_feedback_recovering)
        self.assertEqual(session._manual_speed_commands[1], 500)
        session._servo.send_speed_by_id.assert_not_called()
        self.assertEqual(len(events), 1)
        self.assertIn("manual speed is continuing", events[0])

    def test_monitoring_feedback_error_waits_for_recovery_without_fault(self):
        session = ForceControlSession(load_gui_force_config())
        session._connected = True

        handled = session._handle_manual_feedback_error(RuntimeError("invalid telemetry"))

        self.assertTrue(handled)
        self.assertFalse(session.faulted)
        self.assertTrue(session.manual_feedback_recovering)

    def test_manual_feedback_recovery_does_not_mask_force_control_fault(self):
        session = ForceControlSession(load_gui_force_config())
        session._connected = True
        session._armed = True
        session._running = True
        session._manual_feedback_recovering = True
        session._manual_speed_commands[1] = 500

        handled = session._handle_manual_feedback_error(RuntimeError("invalid telemetry"))

        self.assertFalse(handled)
        self.assertEqual(session._manual_speed_commands[1], 500)
        self.assertFalse(session.faulted)

    def test_fault_message_is_retained_for_gui_diagnostics(self):
        session = ForceControlSession(load_gui_force_config())
        session._servo = Mock()

        session._handle_fault(RuntimeError("invalid telemetry for ID 2"))

        self.assertTrue(session.faulted)
        self.assertEqual(session.fault_message, "invalid telemetry for ID 2")
        self.assertIn(
            "FAULT: all servos stopped: invalid telemetry for ID 2",
            session.drain_events(),
        )

    def test_arm_is_blocked_during_manual_feedback_recovery(self):
        session = ForceControlSession(load_gui_force_config())
        session._connected = True
        session._manual_feedback_recovering = True

        with self.assertRaisesRegex(RuntimeError, "feedback recovery"):
            session.arm(
                kp=5.0,
                ki=0.0,
                kd=0.0,
                max_speed=20.0,
                hard_force_limit_n=3.0,
                control_sign=1.0,
            )

    def test_zero_force_calibration_updates_only_runtime_intercept(self):
        session = ForceControlSession(load_gui_force_config())
        model = {"slope_nmm_per_deg": -2.0}
        state = SimpleNamespace(
            model=model,
            sensor_id=1,
            intercept_nmm=99.0,
            baseline_force_n=4.0,
        )
        session._connected = True
        session._states = {1: state}
        session._feedback = Mock(
            return_value={
                1: {
                    "angle_delta_deg": 3.0,
                    "encoder_position_deg": 720.0,
                    "encoder_received_at": 1.0,
                    "multiturn_angle_model": True,
                }
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            session._intercept_calibration_path = (
                Path(directory) / "torque_intercept_calibration.json"
            )

            result = session.calibrate_current_force_as_zero(1, duration_s=0.01)

        self.assertEqual(model["slope_nmm_per_deg"], -2.0)
        self.assertAlmostEqual(result["intercept_nmm"], 6.0)
        self.assertAlmostEqual(state.intercept_nmm, 6.0)
        self.assertEqual(state.baseline_force_n, 0.0)

    def test_zero_force_calibration_skips_transient_missing_feedback(self):
        session = ForceControlSession(load_gui_force_config())
        model = {"slope_nmm_per_deg": -2.0}
        state = SimpleNamespace(
            model=model,
            sensor_id=1,
            intercept_nmm=99.0,
            baseline_force_n=4.0,
        )
        session._connected = True
        session._states = {1: state}
        valid_feedback = {
            1: {
                "angle_delta_deg": 3.0,
                "encoder_position_deg": 720.0,
                "encoder_received_at": 1.0,
                "multiturn_angle_model": True,
            }
        }
        calls = 0

        def feedback_with_one_dropout():
            nonlocal calls
            calls += 1
            return {} if calls == 1 else valid_feedback

        session._feedback = Mock(side_effect=feedback_with_one_dropout)
        with tempfile.TemporaryDirectory() as directory:
            session._intercept_calibration_path = (
                Path(directory) / "torque_intercept_calibration.json"
            )

            result = session.calibrate_current_force_as_zero(1, duration_s=0.02)

        self.assertGreaterEqual(calls, 2)
        self.assertAlmostEqual(result["intercept_nmm"], 6.0)
        self.assertEqual(model["slope_nmm_per_deg"], -2.0)

    def test_calibration_readiness_allows_transient_feedback_recovery(self):
        session = ForceControlSession(load_gui_force_config())
        session._connected = True
        session._manual_feedback_recovering = True
        session._feedback_errors = {2: "temporary dropout"}

        readiness = session.calibration_readiness_snapshot()

        self.assertIsNone(readiness["all_reason"])
        self.assertTrue(all(reason is None for reason in readiness["by_servo"].values()))

    def test_remember_encoder_position_uses_magnetic_encoder(self):
        session = ForceControlSession(load_gui_force_config())
        state = SimpleNamespace(sensor_id=3)
        session._connected = True
        session._states = {3: state}
        session._feedback = Mock(
            return_value={
                3: {
                    "encoder_position_deg": -361.5,
                    "multiturn_angle_model": True,
                }
            }
        )
        with tempfile.TemporaryDirectory() as directory:
            session._encoder_position_path = Path(directory) / "encoder_positions.json"

            result = session.remember_encoder_position_for_calibration(3)

        self.assertAlmostEqual(result["encoder_position_deg"], -361.5)

    def test_all_tendon_zero_force_calibration_updates_every_runtime_intercept(self):
        session = ForceControlSession(load_gui_force_config())
        states = {}
        feedback = {}
        for servo_id in range(1, 7):
            states[servo_id] = SimpleNamespace(
                model={"slope_nmm_per_deg": -2.0},
                sensor_id=servo_id,
                intercept_nmm=99.0,
                baseline_force_n=4.0,
            )
            feedback[servo_id] = {
                "angle_delta_deg": 3.0,
                "encoder_position_deg": 720.0 + servo_id,
                "encoder_received_at": 1.0,
                "multiturn_angle_model": True,
            }
        session._connected = True
        session._states = states
        session._feedback = Mock(return_value=feedback)

        with tempfile.TemporaryDirectory() as directory:
            calibration_path = Path(directory) / "torque_intercept_calibration.json"
            session._intercept_calibration_path = calibration_path

            result = session.calibrate_all_current_force_as_zero(duration_s=0.01)

            saved = load_torque_intercept_calibration(calibration_path)

        self.assertEqual(set(result), set(range(1, 7)))
        self.assertEqual(set(saved), set(range(1, 7)))
        for servo_id, state in states.items():
            self.assertAlmostEqual(result[servo_id]["intercept_nmm"], 6.0)
            self.assertAlmostEqual(saved[servo_id]["intercept_nmm"], 6.0)
            self.assertAlmostEqual(state.intercept_nmm, 6.0)
            self.assertEqual(state.baseline_force_n, 0.0)

    def test_feedback_uses_older_servo_sample_with_closest_timestamp(self):
        api_python = str(API_ROOT / "python")
        if api_python not in sys.path:
            sys.path.insert(0, api_python)

        session = ForceControlSession(load_gui_force_config())
        state = SimpleNamespace(
            servo_id=1,
            sensor_id=1,
            model={
                "motor_name": "ED",
                "servo_counts_per_rev": 4096.0,
                "servo_zero_raw": 2048.0,
                "servo_direction": "1",
                "encoder_direction": "1",
                "torque_zero_delta_deg": 0.0,
                "slope_nmm_per_deg": -1.0,
                "angle_reference_deg": 0.0,
                "moment_arm_mm": 20.0,
                "pull_angle_deg": 90.0,
            },
            intercept_nmm=0.0,
            previous_snapshot=None,
            latest_force_n=None,
            latest_position=None,
            pending_monitoring_jump=None,
            pending_monitoring_jump_received_at=None,
            last_monitoring_jump_log_at=0.0,
        )
        servo_store = LatestSampleStore()
        encoder_store = LatestSampleStore()
        session._servo = object()
        session._servo_store = servo_store
        session._encoder_stores = {1: encoder_store}
        session._servo_index = {1: 0}
        session._states = {1: state}

        timestamp = time.monotonic() - 0.1
        with session._feedback_history_lock:
            session._servo_feedback_history.extend(
                (
                    (timestamp + 0.025, {"valid": [True], "positions": [2048]}),
                    (timestamp + 0.100, {"valid": [True], "positions": [3072]}),
                )
            )
            session._encoder_feedback_histories[1].append(
                (
                    timestamp + 0.040,
                    {"status": "OK", "relative_multi_turn_deg": 45.0},
                )
            )

        paired = session._feedback()[1]

        self.assertEqual(paired["position"], 2048.0)
        self.assertAlmostEqual(paired["sync_age_s"], 0.015)

    def test_active_control_does_not_hide_a_large_force_change(self):
        api_python = str(API_ROOT / "python")
        if api_python not in sys.path:
            sys.path.insert(0, api_python)

        session = ForceControlSession(load_gui_force_config())
        state = SimpleNamespace(
            servo_id=1,
            sensor_id=1,
            model={
                "motor_name": "ED",
                "servo_counts_per_rev": 4096.0,
                "servo_zero_raw": 2048.0,
                "servo_direction": "1",
                "encoder_direction": "1",
                "torque_zero_delta_deg": 0.0,
                "slope_nmm_per_deg": -10.0,
                "angle_reference_deg": 0.0,
                "moment_arm_mm": 20.0,
                "pull_angle_deg": 90.0,
            },
            intercept_nmm=0.0,
            previous_snapshot={"force_n": 0.0, "angle_delta_deg": 0.0},
            latest_force_n=0.0,
            latest_position=2048.0,
            pending_monitoring_jump=None,
            pending_monitoring_jump_received_at=None,
            last_monitoring_jump_log_at=0.0,
            active=True,
        )
        session._servo = object()
        session._servo_store = LatestSampleStore()
        session._servo_index = {1: 0}
        session._states = {1: state}
        session._armed = True
        session._running = True
        timestamp = time.monotonic()
        with session._feedback_history_lock:
            session._servo_feedback_history.append(
                (timestamp, {"valid": [True], "positions": [2048]})
            )
            session._encoder_feedback_histories[1].append(
                (
                    timestamp,
                    {"status": "OK", "relative_multi_turn_deg": 100.0},
                )
            )

        feedback = session._feedback()

        self.assertLess(feedback[1]["force_n"], -20.0)

    def test_feedback_rejects_when_no_synchronized_pair_exists(self):
        session = ForceControlSession(load_gui_force_config())
        state = SimpleNamespace(
            servo_id=1,
            sensor_id=1,
            model={},
            intercept_nmm=0.0,
            previous_snapshot=None,
            latest_force_n=None,
            latest_position=None,
            pending_monitoring_jump={"force_n": 5.0},
            pending_monitoring_jump_received_at=None,
        )
        session._servo = object()
        session._servo_store = LatestSampleStore()
        session._encoder_stores = {1: LatestSampleStore()}
        session._servo_index = {1: 0}
        session._states = {1: state}
        session._manual_speed_commands[1] = 500
        timestamp = time.monotonic() - 0.1
        state.pending_monitoring_jump_received_at = timestamp
        with session._feedback_history_lock:
            session._servo_feedback_history.append(
                (timestamp, {"valid": [True], "positions": [2048]})
            )
            session._encoder_feedback_histories[1].append(
                (
                    timestamp + 0.080,
                    {"status": "OK", "relative_multi_turn_deg": 0.0},
                )
            )

        feedback = session._feedback()
        error = session.feedback_errors_snapshot()[1]

        self.assertEqual(feedback, {})
        self.assertIn("synchronized feedback unavailable", error)
        self.assertIn("closest timestamp gap 80 ms exceeds 30 ms", error)
        self.assertIsNotNone(state.pending_monitoring_jump)
        self.assertEqual(state.pending_monitoring_jump_received_at, timestamp)

    def test_stationary_monitoring_accepts_wider_timestamp_gap(self):
        session = ForceControlSession(load_gui_force_config())
        timestamp = time.monotonic() - 0.45
        with session._feedback_history_lock:
            session._servo_feedback_history.append(
                (timestamp, {"valid": [True], "positions": [2048]})
            )
            session._encoder_feedback_histories[1].append(
                (
                    timestamp + 0.400,
                    {"status": "OK", "relative_multi_turn_deg": 0.0},
                )
            )

        synchronized, error = session._synchronized_feedback_pair(
            1,
            1,
            0,
            time.monotonic(),
            sync_tolerance_s=0.500,
            feedback_timeout_s=0.500,
        )

        self.assertIsNotNone(synchronized)
        self.assertIsNone(error)
        self.assertAlmostEqual(synchronized[2], 0.400)

    def test_feedback_rejects_encoder_sample_without_multiturn_angle(self):
        session = ForceControlSession(load_gui_force_config())
        state = SimpleNamespace(
            servo_id=1,
            sensor_id=1,
            model={},
            intercept_nmm=0.0,
            previous_snapshot=None,
            latest_force_n=None,
            latest_position=None,
        )
        session._servo = object()
        session._servo_store = LatestSampleStore()
        session._encoder_stores = {1: LatestSampleStore()}
        session._servo_index = {1: 0}
        session._states = {1: state}
        timestamp = time.monotonic()
        with session._feedback_history_lock:
            session._servo_feedback_history.append(
                (timestamp, {"valid": [True], "positions": [2048]})
            )
            session._encoder_feedback_histories[1].append(
                (timestamp, {"status": "OK", "relative_deg": 0.0})
            )

        feedback = session._feedback()
        error = session.feedback_errors_snapshot()[1]

        self.assertEqual(feedback, {})
        self.assertIn("no multi-turn angle", error)

    def test_monitoring_jump_filter_rejects_one_sample_spike(self):
        session = ForceControlSession(load_gui_force_config())
        previous = {
            "force_n": 0.0,
            "angle_delta_deg": 0.0,
            "torque_nmm": 0.0,
            "servo_raw": 2048,
            "encoder_angle_deg": 10.0,
        }
        state = SimpleNamespace(
            servo_id=2,
            previous_snapshot=previous,
            pending_monitoring_jump=None,
            pending_monitoring_jump_received_at=None,
            last_monitoring_jump_log_at=0.0,
        )
        spike = {
            "force_n": -20.0,
            "angle_delta_deg": 65.0,
            "torque_nmm": -400.0,
            "servo_raw": 2048,
            "encoder_angle_deg": 75.0,
        }

        held, message = session._filter_monitoring_jump(
            state, spike, now=10.0, sample_received_at=1.0
        )
        recovered, _ = session._filter_monitoring_jump(
            state, previous, now=10.1, sample_received_at=2.0
        )

        self.assertEqual(held["force_n"], 0.0)
        self.assertIn("monitoring jump held", message)
        self.assertEqual(recovered["force_n"], 0.0)
        self.assertIsNone(state.pending_monitoring_jump)

    def test_monitoring_jump_filter_accepts_persistent_change(self):
        session = ForceControlSession(load_gui_force_config())
        previous = {"force_n": 0.0, "angle_delta_deg": 0.0}
        state = SimpleNamespace(
            servo_id=2,
            previous_snapshot=previous,
            pending_monitoring_jump=None,
            pending_monitoring_jump_received_at=None,
            last_monitoring_jump_log_at=0.0,
        )
        first = {
            "force_n": 5.0,
            "angle_delta_deg": 20.0,
            "servo_raw": 2048,
            "encoder_angle_deg": 30.0,
        }
        second = {
            "force_n": 5.2,
            "angle_delta_deg": 20.5,
            "servo_raw": 2048,
            "encoder_angle_deg": 30.5,
        }

        held, _ = session._filter_monitoring_jump(
            state, first, now=10.0, sample_received_at=1.0
        )
        held_again, _ = session._filter_monitoring_jump(
            state, first, now=10.05, sample_received_at=1.0
        )
        accepted, _ = session._filter_monitoring_jump(
            state, second, now=10.1, sample_received_at=1.2
        )

        self.assertEqual(held["force_n"], 0.0)
        self.assertEqual(held_again["force_n"], 0.0)
        self.assertEqual(accepted["force_n"], 5.2)
        self.assertIsNone(state.pending_monitoring_jump)

    def test_monitoring_jump_filter_restarts_confirmation_after_long_gap(self):
        session = ForceControlSession(load_gui_force_config())
        previous = {"force_n": 0.0, "angle_delta_deg": 0.0}
        state = SimpleNamespace(
            servo_id=2,
            previous_snapshot=previous,
            pending_monitoring_jump=None,
            pending_monitoring_jump_received_at=None,
            last_monitoring_jump_log_at=0.0,
        )
        changed = {
            "force_n": 5.0,
            "angle_delta_deg": 20.0,
            "servo_raw": 2048,
            "encoder_angle_deg": 30.0,
        }

        first, _ = session._filter_monitoring_jump(
            state, changed, now=10.0, sample_received_at=1.0
        )
        after_gap, _ = session._filter_monitoring_jump(
            state, changed, now=11.2, sample_received_at=2.2
        )
        confirmed, _ = session._filter_monitoring_jump(
            state, changed, now=11.4, sample_received_at=2.4
        )

        self.assertEqual(first["force_n"], 0.0)
        self.assertEqual(after_gap["force_n"], 0.0)
        self.assertEqual(confirmed["force_n"], 5.0)


if __name__ == "__main__":
    unittest.main()
