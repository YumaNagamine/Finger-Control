from __future__ import annotations

import tempfile
import sys
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from controller.servo.force_control.gui.session import (
    API_ROOT,
    DEFAULT_CALIBRATION_DIR,
    DEFAULT_FORCE_CONFIG_PATH,
    ForceControlSession,
    MANUAL_SPEED_PRESETS,
    load_gui_force_config,
)
from controller.servo.force_control.runner import LatestSampleStore
from controller.servo.torque_calibration.storage import load_torque_intercept_calibration


class ForceControlGuiConfigTest(unittest.TestCase):
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
                "LU": 10.0,
                "PI": 20.0,
            },
        )
        self.assertEqual(config.ramp_duration_s, 1.0)

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
        self.assertEqual(Path(mapping[0]["calibration_path"]).parent, DEFAULT_CALIBRATION_DIR)

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

    def test_manual_feedback_error_stops_speed_without_latching_fault(self):
        session = ForceControlSession(load_gui_force_config())
        session._connected = True
        session._servo = Mock()
        session._manual_speed_commands[1] = 500

        handled = session._handle_manual_feedback_error(RuntimeError("invalid telemetry"))

        self.assertTrue(handled)
        self.assertFalse(session.faulted)
        self.assertTrue(session.manual_feedback_recovering)
        self.assertTrue(all(speed == 0 for speed in session._manual_speed_commands.values()))
        self.assertEqual(session._servo.send_speed_by_id.call_count, 6)
        self.assertTrue(
            all(call.args[1] == 0 for call in session._servo.send_speed_by_id.call_args_list)
        )

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

    def test_feedback_holds_previous_force_when_streams_are_not_synchronized(self):
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
        )
        servo_store = LatestSampleStore()
        encoder_store = LatestSampleStore()
        session._servo = object()
        session._servo_store = servo_store
        session._encoder_stores = {1: encoder_store}
        session._servo_index = {1: 0}
        session._states = {1: state}

        timestamp = time.monotonic()
        servo_store.publish({"valid": [True], "positions": [2048]}, received_at=timestamp)
        encoder_store.publish(
            {"status": "OK", "relative_multi_turn_deg": 0.0},
            received_at=timestamp,
        )
        initial = session._feedback()[1]

        encoder_store.publish(
            {"status": "OK", "relative_multi_turn_deg": 45.0},
            received_at=timestamp + 0.040,
        )
        held = session._feedback()[1]

        self.assertAlmostEqual(held["force_n"], initial["force_n"])
        self.assertAlmostEqual(held["angle_delta_deg"], initial["angle_delta_deg"])


if __name__ == "__main__":
    unittest.main()
