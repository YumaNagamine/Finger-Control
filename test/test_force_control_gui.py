from __future__ import annotations

import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import Mock

from controller.servo.force_control.gui.session import (
    DEFAULT_CALIBRATION_DIR,
    DEFAULT_FORCE_CONFIG_PATH,
    ForceControlSession,
    MANUAL_SPEED_PRESETS,
    load_gui_force_config,
)


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


if __name__ == "__main__":
    unittest.main()
