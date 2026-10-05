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
        self.assertEqual(config.target_force_levels_n, (0.5, 1.0, 1.5, 2.0))
        self.assertEqual(config.min_target_force_n, 0.5)
        self.assertEqual(config.max_target_force_n, 2.0)
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


if __name__ == "__main__":
    unittest.main()
