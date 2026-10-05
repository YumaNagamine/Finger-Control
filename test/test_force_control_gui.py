from __future__ import annotations

import unittest
from pathlib import Path

from controller.servo.force_control.gui.session import (
    DEFAULT_CALIBRATION_DIR,
    DEFAULT_FORCE_CONFIG_PATH,
    ForceControlSession,
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


if __name__ == "__main__":
    unittest.main()
