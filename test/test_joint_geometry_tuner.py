from __future__ import annotations

import json
import tempfile
import unittest
from datetime import datetime
from pathlib import Path

import numpy as np

from observation.vision.dlc_angle_processor import DLCAngleProcessor
from observation.vision.joint_geometry_tuner import save_adjustments_snapshot


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DLC_CONFIG_PATH = PROJECT_ROOT / "src" / "observation" / "vision" / "config_deeplabcut_angle.json"


class JointGeometryAdjustmentTest(unittest.TestCase):
    def test_medial_rotation_changes_dip_and_pip_angles(self) -> None:
        processor = self._make_processor()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        original, _ = processor.process_keypoints(frame, 0, self._raw_keypoints())

        adjustments = processor.get_adjustments()
        adjustments["theta_rad"] = float(adjustments["theta_rad"]) + 0.2
        processor.update_adjustments(adjustments)
        updated, _ = processor.process_keypoints(frame, 1, self._raw_keypoints())

        self.assertNotAlmostEqual(original["angles"]["DIP"], updated["angles"]["DIP"])
        self.assertNotAlmostEqual(original["angles"]["PIP"], updated["angles"]["PIP"])

    def test_runtime_update_changes_joint_positions(self) -> None:
        processor = self._make_processor()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)
        original, _ = processor.process_keypoints(frame, 0, self._raw_keypoints())

        adjustments = processor.get_adjustments()
        original_joint_shifters = adjustments["joint_shifters"]
        original_mcp_offset = adjustments["mcp_offset"]
        self.assertIsInstance(original_joint_shifters, list)
        self.assertIsInstance(original_mcp_offset, list)
        assert isinstance(original_joint_shifters, list)
        assert isinstance(original_mcp_offset, list)

        adjustments["joint_shifters"] = [
            original_joint_shifters[0] + 10.0,
            original_joint_shifters[1] + 20.0,
        ]
        adjustments["mcp_offset"] = [original_mcp_offset[0] + 30.0, original_mcp_offset[1] - 5.0]
        processor.update_adjustments(adjustments)
        updated, _ = processor.process_keypoints(frame, 1, self._raw_keypoints())

        original_dip = np.asarray(original["joints"]["DIP"])
        original_pip = np.asarray(original["joints"]["PIP"])
        original_mcp = np.asarray(original["joints"]["MCP"])
        updated_dip = np.asarray(updated["joints"]["DIP"])
        updated_pip = np.asarray(updated["joints"]["PIP"])
        updated_mcp = np.asarray(updated["joints"]["MCP"])

        self.assertAlmostEqual(float(np.linalg.norm(updated_dip - original_dip)), 10.0, places=4)
        self.assertAlmostEqual(float(np.linalg.norm(updated_pip - original_pip)), 20.0, places=4)
        np.testing.assert_allclose(updated_mcp - original_mcp, [30.0, -5.0])

    def test_runtime_update_rejects_invalid_values_without_partial_update(self) -> None:
        processor = self._make_processor()
        before = processor.get_adjustments()

        with self.assertRaises(ValueError):
            processor.update_adjustments({"theta_rad": 0.8, "joint_shifters": [1.0]})

        self.assertEqual(processor.get_adjustments(), before)

    def test_snapshot_is_created_next_to_source_without_modifying_source(self) -> None:
        with tempfile.TemporaryDirectory() as temp_dir:
            source = Path(temp_dir) / "config_deeplabcut_angle.json"
            source_contents = '{"source": true}\n'
            source.write_text(source_contents, encoding="utf-8")
            adjustments = self._make_processor().get_adjustments()

            output = save_adjustments_snapshot(
                source,
                adjustments,
                now=datetime(2026, 9, 1, 14, 30, 0),
            )

            self.assertEqual(output.parent, source.parent)
            self.assertEqual(output.name, "config_deeplabcut_angle_adjustments_20260901_143000.json")
            self.assertEqual(source.read_text(encoding="utf-8"), source_contents)
            payload = json.loads(output.read_text(encoding="utf-8"))
            self.assertEqual(payload, {"processing": {"adjustments": adjustments}})

    @staticmethod
    def _make_processor() -> DLCAngleProcessor:
        config = json.loads(DLC_CONFIG_PATH.read_text(encoding="utf-8"))
        return DLCAngleProcessor(config, DLC_CONFIG_PATH.parent, enable_live=False)

    @staticmethod
    def _raw_keypoints() -> dict[str, tuple[float, float, float]]:
        return {
            "fingertip": (30.0, 30.0, 0.9),
            "distal1": (50.0, 40.0, 0.9),
            "middle1": (55.0, 50.0, 0.9),
            "middle2": (80.0, 65.0, 0.9),
            "proximal1": (85.0, 75.0, 0.9),
            "proximal2": (115.0, 100.0, 0.9),
            "meta1": (150.0, 160.0, 0.9),
        }


if __name__ == "__main__":
    unittest.main()
