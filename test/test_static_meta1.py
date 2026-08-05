from __future__ import annotations

import json
import unittest
from pathlib import Path

import numpy as np

from observation.vision.dlc_angle_processor import DLCAngleProcessor
from observation.vision.realtime_joint_angle_source import (
    select_dominant_keypoint_cluster,
)


PROJECT_ROOT = Path(__file__).resolve().parents[1]
DLC_CONFIG_PATH = PROJECT_ROOT / "src" / "observation" / "vision" / "config_deeplabcut_angle.json"


class StaticMeta1Test(unittest.TestCase):
    def test_dominant_cluster_uses_coordinate_median_and_rejects_outliers(self) -> None:
        candidates = [
            (100.0, 200.0, 0.90),
            (101.0, 199.0, 0.92),
            (99.0, 201.0, 0.88),
            (102.0, 200.0, 0.91),
            (100.0, 202.0, 0.89),
            (98.0, 200.0, 0.93),
            (100.0, 198.0, 0.87),
            (400.0, 50.0, 0.95),
            (420.0, 60.0, 0.96),
            (440.0, 70.0, 0.97),
        ]

        cluster = select_dominant_keypoint_cluster(
            candidates,
            radius_px=5.0,
            min_cluster_samples=6,
            min_cluster_ratio=0.6,
        )

        self.assertEqual(cluster.position, (100.0, 200.0))
        self.assertEqual(len(cluster.member_indices), 7)
        self.assertAlmostEqual(cluster.ratio, 0.7)
        self.assertEqual(cluster.likelihood, 0.90)

    def test_dominant_cluster_rejects_insufficient_ratio(self) -> None:
        candidates = [
            (0.0, 0.0, 0.9),
            (1.0, 0.0, 0.9),
            (0.0, 1.0, 0.9),
            (50.0, 50.0, 0.9),
            (51.0, 50.0, 0.9),
            (50.0, 51.0, 0.9),
        ]

        with self.assertRaisesRegex(ValueError, "Dominant cluster is too small"):
            select_dominant_keypoint_cluster(
                candidates,
                radius_px=3.0,
                min_cluster_samples=3,
                min_cluster_ratio=0.6,
            )

    def test_latched_meta1_is_used_when_current_detection_is_missing(self) -> None:
        processor = self._make_processor()
        raw_keypoints = self._raw_keypoints(meta1=(float("nan"), float("nan"), 0.0))
        frame = np.zeros((240, 320, 3), dtype=np.uint8)

        before, _ = processor.process_keypoints(frame, 0, raw_keypoints)
        self.assertEqual(before["keypoints"]["meta1"]["status"], "missing")
        self.assertFalse(np.isfinite(before["angles"]["MCP"]))

        processor.set_latched_keypoint("meta1", (150.0, 160.0), 0.9)
        after, overlay = processor.process_keypoints(frame, 1, raw_keypoints)

        self.assertEqual(after["keypoints"]["meta1"]["status"], "latched")
        self.assertEqual(after["keypoints"]["meta1"]["x"], 150.0)
        self.assertEqual(after["keypoints"]["meta1"]["y"], 160.0)
        self.assertTrue(np.isfinite(after["angles"]["MCP"]))
        self.assertEqual(tuple(overlay[160, 150]), (255, 255, 0))

    def test_unlatched_meta1_keeps_using_each_detected_position(self) -> None:
        processor = self._make_processor()
        frame = np.zeros((240, 320, 3), dtype=np.uint8)

        first, _ = processor.process_keypoints(
            frame,
            0,
            self._raw_keypoints(meta1=(150.0, 160.0, 0.9)),
        )
        second, _ = processor.process_keypoints(
            frame,
            1,
            self._raw_keypoints(meta1=(170.0, 180.0, 0.9)),
        )

        self.assertEqual(first["keypoints"]["meta1"]["status"], "detected")
        self.assertEqual(first["keypoints"]["meta1"]["x"], 150.0)
        self.assertEqual(second["keypoints"]["meta1"]["x"], 170.0)
        self.assertEqual(second["keypoints"]["meta1"]["y"], 180.0)

    @staticmethod
    def _make_processor() -> DLCAngleProcessor:
        config = json.loads(DLC_CONFIG_PATH.read_text(encoding="utf-8"))
        return DLCAngleProcessor(config, DLC_CONFIG_PATH.parent, enable_live=False)

    @staticmethod
    def _raw_keypoints(
        *,
        meta1: tuple[float, float, float],
    ) -> dict[str, tuple[float, float, float]]:
        return {
            "fingertip": (30.0, 30.0, 0.9),
            "distal1": (50.0, 40.0, 0.9),
            "middle1": (55.0, 50.0, 0.9),
            "middle2": (80.0, 65.0, 0.9),
            "proximal1": (85.0, 75.0, 0.9),
            "proximal2": (115.0, 100.0, 0.9),
            "meta1": meta1,
        }


if __name__ == "__main__":
    unittest.main()
