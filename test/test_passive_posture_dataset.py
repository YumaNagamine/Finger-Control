import csv
import json
import sys
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import TestCase

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from observation.vision.passive_posture_dataset import (
    capture_postures,
    export_approved,
    process_dataset,
)


class MockProcessor:
    def __init__(self, likelihood=0.95, missing=False):
        self.likelihood = likelihood
        self.missing = missing
        self.reset_count = 0

    def reset(self):
        self.reset_count += 1

    def process_frame(self, image, frame_index):
        status = "missing" if self.missing else "detected"
        keypoints = {
            name: {"x": 10.0, "y": 10.0, "likelihood": self.likelihood, "status": status}
            for name in ("fingertip", "distal1", "middle1", "middle2", "proximal1", "proximal2", "meta1")
        }
        return (
            {
                "angles": {"DIP": 167.5, "PIP": 161.7, "MCP": 137.9},
                "keypoints": keypoints,
            },
            image.copy(),
        )


class MockCapture:
    def __init__(self):
        self.released = False

    def read(self):
        return True, np.full((120, 160, 3), 127, dtype=np.uint8)

    def release(self):
        self.released = True


def _write_fixture_dataset(root: Path, count=2):
    images = root / "images"
    images.mkdir(parents=True)
    metadata = []
    for index in range(1, count + 1):
        posture_id = f"posture_{index:04d}"
        cv2.imwrite(str(images / f"{posture_id}.png"), np.zeros((120, 160, 3), dtype=np.uint8))
        metadata.append({"posture_id": posture_id, "captured_at": f"2026-10-28T00:00:0{index}+09:00"})
    (root / "capture_metadata.json").write_text(
        json.dumps({"images": metadata}), encoding="utf-8"
    )


class TestPassivePostureDataset(TestCase):
    def test_capture_saves_one_original_image_and_metadata_per_enter(self):
        with TemporaryDirectory() as temporary:
            dataset = Path(temporary) / "dataset_001"
            camera_config = Path(temporary) / "camera.json"
            camera_config.write_text(
                json.dumps({"index": 0, "use_chessboard_calibration": False}),
                encoding="utf-8",
            )
            capture = MockCapture()
            keys = iter((13, ord("q")))

            count = capture_postures(
                dataset,
                camera_config,
                capture_factory=lambda _config: capture,
                show_window=False,
                key_provider=lambda: next(keys),
            )

            self.assertEqual(count, 1)
            self.assertTrue(capture.released)
            self.assertTrue((dataset / "images" / "posture_0001.png").exists())
            metadata = json.loads(
                (dataset / "capture_metadata.json").read_text(encoding="utf-8")
            )
            self.assertEqual(metadata["images"][0]["posture_id"], "posture_0001")
            self.assertEqual(metadata["images"][0]["color_order"], "BGR")

    def test_process_creates_p2p_flexion_csv_and_review_artifacts(self):
        with TemporaryDirectory() as temporary:
            dataset = Path(temporary)
            _write_fixture_dataset(dataset)
            config = dataset / "dlc.json"
            config.write_text("{}", encoding="utf-8")
            processor = MockProcessor()

            rows = process_dataset(dataset, config, processor=processor)

            self.assertEqual(processor.reset_count, 2)
            self.assertEqual([row["status"] for row in rows], ["pending_review", "pending_review"])
            self.assertAlmostEqual(rows[0]["dip_deg"], 12.5)
            self.assertAlmostEqual(rows[0]["pip_deg"], 18.3)
            self.assertAlmostEqual(rows[0]["mcp_deg"], 42.1)
            self.assertTrue((dataset / "annotated" / "posture_0001.png").exists())
            self.assertTrue((dataset / "posture_goals.csv").exists())
            self.assertTrue((dataset / "overview.png").exists())

    def test_low_confidence_is_failed_and_never_exported(self):
        with TemporaryDirectory() as temporary:
            dataset = Path(temporary)
            _write_fixture_dataset(dataset, count=1)
            config = dataset / "dlc.json"
            config.write_text("{}", encoding="utf-8")

            rows = process_dataset(dataset, config, processor=MockProcessor(likelihood=0.1))
            self.assertEqual(rows[0]["status"], "failed")

            goals = dataset / "posture_goals.csv"
            with goals.open("r", newline="", encoding="utf-8-sig") as stream:
                csv_rows = list(csv.DictReader(stream))
            csv_rows[0]["approved"] = "true"
            with goals.open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=csv_rows[0].keys())
                writer.writeheader()
                writer.writerows(csv_rows)

            self.assertEqual(export_approved(dataset), 0)
            with (dataset / "approved_goals.csv").open("r", encoding="utf-8-sig") as stream:
                self.assertEqual(len(list(csv.DictReader(stream))), 0)

    def test_only_explicitly_approved_success_is_exported(self):
        with TemporaryDirectory() as temporary:
            dataset = Path(temporary)
            _write_fixture_dataset(dataset, count=2)
            config = dataset / "dlc.json"
            config.write_text("{}", encoding="utf-8")
            process_dataset(dataset, config, processor=MockProcessor())

            goals = dataset / "posture_goals.csv"
            with goals.open("r", newline="", encoding="utf-8-sig") as stream:
                rows = list(csv.DictReader(stream))
            rows[0]["approved"] = "yes"
            with goals.open("w", newline="", encoding="utf-8-sig") as stream:
                writer = csv.DictWriter(stream, fieldnames=rows[0].keys())
                writer.writeheader()
                writer.writerows(rows)

            self.assertEqual(export_approved(dataset), 1)
            with (dataset / "approved_goals.csv").open("r", encoding="utf-8-sig") as stream:
                approved = list(csv.DictReader(stream))
            self.assertEqual(approved[0]["posture_id"], "posture_0001")
            self.assertEqual(approved[0]["status"], "approved")

