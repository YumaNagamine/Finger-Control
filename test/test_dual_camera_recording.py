from __future__ import annotations

import csv
import tempfile
import unittest
from pathlib import Path

import cv2
import numpy as np

from observation.vision.process_dual_recording import _read_frame_at_index
from observation.vision.record_dual_cameras import _write_paired_timestamps


class FakeCapture:
    def __init__(self, frames: list[np.ndarray]) -> None:
        self.frames = frames
        self.position = 0
        self.read_count = 0
        self.seek_count = 0

    def set(self, property_id: int, value: int) -> bool:
        self.assert_position_property(property_id)
        self.position = int(value)
        self.seek_count += 1
        return True

    def read(self) -> tuple[bool, np.ndarray | None]:
        self.read_count += 1
        if self.position >= len(self.frames):
            return False, None
        frame = self.frames[self.position]
        self.position += 1
        return True, frame

    @staticmethod
    def assert_position_property(property_id: int) -> None:
        if property_id != cv2.CAP_PROP_POS_FRAMES:
            raise AssertionError(f"Unexpected capture property: {property_id}")


class IndexedFrameReadTest(unittest.TestCase):
    def test_sequential_duplicate_and_skipped_frames(self) -> None:
        capture = FakeCapture(
            [np.full((2, 2, 3), value, dtype=np.uint8) for value in range(5)]
        )
        state = (0, None, None)

        values = []
        for target_index in (0, 1, 1, 3):
            ok, frame, next_index, cached_index, cached_frame = _read_frame_at_index(
                capture,
                target_index,
                state[0],
                state[1],
                state[2],
            )
            self.assertTrue(ok)
            self.assertIsNotNone(frame)
            values.append(int(frame[0, 0, 0]))
            state = (next_index, cached_index, cached_frame)

        self.assertEqual(values, [0, 1, 1, 3])
        self.assertEqual(capture.read_count, 3)
        self.assertEqual(capture.seek_count, 1)


class TimestampPairingTest(unittest.TestCase):
    @staticmethod
    def _write_timestamps(path: Path, elapsed_values: list[float]) -> None:
        with path.open("w", newline="", encoding="utf-8") as csv_file:
            writer = csv.writer(csv_file)
            writer.writerow(["frame_idx", "timestamp_iso", "elapsed_s"])
            for frame_index, elapsed_s in enumerate(elapsed_values):
                writer.writerow([frame_index, f"timestamp-{frame_index}", elapsed_s])

    def test_nearest_timestamp_can_reuse_camera_frame(self) -> None:
        with tempfile.TemporaryDirectory() as temporary_directory:
            root = Path(temporary_directory)
            cam0_path = root / "cam0_timestamps.csv"
            cam1_path = root / "cam1_timestamps.csv"
            pair_path = root / "pair_timestamps.csv"
            self._write_timestamps(cam0_path, [0.00, 0.01, 0.02, 0.03])
            self._write_timestamps(cam1_path, [0.00, 0.02, 0.04])

            pair_count = _write_paired_timestamps(cam0_path, cam1_path, pair_path)
            with pair_path.open("r", newline="", encoding="utf-8") as csv_file:
                rows = list(csv.DictReader(csv_file))

            self.assertEqual(pair_count, 4)
            self.assertEqual(
                [int(row["cam1_frame_idx"]) for row in rows],
                [0, 0, 1, 1],
            )


if __name__ == "__main__":
    unittest.main()
