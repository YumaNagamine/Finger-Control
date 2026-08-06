from __future__ import annotations

import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

import numpy as np

from observation.vision.overlay_video_recorder import OverlayVideoRecorder


class FakeVideoWriter:
    def __init__(self) -> None:
        self.frames: list[np.ndarray] = []
        self.released = False

    def isOpened(self) -> bool:
        return True

    def write(self, frame: np.ndarray) -> None:
        self.frames.append(frame.copy())

    def release(self) -> None:
        self.released = True


class OverlayVideoRecorderTest(unittest.TestCase):
    def test_writes_frames_and_releases_writer(self) -> None:
        writer = FakeVideoWriter()
        frame = np.zeros((12, 16, 3), dtype=np.uint8)
        frame[3, 4] = (1, 2, 3)

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "overlay.mp4"
            with patch(
                "observation.vision.overlay_video_recorder.cv2.VideoWriter",
                return_value=writer,
            ) as video_writer:
                recorder = OverlayVideoRecorder(path, fps=5.0, queue_size=2)
                recorder.start()
                self.assertTrue(recorder.write(frame))
                summary = recorder.stop()

        self.assertEqual(summary.path, path)
        self.assertEqual(summary.frames_written, 1)
        self.assertEqual(summary.dropped_frames, 0)
        self.assertTrue(writer.released)
        np.testing.assert_array_equal(writer.frames, [frame])
        video_writer.assert_called_once()

    def test_rejects_changed_frame_dimensions(self) -> None:
        writer = FakeVideoWriter()
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "overlay.mp4"
            with patch(
                "observation.vision.overlay_video_recorder.cv2.VideoWriter",
                return_value=writer,
            ):
                recorder = OverlayVideoRecorder(path, fps=5.0)
                recorder.start()
                self.assertTrue(recorder.write(np.zeros((10, 10, 3), dtype=np.uint8)))
                self.assertTrue(recorder.write(np.zeros((11, 10, 3), dtype=np.uint8)))
                with self.assertRaisesRegex(RuntimeError, "overlay video writer failed"):
                    recorder.stop()


if __name__ == "__main__":
    unittest.main()
