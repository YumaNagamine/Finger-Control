"""Asynchronously save rendered vision overlays without blocking control."""

from __future__ import annotations

import threading
from dataclasses import dataclass
from pathlib import Path
from queue import Full, Queue
from typing import Any

import cv2
import numpy as np


_STOP = object()


@dataclass(frozen=True)
class OverlayVideoSummary:
    path: Path
    frames_written: int
    dropped_frames: int


class OverlayVideoRecorder:
    """Write BGR overlay frames on a background thread.

    Frames are copied into a bounded queue so the camera/control thread does
    not wait for codec or filesystem I/O. If the queue is full, the newest
    frame is dropped and the count is reported in the final summary.
    """

    def __init__(
        self,
        path: Path,
        *,
        fps: float,
        codec: str = "mp4v",
        queue_size: int = 10,
    ) -> None:
        if not np.isfinite(fps) or fps <= 0.0:
            raise ValueError("fps must be finite and greater than zero")
        if len(codec) != 4:
            raise ValueError("codec must contain exactly four characters")
        if queue_size <= 0:
            raise ValueError("queue_size must be greater than zero")

        self._path = Path(path)
        self._fps = float(fps)
        self._codec = codec
        self._queue: Queue[Any] = Queue(maxsize=int(queue_size))
        self._thread: threading.Thread | None = None
        self._writer: cv2.VideoWriter | None = None
        self._frame_shape: tuple[int, int] | None = None
        self._error: BaseException | None = None
        self._state_lock = threading.Lock()
        self._started = False
        self._stopped = False
        self._frames_written = 0
        self._dropped_frames = 0

    @property
    def path(self) -> Path:
        return self._path

    @property
    def dropped_frames(self) -> int:
        with self._state_lock:
            return self._dropped_frames

    def start(self) -> None:
        with self._state_lock:
            if self._started:
                raise RuntimeError("overlay video recorder is already started")
            self._path.parent.mkdir(parents=True, exist_ok=True)
            self._started = True
            self._thread = threading.Thread(
                target=self._run,
                name="overlay-video-writer",
                daemon=True,
            )
            self._thread.start()

    def write(self, frame: np.ndarray) -> bool:
        """Queue one BGR uint8 frame; return False when it was dropped."""
        with self._state_lock:
            if not self._started or self._stopped:
                raise RuntimeError("overlay video recorder is not running")
            self._raise_if_failed_locked()

        image = np.asarray(frame)
        if image.ndim != 3 or image.shape[2] != 3:
            raise ValueError("overlay frame must have shape (height, width, 3)")
        if image.dtype != np.uint8:
            raise ValueError("overlay frame must use uint8 pixels")

        try:
            self._queue.put_nowait(image.copy())
        except Full:
            with self._state_lock:
                self._dropped_frames += 1
            return False
        return True

    def stop(self) -> OverlayVideoSummary:
        with self._state_lock:
            if not self._started:
                raise RuntimeError("overlay video recorder has not been started")
            if self._stopped:
                self._raise_if_failed_locked()
                return OverlayVideoSummary(
                    path=self._path,
                    frames_written=self._frames_written,
                    dropped_frames=self._dropped_frames,
                )
            self._stopped = True
            thread = self._thread

        if thread is not None:
            while thread.is_alive():
                try:
                    self._queue.put(_STOP, timeout=0.1)
                    break
                except Full:
                    continue
            thread.join()

        with self._state_lock:
            self._raise_if_failed_locked()
            return OverlayVideoSummary(
                path=self._path,
                frames_written=self._frames_written,
                dropped_frames=self._dropped_frames,
            )

    def _run(self) -> None:
        try:
            while True:
                item = self._queue.get()
                try:
                    if item is _STOP:
                        return
                    frame = np.asarray(item)
                    if self._writer is None:
                        height, width = frame.shape[:2]
                        self._frame_shape = (int(height), int(width))
                        writer = cv2.VideoWriter(
                            str(self._path),
                            cv2.VideoWriter_fourcc(*self._codec),
                            self._fps,
                            (int(width), int(height)),
                        )
                        if not writer.isOpened():
                            writer.release()
                            raise RuntimeError(
                                f"Failed to open overlay video writer: {self._path}"
                            )
                        self._writer = writer

                    if self._frame_shape is None:
                        raise RuntimeError("video frame size is not initialized")
                    if frame.shape[:2] != self._frame_shape:
                        raise ValueError(
                            "overlay frame dimensions changed during recording"
                        )
                    self._writer.write(frame)
                    with self._state_lock:
                        self._frames_written += 1
                finally:
                    self._queue.task_done()
        except BaseException as exc:
            with self._state_lock:
                self._error = exc
        finally:
            if self._writer is not None:
                self._writer.release()

    def _raise_if_failed_locked(self) -> None:
        if self._error is not None:
            raise RuntimeError("overlay video writer failed") from self._error
