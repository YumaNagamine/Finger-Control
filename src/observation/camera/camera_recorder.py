from __future__ import annotations

import csv
import datetime
import math
import threading
import time
from pathlib import Path

import cv2

from observation.camera.camera_utils import (
    apply_camera_settings,
    fourcc_from_str,
    resolve_backend,
    setup_undistortion_from_config,
    undistort_frame,
)


class CameraRecorder:
    """Record a USB camera to video, optionally with per-frame timestamps."""

    def __init__(
        self,
        camera_config: dict,
        video_path: Path,
        timestamp_csv_path: Path | None = None,
        *,
        warmup_frames: int = 0,
        show_preview: bool = False,
        preview_window_name: str = "Camera Recording",
        preview_escape_is_error: bool = True,
        frame_limit: int | None = None,
    ) -> None:
        self.camera_config = dict(camera_config)
        self.video_path = video_path
        self.timestamp_csv_path = timestamp_csv_path
        self.warmup_frames = warmup_frames
        self.show_preview = show_preview
        self.preview_window_name = preview_window_name
        self.preview_escape_is_error = preview_escape_is_error
        if frame_limit is not None and frame_limit <= 0:
            raise ValueError("frame_limit must be greater than zero when specified")
        self.frame_limit = frame_limit

        self.cap = None
        self.writer = None
        self.calibration = None
        self.timestamp_file = None
        self.timestamp_writer = None
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.error: Exception | None = None
        self.started_at: float | None = None
        self.frame_count = 0
        self.actual_size: tuple[int, int] | None = None
        self.preview_created = False

    def prepare(self) -> None:
        camera_index = int(self.camera_config.get("index", 0))
        backend = resolve_backend(self.camera_config.get("backend"))
        self.cap = (
            cv2.VideoCapture(camera_index, backend)
            if backend is not None
            else cv2.VideoCapture(camera_index)
        )
        if not self.cap.isOpened():
            raise RuntimeError(f"Camera {camera_index} is not available")

        apply_camera_settings(self.cap, self.camera_config)
        self.calibration = setup_undistortion_from_config(
            self.camera_config,
            log_prefix="[camera-recorder]",
        )

        for _ in range(self.warmup_frames):
            self.cap.read()
        ok, initial_frame = self.cap.read()
        if not ok or initial_frame is None:
            raise RuntimeError("Failed to read an initial camera frame")
        initial_frame = undistort_frame(initial_frame, self.calibration)
        height, width = initial_frame.shape[:2]
        self.actual_size = (width, height)

        writer_fourcc = fourcc_from_str(self.camera_config.get("writer_fourcc", "mp4v"))
        if writer_fourcc is None:
            raise ValueError("Camera writer_fourcc must contain exactly four characters")
        target_fps = float(self.camera_config.get("target_fps", 90.0))
        if target_fps <= 0.0 or not math.isfinite(target_fps):
            raise ValueError("Camera target_fps must be finite and greater than zero")
        self.writer = cv2.VideoWriter(
            str(self.video_path),
            writer_fourcc,
            target_fps,
            (width, height),
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {self.video_path}")

        if self.timestamp_csv_path is not None:
            self.timestamp_file = self.timestamp_csv_path.open(
                "w",
                encoding="utf-8",
                newline="",
            )
            self.timestamp_writer = csv.writer(self.timestamp_file)
            self.timestamp_writer.writerow(["frame_idx", "timestamp_iso", "elapsed_s"])

        if self.show_preview:
            cv2.namedWindow(self.preview_window_name, cv2.WINDOW_NORMAL)
            self.preview_created = True
        print(
            f"Camera ready: index={camera_index}, actual={width}x{height}, "
            f"target_fps={target_fps:.1f}"
        )

    def start(self, started_at: float | None = None) -> None:
        if self.cap is None or self.writer is None:
            raise RuntimeError("Call prepare() before starting camera recording")
        if self.thread is not None:
            raise RuntimeError("Camera recorder has already been started")
        self.started_at = time.monotonic() if started_at is None else started_at
        self.thread = threading.Thread(
            target=self._record_loop,
            name="usb-camera-recorder",
            daemon=True,
        )
        self.thread.start()

    @property
    def stop_requested(self) -> bool:
        return self.stop_event.is_set()

    def raise_if_failed(self) -> None:
        with self.lock:
            error = self.error
        if error is not None:
            raise RuntimeError("Camera recording failed") from error

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            if self.thread.is_alive() and self.cap is not None:
                self.cap.release()
                self.thread.join(timeout=1.0)

    def close(self) -> None:
        self.stop()
        if self.timestamp_file is not None:
            self.timestamp_file.flush()
            self.timestamp_file.close()
            self.timestamp_file = None
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.preview_created:
            cv2.destroyWindow(self.preview_window_name)
            self.preview_created = False

    def _record_loop(self) -> None:
        try:
            if self.started_at is None:
                raise RuntimeError("Camera recording has no start time")
            while not self.stop_event.is_set():
                ok, frame = self.cap.read()
                captured_at = time.monotonic()
                if not ok or frame is None:
                    if self.stop_event.is_set():
                        break
                    raise RuntimeError("Camera returned no frame")

                frame = undistort_frame(frame, self.calibration)
                self.writer.write(frame)
                if self.timestamp_writer is not None:
                    timestamp_iso = datetime.datetime.now().isoformat(timespec="milliseconds")
                    self.timestamp_writer.writerow(
                        [
                            self.frame_count,
                            timestamp_iso,
                            f"{captured_at - self.started_at:.6f}",
                        ]
                    )
                self.frame_count += 1
                if self.timestamp_file is not None and self.frame_count % 30 == 0:
                    self.timestamp_file.flush()
                if self.frame_limit is not None and self.frame_count >= self.frame_limit:
                    self.stop_event.set()

                if self.show_preview:
                    cv2.imshow(self.preview_window_name, frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        if self.preview_escape_is_error:
                            raise RuntimeError("Camera preview was stopped with ESC")
                        self.stop_event.set()
        except Exception as exc:
            if not self.stop_event.is_set():
                with self.lock:
                    self.error = exc
