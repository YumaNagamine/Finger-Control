"""Synchronous camera and DLC joint-angle acquisition."""

from __future__ import annotations

import time
import math
from dataclasses import dataclass
from pathlib import Path

import cv2
import numpy as np

from observation.camera.camera_utils import (
    apply_camera_settings,
    resolve_backend,
    setup_undistortion_from_config,
    undistort_frame,
)
from observation.vision.dlc_angle_processor import DLCAngleProcessor
from utils.config_loader import load_config
from utils.path_utils import resolve_path


JOINTS = ("DIP", "PIP", "MCP")


@dataclass(frozen=True)
class JointAngleMeasurement:
    frame_index: int
    captured_at: float
    inference_finished_at: float
    flexion_angles_rad: tuple[float, float, float]
    min_likelihood: float
    inference_ms: float
    valid: bool
    reason: str | None


class RealtimeJointAngleSource:
    def __init__(
        self,
        *,
        dlc_config_path: Path,
        camera_config_path: Path,
        min_likelihood: float,
    ) -> None:
        if not math.isfinite(min_likelihood) or not 0.0 <= min_likelihood <= 1.0:
            raise ValueError("min_likelihood must be in the range 0-1")
        self._dlc_config_path = dlc_config_path.expanduser().resolve()
        self._camera_config_path = camera_config_path.expanduser().resolve()
        self._min_likelihood = float(min_likelihood)
        self._capture: cv2.VideoCapture | None = None
        self._processor: DLCAngleProcessor | None = None
        self._calibration = None

    def open(self) -> None:
        if self._capture is not None:
            raise RuntimeError("Joint-angle source is already open")

        runtime_config = self._resolve_dlc_runtime_config()
        camera_config = load_config(self._camera_config_path)
        processor = DLCAngleProcessor(
            runtime_config,
            self._dlc_config_path.parent,
            enable_live=True,
        )

        camera_index = int(camera_config.get("index", 0))
        backend = resolve_backend(camera_config.get("backend"))
        capture = (
            cv2.VideoCapture(camera_index, backend)
            if backend is not None
            else cv2.VideoCapture(camera_index)
        )
        if not capture.isOpened():
            capture.release()
            raise RuntimeError(f"Camera not available: index={camera_index}")
        apply_camera_settings(capture, camera_config)

        self._processor = processor
        self._capture = capture
        self._calibration = setup_undistortion_from_config(
            camera_config,
            log_prefix="[feedback-camera]",
        )

    def warm_up(self, frame_count: int) -> None:
        if frame_count < 0:
            raise ValueError("frame_count must be non-negative")
        for frame_index in range(frame_count):
            self.read(-(frame_count - frame_index))

    def read(self, frame_index: int) -> JointAngleMeasurement:
        if self._capture is None or self._processor is None:
            raise RuntimeError("Joint-angle source is not open")

        ok, frame = self._capture.read()
        captured_at = time.monotonic()
        if not ok or frame is None:
            raise RuntimeError("Failed to read a frame from the camera")

        frame = undistort_frame(frame, self._calibration)
        result, _overlay = self._processor.process_frame(
            frame_bgr=frame,
            frame_idx=frame_index,
        )
        inference_finished_at = time.monotonic()

        raw_angles = result["angles"]
        flexion_angles = tuple(
            math.radians(180.0 - float(raw_angles.get(joint, float("nan"))))
            for joint in JOINTS
        )
        keypoints = result["keypoints"]
        likelihoods = [
            float(value.get("likelihood", 0.0))
            for value in keypoints.values()
        ]
        statuses = [str(value.get("status", "missing")) for value in keypoints.values()]
        min_likelihood = min(likelihoods, default=0.0)

        reason: str | None = None
        if not np.isfinite(np.asarray(flexion_angles, dtype=np.float64)).all():
            reason = "joint angle is not finite"
        elif any(status == "missing" for status in statuses):
            reason = "one or more DLC keypoints are missing"
        elif min_likelihood < self._min_likelihood:
            reason = (
                f"minimum DLC likelihood {min_likelihood:.3f} is below "
                f"{self._min_likelihood:.3f}"
            )

        return JointAngleMeasurement(
            frame_index=int(frame_index),
            captured_at=captured_at,
            inference_finished_at=inference_finished_at,
            flexion_angles_rad=flexion_angles,
            min_likelihood=min_likelihood,
            inference_ms=float(result.get("inference_ms") or 0.0),
            valid=reason is None,
            reason=reason,
        )

    def close(self) -> None:
        capture = self._capture
        self._capture = None
        self._processor = None
        self._calibration = None
        if capture is not None:
            capture.release()

    def _resolve_dlc_runtime_config(self) -> dict:
        config = load_config(
            self._dlc_config_path,
            required_keys=("input", "dlc", "keypoints", "output"),
        )
        runtime_config = dict(config)
        dlc_config = dict(config["dlc"])
        for key in ("third_party_path", "model_path"):
            resolved = resolve_path(dlc_config.get(key), self._dlc_config_path.parent)
            if resolved is not None:
                dlc_config[key] = str(resolved)
        runtime_config["dlc"] = dlc_config
        return runtime_config
