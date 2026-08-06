"""Synchronous camera and DLC joint-angle acquisition."""

from __future__ import annotations

import math
import time
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
from observation.vision.overlay_video_recorder import (
    OverlayVideoRecorder,
    OverlayVideoSummary,
)
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


@dataclass(frozen=True)
class DominantKeypointCluster:
    position: tuple[float, float]
    likelihood: float
    member_indices: tuple[int, ...]
    ratio: float
    dispersion_px: float


@dataclass(frozen=True)
class Meta1CalibrationResult:
    position: tuple[float, float]
    likelihood: float
    frames_examined: int
    valid_samples: int
    dominant_samples: int
    dominant_ratio: float
    dispersion_px: float


class Meta1CalibrationAbortedError(RuntimeError):
    """Raised when the operator aborts static meta1 calibration."""


def select_dominant_keypoint_cluster(
    candidates: list[tuple[float, float, float]],
    *,
    radius_px: float,
    min_cluster_samples: int,
    min_cluster_ratio: float,
) -> DominantKeypointCluster:
    """Select the densest spatial cluster and return its coordinate median."""
    if not math.isfinite(radius_px) or radius_px <= 0.0:
        raise ValueError("radius_px must be finite and greater than zero")
    if min_cluster_samples <= 0:
        raise ValueError("min_cluster_samples must be greater than zero")
    if not math.isfinite(min_cluster_ratio) or not 0.0 < min_cluster_ratio <= 1.0:
        raise ValueError("min_cluster_ratio must be in the range (0, 1]")
    if not candidates:
        raise ValueError("No valid keypoint candidates were collected")

    values = np.asarray(candidates, dtype=np.float64)
    if values.ndim != 2 or values.shape[1] != 3 or not np.isfinite(values).all():
        raise ValueError("Each candidate must contain finite x, y, and likelihood values")
    if np.any((values[:, 2] < 0.0) | (values[:, 2] > 1.0)):
        raise ValueError("Candidate likelihoods must be in the range 0-1")

    points = values[:, :2]
    pairwise_distances = np.linalg.norm(points[:, None, :] - points[None, :, :], axis=2)
    neighbour_counts = np.count_nonzero(pairwise_distances <= radius_px, axis=1)
    anchor_index = int(np.argmax(neighbour_counts))
    initial_members = np.flatnonzero(pairwise_distances[anchor_index] <= radius_px)
    center = np.median(points[initial_members], axis=0)
    member_indices = np.flatnonzero(np.linalg.norm(points - center, axis=1) <= radius_px)
    center = np.median(points[member_indices], axis=0)
    member_distances = np.linalg.norm(points[member_indices] - center, axis=1)

    member_count = int(member_indices.size)
    ratio = member_count / len(candidates)
    if member_count < min_cluster_samples or ratio < min_cluster_ratio:
        raise ValueError(
            f"Dominant cluster is too small: {member_count}/{len(candidates)} "
            f"samples ({ratio:.0%})"
        )

    return DominantKeypointCluster(
        position=(float(center[0]), float(center[1])),
        likelihood=float(np.median(values[member_indices, 2])),
        member_indices=tuple(int(index) for index in member_indices),
        ratio=float(ratio),
        dispersion_px=float(np.max(member_distances, initial=0.0)),
    )


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
        self._runtime_config: dict | None = None
        self._overlay_recorder: OverlayVideoRecorder | None = None

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
        self._runtime_config = runtime_config
        self._calibration = setup_undistortion_from_config(
            camera_config,
            log_prefix="[feedback-camera]",
        )

    def calibrate_meta1(self) -> Meta1CalibrationResult | None:
        """Optionally determine and latch meta1 before feedback control starts."""
        if self._capture is None or self._processor is None or self._runtime_config is None:
            raise RuntimeError("Joint-angle source is not open")

        raw_settings = self._runtime_config.get("static_keypoints", {}).get("meta1")
        if not isinstance(raw_settings, dict) or not bool(raw_settings.get("enabled", False)):
            return None

        required_samples = int(raw_settings.get("required_valid_samples", 10))
        max_frames = int(raw_settings.get("max_frames", 30))
        min_likelihood = float(raw_settings.get("min_likelihood", self._min_likelihood))
        cluster_radius_px = float(raw_settings.get("cluster_radius_px", 10.0))
        min_cluster_samples = int(raw_settings.get("min_cluster_samples", 6))
        min_cluster_ratio = float(raw_settings.get("min_cluster_ratio", 0.6))
        if required_samples <= 0:
            raise ValueError("static_keypoints.meta1.required_valid_samples must be positive")
        if max_frames < required_samples:
            raise ValueError("static_keypoints.meta1.max_frames must be at least required_valid_samples")
        if not math.isfinite(min_likelihood) or not 0.0 <= min_likelihood <= 1.0:
            raise ValueError("static_keypoints.meta1.min_likelihood must be in the range 0-1")

        window_name = "Static meta1 calibration"
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)
        try:
            while True:
                candidates: list[tuple[float, float, float]] = []
                last_overlay: np.ndarray | None = None
                frames_examined = 0

                for frame_offset in range(max_frames):
                    ok, frame = self._capture.read()
                    if not ok or frame is None:
                        raise RuntimeError("Failed to read a frame during meta1 calibration")
                    frames_examined = frame_offset + 1
                    frame = undistort_frame(frame, self._calibration)
                    result, overlay = self._processor.process_frame(
                        frame_bgr=frame,
                        frame_idx=-(max_frames + frame_offset + 1),
                    )
                    meta1 = result["keypoints"].get("meta1", {})
                    x_value = meta1.get("x")
                    y_value = meta1.get("y")
                    likelihood = float(meta1.get("likelihood", 0.0))
                    if (
                        meta1.get("status") == "detected"
                        and x_value is not None
                        and y_value is not None
                        and math.isfinite(float(x_value))
                        and math.isfinite(float(y_value))
                        and math.isfinite(likelihood)
                        and likelihood >= min_likelihood
                    ):
                        candidates.append((float(x_value), float(y_value), likelihood))

                    self._draw_meta1_candidates(overlay, candidates)
                    self._draw_status_lines(
                        overlay,
                        (
                            "Collecting static meta1 candidates",
                            f"valid: {len(candidates)}/{required_samples}  frame: {frames_examined}/{max_frames}",
                            "ESC: abort",
                        ),
                    )
                    cv2.imshow(window_name, overlay)
                    last_overlay = overlay
                    key = cv2.waitKeyEx(1) & 0xFF
                    if key == 27 or self._window_was_closed(window_name):
                        raise Meta1CalibrationAbortedError("Static meta1 calibration was aborted")
                    if len(candidates) >= required_samples:
                        break

                cluster: DominantKeypointCluster | None = None
                failure_reason: str | None = None
                if len(candidates) < required_samples:
                    failure_reason = (
                        f"Only {len(candidates)}/{required_samples} valid candidates "
                        f"were found in {max_frames} frames"
                    )
                else:
                    try:
                        cluster = select_dominant_keypoint_cluster(
                            candidates,
                            radius_px=cluster_radius_px,
                            min_cluster_samples=min_cluster_samples,
                            min_cluster_ratio=min_cluster_ratio,
                        )
                    except ValueError as exc:
                        failure_reason = str(exc)

                if last_overlay is None:
                    raise RuntimeError("No camera frame was available for meta1 calibration")
                confirmation = last_overlay.copy()
                self._draw_meta1_candidates(
                    confirmation,
                    candidates,
                    cluster.member_indices if cluster is not None else (),
                )
                if cluster is not None:
                    print(
                        "[meta1-calibration] calculated: "
                        f"position=({cluster.position[0]:.1f}, {cluster.position[1]:.1f}), "
                        f"likelihood={cluster.likelihood:.3f}, "
                        f"cluster={len(cluster.member_indices)}/{len(candidates)}, "
                        f"dispersion={cluster.dispersion_px:.1f}px",
                        flush=True,
                    )
                    selected = (int(round(cluster.position[0])), int(round(cluster.position[1])))
                    cv2.circle(confirmation, selected, 18, (255, 255, 0), 4, cv2.LINE_AA)
                    cv2.drawMarker(
                        confirmation,
                        selected,
                        (255, 255, 0),
                        cv2.MARKER_CROSS,
                        44,
                        4,
                        cv2.LINE_AA,
                    )
                    lines = (
                        f"Selected meta1: ({cluster.position[0]:.1f}, {cluster.position[1]:.1f})",
                        f"cluster: {len(cluster.member_indices)}/{len(candidates)}  dispersion: {cluster.dispersion_px:.1f}px",
                        "ENTER: accept  ESC: abort",
                    )
                else:
                    lines = (
                        "Static meta1 calibration failed",
                        failure_reason or "No dominant cluster was found",
                        "ENTER: retry  ESC: abort",
                    )
                self._draw_status_lines(confirmation, lines)
                cv2.imshow(window_name, confirmation)

                while True:
                    key = cv2.waitKeyEx(50) & 0xFF
                    if key == 27 or self._window_was_closed(window_name):
                        raise Meta1CalibrationAbortedError("Static meta1 calibration was aborted")
                    if key in (10, 13):
                        break

                if cluster is None:
                    continue

                self._processor.set_latched_keypoint(
                    "meta1",
                    cluster.position,
                    cluster.likelihood,
                )
                return Meta1CalibrationResult(
                    position=cluster.position,
                    likelihood=cluster.likelihood,
                    frames_examined=frames_examined,
                    valid_samples=len(candidates),
                    dominant_samples=len(cluster.member_indices),
                    dominant_ratio=cluster.ratio,
                    dispersion_px=cluster.dispersion_px,
                )
        finally:
            try:
                cv2.destroyWindow(window_name)
            except cv2.error:
                pass

    @staticmethod
    def _draw_meta1_candidates(
        frame: np.ndarray,
        candidates: list[tuple[float, float, float]],
        dominant_indices: tuple[int, ...] = (),
    ) -> None:
        dominant = set(dominant_indices)
        for index, (x_value, y_value, _likelihood) in enumerate(candidates):
            color = (0, 255, 255) if index in dominant else (180, 180, 180)
            cv2.circle(
                frame,
                (int(round(x_value)), int(round(y_value))),
                4,
                color,
                1,
                cv2.LINE_AA,
            )

    @staticmethod
    def _draw_status_lines(frame: np.ndarray, lines: tuple[str, ...]) -> None:
        for index, line in enumerate(lines):
            cv2.putText(
                frame,
                line,
                (20, 30 + index * 26),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.65,
                (255, 255, 255),
                2,
                cv2.LINE_AA,
            )

    @staticmethod
    def _window_was_closed(window_name: str) -> bool:
        try:
            return cv2.getWindowProperty(window_name, cv2.WND_PROP_VISIBLE) < 1.0
        except cv2.error:
            return True

    def warm_up(self, frame_count: int) -> None:
        if frame_count < 0:
            raise ValueError("frame_count must be non-negative")
        for frame_index in range(frame_count):
            self.read(-(frame_count - frame_index))

    def start_overlay_recording(
        self,
        path: Path,
        *,
        fps: float,
        codec: str = "mp4v",
        queue_size: int = 10,
    ) -> Path:
        if self._capture is None or self._processor is None:
            raise RuntimeError("Joint-angle source is not open")
        if self._overlay_recorder is not None:
            raise RuntimeError("Overlay video recording is already active")
        recorder = OverlayVideoRecorder(
            path,
            fps=fps,
            codec=codec,
            queue_size=queue_size,
        )
        recorder.start()
        self._overlay_recorder = recorder
        return recorder.path

    def stop_overlay_recording(self) -> OverlayVideoSummary | None:
        recorder = self._overlay_recorder
        self._overlay_recorder = None
        if recorder is None:
            return None
        return recorder.stop()

    def read(self, frame_index: int) -> JointAngleMeasurement:
        if self._capture is None or self._processor is None:
            raise RuntimeError("Joint-angle source is not open")

        ok, frame = self._capture.read()
        captured_at = time.monotonic()
        if not ok or frame is None:
            raise RuntimeError("Failed to read a frame from the camera")

        frame = undistort_frame(frame, self._calibration)
        result, overlay = self._processor.process_frame(
            frame_bgr=frame,
            frame_idx=frame_index,
        )
        if self._overlay_recorder is not None:
            self._overlay_recorder.write(overlay)
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
        try:
            self.stop_overlay_recording()
        finally:
            capture = self._capture
            self._capture = None
            self._processor = None
            self._runtime_config = None
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
