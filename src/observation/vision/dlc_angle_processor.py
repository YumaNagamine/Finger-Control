from __future__ import annotations

import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

from utils.path_utils import resolve_path


@dataclass
class AngleDefinition:
    name: str
    segment_a: str
    segment_b: str


class _DLCLivePredictor:
    """Thin runtime wrapper around dlclive for per-frame keypoint inference."""

    def __init__(self, dlc_cfg: dict, keypoint_names: list[str], config_base_dir: Path):
        self._keypoint_names = keypoint_names
        self._convert_to_rgb = bool(dlc_cfg.get("convert_bgr_to_rgb", True))
        self._default_conf = float(dlc_cfg.get("default_likelihood", 0.0))
        self._initialized = False
        self._dlc = None

        third_party_path = resolve_path(dlc_cfg.get("third_party_path"), config_base_dir)
        if third_party_path is not None and third_party_path.exists():
            sys.path.insert(0, str(third_party_path))

        model_path = resolve_path(dlc_cfg.get("model_path"), config_base_dir)
        if model_path is None:
            raise ValueError("dlc.model_path must be provided in config.")
        if not model_path.exists():
            raise FileNotFoundError(f"DLC model path not found: {model_path}")

        try:
            from dlclive import DLCLive
        except Exception as exc:  # pragma: no cover
            raise RuntimeError(
                "Failed to import dlclive. Install dlclive (and deeplabcut runtime deps), "
                "or update dlc.third_party_path / PYTHONPATH."
            ) from exc

        try:
            self._dlc = DLCLive(str(model_path), pcutoff=float(dlc_cfg.get("pcutoff", 0.0)))
        except TypeError:
            self._dlc = DLCLive(str(model_path))

    def reset(self) -> None:
        self._initialized = False

    def infer(self, frame_bgr: np.ndarray) -> dict[str, tuple[float, float, float]]:
        if self._dlc is None:
            raise RuntimeError("DLC predictor is not initialized.")

        frame_input = frame_bgr
        if self._convert_to_rgb:
            frame_input = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)

        if not self._initialized:
            self._dlc.init_inference(frame_input)
            self._initialized = True

        pose = self._dlc.get_pose(frame_input)
        pose_arr = np.asarray(pose, dtype=np.float32)

        if pose_arr.ndim == 3:
            pose_arr = pose_arr[0]
        if pose_arr.ndim != 2:
            raise RuntimeError(f"Unexpected pose shape from dlclive: {pose_arr.shape}")
        if pose_arr.shape[0] < len(self._keypoint_names):
            raise RuntimeError(
                f"Pose count mismatch: expected >= {len(self._keypoint_names)}, got {pose_arr.shape[0]}"
            )

        if pose_arr.shape[1] == 2:
            conf = np.full((pose_arr.shape[0],), self._default_conf, dtype=np.float32)
        elif pose_arr.shape[1] >= 3:
            conf = pose_arr[:, 2]
        else:
            raise RuntimeError(f"Unexpected pose vector length: {pose_arr.shape[1]}")

        output: dict[str, tuple[float, float, float]] = {}
        for idx, name in enumerate(self._keypoint_names):
            x_val = float(pose_arr[idx, 0])
            y_val = float(pose_arr[idx, 1])
            c_val = float(conf[idx])
            output[name] = (x_val, y_val, c_val)
        return output


class DLCAngleProcessor:
    """
    DLC keypoint inference + angle estimation pipeline.

    Angles follow the same formulation as AngleProcessor._calculate_angle:
    angle_0 = angle(distal, medial)
    angle_1 = angle(medial, proximal)
    angle_2 = angle(proximal, palm)
    """

    def __init__(self, config: dict, config_base_dir: Path):
        self._config = config
        keypoint_cfg = config["keypoints"]
        processing_cfg = config.get("processing", {})
        draw_cfg = config.get("draw", {})

        self.keypoint_names = [str(name) for name in keypoint_cfg.get("names", [])]
        if not self.keypoint_names:
            raise ValueError("keypoints.names must contain at least one keypoint.")

        self._segments = self._parse_segments(keypoint_cfg.get("segments", {}))
        self._angles = self._parse_angles(config.get("angles", []))

        self._confidence_threshold = float(processing_cfg.get("confidence_threshold", 0.6))
        self._ema_alpha = float(processing_cfg.get("ema_alpha", 0.4))
        self._hold_last_frames = max(0, int(processing_cfg.get("hold_last_frames", 4)))

        self._show_keypoint_labels = bool(draw_cfg.get("show_keypoint_labels", True))
        self._keypoint_radius = max(1, int(draw_cfg.get("keypoint_radius", 4)))
        self._line_thickness = max(1, int(draw_cfg.get("line_thickness", 2)))

        self._last_valid: dict[str, np.ndarray] = {}
        self._ema_state: dict[str, np.ndarray] = {}
        self._missing_count: dict[str, int] = {name: 0 for name in self.keypoint_names}

        self._predictor = _DLCLivePredictor(config["dlc"], self.keypoint_names, config_base_dir)

    @staticmethod
    def _parse_segments(raw_segments: dict[str, Any]) -> dict[str, tuple[str, str]]:
        segments: dict[str, tuple[str, str]] = {}
        for segment_name, segment_value in raw_segments.items():
            if isinstance(segment_value, dict):
                start = str(segment_value.get("start", ""))
                end = str(segment_value.get("end", ""))
            elif isinstance(segment_value, (list, tuple)) and len(segment_value) == 2:
                start, end = str(segment_value[0]), str(segment_value[1])
            else:
                raise ValueError(
                    f"Invalid segment '{segment_name}'. Expected {{start,end}} or [start,end], got {segment_value!r}"
                )
            if not start or not end:
                raise ValueError(f"Segment '{segment_name}' must define both start and end keypoints.")
            segments[str(segment_name)] = (start, end)
        return segments

    @staticmethod
    def _parse_angles(raw_angles: list[dict[str, Any]]) -> list[AngleDefinition]:
        if not raw_angles:
            return [
                AngleDefinition(name="angle_0", segment_a="distal", segment_b="medial"),
                AngleDefinition(name="angle_1", segment_a="medial", segment_b="proximal"),
                AngleDefinition(name="angle_2", segment_a="proximal", segment_b="palm"),
            ]
        definitions: list[AngleDefinition] = []
        for angle in raw_angles:
            definitions.append(
                AngleDefinition(
                    name=str(angle["name"]),
                    segment_a=str(angle["segment_a"]),
                    segment_b=str(angle["segment_b"]),
                )
            )
        return definitions

    @property
    def angle_names(self) -> list[str]:
        return [angle_def.name for angle_def in self._angles]

    def reset(self) -> None:
        self._last_valid.clear()
        self._ema_state.clear()
        self._missing_count = {name: 0 for name in self.keypoint_names}
        self._predictor.reset()

    @staticmethod
    def _calculate_angle(segment_a: list[tuple[float, float]], segment_b: list[tuple[float, float]]) -> float:
        try:
            distalis = np.array(segment_a, dtype=np.float32)
            proximal = np.array(segment_b, dtype=np.float32)
            distalis_vec = distalis[0] - distalis[1]
            proximal_vec = proximal[0] - proximal[1]

            dot_product = float(np.dot(proximal_vec, distalis_vec))
            cross_product = float(np.cross(proximal_vec, distalis_vec))
            norm_distalis = float(np.linalg.norm(distalis_vec))
            norm_proximal = float(np.linalg.norm(proximal_vec))
            if norm_distalis == 0.0 or norm_proximal == 0.0:
                return float("nan")
            angle_rad = float(np.arctan2(cross_product, dot_product))
            angle_degree = float(np.degrees(angle_rad))
            if angle_rad < 0:
                return abs(angle_degree) + 180.0
            return 180.0 - angle_degree
        except Exception:
            return float("nan")

    def _track_keypoints(
        self,
        raw_keypoints: dict[str, tuple[float, float, float]],
    ) -> dict[str, dict[str, float | str | None]]:
        tracked: dict[str, dict[str, float | str | None]] = {}
        for name in self.keypoint_names:
            x_val, y_val, conf = raw_keypoints[name]
            valid = np.isfinite(x_val) and np.isfinite(y_val) and conf >= self._confidence_threshold

            if valid:
                point = np.array([x_val, y_val], dtype=np.float32)
                if 0.0 < self._ema_alpha < 1.0 and name in self._ema_state:
                    point = (self._ema_alpha * point) + ((1.0 - self._ema_alpha) * self._ema_state[name])
                self._ema_state[name] = point
                self._last_valid[name] = point
                self._missing_count[name] = 0
                status = "detected"
                tracked[name] = {
                    "x": float(point[0]),
                    "y": float(point[1]),
                    "likelihood": float(conf),
                    "status": status,
                }
                continue

            self._missing_count[name] += 1
            if name in self._last_valid and self._missing_count[name] <= self._hold_last_frames:
                hold_point = self._last_valid[name]
                tracked[name] = {
                    "x": float(hold_point[0]),
                    "y": float(hold_point[1]),
                    "likelihood": float(conf),
                    "status": "held_last",
                }
            else:
                tracked[name] = {
                    "x": None,
                    "y": None,
                    "likelihood": float(conf),
                    "status": "missing",
                }
        return tracked

    def _segments_from_keypoints(
        self,
        tracked_keypoints: dict[str, dict[str, float | str | None]],
    ) -> dict[str, list[tuple[float, float]] | None]:
        segments: dict[str, list[tuple[float, float]] | None] = {}
        for segment_name, (start_name, end_name) in self._segments.items():
            p_start = tracked_keypoints.get(start_name)
            p_end = tracked_keypoints.get(end_name)
            if p_start is None or p_end is None:
                segments[segment_name] = None
                continue

            x0, y0 = p_start["x"], p_start["y"]
            x1, y1 = p_end["x"], p_end["y"]
            if x0 is None or y0 is None or x1 is None or y1 is None:
                segments[segment_name] = None
                continue
            segments[segment_name] = [(float(x0), float(y0)), (float(x1), float(y1))]
        return segments

    def _compute_angles(self, segments: dict[str, list[tuple[float, float]] | None]) -> dict[str, float]:
        angles: dict[str, float] = {}
        for angle_def in self._angles:
            seg_a = segments.get(angle_def.segment_a)
            seg_b = segments.get(angle_def.segment_b)
            if seg_a is None or seg_b is None:
                angles[angle_def.name] = float("nan")
                continue
            angles[angle_def.name] = self._calculate_angle(seg_a, seg_b)
        return angles

    def _draw_overlay(
        self,
        frame_bgr: np.ndarray,
        frame_idx: int,
        tracked_keypoints: dict[str, dict[str, float | str | None]],
        segments: dict[str, list[tuple[float, float]] | None],
        angles: dict[str, float],
        inference_ms: float,
    ) -> np.ndarray:
        out = frame_bgr.copy()
        segment_colors = {
            "distal": (255, 0, 0),
            "medial": (127, 0, 255),
            "proximal": (0, 127, 0),
            "palm": (0, 127, 255),
        }

        for segment_name, points in segments.items():
            if points is None:
                continue
            color = segment_colors.get(segment_name, (255, 255, 255))
            p0 = (int(points[0][0]), int(points[0][1]))
            p1 = (int(points[1][0]), int(points[1][1]))
            cv2.line(out, p0, p1, color, self._line_thickness)

        for name, point in tracked_keypoints.items():
            x_val, y_val = point["x"], point["y"]
            if x_val is None or y_val is None:
                continue
            status = str(point["status"])
            if status == "detected":
                color = (0, 255, 0)
            elif status == "held_last":
                color = (0, 180, 255)
            else:
                color = (0, 0, 255)
            center = (int(float(x_val)), int(float(y_val)))
            cv2.circle(out, center, self._keypoint_radius, color, -1)
            if self._show_keypoint_labels:
                cv2.putText(
                    out,
                    name,
                    (center[0] + 5, center[1] - 5),
                    cv2.FONT_HERSHEY_SIMPLEX,
                    0.45,
                    (255, 255, 255),
                    1,
                    cv2.LINE_AA,
                )

        cv2.putText(out, f"frame: {frame_idx}", (25, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2)
        cv2.putText(out, f"infer: {inference_ms:.1f} ms", (25, 55), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 0), 2)
        y_cursor = 85
        for angle_name in self.angle_names:
            angle_val = angles.get(angle_name, float("nan"))
            if np.isnan(angle_val):
                label = f"{angle_name}: nan"
            else:
                label = f"{angle_name}: {angle_val:.1f}"
            cv2.putText(out, label, (25, y_cursor), cv2.FONT_HERSHEY_SIMPLEX, 0.62, (255, 255, 0), 2)
            y_cursor += 25
        return out

    def process_frame(self, frame_bgr: np.ndarray, frame_idx: int) -> tuple[dict[str, Any], np.ndarray]:
        t0 = time.perf_counter()
        raw_keypoints = self._predictor.infer(frame_bgr)
        inference_ms = (time.perf_counter() - t0) * 1000.0

        tracked_keypoints = self._track_keypoints(raw_keypoints)
        segments = self._segments_from_keypoints(tracked_keypoints)
        angles = self._compute_angles(segments)
        overlay = self._draw_overlay(frame_bgr, frame_idx, tracked_keypoints, segments, angles, inference_ms)

        result = {
            "frame_idx": frame_idx,
            "keypoints": tracked_keypoints,
            "segments": segments,
            "angles": angles,
            "inference_ms": inference_ms,
        }
        return result, overlay
