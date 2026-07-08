from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class ArucoMarker:
    marker_id: int
    x: float
    y: float
    side_px: float
    corners: np.ndarray


class WeightDisplacementProcessor:
    """Detect ArUco markers and estimate per-marker vertical displacement."""

    def __init__(self, processing_cfg: dict, measurement_cfg: dict, mm_per_pixel: float):
        self.mm_per_pixel_default = float(mm_per_pixel)
        self.num_weights = int(measurement_cfg.get("num_weights", 6))
        self.baseline_frames = max(1, int(measurement_cfg.get("baseline_frames", 3)))
        self.positive_direction = str(measurement_cfg.get("positive_direction", "up")).strip().lower()
        self.marker_ids = [int(i) for i in measurement_cfg.get("marker_ids", list(range(self.num_weights)))]
        if len(self.marker_ids) != self.num_weights:
            raise ValueError("measurement.marker_ids length must match measurement.num_weights.")

        aruco_cfg = processing_cfg.get("aruco", {})
        self.dictionary_name = str(aruco_cfg.get("dictionary", "DICT_4X4_50"))
        self.marker_length_mm = float(aruco_cfg.get("marker_length_mm", 18.0))

        aruco = getattr(cv2, "aruco", None)
        if aruco is None:
            raise RuntimeError(
                "cv2.aruco is not available. Install a build that includes ArUco (e.g. opencv-contrib-python)."
            )
        if not hasattr(aruco, self.dictionary_name):
            raise ValueError(f"Unsupported ArUco dictionary: {self.dictionary_name}")
        self._aruco = aruco
        self._dictionary = aruco.getPredefinedDictionary(getattr(aruco, self.dictionary_name))
        if hasattr(aruco, "ArucoDetector") and hasattr(aruco, "DetectorParameters"):
            self._detector = aruco.ArucoDetector(self._dictionary, aruco.DetectorParameters())
        else:
            self._detector = None
            self._detector_params = aruco.DetectorParameters_create()

        self._states = {
            marker_id: {
                "baseline_samples": [],
                "scale_samples": [],
                "baseline_y": None,
                "mm_per_px": None,
            }
            for marker_id in self.marker_ids
        }

    @staticmethod
    def _edge_mean_px(corners: np.ndarray) -> float:
        pts = corners.reshape(4, 2)
        edge_lengths = []
        for i in range(4):
            p0 = pts[i]
            p1 = pts[(i + 1) % 4]
            dx = float(p0[0] - p1[0])
            dy = float(p0[1] - p1[1])
            edge_lengths.append((dx * dx + dy * dy) ** 0.5)
        return float(sum(edge_lengths) / 4.0)

    def _detect_markers(self, frame_bgr: np.ndarray) -> dict[int, ArucoMarker]:
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        if self._detector is not None:
            corners, ids, _ = self._detector.detectMarkers(gray)
        else:
            corners, ids, _ = self._aruco.detectMarkers(gray, self._dictionary, parameters=self._detector_params)

        detected: dict[int, ArucoMarker] = {}
        if ids is None or len(ids) == 0:
            return detected

        for idx, marker_id_raw in enumerate(ids.flatten().tolist()):
            marker_id = int(marker_id_raw)
            if marker_id not in self._states:
                continue
            marker_corners = corners[idx].reshape(4, 2)
            center = marker_corners.mean(axis=0)
            side_px = self._edge_mean_px(marker_corners)
            detected[marker_id] = ArucoMarker(
                marker_id=marker_id,
                x=float(center[0]),
                y=float(center[1]),
                side_px=float(side_px),
                corners=marker_corners,
            )
        return detected

    def _update_state(self, marker_id: int, marker: ArucoMarker) -> dict:
        state = self._states[marker_id]
        y_value = float(marker.y)
        mm_per_px = state["mm_per_px"]

        quality = "warming_up"
        if state["baseline_y"] is None:
            baseline_samples: list[float] = state["baseline_samples"]
            scale_samples: list[float] = state["scale_samples"]
            baseline_samples.append(y_value)
            if marker.side_px > 0:
                scale_samples.append(float(self.marker_length_mm / marker.side_px))

            if len(baseline_samples) >= self.baseline_frames:
                state["baseline_y"] = float(sum(baseline_samples) / len(baseline_samples))
                if scale_samples:
                    state["mm_per_px"] = float(sum(scale_samples) / len(scale_samples))
                else:
                    state["mm_per_px"] = float(self.mm_per_pixel_default)
                mm_per_px = state["mm_per_px"]
            else:
                mm_per_px = None
        else:
            quality = "ok"

        baseline_y = state["baseline_y"]
        disp_px: float | None = 0.0
        disp_mm: float | None = 0.0

        if baseline_y is not None and mm_per_px is not None:
            delta = float(y_value - baseline_y)
            if self.positive_direction == "up":
                delta = -delta
            disp_px = delta
            disp_mm = float(delta * mm_per_px)
            quality = "ok"
        else:
            quality = "warming_up"

        return {
            "weight_id": marker_id,
            "marker_id": marker_id,
            "detected": True,
            "x_px": float(marker.x),
            "y_px": y_value,
            "baseline_y_px": None if baseline_y is None else float(baseline_y),
            "disp_px": disp_px,
            "disp_mm": disp_mm,
            "mm_per_px": mm_per_px,
            "quality": quality,
        }

    def _missing_row(self, marker_id: int) -> dict:
        state = self._states[marker_id]
        baseline_y = state["baseline_y"]
        mm_per_px = state["mm_per_px"]
        quality = "missing"
        if baseline_y is None or mm_per_px is None:
            quality = "missing_warming_up"
        return {
            "weight_id": marker_id,
            "marker_id": marker_id,
            "detected": False,
            "x_px": None,
            "y_px": None,
            "baseline_y_px": None if baseline_y is None else float(baseline_y),
            "disp_px": None,
            "disp_mm": None,
            "mm_per_px": None if mm_per_px is None else float(mm_per_px),
            "quality": quality,
        }

    def process_frame(self, frame_bgr: np.ndarray, frame_idx: int) -> tuple[list[dict], np.ndarray]:
        detected = self._detect_markers(frame_bgr)
        results: list[dict] = []
        for marker_id in self.marker_ids:
            marker = detected.get(marker_id)
            if marker is None:
                row = self._missing_row(marker_id)
            else:
                row = self._update_state(marker_id, marker)
            row["frame_idx"] = frame_idx
            results.append(row)

        overlay = self._draw_overlay(frame_bgr, detected, results)
        return results, overlay

    def _draw_overlay(
        self,
        frame_bgr: np.ndarray,
        detected: dict[int, ArucoMarker],
        results: list[dict],
    ) -> np.ndarray:
        out = frame_bgr.copy()
        result_map = {int(row["marker_id"]): row for row in results}
        missing_ids: list[int] = []
        for marker_id in self.marker_ids:
            marker = detected.get(marker_id)
            if marker is None:
                missing_ids.append(marker_id)
                continue
            res = result_map.get(marker_id, {})
            quality = str(res.get("quality", "warming_up"))
            color = (0, 255, 0) if quality == "ok" else (0, 180, 255)

            corners_i32 = marker.corners.astype(np.int32).reshape((-1, 1, 2))
            cv2.polylines(out, [corners_i32], isClosed=True, color=color, thickness=2)
            cv2.circle(out, (int(marker.x), int(marker.y)), 4, color, -1)

            label = f"id={marker_id}"
            disp_mm = res.get("disp_mm")
            if disp_mm is not None:
                label += f" {float(disp_mm):+.2f} mm"
            else:
                label += " --"
            cv2.putText(
                out,
                label,
                (int(marker.x) + 6, int(marker.y) - 6),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

        if missing_ids:
            text = "missing ids: " + ", ".join(str(marker_id) for marker_id in missing_ids)
            cv2.putText(
                out,
                text,
                (12, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.6,
                (0, 0, 255),
                2,
                cv2.LINE_AA,
            )

        return out
