from __future__ import annotations

from dataclasses import dataclass

import cv2
import numpy as np


@dataclass
class MarkerCandidate:
    x: float
    y: float
    area: int
    bbox: tuple[int, int, int, int]


class WeightDisplacementProcessor:
    """Detect colored markers and estimate per-lane vertical displacement."""

    def __init__(self, processing_cfg: dict, measurement_cfg: dict, mm_per_pixel: float):
        self.mm_per_pixel = float(mm_per_pixel)
        self.num_weights = int(measurement_cfg.get("num_weights", 6))
        self.baseline_frames = max(1, int(measurement_cfg.get("baseline_frames", 30)))
        self.positive_direction = str(measurement_cfg.get("positive_direction", "up")).strip().lower()

        components_cfg = processing_cfg.get("components", {})
        self.min_area = int(components_cfg.get("min_area", 80))
        self.max_area = int(components_cfg.get("max_area", 0))

        morphology_cfg = processing_cfg.get("morphology", {})
        self.kernel_size = max(1, int(morphology_cfg.get("kernel_size", 5)))
        if self.kernel_size % 2 == 0:
            self.kernel_size += 1
        self.open_iterations = max(0, int(morphology_cfg.get("open_iterations", 1)))
        self.close_iterations = max(0, int(morphology_cfg.get("close_iterations", 1)))

        color_cfg = processing_cfg.get("color", {})
        self.color_mode = str(color_cfg.get("mode", "target_tolerance"))
        self.lab_target = color_cfg.get("lab_target", {"l": 150, "a": 170, "b": 150})
        self.lab_tolerance = color_cfg.get("lab_tolerance", {"l": 60, "ab": 20})
        self.lab_range = color_cfg.get("lab_range", {})

        tracking_cfg = processing_cfg.get("tracking", {})
        self.hold_last_value_on_missing = bool(tracking_cfg.get("hold_last_value_on_missing", True))

        lanes_cfg = processing_cfg.get("lanes", {})
        self.lane_x_ranges = lanes_cfg.get("x_ranges", [])
        self.lane_y_range = lanes_cfg.get("y_range", [])

        self._states = [
            {
                "baseline_samples": [],
                "baseline_y": None,
                "last_y": None,
            }
            for _ in range(self.num_weights)
        ]

    @staticmethod
    def _clip_i32(val: int, low: int, high: int) -> int:
        return max(low, min(high, int(val)))

    def _resolve_lanes(self, frame_width: int) -> list[tuple[int, int]]:
        if isinstance(self.lane_x_ranges, list) and len(self.lane_x_ranges) == self.num_weights:
            lanes = []
            for pair in self.lane_x_ranges:
                if not isinstance(pair, (list, tuple)) or len(pair) != 2:
                    raise ValueError("Each lanes.x_ranges entry must be [x_min, x_max].")
                x0 = self._clip_i32(pair[0], 0, frame_width - 1)
                x1 = self._clip_i32(pair[1], x0 + 1, frame_width)
                lanes.append((x0, x1))
            return lanes

        lane_w = frame_width / float(self.num_weights)
        lanes = []
        for i in range(self.num_weights):
            x0 = int(round(i * lane_w))
            x1 = int(round((i + 1) * lane_w))
            lanes.append((self._clip_i32(x0, 0, frame_width - 1), self._clip_i32(x1, x0 + 1, frame_width)))
        return lanes

    def _build_mask(self, frame_bgr: np.ndarray) -> np.ndarray:
        lab = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2Lab)
        l_ch, a_ch, b_ch = cv2.split(lab)

        if self.color_mode == "range":
            l_min, l_max = self.lab_range.get("l", [0, 255])
            a_min, a_max = self.lab_range.get("a", [0, 255])
            b_min, b_max = self.lab_range.get("b", [0, 255])
            lower = np.array([int(l_min), int(a_min), int(b_min)], dtype=np.uint8)
            upper = np.array([int(l_max), int(a_max), int(b_max)], dtype=np.uint8)
            mask = cv2.inRange(lab, lower, upper)
        else:
            l0 = int(self.lab_target.get("l", 150))
            a0 = int(self.lab_target.get("a", 170))
            b0 = int(self.lab_target.get("b", 150))
            tol_l = int(self.lab_tolerance.get("l", 60))
            tol_ab = int(self.lab_tolerance.get("ab", 20))

            l_diff = np.abs(l_ch.astype(np.int16) - l0)
            da = a_ch.astype(np.int16) - a0
            db = b_ch.astype(np.int16) - b0
            ab2 = da * da + db * db

            mask_l = l_diff <= max(0, tol_l)
            mask_ab = ab2 <= max(0, tol_ab) * max(0, tol_ab)
            mask = np.where(mask_l & mask_ab, 255, 0).astype(np.uint8)

        kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (self.kernel_size, self.kernel_size))
        if self.open_iterations > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_OPEN, kernel, iterations=self.open_iterations)
        if self.close_iterations > 0:
            mask = cv2.morphologyEx(mask, cv2.MORPH_CLOSE, kernel, iterations=self.close_iterations)
        return mask

    def _extract_candidates(self, mask: np.ndarray) -> list[MarkerCandidate]:
        num_labels, _labels, stats, centroids = cv2.connectedComponentsWithStats(mask)
        candidates: list[MarkerCandidate] = []
        for i in range(1, num_labels):
            area = int(stats[i, cv2.CC_STAT_AREA])
            if area < self.min_area:
                continue
            if self.max_area > 0 and area > self.max_area:
                continue

            x = float(centroids[i][0])
            y = float(centroids[i][1])
            bbox = (
                int(stats[i, cv2.CC_STAT_LEFT]),
                int(stats[i, cv2.CC_STAT_TOP]),
                int(stats[i, cv2.CC_STAT_WIDTH]),
                int(stats[i, cv2.CC_STAT_HEIGHT]),
            )
            candidates.append(MarkerCandidate(x=x, y=y, area=area, bbox=bbox))
        return candidates

    def _candidate_in_lane(self, cand: MarkerCandidate, lane: tuple[int, int], frame_height: int) -> bool:
        x0, x1 = lane
        if not (x0 <= cand.x < x1):
            return False
        if isinstance(self.lane_y_range, (list, tuple)) and len(self.lane_y_range) == 2:
            y0 = self._clip_i32(self.lane_y_range[0], 0, frame_height - 1)
            y1 = self._clip_i32(self.lane_y_range[1], y0 + 1, frame_height)
            return y0 <= cand.y < y1
        return True

    def _pick_candidate(self, lane_idx: int, lane_candidates: list[MarkerCandidate]) -> MarkerCandidate | None:
        if not lane_candidates:
            return None
        state = self._states[lane_idx]
        last_y = state["last_y"]
        if last_y is None:
            return max(lane_candidates, key=lambda c: c.area)
        return min(lane_candidates, key=lambda c: abs(c.y - float(last_y)))

    def _update_state(self, lane_idx: int, candidate: MarkerCandidate | None) -> tuple[float | None, float | None, str]:
        state = self._states[lane_idx]
        y_value: float | None = None
        quality = "missing"

        if candidate is not None:
            y_value = float(candidate.y)
            state["last_y"] = y_value
            quality = "warming_up"
        elif self.hold_last_value_on_missing and state["last_y"] is not None:
            y_value = float(state["last_y"])
            quality = "held_last"

        if candidate is not None and state["baseline_y"] is None:
            samples: list[float] = state["baseline_samples"]
            samples.append(float(candidate.y))
            if len(samples) >= self.baseline_frames:
                state["baseline_y"] = float(sum(samples) / len(samples))

        baseline_y = state["baseline_y"]
        disp_px: float | None = None
        if baseline_y is not None and y_value is not None:
            delta = float(y_value - baseline_y)
            if self.positive_direction == "up":
                delta = -delta
            disp_px = delta
            if quality == "warming_up":
                quality = "ok"

        return y_value, disp_px, quality

    def process_frame(self, frame_bgr: np.ndarray, frame_idx: int) -> tuple[list[dict], np.ndarray]:
        h, w = frame_bgr.shape[:2]
        lanes = self._resolve_lanes(w)
        mask = self._build_mask(frame_bgr)
        candidates = self._extract_candidates(mask)

        results: list[dict] = []
        for lane_idx, lane in enumerate(lanes):
            lane_candidates = [c for c in candidates if self._candidate_in_lane(c, lane, h)]
            chosen = self._pick_candidate(lane_idx, lane_candidates)
            y_value, disp_px, quality = self._update_state(lane_idx, chosen)

            baseline_y = self._states[lane_idx]["baseline_y"]
            results.append(
                {
                    "weight_id": lane_idx,
                    "detected": chosen is not None,
                    "x_px": None if chosen is None else float(chosen.x),
                    "y_px": y_value,
                    "baseline_y_px": None if baseline_y is None else float(baseline_y),
                    "disp_px": disp_px,
                    "disp_mm": None if disp_px is None else float(disp_px * self.mm_per_pixel),
                    "quality": quality,
                    "frame_idx": frame_idx,
                }
            )

        overlay = self._draw_overlay(frame_bgr, mask, lanes, results)
        return results, overlay

    def _draw_overlay(
        self,
        frame_bgr: np.ndarray,
        mask: np.ndarray,
        lanes: list[tuple[int, int]],
        results: list[dict],
    ) -> np.ndarray:
        out = frame_bgr.copy()
        h, w = out.shape[:2]

        for lane_idx, lane in enumerate(lanes):
            x0, x1 = lane
            cv2.rectangle(out, (x0, 0), (x1, h - 1), (80, 80, 80), 1)
            res = results[lane_idx]

            if res["y_px"] is not None:
                color = (0, 255, 0) if res["quality"] in {"ok", "warming_up"} else (0, 180, 255)
                x_draw = int((x0 + x1) / 2) if res["x_px"] is None else int(res["x_px"])
                y_draw = int(res["y_px"])
                cv2.circle(out, (x_draw, y_draw), 6, color, -1)

            label = f"id={lane_idx}"
            if res["disp_mm"] is not None:
                label += f" {res['disp_mm']:+.2f} mm"
            else:
                label += " --"
            cv2.putText(
                out,
                label,
                (x0 + 6, 24),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.5,
                (255, 255, 0),
                1,
                cv2.LINE_AA,
            )

        # Small binary-mask preview in the corner.
        mask_bgr = cv2.cvtColor(mask, cv2.COLOR_GRAY2BGR)
        preview_h = max(60, int(h * 0.2))
        preview_w = max(80, int(w * 0.2))
        preview = cv2.resize(mask_bgr, (preview_w, preview_h), interpolation=cv2.INTER_NEAREST)
        out[0:preview_h, 0:preview_w] = preview

        return out
