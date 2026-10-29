"""Runtime evaluation of saved moment-arm polynomial models."""

from __future__ import annotations

import hashlib
import json
import math
from pathlib import Path
from typing import Callable, Sequence

import numpy as np

from controller.servo_mapping import TENDONS


JOINTS = ("DIP", "PIP", "MCP")


class MomentArmRuntime:
    def __init__(
        self,
        coefficients: dict[str, dict[str, np.ndarray]],
        *,
        evaluation_ranges_rad: dict[str, tuple[float, float]] | None = None,
        model_info: dict[str, object] | None = None,
        extrapolation_event_callback: Callable[[dict[str, object]], None] | None = None,
    ) -> None:
        self._coefficients = coefficients
        self._evaluation_ranges_rad = dict(evaluation_ranges_rad or {})
        self._model_info = dict(model_info or {})
        self._extrapolation_event_callback = extrapolation_event_callback

    @property
    def evaluation_ranges_rad(self) -> dict[str, tuple[float, float]]:
        return dict(self._evaluation_ranges_rad)

    @property
    def model_info(self) -> dict[str, object]:
        return dict(self._model_info)

    def set_extrapolation_event_callback(
        self,
        callback: Callable[[dict[str, object]], None] | None,
    ) -> None:
        """Set the session event sink used for constant extrapolation records."""

        self._extrapolation_event_callback = callback

    @classmethod
    def from_directory(cls, directory: Path) -> "MomentArmRuntime":
        root = directory.expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Moment-arm directory not found: {root}")

        coefficients: dict[str, dict[str, np.ndarray]] = {}
        evaluation_ranges_rad: dict[str, tuple[float, float]] = {}
        joint_info: dict[str, object] = {}
        for joint in JOINTS:
            path = root / f"moment_arm_{joint}_average.json"
            if not path.is_file():
                raise FileNotFoundError(f"Moment-arm JSON not found: {path}")
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            raw_by_tendon = payload.get("moment_arm_coefficients")
            if not isinstance(raw_by_tendon, dict):
                raise ValueError(f"Missing moment_arm_coefficients in {path}")

            tendon_coefficients: dict[str, np.ndarray] = {}
            for tendon in TENDONS:
                raw_values = raw_by_tendon.get(tendon)
                if not isinstance(raw_values, list) or not raw_values:
                    raise ValueError(f"Missing coefficients for {tendon} in {path}")
                values = np.asarray(raw_values, dtype=np.float64)
                if values.ndim != 1 or not np.isfinite(values).all():
                    raise ValueError(f"Invalid coefficients for {tendon} in {path}")
                tendon_coefficients[tendon] = values
            coefficients[joint] = tendon_coefficients

            raw_range, range_source = _evaluation_range_deg(payload, joint, path)
            low_deg, high_deg = raw_range
            evaluation_ranges_rad[joint] = (
                math.radians(low_deg),
                math.radians(high_deg),
            )
            joint_info[joint] = {
                "path": str(path),
                "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
                "source_id": payload.get("source_id"),
                "source_trial": payload.get("source_trial"),
                "evaluation_range_deg": raw_range,
                "evaluation_range_source": range_source,
                "fit_degree": payload.get("moment_arm_fit_degree"),
            }
        return cls(
            coefficients,
            evaluation_ranges_rad=evaluation_ranges_rad,
            model_info={"directory": str(root), "joints": joint_info},
        )

    def validate_domain(self, joint_angles_rad: Sequence[float]) -> None:
        """Validate angles and fit metadata without limiting physical angles."""

        angles = np.asarray(joint_angles_rad, dtype=np.float64)
        if angles.shape != (len(JOINTS),) or not np.isfinite(angles).all():
            raise ValueError("joint_angles_rad must contain finite DIP, PIP, and MCP")
        if any(joint not in self._evaluation_ranges_rad for joint in JOINTS):
            raise ValueError("Moment-arm evaluation ranges are missing")
        for joint in JOINTS:
            low, high = self._evaluation_ranges_rad[joint]
            if not math.isfinite(low) or not math.isfinite(high) or low >= high:
                raise ValueError(f"Invalid Moment-arm evaluation range for {joint}")

    def matrix(
        self,
        joint_angles_rad: Sequence[float],
    ) -> np.ndarray:
        angles = np.asarray(joint_angles_rad, dtype=np.float64)
        if angles.shape != (len(JOINTS),):
            raise ValueError("joint_angles_rad must contain DIP, PIP, and MCP")
        if not np.isfinite(angles).all():
            raise ValueError("joint_angles_rad contains a non-finite value")
        self.validate_domain(angles)

        result = np.empty((len(TENDONS), len(JOINTS)), dtype=np.float64)
        for joint_index, (joint, angle) in enumerate(zip(JOINTS, angles)):
            evaluated_angle = self._clamped_angle(joint, float(angle))
            by_tendon = self._coefficients[joint]
            for tendon_index, tendon in enumerate(TENDONS):
                result[tendon_index, joint_index] = float(
                    np.polyval(by_tendon[tendon], evaluated_angle)
                )
        return result

    def integrate_excursion(
        self,
        start_angles_rad: Sequence[float],
        end_angles_rad: Sequence[float],
    ) -> np.ndarray:
        """Integrate signed tendon excursion, splitting at fit-range edges.

        Existing midpoint quadrature is retained inside the fitting range.
        Each interval is split at fitting boundaries so the endpoint value is
        held constant only over its out-of-range segment.
        """

        start = np.asarray(start_angles_rad, dtype=np.float64)
        end = np.asarray(end_angles_rad, dtype=np.float64)
        self.validate_domain(start)
        self.validate_domain(end)
        excursion = np.zeros(len(TENDONS), dtype=np.float64)

        for joint_index, joint in enumerate(JOINTS):
            q_start = float(start[joint_index])
            q_end = float(end[joint_index])
            delta = q_end - q_start
            if delta == 0.0:
                continue
            low, high = self._evaluation_ranges_rad[joint]
            fractions = [0.0, 1.0]
            for boundary in (low, high):
                fraction = (boundary - q_start) / delta
                if 0.0 < fraction < 1.0:
                    fractions.append(fraction)
            fractions.sort()

            by_tendon = self._coefficients[joint]
            for fraction_start, fraction_end in zip(fractions, fractions[1:]):
                segment_start = q_start + fraction_start * delta
                segment_end = q_start + fraction_end * delta
                segment_midpoint = 0.5 * (segment_start + segment_end)
                evaluation_angle = self._clamped_angle(joint, segment_midpoint)
                for tendon_index, tendon in enumerate(TENDONS):
                    moment_arm = float(
                        np.polyval(by_tendon[tendon], evaluation_angle)
                    )
                    excursion[tendon_index] += moment_arm * (
                        segment_end - segment_start
                    )

        if not np.isfinite(excursion).all():
            raise ValueError("Moment-arm excursion integration returned non-finite values")
        return excursion

    def _clamped_angle(self, joint: str, requested_angle_rad: float) -> float:
        low, high = self._evaluation_ranges_rad[joint]
        if requested_angle_rad < low:
            endpoint = low
        elif requested_angle_rad > high:
            endpoint = high
        else:
            return requested_angle_rad
        self._record_extrapolation(joint, requested_angle_rad, low, high)
        return endpoint

    def _record_extrapolation(
        self,
        joint: str,
        requested_angle_rad: float,
        low_rad: float,
        high_rad: float,
    ) -> None:
        callback = self._extrapolation_event_callback
        if callback is None:
            return
        endpoint_rad = low_rad if requested_angle_rad < low_rad else high_rad
        callback(
            {
                "joint": joint,
                "requested_angle_rad": float(requested_angle_rad),
                "requested_angle_deg": math.degrees(requested_angle_rad),
                "fitting_range_rad": {"q_min": low_rad, "q_max": high_rad},
                "fitting_range_deg": {
                    "q_min": math.degrees(low_rad),
                    "q_max": math.degrees(high_rad),
                },
                "used_endpoint": "q_min" if endpoint_rad == low_rad else "q_max",
                "used_endpoint_rad": endpoint_rad,
                "used_endpoint_deg": math.degrees(endpoint_rad),
                "constant_extrapolation_applied": True,
            }
        )

    def correction_excursion(
        self,
        joint_angles_rad: Sequence[float],
        joint_correction_rad: Sequence[float],
    ) -> np.ndarray:
        correction = np.asarray(joint_correction_rad, dtype=np.float64)
        if correction.shape != (len(JOINTS),):
            raise ValueError("joint_correction_rad must contain DIP, PIP, and MCP")
        if not np.isfinite(correction).all():
            raise ValueError("joint_correction_rad contains a non-finite value")
        return self.matrix(joint_angles_rad) @ correction


def _evaluation_range_deg(
    payload: dict[str, object],
    joint: str,
    path: Path,
) -> tuple[tuple[float, float], str]:
    """Resolve explicit model domain or the common measured fit-data domain."""

    raw_range = payload.get("evaluation_range_deg")
    if raw_range is not None:
        if not isinstance(raw_range, list) or len(raw_range) != 2:
            raise ValueError(f"Invalid evaluation_range_deg in {path}")
        low_deg, high_deg = (float(value) for value in raw_range)
        if (
            not math.isfinite(low_deg)
            or not math.isfinite(high_deg)
            or low_deg >= high_deg
        ):
            raise ValueError(f"Invalid evaluation_range_deg in {path}")
        return (low_deg, high_deg), "evaluation_range_deg"

    metrics = payload.get("fit_metrics")
    if not isinstance(metrics, dict):
        raise ValueError(
            f"Missing evaluation_range_deg or fit_metrics for {joint} in {path}"
        )
    ranges: list[tuple[float, float]] = []
    for tendon in TENDONS:
        item = metrics.get(tendon)
        if not isinstance(item, dict):
            raise ValueError(
                f"Missing fit range metadata for {joint}-{tendon} in {path}"
            )
        try:
            low_deg = float(item["angle_min_deg"])
            high_deg = float(item["angle_max_deg"])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(
                f"Invalid fit range metadata for {joint}-{tendon} in {path}"
            ) from exc
        if (
            not math.isfinite(low_deg)
            or not math.isfinite(high_deg)
            or low_deg >= high_deg
        ):
            raise ValueError(
                f"Invalid fit range metadata for {joint}-{tendon} in {path}"
            )
        ranges.append((low_deg, high_deg))

    low_deg = max(item[0] for item in ranges)
    high_deg = min(item[1] for item in ranges)
    if low_deg >= high_deg:
        raise ValueError(f"Tendon fit ranges do not overlap for {joint} in {path}")
    return (low_deg, high_deg), "intersection_of_all_tendon_fit_data_ranges"
