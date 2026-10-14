"""Calibration and PID settings for CSV-to-magnetic-encoder playback."""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from controller.servo_mapping import TENDONS


@dataclass(frozen=True)
class EncoderExcursionCalibration:
    """Per-tendon excursion conversion and encoder PID settings."""

    degrees_per_mm: tuple[float, ...]
    encoder_directions: tuple[float, ...]
    kp: float
    ki: float
    kd: float
    max_speed: float
    max_acceleration: float
    hard_error_margin_deg: float
    minimum_hard_error_deg: float
    return_tolerance_deg: float
    calibration_status: str


def load_encoder_excursion_calibration(path: Path) -> EncoderExcursionCalibration:
    """Load and validate encoder position conversion and PID settings."""

    with Path(path).expanduser().open("r", encoding="utf-8") as handle:
        payload: Any = json.load(handle)
    if not isinstance(payload, dict) or payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported encoder calibration schema: {path}")

    scales = _tendon_values(payload.get("encoder_degrees_per_mm"), "encoder_degrees_per_mm")
    directions = _tendon_values(payload.get("encoder_direction"), "encoder_direction")
    if any(value <= 0.0 for value in scales):
        raise ValueError("encoder_degrees_per_mm values must be positive")
    if any(value not in (-1.0, 1.0) for value in directions):
        raise ValueError("encoder_direction values must be either -1 or 1")

    pid = payload.get("pid")
    if not isinstance(pid, dict):
        raise ValueError("pid must be an object")
    values = {
        name: _finite_positive(pid, name)
        for name in (
            "kp",
            "max_speed",
            "max_acceleration",
            "hard_error_margin_deg",
            "minimum_hard_error_deg",
            "return_tolerance_deg",
        )
    }
    values["ki"] = _finite(pid, "ki")
    values["kd"] = _finite(pid, "kd")

    metadata = payload.get("metadata", {})
    if not isinstance(metadata, dict):
        raise ValueError("metadata must be an object")
    calibration_status = str(metadata.get("calibration_status", "nominal"))
    return EncoderExcursionCalibration(
        degrees_per_mm=scales,
        encoder_directions=directions,
        kp=values["kp"],
        ki=values["ki"],
        kd=values["kd"],
        max_speed=values["max_speed"],
        max_acceleration=values["max_acceleration"],
        hard_error_margin_deg=values["hard_error_margin_deg"],
        minimum_hard_error_deg=values["minimum_hard_error_deg"],
        return_tolerance_deg=values["return_tolerance_deg"],
        calibration_status=calibration_status,
    )


def _tendon_values(raw: Any, name: str) -> tuple[float, ...]:
    if not isinstance(raw, dict):
        raise ValueError(f"{name} must be an object keyed by tendon")
    values: list[float] = []
    for tendon in TENDONS:
        try:
            value = float(raw[tendon])
        except (KeyError, TypeError, ValueError) as exc:
            raise ValueError(f"{name}.{tendon} must be numeric") from exc
        if not math.isfinite(value):
            raise ValueError(f"{name}.{tendon} must be finite")
        values.append(value)
    return tuple(values)


def _finite_positive(raw: dict[str, Any], name: str) -> float:
    value = _finite(raw, name)
    if value <= 0.0:
        raise ValueError(f"pid.{name} must be positive")
    return value


def _finite(raw: dict[str, Any], name: str) -> float:
    try:
        value = float(raw[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"pid.{name} must be numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"pid.{name} must be finite")
    return value
