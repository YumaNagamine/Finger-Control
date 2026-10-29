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
    calibration_basis: str = "unspecified"
    approval_status: str = "unapproved"
    spool_radius_mm: float | None = None

    @property
    def p2p_conversion_approved(self) -> bool:
        """Whether the conversion has an accepted, explicit provenance."""

        return encoder_conversion_is_p2p_approved(self)


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
    calibration_basis = str(metadata.get("calibration_basis", "unspecified"))
    approval_status = str(metadata.get("approval_status", "unapproved"))
    spool_radius_mm = (
        _finite_positive(metadata, "spool_radius_mm")
        if "spool_radius_mm" in metadata
        else None
    )
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
        calibration_basis=calibration_basis,
        approval_status=approval_status,
        spool_radius_mm=spool_radius_mm,
    )


def encoder_conversion_is_p2p_approved(
    calibration: EncoderExcursionCalibration,
) -> bool:
    """Accept measured calibration or explicitly approved spool geometry.

    Geometry-based approval is intentionally separate from
    ``calibration_status`` so a nominal coefficient is never labeled as
    measured. The stored scale must agree with the declared spool radius.
    """

    if (
        str(getattr(calibration, "calibration_status", ""))
        .strip()
        .lower()
        == "measured"
    ):
        return True
    if (
        str(getattr(calibration, "calibration_basis", ""))
        .strip()
        .lower()
        != "spool_radius_geometry"
        or str(getattr(calibration, "approval_status", ""))
        .strip()
        .lower()
        != "approved"
    ):
        return False

    try:
        radius_mm = float(getattr(calibration, "spool_radius_mm"))
        degrees_per_mm = tuple(
            float(value) for value in getattr(calibration, "degrees_per_mm")
        )
    except (AttributeError, TypeError, ValueError, OverflowError):
        return False
    if (
        not math.isfinite(radius_mm)
        or radius_mm <= 0.0
        or len(degrees_per_mm) != len(TENDONS)
    ):
        return False
    expected_degrees_per_mm = 360.0 / (2.0 * math.pi * radius_mm)
    return all(
        math.isfinite(value)
        and value > 0.0
        and math.isclose(
            value,
            expected_degrees_per_mm,
            rel_tol=1e-5,
            abs_tol=1e-8,
        )
        for value in degrees_per_mm
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
