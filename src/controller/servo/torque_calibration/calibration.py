"""Fixed-slope intercept calculations for force-gauge calibration."""

from __future__ import annotations

import sys
import time
from pathlib import Path
from statistics import fmean, pstdev
from typing import Any, Mapping, Sequence

from controller.servo.hardware_session import API_ROOT


def load_servo_calibration_models(calibration_dir: Path) -> dict[int, dict[str, Any]]:
    """Load the existing six servo models without modifying their source files."""

    api_python = str(API_ROOT / "python")
    if api_python not in sys.path:
        sys.path.insert(0, api_python)
    from drivers.as5048a.calibration_store import load_calibration_points
    from drivers.as5048a.torque import calibration_model_from_points

    models: dict[int, dict[str, Any]] = {}
    calibration_root = Path(calibration_dir).expanduser().resolve()
    for servo_id in range(1, 7):
        path = calibration_root / f"spring_torque_calibration_servo_id_{servo_id}.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Calibration file not found: {path}")
        points = load_calibration_points(str(path), servo_id, f"servo-{servo_id}")
        model = calibration_model_from_points(points)
        if int(model["servo_id"]) != servo_id:
            raise ValueError(
                f"Calibration servo ID mismatch for {path}: "
                f"{model['servo_id']} != {servo_id}"
            )
        models[servo_id] = model
    return models


def calculate_calibration_point(
    model: Mapping[str, Any],
    *,
    force_n: float,
    angle_delta_deg: float,
    torque_direction: int,
) -> dict[str, float | int]:
    """Calculate one fixed-slope intercept point from a manual force reading."""

    api_python = str(API_ROOT / "python")
    if api_python not in sys.path:
        sys.path.insert(0, api_python)
    from drivers.as5048a.torque import force_to_torque_nmm

    force_n = float(force_n)
    angle_delta_deg = float(angle_delta_deg)
    direction = int(torque_direction)
    if force_n < 0.0:
        raise ValueError("Force-gauge reading must not be negative")
    if direction not in (-1, 1):
        raise ValueError("Torque direction must be -1 or 1")
    torque_nmm = force_to_torque_nmm(
        force_n,
        float(model["moment_arm_mm"]),
        float(model["pull_angle_deg"]),
        direction,
    )
    slope = float(model["slope_nmm_per_deg"])
    intercept = torque_nmm - slope * angle_delta_deg
    return {
        "force_n": force_n,
        "torque_direction": direction,
        "angle_delta_deg": angle_delta_deg,
        "torque_nmm": float(torque_nmm),
        "intercept_nmm": float(intercept),
        "captured_at_unix_s": time.time(),
    }


def fit_fixed_slope_intercept(
    model: Mapping[str, Any],
    points: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    """Fit only the intercept while keeping the existing slope fixed."""

    if len(points) < 2:
        raise ValueError("At least two force-gauge points are required")
    intercepts = [float(point["intercept_nmm"]) for point in points]
    return {
        "intercept_nmm": float(fmean(intercepts)),
        "intercept_std_nmm": float(pstdev(intercepts)),
        "intercept_min_nmm": float(min(intercepts)),
        "intercept_max_nmm": float(max(intercepts)),
        "point_count": len(intercepts),
        "slope_nmm_per_deg": float(model["slope_nmm_per_deg"]),
        "encoder_sensor_id": int(model["sensor_id"]),
        "moment_arm_mm": float(model["moment_arm_mm"]),
        "pull_angle_deg": float(model["pull_angle_deg"]),
        "points": [dict(point) for point in points],
    }
