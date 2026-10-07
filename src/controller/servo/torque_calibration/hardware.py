"""Read synchronized servo and AS5048A angle differences for calibration."""

from __future__ import annotations

import math
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping

from controller.servo.hardware_session import API_ROOT, ServoHardwareSession


@dataclass(frozen=True)
class AngleCapture:
    mean_delta_deg: float
    std_delta_deg: float
    min_delta_deg: float
    max_delta_deg: float
    sample_count: int


def capture_angle_window(
    session: ServoHardwareSession,
    model: Mapping[str, Any],
    servo_id: int,
    duration_s: float,
    *,
    max_age_s: float = 0.5,
    stop_event: threading.Event | None = None,
) -> AngleCapture:
    """Capture a stationary angle window without sending any servo command."""

    duration_s = float(duration_s)
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("Capture duration must be a finite value greater than zero")
    if not session.connected or session.system is None:
        raise RuntimeError("Connect the servo and encoder before capturing a point")

    api_python = str(API_ROOT / "python")
    if api_python not in sys.path:
        sys.path.insert(0, api_python)
    from drivers.as5048a.torque import compute_torque_snapshot

    sensor_id = int(model["sensor_id"])
    servo_index = session.system.index_for_servo_id(int(servo_id))
    if servo_index is None:
        raise ValueError(f"Servo ID is not configured: {servo_id}")

    previous_snapshot: dict[str, Any] | None = None
    previous_pair: tuple[float, float] | None = None
    sample_count = 0
    mean_delta = 0.0
    m2_delta = 0.0
    min_delta = math.inf
    max_delta = -math.inf
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        if stop_event is not None and stop_event.is_set():
            raise RuntimeError("Capture cancelled")
        session.check_health()
        feedback = session.latest_feedback(max_age_s=max_age_s)
        if feedback is None:
            time.sleep(0.005)
            continue
        sample = feedback.encoder_by_sensor.get(sensor_id)
        encoder_received_at = feedback.encoder_received_at.get(sensor_id)
        if sample is None or encoder_received_at is None:
            time.sleep(0.005)
            continue
        pair = (feedback.servo_received_at, encoder_received_at)
        if pair == previous_pair:
            time.sleep(0.003)
            continue
        previous_pair = pair
        if not feedback.servo.get("valid", [True] * len(session.servo_ids))[servo_index]:
            continue
        sync_age_s = abs(encoder_received_at - feedback.servo_received_at)
        telemetry_age_s = max(
            feedback.servo_age_s,
            feedback.encoder_age_s(sensor_id) or 0.0,
            sync_age_s,
        )
        if telemetry_age_s > max_age_s:
            continue
        snapshot = compute_torque_snapshot(
            sample=sample,
            servo_id=servo_id,
            motor_name=str(model.get("motor_name", f"servo-{servo_id}")),
            servo_raw=feedback.servo["positions"][servo_index],
            counts_per_rev=model["servo_counts_per_rev"],
            servo_zero_raw=model["servo_zero_raw"],
            servo_direction=model["servo_direction"],
            encoder_direction=model["encoder_direction"],
            torque_zero_delta_deg=model["torque_zero_delta_deg"],
            calibration_slope=model["slope_nmm_per_deg"],
            calibration_intercept=0.0,
            calibration_angle_reference_deg=model["angle_reference_deg"],
            spring_k=0.0,
            moment_arm_mm=model["moment_arm_mm"],
            pull_angle_deg=model["pull_angle_deg"],
            telemetry_age_s=telemetry_age_s,
            telemetry_stale_after_s=max_age_s,
            previous_snapshot=previous_snapshot,
        )
        if snapshot is None:
            continue
        delta = float(snapshot["angle_delta_deg"])
        if math.isfinite(delta):
            sample_count += 1
            delta_from_mean = delta - mean_delta
            mean_delta += delta_from_mean / sample_count
            m2_delta += delta_from_mean * (delta - mean_delta)
            min_delta = min(min_delta, delta)
            max_delta = max(max_delta, delta)
            previous_snapshot = snapshot
        time.sleep(0.003)

    if sample_count == 0:
        raise RuntimeError(
            f"No valid synchronized angle samples were captured for servo {servo_id}"
        )
    return AngleCapture(
        mean_delta_deg=float(mean_delta),
        std_delta_deg=float(math.sqrt(max(0.0, m2_delta / sample_count))),
        min_delta_deg=float(min_delta),
        max_delta_deg=float(max_delta),
        sample_count=sample_count,
    )
