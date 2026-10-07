"""Calibrated AS5048A/servo force estimates for CSV playback logging."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Any

from controller.servo.hardware_session import API_ROOT, HardwareFeedback


class CalibratedForceEstimator:
    """Reuse ServoHardwareAPI calibration math without owning serial I/O."""

    def __init__(
        self,
        *,
        calibration_dir: Path,
        zero_offset_path: Path,
        servo_ids: tuple[int, ...],
        tendon_by_servo_id: dict[int, str],
        telemetry_stale_after_s: float = 0.2,
    ) -> None:
        api_python = str(API_ROOT / "python")
        if api_python not in sys.path:
            sys.path.insert(0, api_python)
        from drivers.as5048a.calibration_store import load_calibration_points, load_zero_offsets
        from drivers.as5048a.torque import calibration_model_from_points

        self.servo_ids = tuple(servo_ids)
        self.tendon_by_servo_id = dict(tendon_by_servo_id)
        self.telemetry_stale_after_s = float(telemetry_stale_after_s)
        self._models: dict[int, dict[str, Any]] = {}
        self._previous: dict[int, dict[str, Any] | None] = {}
        offsets = load_zero_offsets(str(zero_offset_path)) if Path(zero_offset_path).is_file() else {}
        calibration_root = Path(calibration_dir).expanduser().resolve()
        for servo_id in self.servo_ids:
            path = calibration_root / f"spring_torque_calibration_servo_id_{servo_id}.csv"
            if not path.is_file():
                raise FileNotFoundError(f"Calibration file not found: {path}")
            tendon = self.tendon_by_servo_id[servo_id]
            points = load_calibration_points(str(path), servo_id, tendon)
            model = calibration_model_from_points(points)
            if int(model["servo_id"]) != int(servo_id):
                raise ValueError(f"Calibration servo ID mismatch for {path}")
            intercept = float(model["intercept_nmm"])
            offset = offsets.get(int(model["sensor_id"]))
            if offset is not None:
                if int(offset["servo_id"]) != int(servo_id):
                    raise ValueError(
                        f"Zero-offset servo ID mismatch for sensor {model['sensor_id']}"
                    )
                intercept = -float(model["slope_nmm_per_deg"]) * float(
                    offset["mean_delta_deg"]
                )
            model = dict(model)
            model["effective_intercept_nmm"] = intercept
            self._models[int(servo_id)] = model
            self._previous[int(servo_id)] = None

    def estimate(
        self,
        feedback: HardwareFeedback,
        servo_index_by_id: dict[int, int],
    ) -> dict[int, dict[str, Any]]:
        from drivers.as5048a.torque import compute_torque_snapshot

        result: dict[int, dict[str, Any]] = {}
        telemetry = feedback.servo
        for servo_id in self.servo_ids:
            model = self._models[servo_id]
            sensor_id = int(model["sensor_id"])
            sample = feedback.encoder_by_sensor.get(sensor_id)
            encoder_age_s = feedback.encoder_age_s(sensor_id)
            sync_age_s = None
            if sensor_id in feedback.encoder_received_at:
                sync_age_s = abs(
                    feedback.encoder_received_at[sensor_id] - feedback.servo_received_at
                )
            telemetry_age_s = None
            ages = [feedback.servo_age_s]
            if encoder_age_s is not None:
                ages.append(encoder_age_s)
            if sync_age_s is not None:
                ages.append(sync_age_s)
            if ages:
                telemetry_age_s = max(ages)
            base = {
                "servo_id": servo_id,
                "tendon": self.tendon_by_servo_id[servo_id],
                "force_n": None,
                "torque_nmm": None,
                "angle_delta_deg": None,
                "servo_age_s": feedback.servo_age_s,
                "encoder_age_s": encoder_age_s,
                "sync_age_s": sync_age_s,
                "status": "NO_ENCODER_SAMPLE" if sample is None else "UNAVAILABLE",
            }
            if sample is None:
                result[servo_id] = base
                continue
            index = servo_index_by_id[servo_id]
            valid = telemetry.get("valid", [True] * len(self.servo_ids))[index]
            if not valid:
                base["status"] = "SERVO_TELEMETRY_INVALID"
                result[servo_id] = base
                continue
            snapshot = compute_torque_snapshot(
                sample=sample,
                servo_id=servo_id,
                motor_name=self.tendon_by_servo_id[servo_id],
                servo_raw=telemetry["positions"][index],
                counts_per_rev=model["servo_counts_per_rev"],
                servo_zero_raw=model["servo_zero_raw"],
                servo_direction=model["servo_direction"],
                encoder_direction=model["encoder_direction"],
                torque_zero_delta_deg=model["torque_zero_delta_deg"],
                calibration_slope=model["slope_nmm_per_deg"],
                calibration_intercept=model["effective_intercept_nmm"],
                calibration_angle_reference_deg=model["angle_reference_deg"],
                spring_k=0.0,
                moment_arm_mm=model["moment_arm_mm"],
                pull_angle_deg=model["pull_angle_deg"],
                telemetry_age_s=telemetry_age_s,
                telemetry_stale_after_s=self.telemetry_stale_after_s,
                previous_snapshot=self._previous[servo_id],
            )
            if snapshot is not None:
                self._previous[servo_id] = snapshot
            if snapshot is not None and snapshot.get("force_n") is not None:
                base.update(
                    force_n=float(snapshot["force_n"]),
                    torque_nmm=snapshot.get("torque_nmm"),
                    angle_delta_deg=snapshot.get("angle_delta_deg"),
                    status=str(snapshot.get("sample_status") or "OK"),
                )
            else:
                base["status"] = "FORCE_UNAVAILABLE"
            result[servo_id] = base
        return result
