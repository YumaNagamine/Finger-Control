"""Persistence for fixed-slope torque-intercept calibration results."""

from __future__ import annotations

import json
import math
import os
import shutil
import tempfile
import time
from pathlib import Path
from typing import Any, Mapping

from controller.servo.hardware_session import API_ROOT


INTERCEPT_CALIBRATION_FORMAT = "servo_torque_intercept_calibration"
INTERCEPT_CALIBRATION_VERSION = 1
DEFAULT_INTERCEPT_CALIBRATION_PATH = (
    API_ROOT / "calibration" / "current" / "torque_intercept_calibration.json"
)
DEFAULT_ENCODER_POSITION_PATH = (
    API_ROOT / "calibration" / "current" / "torque_calibration_encoder_positions.json"
)


def _minimum_point_count(record: Mapping[str, Any]) -> int:
    return 1 if record.get("method") == "loose_position_zero" else 2


def _finite(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number


def _servo_id(value: Any, field_name: str = "servo_id") -> int:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be an integer")
    try:
        result = int(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be an integer") from exc
    if result not in range(1, 7):
        raise ValueError(f"{field_name} must be between 1 and 6")
    return result


def load_torque_intercept_calibration(
    path: Path | str = DEFAULT_INTERCEPT_CALIBRATION_PATH,
) -> dict[int, dict[str, Any]]:
    """Load fixed-slope intercept records, returning an empty mapping if absent."""

    calibration_path = Path(path)
    if not calibration_path.is_file():
        return {}
    try:
        with calibration_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid torque-intercept calibration JSON: {calibration_path}") from exc

    if not isinstance(payload, dict):
        raise ValueError("Torque-intercept calibration must be a JSON object")
    if payload.get("format") != INTERCEPT_CALIBRATION_FORMAT:
        raise ValueError(
            f"Unsupported torque-intercept calibration format: {payload.get('format')}"
        )
    if payload.get("version") != INTERCEPT_CALIBRATION_VERSION:
        raise ValueError(
            f"Unsupported torque-intercept calibration version: {payload.get('version')}"
        )
    channels = payload.get("channels")
    if not isinstance(channels, dict):
        raise ValueError("Torque-intercept calibration must contain a channels object")

    records: dict[int, dict[str, Any]] = {}
    for key, raw_record in channels.items():
        if not isinstance(raw_record, dict):
            raise ValueError(f"Channel {key} must be an object")
        servo_id = _servo_id(raw_record.get("servo_id", key))
        sensor_id = _servo_id(
            raw_record.get("encoder_sensor_id", servo_id),
            "encoder_sensor_id",
        )
        record = dict(raw_record)
        record["servo_id"] = servo_id
        record["encoder_sensor_id"] = sensor_id
        record["slope_nmm_per_deg"] = _finite(
            record.get("slope_nmm_per_deg"),
            "slope_nmm_per_deg",
        )
        record["intercept_nmm"] = _finite(record.get("intercept_nmm"), "intercept_nmm")
        point_count = record.get("point_count", 0)
        if isinstance(point_count, bool):
            raise ValueError("point_count must be an integer")
        try:
            record["point_count"] = int(point_count)
        except (TypeError, ValueError) as exc:
            raise ValueError("point_count must be an integer") from exc
        minimum = _minimum_point_count(record)
        if record["point_count"] < minimum:
            raise ValueError(f"point_count must be at least {minimum}")
        records[servo_id] = record
    return records


def intercept_for_model(
    records: Mapping[int, Mapping[str, Any]],
    model: Mapping[str, Any],
) -> float | None:
    """Return a validated calibrated intercept for a live model, if present."""

    servo_id = _servo_id(model.get("servo_id"))
    record = records.get(servo_id)
    if record is None:
        return None
    sensor_id = _servo_id(model.get("sensor_id"), "sensor_id")
    if int(record["encoder_sensor_id"]) != sensor_id:
        raise ValueError(
            f"Torque-intercept sensor mismatch for servo {servo_id}: "
            f"{record['encoder_sensor_id']} != {sensor_id}"
        )
    model_slope = _finite(model.get("slope_nmm_per_deg"), "model slope")
    saved_slope = _finite(record.get("slope_nmm_per_deg"), "saved slope")
    if not math.isclose(model_slope, saved_slope, rel_tol=1e-6, abs_tol=1e-6):
        raise ValueError(
            f"Torque-intercept calibration for servo {servo_id} was created with a "
            f"different slope ({saved_slope} != {model_slope}); recalibrate the intercept."
        )
    return _finite(record.get("intercept_nmm"), "intercept_nmm")


def save_torque_intercept_calibration(
    path: Path | str,
    records: Mapping[int, Mapping[str, Any]],
) -> Path:
    """Atomically save records and preserve one recoverable backup when replacing a file."""

    calibration_path = Path(path)
    calibration_path.parent.mkdir(parents=True, exist_ok=True)
    channels: dict[str, dict[str, Any]] = {}
    for raw_servo_id, raw_record in sorted(records.items(), key=lambda item: int(item[0])):
        servo_id = _servo_id(raw_servo_id)
        if not isinstance(raw_record, Mapping):
            raise ValueError(f"Record for servo {servo_id} must be an object")
        record = dict(raw_record)
        record["servo_id"] = servo_id
        record["encoder_sensor_id"] = _servo_id(
            record.get("encoder_sensor_id", servo_id),
            "encoder_sensor_id",
        )
        record["slope_nmm_per_deg"] = _finite(
            record.get("slope_nmm_per_deg"),
            "slope_nmm_per_deg",
        )
        record["intercept_nmm"] = _finite(record.get("intercept_nmm"), "intercept_nmm")
        record["point_count"] = int(record.get("point_count", 0))
        minimum = _minimum_point_count(record)
        if record["point_count"] < minimum:
            raise ValueError(f"point_count must be at least {minimum}")
        channels[str(servo_id)] = record

    methods = {
        str(record.get("method", "fixed_slope_force_gauge"))
        for record in channels.values()
    }

    payload = {
        "format": INTERCEPT_CALIBRATION_FORMAT,
        "version": INTERCEPT_CALIBRATION_VERSION,
        "saved_at_unix_s": time.time(),
        "method": methods.pop() if len(methods) == 1 else "mixed",
        "channels": channels,
    }

    backup_path = calibration_path.with_suffix(calibration_path.suffix + ".bak")
    if calibration_path.exists():
        shutil.copy2(calibration_path, backup_path)

    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            dir=calibration_path.parent,
            prefix=f".{calibration_path.name}.",
            suffix=".tmp",
            delete=False,
        ) as handle:
            temporary_path = handle.name
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, calibration_path)
    except Exception:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass
        raise
    return calibration_path


def load_encoder_calibration_positions(
    path: Path | str = DEFAULT_ENCODER_POSITION_PATH,
) -> dict[int, dict[str, Any]]:
    """Load remembered magnetic-encoder positions for future automatic calibration."""

    position_path = Path(path)
    if not position_path.is_file():
        return {}
    try:
        with position_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid encoder calibration-position JSON: {position_path}") from exc
    if not isinstance(payload, dict) or payload.get("format") != "torque_calibration_encoder_positions":
        raise ValueError("Unsupported encoder calibration-position format")
    if payload.get("version") != 1 or not isinstance(payload.get("channels"), dict):
        raise ValueError("Unsupported encoder calibration-position version or channels")

    records: dict[int, dict[str, Any]] = {}
    for key, raw_record in payload["channels"].items():
        if not isinstance(raw_record, dict):
            raise ValueError(f"Encoder calibration position {key} must be an object")
        record = dict(raw_record)
        servo_id = _servo_id(record.get("servo_id", key))
        record["servo_id"] = servo_id
        record["encoder_sensor_id"] = _servo_id(
            record.get("encoder_sensor_id", servo_id), "encoder_sensor_id"
        )
        record["encoder_position_deg"] = _finite(
            record.get("encoder_position_deg"), "encoder_position_deg"
        )
        records[servo_id] = record
    return records


def save_encoder_calibration_positions(
    path: Path | str,
    records: Mapping[int, Mapping[str, Any]],
) -> Path:
    """Atomically overwrite remembered magnetic-encoder calibration positions."""

    position_path = Path(path)
    position_path.parent.mkdir(parents=True, exist_ok=True)
    channels: dict[str, dict[str, Any]] = {}
    for raw_servo_id, raw_record in sorted(records.items(), key=lambda item: int(item[0])):
        servo_id = _servo_id(raw_servo_id)
        record = dict(raw_record)
        record["servo_id"] = servo_id
        record["encoder_sensor_id"] = _servo_id(
            record.get("encoder_sensor_id", servo_id), "encoder_sensor_id"
        )
        record["encoder_position_deg"] = _finite(
            record.get("encoder_position_deg"), "encoder_position_deg"
        )
        channels[str(servo_id)] = record

    payload = {
        "format": "torque_calibration_encoder_positions",
        "version": 1,
        "saved_at_unix_s": time.time(),
        "channels": channels,
    }
    backup_path = position_path.with_suffix(position_path.suffix + ".bak")
    if position_path.exists():
        shutil.copy2(position_path, backup_path)
    temporary_path: str | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w", encoding="utf-8", dir=position_path.parent,
            prefix=f".{position_path.name}.", suffix=".tmp", delete=False,
        ) as handle:
            temporary_path = handle.name
            json.dump(payload, handle, ensure_ascii=False, indent=2)
            handle.write("\n")
            handle.flush()
            os.fsync(handle.fileno())
        os.replace(temporary_path, position_path)
    except Exception:
        if temporary_path is not None:
            try:
                os.unlink(temporary_path)
            except OSError:
                pass
        raise
    return position_path
