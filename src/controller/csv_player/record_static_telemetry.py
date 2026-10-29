"""Read-only static Servo, Encoder, and DLC telemetry capture and analysis.

Run this only after closing other applications that may own the camera or the
Servo/Encoder serial ports. The recorder opens the Servo transport read-only;
it does not create a Servo command dispatcher or send Servo commands.
"""

from __future__ import annotations

import argparse
import bisect
import json
import math
import statistics
import sys
import threading
import time
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


SRC_ROOT = Path(__file__).resolve().parents[2]
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from controller.servo.hardware_session import ServoHardwareSession  # noqa: E402
from controller.servo_mapping import (  # noqa: E402
    ENCODER_SENSOR_IDS_BY_SERVO_ID,
    SERVO_IDS,
    SERVO_IDS_BY_TENDON,
    SERVO_NAMES_BY_ID,
    TENDONS,
)
from observation.camera.camera_param_resolver import resolve_param_path  # noqa: E402
from observation.vision.realtime_joint_angle_source import (  # noqa: E402
    JOINTS,
    RealtimeJointAngleSource,
    TemporalAngleGate,
)


DEFAULT_SERVO_PORT = "COM3"
DEFAULT_ENCODER_PORT = "COM5"
DEFAULT_DLC_CONFIG_PATH = (
    SRC_ROOT / "observation" / "vision" / "config_deeplabcut_angle_control.json"
)
DEFAULT_CAMERA_CONFIG_PATH = resolve_param_path("camera_config_dlc.json")
ENCODER_COUNTS_PER_REVOLUTION = 1 << 14
CURRENT_Q0_FRAME_COUNT = 5
CURRENT_Q0_MAX_SPREAD_DEG = 2.0
CURRENT_DLC_TEMPORAL_JUMP_DEG = 30.0
CURRENT_DLC_TEMPORAL_MAX_HOLD_FRAMES = 5


class JsonlEventWriter:
    """Small synchronous JSONL writer for bounded-rate telemetry collection."""

    def __init__(self, path: Path, *, run_id: str) -> None:
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self._handle = self.path.open("x", encoding="utf-8", newline="\n")
        self._lock = threading.Lock()
        self._run_id = str(run_id)
        self._sequence = 0
        self._last_flush_at = time.monotonic()

    def log(
        self,
        event_type: str,
        *,
        event_monotonic_s: float,
        payload: Mapping[str, Any],
    ) -> None:
        with self._lock:
            event = {
                "schema_version": 1,
                "run_id": self._run_id,
                "event_sequence": self._sequence,
                "event_type": str(event_type),
                "event_monotonic_s": float(event_monotonic_s),
                "payload": dict(payload),
            }
            encoded = json.dumps(event, ensure_ascii=False, allow_nan=False)
            self._handle.write(encoded + "\n")
            self._sequence += 1
            now = time.monotonic()
            if now - self._last_flush_at >= 0.5:
                self._handle.flush()
                self._last_flush_at = now

    def close(self) -> None:
        with self._lock:
            if not self._handle.closed:
                self._handle.flush()
                self._handle.close()


class _TelemetryDrain(threading.Thread):
    def __init__(
        self,
        hardware: ServoHardwareSession,
        writer: JsonlEventWriter,
        stop_event: threading.Event,
    ) -> None:
        super().__init__(name="static-telemetry-drain", daemon=True)
        self.hardware = hardware
        self.writer = writer
        self.stop_event = stop_event
        self.error: str | None = None
        self._counts: dict[str, int] = defaultdict(int)

    @property
    def counts(self) -> dict[str, int]:
        return dict(self._counts)

    def run(self) -> None:
        try:
            while not self.stop_event.wait(0.01):
                self._drain()
            self._drain()
        except Exception as exc:  # surfaced by the main thread after capture
            self.error = f"{type(exc).__name__}: {exc}"

    def _drain(self) -> None:
        for record in self.hardware.drain_telemetry_samples():
            signal = record.get("signal")
            if signal == "servo_frame":
                self._write_servo_frame(record)
            elif signal == "servo_line":
                self._write_invalid_line("servo", record)
            elif signal == "encoder_line":
                self._write_encoder_line(record)

    def _write_invalid_line(
        self,
        signal: str,
        record: Mapping[str, Any],
    ) -> None:
        self.writer.log(
            "static_telemetry_sample",
            event_monotonic_s=float(record["received_monotonic_s"]),
            payload={
                "signal": signal,
                "channel_id": None,
                "tendon": None,
                "received_monotonic_s": record.get("received_monotonic_s"),
                "device_ms": None,
                "valid": False,
                "raw_line": record.get("raw_line"),
                "parsed": record.get("parsed"),
            },
        )
        self._counts[f"{signal}_invalid_lines"] += 1

    def _write_servo_frame(self, record: Mapping[str, Any]) -> None:
        raw = record.get("raw")
        if not isinstance(raw, Mapping):
            return
        positions = raw.get("positions", ())
        raw_positions = raw.get("raw_positions", ())
        valid = raw.get("valid", ())
        loads = raw.get("loads", ())
        speeds = raw.get("speeds", ())
        for index, servo_id in enumerate(SERVO_IDS):
            position = _sequence_item(positions, index)
            if position is None:
                continue
            tendon = SERVO_NAMES_BY_ID[index]
            payload = {
                "signal": "servo",
                "channel_id": servo_id,
                "tendon": tendon,
                "received_monotonic_s": record.get("received_monotonic_s"),
                "device_ms": record.get("device_ms"),
                "position_count": position,
                "raw_position_count": _sequence_item(raw_positions, index),
                "load_raw": _sequence_item(loads, index),
                "speed_raw": _sequence_item(speeds, index),
                "valid": bool(_sequence_item(valid, index, default=False)),
                "position_field": "accumulated_pos_or_firmware_position",
                "position_unit_note": (
                    "integer firmware telemetry unit; mechanical scale not verified"
                ),
                "protocol_fields": raw.get("parts"),
            }
            self.writer.log(
                "static_telemetry_sample",
                event_monotonic_s=float(record["received_monotonic_s"]),
                payload=payload,
            )
            self._counts["servo"] += 1

    def _write_encoder_line(self, record: Mapping[str, Any]) -> None:
        sample = record.get("raw")
        if not isinstance(sample, Mapping):
            sample = {}
        sensor_id = record.get("sensor_id", sample.get("sensor_id"))
        tendon = _tendon_by_sensor_id(sensor_id)
        payload = {
            "signal": "encoder",
            "channel_id": sensor_id,
            "tendon": tendon,
            "received_monotonic_s": record.get("received_monotonic_s"),
            "device_ms": record.get("device_ms"),
            "raw_count": sample.get("raw"),
            "angle_deg": sample.get("angle_deg"),
            "relative_deg": sample.get("relative_deg"),
            "zero_deg": sample.get("zero_deg"),
            "status": record.get("status"),
            "parsed_kind": record.get("parsed_kind"),
            "valid": bool(record.get("valid", False)),
            "raw_line": record.get("raw_line"),
            "count_modulus": ENCODER_COUNTS_PER_REVOLUTION,
            "count_wrap_rule": "circular difference modulo 16384",
        }
        self.writer.log(
            "static_telemetry_sample",
            event_monotonic_s=float(record["received_monotonic_s"]),
            payload=payload,
        )
        self._counts["encoder"] += 1


def _sequence_item(value: Any, index: int, *, default: Any = None) -> Any:
    try:
        return value[index]
    except (IndexError, KeyError, TypeError):
        return default


def _tendon_by_sensor_id(sensor_id: Any) -> str | None:
    try:
        target_id = int(sensor_id)
    except (TypeError, ValueError, OverflowError):
        return None
    for servo_id, encoder_id in ENCODER_SENSOR_IDS_BY_SERVO_ID.items():
        if encoder_id == target_id:
            try:
                return TENDONS[SERVO_IDS_BY_TENDON.index(servo_id)]
            except ValueError:
                return None
    return None


def circular_count_delta(start: int, end: int, modulus: int) -> int:
    if modulus <= 0:
        raise ValueError("modulus must be positive")
    difference = abs(int(end) - int(start)) % modulus
    return min(difference, modulus - difference)


def _percentile(values: Sequence[float], percentile: float) -> float | None:
    if not values:
        return None
    ordered = sorted(float(value) for value in values)
    position = (len(ordered) - 1) * percentile / 100.0
    low = int(math.floor(position))
    high = int(math.ceil(position))
    if low == high:
        return ordered[low]
    fraction = position - low
    return ordered[low] * (1.0 - fraction) + ordered[high] * fraction


def _delta_summary(
    samples: Sequence[tuple[float, float]],
    lag_s: float,
    *,
    circular_modulus: int | None = None,
    rejection_threshold: float | None = None,
) -> dict[str, Any]:
    ordered = sorted(
        ((float(timestamp), float(value)) for timestamp, value in samples),
        key=lambda item: item[0],
    )
    times = [item[0] for item in ordered]
    deltas: list[float] = []
    actual_lags: list[float] = []
    for index, (timestamp, start_value) in enumerate(ordered):
        next_index = bisect.bisect_left(times, timestamp + lag_s, lo=index + 1)
        if next_index >= len(ordered):
            continue
        next_time, end_value = ordered[next_index]
        if circular_modulus is None:
            delta = abs(end_value - start_value)
        else:
            delta = float(
                circular_count_delta(
                    int(round(start_value)),
                    int(round(end_value)),
                    circular_modulus,
                )
            )
        deltas.append(delta)
        actual_lags.append(next_time - timestamp)
    result: dict[str, Any] = {
        "requested_lag_s": lag_s,
        "pair_count": len(deltas),
        "actual_lag_median_s": statistics.median(actual_lags)
        if actual_lags
        else None,
        "abs_delta_count_or_deg": {
            "p50": _percentile(deltas, 50.0),
            "p90": _percentile(deltas, 90.0),
            "p95": _percentile(deltas, 95.0),
            "p99": _percentile(deltas, 99.0),
            "max": max(deltas) if deltas else None,
        },
    }
    if rejection_threshold is not None:
        result["current_guard_rejection_fraction"] = (
            sum(value > rejection_threshold for value in deltas) / len(deltas)
            if deltas
            else None
        )
        result["current_guard_rejection_count"] = sum(
            value > rejection_threshold for value in deltas
        )
    return result


def _linear_drift_per_minute(samples: Sequence[tuple[float, float]]) -> float | None:
    if len(samples) < 2:
        return None
    ordered = sorted(samples)
    t0 = ordered[0][0]
    times = [timestamp - t0 for timestamp, _ in ordered]
    values = [value for _, value in ordered]
    mean_t = statistics.mean(times)
    mean_v = statistics.mean(values)
    denominator = sum((value - mean_t) ** 2 for value in times)
    if denominator <= 0.0:
        return None
    slope_per_second = sum(
        (timestamp - mean_t) * (value - mean_v)
        for timestamp, value in zip(times, values)
    ) / denominator
    return slope_per_second * 60.0


def _dlc_guard_summary(
    dlc_samples: Sequence[Mapping[str, Any]],
    lag_s: float,
    *,
    frame_count: int = CURRENT_Q0_FRAME_COUNT,
    max_spread_deg: float = CURRENT_Q0_MAX_SPREAD_DEG,
) -> dict[str, Any]:
    usable = []
    for sample in sorted(
        dlc_samples,
        key=lambda item: float(item.get("captured_at_monotonic_s", -math.inf)),
    ):
        angles = sample.get("p2p_angles_deg", sample.get("angles_deg"))
        captured_at = _finite_number(sample.get("captured_at_monotonic_s"))
        p2p_valid = bool(sample.get("p2p_valid", sample.get("valid")))
        p2p_held = bool(sample.get("p2p_held_last", sample.get("held_last", False)))
        if (
            not p2p_valid
            or p2p_held
            or captured_at is None
            or not isinstance(angles, Mapping)
        ):
            continue
        joint_angles = {joint: _finite_number(angles.get(joint)) for joint in JOINTS}
        if any(value is None for value in joint_angles.values()):
            continue
        usable.append((captured_at, joint_angles))
    times = [item[0] for item in usable]
    windows = 0
    spread_rejections = 0
    start_checks = 0
    start_change_rejections = 0
    combined_rejections = 0
    for start_index in range(0, len(usable) - frame_count + 1):
        window = usable[start_index : start_index + frame_count]
        q0_time, q0_angles = window[-1]
        spreads = {
            joint: max(sample[1][joint] for sample in window)
            - min(sample[1][joint] for sample in window)
            for joint in JOINTS
        }
        spread_rejected = any(value > max_spread_deg for value in spreads.values())
        windows += 1
        spread_rejections += int(spread_rejected)
        runner_index = bisect.bisect_left(times, q0_time + lag_s, lo=start_index + frame_count)
        if runner_index >= len(usable):
            continue
        start_checks += 1
        _, runner_angles = usable[runner_index]
        movement_rejected = any(
            abs(runner_angles[joint] - q0_angles[joint]) > spreads[joint] + 1e-9
            for joint in JOINTS
        )
        start_change_rejections += int(movement_rejected)
        combined_rejections += int(spread_rejected or movement_rejected)
    return {
        "frame_count": frame_count,
        "max_spread_deg": max_spread_deg,
        "q0_windows": windows,
        "windows_with_runner_sample": start_checks,
        "spread_gate_reject_count": spread_rejections,
        "spread_gate_reject_fraction": (
            spread_rejections / windows if windows else None
        ),
        "q0_to_runner_gate_reject_count": start_change_rejections,
        "q0_to_runner_gate_reject_fraction": (
            start_change_rejections / start_checks if start_checks else None
        ),
        "combined_reject_count": combined_rejections,
        "combined_reject_fraction": (
            combined_rejections / start_checks if start_checks else None
        ),
        "note": (
            "Uses p2p_angles_deg/p2p_valid when present, including the current "
            "temporal gate and excluding held frames; otherwise uses raw DLC values."
        ),
    }


def _finite_number(value: Any) -> float | None:
    try:
        number = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return number if math.isfinite(number) else None


def _load_jsonl(path: Path) -> Iterable[dict[str, Any]]:
    with Path(path).open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            try:
                event = json.loads(line)
            except json.JSONDecodeError as exc:
                raise ValueError(f"Invalid JSONL at {path}:{line_number}") from exc
            if isinstance(event, dict):
                yield event


def _actual_p2p_lags(paths: Sequence[Path]) -> dict[tuple[str, str], list[float]]:
    lags: dict[tuple[str, str], list[float]] = defaultdict(list)
    for path in paths:
        for event in _load_jsonl(path):
            if event.get("event_type") != "point_to_point_start_alignment_check":
                continue
            payload = event.get("payload")
            if not isinstance(payload, Mapping):
                continue
            timing = payload.get("time_deltas_s")
            if not isinstance(timing, Mapping):
                continue
            for signal, key in (("servo", "servo_by_tendon"), ("encoder", "encoder_by_tendon")):
                values = timing.get(key)
                if not isinstance(values, Mapping):
                    continue
                for tendon, lag in values.items():
                    numeric = _finite_number(lag)
                    if numeric is not None and numeric > 0.0:
                        lags[(signal, str(tendon))].append(numeric)
            dlc_lag = _finite_number(timing.get("dlc_q0_to_runner"))
            if (
                dlc_lag is not None
                and dlc_lag > 0.0
                and payload.get("runner_dlc_snapshot_used_for_validation")
            ):
                lags[("dlc", "all")].append(dlc_lag)
    return dict(lags)


def analyze_static_telemetry(
    path: Path,
    *,
    lags_s: Sequence[float] = (),
    p2p_event_paths: Sequence[Path] = (),
) -> dict[str, Any]:
    generic_lags = tuple(float(value) for value in lags_s)
    if any(not math.isfinite(value) or value <= 0.0 for value in generic_lags):
        raise ValueError("All lags must be finite and positive")
    events = list(_load_jsonl(Path(path)))
    samples = [
        event.get("payload", {})
        for event in events
        if event.get("event_type") == "static_telemetry_sample"
        and isinstance(event.get("payload"), Mapping)
    ]
    actual_lags = _actual_p2p_lags(p2p_event_paths)

    def channel_lags(signal: str, tendon: str) -> tuple[float, ...]:
        observed = actual_lags.get((signal, tendon), ())
        if signal == "dlc":
            observed = actual_lags.get(("dlc", "all"), ())
        return tuple(observed) if observed else generic_lags

    report: dict[str, Any] = {
        "input": str(Path(path).resolve()),
        "event_count": len(events),
        "lag_sources": {
            "p2p_alignment_events": [str(Path(item).resolve()) for item in p2p_event_paths],
            "explicit_lags_s": list(generic_lags),
            "actual_p2p_lag_channels": len(actual_lags),
        },
        "servo": {},
        "encoder": {},
        "dlc": {},
        "tolerance_decision": None,
        "tolerance_note": (
            "This report describes observed noise and current-guard rejection rates. "
            "It does not approve physical limits or select start tolerances."
        ),
    }

    for signal in ("servo", "encoder"):
        for tendon in TENDONS:
            channel_samples: list[tuple[float, float]] = []
            invalid_count = 0
            for sample in samples:
                if sample.get("signal") != signal or sample.get("tendon") != tendon:
                    continue
                timestamp = _finite_number(sample.get("received_monotonic_s"))
                value = _finite_number(
                    sample.get("position_count")
                    if signal == "servo"
                    else sample.get("raw_count")
                )
                if sample.get("valid") and timestamp is not None and value is not None:
                    channel_samples.append((timestamp, value))
                else:
                    invalid_count += 1
            summaries = []
            for lag in channel_lags(signal, tendon):
                summary = _delta_summary(
                    channel_samples,
                    lag,
                    circular_modulus=(
                        ENCODER_COUNTS_PER_REVOLUTION if signal == "encoder" else None
                    ),
                    rejection_threshold=1.0,
                )
                summaries.append(summary)
            report[signal][tendon] = {
                "valid_sample_count": len(channel_samples),
                "invalid_sample_count": invalid_count,
                "lagged_position_deltas": summaries,
                "position_unit": (
                    "Servo firmware integer position units; mechanical meaning unverified"
                    if signal == "servo"
                    else "AS5048A raw counts modulo 16384"
                ),
            }

    dlc_samples = [sample for sample in samples if sample.get("signal") == "dlc"]
    for joint in JOINTS:
        series = []
        invalid = 0
        for sample in dlc_samples:
            angles = sample.get("angles_deg")
            timestamp = _finite_number(sample.get("captured_at_monotonic_s"))
            value = _finite_number(angles.get(joint)) if isinstance(angles, Mapping) else None
            if sample.get("valid") and timestamp is not None and value is not None:
                series.append((timestamp, value))
            else:
                invalid += 1
        median = statistics.median(value for _, value in series) if series else None
        mad = (
            statistics.median(abs(value - median) for _, value in series)
            if series and median is not None
            else None
        )
        lag_summaries = [
            _delta_summary(series, lag, rejection_threshold=None)
            for lag in channel_lags("dlc", joint)
        ]
        report["dlc"][joint] = {
            "valid_sample_count": len(series),
            "invalid_sample_count": invalid,
            "median_deg": median,
            "median_absolute_deviation_deg": mad,
            "linear_drift_deg_per_min": _linear_drift_per_minute(series),
            "lagged_angle_deltas": lag_summaries,
        }

    report["dlc"]["current_q0_guard_static_rejection_estimate"] = [
        {
            "lag_s": lag,
            **_dlc_guard_summary(dlc_samples, lag),
        }
        for lag in channel_lags("dlc", "all")
    ]
    return report


def _number_or_none(value: Any) -> float | None:
    number = _finite_number(value)
    return number


def _record_capture(args: argparse.Namespace) -> int:
    if not math.isfinite(args.duration_s) or args.duration_s <= 0.0:
        raise ValueError("--duration-s must be finite and positive")
    if not math.isfinite(args.dlc_hz) or args.dlc_hz <= 0.0:
        raise ValueError("--dlc-hz must be finite and positive")
    if not math.isfinite(args.encoder_rate_hz) or args.encoder_rate_hz <= 0.0:
        raise ValueError("--encoder-rate-hz must be finite and positive")

    output_path = Path(args.output).expanduser().resolve()
    run_id = datetime.now().strftime("static_%Y%m%d_%H%M%S")
    writer = JsonlEventWriter(output_path, run_id=run_id)
    hardware = ServoHardwareSession(
        servo_ids=SERVO_IDS,
        servo_names=SERVO_NAMES_BY_ID,
        encoder_sensor_ids=tuple(
            ENCODER_SENSOR_IDS_BY_SERVO_ID[servo_id] for servo_id in SERVO_IDS
        ),
        servo_port=args.servo_port,
        encoder_port=args.encoder_port,
        baudrate=921600,
        serial_timeout_s=0.05,
        encoder_rate_hz=args.encoder_rate_hz,
        batch_encoder_reads=True,
        encoder_batch_max_lines=128,
        read_only=True,
        capture_telemetry=True,
    )
    source = RealtimeJointAngleSource(
        dlc_config_path=Path(args.dlc_config),
        camera_config_path=Path(args.camera_config),
        min_likelihood=0.3,
        temporal_outlier_enabled=False,
    )
    drain_stop = threading.Event()
    drain_thread: _TelemetryDrain | None = None
    p2p_temporal_gate = TemporalAngleGate(
        jump_threshold_deg=CURRENT_DLC_TEMPORAL_JUMP_DEG,
        max_hold_frames=CURRENT_DLC_TEMPORAL_MAX_HOLD_FRAMES,
    )
    error: str | None = None
    recording_started_at: float | None = None
    dlc_frame_count = 0
    try:
        source.open()
        hardware.connect(startup_timeout_s=5.0)
        hardware.wait_for_feedback(5.0, require_encoder=True)
        hardware.drain_telemetry_samples()
        recording_started_at = time.monotonic()
        writer.log(
            "static_telemetry_started",
            event_monotonic_s=recording_started_at,
            payload={
                "duration_s": args.duration_s,
                "dlc_rate_hz": args.dlc_hz,
                "encoder_rate_hz": args.encoder_rate_hz,
                "servo_port": args.servo_port,
                "encoder_port": args.encoder_port,
                "servo_ids": list(SERVO_IDS),
                "tendon_order": list(TENDONS),
                "servo_name_by_id": dict(zip(SERVO_IDS, SERVO_NAMES_BY_ID)),
                "encoder_sensor_id_by_servo_id": {
                    str(key): value
                    for key, value in ENCODER_SENSOR_IDS_BY_SERVO_ID.items()
                },
                "encoder_counts_per_revolution": ENCODER_COUNTS_PER_REVOLUTION,
                "servo_position_field": (
                    "firmware accumulated_pos/position integer; mechanical scale unverified"
                ),
                "servo_commands_enabled": False,
                "dlc_temporal_gate_enabled_for_capture": False,
                "p2p_temporal_gate_simulated": True,
                "p2p_q0_frame_count": CURRENT_Q0_FRAME_COUNT,
                "p2p_q0_max_spread_deg": CURRENT_Q0_MAX_SPREAD_DEG,
                "p2p_dlc_temporal_jump_deg": CURRENT_DLC_TEMPORAL_JUMP_DEG,
                "p2p_dlc_temporal_max_hold_frames": CURRENT_DLC_TEMPORAL_MAX_HOLD_FRAMES,
                "timestamp_timebase": "host monotonic seconds; device_ms retained when available",
            },
        )
        drain_thread = _TelemetryDrain(hardware, writer, drain_stop)
        drain_thread.start()
        frame_index = 0
        period_s = 1.0 / args.dlc_hz
        deadline = recording_started_at + args.duration_s
        while time.monotonic() < deadline:
            loop_started = time.monotonic()
            measurement = source.read(frame_index)
            angles_deg = {
                joint: _number_or_none(math.degrees(angle))
                for joint, angle in zip(JOINTS, measurement.flexion_angles_rad)
            }
            if measurement.valid:
                p2p_decision = p2p_temporal_gate.apply(
                    measurement.flexion_angles_rad
                )
                p2p_angles_rad = p2p_decision.angles_rad
                p2p_valid = p2p_decision.valid
                p2p_held_last = p2p_decision.held_last
                p2p_reason = p2p_decision.reason
            else:
                p2p_angles_rad = measurement.flexion_angles_rad
                p2p_valid = False
                p2p_held_last = False
                p2p_reason = measurement.reason
            writer.log(
                "static_telemetry_sample",
                event_monotonic_s=measurement.inference_finished_at,
                payload={
                    "signal": "dlc",
                    "channel_id": "DIP_PIP_MCP",
                    "frame_index": measurement.frame_index,
                    "captured_at_monotonic_s": measurement.captured_at,
                    "inference_finished_at_monotonic_s": measurement.inference_finished_at,
                    "angles_deg": angles_deg,
                    "angles_rad": {
                        joint: _number_or_none(angle)
                        for joint, angle in zip(JOINTS, measurement.flexion_angles_rad)
                    },
                    "p2p_angles_deg": {
                        joint: _number_or_none(math.degrees(angle))
                        for joint, angle in zip(JOINTS, p2p_angles_rad)
                    },
                    "p2p_valid": bool(p2p_valid),
                    "p2p_held_last": bool(p2p_held_last),
                    "p2p_reason": p2p_reason,
                    "min_likelihood": measurement.min_likelihood,
                    "inference_ms": measurement.inference_ms,
                    "valid": bool(measurement.valid),
                    "held_last": bool(measurement.held_last),
                    "reason": measurement.reason,
                },
            )
            frame_index += 1
            dlc_frame_count += 1
            time.sleep(max(0.0, period_s - (time.monotonic() - loop_started)))
    except KeyboardInterrupt:
        error = "KeyboardInterrupt"
    except Exception as exc:
        error = f"{type(exc).__name__}: {exc}"
    finally:
        try:
            source.close()
        except Exception as exc:
            error = error or f"DLC close: {type(exc).__name__}: {exc}"
        hardware.close()
        drain_stop.set()
        if drain_thread is not None:
            drain_thread.join(timeout=5.0)
            if drain_thread.is_alive():
                error = error or "Telemetry drain thread did not stop"
            else:
                drain_thread._drain()
                if drain_thread.error:
                    error = error or drain_thread.error
        stats = hardware.telemetry_capture_stats()
        writer.log(
            "static_telemetry_finished",
            event_monotonic_s=time.monotonic(),
            payload={
                "recording_started_monotonic_s": recording_started_at,
                "elapsed_s": (
                    None
                    if recording_started_at is None
                    else time.monotonic() - recording_started_at
                ),
                "dlc_frame_count": dlc_frame_count,
                "hardware_samples": stats,
                "drained_hardware_records": (
                    {} if drain_thread is None else drain_thread.counts
                ),
                "error": error,
            },
        )
        writer.close()
    if error:
        print(f"Static capture ended with error: {error}", file=sys.stderr)
        return 1
    print(f"Saved read-only telemetry: {output_path}")
    return 0


def _analyze(args: argparse.Namespace) -> int:
    if not args.lags_s and not args.p2p_events:
        raise ValueError("Provide --lags-s or --p2p-events with P2P alignment logs")
    report = analyze_static_telemetry(
        Path(args.input),
        lags_s=args.lags_s or (),
        p2p_event_paths=[Path(path) for path in args.p2p_events or ()],
    )
    rendered = json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False)
    if args.output:
        Path(args.output).write_text(rendered + "\n", encoding="utf-8")
        print(f"Saved analysis: {Path(args.output).resolve()}")
    else:
        print(rendered)
    return 0


def _build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description=__doc__)
    subparsers = parser.add_subparsers(dest="command", required=True)
    record = subparsers.add_parser("record", help="capture static telemetry without Servo commands")
    record.add_argument("--output", required=True, help="new JSONL output file")
    record.add_argument("--duration-s", required=True, type=float)
    record.add_argument("--servo-port", default=DEFAULT_SERVO_PORT)
    record.add_argument("--encoder-port", default=DEFAULT_ENCODER_PORT)
    record.add_argument("--dlc-config", type=Path, default=DEFAULT_DLC_CONFIG_PATH)
    record.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA_CONFIG_PATH)
    record.add_argument("--encoder-rate-hz", type=float, default=35.0)
    record.add_argument("--dlc-hz", type=float, default=5.0)
    analyze = subparsers.add_parser("analyze", help="summarize static noise and guard rejection rates")
    analyze.add_argument("--input", required=True, help="static telemetry JSONL")
    analyze.add_argument("--lags-s", type=float, nargs="*", default=())
    analyze.add_argument("--p2p-events", nargs="*", default=())
    analyze.add_argument("--output", help="optional JSON report path")
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    args = _build_parser().parse_args(argv)
    if args.command == "record":
        return _record_capture(args)
    if args.command == "analyze":
        return _analyze(args)
    raise ValueError(f"Unsupported command: {args.command}")


if __name__ == "__main__":  # pragma: no cover - exercised through the CLI
    raise SystemExit(main())
