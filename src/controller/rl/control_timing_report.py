"""Build a component timing budget from one raw CSV-player session."""

from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Any, Iterable, Mapping


def _finite(values: Iterable[object]) -> list[float]:
    result: list[float] = []
    for raw in values:
        try:
            value = float(raw)
        except (TypeError, ValueError):
            continue
        if math.isfinite(value):
            result.append(value)
    return result


def _percentile(sorted_values: list[float], percentile: float) -> float | None:
    if not sorted_values:
        return None
    if len(sorted_values) == 1:
        return sorted_values[0]
    rank = (len(sorted_values) - 1) * percentile / 100.0
    lower = int(math.floor(rank))
    upper = int(math.ceil(rank))
    if lower == upper:
        return sorted_values[lower]
    fraction = rank - lower
    return sorted_values[lower] * (1.0 - fraction) + sorted_values[upper] * fraction


def _stats(values: Iterable[object], *, scale: float = 1.0) -> dict[str, Any]:
    finite = sorted(value * scale for value in _finite(values))
    if not finite:
        return {"count": 0, "mean": None, "p50": None, "p95": None, "max": None}
    return {
        "count": len(finite),
        "mean": sum(finite) / len(finite),
        "p50": _percentile(finite, 50.0),
        "p95": _percentile(finite, 95.0),
        "max": finite[-1],
    }


def _load_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            event = json.loads(text)
            if not isinstance(event, dict):
                raise ValueError(f"{path}:{line_number} is not an object")
            events.append(event)
    return events


def build_control_timing_report(session_dir: str | Path) -> dict[str, Any]:
    root = Path(session_dir)
    events = _load_events(root / "raw_events.jsonl")
    by_type: dict[str, list[Mapping[str, Any]]] = {}
    for event in events:
        by_type.setdefault(str(event.get("event_type", "")), []).append(event)

    ticks = by_type.get("pid_tick", [])
    writes = by_type.get("transport_write", [])
    visions = by_type.get("vision_observation", [])
    tick_payloads = [event.get("payload", {}) for event in ticks]
    write_payloads = [event.get("payload", {}) for event in writes]
    vision_payloads = [event.get("payload", {}) for event in visions]

    encoder_ages = []
    stage_values: dict[str, list[object]] = {}
    for payload in tick_payloads:
        tendons = payload.get("tendons", {})
        if isinstance(tendons, Mapping):
            encoder_ages.extend(
                item.get("encoder_age_s")
                for item in tendons.values()
                if isinstance(item, Mapping)
            )
        stages = payload.get("stage_durations_s", {})
        if isinstance(stages, Mapping):
            for name, value in stages.items():
                stage_values.setdefault(str(name), []).append(value)

    queue_delays = []
    write_durations = []
    submit_to_finish = []
    transport_status: dict[str, int] = {}
    for payload in write_payloads:
        status = str(payload.get("transport_status", "unknown"))
        transport_status[status] = transport_status.get(status, 0) + 1
        submitted = payload.get("dispatcher_submitted_monotonic_s")
        started = payload.get("transport_write_started_monotonic_s")
        finished = payload.get("transport_write_finished_monotonic_s")
        try:
            if submitted is not None and started is not None:
                queue_delays.append(float(started) - float(submitted))
            if started is not None and finished is not None:
                write_durations.append(float(finished) - float(started))
            if submitted is not None and finished is not None:
                submit_to_finish.append(float(finished) - float(submitted))
        except (TypeError, ValueError):
            continue

    vision_capture_times = _finite(
        payload.get("capture_monotonic_s") for payload in vision_payloads
    )
    vision_capture_times.sort()
    vision_periods = [
        current - previous
        for previous, current in zip(vision_capture_times, vision_capture_times[1:])
    ]

    manifest_path = root / "session_manifest.json"
    manifest = _load_json(manifest_path) if manifest_path.exists() else {}
    raw_stats_path = root / "raw_event_stats.json"
    raw_stats = (
        _load_json(raw_stats_path)
        if raw_stats_path.exists()
        else manifest.get("raw_logger_stats")
    )
    encoder_path = root / "encoder_diagnostics.json"
    encoder_diagnostics = _load_json(encoder_path) if encoder_path.exists() else None

    sensor_stats = {}
    if isinstance(encoder_diagnostics, Mapping):
        raw_sensor_stats = encoder_diagnostics.get("sensor_stats", {})
        if isinstance(raw_sensor_stats, Mapping):
            sensor_stats = {
                str(sensor_id): {
                    "sample_count": stats.get("sample_count"),
                    "max_receive_gap_ms": (
                        None
                        if stats.get("max_receive_gap_s") is None
                        else float(stats["max_receive_gap_s"]) * 1000.0
                    ),
                    "last_status": stats.get("last_status"),
                }
                for sensor_id, stats in raw_sensor_stats.items()
                if isinstance(stats, Mapping)
            }

    configured_control_hz = manifest.get("control_hz")
    expected_period_ms = (
        None
        if configured_control_hz in (None, 0)
        else 1000.0 / float(configured_control_hz)
    )
    control_period = _stats(
        (payload.get("dt_s") for payload in tick_payloads),
        scale=1000.0,
    )
    measured_hz = (
        None
        if control_period["mean"] in (None, 0)
        else 1000.0 / float(control_period["mean"])
    )
    bottleneck_candidates = {
        "control_period_p95_ms": control_period["p95"],
        "servo_feedback_age_p95_ms": _stats(
            (payload.get("servo_feedback_age_s") for payload in tick_payloads),
            scale=1000.0,
        )["p95"],
        "encoder_age_p95_ms": _stats(encoder_ages, scale=1000.0)["p95"],
        "dispatcher_queue_p95_ms": _stats(queue_delays, scale=1000.0)["p95"],
        "transport_write_p95_ms": _stats(write_durations, scale=1000.0)["p95"],
        "vision_inference_p95_ms": _stats(
            (payload.get("inference_ms") for payload in vision_payloads)
        )["p95"],
        "logger_max_write_lag_ms": (
            None
            if not isinstance(raw_stats, Mapping)
            or raw_stats.get("max_write_lag_s") is None
            else float(raw_stats["max_write_lag_s"]) * 1000.0
        ),
        **{
            f"stage_{name}_p95_ms": _stats(values, scale=1000.0)["p95"]
            for name, values in stage_values.items()
        },
    }
    ranked = sorted(
        (
            (name, float(value))
            for name, value in bottleneck_candidates.items()
            if value is not None and math.isfinite(float(value))
        ),
        key=lambda item: item[1],
        reverse=True,
    )
    return {
        "schema_version": 1,
        "session_dir": str(root.resolve()),
        "configured_control_hz": configured_control_hz,
        "expected_control_period_ms": expected_period_ms,
        "measured_control_hz": measured_hz,
        "control_period_ms": control_period,
        "tick_lateness_ms": _stats(
            (payload.get("tick_lateness_s") for payload in tick_payloads),
            scale=1000.0,
        ),
        "servo_feedback_age_ms": _stats(
            (payload.get("servo_feedback_age_s") for payload in tick_payloads),
            scale=1000.0,
        ),
        "encoder_age_ms": _stats(encoder_ages, scale=1000.0),
        "control_stage_duration_ms": {
            name: _stats(values, scale=1000.0)
            for name, values in sorted(stage_values.items())
        },
        "dispatcher_queue_ms": _stats(queue_delays, scale=1000.0),
        "transport_write_ms": _stats(write_durations, scale=1000.0),
        "dispatcher_submit_to_write_finish_ms": _stats(
            submit_to_finish,
            scale=1000.0,
        ),
        "transport_status_counts": transport_status,
        "vision_inference_ms": _stats(
            (payload.get("inference_ms") for payload in vision_payloads)
        ),
        "vision_capture_period_ms": _stats(vision_periods, scale=1000.0),
        "raw_logger_stats": raw_stats,
        "encoder_sensor_stats": sensor_stats,
        "bottleneck_ranking": [
            {"metric": name, "value_ms": value} for name, value in ranked
        ],
        "interpretation": (
            "The ranking identifies large observed latencies; it does not by itself "
            "prove which component blocked the control thread."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Summarize control-loop timing from one session directory."
    )
    parser.add_argument("session_dir", type=Path)
    parser.add_argument("--output", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = build_control_timing_report(args.session_dir)
    output = args.output or args.session_dir / "control_timing_report.json"
    output.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(output)


if __name__ == "__main__":
    main()
