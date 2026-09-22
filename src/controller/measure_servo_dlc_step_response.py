"""Measure one safe servo step and its DLC-observed joint response.

The command is intentionally single-trial.  It records raw telemetry and
vision timelines so latency thresholds can be changed and the result
recomputed without moving the robot again.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import queue
import sys
import threading
import time
import uuid
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Sequence

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from controller.servo_mapping import SERVO_IDS_BY_TENDON, TENDONS
from controller.step_response_analysis import (
    TimedScalar,
    TimedServoSample,
    detect_joint_settling,
    detect_onset,
    detect_servo_settling,
    robust_noise,
    threshold_from_noise,
)
from observation.vision.realtime_joint_angle_source import (
    JOINTS,
    JointAngleMeasurement,
    RealtimeJointAngleSource,
)
from servo.control import (
    PositionControlConfig,
    ReliablePositionController,
    TelemetryMonitor,
    TelemetrySnapshot,
)
from utils.config_loader import load_config
from utils.path_utils import resolve_path


PROJECT_ROOT = SRC_ROOT.parent
DEFAULT_CONFIG_PATH = SRC_ROOT / "controller" / "config_dlc_excursion_feedback.json"
SERVO_IDS = SERVO_IDS_BY_TENDON


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Measure one servo step and its DLC joint response."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument("--tendon", choices=TENDONS, required=True)
    parser.add_argument("--position-step-count", type=int, required=True)
    parser.add_argument(
        "--measurement-max-step-count",
        type=int,
        default=50,
        help="Hard upper bound for this measurement, before the existing per-tendon limit.",
    )
    parser.add_argument(
        "--movement-direction",
        choices=("increase", "decrease"),
        required=True,
    )
    parser.add_argument("--observation-timeout-s", type=float)
    parser.add_argument("--baseline-sample-count", type=int)
    parser.add_argument("--servo-onset-threshold-count", type=float)
    parser.add_argument("--joint-onset-threshold-deg", type=float)
    parser.add_argument("--consecutive-detection-count", type=int)
    parser.add_argument("--noise-scale", type=float)
    parser.add_argument("--servo-settling-band-count", type=float)
    parser.add_argument("--servo-speed-threshold", type=float)
    parser.add_argument("--joint-settling-band-deg", type=float)
    parser.add_argument("--joint-settling-slope-deg", type=float)
    parser.add_argument("--settling-consecutive-count", type=int)
    parser.add_argument("--settling-terminal-window", type=int)
    parser.add_argument(
        "--return-to-start",
        dest="return_to_start",
        action="store_true",
        default=None,
    )
    parser.add_argument(
        "--no-return-to-start",
        dest="return_to_start",
        action="store_false",
    )
    parser.add_argument(
        "--output-directory",
        default=None,
        help="Override paths.log_directory/step_response.",
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Send one command to the physical servo. Without this flag, dry-run only.",
    )
    return parser.parse_args(argv)


def _project_path(raw_path: str, option_name: str) -> Path:
    path = resolve_path(raw_path, PROJECT_ROOT)
    if path is None:
        raise ValueError(f"{option_name} must be a non-empty path")
    return path


def _ordered_ints(mapping: dict, names: Sequence[str], option_name: str) -> tuple[int, ...]:
    missing = [name for name in names if name not in mapping]
    if missing:
        raise ValueError(f"{option_name} is missing values for {missing}")
    return tuple(int(mapping[name]) for name in names)


def _position_limits(raw_limits: dict) -> tuple[tuple[int, int], ...]:
    result: list[tuple[int, int]] = []
    for tendon in TENDONS:
        values = raw_limits.get(tendon)
        if not isinstance(values, list) or len(values) != 2:
            raise ValueError(f"position_limits.{tendon} must contain [min, max]")
        result.append((int(values[0]), int(values[1])))
    return tuple(result)


def resolve_analysis_args(
    args: argparse.Namespace,
    config: dict,
) -> argparse.Namespace:
    """Fill omitted analysis options from the step-response configuration."""

    step_response = config.get("step_response")
    if not isinstance(step_response, dict):
        raise ValueError("step_response must be an object")
    detection = step_response.get("detection")
    settling = step_response.get("settling")
    if not isinstance(detection, dict) or not isinstance(settling, dict):
        raise ValueError("step_response detection and settling must be objects")

    defaults = {
        "observation_timeout_s": step_response.get("observation_timeout_s"),
        "baseline_sample_count": step_response.get("baseline_sample_count"),
        "servo_onset_threshold_count": detection.get(
            "servo_onset_threshold_count"
        ),
        "joint_onset_threshold_deg": detection.get("joint_onset_threshold_deg"),
        "consecutive_detection_count": detection.get("consecutive_count"),
        "noise_scale": detection.get("noise_scale"),
        "servo_settling_band_count": settling.get(
            "servo_position_band_count"
        ),
        "servo_speed_threshold": settling.get("servo_speed_threshold"),
        "joint_settling_band_deg": settling.get("joint_position_band_deg"),
        "joint_settling_slope_deg": settling.get(
            "joint_slope_threshold_deg"
        ),
        "settling_consecutive_count": settling.get("consecutive_count"),
        "settling_terminal_window": settling.get("terminal_window"),
    }
    missing = [name for name, value in defaults.items() if value is None]
    if missing:
        raise ValueError(f"step_response is missing values for {missing}")
    for name, value in defaults.items():
        if getattr(args, name, None) is None:
            setattr(args, name, value)
    return args


def _make_angle_source(config: dict) -> RealtimeJointAngleSource:
    paths = config["paths"]
    vision = config["vision"]
    temporal_outlier = vision.get("temporal_outlier", {})
    if not isinstance(temporal_outlier, dict):
        raise ValueError("vision.temporal_outlier must be an object")
    return RealtimeJointAngleSource(
        dlc_config_path=_project_path(paths["dlc_config"], "paths.dlc_config"),
        camera_config_path=_project_path(paths["camera_config"], "paths.camera_config"),
        min_likelihood=float(vision["min_likelihood"]),
        temporal_outlier_enabled=bool(temporal_outlier.get("enabled", False)),
        temporal_jump_threshold_deg=float(temporal_outlier.get("base_jump_deg", 10.0)),
        temporal_max_hold_frames=int(temporal_outlier.get("max_hold_frames", 1)),
    )


def _make_position_controller(
    api,
    telemetry: TelemetryMonitor,
    config: dict,
    position_limits: Sequence[tuple[int, int]],
) -> ReliablePositionController:
    hardware = config["hardware"]
    safety = config["safety"]
    return ReliablePositionController(
        api,
        telemetry,
        PositionControlConfig(
            telemetry_stale_s=float(safety["telemetry_stale_s"]),
            telemetry_wait_s=float(hardware["telemetry_wait_s"]),
            id_map_reset_wait_s=0.1,
            speed_init_wait_s=float(hardware["prepare_wait_s"]),
            prime_command_count=int(hardware["prime_command_count"]),
            prime_interval_s=float(hardware["prime_wait_s"]),
            start_observation_s=0.3,
            start_min_delta=3,
            position_tolerance=int(hardware["return_tolerance"]),
            speed_tolerance=5,
            stable_frame_count=3,
            arrival_timeout_s=float(hardware["telemetry_wait_s"]),
            max_start_retries=1,
            reset_id_map_on_prepare=True,
            multiturn=bool(hardware["multiturn"]),
            position_min=min(value[0] for value in position_limits),
            position_max=max(value[1] for value in position_limits),
        ),
    )


class TelemetryRecorder:
    """Drain a TelemetryMonitor subscription without reading the serial port."""

    def __init__(self, telemetry: TelemetryMonitor) -> None:
        self._telemetry = telemetry
        self._subscription: queue.Queue[TelemetrySnapshot | None] | None = None
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self.records: list[TelemetrySnapshot] = []
        self.error: Exception | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("TelemetryRecorder has already been started")
        self._subscription = self._telemetry.subscribe()
        self._thread = threading.Thread(
            target=self._run,
            name="step-response-telemetry-recorder",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        subscription = self._subscription
        if subscription is None:
            return
        self._telemetry.unsubscribe(subscription)
        subscription.put(None)
        if self._thread is not None:
            self._thread.join(timeout=2.0)
        self._subscription = None

    def _run(self) -> None:
        assert self._subscription is not None
        try:
            while not self._stop.is_set():
                snapshot = self._subscription.get()
                if snapshot is None:
                    return
                self.records.append(snapshot)
        except Exception as exc:
            self.error = exc


def _measurement_to_row(measurement: JointAngleMeasurement, phase: str) -> dict[str, object]:
    return {
        "phase": phase,
        "frame_index": measurement.frame_index,
        "captured_at": measurement.captured_at,
        "inference_finished_at": measurement.inference_finished_at,
        "inference_ms": measurement.inference_ms,
        "DIP_angle_rad": measurement.flexion_angles_rad[0],
        "PIP_angle_rad": measurement.flexion_angles_rad[1],
        "MCP_angle_rad": measurement.flexion_angles_rad[2],
        "min_likelihood": measurement.min_likelihood,
        "valid": int(measurement.valid),
        "held_last": int(measurement.held_last),
        "reason": measurement.reason or "",
    }


def _measurement_is_usable(measurement: JointAngleMeasurement) -> bool:
    return bool(measurement.valid and not measurement.held_last)


def _read_baseline(
    angle_source: RealtimeJointAngleSource,
    *,
    sample_count: int,
    timeout_s: float,
    start_frame_index: int,
) -> tuple[list[JointAngleMeasurement], int]:
    deadline = time.monotonic() + timeout_s
    measurements: list[JointAngleMeasurement] = []
    frame_index = start_frame_index
    while len([m for m in measurements if _measurement_is_usable(m)]) < sample_count:
        if time.monotonic() >= deadline:
            raise TimeoutError("Did not obtain enough valid DLC baseline samples")
        measurement = angle_source.read(frame_index)
        measurements.append(measurement)
        frame_index += 1
    return measurements, frame_index


def _return_to_start(
    controller: ReliablePositionController,
    telemetry: TelemetryMonitor,
    start_positions: Sequence[int],
    config: dict,
) -> float:
    hardware = config["hardware"]
    safety = config["safety"]
    return_requested_at = time.monotonic()
    controller.stream_positions(
        SERVO_IDS,
        start_positions,
        time_ms=int(hardware["return_time_ms"]),
    )
    deadline = return_requested_at + float(hardware["return_timeout_s"])
    tolerance = int(hardware["return_tolerance"])
    sequence = 0
    latest = telemetry.latest(float(safety["telemetry_stale_s"]))
    if latest is not None:
        sequence = latest.sequence
    while time.monotonic() < deadline:
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0.0:
            break
        try:
            latest = telemetry.wait_for_newer(sequence, min(0.25, remaining_s))
        except TimeoutError:
            continue
        sequence = latest.sequence
        if all(
            abs(target - latest.positions[servo_id]) <= tolerance
            for servo_id, target in zip(SERVO_IDS, start_positions)
        ):
            return return_requested_at
    raise TimeoutError("Servos did not return to the initial positions before timeout")


def _phase_for_timestamp(
    timestamp: float,
    *,
    command_requested_at: float | None,
    return_requested_at: float | None,
) -> str:
    if return_requested_at is not None and timestamp >= return_requested_at:
        return "return"
    if command_requested_at is not None and timestamp >= command_requested_at:
        return "response"
    return "baseline"


def _latency_ms(at: float | None, commanded_at: float | None) -> float | None:
    if at is None or commanded_at is None:
        return None
    return (float(at) - float(commanded_at)) * 1000.0


def _finite_median(values: Sequence[float]) -> float | None:
    finite = [float(value) for value in values if math.isfinite(float(value))]
    return None if not finite else float(median(finite))


def analyse_step_response(
    *,
    args: argparse.Namespace,
    measurements: Sequence[JointAngleMeasurement],
    telemetry_records: Sequence[TelemetrySnapshot],
    commanded_at: float | None,
    command_requested_at: float | None,
    return_requested_at: float | None,
    servo_id: int,
    start_position: int | None,
    target_position: int | None,
) -> dict[str, object]:
    baseline_measurements = [
        m for m in measurements if command_requested_at is None or m.captured_at < command_requested_at
    ]
    response_measurements = [
        m
        for m in measurements
        if commanded_at is not None
        and m.captured_at > commanded_at
        and (return_requested_at is None or m.captured_at < return_requested_at)
    ]
    baseline_telemetry = [
        s for s in telemetry_records
        if command_requested_at is None or s.received_at < command_requested_at
    ]
    response_telemetry = [
        s for s in telemetry_records
        if commanded_at is not None
        and s.received_at > commanded_at
        and (return_requested_at is None or s.received_at < return_requested_at)
    ]

    baseline_position_values = [s.positions[servo_id] for s in baseline_telemetry]
    baseline_position = (
        float(start_position)
        if start_position is not None
        else _finite_median(baseline_position_values)
    )
    servo_noise = robust_noise(baseline_position_values)
    servo_threshold = threshold_from_noise(
        args.servo_onset_threshold_count,
        servo_noise,
        args.noise_scale,
    )
    servo_samples = [
        TimedScalar(s.received_at, s.positions[servo_id])
        for s in response_telemetry
    ]
    direction = 1 if args.movement_direction == "increase" else -1
    servo_onset = None
    if baseline_position is not None:
        servo_onset = detect_onset(
            servo_samples,
            baseline=baseline_position,
            threshold=servo_threshold,
            consecutive_count=args.consecutive_detection_count,
            direction=direction,
        )

    servo_settling = None
    if target_position is not None:
        servo_settling = detect_servo_settling(
            [
                TimedServoSample(
                    s.received_at,
                    s.positions[servo_id],
                    None if s.speeds is None else s.speeds[servo_id],
                )
                for s in response_telemetry
            ],
            onset_at=None if servo_onset is None else servo_onset.first_at,
            target_position=target_position,
            position_band=args.servo_settling_band_count,
            speed_threshold=args.servo_speed_threshold,
            consecutive_count=args.settling_consecutive_count,
        )

    joint_onsets: dict[str, object] = {}
    joint_settlings: dict[str, object] = {}
    joint_noise: dict[str, float] = {}
    joint_thresholds: dict[str, float] = {}
    joint_onset_times: list[float] = []
    for index, joint in enumerate(JOINTS):
        baseline_values = [
            m.flexion_angles_rad[index]
            for m in baseline_measurements
            if _measurement_is_usable(m)
        ]
        baseline_angle = _finite_median(baseline_values)
        noise = robust_noise(baseline_values)
        threshold = threshold_from_noise(
            math.radians(args.joint_onset_threshold_deg),
            noise,
            args.noise_scale,
        )
        joint_noise[joint] = noise
        joint_thresholds[joint] = threshold
        joint_response_measurements = [
            m for m in response_measurements if _measurement_is_usable(m)
        ]
        samples = [
            TimedScalar(m.captured_at, m.flexion_angles_rad[index])
            for m in joint_response_measurements
        ]
        onset = None
        if baseline_angle is not None:
            onset = detect_onset(
                samples,
                baseline=baseline_angle,
                threshold=threshold,
                consecutive_count=args.consecutive_detection_count,
            )
        if onset is not None:
            joint_onset_times.append(onset.first_at)
            joint_onsets[joint] = {
                "first_at": onset.first_at,
                "confirmed_at": onset.confirmed_at,
                "onset_latency_ms": _latency_ms(onset.first_at, commanded_at),
                "available_latency_ms": _latency_ms(
                    joint_response_measurements[onset.first_index].inference_finished_at,
                    commanded_at,
                ),
                "threshold_rad": threshold,
            }
        else:
            joint_onsets[joint] = None

        settling = detect_joint_settling(
            samples,
            onset_at=None if onset is None else onset.first_at,
            terminal_window=args.settling_terminal_window,
            position_band=math.radians(args.joint_settling_band_deg),
            slope_threshold=math.radians(args.joint_settling_slope_deg),
            consecutive_count=args.settling_consecutive_count,
        )
        if settling is not None:
            joint_settlings[joint] = {
                "first_at": settling.first_at,
                "confirmed_at": settling.confirmed_at,
                "settling_time_ms": _latency_ms(settling.first_at, commanded_at),
            }
        else:
            joint_settlings[joint] = None

    first_joint_onset = min(joint_onset_times) if joint_onset_times else None
    first_joint_measurement = next(
        (m for m in response_measurements if m.captured_at == first_joint_onset),
        None,
    )
    processing_latencies = [
        (m.inference_finished_at - m.captured_at) * 1000.0
        for m in response_measurements
        if math.isfinite(m.inference_finished_at - m.captured_at)
    ]
    rejection_reasons: list[str] = []
    if servo_onset is None:
        rejection_reasons.append("servo_onset_not_detected")
    if first_joint_onset is None:
        rejection_reasons.append("joint_onset_not_detected")
    if servo_settling is None:
        rejection_reasons.append("servo_settling_not_detected")
    if any(value is None for value in joint_settlings.values()):
        rejection_reasons.append("joint_settling_not_detected")

    return {
        "servo_motion_started_at": None if servo_onset is None else servo_onset.first_at,
        "joint_motion_observed_at": first_joint_onset,
        "servo_settled_at": None if servo_settling is None else servo_settling.first_at,
        "joint_settled_at": (
            min(
                value["first_at"]
                for value in joint_settlings.values()
                if value is not None
            )
            if any(value is not None for value in joint_settlings.values())
            else None
        ),
        "servo_onset_latency_ms": None if servo_onset is None else _latency_ms(servo_onset.first_at, commanded_at),
        "joint_onset_latency_ms": _latency_ms(first_joint_onset, commanded_at),
        "vision_processing_latency_ms": _finite_median(processing_latencies),
        "vision_processing_p95_ms": (
            sorted(processing_latencies)[int(0.95 * (len(processing_latencies) - 1))]
            if processing_latencies
            else None
        ),
        "joint_available_latency_ms": (
            None
            if first_joint_measurement is None
            else _latency_ms(first_joint_measurement.inference_finished_at, commanded_at)
        ),
        "servo_settling_time_ms": (
            None
            if servo_settling is None
            else _latency_ms(servo_settling.first_at, commanded_at)
        ),
        "joint_settling_time_ms": (
            min(value["settling_time_ms"] for value in joint_settlings.values() if value is not None)
            if any(value is not None for value in joint_settlings.values())
            else None
        ),
        "onset_detected": bool(servo_onset is not None and first_joint_onset is not None),
        "settling_detected": bool(
            servo_settling is not None
            and all(value is not None for value in joint_settlings.values())
        ),
        "rejection_reason": ";".join(rejection_reasons) or None,
        "servo_noise_count": servo_noise,
        "servo_effective_threshold_count": servo_threshold,
        "joint_noise_rad": joint_noise,
        "joint_effective_threshold_rad": joint_thresholds,
        "joint_onsets": joint_onsets,
        "joint_settlings": joint_settlings,
        "response_telemetry_count": len(response_telemetry),
        "response_vision_count": len(response_measurements),
    }


def _save_trial(
    trial_directory: Path,
    metadata: dict[str, object],
    telemetry_records: Sequence[TelemetrySnapshot],
    measurements: Sequence[JointAngleMeasurement],
    *,
    command_requested_at: float | None,
    return_requested_at: float | None,
    servo_id: int,
    target_position: int | None,
) -> None:
    trial_directory.mkdir(parents=True, exist_ok=True)
    telemetry_path = trial_directory / "telemetry.csv"
    with telemetry_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = [
            "phase", "sequence", "received_at", "firmware_timestamp_ms",
            "target_position", "position", "speed", "load",
            "positions_json", "speeds_json", "loads_json",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for snapshot in telemetry_records:
            speed = "" if snapshot.speeds is None else snapshot.speeds[servo_id]
            writer.writerow(
                {
                    "phase": _phase_for_timestamp(
                        snapshot.received_at,
                        command_requested_at=command_requested_at,
                        return_requested_at=return_requested_at,
                    ),
                    "sequence": snapshot.sequence,
                    "received_at": snapshot.received_at,
                    "firmware_timestamp_ms": snapshot.timestamp_ms,
                    "target_position": (
                        "" if target_position is None else target_position
                    ),
                    "position": snapshot.positions[servo_id],
                    "speed": speed,
                    "load": snapshot.loads[servo_id],
                    "positions_json": json.dumps(snapshot.positions),
                    "speeds_json": "" if snapshot.speeds is None else json.dumps(snapshot.speeds),
                    "loads_json": json.dumps(snapshot.loads),
                }
            )

    vision_path = trial_directory / "vision.csv"
    with vision_path.open("w", newline="", encoding="utf-8") as handle:
        fieldnames = list(_measurement_to_row(measurements[0], "baseline")) if measurements else [
            "phase", "frame_index", "captured_at", "inference_finished_at",
            "inference_ms", "DIP_angle_rad", "PIP_angle_rad", "MCP_angle_rad",
            "min_likelihood", "valid", "held_last", "reason",
        ]
        writer = csv.DictWriter(handle, fieldnames=fieldnames)
        writer.writeheader()
        for measurement in measurements:
            writer.writerow(
                _measurement_to_row(
                    measurement,
                    _phase_for_timestamp(
                        measurement.captured_at,
                        command_requested_at=command_requested_at,
                        return_requested_at=return_requested_at,
                    ),
                )
            )

    (trial_directory / "trial.json").write_text(
        json.dumps(metadata, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )


def _validate_measurement_args(args: argparse.Namespace) -> None:
    if args.position_step_count <= 0:
        raise ValueError("--position-step-count must be positive")
    if args.measurement_max_step_count <= 0:
        raise ValueError("--measurement-max-step-count must be positive")
    if args.position_step_count > args.measurement_max_step_count:
        raise ValueError(
            "--position-step-count must not exceed --measurement-max-step-count"
        )
    if args.observation_timeout_s <= 0.0:
        raise ValueError("--observation-timeout-s must be positive")
    if args.baseline_sample_count <= 0:
        raise ValueError("--baseline-sample-count must be positive")
    if args.consecutive_detection_count <= 0 or args.settling_consecutive_count <= 0:
        raise ValueError("consecutive counts must be positive")
    if args.settling_terminal_window <= 0:
        raise ValueError("--settling-terminal-window must be positive")
    non_negative = (
        "servo_onset_threshold_count", "joint_onset_threshold_deg", "noise_scale",
        "servo_settling_band_count", "servo_speed_threshold",
        "joint_settling_band_deg", "joint_settling_slope_deg",
    )
    for name in non_negative:
        if not math.isfinite(float(getattr(args, name))) or float(getattr(args, name)) < 0.0:
            raise ValueError(f"--{name.replace('_', '-')} must be finite and non-negative")


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(
        config_path,
        required_keys=("control", "paths", "vision", "safety", "hardware"),
    )
    resolve_analysis_args(args, config)
    _validate_measurement_args(args)
    tendon_index = TENDONS.index(args.tendon)
    servo_id = SERVO_IDS[tendon_index]
    position_limits = _position_limits(config["safety"]["position_limits"])
    max_position_steps = _ordered_ints(
        config["safety"]["max_position_step"],
        TENDONS,
        "safety.max_position_step",
    )
    if args.position_step_count > max_position_steps[tendon_index]:
        raise ValueError(
            f"position step {args.position_step_count} exceeds "
            f"{args.tendon} max_position_step {max_position_steps[tendon_index]}"
        )
    direction = 1 if args.movement_direction == "increase" else -1
    dry_positions = tuple(int(value) for value in config["hardware"]["dry_run_start_positions"])
    if len(dry_positions) != len(TENDONS):
        raise ValueError("hardware.dry_run_start_positions must contain six values")
    dry_target = dry_positions[servo_id] + direction * args.position_step_count
    if not position_limits[tendon_index][0] <= dry_target <= position_limits[tendon_index][1]:
        raise ValueError(f"dry-run target {dry_target} is outside {args.tendon} position limits")

    base_output = (
        Path(args.output_directory).expanduser().resolve()
        if args.output_directory
        else _project_path(config["paths"]["log_directory"], "paths.log_directory") / "step_response"
    )
    trial_id = (
        datetime.now().strftime("%Y%m%d_%H%M%S")
        + f"_{args.tendon}_{args.movement_direction}_{uuid.uuid4().hex[:8]}"
    )
    trial_directory = base_output / trial_id
    return_to_start = (
        bool(config["hardware"].get("return_to_start", False))
        if args.return_to_start is None
        else bool(args.return_to_start)
    )
    metadata: dict[str, object] = {
        "schema_version": 1,
        "trial_id": trial_id,
        "tendon": args.tendon,
        "servo_id": servo_id,
        "movement_direction": args.movement_direction,
        "position_step_count": args.position_step_count,
        "start_position": None,
        "target_position": None,
        "command_requested_at": None,
        "commanded_at": None,
        "return_requested_at": None,
        "return_to_start": return_to_start,
        "config_path": str(config_path),
        "config": config,
        "measurement_args": vars(args),
        "completion_reason": None,
        "analysis": None,
    }
    if not args.execute:
        metadata["start_position"] = dry_positions[servo_id]
        metadata["target_position"] = dry_target
        metadata["completion_reason"] = "dry_run"
        _save_trial(
            trial_directory,
            metadata,
            (),
            (),
            command_requested_at=None,
            return_requested_at=None,
            servo_id=servo_id,
            target_position=dry_target,
        )
        print(f"DRY RUN: no camera, serial, prepare, or servo command was used.")
        print(f"Dry-run trial: {trial_directory}")
        return 0

    angle_source = _make_angle_source(config)
    api = None
    telemetry: TelemetryMonitor | None = None
    recorder: TelemetryRecorder | None = None
    controller: ReliablePositionController | None = None
    measurements: list[JointAngleMeasurement] = []
    start_positions: tuple[int, ...] | None = None
    command_requested_at: float | None = None
    commanded_at: float | None = None
    return_requested_at: float | None = None
    error_text: str | None = None
    stop_sent = False
    tracking_counts = [0] * len(SERVO_IDS)

    try:
        angle_source.open()
        angle_source.warm_up(int(config["vision"]["warmup_frames"]))
        from servo.servo_APIs import ServoAPI

        hardware = config["hardware"]
        api = ServoAPI(
            port=str(hardware["serial_port"]),
            baud_rate=int(hardware["baud_rate"]),
            timeout=float(hardware["serial_timeout_s"]),
        )
        telemetry = TelemetryMonitor(
            api,
            num_servos=len(TENDONS),
            read_timeout_s=float(hardware["serial_timeout_s"]),
        )
        controller = _make_position_controller(api, telemetry, config, position_limits)
        telemetry.start()
        recorder = TelemetryRecorder(telemetry)
        recorder.start()
        controller.prepare(
            SERVO_IDS,
            force_init_servo_ids=tuple(range(len(TENDONS))),
        )
        angle_source.reset_temporal_history()
        angle_source.warm_up(int(config["vision"]["warmup_frames"]))
        start_positions = tuple(controller.current_position(servo) for servo in SERVO_IDS)
        start_position = start_positions[tendon_index]
        target_position = start_position + direction * args.position_step_count
        if not position_limits[tendon_index][0] <= target_position <= position_limits[tendon_index][1]:
            raise ValueError(f"target {target_position} is outside {args.tendon} position limits")
        metadata["start_position"] = start_position
        metadata["target_position"] = target_position

        measurements, next_frame_index = _read_baseline(
            angle_source,
            sample_count=args.baseline_sample_count,
            timeout_s=args.observation_timeout_s,
            start_frame_index=0,
        )
        latest = telemetry.latest(float(config["safety"]["telemetry_stale_s"]))
        if latest is None:
            raise RuntimeError("Fresh telemetry is not available before command")
        if not _measurement_is_usable(measurements[-1]):
            raise RuntimeError("DLC observation is not valid before command")

        command_requested_at = time.monotonic()
        stream_result = controller.stream_positions(
            (servo_id,),
            (target_position,),
            time_ms=int(config["control"].get("stream_time_ms", 0)),
        )
        commanded_at = stream_result.commanded_at
        metadata["command_requested_at"] = command_requested_at
        metadata["commanded_at"] = commanded_at

        response_deadline = command_requested_at + args.observation_timeout_s
        tracking_limit = int(config["safety"]["tracking_error_limit"])
        tracking_cycles = int(config["safety"]["tracking_error_cycles"])
        expected_targets = list(start_positions)
        expected_targets[tendon_index] = target_position
        last_tracking_sequence = stream_result.telemetry_sequence
        while time.monotonic() < response_deadline:
            telemetry.raise_if_failed()
            tracking_snapshot = telemetry.latest(
                float(config["safety"]["telemetry_stale_s"])
            )
            if tracking_snapshot is None:
                raise RuntimeError("Telemetry became stale during response observation")
            if tracking_snapshot.sequence > last_tracking_sequence:
                last_tracking_sequence = tracking_snapshot.sequence
                for index, servo in enumerate(SERVO_IDS):
                    error = expected_targets[index] - tracking_snapshot.positions[servo]
                    if abs(error) > tracking_limit:
                        tracking_counts[index] += 1
                    else:
                        tracking_counts[index] = 0
                    if tracking_counts[index] >= tracking_cycles:
                        raise RuntimeError(
                            f"Servo {servo} tracking error {error:+d} exceeded "
                            f"{tracking_limit} for {tracking_counts[index]} telemetry samples"
                        )
            measurement = angle_source.read(next_frame_index)
            next_frame_index += 1
            measurements.append(measurement)
            if not measurement.valid:
                raise RuntimeError(
                    f"Invalid DLC measurement after command: {measurement.reason}"
                )

        if return_to_start:
            return_requested_at = _return_to_start(
                controller,
                telemetry,
                start_positions,
                config,
            )
            metadata["return_requested_at"] = return_requested_at
        controller.stop_all()
        stop_sent = True
        metadata["completion_reason"] = "observation_complete"
    except KeyboardInterrupt:
        error_text = "interrupted"
    except Exception as exc:
        error_text = f"{type(exc).__name__}: {exc}"
    finally:
        if error_text is not None and controller is not None:
            try:
                controller.stop_all()
                stop_sent = True
            except Exception as exc:
                error_text += f"; stop_all failed: {exc}"
        if recorder is not None:
            recorder.stop()
            if recorder.error is not None and error_text is None:
                error_text = f"Telemetry recorder failed: {recorder.error}"
        if telemetry is not None:
            telemetry.stop()
        if api is not None:
            api.close()
        angle_source.close()

    if recorder is not None and recorder.error is not None and error_text is None:
        error_text = f"Telemetry recorder failed: {recorder.error}"
    telemetry_records = () if recorder is None else tuple(recorder.records)
    if commanded_at is not None:
        metadata["analysis"] = analyse_step_response(
            args=args,
            measurements=measurements,
            telemetry_records=telemetry_records,
            commanded_at=commanded_at,
            command_requested_at=command_requested_at,
            return_requested_at=return_requested_at,
            servo_id=servo_id,
            start_position=metadata["start_position"],
            target_position=metadata["target_position"],
        )
    if error_text is not None:
        metadata["completion_reason"] = "fault"
        metadata["error"] = error_text
    elif metadata["completion_reason"] is None:
        metadata["completion_reason"] = "observation_complete"
    _save_trial(
        trial_directory,
        metadata,
        telemetry_records,
        measurements,
        command_requested_at=command_requested_at,
        return_requested_at=return_requested_at,
        servo_id=servo_id,
        target_position=metadata["target_position"],
    )
    print(f"Trial: {trial_directory}")
    if error_text is not None:
        print(f"FAILED: {error_text}", file=sys.stderr)
        return 1
    print("Step-response observation completed.")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
