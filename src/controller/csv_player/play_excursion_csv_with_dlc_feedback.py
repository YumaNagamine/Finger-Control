"""Run synchronized predicted-excursion playback with DLC angle feedback."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Sequence

SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from controller.csv_player.excursion_player import load_position_calibration
from controller.feedback import (
    FeedbackCommandBuilder,
    FeedbackTrajectory,
    JOINTS,
    JointFeedbackController,
    MomentArmRuntime,
)
from controller.servo_mapping import SERVO_IDS_BY_TENDON, TENDONS
from observation.vision.realtime_joint_angle_source import (
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


class TrackingSupervisor:
    def __init__(
        self,
        *,
        servo_ids: Sequence[int],
        error_limit: int,
        consecutive_cycles: int,
    ) -> None:
        if error_limit < 0:
            raise ValueError("tracking error limit must be non-negative")
        if consecutive_cycles <= 0:
            raise ValueError("tracking error cycles must be greater than zero")
        self._servo_ids = tuple(int(value) for value in servo_ids)
        self._error_limit = int(error_limit)
        self._consecutive_cycles = int(consecutive_cycles)
        self._error_counts = [0] * len(self._servo_ids)
        self._last_targets: tuple[int, ...] | None = None

    def check(self, snapshot: TelemetrySnapshot) -> None:
        if self._last_targets is None:
            return
        for index, (servo_id, target) in enumerate(
            zip(self._servo_ids, self._last_targets)
        ):
            error = target - snapshot.positions[servo_id]
            if abs(error) > self._error_limit:
                self._error_counts[index] += 1
            else:
                self._error_counts[index] = 0
            if self._error_counts[index] >= self._consecutive_cycles:
                tendon = TENDONS[index]
                raise RuntimeError(
                    f"{tendon} tracking error {error:+d} exceeded "
                    f"{self._error_limit} for {self._error_counts[index]} cycles"
                )

    def record_targets(self, targets: Sequence[int]) -> None:
        values = tuple(int(value) for value in targets)
        if len(values) != len(self._servo_ids):
            raise ValueError("targets must match servo_ids")
        self._last_targets = values


class CycleLogger:
    def __init__(self, output_directory: Path) -> None:
        output_directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = output_directory / f"dlc_feedback_{timestamp}.csv"
        self._handle = self.path.open("w", newline="", encoding="utf-8")
        self._writer: csv.DictWriter | None = None

    def write(self, row: dict[str, object]) -> None:
        if self._writer is None:
            self._writer = csv.DictWriter(self._handle, fieldnames=list(row))
            self._writer.writeheader()
        self._writer.writerow(row)
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Play a predicted-excursion CSV at a fixed rate and add synchronous "
            "DLC joint-angle feedback."
        )
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG_PATH))
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Send commands to the physical servos. Without this flag, run a dry-run.",
    )
    return parser.parse_args()


def _project_path(raw_path: str, option_name: str) -> Path:
    path = resolve_path(raw_path, PROJECT_ROOT)
    if path is None:
        raise ValueError(f"{option_name} must be a non-empty path")
    return path


def _ordered_floats(mapping: dict, names: Sequence[str], option_name: str) -> tuple[float, ...]:
    missing = [name for name in names if name not in mapping]
    if missing:
        raise ValueError(f"{option_name} is missing values for {missing}")
    values = tuple(float(mapping[name]) for name in names)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{option_name} contains a non-finite value")
    return values


def _ordered_ints(mapping: dict, names: Sequence[str], option_name: str) -> tuple[int, ...]:
    missing = [name for name in names if name not in mapping]
    if missing:
        raise ValueError(f"{option_name} is missing values for {missing}")
    return tuple(int(mapping[name]) for name in names)


def _position_limits(raw_limits: dict) -> tuple[tuple[int, int], ...]:
    missing = [tendon for tendon in TENDONS if tendon not in raw_limits]
    if missing:
        raise ValueError(f"position_limits is missing values for {missing}")
    result: list[tuple[int, int]] = []
    for tendon in TENDONS:
        values = raw_limits[tendon]
        if not isinstance(values, list) or len(values) != 2:
            raise ValueError(f"position_limits.{tendon} must contain [min, max]")
        result.append((int(values[0]), int(values[1])))
    return tuple(result)


def _make_feedback_components(config: dict):
    paths = config["paths"]
    feedback_config = config["feedback"]
    safety_config = config["safety"]

    trajectory = FeedbackTrajectory.from_csv(
        _project_path(paths["prediction_csv"], "paths.prediction_csv"),
        direction_deadband_rad=math.radians(
            float(feedback_config.get("direction_deadband_deg", 0.2))
        ),
    )
    moment_arm = MomentArmRuntime.from_directory(
        _project_path(
            paths["moment_arm_directory"],
            "paths.moment_arm_directory",
        )
    )
    feedback_controller = JointFeedbackController(
        moment_arm,
        kp=_ordered_floats(feedback_config["kp"], JOINTS, "feedback.kp"),
        max_joint_correction_rad=tuple(
            math.radians(value)
            for value in _ordered_floats(
                feedback_config["max_joint_correction_deg"],
                JOINTS,
                "feedback.max_joint_correction_deg",
            )
        ),
        max_excursion_correction_mm=_ordered_floats(
            feedback_config["max_excursion_correction_mm"],
            TENDONS,
            "feedback.max_excursion_correction_mm",
        ),
    )
    position_units_per_mm = load_position_calibration(
        _project_path(
            paths["servo_calibration"],
            "paths.servo_calibration",
        )
    )
    position_limits = _position_limits(safety_config["position_limits"])
    max_position_step = _ordered_ints(
        safety_config["max_position_step"],
        TENDONS,
        "safety.max_position_step",
    )
    return (
        trajectory,
        feedback_controller,
        position_units_per_mm,
        position_limits,
        max_position_step,
    )


def _make_angle_source(config: dict) -> RealtimeJointAngleSource:
    paths = config["paths"]
    vision = config["vision"]
    return RealtimeJointAngleSource(
        dlc_config_path=_project_path(paths["dlc_config"], "paths.dlc_config"),
        camera_config_path=_project_path(paths["camera_config"], "paths.camera_config"),
        min_likelihood=float(vision["min_likelihood"]),
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


def _build_log_row(
    *,
    cycle_index: int,
    scheduled_at: float,
    started_at: float,
    command_deadline_at: float,
    measurement: JointAngleMeasurement,
    reference,
    feedback,
    command,
    commanded_at: float,
    cycle_finished_at: float,
    snapshot: TelemetrySnapshot | None,
) -> dict[str, object]:
    row: dict[str, object] = {
        "cycle_index": cycle_index,
        "scheduled_s": scheduled_at - started_at,
        "command_deadline_s": command_deadline_at - started_at,
        "captured_s": measurement.captured_at - started_at,
        "inference_finished_s": measurement.inference_finished_at - started_at,
        "commanded_s": commanded_at - started_at,
        "cycle_duration_ms": (cycle_finished_at - scheduled_at) * 1000.0,
        "inference_ms": measurement.inference_ms,
        "min_likelihood": measurement.min_likelihood,
    }
    for index, joint in enumerate(JOINTS):
        row[f"{joint}_reference_rad"] = reference.joint_angles_rad[index]
        row[f"{joint}_measured_rad"] = measurement.flexion_angles_rad[index]
        row[f"{joint}_error_rad"] = feedback.error_rad[index]
    for index, tendon in enumerate(TENDONS):
        row[f"{tendon}_nominal_excursion_mm"] = command.nominal_excursions_mm[index]
        row[f"{tendon}_feedback_excursion_mm"] = command.feedback_excursions_mm[index]
        row[f"{tendon}_total_excursion_mm"] = command.total_excursions_mm[index]
        row[f"{tendon}_target_position"] = command.target_positions[index]
        if snapshot is not None:
            servo_id = SERVO_IDS[index]
            row[f"{tendon}_actual_position"] = snapshot.positions[servo_id]
            row[f"{tendon}_load"] = snapshot.loads[servo_id]
        else:
            row[f"{tendon}_actual_position"] = ""
            row[f"{tendon}_load"] = ""
    return row


def _run_loop(
    *,
    config: dict,
    trajectory: FeedbackTrajectory,
    feedback_controller: JointFeedbackController,
    angle_source: RealtimeJointAngleSource,
    command_builder: FeedbackCommandBuilder,
    logger: CycleLogger,
    controller: ReliablePositionController | None = None,
    telemetry: TelemetryMonitor | None = None,
) -> None:
    control = config["control"]
    safety = config["safety"]
    frequency_hz = float(control["frequency_hz"])
    if not math.isfinite(frequency_hz) or frequency_hz <= 0.0:
        raise ValueError("control.frequency_hz must be finite and greater than zero")
    period_s = 1.0 / frequency_hz
    max_overrun_s = float(control["max_cycle_overrun_s"])
    stream_time_ms = int(control.get("stream_time_ms", 0))
    supervisor = TrackingSupervisor(
        servo_ids=SERVO_IDS,
        error_limit=int(safety["tracking_error_limit"]),
        consecutive_cycles=int(safety["tracking_error_cycles"]),
    )

    started_at = time.monotonic()
    cycle_index = 0
    while True:
        scheduled_at = started_at + cycle_index * period_s
        remaining_s = scheduled_at - time.monotonic()
        if remaining_s > 0.0:
            time.sleep(remaining_s)

        elapsed_s = scheduled_at - started_at
        if elapsed_s > trajectory.duration_s:
            break

        reference = trajectory.sample(elapsed_s)
        measurement = angle_source.read(cycle_index)
        if not measurement.valid:
            raise RuntimeError(
                f"Invalid DLC measurement at cycle {cycle_index}: {measurement.reason}"
            )
        feedback = feedback_controller.compute(
            reference.joint_angles_rad,
            measurement.flexion_angles_rad,
            reference.motion_directions,
        )
        command = command_builder.build(
            reference.nominal_excursions_mm,
            feedback.excursion_correction_mm,
        )
        command_deadline_at = scheduled_at + period_s
        remaining_to_command_s = command_deadline_at - time.monotonic()
        if remaining_to_command_s > 0.0:
            time.sleep(remaining_to_command_s)
        elif -remaining_to_command_s > max_overrun_s:
            raise RuntimeError(
                f"Control cycle {cycle_index} missed its command deadline by "
                f"{-remaining_to_command_s:.3f} s"
            )

        snapshot: TelemetrySnapshot | None = None
        commanded_at = time.monotonic()
        if controller is not None:
            if telemetry is None:
                raise RuntimeError("telemetry is required for hardware execution")
            snapshot = telemetry.latest(float(safety["telemetry_stale_s"]))
            if snapshot is None:
                raise RuntimeError("Fresh servo telemetry is not available")
            supervisor.check(snapshot)
            stream_result = controller.stream_positions(
                SERVO_IDS,
                command.target_positions,
                time_ms=stream_time_ms,
            )
            commanded_at = stream_result.commanded_at
            supervisor.record_targets(command.target_positions)
        else:
            print(
                f"DRY cycle={cycle_index} t={elapsed_s:.3f}s "
                f"targets={command.target_positions}",
                flush=True,
            )

        cycle_finished_at = time.monotonic()
        logger.write(
            _build_log_row(
                cycle_index=cycle_index,
                scheduled_at=scheduled_at,
                started_at=started_at,
                command_deadline_at=command_deadline_at,
                measurement=measurement,
                reference=reference,
                feedback=feedback,
                command=command,
                commanded_at=commanded_at,
                cycle_finished_at=cycle_finished_at,
                snapshot=snapshot,
            )
        )

        overrun_s = cycle_finished_at - command_deadline_at
        if overrun_s > max_overrun_s:
            raise RuntimeError(
                f"Control cycle {cycle_index} overran its deadline by "
                f"{overrun_s:.3f} s"
            )
        cycle_index += 1


def _return_to_start(
    *,
    controller: ReliablePositionController,
    telemetry: TelemetryMonitor,
    start_positions: Sequence[int],
    config: dict,
) -> None:
    hardware = config["hardware"]
    safety = config["safety"]
    controller.stream_positions(
        SERVO_IDS,
        start_positions,
        time_ms=int(hardware["return_time_ms"]),
    )
    deadline = time.monotonic() + float(hardware["return_timeout_s"])
    tolerance = int(hardware["return_tolerance"])
    while time.monotonic() < deadline:
        snapshot = telemetry.latest(float(safety["telemetry_stale_s"]))
        if snapshot is not None and all(
            abs(target - snapshot.positions[servo_id]) <= tolerance
            for servo_id, target in zip(SERVO_IDS, start_positions)
        ):
            print("All servos returned to their initial positions.")
            return
        time.sleep(0.02)
    raise TimeoutError("Servos did not return to their initial positions before timeout")


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).expanduser().resolve()
    config = load_config(
        config_path,
        required_keys=("control", "paths", "feedback", "vision", "safety", "hardware"),
    )
    (
        trajectory,
        feedback_controller,
        position_units_per_mm,
        position_limits,
        max_position_step,
    ) = _make_feedback_components(config)
    angle_source = _make_angle_source(config)
    log_directory = _project_path(
        config["paths"]["log_directory"],
        "paths.log_directory",
    )

    print("Opening camera and initializing DLC before servo control.")
    angle_source.open()
    try:
        meta1_calibration = angle_source.calibrate_meta1()
        if meta1_calibration is not None:
            print(
                "Static meta1 accepted: "
                f"position=({meta1_calibration.position[0]:.1f}, "
                f"{meta1_calibration.position[1]:.1f}), "
                f"cluster={meta1_calibration.dominant_samples}/"
                f"{meta1_calibration.valid_samples}, "
                f"dispersion={meta1_calibration.dispersion_px:.1f}px"
            )
        angle_source.warm_up(int(config["vision"]["warmup_frames"]))

        if not args.execute:
            start_positions = tuple(
                int(value)
                for value in config["hardware"]["dry_run_start_positions"]
            )
            initial_sample = trajectory.sample(0.0)
            command_builder = FeedbackCommandBuilder(
                start_positions=start_positions,
                initial_excursions_mm=initial_sample.nominal_excursions_mm,
                position_units_per_mm=position_units_per_mm,
                position_limits=position_limits,
                max_position_step=max_position_step,
            )
            logger = CycleLogger(log_directory)
            try:
                print("DRY RUN: camera and DLC are active; no servo commands will be sent.")
                _run_loop(
                    config=config,
                    trajectory=trajectory,
                    feedback_controller=feedback_controller,
                    angle_source=angle_source,
                    command_builder=command_builder,
                    logger=logger,
                )
            finally:
                logger.close()
            print(f"Dry-run log: {logger.path}")
            return

        from servo.servo_APIs import ServoAPI

        hardware = config["hardware"]
        with ServoAPI(
            port=str(hardware["serial_port"]),
            baud_rate=int(hardware["baud_rate"]),
            timeout=float(hardware["serial_timeout_s"]),
        ) as api:
            telemetry = TelemetryMonitor(
                api,
                num_servos=len(TENDONS),
                read_timeout_s=float(hardware["serial_timeout_s"]),
            )
            controller = _make_position_controller(
                api,
                telemetry,
                config,
                position_limits,
            )
            telemetry.start()
            logger: CycleLogger | None = None
            completed = False
            try:
                controller.prepare(
                    SERVO_IDS,
                    force_init_servo_ids=tuple(range(len(TENDONS))),
                )
                start_positions = tuple(
                    controller.current_position(servo_id)
                    for servo_id in SERVO_IDS
                )
                initial_sample = trajectory.sample(0.0)
                command_builder = FeedbackCommandBuilder(
                    start_positions=start_positions,
                    initial_excursions_mm=initial_sample.nominal_excursions_mm,
                    position_units_per_mm=position_units_per_mm,
                    position_limits=position_limits,
                    max_position_step=max_position_step,
                )
                post_prepare_warmup_frames = int(config["vision"]["warmup_frames"])
                print(
                    "Re-warming camera and DLC after servo preparation "
                    f"({post_prepare_warmup_frames} frame(s))."
                )
                angle_source.warm_up(post_prepare_warmup_frames)
                logger = CycleLogger(log_directory)
                print("Executing synchronized DLC feedback control. Press Ctrl-C to stop.")
                _run_loop(
                    config=config,
                    trajectory=trajectory,
                    feedback_controller=feedback_controller,
                    angle_source=angle_source,
                    command_builder=command_builder,
                    logger=logger,
                    controller=controller,
                    telemetry=telemetry,
                )
                completed = True
                if bool(hardware["return_to_start"]):
                    _return_to_start(
                        controller=controller,
                        telemetry=telemetry,
                        start_positions=start_positions,
                        config=config,
                    )
            finally:
                try:
                    controller.stop_all()
                finally:
                    telemetry.stop()
                    if logger is not None:
                        logger.close()
            if completed and logger is not None:
                print(f"Feedback-control log: {logger.path}")
    finally:
        angle_source.close()


if __name__ == "__main__":
    main()
