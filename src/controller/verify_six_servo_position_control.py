"""Verify a small simultaneous position move on all six unloaded servos."""

from __future__ import annotations

import argparse
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from servo.control import (
    PositionControlConfig,
    ReliablePositionController,
    TelemetryMonitor,
    TelemetrySnapshot,
)


SERVO_IDS = tuple(range(6))
MULTITURN_POSITION_MIN = -28672
MULTITURN_POSITION_MAX = 28672
DEFAULT_PORT = "COM3"
DEFAULT_BAUD_RATE = 921600
DEFAULT_SERIAL_TIMEOUT_S = 0.2
DEFAULT_DELTA = 100
DEFAULT_MOVE_TIME_MS = 1000
DEFAULT_START_OBSERVATION_S = 0.3
DEFAULT_START_MIN_DELTA = 3
DEFAULT_TOLERANCE = 10
DEFAULT_SPEED_TOLERANCE = 5
DEFAULT_STABLE_FRAME_COUNT = 3
DEFAULT_ARRIVAL_TIMEOUT_S = 4.0
DEFAULT_DISPLAY_INTERVAL_S = 0.1


@dataclass(frozen=True)
class PhaseResult:
    name: str
    targets: tuple[int, ...]
    final_positions: tuple[int, ...]
    first_command_misses: tuple[int, ...]
    send_elapsed_ms: float


def build_targets(start_positions: Sequence[int], delta: int) -> tuple[int, ...]:
    if len(start_positions) != len(SERVO_IDS):
        raise ValueError(f"Expected {len(SERVO_IDS)} start positions")
    if delta == 0:
        raise ValueError("delta must be non-zero")

    targets = tuple(int(position) + int(delta) for position in start_positions)
    for servo_id, target in zip(SERVO_IDS, targets):
        if not MULTITURN_POSITION_MIN <= target <= MULTITURN_POSITION_MAX:
            raise ValueError(
                f"Servo {servo_id} target {target} is outside "
                f"{MULTITURN_POSITION_MIN}-{MULTITURN_POSITION_MAX}"
            )
    return targets


def first_command_misses(
    baselines: Sequence[int],
    targets: Sequence[int],
    actual_positions: Sequence[int],
    *,
    start_min_delta: int,
    tolerance: int,
) -> tuple[int, ...]:
    if not (
        len(baselines) == len(targets) == len(actual_positions) == len(SERVO_IDS)
    ):
        raise ValueError("Expected six baseline, target, and actual positions")

    missed: list[int] = []
    for servo_id, baseline, target, actual in zip(
        SERVO_IDS, baselines, targets, actual_positions
    ):
        if abs(target - actual) <= tolerance:
            continue
        direction = 1 if target >= baseline else -1
        if direction * (actual - baseline) < start_min_delta:
            missed.append(servo_id)
    return tuple(missed)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Move all six unloaded servos by a small amount in one command batch, "
            "check first-command start, then return them to their measured starts."
        )
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually move the servos. Without this flag, only the plan is printed.",
    )
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--baud-rate", type=int, default=DEFAULT_BAUD_RATE)
    parser.add_argument("--delta", type=int, default=DEFAULT_DELTA)
    parser.add_argument("--move-time-ms", type=int, default=DEFAULT_MOVE_TIME_MS)
    parser.add_argument(
        "--arrival-timeout-s", type=float, default=DEFAULT_ARRIVAL_TIMEOUT_S
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if args.delta == 0:
        raise ValueError("--delta must be non-zero")
    if not 1 <= args.move_time_ms <= 65535:
        raise ValueError("--move-time-ms must be in the range 1-65535")
    if not math.isfinite(args.arrival_timeout_s) or args.arrival_timeout_s <= 0.0:
        raise ValueError("--arrival-timeout-s must be finite and positive")
    if args.arrival_timeout_s < args.move_time_ms / 1000.0:
        raise ValueError("--arrival-timeout-s must cover --move-time-ms")
    if args.baud_rate <= 0:
        raise ValueError("--baud-rate must be positive")


def build_controller_config(args: argparse.Namespace) -> PositionControlConfig:
    return PositionControlConfig(
        telemetry_stale_s=0.5,
        telemetry_wait_s=3.0,
        id_map_reset_wait_s=0.1,
        speed_init_wait_s=0.2,
        prime_command_count=2,
        prime_interval_s=0.2,
        start_observation_s=DEFAULT_START_OBSERVATION_S,
        start_min_delta=DEFAULT_START_MIN_DELTA,
        position_tolerance=DEFAULT_TOLERANCE,
        speed_tolerance=DEFAULT_SPEED_TOLERANCE,
        stable_frame_count=DEFAULT_STABLE_FRAME_COUNT,
        arrival_timeout_s=args.arrival_timeout_s,
        max_start_retries=1,
        reset_id_map_on_prepare=True,
        multiturn=True,
        position_min=MULTITURN_POSITION_MIN,
        position_max=MULTITURN_POSITION_MAX,
    )


def wait_for_observation(
    telemetry: TelemetryMonitor,
    sequence: int,
    duration_s: float,
) -> TelemetrySnapshot:
    deadline = time.monotonic() + duration_s
    snapshot: TelemetrySnapshot | None = None
    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        try:
            snapshot = telemetry.wait_for_newer(sequence, min(remaining, 0.1))
            sequence = snapshot.sequence
        except TimeoutError:
            pass
    if snapshot is None:
        snapshot = telemetry.latest(0.5)
    if snapshot is None:
        raise RuntimeError("Telemetry was unavailable during start observation")
    return snapshot


def wait_for_all_settled(
    telemetry: TelemetryMonitor,
    targets: Sequence[int],
    *,
    sequence: int,
    timeout_s: float,
) -> TelemetrySnapshot:
    deadline = time.monotonic() + timeout_s
    stable_frames = [0] * len(SERVO_IDS)
    next_display_at = 0.0
    snapshot: TelemetrySnapshot | None = None

    while time.monotonic() < deadline:
        remaining = deadline - time.monotonic()
        snapshot = telemetry.wait_for_newer(sequence, remaining)
        sequence = snapshot.sequence
        if snapshot.speeds is None:
            raise RuntimeError("Six-servo test requires telemetry with speed values")

        for servo_id, target in zip(SERVO_IDS, targets):
            settled = (
                abs(target - snapshot.positions[servo_id]) <= DEFAULT_TOLERANCE
                and abs(snapshot.speeds[servo_id]) <= DEFAULT_SPEED_TOLERANCE
            )
            stable_frames[servo_id] = stable_frames[servo_id] + 1 if settled else 0

        now = time.monotonic()
        if now >= next_display_at:
            summary = " ".join(
                f"{servo_id}:{snapshot.positions[servo_id]}"
                f"({target - snapshot.positions[servo_id]:+d})"
                for servo_id, target in zip(SERVO_IDS, targets)
            )
            print(f"  positions(error) {summary}", flush=True)
            next_display_at = now + DEFAULT_DISPLAY_INTERVAL_S

        if all(count >= DEFAULT_STABLE_FRAME_COUNT for count in stable_frames):
            return snapshot

    actual = None if snapshot is None else snapshot.positions
    raise TimeoutError(f"All servos did not settle before timeout; actual={actual}")


def run_phase(
    name: str,
    controller: ReliablePositionController,
    telemetry: TelemetryMonitor,
    baselines: Sequence[int],
    targets: Sequence[int],
    *,
    move_time_ms: int,
    arrival_timeout_s: float,
) -> PhaseResult:
    print(f"\n{name}: sending one target batch")
    command = controller.stream_positions(SERVO_IDS, targets, time_ms=move_time_ms)
    send_elapsed_ms = (time.monotonic() - command.commanded_at) * 1000.0
    print(f"  six command packets sent in {send_elapsed_ms:.3f} ms")

    observed = wait_for_observation(
        telemetry,
        command.telemetry_sequence,
        DEFAULT_START_OBSERVATION_S,
    )
    missed = first_command_misses(
        baselines,
        targets,
        observed.positions,
        start_min_delta=DEFAULT_START_MIN_DELTA,
        tolerance=DEFAULT_TOLERANCE,
    )
    if missed:
        print(f"  first-command start not observed for servo IDs: {missed}")
        retry_targets = tuple(targets[servo_id] for servo_id in missed)
        retry = controller.stream_positions(missed, retry_targets, time_ms=move_time_ms)
        settle_sequence = retry.telemetry_sequence
    else:
        print("  first-command start observed for all servo IDs")
        settle_sequence = observed.sequence

    final = wait_for_all_settled(
        telemetry,
        targets,
        sequence=settle_sequence,
        timeout_s=arrival_timeout_s,
    )
    return PhaseResult(
        name=name,
        targets=tuple(targets),
        final_positions=final.positions,
        first_command_misses=missed,
        send_elapsed_ms=send_elapsed_ms,
    )


def execute(args: argparse.Namespace) -> None:
    from servo.servo_APIs import ServoAPI

    with ServoAPI(
        port=args.port,
        baud_rate=args.baud_rate,
        timeout=DEFAULT_SERIAL_TIMEOUT_S,
    ) as api:
        telemetry = TelemetryMonitor(
            api,
            num_servos=len(SERVO_IDS),
            read_timeout_s=DEFAULT_SERIAL_TIMEOUT_S,
        )
        controller = ReliablePositionController(
            api,
            telemetry,
            build_controller_config(args),
        )
        telemetry.start()
        try:
            print("Preparing all six servos for multi-turn position control.")
            controller.prepare(SERVO_IDS, force_init_servo_ids=SERVO_IDS)
            starts = tuple(controller.current_position(servo_id) for servo_id in SERVO_IDS)
            targets = build_targets(starts, args.delta)
            print("Measured plan:")
            for servo_id, start, target in zip(SERVO_IDS, starts, targets):
                print(f"  servo {servo_id}: {start} -> {target} -> {start}")

            outward = run_phase(
                "OUT",
                controller,
                telemetry,
                starts,
                targets,
                move_time_ms=args.move_time_ms,
                arrival_timeout_s=args.arrival_timeout_s,
            )
            returned = run_phase(
                "RETURN",
                controller,
                telemetry,
                outward.final_positions,
                starts,
                move_time_ms=args.move_time_ms,
                arrival_timeout_s=args.arrival_timeout_s,
            )

            print("\nPASS: all six servos moved and returned within tolerance.")
            print(f"  OUT first-command misses: {outward.first_command_misses or 'none'}")
            print(
                f"  RETURN first-command misses: "
                f"{returned.first_command_misses or 'none'}"
            )
            for servo_id, start, actual in zip(
                SERVO_IDS, starts, returned.final_positions
            ):
                print(
                    f"  servo {servo_id}: start={start}, final={actual}, "
                    f"error={start - actual:+d}"
                )
        finally:
            try:
                controller.stop_all()
            finally:
                telemetry.stop()


def main() -> None:
    args = parse_args()
    validate_args(args)
    if not args.execute:
        print("DRY RUN: no serial port will be opened and no servo will move.")
        print(
            f"Plan: servo IDs {SERVO_IDS}, delta={args.delta:+d}, "
            f"move_time_ms={args.move_time_ms}, then return to measured starts."
        )
        print("Re-run with --execute only after removing all tendons.")
        return
    execute(args)


if __name__ == "__main__":
    main()
