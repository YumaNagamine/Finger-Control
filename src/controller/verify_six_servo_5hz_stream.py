"""Exercise all six unloaded servos with a 5 Hz position-command stream."""

from __future__ import annotations

import argparse
import csv
import math
import sys
import time
from pathlib import Path
from typing import Sequence


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from controller.verify_six_servo_position_control import (
    DEFAULT_ARRIVAL_TIMEOUT_S,
    DEFAULT_BAUD_RATE,
    DEFAULT_PORT,
    DEFAULT_SERIAL_TIMEOUT_S,
    MULTITURN_POSITION_MAX,
    MULTITURN_POSITION_MIN,
    SERVO_IDS,
    build_controller_config,
    run_phase,
)
from servo.control import ReliablePositionController, TelemetryMonitor


DEFAULT_FREQUENCY_HZ = 5.0
DEFAULT_DURATION_S = 30.0
DEFAULT_AMPLITUDE = 100
DEFAULT_TRAJECTORY_PERIOD_S = 4.0
DEFAULT_MOVE_TIME_MS = 200
DEFAULT_STARTUP_WAIT_S = 5.0
DEFAULT_LOG_DIRECTORY = SRC_ROOT.parent / "logs" / "servo_5hz_test"


def build_sine_targets(
    start_positions: Sequence[int],
    amplitude: int,
    elapsed_s: float,
    trajectory_period_s: float,
) -> tuple[int, ...]:
    if len(start_positions) != len(SERVO_IDS):
        raise ValueError(f"Expected {len(SERVO_IDS)} start positions")
    if amplitude <= 0:
        raise ValueError("amplitude must be positive")
    if not math.isfinite(elapsed_s) or elapsed_s < 0.0:
        raise ValueError("elapsed_s must be finite and non-negative")
    if not math.isfinite(trajectory_period_s) or trajectory_period_s <= 0.0:
        raise ValueError("trajectory_period_s must be finite and positive")

    offset = round(
        amplitude * math.sin(2.0 * math.pi * elapsed_s / trajectory_period_s)
    )
    targets = tuple(int(position) + offset for position in start_positions)
    for servo_id, target in zip(SERVO_IDS, targets):
        if not MULTITURN_POSITION_MIN <= target <= MULTITURN_POSITION_MAX:
            raise ValueError(
                f"Servo {servo_id} target {target} is outside "
                f"{MULTITURN_POSITION_MIN}-{MULTITURN_POSITION_MAX}"
            )
    return targets


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Send a small sinusoidal multi-turn position trajectory to all six "
            "unloaded servos at a fixed control frequency."
        )
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Actually move the servos. Without this flag, only the plan is printed.",
    )
    parser.add_argument("--port", default=DEFAULT_PORT)
    parser.add_argument("--baud-rate", type=int, default=DEFAULT_BAUD_RATE)
    parser.add_argument("--frequency-hz", type=float, default=DEFAULT_FREQUENCY_HZ)
    parser.add_argument("--duration-s", type=float, default=DEFAULT_DURATION_S)
    parser.add_argument("--amplitude", type=int, default=DEFAULT_AMPLITUDE)
    parser.add_argument(
        "--trajectory-period-s",
        type=float,
        default=DEFAULT_TRAJECTORY_PERIOD_S,
    )
    parser.add_argument("--move-time-ms", type=int, default=DEFAULT_MOVE_TIME_MS)
    parser.add_argument(
        "--startup-wait-s",
        type=float,
        default=DEFAULT_STARTUP_WAIT_S,
        help="Wait after opening the ESP32 serial port before reading telemetry.",
    )
    parser.add_argument(
        "--arrival-timeout-s",
        type=float,
        default=DEFAULT_ARRIVAL_TIMEOUT_S,
    )
    parser.add_argument(
        "--log-directory",
        type=Path,
        default=DEFAULT_LOG_DIRECTORY,
    )
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    positive_floats = {
        "--frequency-hz": args.frequency_hz,
        "--duration-s": args.duration_s,
        "--trajectory-period-s": args.trajectory_period_s,
        "--arrival-timeout-s": args.arrival_timeout_s,
    }
    for name, value in positive_floats.items():
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"{name} must be finite and positive")
    if args.amplitude <= 0:
        raise ValueError("--amplitude must be positive")
    if not 1 <= args.move_time_ms <= 65535:
        raise ValueError("--move-time-ms must be in the range 1-65535")
    if not math.isfinite(args.startup_wait_s) or args.startup_wait_s < 0.0:
        raise ValueError("--startup-wait-s must be finite and non-negative")
    if args.baud_rate <= 0:
        raise ValueError("--baud-rate must be positive")


def create_log_writer(
    directory: Path,
) -> tuple[object, csv.writer, Path]:
    directory.mkdir(parents=True, exist_ok=True)
    timestamp = time.strftime("%Y%m%d_%H%M%S")
    path = directory / f"six_servo_5hz_{timestamp}.csv"
    handle = path.open("w", newline="", encoding="utf-8")
    writer = csv.writer(handle)
    header = [
        "cycle",
        "scheduled_elapsed_s",
        "send_started_elapsed_s",
        "send_elapsed_ms",
        "schedule_lateness_ms",
        "deadline_missed",
        "telemetry_age_ms",
    ]
    for servo_id in SERVO_IDS:
        header.extend(
            [
                f"target_{servo_id}",
                f"actual_{servo_id}",
                f"previous_target_error_{servo_id}",
            ]
        )
    writer.writerow(header)
    return handle, writer, path


def run_stream(
    controller: ReliablePositionController,
    telemetry: TelemetryMonitor,
    start_positions: Sequence[int],
    args: argparse.Namespace,
) -> tuple[tuple[int, ...], Path]:
    control_period_s = 1.0 / args.frequency_hz
    cycle_count = max(1, math.ceil(args.duration_s * args.frequency_hz))
    handle, writer, log_path = create_log_writer(args.log_directory)
    started_at = time.monotonic()
    previous_targets = tuple(int(value) for value in start_positions)
    max_send_ms = 0.0
    max_lateness_ms = 0.0
    max_previous_errors = [0] * len(SERVO_IDS)
    deadline_misses = 0
    stale_frames = 0

    try:
        for cycle in range(cycle_count):
            scheduled_at = started_at + cycle * control_period_s
            remaining_s = scheduled_at - time.monotonic()
            if remaining_s > 0.0:
                time.sleep(remaining_s)

            send_started_at = time.monotonic()
            elapsed_s = cycle * control_period_s
            targets = build_sine_targets(
                start_positions,
                args.amplitude,
                elapsed_s,
                args.trajectory_period_s,
            )
            snapshot = telemetry.latest(0.5)
            command = controller.stream_positions(
                SERVO_IDS,
                targets,
                time_ms=args.move_time_ms,
            )
            send_finished_at = time.monotonic()

            send_elapsed_ms = (send_finished_at - send_started_at) * 1000.0
            lateness_ms = max(0.0, (send_started_at - scheduled_at) * 1000.0)
            deadline_missed = send_finished_at > scheduled_at + control_period_s
            max_send_ms = max(max_send_ms, send_elapsed_ms)
            max_lateness_ms = max(max_lateness_ms, lateness_ms)
            deadline_misses += int(deadline_missed)

            if snapshot is None:
                stale_frames += 1
                actual_positions = ("",) * len(SERVO_IDS)
                telemetry_age_ms: float | str = ""
                previous_errors = ("",) * len(SERVO_IDS)
            else:
                actual_positions = snapshot.positions
                telemetry_age_ms = (
                    send_started_at - snapshot.received_at
                ) * 1000.0
                previous_errors = tuple(
                    previous_target - actual
                    for previous_target, actual in zip(
                        previous_targets, actual_positions
                    )
                )
                for servo_id, error in zip(SERVO_IDS, previous_errors):
                    max_previous_errors[servo_id] = max(
                        max_previous_errors[servo_id], abs(error)
                    )

            row: list[object] = [
                cycle,
                elapsed_s,
                send_started_at - started_at,
                send_elapsed_ms,
                lateness_ms,
                int(deadline_missed),
                telemetry_age_ms,
            ]
            for target, actual, previous_error in zip(
                targets, actual_positions, previous_errors
            ):
                row.extend((target, actual, previous_error))
            writer.writerow(row)
            previous_targets = targets

            if cycle % max(1, round(args.frequency_hz)) == 0:
                print(
                    f"  t={elapsed_s:5.1f}s cycle={cycle:3d} "
                    f"send={send_elapsed_ms:.3f}ms "
                    f"late={lateness_ms:.3f}ms "
                    f"deadline_misses={deadline_misses}",
                    flush=True,
                )

        latest = telemetry.latest(0.5)
        if latest is None:
            raise RuntimeError("Fresh telemetry was unavailable after streaming")

        print("\n5 Hz stream summary:")
        print(f"  cycles: {cycle_count}")
        print(f"  maximum command-batch time: {max_send_ms:.3f} ms")
        print(f"  maximum schedule lateness: {max_lateness_ms:.3f} ms")
        print(f"  deadline misses: {deadline_misses}")
        print(f"  stale telemetry frames: {stale_frames}")
        print(f"  maximum one-cycle errors: {tuple(max_previous_errors)}")
        return latest.positions, log_path
    finally:
        handle.close()


def execute(args: argparse.Namespace) -> None:
    from servo.servo_APIs import ServoAPI

    with ServoAPI(
        port=args.port,
        baud_rate=args.baud_rate,
        timeout=DEFAULT_SERIAL_TIMEOUT_S,
    ) as api:
        if args.startup_wait_s > 0.0:
            print(f"Waiting {args.startup_wait_s:.1f} s for ESP32 startup.")
            time.sleep(args.startup_wait_s)

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
            print(f"Start positions: {starts}")
            print(
                f"Streaming at {args.frequency_hz:.3f} Hz for "
                f"{args.duration_s:.3f} s, amplitude=+/-{args.amplitude}."
            )
            final_stream_positions, log_path = run_stream(
                controller,
                telemetry,
                starts,
                args,
            )
            returned = run_phase(
                "FINAL RETURN",
                controller,
                telemetry,
                final_stream_positions,
                starts,
                move_time_ms=max(args.move_time_ms, 1000),
                arrival_timeout_s=args.arrival_timeout_s,
            )
            print("\nPASS: 5 Hz stream completed and all servos returned.")
            print(f"Log: {log_path}")
            print(
                "Return first-command misses: "
                f"{returned.first_command_misses or 'none'}"
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
            f"Plan: {args.frequency_hz:.3f} Hz for {args.duration_s:.3f} s, "
            f"amplitude=+/-{args.amplitude}, "
            f"trajectory period={args.trajectory_period_s:.3f} s."
        )
        print("Re-run with --execute only after removing all tendons.")
        return
    execute(args)


if __name__ == "__main__":
    main()
