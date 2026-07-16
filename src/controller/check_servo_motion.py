"""Check each servo by moving it away from and back to its start position.

The tendon/servo order is FDP, FDS, EI, DI, PI, LUM, mapped to servo IDs
5, 4, 2, 3, 1, 0. Positive ``--distance-mm`` follows the signed conversion
defined in ``excursion_servo_calibration.json``.

By default this script moves the physical servos. Pass ``--dry-run`` to print
the planned commands without opening the serial port or moving any servo.
So MAKE SURE THAT YOU CONNNECT THE SERVOS AND EACH TENDONS CORRECTLY.

Examples::

    python -m controller.check_servo_motion --distance-mm 2.0
    python -m controller.check_servo_motion --dry-run --distance-mm 2.0
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path
from typing import Sequence


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from controller.servo_mapping import (
    SERVO_IDS_BY_TENDON,
    TENDONS,
)
from controller.csv_player.excursion_player import (
    SERVO_POSITION_MAX,
    SERVO_POSITION_MIN,
    load_position_calibration,
)
from servo.control import (
    PositionControlConfig,
    PositionProgress,
    ReliablePositionController,
    RetryPolicy,
    TelemetryMonitor,
)


CALIBRATION_PATH = SRC_ROOT / "controller" / "excursion_servo_calibration.json"
SERVO_IDS = SERVO_IDS_BY_TENDON
SIMULATION_START_POSITIONS = (2048, 2048, 2048, 2048, 2048, 2048)

DEFAULT_DISTANCE_MM = 10.0
DEFAULT_MOVE_TIME_MS = 1000
DEFAULT_HOLD_TIME_S = 0.5
DEFAULT_TOLERANCE = 10
DEFAULT_ARRIVAL_TIMEOUT_S = 3.0
DEFAULT_TELEMETRY_WAIT_S = 3.0
DEFAULT_TELEMETRY_STALE_S = 0.5
DEFAULT_DISPLAY_INTERVAL_S = 0.1
DEFAULT_SERIAL_PORT = "COM3"
DEFAULT_BAUD_RATE = 921600
DEFAULT_SERIAL_TIMEOUT_S = 0.2
DEFAULT_SPEED_TOLERANCE = 5
DEFAULT_STABLE_FRAME_COUNT = 3
DEFAULT_ID_MAP_RESET_WAIT_S = 0.1
DEFAULT_POSITION_MODE_PREPARE_WAIT_S = 0.2
DEFAULT_POSITION_MODE_PRIME_WAIT_S = 0.2
DEFAULT_POSITION_MODE_PRIME_COMMAND_COUNT = 2
DEFAULT_START_OBSERVATION_S = 0.3
DEFAULT_START_MIN_DELTA = 3
DEFAULT_MAX_START_RETRIES = 1


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Move each servo from its measured start position by a fixed "
            "distance, then return it to the start position."
        )
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Print planned commands without opening the serial port or moving servos.",
    )
    parser.add_argument(
        "--distance-mm",
        type=float,
        default=DEFAULT_DISTANCE_MM,
        help=f"Excursion applied to each tendon (default: {DEFAULT_DISTANCE_MM} mm).",
    )
    parser.add_argument(
        "--move-time-ms",
        type=int,
        default=DEFAULT_MOVE_TIME_MS,
        help=f"Time for each outward/return movement (default: {DEFAULT_MOVE_TIME_MS} ms).",
    )
    parser.add_argument(
        "--hold-time-s",
        type=float,
        default=DEFAULT_HOLD_TIME_S,
        help=f"Pause at the moved position (default: {DEFAULT_HOLD_TIME_S} s).",
    )
    parser.add_argument(
        "--tolerance",
        type=int,
        default=DEFAULT_TOLERANCE,
        help=f"Arrival tolerance in position units (default: {DEFAULT_TOLERANCE}).",
    )
    parser.add_argument(
        "--arrival-timeout-s",
        type=float,
        default=DEFAULT_ARRIVAL_TIMEOUT_S,
        help=f"Maximum wait for each arrival (default: {DEFAULT_ARRIVAL_TIMEOUT_S} s).",
    )
    parser.add_argument("--port", default=DEFAULT_SERIAL_PORT)
    parser.add_argument("--baud-rate", type=int, default=DEFAULT_BAUD_RATE)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> None:
    if not math.isfinite(args.distance_mm) or args.distance_mm == 0.0:
        raise ValueError("--distance-mm must be finite and non-zero")
    if not 1 <= args.move_time_ms <= 65535:
        raise ValueError("--move-time-ms must be in the range 1-65535")
    if not math.isfinite(args.hold_time_s) or args.hold_time_s < 0.0:
        raise ValueError("--hold-time-s must be finite and non-negative")
    if args.tolerance < 0:
        raise ValueError("--tolerance must be non-negative")
    if not math.isfinite(args.arrival_timeout_s) or args.arrival_timeout_s <= 0.0:
        raise ValueError("--arrival-timeout-s must be finite and greater than zero")
    if args.arrival_timeout_s < args.move_time_ms / 1000.0:
        raise ValueError("--arrival-timeout-s must cover --move-time-ms")
    if args.baud_rate <= 0:
        raise ValueError("--baud-rate must be greater than zero")


def build_target_positions(
    start_positions: Sequence[int],
    position_units_per_mm: Sequence[float],
    distance_mm: float,
) -> tuple[int, ...]:
    if len(start_positions) != len(TENDONS):
        raise ValueError(f"Expected {len(TENDONS)} start positions")
    if len(position_units_per_mm) != len(TENDONS):
        raise ValueError(f"Expected {len(TENDONS)} calibration values")

    targets = tuple(
        round(start + distance_mm * units_per_mm)
        for start, units_per_mm in zip(start_positions, position_units_per_mm)
    )
    for tendon, servo_id, target in zip(TENDONS, SERVO_IDS, targets):
        if not SERVO_POSITION_MIN <= target <= SERVO_POSITION_MAX:
            raise ValueError(
                f"{tendon} (servo {servo_id}) target {target} is outside "
                f"{SERVO_POSITION_MIN}-{SERVO_POSITION_MAX}. Reduce --distance-mm "
                "or check the signed calibration."
            )
    return targets


def build_position_control_config(args: argparse.Namespace) -> PositionControlConfig:
    return PositionControlConfig(
        telemetry_stale_s=DEFAULT_TELEMETRY_STALE_S,
        telemetry_wait_s=DEFAULT_TELEMETRY_WAIT_S,
        id_map_reset_wait_s=DEFAULT_ID_MAP_RESET_WAIT_S,
        speed_init_wait_s=DEFAULT_POSITION_MODE_PREPARE_WAIT_S,
        prime_command_count=DEFAULT_POSITION_MODE_PRIME_COMMAND_COUNT,
        prime_interval_s=DEFAULT_POSITION_MODE_PRIME_WAIT_S,
        start_observation_s=DEFAULT_START_OBSERVATION_S,
        start_min_delta=DEFAULT_START_MIN_DELTA,
        position_tolerance=args.tolerance,
        speed_tolerance=DEFAULT_SPEED_TOLERANCE,
        stable_frame_count=DEFAULT_STABLE_FRAME_COUNT,
        arrival_timeout_s=args.arrival_timeout_s,
        max_start_retries=DEFAULT_MAX_START_RETRIES,
        reset_id_map_on_prepare=True,
        position_min=SERVO_POSITION_MIN,
        position_max=SERVO_POSITION_MAX,
    )


def print_plan(
    start_positions: Sequence[int],
    target_positions: Sequence[int],
    distance_mm: float,
    *,
    simulation: bool,
) -> None:
    mode = "SIMULATION" if simulation else "HARDWARE"
    print(f"Mode: {mode}; excursion: {distance_mm:+.3f} mm")
    for tendon, servo_id, start, target in zip(
        TENDONS, SERVO_IDS, start_positions, target_positions
    ):
        print(
            f"  {tendon} (servo {servo_id}): "
            f"{start} -> {target} -> {start} ({target - start:+d} units)"
        )


def prepare_position_control(controller: ReliablePositionController) -> None:
    print(
        "Preparing position control: "
        f"reset_ids=True, speed_init_servos=0-{len(TENDONS) - 1}, "
        f"position_servos={SERVO_IDS}, "
        f"prime_commands={DEFAULT_POSITION_MODE_PRIME_COMMAND_COUNT}"
    )
    controller.prepare(
        SERVO_IDS,
        force_init_servo_ids=tuple(range(len(TENDONS))),
    )


def read_prepared_positions(
    controller: ReliablePositionController,
) -> tuple[int, ...]:
    return tuple(controller.current_position(servo_id) for servo_id in SERVO_IDS)


def move_servo(
    controller: ReliablePositionController,
    servo_id: int,
    target_position: int,
    *,
    phase: str,
    move_time_ms: int,
) -> None:
    next_display_at = 0.0

    def display_progress(progress: PositionProgress) -> None:
        nonlocal next_display_at
        now = time.monotonic()
        if now < next_display_at:
            return
        print(
            f"{phase} phase={progress.phase}, attempt={progress.attempt}, "
            f"servo={progress.servo_id}, target={progress.target_position}, "
            f"actual={progress.actual_position}, "
            f"error={progress.target_position - progress.actual_position:+d}, "
            f"speed={progress.speed}, load={progress.load}, "
            f"stable={progress.stable_frames}/{DEFAULT_STABLE_FRAME_COUNT}",
            flush=True,
        )
        next_display_at = now + DEFAULT_DISPLAY_INTERVAL_S

    result = controller.move_and_wait(
        servo_id,
        target_position,
        time_ms=move_time_ms,
        retry_policy=RetryPolicy.ON_NO_START,
        progress_callback=display_progress,
    )
    print(
        f"{phase} stopped: servo={servo_id}, actual={result.final_position}, "
        f"target={target_position}, error={target_position - result.final_position:+d}, "
        f"speed={result.final_speed}, retries={result.retries}"
    )


def restore_all(
    controller: ReliablePositionController,
    start_positions: Sequence[int],
    *,
    move_time_ms: int,
) -> None:
    print("Returning all servos to their measured start positions.")
    for tendon, servo_id, start in zip(TENDONS, SERVO_IDS, start_positions):
        move_servo(
            controller,
            servo_id,
            start,
            phase=f"{tendon} RESTORE",
            move_time_ms=move_time_ms,
        )


def execute_check(
    args: argparse.Namespace,
    position_units_per_mm: Sequence[float],
) -> None:
    # Import pyserial-backed code only when hardware execution was requested.
    from servo.servo_APIs import ServoAPI

    with ServoAPI(
        port=args.port,
        baud_rate=args.baud_rate,
        timeout=DEFAULT_SERIAL_TIMEOUT_S,
    ) as api:
        telemetry_monitor = TelemetryMonitor(
            api,
            num_servos=len(TENDONS),
            read_timeout_s=DEFAULT_SERIAL_TIMEOUT_S,
        )
        controller = ReliablePositionController(
            api,
            telemetry_monitor,
            build_position_control_config(args),
        )
        telemetry_monitor.start()
        start_positions: tuple[int, ...] | None = None
        completed = False
        try:
            prepare_position_control(controller)
            start_positions = read_prepared_positions(controller)
            target_positions = build_target_positions(
                start_positions,
                position_units_per_mm,
                args.distance_mm,
            )
            print_plan(
                start_positions,
                target_positions,
                args.distance_mm,
                simulation=False,
            )
            print("Starting hardware check. Press Ctrl-C to return all servos and stop.")
            for tendon, servo_id, start, target in zip(
                TENDONS, SERVO_IDS, start_positions, target_positions
            ):
                print(f"\n[{tendon}] servo {servo_id}: moving {start} -> {target}")
                move_servo(
                    controller,
                    servo_id,
                    target,
                    phase=f"{tendon} OUT",
                    move_time_ms=args.move_time_ms,
                )
                if args.hold_time_s > 0.0:
                    time.sleep(args.hold_time_s)

                print(f"[{tendon}] servo {servo_id}: returning {target} -> {start}")
                move_servo(
                    controller,
                    servo_id,
                    start,
                    phase=f"{tendon} RETURN",
                    move_time_ms=args.move_time_ms,
                )
            completed = True
        finally:
            try:
                if start_positions is not None:
                    if not completed:
                        controller.stop_all()
                        prepare_position_control(controller)
                    restore_all(
                        controller,
                        start_positions,
                        move_time_ms=args.move_time_ms,
                    )
            except Exception as restore_error:
                print(f"WARNING: automatic restoration failed: {restore_error}")
            finally:
                try:
                    controller.stop_all()
                finally:
                    telemetry_monitor.stop()

        if completed:
            print("Servo motion check completed; all servos were returned and stopped.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    position_units_per_mm = load_position_calibration(CALIBRATION_PATH)

    if args.dry_run:
        target_positions = build_target_positions(
            SIMULATION_START_POSITIONS,
            position_units_per_mm,
            args.distance_mm,
        )
        print("DRY RUN: no serial port will be opened and no commands will be sent.")
        print_plan(
            SIMULATION_START_POSITIONS,
            target_positions,
            args.distance_mm,
            simulation=True,
        )
        return

    execute_check(args, position_units_per_mm)


if __name__ == "__main__":
    main()
