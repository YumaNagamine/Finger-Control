"""Check each servo by moving it away from and back to its start position.

The tendon/servo order is FDP, FDS, EI, DI, PI, LUM, mapped to servo IDs
0, 1, 2, 3, 4, 5. Positive ``--distance-mm`` follows the signed conversion
defined in ``excursion_servo_calibration.json``.

Without ``--execute`` this script only prints the planned commands. To move
the physical servos, explicitly pass ``--execute``.

Examples::

    python -m controller.check_servo_motion --distance-mm 2.0
    python -m controller.check_servo_motion --execute --distance-mm 2.0
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

from controller.csv_player.excursion_player import (
    SERVO_POSITION_MAX,
    SERVO_POSITION_MIN,
    TENDONS,
    TelemetryMonitor,
    load_position_calibration,
)


CALIBRATION_PATH = SRC_ROOT / "controller" / "excursion_servo_calibration.json"
SERVO_IDS = (0, 1, 2, 3, 4, 5)
SIMULATION_START_POSITIONS = (2048, 2048, 2048, 2048, 2048, 2048)

DEFAULT_DISTANCE_MM = 2.0
DEFAULT_MOVE_TIME_MS = 1000
DEFAULT_HOLD_TIME_S = 0.5
DEFAULT_TOLERANCE = 10
DEFAULT_ARRIVAL_TIMEOUT_S = 3.0
DEFAULT_TELEMETRY_WAIT_S = 3.0
DEFAULT_TELEMETRY_STALE_S = 0.5
DEFAULT_DISPLAY_INTERVAL_S = 0.1
DEFAULT_SERIAL_PORT = "COM7"
DEFAULT_BAUD_RATE = 921600
DEFAULT_SERIAL_TIMEOUT_S = 0.2


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Move each servo from its measured start position by a fixed "
            "distance, then return it to the start position."
        )
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Send commands to the physical servos (default: simulation only).",
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


def read_start_positions(api, timeout_s: float) -> tuple[int, ...]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        frame = api.try_read_telemetry()
        if frame is not None and len(frame.positions) == len(TENDONS):
            return tuple(frame.positions[servo_id] for servo_id in SERVO_IDS)
    raise TimeoutError(f"No valid six-servo telemetry received within {timeout_s:.1f} s")


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


def wait_for_positions(
    telemetry_monitor: TelemetryMonitor,
    servo_ids: Sequence[int],
    targets: Sequence[int],
    *,
    phase: str,
    tolerance: int,
    timeout_s: float,
) -> None:
    deadline = time.monotonic() + timeout_s
    next_display_at = 0.0

    while True:
        telemetry_monitor.raise_if_failed()
        actual_positions, telemetry_age_s = telemetry_monitor.latest_positions(
            servo_ids,
            DEFAULT_TELEMETRY_STALE_S,
        )
        now = time.monotonic()

        if now >= next_display_at:
            if actual_positions is None:
                values = " | ".join(
                    f"servo {servo_id}={target}/N/A"
                    for servo_id, target in zip(servo_ids, targets)
                )
            else:
                values = " | ".join(
                    f"servo {servo_id}={target}/{actual}({target - actual:+d})"
                    for servo_id, target, actual in zip(
                        servo_ids, targets, actual_positions
                    )
                )
            age = "N/A" if telemetry_age_s is None else f"{telemetry_age_s:.3f}s"
            print(
                f"{phase} telemetry_age={age} target/actual(error): {values}",
                flush=True,
            )
            next_display_at = now + DEFAULT_DISPLAY_INTERVAL_S

        if actual_positions is not None and all(
            abs(target - actual) <= tolerance
            for target, actual in zip(targets, actual_positions)
        ):
            return
        if now >= deadline:
            raise TimeoutError(
                f"{phase} did not arrive within {timeout_s:.1f} s "
                f"and tolerance +/-{tolerance} position units"
            )
        time.sleep(min(0.02, max(0.0, deadline - now)))


def restore_all(
    api,
    telemetry_monitor: TelemetryMonitor,
    start_positions: Sequence[int],
    *,
    move_time_ms: int,
    tolerance: int,
    timeout_s: float,
) -> None:
    print("Returning all servos to their measured start positions.")
    for servo_id, start in zip(SERVO_IDS, start_positions):
        api.set_position(servo_id, start, time_ms=move_time_ms)
    wait_for_positions(
        telemetry_monitor,
        SERVO_IDS,
        start_positions,
        phase="RESTORE",
        tolerance=tolerance,
        timeout_s=timeout_s,
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
        start_positions = read_start_positions(api, DEFAULT_TELEMETRY_WAIT_S)
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

        telemetry_monitor = TelemetryMonitor(api, DEFAULT_SERIAL_TIMEOUT_S)
        telemetry_monitor.start()
        completed = False
        try:
            print("Starting hardware check. Press Ctrl-C to return all servos and stop.")
            for tendon, servo_id, start, target in zip(
                TENDONS, SERVO_IDS, start_positions, target_positions
            ):
                print(f"\n[{tendon}] servo {servo_id}: moving {start} -> {target}")
                api.set_position(servo_id, target, time_ms=args.move_time_ms)
                wait_for_positions(
                    telemetry_monitor,
                    (servo_id,),
                    (target,),
                    phase=f"{tendon} OUT",
                    tolerance=args.tolerance,
                    timeout_s=args.arrival_timeout_s,
                )
                if args.hold_time_s > 0.0:
                    time.sleep(args.hold_time_s)

                print(f"[{tendon}] servo {servo_id}: returning {target} -> {start}")
                api.set_position(servo_id, start, time_ms=args.move_time_ms)
                wait_for_positions(
                    telemetry_monitor,
                    (servo_id,),
                    (start,),
                    phase=f"{tendon} RETURN",
                    tolerance=args.tolerance,
                    timeout_s=args.arrival_timeout_s,
                )
            completed = True
        finally:
            try:
                restore_all(
                    api,
                    telemetry_monitor,
                    start_positions,
                    move_time_ms=args.move_time_ms,
                    tolerance=args.tolerance,
                    timeout_s=args.arrival_timeout_s,
                )
            except Exception as restore_error:
                print(f"WARNING: automatic restoration failed: {restore_error}")
            finally:
                try:
                    api.stop_all()
                finally:
                    telemetry_monitor.stop()

        if completed:
            print("Servo motion check completed; all servos were returned and stopped.")


def main() -> None:
    args = parse_args()
    validate_args(args)
    position_units_per_mm = load_position_calibration(CALIBRATION_PATH)

    if not args.execute:
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
