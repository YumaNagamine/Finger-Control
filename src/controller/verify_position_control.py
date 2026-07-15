"""Reproduce servoGUI_api position commands without starting the GUI.

This diagnostic follows the GUI sequence:

1. Reset the firmware's RAM ID map.
2. Put all six motors in wheel mode at speed zero.
3. Wait briefly.
4. Alternate one servo between center + offset and center - offset using
   position commands with time_ms=0.
5. Return the selected servo to its measured initial position and stop all.

Without ``--execute`` the script only prints the commands it would send.
"""

from __future__ import annotations

import argparse
import math
import sys
import time
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))


NUM_SERVOS = 6
POSITION_CENTER = 2048
POSITION_MIN = 0
POSITION_MAX = 4095

DEFAULT_SERVO_ID = 0
DEFAULT_OFFSET = 1000
DEFAULT_CYCLES = 2
DEFAULT_SETTLE_S = 1.0
DEFAULT_TELEMETRY_WAIT_S = 3.0
DEFAULT_DISPLAY_INTERVAL_S = 0.1
DEFAULT_SERIAL_PORT = "COM3"
DEFAULT_BAUD_RATE = 921600
DEFAULT_SERIAL_TIMEOUT_S = 0.03


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Reproduce the servoGUI_api position-mode command sequence and "
            "display measured position, speed, and load."
        )
    )
    parser.add_argument(
        "--execute",
        action="store_true",
        help="Open the serial port and move the physical servo.",
    )
    parser.add_argument(
        "--servo-id",
        type=int,
        default=DEFAULT_SERVO_ID,
        help=f"GUI motor/servo ID to test (default: {DEFAULT_SERVO_ID}).",
    )
    parser.add_argument(
        "--offset",
        type=int,
        default=DEFAULT_OFFSET,
        help=(
            "Position offset on either side of center 2048 "
            f"(default: {DEFAULT_OFFSET})."
        ),
    )
    parser.add_argument(
        "--cycles",
        type=int,
        default=DEFAULT_CYCLES,
        help=f"Number of positive/negative pairs (default: {DEFAULT_CYCLES}).",
    )
    parser.add_argument(
        "--settle-s",
        type=float,
        default=DEFAULT_SETTLE_S,
        help=f"Telemetry observation time after each command (default: {DEFAULT_SETTLE_S} s).",
    )
    parser.add_argument("--port", default=DEFAULT_SERIAL_PORT)
    parser.add_argument("--baud-rate", type=int, default=DEFAULT_BAUD_RATE)
    return parser.parse_args()


def validate_args(args: argparse.Namespace) -> tuple[int, int]:
    if not 0 <= args.servo_id < NUM_SERVOS:
        raise ValueError(f"--servo-id must be 0-{NUM_SERVOS - 1}")
    if args.offset <= 0:
        raise ValueError("--offset must be greater than zero")
    if args.cycles <= 0:
        raise ValueError("--cycles must be greater than zero")
    if not math.isfinite(args.settle_s) or args.settle_s <= 0.0:
        raise ValueError("--settle-s must be finite and greater than zero")
    if args.baud_rate <= 0:
        raise ValueError("--baud-rate must be greater than zero")

    positive_target = POSITION_CENTER + args.offset
    negative_target = POSITION_CENTER - args.offset
    for target in (positive_target, negative_target):
        if not POSITION_MIN <= target <= POSITION_MAX:
            raise ValueError(
                f"Target {target} is outside {POSITION_MIN}-{POSITION_MAX}; "
                "reduce --offset"
            )
    return positive_target, negative_target


def read_fresh_telemetry(api, timeout_s: float):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        frame = api.try_read_telemetry()
        if frame is not None and len(frame.positions) == NUM_SERVOS:
            return frame
    raise TimeoutError(
        f"No valid six-servo telemetry received within {timeout_s:.1f} s"
    )


def observe_telemetry(
    api,
    servo_id: int,
    target_position: int,
    duration_s: float,
    phase: str,
):
    deadline = time.monotonic() + duration_s
    next_display_at = 0.0
    latest_frame = None

    while time.monotonic() < deadline:
        frame = api.try_read_telemetry()
        if frame is None or len(frame.positions) != NUM_SERVOS:
            continue

        latest_frame = frame
        now = time.monotonic()
        if now < next_display_at:
            continue

        actual_position = frame.positions[servo_id]
        actual_load = frame.loads[servo_id]
        actual_speed = (
            None
            if frame.speeds is None or len(frame.speeds) != NUM_SERVOS
            else frame.speeds[servo_id]
        )
        speed_text = "N/A" if actual_speed is None else str(actual_speed)
        print(
            f"{phase}: servo={servo_id}, target={target_position}, "
            f"actual={actual_position}, "
            f"error={target_position - actual_position:+d}, "
            f"speed={speed_text}, load={actual_load}",
            flush=True,
        )
        next_display_at = now + DEFAULT_DISPLAY_INTERVAL_S

    if latest_frame is None:
        raise TimeoutError(f"No telemetry received during {phase}")
    return latest_frame


def print_dry_run(
    servo_id: int,
    positive_target: int,
    negative_target: int,
    cycles: int,
) -> None:
    print("DRY RUN: no serial port will be opened and no command will be sent.")
    print("RESET_IDS")
    for motor_id in range(NUM_SERVOS):
        print(f"{motor_id},0,1")
    for cycle in range(1, cycles + 1):
        print(f"cycle {cycle}: x,{servo_id},{positive_target},0")
        print(f"cycle {cycle}: x,{servo_id},{negative_target},0")
    print(
        f"Finally: x,{servo_id},<measured-initial-position>,0, "
        "then stop_all"
    )


def run_hardware(
    args: argparse.Namespace,
    positive_target: int,
    negative_target: int,
) -> None:
    from servo.servo_APIs import ServoAPI

    with ServoAPI(
        port=args.port,
        baud_rate=args.baud_rate,
        timeout=DEFAULT_SERIAL_TIMEOUT_S,
    ) as api:
        initial_position = None
        try:
            print("Sending the same initialization sequence as servoGUI_api.")
            api.reset_ids()
            time.sleep(0.1)
            for motor_id in range(NUM_SERVOS):
                api.set_speed(motor_id, 0, force_init=True)
            time.sleep(0.2)

            initial_frame = read_fresh_telemetry(api, DEFAULT_TELEMETRY_WAIT_S)
            initial_position = initial_frame.positions[args.servo_id]
            print(
                f"Ready: servo={args.servo_id}, initial={initial_position}, "
                f"targets=({positive_target}, {negative_target}), time_ms=0"
            )

            for cycle in range(1, args.cycles + 1):
                for label, target in (
                    ("POSITIVE", positive_target),
                    ("NEGATIVE", negative_target),
                ):
                    phase = f"cycle={cycle} {label}"
                    print(
                        f"Command: x,{args.servo_id},{target},0",
                        flush=True,
                    )
                    api.set_position(args.servo_id, target, time_ms=0)
                    observe_telemetry(
                        api,
                        args.servo_id,
                        target,
                        args.settle_s,
                        phase,
                    )
        finally:
            try:
                if initial_position is not None:
                    print(
                        f"Returning servo {args.servo_id} to initial "
                        f"position {initial_position}."
                    )
                    api.set_position(args.servo_id, initial_position, time_ms=0)
                    try:
                        observe_telemetry(
                            api,
                            args.servo_id,
                            initial_position,
                            args.settle_s,
                            "RETURN",
                        )
                    except Exception as exc:
                        print(f"WARNING: return telemetry check failed: {exc}")
            finally:
                api.stop_all()

    print("GUI-equivalent position verification completed; stop_all was sent.")


def main() -> None:
    args = parse_args()
    positive_target, negative_target = validate_args(args)

    if not args.execute:
        print_dry_run(
            args.servo_id,
            positive_target,
            negative_target,
            args.cycles,
        )
        return

    run_hardware(args, positive_target, negative_target)


if __name__ == "__main__":
    main()
