"""Verify ST3215 multi-turn position control on one unloaded physical servo."""

from __future__ import annotations

import argparse
import csv
import sys
import time
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path

SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from servo.control import TelemetryMonitor
from servo.servo_APIs import ServoAPI


DEFAULT_LOG_DIRECTORY = SRC_ROOT.parent / "logs" / "servo_multiturn_test"


@dataclass(frozen=True)
class BoundaryTestPlan:
    initial_position: int
    setup_position: int
    forward_target: int
    reverse_target: int


def wrapped_position(position: int) -> int:
    return int(position) % 4096


def shortest_delta_to_raw(position: int, target_raw: int) -> int:
    if not 0 <= target_raw <= 4095:
        raise ValueError("target_raw must be in the range 0-4095")
    current_raw = wrapped_position(position)
    return ((target_raw - current_raw + 2048) % 4096) - 2048


def build_boundary_test_plan(
    initial_position: int,
    *,
    start_raw: int = 4000,
    delta: int = 1000,
) -> BoundaryTestPlan:
    if not 0 <= start_raw <= 4095:
        raise ValueError("start_raw must be in the range 0-4095")
    if not 1 <= delta <= 32767:
        raise ValueError("delta must be in the range 1-32767")
    if start_raw + delta < 4096:
        raise ValueError("start_raw + delta must cross the 4095-to-0 boundary")

    setup_position = initial_position + shortest_delta_to_raw(
        initial_position,
        start_raw,
    )
    forward_target = setup_position + delta
    if not -28672 <= setup_position <= 28672:
        raise ValueError("setup position exceeds the multi-loop range")
    if not -28672 <= forward_target <= 28672:
        raise ValueError("forward target exceeds the multi-loop range")
    return BoundaryTestPlan(
        initial_position=initial_position,
        setup_position=setup_position,
        forward_target=forward_target,
        reverse_target=setup_position,
    )


class ResultLogger:
    def __init__(self, output_directory: Path) -> None:
        output_directory.mkdir(parents=True, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        self.path = output_directory / f"multiturn_position_{timestamp}.csv"
        self._handle = self.path.open("w", newline="", encoding="utf-8")
        self._writer = csv.DictWriter(
            self._handle,
            fieldnames=(
                "phase",
                "timestamp_ms",
                "target_extended",
                "actual_extended",
                "actual_wrapped",
                "error",
                "speed",
                "load",
            ),
        )
        self._writer.writeheader()

    def write(self, phase: str, target: int, snapshot, servo_id: int) -> None:
        actual = snapshot.positions[servo_id]
        speed = snapshot.speeds[servo_id] if snapshot.speeds is not None else ""
        self._writer.writerow(
            {
                "phase": phase,
                "timestamp_ms": snapshot.timestamp_ms,
                "target_extended": target,
                "actual_extended": actual,
                "actual_wrapped": wrapped_position(actual),
                "error": target - actual,
                "speed": speed,
                "load": snapshot.loads[servo_id],
            }
        )
        self._handle.flush()

    def close(self) -> None:
        self._handle.close()


def wait_for_target(
    telemetry: TelemetryMonitor,
    *,
    servo_id: int,
    target: int,
    tolerance: int,
    speed_tolerance: int,
    stable_frames: int,
    timeout_s: float,
    phase: str,
    logger: ResultLogger,
):
    deadline = time.monotonic() + timeout_s
    snapshot = telemetry.latest(timeout_s)
    sequence = snapshot.sequence if snapshot is not None else 0
    stable_count = 0

    while time.monotonic() < deadline:
        remaining_s = deadline - time.monotonic()
        try:
            snapshot = telemetry.wait_for_newer(sequence, min(0.25, remaining_s))
        except TimeoutError:
            continue
        sequence = snapshot.sequence
        logger.write(phase, target, snapshot, servo_id)

        actual = snapshot.positions[servo_id]
        speed = snapshot.speeds[servo_id] if snapshot.speeds is not None else 0
        if abs(target - actual) <= tolerance and abs(speed) <= speed_tolerance:
            stable_count += 1
        else:
            stable_count = 0
        if stable_count >= stable_frames:
            print(
                f"{phase}: target={target}, actual={actual}, "
                f"wrapped={wrapped_position(actual)}, speed={speed}"
            )
            return snapshot

    actual_text = "unavailable" if snapshot is None else str(snapshot.positions[servo_id])
    raise TimeoutError(
        f"{phase}: target {target} was not reached; actual={actual_text}"
    )


def validate_crossings(
    plan: BoundaryTestPlan,
    forward_actual: int,
    reverse_actual: int,
    tolerance: int,
) -> None:
    expected_forward_raw = wrapped_position(plan.forward_target)
    expected_reverse_raw = wrapped_position(plan.reverse_target)
    if abs(wrapped_position(forward_actual) - expected_forward_raw) > tolerance:
        raise AssertionError(
            "4095-to-0 crossing produced an unexpected wrapped position"
        )
    if abs(wrapped_position(reverse_actual) - expected_reverse_raw) > tolerance:
        raise AssertionError(
            "0-to-4095 reverse crossing produced an unexpected wrapped position"
        )
    if forward_actual <= plan.setup_position:
        raise AssertionError("forward crossing did not increase the extended position")
    if reverse_actual >= forward_actual:
        raise AssertionError("reverse crossing did not decrease the extended position")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Exercise both 4095-to-0 and 0-to-4095 crossings on one unloaded "
            "ST3215 servo. Remove the tendon before running."
        )
    )
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--port", default="COM3")
    parser.add_argument("--baud-rate", type=int, default=921600)
    parser.add_argument("--serial-timeout-s", type=float, default=0.2)
    parser.add_argument("--startup-wait-s", type=float, default=2.0)
    parser.add_argument("--servo-id", type=int, default=0)
    parser.add_argument("--start-raw", type=int, default=4000)
    parser.add_argument("--delta", type=int, default=1000)
    parser.add_argument("--time-ms", type=int, default=0)
    parser.add_argument("--tolerance", type=int, default=10)
    parser.add_argument("--speed-tolerance", type=int, default=5)
    parser.add_argument("--stable-frames", type=int, default=3)
    parser.add_argument("--timeout-s", type=float, default=8.0)
    parser.add_argument("--repeat-observation-s", type=float, default=0.5)
    parser.add_argument("--log-directory", default=str(DEFAULT_LOG_DIRECTORY))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    if not args.execute:
        print(
            "DRY RUN: no servo command was sent. Re-run with --execute only "
            "after removing the tendon."
        )
        print(
            f"Planned test: raw {args.start_raw} -> "
            f"{wrapped_position(args.start_raw + args.delta)} -> {args.start_raw}"
        )
        return
    if not 0 <= args.servo_id < 6:
        raise ValueError("--servo-id must be in the range 0-5")
    if not 0 <= args.time_ms <= 65535:
        raise ValueError("--time-ms must be in the range 0-65535")
    if args.tolerance < 0 or args.speed_tolerance < 0:
        raise ValueError("tolerances must be non-negative")
    if args.startup_wait_s < 0.0:
        raise ValueError("--startup-wait-s must be non-negative")
    if args.stable_frames <= 0 or args.timeout_s <= 0.0:
        raise ValueError("stable frames and timeout must be greater than zero")

    logger = ResultLogger(Path(args.log_directory).expanduser().resolve())
    with ServoAPI(
        port=args.port,
        baud_rate=args.baud_rate,
        timeout=args.serial_timeout_s,
    ) as api:
        telemetry = TelemetryMonitor(
            api,
            num_servos=6,
            read_timeout_s=args.serial_timeout_s,
        )
        telemetry.start()
        try:
            time.sleep(args.startup_wait_s)
            initial_snapshot = telemetry.wait_for_newer(0, args.timeout_s)
            api.stop_all()
            stopped_snapshot = telemetry.wait_for_newer(
                initial_snapshot.sequence,
                args.timeout_s,
            )
            initial_position = stopped_snapshot.positions[args.servo_id]
            plan = build_boundary_test_plan(
                initial_position,
                start_raw=args.start_raw,
                delta=args.delta,
            )
            print(
                f"Initial extended={plan.initial_position}, "
                f"raw={wrapped_position(plan.initial_position)}"
            )

            api.set_multiturn_position(
                args.servo_id,
                plan.setup_position,
                args.time_ms,
            )
            wait_for_target(
                telemetry,
                servo_id=args.servo_id,
                target=plan.setup_position,
                tolerance=args.tolerance,
                speed_tolerance=args.speed_tolerance,
                stable_frames=args.stable_frames,
                timeout_s=args.timeout_s,
                phase="SETUP",
                logger=logger,
            )

            api.set_multiturn_position(
                args.servo_id,
                plan.forward_target,
                args.time_ms,
            )
            forward_snapshot = wait_for_target(
                telemetry,
                servo_id=args.servo_id,
                target=plan.forward_target,
                tolerance=args.tolerance,
                speed_tolerance=args.speed_tolerance,
                stable_frames=args.stable_frames,
                timeout_s=args.timeout_s,
                phase="FORWARD_4095_TO_0",
                logger=logger,
            )

            api.set_multiturn_position(
                args.servo_id,
                plan.forward_target,
                args.time_ms,
            )
            time.sleep(args.repeat_observation_s)
            repeated_snapshot = telemetry.latest(args.timeout_s)
            if repeated_snapshot is None:
                raise RuntimeError("Telemetry became unavailable after repeated target")
            logger.write(
                "REPEATED_ABSOLUTE",
                plan.forward_target,
                repeated_snapshot,
                args.servo_id,
            )
            if abs(
                repeated_snapshot.positions[args.servo_id] - plan.forward_target
            ) > args.tolerance:
                raise AssertionError("Repeated absolute target caused position drift")

            api.set_multiturn_position(
                args.servo_id,
                plan.reverse_target,
                args.time_ms,
            )
            reverse_snapshot = wait_for_target(
                telemetry,
                servo_id=args.servo_id,
                target=plan.reverse_target,
                tolerance=args.tolerance,
                speed_tolerance=args.speed_tolerance,
                stable_frames=args.stable_frames,
                timeout_s=args.timeout_s,
                phase="REVERSE_0_TO_4095",
                logger=logger,
            )
            validate_crossings(
                plan,
                forward_snapshot.positions[args.servo_id],
                reverse_snapshot.positions[args.servo_id],
                args.tolerance,
            )

            api.set_multiturn_position(
                args.servo_id,
                plan.initial_position,
                args.time_ms,
            )
            wait_for_target(
                telemetry,
                servo_id=args.servo_id,
                target=plan.initial_position,
                tolerance=args.tolerance,
                speed_tolerance=args.speed_tolerance,
                stable_frames=args.stable_frames,
                timeout_s=args.timeout_s,
                phase="RETURN",
                logger=logger,
            )
            print("PASS: forward crossing, repeated target, and reverse crossing")
            print(f"Log: {logger.path}")
        finally:
            try:
                api.stop_all()
            finally:
                telemetry.stop()
                logger.close()


if __name__ == "__main__":
    main()
