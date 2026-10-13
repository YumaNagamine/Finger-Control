"""Run servo PID control from magnetic-encoder position feedback."""

from __future__ import annotations

import argparse
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional


CONTROLLER_ROOT = Path(__file__).resolve().parents[2]
SRC_ROOT = CONTROLLER_ROOT.parent
PROJECT_ROOT = SRC_ROOT.parent
API_ROOT = PROJECT_ROOT / "third_party" / "ServoHardwareAPI"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from controller.servo.magneticEncoderPos_control import (  # noqa: E402
    MagneticEncoderPositionControlConfig,
    MagneticEncoderPositionFeedback,
    MagneticEncoderPositionPIDController,
    PID,
    PIDConfig,
    encoder_position_from_sample,
)
from controller.servo.control_common import LatestSampleStore, ReaderTask  # noqa: E402


SERVO_NAMES = ["ED", "DI", "FDS", "FDP", "LUM", "PI"]


def _parse_args(argv: Optional[list[str]] = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--servo-port", required=True)
    parser.add_argument("--encoder-port", required=True)
    parser.add_argument("--servo-id", required=True, type=int, choices=range(1, 7))
    parser.add_argument(
        "--encoder-sensor-id",
        type=int,
        choices=range(1, 7),
        help="AS5048A sensor channel; defaults to the selected servo ID.",
    )
    parser.add_argument(
        "--target-position-deg",
        "--target-position",
        dest="target_position_deg",
        required=True,
        type=float,
        help="Target AS5048A relative multi-turn position in degrees.",
    )
    parser.add_argument("--kp", required=True, type=float)
    parser.add_argument("--ki", default=0.0, type=float)
    parser.add_argument("--kd", default=0.0, type=float)
    parser.add_argument("--max-speed", default=100.0, type=float)
    parser.add_argument("--hard-position-error-deg", required=True, type=float)
    parser.add_argument("--min-position-deg", type=float)
    parser.add_argument("--max-position-deg", type=float)
    parser.add_argument("--control-sign", choices=(-1.0, 1.0), default=1.0, type=float)
    parser.add_argument(
        "--encoder-direction",
        choices=(-1.0, 1.0),
        default=1.0,
        type=float,
        help="Sign applied to the encoder position before PID calculation.",
    )
    parser.add_argument("--control-hz", default=20.0, type=float)
    parser.add_argument("--encoder-rate-hz", default=70.0, type=float)
    parser.add_argument("--feedback-timeout-s", default=0.2, type=float)
    parser.add_argument("--startup-timeout-s", default=3.0, type=float)
    parser.add_argument("--seconds", default=10.0, type=float)
    parser.add_argument("--print-period-s", default=0.2, type=float)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args(argv)


def _validate_args(args: argparse.Namespace) -> None:
    positive_values = (
        args.control_hz,
        args.encoder_rate_hz,
        args.feedback_timeout_s,
        args.startup_timeout_s,
        args.seconds,
        args.print_period_s,
        args.max_speed,
        args.hard_position_error_deg,
    )
    if any(value <= 0.0 for value in positive_values):
        raise ValueError("timing, speed, and position-error limits must be positive")
    if args.min_position_deg is not None and args.max_position_deg is not None:
        if args.min_position_deg >= args.max_position_deg:
            raise ValueError("min-position-deg must be smaller than max-position-deg")


def _accept_servo(parsed: Any) -> Optional[dict]:
    if isinstance(parsed, dict) and parsed.get("kind") == "telemetry":
        return parsed
    return None


def _make_encoder_acceptor(sensor_id: int):
    def accept(parsed: Any) -> Optional[dict]:
        if not isinstance(parsed, (tuple, list)) or len(parsed) != 2:
            return None
        if parsed[0] not in ("sensor_data", "data"):
            return None
        sample = parsed[1]
        if not isinstance(sample, dict):
            return None
        sample_sensor_id = sample.get("sensor_id")
        if sample_sensor_id is not None and int(sample_sensor_id) != sensor_id:
            return None
        return sample

    return accept


def run(args: argparse.Namespace) -> int:
    if not API_ROOT.is_dir():
        raise FileNotFoundError(f"ServoHardwareAPI was not found: {API_ROOT}")
    sys.path.insert(0, str(API_ROOT / "python"))

    from drivers import (  # pylint: disable=import-outside-toplevel
        AS5048ABridgeDriver,
        SerialTransport,
        ServoControllerDriver,
        ServoSystem,
    )

    sensor_id = args.encoder_sensor_id or args.servo_id
    system = ServoSystem(
        count=6,
        servo_ids=[1, 2, 3, 4, 5, 6],
        names=SERVO_NAMES,
        as5048a_sensor_map=[1, 2, 3, 4, 5, 6],
    )
    servo_index = system.index_for_servo_id(args.servo_id)
    if servo_index is None:
        raise ValueError(f"Servo ID is not configured: {args.servo_id}")

    pid = PID(
        PIDConfig(
            kp=args.kp,
            ki=args.ki,
            kd=args.kd,
            output_min=-args.max_speed,
            output_max=args.max_speed,
            integral_min=-args.max_speed,
            integral_max=args.max_speed,
        )
    )
    controller = MagneticEncoderPositionPIDController(
        pid,
        MagneticEncoderPositionControlConfig(
            target_position_deg=args.target_position_deg,
            max_speed=args.max_speed,
            hard_position_error_deg=args.hard_position_error_deg,
            max_feedback_age_s=args.feedback_timeout_s,
            min_position_deg=args.min_position_deg,
            max_position_deg=args.max_position_deg,
            control_sign=args.control_sign,
        ),
    )

    servo_transport = None
    encoder_transport = None
    servo = None
    encoder = None
    servo_reader = None
    encoder_reader = None
    stop_event = threading.Event()
    try:
        servo_transport = SerialTransport(
            args.servo_port,
            baudrate=921600,
            timeout=min(0.05, args.feedback_timeout_s),
        ).open()
        encoder_transport = SerialTransport(
            args.encoder_port,
            baudrate=921600,
            timeout=min(0.05, args.feedback_timeout_s),
        ).open()
        servo = ServoControllerDriver(servo_transport, servo_system=system)
        encoder = AS5048ABridgeDriver(encoder_transport)
        servo_store = LatestSampleStore()
        encoder_store = LatestSampleStore()
        servo_reader = ReaderTask(
            servo.read_parsed_line,
            _accept_servo,
            servo_store,
            stop_event,
            name="magnetic-position-servo-reader",
        )
        encoder_reader = ReaderTask(
            encoder.read_parsed_line,
            _make_encoder_acceptor(sensor_id),
            encoder_store,
            stop_event,
            name="magnetic-position-encoder-reader",
        )

        encoder.stop_stream()
        encoder.set_rate_hz(args.encoder_rate_hz)
        encoder.start_all_stream()
        servo_reader.start()
        encoder_reader.start()
        servo_store.wait_for_sample(args.startup_timeout_s, stop_event=stop_event)
        encoder_store.wait_for_sample(args.startup_timeout_s, stop_event=stop_event)

        deadline = time.monotonic() + args.seconds
        next_tick = time.monotonic()
        last_print_at = 0.0
        period_s = 1.0 / args.control_hz
        while time.monotonic() < deadline:
            if servo_store.error:
                raise RuntimeError("Servo reader failed") from servo_store.error
            if encoder_store.error:
                raise RuntimeError("Magnetic encoder reader failed") from encoder_store.error

            servo_record = servo_store.snapshot()
            encoder_record = encoder_store.snapshot()
            if servo_record is None or encoder_record is None:
                raise MagneticEncoderPositionSafetyError(
                    "Servo or magnetic encoder feedback disappeared"
                )

            now = time.monotonic()
            servo_age_s = max(0.0, now - servo_record.received_at)
            encoder_age_s = max(0.0, now - encoder_record.received_at)
            telemetry = servo_record.value
            if not telemetry.get("valid", [True] * system.count)[servo_index]:
                raise MagneticEncoderPositionSafetyError(
                    "Servo telemetry is marked invalid"
                )

            sample = encoder_record.value
            position_deg = encoder_position_from_sample(
                sample,
                encoder_direction=args.encoder_direction,
            )
            output = controller.step(
                MagneticEncoderPositionFeedback(
                    position_deg=position_deg,
                    servo_age_s=servo_age_s,
                    encoder_age_s=encoder_age_s,
                    status=str(sample.get("status", "")),
                ),
                now=now,
            )
            servo.send_speed_by_id(args.servo_id, output.speed_command)

            if now - last_print_at >= args.print_period_s:
                sync_age_s = abs(encoder_record.received_at - servo_record.received_at)
                print(
                    "ID{} encoder={:.4f} deg target={:.4f} deg error={:+.4f} deg "
                    "speed={} sync={:.1f} ms".format(
                        args.servo_id,
                        position_deg,
                        args.target_position_deg,
                        output.position_error_deg,
                        output.speed_command,
                        sync_age_s * 1000.0,
                    )
                )
                last_print_at = now

            next_tick += period_s
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            else:
                next_tick = time.monotonic()
    finally:
        stop_event.set()
        if servo is not None:
            try:
                servo.send_speed_by_id(args.servo_id, 0)
            except Exception as exc:
                print(f"WARNING: failed to stop servo: {exc}", file=sys.stderr)
        if encoder is not None:
            try:
                encoder.stop_stream()
            except Exception:
                pass
        if servo_reader is not None:
            servo_reader.join(1.0)
        if encoder_reader is not None:
            encoder_reader.join(1.0)
        if encoder_transport is not None:
            encoder_transport.close()
        if servo_transport is not None:
            servo_transport.close()

    return 0


def main(argv: Optional[list[str]] = None) -> int:
    args = _parse_args(argv)
    _validate_args(args)
    if not args.execute:
        print("DRY RUN: no serial port will be opened and no servo command will be sent.")
        print(
            f"Plan: servo={args.servo_id}, sensor={args.encoder_sensor_id or args.servo_id}, "
            f"target_position={args.target_position_deg:.4f} deg, "
            f"kp={args.kp:.6g}, ki={args.ki:.6g}, kd={args.kd:.6g}, "
            f"control={args.control_hz:.1f} Hz, max_speed={args.max_speed:.1f}."
        )
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
