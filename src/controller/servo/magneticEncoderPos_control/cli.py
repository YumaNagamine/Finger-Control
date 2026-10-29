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
    target_group = parser.add_mutually_exclusive_group(required=True)
    target_group.add_argument(
        "--target-position-deg",
        "--target-position",
        dest="target_position_deg",
        type=float,
        help="Target AS5048A relative multi-turn position in degrees.",
    )
    target_group.add_argument(
        "--return-to-calibration-position",
        action="store_true",
        help="Load this servo's remembered calibration position and return to it.",
    )
    parser.add_argument("--kp", required=True, type=float)
    parser.add_argument("--ki", default=0.0, type=float)
    parser.add_argument("--kd", default=0.0, type=float)
    parser.add_argument("--max-speed", default=100.0, type=float)
    parser.add_argument("--hard-position-error-deg", required=True, type=float)
    parser.add_argument("--min-position-deg", type=float)
    parser.add_argument("--max-position-deg", type=float)
    parser.add_argument("--control-sign", choices=(-1.0, 1.0), type=float)
    parser.add_argument(
        "--encoder-direction",
        choices=(-1.0, 1.0),
        type=float,
        help="Sign applied to the encoder position before PID calculation.",
    )
    parser.add_argument(
        "--calibration-position-file",
        type=Path,
        help="JSON file containing remembered encoder positions.",
    )
    parser.add_argument(
        "--calibration-dir",
        type=Path,
        default=API_ROOT / "calibration" / "current",
        help="Calibration directory, used to find saved positions and legacy encoder direction.",
    )
    parser.add_argument("--control-hz", default=20.0, type=float)
    parser.add_argument("--encoder-rate-hz", default=70.0, type=float)
    parser.add_argument("--feedback-timeout-s", default=0.2, type=float)
    parser.add_argument("--startup-timeout-s", default=3.0, type=float)
    parser.add_argument("--seconds", default=10.0, type=float)
    parser.add_argument("--return-timeout-s", default=60.0, type=float)
    parser.add_argument("--target-tolerance-deg", default=1.0, type=float)
    parser.add_argument("--settle-time-s", default=0.25, type=float)
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
        args.return_timeout_s,
        args.target_tolerance_deg,
        args.settle_time_s,
        args.print_period_s,
        args.max_speed,
        args.hard_position_error_deg,
    )
    if any(value <= 0.0 for value in positive_values):
        raise ValueError("timing, speed, and position-error limits must be positive")
    if args.min_position_deg is not None and args.max_position_deg is not None:
        if args.min_position_deg >= args.max_position_deg:
            raise ValueError("min-position-deg must be smaller than max-position-deg")
    if args.control_sign not in (None, -1.0, 1.0):
        raise ValueError("control-sign must be either -1 or 1")
    if args.encoder_direction not in (None, -1.0, 1.0):
        raise ValueError("encoder-direction must be either -1 or 1")


def _legacy_encoder_direction(args: argparse.Namespace, sensor_id: int) -> float:
    calibration_path = (
        args.calibration_dir / f"spring_torque_calibration_servo_id_{args.servo_id}.csv"
    )
    if not calibration_path.is_file():
        raise ValueError(
            "Saved position predates encoder-direction metadata. Provide "
            "--encoder-direction or a calibration directory containing "
            f"{calibration_path.name}."
        )
    api_python = str(API_ROOT / "python")
    if api_python not in sys.path:
        sys.path.insert(0, api_python)
    from drivers.as5048a.calibration_store import (  # pylint: disable=import-outside-toplevel
        load_calibration_points,
    )
    from drivers.as5048a.torque import (  # pylint: disable=import-outside-toplevel
        calibration_model_from_points,
    )

    points = load_calibration_points(
        str(calibration_path), args.servo_id, f"servo-{args.servo_id}"
    )
    model = calibration_model_from_points(points)
    if int(model["sensor_id"]) != sensor_id:
        raise ValueError(
            "Saved position encoder channel does not match the calibration model: "
            f"{sensor_id} != {model['sensor_id']}"
        )
    return -1.0 if float(model["encoder_direction"]) < 0.0 else 1.0


def _resolve_target_settings(args: argparse.Namespace) -> tuple[float, int, float, float]:
    """Resolve either a direct target or a persisted calibration target."""

    sensor_id = args.encoder_sensor_id or args.servo_id
    encoder_direction = 1.0 if args.encoder_direction is None else args.encoder_direction
    control_sign = 1.0 if args.control_sign is None else args.control_sign
    if not args.return_to_calibration_position:
        return float(args.target_position_deg), sensor_id, encoder_direction, control_sign

    from controller.servo.torque_calibration.storage import (  # pylint: disable=import-outside-toplevel
        DEFAULT_ENCODER_POSITION_PATH,
        load_encoder_calibration_positions,
    )

    position_path = args.calibration_position_file or (
        Path(args.calibration_dir) / DEFAULT_ENCODER_POSITION_PATH.name
    )
    positions = load_encoder_calibration_positions(position_path)
    record = positions.get(args.servo_id)
    if record is None:
        raise ValueError(
            f"No remembered calibration position for servo {args.servo_id} in {position_path}"
        )
    saved_sensor_id = int(record["encoder_sensor_id"])
    if args.encoder_sensor_id is not None and args.encoder_sensor_id != saved_sensor_id:
        raise ValueError(
            "--encoder-sensor-id does not match the remembered calibration channel: "
            f"{args.encoder_sensor_id} != {saved_sensor_id}"
        )
    sensor_id = saved_sensor_id
    saved_encoder_direction = (
        float(record["encoder_direction"])
        if "encoder_direction" in record
        else _legacy_encoder_direction(args, sensor_id)
    )
    target_position_deg = float(record["encoder_position_deg"])
    if args.encoder_direction is not None:
        encoder_direction = args.encoder_direction
        target_position_deg *= encoder_direction / saved_encoder_direction
    else:
        encoder_direction = saved_encoder_direction
    if args.control_sign is not None:
        control_sign = args.control_sign
    else:
        control_sign = float(record.get("control_sign", 1.0))
    return (
        target_position_deg,
        sensor_id,
        encoder_direction,
        control_sign,
    )


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


def run(
    args: argparse.Namespace,
    target_settings: Optional[tuple[float, int, float, float]] = None,
) -> int:
    if target_settings is None:
        target_settings = _resolve_target_settings(args)
    target_position_deg, sensor_id, encoder_direction, control_sign = target_settings
    if not API_ROOT.is_dir():
        raise FileNotFoundError(f"ServoHardwareAPI was not found: {API_ROOT}")
    sys.path.insert(0, str(API_ROOT / "python"))

    from drivers import (  # pylint: disable=import-outside-toplevel
        AS5048ABridgeDriver,
        SerialTransport,
        ServoControllerDriver,
        ServoSystem,
    )

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
            target_position_deg=target_position_deg,
            max_speed=args.max_speed,
            hard_position_error_deg=args.hard_position_error_deg,
            max_feedback_age_s=args.feedback_timeout_s,
            min_position_deg=args.min_position_deg,
            max_position_deg=args.max_position_deg,
            control_sign=control_sign,
            allow_multiturn_uncertain=args.return_to_calibration_position,
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

        run_duration_s = (
            args.return_timeout_s
            if args.return_to_calibration_position
            else args.seconds
        )
        deadline = time.monotonic() + run_duration_s
        next_tick = time.monotonic()
        last_print_at = 0.0
        reached_since = None
        reached_target = False
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
                encoder_direction=encoder_direction,
                allow_multiturn_uncertain=args.return_to_calibration_position,
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
            speed_command = output.speed_command
            if args.return_to_calibration_position:
                if abs(output.position_error_deg) <= args.target_tolerance_deg:
                    speed_command = 0
                    if reached_since is None:
                        reached_since = now
                    elif now - reached_since >= args.settle_time_s:
                        reached_target = True
                else:
                    reached_since = None
            servo.send_speed_by_id(args.servo_id, speed_command)

            if now - last_print_at >= args.print_period_s:
                sync_age_s = abs(encoder_record.received_at - servo_record.received_at)
                print(
                    "ID{} encoder={:.4f} deg target={:.4f} deg error={:+.4f} deg "
                    "speed={} sync={:.1f} ms".format(
                        args.servo_id,
                        position_deg,
                        target_position_deg,
                        output.position_error_deg,
                        speed_command,
                        sync_age_s * 1000.0,
                    )
                )
                last_print_at = now

            if reached_target:
                print(
                    f"ID{args.servo_id} reached remembered calibration position "
                    f"{target_position_deg:.4f} deg."
                )
                break

            next_tick += period_s
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0.0:
                time.sleep(sleep_s)
            else:
                next_tick = time.monotonic()

        if args.return_to_calibration_position and not reached_target:
            raise TimeoutError(
                f"Servo {args.servo_id} did not reach remembered position within "
                f"{args.return_timeout_s:g} s"
            )
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
    target_settings = _resolve_target_settings(args)
    if not args.execute:
        print("DRY RUN: no serial port will be opened and no servo command will be sent.")
        print(
            f"Plan: servo={args.servo_id}, sensor={target_settings[1]}, "
            f"target_position={target_settings[0]:.4f} deg, "
            f"kp={args.kp:.6g}, ki={args.ki:.6g}, kd={args.kd:.6g}, "
            f"encoder_direction={target_settings[2]:+g}, "
            f"control_sign={target_settings[3]:+g}, "
            f"control={args.control_hz:.1f} Hz, max_speed={args.max_speed:.1f}."
        )
        return 0
    return run(args, target_settings)


if __name__ == "__main__":
    raise SystemExit(main())
