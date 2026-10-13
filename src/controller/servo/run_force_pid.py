"""Run calibrated force PI/PID control through ServoHardwareAPI.

The command is dry-run unless ``--execute`` is supplied. Before using
``--execute``, verify the sign, calibration, mechanical limits, and emergency
stop procedure with the tendon unloaded.
"""

from __future__ import annotations

import argparse
import json
import sys
import threading
import time
from pathlib import Path
from typing import Any, Optional


CONTROLLER_ROOT = Path(__file__).resolve().parents[1]
SRC_ROOT = CONTROLLER_ROOT.parent
PROJECT_ROOT = SRC_ROOT.parent
API_ROOT = PROJECT_ROOT / "third_party" / "ServoHardwareAPI"
DEFAULT_FORCE_CONFIG_PATH = CONTROLLER_ROOT / "config_force_control.json"
if str(SRC_ROOT) not in sys.path:
    sys.path.insert(0, str(SRC_ROOT))

from controller.servo.force_control import (  # noqa: E402
    ForceControlConfig,
    ForceFeedback,
    ForcePIDController,
    ForceSafetyError,
    LatestSampleStore,
    PID,
    PIDConfig,
    ReaderTask,
)
from controller.servo.torque_calibration.storage import (  # noqa: E402
    intercept_for_model,
    load_torque_intercept_calibration,
)


SERVO_NAMES = ["ED", "DI", "FDS", "FDP", "LU", "PI"]


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--servo-port", required=True)
    parser.add_argument("--encoder-port", required=True)
    parser.add_argument("--servo-id", required=True, type=int, choices=range(1, 7))
    parser.add_argument("--calibration-file", required=True, type=Path)
    parser.add_argument(
        "--zero-offset-file",
        type=Path,
        default=API_ROOT / "calibration" / "current" / "as5048a_unloaded_zero_offsets.json",
    )
    parser.add_argument(
        "--intercept-calibration-file",
        type=Path,
        default=API_ROOT / "calibration" / "current" / "torque_intercept_calibration.json",
        help="Fixed-slope force-gauge torque-intercept calibration JSON.",
    )
    parser.add_argument("--no-zero-offset", action="store_true")
    parser.add_argument("--target-force-n", required=True, type=float)
    parser.add_argument("--kp", required=True, type=float)
    parser.add_argument("--ki", default=0.0, type=float)
    parser.add_argument("--kd", default=0.0, type=float)
    parser.add_argument("--max-speed", default=100.0, type=float)
    parser.add_argument("--hard-force-limit-n", required=True, type=float)
    parser.add_argument(
        "--force-config",
        type=Path,
        default=DEFAULT_FORCE_CONFIG_PATH,
        help="JSON file containing the minimum target force.",
    )
    parser.add_argument("--control-sign", choices=(-1.0, 1.0), default=1.0, type=float)
    parser.add_argument("--min-position", type=float)
    parser.add_argument("--max-position", type=float)
    parser.add_argument("--control-hz", default=20.0, type=float)
    parser.add_argument("--encoder-rate-hz", default=70.0, type=float)
    parser.add_argument("--feedback-timeout-s", default=0.2, type=float)
    parser.add_argument("--startup-timeout-s", default=3.0, type=float)
    parser.add_argument("--seconds", default=10.0, type=float)
    parser.add_argument("--print-period-s", default=0.2, type=float)
    parser.add_argument("--execute", action="store_true")
    return parser.parse_args()


def _path_from_project(path: Path) -> Path:
    return path if path.is_absolute() else PROJECT_ROOT / path


def _load_force_control_config(path: Path) -> float:
    config_path = _path_from_project(path)
    try:
        with config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
    except FileNotFoundError as exc:
        raise FileNotFoundError(
            f"Force-control config was not found: {config_path}"
        ) from exc
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid force-control JSON: {config_path}") from exc

    if not isinstance(config, dict) or not isinstance(config.get("default"), dict):
        raise ValueError("Force-control config must contain a default object")

    value = config["default"].get("min_target_force_n")
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError("min_target_force_n must be a number")
    min_target_force_n = float(value)
    if min_target_force_n < 0.0:
        raise ValueError("min_target_force_n must not be negative")
    return min_target_force_n


def _load_calibration(
    args: argparse.Namespace,
    load_calibration_points,
    calibration_model_from_points,
    load_zero_offsets,
):
    calibration_path = _path_from_project(args.calibration_file)
    points = load_calibration_points(
        calibration_path,
        args.servo_id,
        f"servo-{args.servo_id}",
    )
    model = calibration_model_from_points(points)
    if model["servo_id"] != args.servo_id:
        raise ValueError(
            "Calibration servo ID does not match --servo-id: "
            f"{model['servo_id']} != {args.servo_id}"
        )

    intercept_nmm = model["intercept_nmm"]
    intercept_path = _path_from_project(
        getattr(
            args,
            "intercept_calibration_file",
            API_ROOT / "calibration" / "current" / "torque_intercept_calibration.json",
        )
    )
    intercept_records = load_torque_intercept_calibration(intercept_path)
    calibrated_intercept = intercept_for_model(intercept_records, model)
    zero_path = _path_from_project(args.zero_offset_file)
    if calibrated_intercept is not None:
        intercept_nmm = calibrated_intercept
    elif not args.no_zero_offset and zero_path.exists():
        offsets = load_zero_offsets(zero_path)
        offset = offsets.get(model["sensor_id"])
        if offset is None or offset["servo_id"] != args.servo_id:
            raise ValueError("Zero-offset metadata does not match the selected servo")
        intercept_nmm = -model["slope_nmm_per_deg"] * offset["mean_delta_deg"]

    return model, intercept_nmm


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


def _validate_args(args: argparse.Namespace) -> None:
    if args.control_hz <= 0.0 or args.encoder_rate_hz <= 0.0:
        raise ValueError("control and encoder rates must be greater than zero")
    if args.seconds <= 0.0 or args.feedback_timeout_s <= 0.0:
        raise ValueError("seconds and feedback timeout must be greater than zero")
    if args.min_position is not None and args.max_position is not None:
        if args.min_position >= args.max_position:
            raise ValueError("min-position must be smaller than max-position")
    if args.target_force_n < args.min_target_force_n:
        raise ValueError(
            "target force must be at least the configured minimum target force"
        )
    if abs(args.target_force_n) >= args.hard_force_limit_n:
        raise ValueError("target force must be below the hard force limit")


def run(args: argparse.Namespace) -> int:
    if not API_ROOT.exists():
        raise FileNotFoundError(f"ServoHardwareAPI was not found: {API_ROOT}")
    sys.path.insert(0, str(API_ROOT / "python"))

    from drivers import (  # pylint: disable=import-outside-toplevel
        AS5048ABridgeDriver,
        SerialTransport,
        ServoControllerDriver,
        ServoSystem,
    )
    from drivers.as5048a.calibration_store import (  # pylint: disable=import-outside-toplevel
        load_calibration_points,
        load_zero_offsets,
    )
    from drivers.as5048a.torque import (  # pylint: disable=import-outside-toplevel
        calibration_model_from_points,
        compute_torque_snapshot,
    )

    model, intercept_nmm = _load_calibration(
        args,
        load_calibration_points,
        calibration_model_from_points,
        load_zero_offsets,
    )

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
    controller = ForcePIDController(
        pid,
        config=ForceControlConfig(
            target_force_n=args.target_force_n,
            max_speed=args.max_speed,
            hard_force_limit_n=args.hard_force_limit_n,
            min_target_force_n=args.min_target_force_n,
            max_feedback_age_s=args.feedback_timeout_s,
            min_position=args.min_position,
            max_position=args.max_position,
            control_sign=args.control_sign,
        ),
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
    stop_event = threading.Event()
    servo_reader = ReaderTask(
        servo.read_parsed_line,
        _accept_servo,
        servo_store,
        stop_event,
        name="force-pid-servo-reader",
    )
    encoder_reader = ReaderTask(
        encoder.read_parsed_line,
        _make_encoder_acceptor(model["sensor_id"]),
        encoder_store,
        stop_event,
        name="force-pid-encoder-reader",
    )

    previous_snapshot = None
    last_print_at = 0.0
    period_s = 1.0 / args.control_hz
    try:
        encoder.stop_stream()
        encoder.set_rate_hz(args.encoder_rate_hz)
        encoder.start_all_stream()
        servo_reader.start()
        encoder_reader.start()
        servo_store.wait_for_sample(args.startup_timeout_s)
        encoder_store.wait_for_sample(args.startup_timeout_s)

        deadline = time.monotonic() + args.seconds
        next_tick = time.monotonic()
        while time.monotonic() < deadline:
            if servo_store.error:
                raise RuntimeError("Servo reader failed") from servo_store.error
            if encoder_store.error:
                raise RuntimeError("Encoder reader failed") from encoder_store.error

            servo_record = servo_store.snapshot()
            encoder_record = encoder_store.snapshot()
            if servo_record is None or encoder_record is None:
                raise ForceSafetyError("Feedback disappeared during control")

            now = time.monotonic()
            servo_age_s = max(0.0, now - servo_record.received_at)
            encoder_age_s = max(0.0, now - encoder_record.received_at)
            telemetry = servo_record.value
            if not telemetry["valid"][servo_index]:
                raise ForceSafetyError("Servo telemetry is marked invalid")

            sync_age_s = abs(encoder_record.received_at - servo_record.received_at)
            snapshot = compute_torque_snapshot(
                sample=encoder_record.value,
                servo_id=args.servo_id,
                motor_name=model["motor_name"],
                servo_raw=telemetry["positions"][servo_index],
                counts_per_rev=model["servo_counts_per_rev"],
                servo_zero_raw=model["servo_zero_raw"],
                servo_direction=model["servo_direction"],
                encoder_direction=model["encoder_direction"],
                torque_zero_delta_deg=model["torque_zero_delta_deg"],
                calibration_slope=model["slope_nmm_per_deg"],
                calibration_intercept=intercept_nmm,
                calibration_angle_reference_deg=model["angle_reference_deg"],
                spring_k=0.0,
                moment_arm_mm=model["moment_arm_mm"],
                pull_angle_deg=model["pull_angle_deg"],
                telemetry_age_s=sync_age_s,
                telemetry_stale_after_s=args.feedback_timeout_s,
                previous_snapshot=previous_snapshot,
            )
            if snapshot is None or snapshot["force_n"] is None:
                raise ForceSafetyError("Force estimate is unavailable")
            previous_snapshot = snapshot

            output = controller.step(
                ForceFeedback(
                    force_n=float(snapshot["force_n"]),
                    servo_position=float(telemetry["positions"][servo_index]),
                    servo_age_s=servo_age_s,
                    encoder_age_s=encoder_age_s,
                ),
                now=now,
            )
            servo.send_speed_by_id(args.servo_id, output.speed_command)

            if now - last_print_at >= args.print_period_s:
                print(
                    "ID{} force={:.4f} N target={:.4f} N speed={} "
                    "position={} sync={:.1f} ms".format(
                        args.servo_id,
                        snapshot["force_n"],
                        args.target_force_n,
                        output.speed_command,
                        telemetry["positions"][servo_index],
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
        try:
            servo.send_speed_by_id(args.servo_id, 0)
        except Exception as exc:
            print(f"WARNING: failed to stop servo: {exc}", file=sys.stderr)
        try:
            encoder.stop_stream()
        except Exception:
            pass
        servo_reader.join(1.0)
        encoder_reader.join(1.0)
        encoder_transport.close()
        servo_transport.close()

    return 0


def main() -> int:
    args = _parse_args()
    args.min_target_force_n = _load_force_control_config(args.force_config)
    _validate_args(args)
    if not args.execute:
        print("DRY RUN: no serial port will be opened and no servo command will be sent.")
        print(
            f"Plan: servo={args.servo_id}, target_force={args.target_force_n:.4f} N, "
            f"min_target_force={args.min_target_force_n:.4f} N, "
            f"control={args.control_hz:.1f} Hz, max_speed={args.max_speed:.1f}."
        )
        return 0
    return run(args)


if __name__ == "__main__":
    raise SystemExit(main())
