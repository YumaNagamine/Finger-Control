"""Play predicted tendon excursions with feedback speed control.

The CSV is treated as a continuous excursion trajectory.  The player samples
that trajectory at ``CONTROL_HZ``, compares it with current servo telemetry,
and sends bounded speed commands through the project-owned servo session.
When enabled, calibrated force estimates from the AS5048A stream are written
to the same control trace.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from controller.csv_player.excursion_player import load_position_calibration
from controller.csv_player.force_estimator import CalibratedForceEstimator
from controller.csv_player.recorder import ControlTraceWriter
from controller.csv_player.runner import PlaybackConfig, SpeedTrajectoryRunner
from controller.csv_player.speed_controller import SpeedControlConfig, SpeedController
from controller.csv_player.trajectory import ExcursionTrajectory
from controller.servo.hardware_session import API_ROOT, ServoHardwareSession
from controller.servo_mapping import (
    ENCODER_SENSOR_IDS_BY_SERVO_ID,
    SERVO_IDS,
    SERVO_IDS_BY_TENDON,
    SERVO_NAMES_BY_ID,
    TENDONS,
)


# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------
PROJECT_ROOT = SRC_ROOT.parent
PREDICTION_CSV_PATH = (
    PROJECT_ROOT
    / "logs"
    / "dual_camera"
    / "excursion_predictions"
    / "dual_processed_controlTestSimpleExtension_20260727_184321_prediction_20260727.csv"
)
CALIBRATION_PATH = SRC_ROOT / "controller" / "excursion_servo_calibration.json"

EXECUTE = True
SERIAL_PORT = "COM3"
ENCODER_PORT = "COM5"
BAUD_RATE = 921600
SERIAL_TIMEOUT_S = 0.05
ENCODER_RATE_HZ = 70.0
RECORD_FORCE = True
FORCE_CALIBRATION_DIR = API_ROOT / "calibration" / "current"
ZERO_OFFSET_PATH = FORCE_CALIBRATION_DIR / "as5048a_unloaded_zero_offsets.json"

TIME_SCALE = 1.0
CONTROL_HZ = 50.0
KP = 0.5
MAX_SPEED = 500
MAX_ACCELERATION = 3000.0
POSITION_DEADBAND = 3.0
TELEMETRY_STALE_S = 0.2
TELEMETRY_WAIT_S = 3.0
DISPLAY_INTERVAL_S = 0.25
RETURN_TO_START = True
RETURN_TOLERANCE = 10.0
RETURN_TIMEOUT_S = 5.0
RETURN_STABLE_FRAMES = 3
START_POSITIONS: tuple[int, ...] | None = (2048, 2048, 2048, 2048, 2048, 2048)
OUTPUT_ROOT = PROJECT_ROOT / "logs" / "excursion_playback"


def _build_trajectory(csv_path: Path) -> ExcursionTrajectory:
    position_units_per_mm = load_position_calibration(CALIBRATION_PATH)
    return ExcursionTrajectory.from_csv(
        csv_path,
        position_units_per_mm,
        time_scale=TIME_SCALE,
    )


def _print_dry_run(trajectory: ExcursionTrajectory, csv_path: Path) -> None:
    if START_POSITIONS is None:
        raise ValueError("Dry-run requires START_POSITIONS")
    targets_at_start = trajectory.positions_at(0.0, START_POSITIONS)
    targets_at_end = trajectory.positions_at(trajectory.duration_s, START_POSITIONS)
    print("DRY RUN: no serial ports will be opened")
    print(f"CSV: {csv_path}")
    print(f"Duration: {trajectory.duration_s:.3f} s")
    print(
        "Mapping: "
        + ", ".join(
            f"{tendon}->servo {servo_id}"
            for tendon, servo_id in zip(TENDONS, SERVO_IDS_BY_TENDON)
        )
    )
    print("Start positions: " + ", ".join(map(str, START_POSITIONS)))
    for tendon, start, end in zip(TENDONS, targets_at_start, targets_at_end):
        print(f"  {tendon}: target {start:.1f} -> {end:.1f}")


def main() -> None:
    csv_path = PREDICTION_CSV_PATH.expanduser().resolve()
    trajectory = _build_trajectory(csv_path)
    if not EXECUTE:
        _print_dry_run(trajectory, csv_path)
        return

    if RECORD_FORCE and not ENCODER_PORT:
        raise ValueError("ENCODER_PORT is required when RECORD_FORCE is enabled")

    session_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=False)
    trace_path = session_dir / "control_trace.csv"
    manifest_path = session_dir / "session_manifest.json"
    trace = ControlTraceWriter(trace_path, SERVO_IDS_BY_TENDON)
    hardware = ServoHardwareSession(
        servo_ids=SERVO_IDS,
        servo_names=SERVO_NAMES_BY_ID,
        encoder_sensor_ids=tuple(
            ENCODER_SENSOR_IDS_BY_SERVO_ID[servo_id] for servo_id in SERVO_IDS
        ),
        servo_port=SERIAL_PORT,
        encoder_port=ENCODER_PORT if RECORD_FORCE else None,
        baudrate=BAUD_RATE,
        serial_timeout_s=SERIAL_TIMEOUT_S,
        encoder_rate_hz=ENCODER_RATE_HZ,
    )
    force_estimator = None
    start_positions_by_id = None
    completed = False
    error_text = None
    try:
        hardware.connect(startup_timeout_s=TELEMETRY_WAIT_S)
        feedback = hardware.wait_for_feedback(
            TELEMETRY_WAIT_S,
            require_encoder=RECORD_FORCE,
        )
        start_positions_by_id = hardware.position_by_servo_id(feedback)
        tendon_by_servo_id = dict(zip(SERVO_IDS_BY_TENDON, TENDONS))
        if RECORD_FORCE:
            force_estimator = CalibratedForceEstimator(
                calibration_dir=FORCE_CALIBRATION_DIR,
                zero_offset_path=ZERO_OFFSET_PATH,
                servo_ids=SERVO_IDS,
                tendon_by_servo_id=tendon_by_servo_id,
                telemetry_stale_after_s=TELEMETRY_STALE_S,
            )
        runner = SpeedTrajectoryRunner(
            trajectory=trajectory,
            servo_ids_by_tendon=SERVO_IDS_BY_TENDON,
            hardware=hardware,
            speed_controller=SpeedController(
                SpeedControlConfig(
                    kp=KP,
                    max_speed=MAX_SPEED,
                    max_acceleration=MAX_ACCELERATION,
                    position_deadband=POSITION_DEADBAND,
                )
            ),
            config=PlaybackConfig(
                control_hz=CONTROL_HZ,
                telemetry_stale_s=TELEMETRY_STALE_S,
                telemetry_wait_s=TELEMETRY_WAIT_S,
                display_interval_s=DISPLAY_INTERVAL_S,
                return_to_start=RETURN_TO_START,
                return_tolerance=RETURN_TOLERANCE,
                return_timeout_s=RETURN_TIMEOUT_S,
                return_stable_frames=RETURN_STABLE_FRAMES,
            ),
            recorder=trace,
            force_estimator=force_estimator,
        )
        print("Executing feedback speed trajectory. Press Ctrl-C to stop.")
        runner.run(start_positions_by_id)
        completed = True
    except BaseException as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        hardware.close()
        trace.close()
        manifest = {
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "input_csv_path": str(csv_path),
            "position_calibration_path": str(CALIBRATION_PATH.resolve()),
            "force_calibration_dir": str(FORCE_CALIBRATION_DIR.resolve()),
            "zero_offset_path": str(ZERO_OFFSET_PATH.resolve()),
            "servo_port": SERIAL_PORT,
            "encoder_port": ENCODER_PORT if RECORD_FORCE else None,
            "servo_ids": list(SERVO_IDS),
            "servo_names_by_id": list(SERVO_NAMES_BY_ID),
            "tendon_order": list(TENDONS),
            "servo_ids_by_tendon": list(SERVO_IDS_BY_TENDON),
            "control_hz": CONTROL_HZ,
            "speed_control": {
                "kp": KP,
                "max_speed": MAX_SPEED,
                "max_acceleration": MAX_ACCELERATION,
                "position_deadband": POSITION_DEADBAND,
            },
            "start_positions_by_id": start_positions_by_id,
            "completed": completed,
            "error": error_text,
            "outputs": {"control_trace": str(trace_path)},
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Saved playback session: {session_dir}")


if __name__ == "__main__":
    main()
