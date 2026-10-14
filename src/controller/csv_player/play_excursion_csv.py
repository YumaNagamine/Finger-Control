"""Play predicted tendon excursions using magnetic-encoder position feedback.

The CSV is treated as a continuous excursion trajectory. Each excursion is
converted to an encoder-angle target relative to the measured starting angle,
and the encoder position controller sends bounded servo-speed commands.
"""

from __future__ import annotations

import json
import sys
from datetime import datetime
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from controller.csv_player.encoder_calibration import (
    EncoderExcursionCalibration,
    load_encoder_excursion_calibration,
)
from controller.csv_player.playback_service import (
    EncoderPlaybackService,
    build_encoder_playback_config,
)
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
ENCODER_CALIBRATION_PATH = SRC_ROOT / "controller" / "excursion_encoder_calibration.json"

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
TELEMETRY_STALE_S = 0.2
TELEMETRY_WAIT_S = 3.0
DISPLAY_INTERVAL_S = 0.25
RETURN_TO_START = True
RETURN_TIMEOUT_S = 5.0
RETURN_STABLE_FRAMES = 3
OUTPUT_ROOT = PROJECT_ROOT / "logs" / "excursion_playback"


def _build_trajectory(csv_path: Path) -> ExcursionTrajectory:
    return ExcursionTrajectory.from_csv(csv_path, time_scale=TIME_SCALE)


def _print_dry_run(
    trajectory: ExcursionTrajectory,
    csv_path: Path,
    calibration: EncoderExcursionCalibration,
) -> None:
    excursions_at_end = trajectory.excursions_at(trajectory.duration_s)
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
    print("CSV initial excursion is treated as zero encoder-angle offset.")
    for index, (tendon, initial, end) in enumerate(
        zip(TENDONS, trajectory.initial_excursion_mm, excursions_at_end)
    ):
        angle_delta = (end - initial) * calibration.degrees_per_mm[index]
        print(
            f"  {tendon}: excursion {initial:.3f} -> {end:.3f} mm; "
            f"encoder target change {angle_delta:+.3f} deg"
        )


def main() -> None:
    csv_path = PREDICTION_CSV_PATH.expanduser().resolve()
    calibration = load_encoder_excursion_calibration(ENCODER_CALIBRATION_PATH)
    if calibration.calibration_status.lower() == "nominal":
        print(
            "WARNING: encoder degrees/mm uses the nominal 20 mm spool radius; "
            "tension-dependent calibration and pull direction should be verified."
        )
    trajectory = _build_trajectory(csv_path)
    if not EXECUTE:
        _print_dry_run(trajectory, csv_path, calibration)
        return

    if not ENCODER_PORT:
        raise ValueError("ENCODER_PORT is required for magnetic-encoder position control")

    session_dir = OUTPUT_ROOT / datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir.mkdir(parents=True, exist_ok=False)
    trace_path = session_dir / "control_trace.csv"
    manifest_path = session_dir / "session_manifest.json"
    playback_config = build_encoder_playback_config(
        calibration,
        control_hz=CONTROL_HZ,
        telemetry_stale_s=TELEMETRY_STALE_S,
        telemetry_wait_s=TELEMETRY_WAIT_S,
        display_interval_s=DISPLAY_INTERVAL_S,
        return_to_start=RETURN_TO_START,
        return_timeout_s=RETURN_TIMEOUT_S,
        return_stable_frames=RETURN_STABLE_FRAMES,
    )
    playback = EncoderPlaybackService(
        trajectory=trajectory,
        calibration=calibration,
        trace_path=trace_path,
        run_id=session_dir.name,
        force_calibration_dir=FORCE_CALIBRATION_DIR,
        zero_offset_path=ZERO_OFFSET_PATH,
        config=playback_config,
        record_force=RECORD_FORCE,
    )
    hardware = ServoHardwareSession(
        servo_ids=SERVO_IDS,
        servo_names=SERVO_NAMES_BY_ID,
        encoder_sensor_ids=tuple(
            ENCODER_SENSOR_IDS_BY_SERVO_ID[servo_id] for servo_id in SERVO_IDS
        ),
        servo_port=SERIAL_PORT,
        encoder_port=ENCODER_PORT,
        baudrate=BAUD_RATE,
        serial_timeout_s=SERIAL_TIMEOUT_S,
        encoder_rate_hz=ENCODER_RATE_HZ,
    )
    completed = False
    error_text = None
    try:
        hardware.connect(startup_timeout_s=TELEMETRY_WAIT_S)
        print("Executing magnetic-encoder position PID trajectory. Press Ctrl-C to stop.")
        playback.run(hardware)
        completed = True
    except BaseException as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        hardware.close()
        playback.close()
        manifest = {
            "run_id": session_dir.name,
            "timestamp_timebase": "time.monotonic seconds; shared within this run",
            "created_at": datetime.now().isoformat(timespec="seconds"),
            "input_csv_path": str(csv_path),
            "encoder_calibration_path": str(ENCODER_CALIBRATION_PATH.resolve()),
            "force_calibration_dir": str(FORCE_CALIBRATION_DIR.resolve()),
            "zero_offset_path": str(ZERO_OFFSET_PATH.resolve()),
            "servo_port": SERIAL_PORT,
            "encoder_port": ENCODER_PORT,
            "servo_ids": list(SERVO_IDS),
            "servo_names_by_id": list(SERVO_NAMES_BY_ID),
            "tendon_order": list(TENDONS),
            "servo_ids_by_tendon": list(SERVO_IDS_BY_TENDON),
            "control_hz": CONTROL_HZ,
            "encoder_position_control": {
                "controller": "magnetic_encoder_pid",
                "kp": calibration.kp,
                "ki": calibration.ki,
                "kd": calibration.kd,
                "encoder_degrees_per_mm": dict(
                    zip(TENDONS, calibration.degrees_per_mm)
                ),
                "encoder_direction": dict(
                    zip(TENDONS, calibration.encoder_directions)
                ),
                "max_speed": calibration.max_speed,
                "max_acceleration": calibration.max_acceleration,
                "hard_error_margin_deg": calibration.hard_error_margin_deg,
                "minimum_hard_error_deg": calibration.minimum_hard_error_deg,
                "return_tolerance_deg": calibration.return_tolerance_deg,
                "calibration_status": calibration.calibration_status,
            },
            "completed": completed,
            "error": error_text,
            "outputs": {"control_trace": str(trace_path)},
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Saved playback session: {session_dir}")


if __name__ == "__main__":
    main()
