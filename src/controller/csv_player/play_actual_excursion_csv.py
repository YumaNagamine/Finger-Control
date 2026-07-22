"""Experimentally play measured tendon excursions on the six servos.

This executable intentionally keeps the actual-excursion experiment isolated
from the regular predicted-excursion player.  It reads the smoothed measured
excursion columns written by ``predict_excursion_for_control.py`` and reuses
``ExcursionPlayer`` only for command-frame construction and timed playback.

The input is replayed relative to its first row::

    target_position = measured_start_position
                      + (actual_excursion - first_actual_excursion)
                      * position_units_per_mm

Set ``EXECUTE = True`` only after checking the dry-run command preview.  A
hardware run writes a structured servo trace, a tracking plot, and a manifest
under ``OUTPUT_ROOT``.  Plotting is performed after the servos have stopped so
that Matplotlib cannot disturb control timing.
"""

from __future__ import annotations

import csv
import datetime
import json
import math
import sys
import time
from pathlib import Path
from typing import Sequence


SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from controller.csv_player.excursion_player import (
    CommandFrame,
    ExcursionPlayer,
    PlaybackStatus,
    load_position_calibration,
)
from controller.servo_mapping import SERVO_IDS_BY_TENDON, TENDONS
from servo.control import (
    AccumulatedPositionControlConfig,
    PositionControlConfig,
    ReliablePositionController,
    TelemetryMonitor as PositionTelemetryMonitor,
)


# ---------------------------------------------------------------------------
# User settings
# All six-value tuples use this order: FDP, FDS, EI, DI, PI, LUM.
# ---------------------------------------------------------------------------
PROJECT_ROOT = SRC_ROOT.parent
EXCURSION_CSV_PATH = (
    PROJECT_ROOT
    / "logs"
    / "dual_camera"
    / "excursion_predictions"
    / "dual_processed_controlTest_20260708_111151_prediction_20260708.csv"
)
ACTUAL_EXCURSION_COLUMNS = tuple(
    f"{tendon}_actual_excursion_smoothed_mm" for tendon in TENDONS
)

CALIBRATION_PATH = SRC_ROOT / "controller" / "excursion_servo_calibration.json"
SERVO_IDS = SERVO_IDS_BY_TENDON

# Used only for dry-run command generation.  Hardware execution always reads
# the actual starting positions from servo telemetry.
START_POSITIONS: tuple[int, ...] | None = (2048, 2048, 2048, 2048, 2048, 2048)

TIME_SCALE = 1.0
EXECUTE = True
SERIAL_PORT = "COM3"
BAUD_RATE = 921600
SERIAL_TIMEOUT_S = 0.2
TELEMETRY_WAIT_S = 3.0
MAX_LAG_S = 0.5
LIVE_DISPLAY_INTERVAL_S = 0.1
TELEMETRY_STALE_S = 0.5

POSITION_MODE_PREPARE_WAIT_S = 0.2
POSITION_MODE_PRIME_WAIT_S = 0.2
POSITION_MODE_PRIME_COMMAND_COUNT = 2
START_OBSERVATION_S = 0.3
START_MIN_DELTA = 3
SPEED_TOLERANCE = 5
STABLE_FRAME_COUNT = 3
MAX_START_RETRIES = 1
WHEEL_SWITCH_TO_POSITION_THRESHOLD = 150
WHEEL_KP = 0.5
WHEEL_KD = 0.1
WHEEL_MIN_SPEED = 30
WHEEL_MAX_SPEED = 400
WHEEL_COMMAND_LIFETIME_MS = 200
WHEEL_TELEMETRY_TIMEOUT_S = 0.15
WHEEL_ARRIVAL_TIMEOUT_S = 20.0
WHEEL_STOP_TIMEOUT_S = 1.0


RETURN_TO_START = True
RETURN_TO_START_TIME_MS = 2000
RETURN_TO_START_TOLERANCE = 10
RETURN_TO_START_TIMEOUT_S = 5.0

OUTPUT_ROOT = PROJECT_ROOT / "logs" / "actual_excursion_playback"

# Optional preflight limits.  Leave as None until experiment-specific safe
# limits are known.  Accumulated control intentionally allows command frames
# outside the firmware's single-turn position range.
MAX_ABS_EXCURSION_FROM_START_MM: float | None = None
MAX_EXCURSION_STEP_MM: float | None = None
MAX_EXCURSION_RATE_MM_S: float | None = None

SERVO_POSITION_MIN = 0
SERVO_POSITION_MAX = 4095


def _six_values(values: Sequence[float | int], option_name: str) -> None:
    if len(values) != len(TENDONS):
        raise ValueError(
            f"{option_name} requires {len(TENDONS)} values in this order: "
            f"{', '.join(TENDONS)}"
        )


def _validate_optional_positive_limit(value: float | None, name: str) -> None:
    if value is not None and (not math.isfinite(value) or value <= 0.0):
        raise ValueError(f"{name} must be finite and greater than zero, or None")


def validate_settings() -> None:
    _six_values(SERVO_IDS, "SERVO_IDS")
    if len(set(SERVO_IDS)) != len(SERVO_IDS):
        raise ValueError("SERVO_IDS must not contain duplicates")
    if any(not 0 <= servo_id <= 5 for servo_id in SERVO_IDS):
        raise ValueError("Every SERVO_IDS value must be in the firmware range 0-5")
    if TIME_SCALE <= 0.0 or not math.isfinite(TIME_SCALE):
        raise ValueError("TIME_SCALE must be finite and greater than zero")
    if MAX_LAG_S < 0.0 or not math.isfinite(MAX_LAG_S):
        raise ValueError("MAX_LAG_S must be finite and non-negative")
    if LIVE_DISPLAY_INTERVAL_S <= 0.0 or not math.isfinite(LIVE_DISPLAY_INTERVAL_S):
        raise ValueError("LIVE_DISPLAY_INTERVAL_S must be finite and greater than zero")
    if TELEMETRY_STALE_S <= 0.0 or not math.isfinite(TELEMETRY_STALE_S):
        raise ValueError("TELEMETRY_STALE_S must be finite and greater than zero")
    if RETURN_TO_START:
        if not 1 <= RETURN_TO_START_TIME_MS <= 65535:
            raise ValueError("RETURN_TO_START_TIME_MS must be in the range 1-65535")
        if RETURN_TO_START_TOLERANCE < 0:
            raise ValueError("RETURN_TO_START_TOLERANCE must be non-negative")
        if RETURN_TO_START_TIMEOUT_S <= 0.0 or not math.isfinite(
            RETURN_TO_START_TIMEOUT_S
        ):
            raise ValueError("RETURN_TO_START_TIMEOUT_S must be finite and greater than zero")
        if RETURN_TO_START_TIMEOUT_S < RETURN_TO_START_TIME_MS / 1000.0:
            raise ValueError("RETURN_TO_START_TIMEOUT_S must cover RETURN_TO_START_TIME_MS")

    _validate_optional_positive_limit(
        MAX_ABS_EXCURSION_FROM_START_MM,
        "MAX_ABS_EXCURSION_FROM_START_MM",
    )
    _validate_optional_positive_limit(MAX_EXCURSION_STEP_MM, "MAX_EXCURSION_STEP_MM")
    _validate_optional_positive_limit(MAX_EXCURSION_RATE_MM_S, "MAX_EXCURSION_RATE_MM_S")


def load_actual_excursions(
    csv_path: Path,
) -> tuple[list[float], list[tuple[float, ...]]]:
    """Load the six smoothed measured-excursion series and preflight them."""

    path = csv_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Actual-excursion CSV not found: {path}")

    required = ("elapsed_s", *ACTUAL_EXCURSION_COLUMNS)
    numeric_rows: list[tuple[float, ...]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = [column for column in required if column not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(f"Actual-excursion CSV is missing columns: {missing}")

        for row_number, row in enumerate(reader, start=2):
            try:
                values = tuple(float(row[column]) for column in required)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Actual-excursion CSV row {row_number} contains a non-numeric value"
                ) from exc
            if not all(math.isfinite(value) for value in values):
                raise ValueError(
                    f"Actual-excursion CSV row {row_number} contains a non-finite value"
                )
            numeric_rows.append(values)

    if not numeric_rows:
        raise ValueError("Actual-excursion CSV contains no rows")

    raw_times = [row[0] for row in numeric_rows]
    if any(current < previous for previous, current in zip(raw_times, raw_times[1:])):
        raise ValueError("elapsed_s must be monotonically non-decreasing")

    first_time = raw_times[0]
    elapsed_times = [float(value - first_time) for value in raw_times]
    excursions = [tuple(row[1:]) for row in numeric_rows]
    validate_excursion_limits(elapsed_times, excursions)
    return elapsed_times, excursions


def validate_excursion_limits(
    elapsed_times: Sequence[float],
    excursions: Sequence[Sequence[float]],
) -> None:
    """Apply optional experiment-specific displacement, step, and rate limits."""

    first_excursion = excursions[0]
    for row_index, excursion_row in enumerate(excursions):
        for tendon_index, (tendon, excursion, initial) in enumerate(
            zip(TENDONS, excursion_row, first_excursion)
        ):
            displacement = excursion - initial
            if (
                MAX_ABS_EXCURSION_FROM_START_MM is not None
                and abs(displacement) > MAX_ABS_EXCURSION_FROM_START_MM
            ):
                raise ValueError(
                    f"Row {row_index}, {tendon}: excursion from start "
                    f"{displacement:+.3f} mm exceeds "
                    f"MAX_ABS_EXCURSION_FROM_START_MM="
                    f"{MAX_ABS_EXCURSION_FROM_START_MM:.3f}"
                )

            if row_index == 0:
                continue
            step = excursion - excursions[row_index - 1][tendon_index]
            if MAX_EXCURSION_STEP_MM is not None and abs(step) > MAX_EXCURSION_STEP_MM:
                raise ValueError(
                    f"Row {row_index}, {tendon}: excursion step {step:+.3f} mm exceeds "
                    f"MAX_EXCURSION_STEP_MM={MAX_EXCURSION_STEP_MM:.3f}"
                )

            if MAX_EXCURSION_RATE_MM_S is None:
                continue
            delta_time = elapsed_times[row_index] - elapsed_times[row_index - 1]
            if delta_time <= 0.0:
                if step != 0.0:
                    raise ValueError(
                        f"Row {row_index}, {tendon}: non-zero excursion step at a "
                        "duplicate elapsed_s value"
                    )
                continue
            rate = abs(step * TIME_SCALE / delta_time)
            if rate > MAX_EXCURSION_RATE_MM_S:
                raise ValueError(
                    f"Row {row_index}, {tendon}: excursion rate {rate:.3f} mm/s exceeds "
                    f"MAX_EXCURSION_RATE_MM_S={MAX_EXCURSION_RATE_MM_S:.3f}"
                )


class ExperimentalTraceRecorder:
    """Collect playback statuses and write them as a structured CSV."""

    def __init__(self) -> None:
        self.statuses: list[PlaybackStatus] = []

    def write(self, status: PlaybackStatus) -> None:
        self.statuses.append(status)

    def write_csv(self, output_path: Path) -> None:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        with output_path.open("w", encoding="utf-8", newline="") as csv_file:
            writer = csv.writer(csv_file)
            header = [
                "phase",
                "elapsed_s",
                "scheduled_s",
                "row_index",
                "move_time_ms",
                "telemetry_age_s",
            ]
            for tendon in TENDONS:
                header.extend(
                    [
                        f"{tendon}_target_position",
                        f"{tendon}_actual_position",
                        f"{tendon}_position_error",
                    ]
                )
            writer.writerow(header)

            for status in self.statuses:
                row: list[object] = [
                    status.phase,
                    f"{status.elapsed_s:.6f}",
                    "" if status.scheduled_s is None else f"{status.scheduled_s:.6f}",
                    "" if status.row_index is None else status.row_index,
                    status.move_time_ms,
                    ""
                    if status.telemetry_age_s is None
                    else f"{status.telemetry_age_s:.6f}",
                ]
                if status.actual_positions is None:
                    for target in status.target_positions:
                        row.extend([target, "", ""])
                else:
                    for target, actual in zip(
                        status.target_positions,
                        status.actual_positions,
                    ):
                        row.extend([target, actual, target - actual])
                writer.writerow(row)


def plot_servo_tracking(
    statuses: Sequence[PlaybackStatus],
    start_positions: Sequence[int],
    position_units_per_mm: Sequence[float],
    output_path: Path,
) -> None:
    """Plot target and telemetry-equivalent excursions for the playback phase."""

    import matplotlib

    matplotlib.use("Agg")

    import matplotlib.pyplot as plt

    playback_statuses = [status for status in statuses if status.phase == "playback"]
    if not playback_statuses:
        raise ValueError("No playback statuses are available for plotting")

    _six_values(start_positions, "start_positions")
    _six_values(position_units_per_mm, "position_units_per_mm")
    elapsed_values = [status.elapsed_s for status in playback_statuses]

    figure, axes = plt.subplots(
        nrows=3,
        ncols=2,
        figsize=(14, 10),
        sharex=True,
    )
    axes_flat = axes.flatten()

    for tendon_index, (axis, tendon, start, scale) in enumerate(
        zip(axes_flat, TENDONS, start_positions, position_units_per_mm)
    ):
        target_excursions = [
            (status.target_positions[tendon_index] - start) / scale
            for status in playback_statuses
        ]
        actual_excursions = [
            math.nan
            if status.actual_positions is None
            else (status.actual_positions[tendon_index] - start) / scale
            for status in playback_statuses
        ]

        axis.plot(
            elapsed_values,
            target_excursions,
            label="Target (actual-excursion input)",
            linewidth=1.8,
        )
        axis.plot(
            elapsed_values,
            actual_excursions,
            label="Servo telemetry",
            linewidth=1.8,
        )
        axis.axhline(0.0, color="black", linewidth=0.8)
        axis.set_title(tendon)
        axis.set_ylabel("Excursion from start [mm]")
        axis.grid(True, alpha=0.3)
        axis.legend(loc="best")

    for axis in axes_flat[-2:]:
        axis.set_xlabel("Elapsed time [s]")

    figure.suptitle(
        "Actual-excursion command vs servo telemetry",
        fontsize=14,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    figure.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(figure)



def create_position_controller(
    api,
    telemetry: PositionTelemetryMonitor,
) -> ReliablePositionController:
    return ReliablePositionController(
        api,
        telemetry,
        PositionControlConfig(
            telemetry_stale_s=TELEMETRY_STALE_S,
            telemetry_wait_s=TELEMETRY_WAIT_S,
            id_map_reset_wait_s=0.1,
            speed_init_wait_s=POSITION_MODE_PREPARE_WAIT_S,
            prime_command_count=POSITION_MODE_PRIME_COMMAND_COUNT,
            prime_interval_s=POSITION_MODE_PRIME_WAIT_S,
            start_observation_s=START_OBSERVATION_S,
            start_min_delta=START_MIN_DELTA,
            position_tolerance=RETURN_TO_START_TOLERANCE,
            speed_tolerance=SPEED_TOLERANCE,
            stable_frame_count=STABLE_FRAME_COUNT,
            arrival_timeout_s=TELEMETRY_WAIT_S,
            max_start_retries=MAX_START_RETRIES,
            reset_id_map_on_prepare=True,
            position_min=SERVO_POSITION_MIN,
            position_max=SERVO_POSITION_MAX,
        ),
        AccumulatedPositionControlConfig(
            switch_to_position_threshold=WHEEL_SWITCH_TO_POSITION_THRESHOLD,
            wheel_kp=WHEEL_KP,
            wheel_kd=WHEEL_KD,
            wheel_min_speed=WHEEL_MIN_SPEED,
            wheel_max_speed=WHEEL_MAX_SPEED,
            wheel_command_lifetime_ms=WHEEL_COMMAND_LIFETIME_MS,
            wheel_telemetry_timeout_s=WHEEL_TELEMETRY_TIMEOUT_S,
            wheel_arrival_timeout_s=WHEEL_ARRIVAL_TIMEOUT_S,
            wheel_stop_timeout_s=WHEEL_STOP_TIMEOUT_S,
        ),
    )


def prepare_position_control(
    controller: ReliablePositionController,
) -> tuple[int, ...]:
    print(
        "Preparing position control: "
        f"reset_ids=True, speed_init_servos=0-{len(TENDONS) - 1}, "
        f"position_servos={SERVO_IDS}, "
        f"prime_commands={POSITION_MODE_PRIME_COMMAND_COUNT}"
    )
    controller.prepare(
        SERVO_IDS,
        force_init_servo_ids=tuple(range(len(TENDONS))),
    )
    start_positions = tuple(
        controller.current_position(servo_id) for servo_id in SERVO_IDS
    )
    for servo_id, start in zip(SERVO_IDS, start_positions):
        controller.set_accumulated_reference(servo_id, start)
    return start_positions


def build_player(position_units_per_mm: Sequence[float]) -> ExcursionPlayer:
    return ExcursionPlayer(
        servo_ids=SERVO_IDS,
        position_units_per_mm=position_units_per_mm,
        time_scale=TIME_SCALE,
        max_lag_s=MAX_LAG_S,
        live_display_interval_s=LIVE_DISPLAY_INTERVAL_S,
        telemetry_stale_s=TELEMETRY_STALE_S,
        allow_out_of_range_positions=True,
    )


def play_accumulated_frames(
    player: ExcursionPlayer,
    controller: ReliablePositionController,
    frames: Sequence[CommandFrame],
    *,
    started_at: float,
    recorder: ExperimentalTraceRecorder,
) -> None:
    if not frames:
        raise ValueError("frames must not be empty")

    controller.begin_accumulated_control(SERVO_IDS)
    next_display_at = started_at
    try:
        for row_index, frame in enumerate(frames):
            scheduled_at = started_at + frame.elapsed_s
            remaining_s = scheduled_at - time.monotonic()
            if remaining_s > 0.0:
                time.sleep(remaining_s)

            now = time.monotonic()
            lag_s = now - scheduled_at
            if lag_s > MAX_LAG_S:
                raise RuntimeError(
                    f"Control timing lag at row {row_index} is {lag_s:.3f} s, "
                    f"exceeding MAX_LAG_S={MAX_LAG_S:.3f}"
                )

            command = controller.command_accumulated_positions(
                SERVO_IDS,
                frame.positions,
            )
            now = time.monotonic()
            status = PlaybackStatus(
                phase="playback",
                elapsed_s=now - started_at,
                scheduled_s=frame.elapsed_s,
                row_index=row_index,
                move_time_ms=frame.move_time_ms,
                target_positions=frame.positions,
                actual_positions=command.actual_positions,
                telemetry_age_s=None,
            )
            recorder.write(status)
            if now >= next_display_at or row_index == len(frames) - 1:
                player.print_status(status)
                next_display_at = now + LIVE_DISPLAY_INTERVAL_S

        settle_deadline = (
            time.monotonic()
            + controller.accumulated_config.wheel_arrival_timeout_s
        )
        while any(speed != 0 for speed in command.speed_commands):
            if time.monotonic() >= settle_deadline:
                raise TimeoutError(
                    "Servos did not enter the final position-mode switching region"
                )
            command = controller.command_accumulated_positions(
                SERVO_IDS,
                frames[-1].positions,
            )
            now = time.monotonic()
            status = PlaybackStatus(
                phase="settling",
                elapsed_s=now - started_at,
                scheduled_s=None,
                row_index=None,
                move_time_ms=0,
                target_positions=frames[-1].positions,
                actual_positions=command.actual_positions,
                telemetry_age_s=None,
            )
            recorder.write(status)
            if now >= next_display_at:
                player.print_status(status)
                next_display_at = now + LIVE_DISPLAY_INTERVAL_S

        controller.end_accumulated_control(
            SERVO_IDS,
            frames[-1].positions,
        )
    except Exception:
        controller.stop_all()
        raise


def move_all_accumulated_and_wait(
    player: ExcursionPlayer,
    controller: ReliablePositionController,
    targets: Sequence[int],
    *,
    experiment_started_at: float,
    timeout_s: float,
    recorder: ExperimentalTraceRecorder,
) -> None:
    target_tuple = tuple(int(value) for value in targets)
    deadline = time.monotonic() + timeout_s
    next_display_at = time.monotonic()
    controller.begin_accumulated_control(SERVO_IDS)
    try:
        while True:
            if time.monotonic() >= deadline:
                raise TimeoutError(
                    f"Servos did not approach accumulated targets within {timeout_s:.1f} s"
                )
            command = controller.command_accumulated_positions(
                SERVO_IDS,
                target_tuple,
            )
            now = time.monotonic()
            reached_switch_region = all(speed == 0 for speed in command.speed_commands)
            status = PlaybackStatus(
                phase="return",
                elapsed_s=now - experiment_started_at,
                scheduled_s=None,
                row_index=None,
                move_time_ms=0,
                target_positions=target_tuple,
                actual_positions=command.actual_positions,
                telemetry_age_s=None,
            )
            recorder.write(status)
            if now >= next_display_at or reached_switch_region:
                player.print_status(status)
                next_display_at = now + LIVE_DISPLAY_INTERVAL_S
            if reached_switch_region:
                break

        controller.end_accumulated_control(SERVO_IDS, target_tuple)
    except Exception:
        controller.stop_all()
        raise


def write_manifest(
    output_path: Path,
    *,
    session_dir: Path,
    csv_path: Path,
    position_units_per_mm: Sequence[float],
    start_positions: Sequence[int] | None,
    playback_completed: bool,
    returned_to_start: bool,
    error_text: str | None,
    trace_path: Path,
    plot_path: Path,
    plot_error: str | None,
) -> None:
    payload = {
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "experiment": "actual_excursion_smoothed_playback",
        "control_mode": "host_hybrid_accumulated_position",
        "accumulated_position_control": {
            "switch_to_position_threshold": WHEEL_SWITCH_TO_POSITION_THRESHOLD,
            "kp": WHEEL_KP,
            "kd": WHEEL_KD,
            "min_speed": WHEEL_MIN_SPEED,
            "max_speed": WHEEL_MAX_SPEED,
            "command_lifetime_ms": WHEEL_COMMAND_LIFETIME_MS,
            "telemetry_timeout_s": WHEEL_TELEMETRY_TIMEOUT_S,
            "arrival_timeout_s": WHEEL_ARRIVAL_TIMEOUT_S,
            "stop_timeout_s": WHEEL_STOP_TIMEOUT_S,
        },
        "session_dir": str(session_dir),
        "input_csv_path": str(csv_path),
        "actual_excursion_columns": list(ACTUAL_EXCURSION_COLUMNS),
        "servo_calibration_path": str(CALIBRATION_PATH.resolve()),
        "position_units_per_mm": {
            tendon: value for tendon, value in zip(TENDONS, position_units_per_mm)
        },
        "servo_ids": list(SERVO_IDS),
        "initial_positions": None if start_positions is None else list(start_positions),
        "time_scale": TIME_SCALE,
        "preflight_limits": {
            "max_abs_excursion_from_start_mm": MAX_ABS_EXCURSION_FROM_START_MM,
            "max_excursion_step_mm": MAX_EXCURSION_STEP_MM,
            "max_excursion_rate_mm_s": MAX_EXCURSION_RATE_MM_S,
        },
        "playback_completed": playback_completed,
        "returned_to_start": returned_to_start,
        "error": error_text,
        "plot_error": plot_error,
        "outputs": {
            "servo_trace": str(trace_path),
            "tracking_plot": str(plot_path),
        },
    }
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def main() -> None:
    validate_settings()
    csv_path = EXCURSION_CSV_PATH.expanduser().resolve()
    elapsed_times, excursions = load_actual_excursions(csv_path)
    position_units_per_mm = load_position_calibration(CALIBRATION_PATH)
    player = build_player(position_units_per_mm)

    print("Excursion source: measured, smoothed actual excursion")
    print("Columns: " + ", ".join(ACTUAL_EXCURSION_COLUMNS))

    print("Control: host-side hybrid accumulated position")
    print(
        "CAUTION: wheel-mode gains and switching thresholds require hardware tuning."
    )
    if not EXECUTE:
        if START_POSITIONS is None:
            raise ValueError("Dry-run requires START_POSITIONS because no telemetry is opened")
        frames = player.build_command_frames(elapsed_times, excursions, START_POSITIONS)
        print("DRY RUN: no serial commands will be sent")
        player.print_summary(csv_path, frames, START_POSITIONS)
        player.print_simulation_commands(frames)
        return

    session_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = OUTPUT_ROOT / session_name
    session_dir.mkdir(parents=True, exist_ok=False)
    trace_path = session_dir / "servo_trace.csv"
    plot_path = session_dir / "actual_excursion_tracking.png"
    manifest_path = session_dir / "session_manifest.json"

    recorder = ExperimentalTraceRecorder()
    start_positions: tuple[int, ...] | None = None
    playback_completed = False
    returned_to_start = False
    error_text: str | None = None

    try:
        # Delay importing the serial-backed API until hardware execution is requested.
        from servo.servo_APIs import ServoAPI

        with ServoAPI(
            port=SERIAL_PORT,
            baud_rate=BAUD_RATE,
            timeout=SERIAL_TIMEOUT_S,
        ) as api:
            telemetry = PositionTelemetryMonitor(
                api,
                num_servos=len(TENDONS),
                read_timeout_s=SERIAL_TIMEOUT_S,
            )
            controller = create_position_controller(api, telemetry)
            telemetry.start()
            try:
                start_positions = prepare_position_control(controller)
                frames = player.build_command_frames(
                    elapsed_times,
                    excursions,
                    start_positions,
                )
                player.print_summary(csv_path, frames, start_positions)
                experiment_started_at = time.monotonic()
                print(
                    "Executing accumulated actual-excursion trajectory. "
                    "Press Ctrl-C to stop all servos."
                )
                play_accumulated_frames(
                    player,
                    controller,
                    frames,
                    started_at=experiment_started_at,
                    recorder=recorder,
                )
                playback_completed = True

                if RETURN_TO_START:
                    move_all_accumulated_and_wait(
                        player,
                        controller,
                        start_positions,
                        experiment_started_at=experiment_started_at,
                        timeout_s=RETURN_TO_START_TIMEOUT_S,
                        recorder=recorder,
                    )
                    returned_to_start = True
            finally:
                try:
                    controller.stop_all()
                finally:
                    telemetry.stop()
    except BaseException as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        recorder.write_csv(trace_path)
        plot_error: str | None = None
        if start_positions is not None and recorder.statuses:
            try:
                plot_servo_tracking(
                    recorder.statuses,
                    start_positions,
                    position_units_per_mm,
                    plot_path,
                )
            except Exception as exc:
                plot_error = f"{type(exc).__name__}: {exc}"
                print(f"Could not create tracking plot: {plot_error}")

        write_manifest(
            manifest_path,
            session_dir=session_dir,
            csv_path=csv_path,
            position_units_per_mm=position_units_per_mm,
            start_positions=start_positions,
            playback_completed=playback_completed,
            returned_to_start=returned_to_start,
            error_text=error_text,
            trace_path=trace_path,
            plot_path=plot_path,
            plot_error=plot_error,
        )
        print(f"Saved actual-excursion session: {session_dir}")

    print("Playback completed; stop_all() was sent.")


if __name__ == "__main__":
    main()
