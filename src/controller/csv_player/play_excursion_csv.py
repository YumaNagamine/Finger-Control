"""Play predicted tendon excursions as time-aligned servo position commands.

The input is the CSV written by ``predict_excursion_for_control.py``.  Each
predicted excursion is treated as a displacement relative to its value in the
first CSV row::

    target_position = start_position
                      + excursion_delta_mm * position_units_per_mm

``position_units_per_mm`` is signed: use its sign to describe which servo
rotation direction winds the corresponding tendon.  It must be measured for
the actual spool/tendon mechanism; this script deliberately does not guess it.

The tendon order used by all six-value settings is: FDP, FDS, EI, DI, PI, LUM.

Set ``EXECUTE = True`` in the user settings block to send commands to the
physical servos. Leave it ``False`` for a dry-run simulation.
"""

from __future__ import annotations

import csv
import math
import sys
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from controller.servo_mapping import SERVO_IDS_BY_TENDON, TENDONS
from controller.csv_player.excursion_player import (
    ExcursionPlayer,
    load_position_calibration,
)
from servo.control import (
    PositionControlConfig,
    ReliablePositionController,
    TelemetryMonitor as PositionTelemetryMonitor,
)


EXCURSION_COLUMNS = tuple(f"{tendon}_predicted_excursion_mm" for tendon in TENDONS)
SERVO_POSITION_MIN = 0
SERVO_POSITION_MAX = 4095


# ---------------------------------------------------------------------------
# User settings
# All six-value tuples use this order: FDP, FDS, EI, DI, PI, LUM.
# ---------------------------------------------------------------------------
PROJECT_ROOT = SRC_ROOT.parent
PREDICTION_CSV_PATH = (
    PROJECT_ROOT
    / "logs"
    / "dual_camera"
    / "excursion_predictions"
    / "dual_processed_controlTest_20260708_111151_prediction_20260708.csv"
)

# Required signed tendon calibration is shared by all excursion players.
# Edit the JSON file when the measured mechanism calibration changes.
CALIBRATION_PATH = (
    SRC_ROOT / "controller" / "excursion_servo_calibration.json"
) # 直径40mmなので 1周125mm, 4096stepあるので0.030679mm/step, 32.595 step/mm

SERVO_IDS = SERVO_IDS_BY_TENDON

# Dry-run-only representative raw positions. Hardware execution always reads
# the actual starting positions from servo telemetry.
START_POSITIONS: tuple[int, ...] | None = (2048, 2048, 2048, 2048, 2048, 2048)

TIME_SCALE = 1.0
EXECUTE = False
SERIAL_PORT = "COM7"
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
RETURN_TO_START = True
RETURN_TO_START_TIME_MS = 2000
RETURN_TO_START_TOLERANCE = 10
RETURN_TO_START_TIMEOUT_S = 5.0


@dataclass(frozen=True)
class CommandFrame:
    elapsed_s: float
    positions: tuple[int, ...]
    move_time_ms: int


@dataclass(frozen=True)
class TelemetrySnapshot:
    received_at: float
    positions: tuple[int, ...]


class TelemetryMonitor:
    def __init__(self, api) -> None:
        self._api = api
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._latest: TelemetrySnapshot | None = None
        self._error: Exception | None = None

    def start(self) -> None:
        self._thread = threading.Thread(
            target=self._read_loop,
            name="servo-telemetry",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=SERIAL_TIMEOUT_S + 1.0)

    def raise_if_failed(self) -> None:
        with self._lock:
            error = self._error
        if error is not None:
            raise RuntimeError("Telemetry reader failed") from error

    def latest_positions(
        self,
        servo_ids: Sequence[int],
        max_age_s: float,
    ) -> tuple[tuple[int, ...] | None, float | None]:
        with self._lock:
            snapshot = self._latest
        if snapshot is None:
            return None, None

        age_s = max(0.0, time.monotonic() - snapshot.received_at)
        if age_s > max_age_s:
            return None, age_s
        return tuple(snapshot.positions[servo_id] for servo_id in servo_ids), age_s

    def _read_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                frame = self._api.try_read_telemetry()
                if frame is None:
                    continue
                _six_values(frame.positions, "telemetry positions")
                snapshot = TelemetrySnapshot(
                    received_at=time.monotonic(),
                    positions=tuple(frame.positions),
                )
                with self._lock:
                    self._latest = snapshot
        except Exception as exc:
            if not self._stop_event.is_set():
                with self._lock:
                    self._error = exc


def _six_values(values: Sequence[float | int], option_name: str) -> None:
    if len(values) != len(TENDONS):
        raise ValueError(
            f"{option_name} requires {len(TENDONS)} values in this order: "
            f"{', '.join(TENDONS)}"
        )


def load_excursions(csv_path: Path) -> tuple[list[float], list[tuple[float, ...]]]:
    if not csv_path.is_file():
        raise FileNotFoundError(f"Prediction CSV not found: {csv_path}")

    required = ("elapsed_s", *EXCURSION_COLUMNS)
    numeric_rows: list[tuple[float, ...]] = []
    with csv_path.open("r", encoding="utf-8-sig", newline="") as csv_file:
        reader = csv.DictReader(csv_file)
        missing = [column for column in required if column not in (reader.fieldnames or ())]
        if missing:
            raise ValueError(f"Prediction CSV is missing columns: {missing}")

        for row_number, row in enumerate(reader, start=2):
            try:
                values = tuple(float(row[column]) for column in required)
            except (TypeError, ValueError) as exc:
                raise ValueError(f"Prediction CSV row {row_number} contains a non-numeric value") from exc
            if not all(math.isfinite(value) for value in values):
                raise ValueError(f"Prediction CSV row {row_number} contains a non-finite value")
            numeric_rows.append(values)

    if not numeric_rows:
        raise ValueError("Prediction CSV contains no rows")

    raw_times = [row[0] for row in numeric_rows]
    if any(current < previous for previous, current in zip(raw_times, raw_times[1:])):
        raise ValueError("elapsed_s must be monotonically non-decreasing")

    first_time = raw_times[0]
    elapsed_times = [float(value - first_time) for value in raw_times]
    excursions = [row[1:] for row in numeric_rows]
    return elapsed_times, excursions


def build_command_frames(
    elapsed_times: Sequence[float],
    excursions: Sequence[Sequence[float]],
    start_positions: Sequence[int],
    position_units_per_mm: Sequence[float],
    time_scale: float,
) -> list[CommandFrame]:
    _six_values(start_positions, "start positions")
    _six_values(position_units_per_mm, "position units per mm")
    if time_scale <= 0.0 or not math.isfinite(time_scale):
        raise ValueError("time_scale must be a finite value greater than zero")
    if len(elapsed_times) != len(excursions) or not elapsed_times:
        raise ValueError("elapsed times and excursions must have the same non-zero length")
    if any(not math.isfinite(value) or value == 0.0 for value in position_units_per_mm):
        raise ValueError("Each position-units-per-mm value must be finite and non-zero")

    first_excursion = excursions[0]
    _six_values(first_excursion, "first excursion row")
    frames: list[CommandFrame] = []

    for row_index, (elapsed_s, excursion_row) in enumerate(zip(elapsed_times, excursions)):
        _six_values(excursion_row, f"excursion row {row_index}")
        positions = tuple(
            round(start + (excursion - initial) * scale)
            for start, excursion, initial, scale in zip(
                start_positions,
                excursion_row,
                first_excursion,
                position_units_per_mm,
            )
        )

        for tendon, position in zip(TENDONS, positions):
            if not SERVO_POSITION_MIN <= position <= SERVO_POSITION_MAX:
                raise ValueError(
                    f"Row {row_index}, {tendon}: target position {position} is outside "
                    f"the firmware range {SERVO_POSITION_MIN}-{SERVO_POSITION_MAX}. "
                    "Check the start position and signed position-units-per-mm calibration."
                )

        if row_index + 1 < len(elapsed_times):
            interval_s = (elapsed_times[row_index + 1] - elapsed_s) / time_scale
            move_time_ms = max(0, round(interval_s * 1000.0))
        else:
            move_time_ms = 0

        frames.append(
            CommandFrame(
                elapsed_s=elapsed_s / time_scale,
                positions=positions,
                move_time_ms=move_time_ms,
            )
        )

    return frames


def read_start_positions(
    api,
    timeout_s: float,
    servo_ids: Sequence[int],
) -> tuple[int, ...]:
    _six_values(servo_ids, "servo IDs")
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        frame = api.try_read_telemetry()
        if frame is not None:
            _six_values(frame.positions, "telemetry positions")
            return tuple(frame.positions[servo_id] for servo_id in servo_ids)
    raise TimeoutError(f"No valid six-servo telemetry received within {timeout_s:.1f} s")


def play_frames(
    api,
    frames: Sequence[CommandFrame],
    servo_ids: Sequence[int],
    max_lag_s: float,
    telemetry_monitor: TelemetryMonitor,
    display_interval_s: float,
    telemetry_stale_s: float,
) -> None:
    _six_values(servo_ids, "servo IDs")
    started_at = time.monotonic()
    next_display_at = started_at
    last_row_index = len(frames) - 1

    for row_index, frame in enumerate(frames):
        deadline = started_at + frame.elapsed_s
        remaining_s = deadline - time.monotonic()
        if remaining_s > 0.0:
            time.sleep(remaining_s)

        lag_s = time.monotonic() - deadline
        if lag_s > max_lag_s:
            raise RuntimeError(
                f"Control timing lag at row {row_index} is {lag_s:.3f} s, "
                f"exceeding --max-lag-s={max_lag_s:.3f}"
            )

        telemetry_monitor.raise_if_failed()
        for servo_id, position in zip(servo_ids, frame.positions):
            api.set_position(servo_id, position, time_ms=frame.move_time_ms)

        now = time.monotonic()
        if now >= next_display_at or row_index == last_row_index:
            actual_positions, telemetry_age_s = telemetry_monitor.latest_positions(
                servo_ids,
                telemetry_stale_s,
            )
            print_live_status(
                row_index,
                frame,
                servo_ids,
                actual_positions,
                telemetry_age_s,
            )
            next_display_at = now + display_interval_s


def return_to_start(
    api,
    telemetry_monitor: TelemetryMonitor,
    servo_ids: Sequence[int],
    start_positions: Sequence[int],
    move_time_ms: int,
    tolerance: int,
    timeout_s: float,
    display_interval_s: float,
    telemetry_stale_s: float,
) -> None:
    _six_values(servo_ids, "servo IDs")
    _six_values(start_positions, "start positions")

    print(f"Returning all servos to their initial positions over {move_time_ms} ms.")
    for servo_id, position in zip(servo_ids, start_positions):
        api.set_position(servo_id, position, time_ms=move_time_ms)

    deadline = time.monotonic() + timeout_s
    next_display_at = 0.0
    while True:
        telemetry_monitor.raise_if_failed()
        actual_positions, telemetry_age_s = telemetry_monitor.latest_positions(
            servo_ids,
            telemetry_stale_s,
        )
        now = time.monotonic()

        if now >= next_display_at:
            if actual_positions is None:
                status = " | ".join(
                    f"{tendon}(servo {servo_id})={target}/N/A"
                    for tendon, servo_id, target in zip(TENDONS, servo_ids, start_positions)
                )
            else:
                status = " | ".join(
                    f"{tendon}(servo {servo_id})={target}/{actual}({target - actual:+d})"
                    for tendon, servo_id, target, actual in zip(
                        TENDONS,
                        servo_ids,
                        start_positions,
                        actual_positions,
                    )
                )
            telemetry_age = (
                "N/A" if telemetry_age_s is None else f"{telemetry_age_s:.3f}s"
            )
            print(
                f"RETURN telemetry_age={telemetry_age} target/actual(error): {status}",
                flush=True,
            )
            next_display_at = now + display_interval_s

        if actual_positions is not None and all(
            abs(target - actual) <= tolerance
            for target, actual in zip(start_positions, actual_positions)
        ):
            print("All servos returned to their initial positions.")
            return

        if now >= deadline:
            raise TimeoutError(
                f"Servos did not return within {timeout_s:.1f} s "
                f"and tolerance +/-{tolerance} position units"
            )
        time.sleep(min(0.02, max(0.0, deadline - now)))


def print_summary(
    csv_path: Path,
    frames: Sequence[CommandFrame],
    start_positions: Sequence[int],
    servo_ids: Sequence[int],
) -> None:
    print(f"CSV: {csv_path}")
    print(f"Rows: {len(frames)}, duration: {frames[-1].elapsed_s:.3f} s")
    mapping = ", ".join(
        f"{tendon}->servo {servo_id}" for tendon, servo_id in zip(TENDONS, servo_ids)
    )
    print(f"Mapping: {mapping}")
    print("Start positions: " + ", ".join(map(str, start_positions)))
    for index, tendon in enumerate(TENDONS):
        targets = [frame.positions[index] for frame in frames]
        print(f"  {tendon}: target range {min(targets)}..{max(targets)}")


def print_simulation_commands(
    frames: Sequence[CommandFrame],
    servo_ids: Sequence[int],
) -> None:
    print("Simulation command preview (no serial commands will be sent):")
    for row_index, frame in enumerate(frames):
        targets = ", ".join(
            f"{tendon}(servo {servo_id})={position}"
            for tendon, servo_id, position in zip(TENDONS, servo_ids, frame.positions)
        )
        print(
            f"SIM row={row_index} t={frame.elapsed_s:.3f}s "
            f"move={frame.move_time_ms}ms {targets}"
        )


def print_live_status(
    row_index: int,
    frame: CommandFrame,
    servo_ids: Sequence[int],
    actual_positions: Sequence[int] | None,
    telemetry_age_s: float | None,
) -> None:
    status_parts: list[str] = []
    if actual_positions is None:
        status_parts.extend(
            f"{tendon}(servo {servo_id})={target}/N/A"
            for tendon, servo_id, target in zip(TENDONS, servo_ids, frame.positions)
        )
    else:
        status_parts.extend(
            f"{tendon}(servo {servo_id})={target}/{actual}({target - actual:+d})"
            for tendon, servo_id, target, actual in zip(
                TENDONS,
                servo_ids,
                frame.positions,
                actual_positions,
            )
        )

    telemetry_age = "N/A" if telemetry_age_s is None else f"{telemetry_age_s:.3f}s"
    print(
        f"LIVE row={row_index} t={frame.elapsed_s:.3f}s "
        f"move={frame.move_time_ms}ms telemetry_age={telemetry_age} "
        "target/actual(error): " + " | ".join(status_parts),
        flush=True,
    )


def prepare_position_control(api) -> tuple[int, ...]:
    telemetry = PositionTelemetryMonitor(
        api,
        num_servos=len(TENDONS),
        read_timeout_s=SERIAL_TIMEOUT_S,
    )
    controller = ReliablePositionController(
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
    )
    print(
        "Preparing position control: "
        f"reset_ids=True, speed_init_servos=0-{len(TENDONS) - 1}, "
        f"position_servos={SERVO_IDS}, "
        f"prime_commands={POSITION_MODE_PRIME_COMMAND_COUNT}"
    )
    telemetry.start()
    try:
        controller.prepare(
            SERVO_IDS,
            force_init_servo_ids=tuple(range(len(TENDONS))),
        )
        return tuple(controller.current_position(servo_id) for servo_id in SERVO_IDS)
    finally:
        telemetry.stop()


def main() -> None:
    _six_values(SERVO_IDS, "SERVO_IDS")
    if len(set(SERVO_IDS)) != len(SERVO_IDS):
        raise ValueError("SERVO_IDS must not contain duplicates")
    if any(not 0 <= servo_id <= 5 for servo_id in SERVO_IDS):
        raise ValueError("Every SERVO_IDS value must be in the firmware range 0-5")
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
        if RETURN_TO_START_TIMEOUT_S <= 0.0 or not math.isfinite(RETURN_TO_START_TIMEOUT_S):
            raise ValueError("RETURN_TO_START_TIMEOUT_S must be finite and greater than zero")
        if RETURN_TO_START_TIMEOUT_S < RETURN_TO_START_TIME_MS / 1000.0:
            raise ValueError("RETURN_TO_START_TIMEOUT_S must cover RETURN_TO_START_TIME_MS")
    position_units_per_mm = load_position_calibration(CALIBRATION_PATH)
    player = ExcursionPlayer(
        servo_ids=SERVO_IDS,
        position_units_per_mm=position_units_per_mm,
        time_scale=TIME_SCALE,
        max_lag_s=MAX_LAG_S,
        live_display_interval_s=LIVE_DISPLAY_INTERVAL_S,
        telemetry_stale_s=TELEMETRY_STALE_S,
    )



    csv_path = PREDICTION_CSV_PATH.expanduser().resolve()

    if not EXECUTE:
        if START_POSITIONS is None:
            raise ValueError("Dry-run requires START_POSITIONS because no telemetry is opened")
        frames = player.load_and_build(csv_path, START_POSITIONS)
        print("DRY RUN: no serial commands will be sent")
        player.print_summary(csv_path, frames, START_POSITIONS)
        player.print_simulation_commands(frames)
        return

    # Delay importing pyserial-backed code until hardware execution is requested.
    from servo.servo_APIs import ServoAPI

    with ServoAPI(
        port=SERIAL_PORT,
        baud_rate=BAUD_RATE,
        timeout=SERIAL_TIMEOUT_S,
    ) as api:
        start_positions = prepare_position_control(api)
        frames = player.load_and_build(csv_path, start_positions)
        player.print_summary(csv_path, frames, start_positions)
        telemetry_monitor = player.create_telemetry_monitor(api, SERIAL_TIMEOUT_S)
        telemetry_monitor.start()
        experiment_started_at = time.monotonic()
        try:
            print("Executing servo trajectory. Press Ctrl-C to stop all servos.")
            player.play(
                api,
                frames,
                telemetry_monitor,
                started_at=experiment_started_at,
            )
            if RETURN_TO_START:
                player.return_to_start(
                    api,
                    start_positions,
                    telemetry_monitor,
                    experiment_started_at=experiment_started_at,
                    move_time_ms=RETURN_TO_START_TIME_MS,
                    tolerance=RETURN_TO_START_TOLERANCE,
                    timeout_s=RETURN_TO_START_TIMEOUT_S,
                )
        finally:
            try:
                api.stop_all()
            finally:
                telemetry_monitor.stop()

    print("Playback completed; stop_all() was sent.")


if __name__ == "__main__":
    main()
