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
"""

from __future__ import annotations

import csv
import math
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))


TENDONS = ("FDP", "FDS", "EI", "DI", "PI", "LUM")
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
    / "REPLACE_WITH_PREDICTION_CSV.csv"
)

# Required calibration. Set signed servo position units per 1 mm of tendon
# excursion. A positive/negative sign selects the winding direction.
POSITION_UNITS_PER_MM: tuple[float, ...] | None = (
    32.595,32.595,32.595,32.595,32.595,32.595
) # 直径40mmなので 1周125mm, 4096stepあるので0.030679mm/step, 32.595 step/mm

SERVO_IDS = (0, 1, 2, 3, 4, 5)

# For dry-run, specify six representative raw positions. During hardware
# execution, None reads the actual starting positions from servo telemetry.
START_POSITIONS: tuple[int, ...] | None = (2048, 2048, 2048, 2048, 2048, 2048)

TIME_SCALE = 1.0
EXECUTE = False
SERIAL_PORT = "COM7"
BAUD_RATE = 921600
SERIAL_TIMEOUT_S = 0.2
TELEMETRY_WAIT_S = 3.0
MAX_LAG_S = 0.5


@dataclass(frozen=True)
class CommandFrame:
    elapsed_s: float
    positions: tuple[int, ...]
    move_time_ms: int


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


def read_start_positions(api, timeout_s: float) -> tuple[int, ...]:
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        frame = api.try_read_telemetry()
        if frame is not None:
            _six_values(frame.positions, "telemetry positions")
            return tuple(frame.positions)
    raise TimeoutError(f"No valid six-servo telemetry received within {timeout_s:.1f} s")


def play_frames(api, frames: Sequence[CommandFrame], servo_ids: Sequence[int], max_lag_s: float) -> None:
    _six_values(servo_ids, "servo IDs")
    started_at = time.monotonic()

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

        for servo_id, position in zip(servo_ids, frame.positions):
            api.set_position(servo_id, position, time_ms=frame.move_time_ms)


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


def main() -> None:
    if POSITION_UNITS_PER_MM is None:
        raise ValueError(
            "Set POSITION_UNITS_PER_MM in the user settings block before running this script"
        )
    _six_values(SERVO_IDS, "SERVO_IDS")
    if len(set(SERVO_IDS)) != len(SERVO_IDS):
        raise ValueError("SERVO_IDS must not contain duplicates")
    if any(not 0 <= servo_id <= 5 for servo_id in SERVO_IDS):
        raise ValueError("Every SERVO_IDS value must be in the firmware range 0-5")
    if MAX_LAG_S < 0.0 or not math.isfinite(MAX_LAG_S):
        raise ValueError("MAX_LAG_S must be finite and non-negative")

    csv_path = PREDICTION_CSV_PATH.expanduser().resolve()
    elapsed_times, excursions = load_excursions(csv_path)

    if not EXECUTE:
        if START_POSITIONS is None:
            raise ValueError("Dry-run requires START_POSITIONS because no telemetry is opened")
        frames = build_command_frames(
            elapsed_times,
            excursions,
            START_POSITIONS,
            POSITION_UNITS_PER_MM,
            TIME_SCALE,
        )
        print("DRY RUN: no serial commands will be sent")
        print_summary(csv_path, frames, START_POSITIONS, SERVO_IDS)
        return

    # Delay importing pyserial-backed code until hardware execution is requested.
    from servo.servo_APIs import ServoAPI

    with ServoAPI(
        port=SERIAL_PORT,
        baud_rate=BAUD_RATE,
        timeout=SERIAL_TIMEOUT_S,
    ) as api:
        start_positions = (
            START_POSITIONS
            if START_POSITIONS is not None
            else read_start_positions(api, TELEMETRY_WAIT_S)
        )
        # Validate the complete trajectory before issuing its first motor command.
        frames = build_command_frames(
            elapsed_times,
            excursions,
            start_positions,
            POSITION_UNITS_PER_MM,
            TIME_SCALE,
        )
        print_summary(csv_path, frames, start_positions, SERVO_IDS)
        print("Executing servo trajectory. Press Ctrl-C to stop all servos.")
        try:
            play_frames(api, frames, SERVO_IDS, MAX_LAG_S)
        finally:
            api.stop_all()

    print("Playback completed; stop_all() was sent.")


if __name__ == "__main__":
    main()
