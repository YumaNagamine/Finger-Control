"""Reusable tendon-excursion CSV playback and servo telemetry support."""

from __future__ import annotations

import csv
import json
import math
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Callable, Sequence
from controller.servo_mapping import TENDONS


EXCURSION_COLUMNS = tuple(f"{tendon}_predicted_excursion_mm" for tendon in TENDONS)
SERVO_POSITION_MIN = 0
SERVO_POSITION_MAX = 4095


@dataclass(frozen=True)
class CommandFrame:
    elapsed_s: float
    positions: tuple[int, ...]
    move_time_ms: int


@dataclass(frozen=True)
class TelemetrySnapshot:
    received_at: float
    positions: tuple[int, ...]


@dataclass(frozen=True)
class PlaybackStatus:
    phase: str
    elapsed_s: float
    scheduled_s: float | None
    row_index: int | None
    move_time_ms: int
    target_positions: tuple[int, ...]
    actual_positions: tuple[int, ...] | None
    telemetry_age_s: float | None


StatusCallback = Callable[[PlaybackStatus], None]
HealthCheck = Callable[[], None]


def _six_values(values: Sequence[float | int], option_name: str) -> None:
    if len(values) != len(TENDONS):
        raise ValueError(
            f"{option_name} requires {len(TENDONS)} values in this order: "
            f"{', '.join(TENDONS)}"
        )


def load_position_calibration(json_path: Path) -> tuple[float, ...]:
    path = json_path.expanduser().resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Servo excursion calibration JSON not found: {path}")

    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid servo excursion calibration JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Servo excursion calibration must be a JSON object: {path}")
    if payload.get("schema_version") != 1:
        raise ValueError(f"Unsupported servo excursion calibration schema_version in {path}")

    raw_values = payload.get("position_units_per_mm")
    if not isinstance(raw_values, dict):
        raise ValueError(f"position_units_per_mm must be an object in {path}")

    expected = set(TENDONS)
    provided = set(raw_values)
    missing = sorted(expected - provided)
    extra = sorted(provided - expected)
    if missing or extra:
        raise ValueError(
            f"Invalid tendon keys in {path}: missing={missing}, extra={extra}"
        )

    values: list[float] = []
    for tendon in TENDONS:
        raw_value = raw_values[tendon]
        if isinstance(raw_value, bool) or not isinstance(raw_value, (int, float)):
            raise ValueError(f"{tendon} position_units_per_mm must be numeric in {path}")
        value = float(raw_value)
        if not math.isfinite(value) or value == 0.0:
            raise ValueError(
                f"{tendon} position_units_per_mm must be finite and non-zero in {path}"
            )
        values.append(value)

    return tuple(values)


class TelemetryMonitor:
    def __init__(self, api, read_timeout_s: float) -> None:
        self._api = api
        self._read_timeout_s = read_timeout_s
        self._stop_event = threading.Event()
        self._lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._latest: TelemetrySnapshot | None = None
        self._error: Exception | None = None

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Telemetry monitor has already been started")
        self._thread = threading.Thread(
            target=self._read_loop,
            name="servo-telemetry",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self._read_timeout_s + 1.0))

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


class ExcursionPlayer:
    def __init__(
        self,
        *,
        servo_ids: Sequence[int],
        position_units_per_mm: Sequence[float],
        time_scale: float = 1.0,
        max_lag_s: float = 0.5,
        live_display_interval_s: float = 0.1,
        telemetry_stale_s: float = 0.5,
    ) -> None:
        _six_values(servo_ids, "servo IDs")
        _six_values(position_units_per_mm, "position units per mm")
        if len(set(servo_ids)) != len(servo_ids):
            raise ValueError("servo IDs must not contain duplicates")
        if any(not 0 <= servo_id <= 5 for servo_id in servo_ids):
            raise ValueError("Every servo ID must be in the firmware range 0-5")
        if any(not math.isfinite(value) or value == 0.0 for value in position_units_per_mm):
            raise ValueError("Each position-units-per-mm value must be finite and non-zero")
        if time_scale <= 0.0 or not math.isfinite(time_scale):
            raise ValueError("time_scale must be finite and greater than zero")
        if max_lag_s < 0.0 or not math.isfinite(max_lag_s):
            raise ValueError("max_lag_s must be finite and non-negative")
        if live_display_interval_s <= 0.0 or not math.isfinite(live_display_interval_s):
            raise ValueError("live_display_interval_s must be finite and greater than zero")
        if telemetry_stale_s <= 0.0 or not math.isfinite(telemetry_stale_s):
            raise ValueError("telemetry_stale_s must be finite and greater than zero")

        self.servo_ids = tuple(int(value) for value in servo_ids)
        self.position_units_per_mm = tuple(float(value) for value in position_units_per_mm)
        self.time_scale = float(time_scale)
        self.max_lag_s = float(max_lag_s)
        self.live_display_interval_s = float(live_display_interval_s)
        self.telemetry_stale_s = float(telemetry_stale_s)

    @staticmethod
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
                    raise ValueError(
                        f"Prediction CSV row {row_number} contains a non-numeric value"
                    ) from exc
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
        self,
        elapsed_times: Sequence[float],
        excursions: Sequence[Sequence[float]],
        start_positions: Sequence[int],
    ) -> list[CommandFrame]:
        _six_values(start_positions, "start positions")
        if len(elapsed_times) != len(excursions) or not elapsed_times:
            raise ValueError("elapsed times and excursions must have the same non-zero length")

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
                    self.position_units_per_mm,
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
                interval_s = (elapsed_times[row_index + 1] - elapsed_s) / self.time_scale
                move_time_ms = max(0, round(interval_s * 1000.0))
            else:
                move_time_ms = 0

            frames.append(
                CommandFrame(
                    elapsed_s=elapsed_s / self.time_scale,
                    positions=positions,
                    move_time_ms=move_time_ms,
                )
            )

        return frames

    def load_and_build(self, csv_path: Path, start_positions: Sequence[int]) -> list[CommandFrame]:
        elapsed_times, excursions = self.load_excursions(csv_path)
        return self.build_command_frames(elapsed_times, excursions, start_positions)

    def read_start_positions(self, api, timeout_s: float) -> tuple[int, ...]:
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            frame = api.try_read_telemetry()
            if frame is not None:
                _six_values(frame.positions, "telemetry positions")
                return tuple(frame.positions[servo_id] for servo_id in self.servo_ids)
        raise TimeoutError(f"No valid six-servo telemetry received within {timeout_s:.1f} s")

    @staticmethod
    def create_telemetry_monitor(api, read_timeout_s: float) -> TelemetryMonitor:
        return TelemetryMonitor(api, read_timeout_s)

    def play(
        self,
        api,
        frames: Sequence[CommandFrame],
        telemetry_monitor: TelemetryMonitor,
        *,
        started_at: float | None = None,
        status_callback: StatusCallback | None = None,
        health_check: HealthCheck | None = None,
        print_live: bool = True,
    ) -> None:
        if not frames:
            raise ValueError("frames must not be empty")
        if started_at is None:
            started_at = time.monotonic()
        next_display_at = started_at
        last_row_index = len(frames) - 1

        for row_index, frame in enumerate(frames):
            deadline = started_at + frame.elapsed_s
            remaining_s = deadline - time.monotonic()
            if remaining_s > 0.0:
                time.sleep(remaining_s)

            now = time.monotonic()
            lag_s = now - deadline
            if lag_s > self.max_lag_s:
                raise RuntimeError(
                    f"Control timing lag at row {row_index} is {lag_s:.3f} s, "
                    f"exceeding max_lag_s={self.max_lag_s:.3f}"
                )

            telemetry_monitor.raise_if_failed()
            if health_check is not None:
                health_check()
            for servo_id, position in zip(self.servo_ids, frame.positions):
                api.set_position(servo_id, position, time_ms=frame.move_time_ms)

            now = time.monotonic()
            actual_positions, telemetry_age_s = telemetry_monitor.latest_positions(
                self.servo_ids,
                self.telemetry_stale_s,
            )
            status = PlaybackStatus(
                phase="playback",
                elapsed_s=now - started_at,
                scheduled_s=frame.elapsed_s,
                row_index=row_index,
                move_time_ms=frame.move_time_ms,
                target_positions=frame.positions,
                actual_positions=actual_positions,
                telemetry_age_s=telemetry_age_s,
            )
            if status_callback is not None:
                status_callback(status)
            if print_live and (now >= next_display_at or row_index == last_row_index):
                self.print_status(status)
                next_display_at = now + self.live_display_interval_s

    def return_to_start(
        self,
        api,
        initial_positions: Sequence[int],
        telemetry_monitor: TelemetryMonitor,
        *,
        experiment_started_at: float,
        move_time_ms: int,
        tolerance: int,
        timeout_s: float,
        status_callback: StatusCallback | None = None,
        health_check: HealthCheck | None = None,
        print_live: bool = True,
    ) -> None:
        _six_values(initial_positions, "initial positions")
        if move_time_ms < 0:
            raise ValueError("move_time_ms must be non-negative")
        if tolerance < 0:
            raise ValueError("tolerance must be non-negative")
        if timeout_s <= 0.0 or not math.isfinite(timeout_s):
            raise ValueError("timeout_s must be finite and greater than zero")

        targets = tuple(int(value) for value in initial_positions)
        telemetry_monitor.raise_if_failed()
        if health_check is not None:
            health_check()

        for servo_id, position in zip(self.servo_ids, targets):
            api.set_position(servo_id, position, time_ms=move_time_ms)

        deadline = time.monotonic() + timeout_s
        next_display_at = time.monotonic()
        while True:
            telemetry_monitor.raise_if_failed()
            if health_check is not None:
                health_check()
            now = time.monotonic()
            actual_positions, telemetry_age_s = telemetry_monitor.latest_positions(
                self.servo_ids,
                self.telemetry_stale_s,
            )
            status = PlaybackStatus(
                phase="return",
                elapsed_s=now - experiment_started_at,
                scheduled_s=None,
                row_index=None,
                move_time_ms=move_time_ms,
                target_positions=targets,
                actual_positions=actual_positions,
                telemetry_age_s=telemetry_age_s,
            )
            if status_callback is not None:
                status_callback(status)

            reached = actual_positions is not None and all(
                abs(target - actual) <= tolerance
                for target, actual in zip(targets, actual_positions)
            )
            if print_live and (now >= next_display_at or reached):
                self.print_status(status)
                next_display_at = now + self.live_display_interval_s
            if reached:
                return
            if now >= deadline:
                raise TimeoutError(
                    f"Servos did not return within {tolerance} position units "
                    f"before the {timeout_s:.1f} s timeout"
                )
            time.sleep(0.02)

    def print_summary(
        self,
        csv_path: Path,
        frames: Sequence[CommandFrame],
        start_positions: Sequence[int],
    ) -> None:
        print(f"CSV: {csv_path}")
        print(f"Rows: {len(frames)}, duration: {frames[-1].elapsed_s:.3f} s")
        mapping = ", ".join(
            f"{tendon}->servo {servo_id}"
            for tendon, servo_id in zip(TENDONS, self.servo_ids)
        )
        print(f"Mapping: {mapping}")
        print("Start positions: " + ", ".join(map(str, start_positions)))
        for index, tendon in enumerate(TENDONS):
            targets = [frame.positions[index] for frame in frames]
            print(f"  {tendon}: target range {min(targets)}..{max(targets)}")

    def print_simulation_commands(self, frames: Sequence[CommandFrame]) -> None:
        print("Simulation command preview (no serial commands will be sent):")
        for row_index, frame in enumerate(frames):
            targets = ", ".join(
                f"{tendon}(servo {servo_id})={position}"
                for tendon, servo_id, position in zip(TENDONS, self.servo_ids, frame.positions)
            )
            print(
                f"SIM row={row_index} t={frame.elapsed_s:.3f}s "
                f"move={frame.move_time_ms}ms {targets}"
            )

    def print_status(self, status: PlaybackStatus) -> None:
        status_parts: list[str] = []
        if status.actual_positions is None:
            status_parts.extend(
                f"{tendon}(servo {servo_id})={target}/N/A"
                for tendon, servo_id, target in zip(
                    TENDONS,
                    self.servo_ids,
                    status.target_positions,
                )
            )
        else:
            status_parts.extend(
                f"{tendon}(servo {servo_id})={target}/{actual}({target - actual:+d})"
                for tendon, servo_id, target, actual in zip(
                    TENDONS,
                    self.servo_ids,
                    status.target_positions,
                    status.actual_positions,
                )
            )

        telemetry_age = (
            "N/A" if status.telemetry_age_s is None else f"{status.telemetry_age_s:.3f}s"
        )
        row_text = "-" if status.row_index is None else str(status.row_index)
        print(
            f"{status.phase.upper()} row={row_text} t={status.elapsed_s:.3f}s "
            f"move={status.move_time_ms}ms telemetry_age={telemetry_age} "
            "target/actual(error): " + " | ".join(status_parts),
            flush=True,
        )
