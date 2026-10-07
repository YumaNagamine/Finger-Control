"""Continuous time trajectories loaded from tendon-excursion CSV files."""

from __future__ import annotations

import bisect
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Sequence

from controller.servo_mapping import TENDONS


EXCURSION_COLUMNS = tuple(f"{tendon}_predicted_excursion_mm" for tendon in TENDONS)


def _six_values(values: Sequence[float | int], name: str) -> None:
    if len(values) != len(TENDONS):
        raise ValueError(
            f"{name} requires {len(TENDONS)} values in this order: "
            f"{', '.join(TENDONS)}"
        )


@dataclass(frozen=True)
class ExcursionTrajectory:
    """A monotonic, linearly interpolated tendon-excursion trajectory."""

    elapsed_times_s: tuple[float, ...]
    excursions_mm: tuple[tuple[float, ...], ...]
    position_units_per_mm: tuple[float, ...]
    time_scale: float = 1.0

    def __post_init__(self) -> None:
        if not self.elapsed_times_s or len(self.elapsed_times_s) != len(self.excursions_mm):
            raise ValueError("Trajectory must contain matching non-empty time and value rows")
        if any(
            current < previous
            for previous, current in zip(self.elapsed_times_s, self.elapsed_times_s[1:])
        ):
            raise ValueError("Trajectory elapsed times must be monotonic")
        _six_values(self.position_units_per_mm, "position_units_per_mm")
        if any(
            not math.isfinite(value) or value == 0.0
            for value in self.position_units_per_mm
        ):
            raise ValueError("position_units_per_mm must be finite and non-zero")
        if self.time_scale <= 0.0 or not math.isfinite(self.time_scale):
            raise ValueError("time_scale must be finite and greater than zero")
        for row_index, row in enumerate(self.excursions_mm):
            _six_values(row, f"excursion row {row_index}")
            if not all(math.isfinite(value) for value in row):
                raise ValueError(f"excursion row {row_index} contains a non-finite value")

    @classmethod
    def from_csv(
        cls,
        csv_path: Path,
        position_units_per_mm: Sequence[float],
        *,
        time_scale: float = 1.0,
    ) -> "ExcursionTrajectory":
        path = Path(csv_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Prediction CSV not found: {path}")

        required = ("elapsed_s", *EXCURSION_COLUMNS)
        rows: list[tuple[float, ...]] = []
        with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
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
                    raise ValueError(
                        f"Prediction CSV row {row_number} contains a non-finite value"
                    )
                rows.append(values)

        if not rows:
            raise ValueError("Prediction CSV contains no rows")
        first_time = rows[0][0]
        elapsed = tuple(float(row[0] - first_time) for row in rows)
        excursions = tuple(tuple(row[1:]) for row in rows)
        return cls(elapsed, excursions, tuple(position_units_per_mm), time_scale)

    @property
    def duration_s(self) -> float:
        return self.elapsed_times_s[-1] / self.time_scale

    @property
    def initial_excursion_mm(self) -> tuple[float, ...]:
        return self.excursions_mm[0]

    def excursions_at(self, elapsed_s: float) -> tuple[float, ...]:
        if not math.isfinite(elapsed_s):
            raise ValueError("elapsed_s must be finite")
        source_time = max(0.0, float(elapsed_s)) * self.time_scale
        if source_time <= self.elapsed_times_s[0]:
            return self.excursions_mm[0]
        if source_time >= self.elapsed_times_s[-1]:
            return self.excursions_mm[-1]

        right = bisect.bisect_right(self.elapsed_times_s, source_time)
        left = max(0, right - 1)
        right = min(len(self.elapsed_times_s) - 1, right)
        left_time = self.elapsed_times_s[left]
        right_time = self.elapsed_times_s[right]
        if right_time <= left_time:
            return self.excursions_mm[right]
        fraction = (source_time - left_time) / (right_time - left_time)
        return tuple(
            left_value + (right_value - left_value) * fraction
            for left_value, right_value in zip(
                self.excursions_mm[left], self.excursions_mm[right]
            )
        )

    def positions_at(
        self,
        elapsed_s: float,
        start_positions: Sequence[float | int],
    ) -> tuple[float, ...]:
        _six_values(start_positions, "start_positions")
        excursions = self.excursions_at(elapsed_s)
        initial = self.initial_excursion_mm
        return tuple(
            float(start) + (excursion - initial_value) * scale
            for start, excursion, initial_value, scale in zip(
                start_positions,
                excursions,
                initial,
                self.position_units_per_mm,
            )
        )
