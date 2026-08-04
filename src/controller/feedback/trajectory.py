"""Time interpolation for predicted-excursion feedback trajectories."""

from __future__ import annotations

import csv
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from controller.servo_mapping import TENDONS
from controller.feedback.moment_arm_runtime import JOINTS


JOINT_COLUMNS = tuple(f"{joint}_flexion_smoothed_angle_rad" for joint in JOINTS)
EXCURSION_COLUMNS = tuple(f"{tendon}_predicted_excursion_mm" for tendon in TENDONS)


@dataclass(frozen=True)
class TrajectorySample:
    elapsed_s: float
    joint_angles_rad: tuple[float, float, float]
    nominal_excursions_mm: tuple[float, float, float, float, float, float]
    motion_directions: tuple[str, str, str]


class FeedbackTrajectory:
    def __init__(
        self,
        elapsed_s: np.ndarray,
        joint_angles_rad: np.ndarray,
        excursions_mm: np.ndarray,
        *,
        direction_deadband_rad: float = math.radians(0.2),
    ) -> None:
        if elapsed_s.ndim != 1 or elapsed_s.size < 2:
            raise ValueError("Trajectory requires at least two elapsed-time samples")
        if joint_angles_rad.shape != (elapsed_s.size, len(JOINTS)):
            raise ValueError("joint_angles_rad must have shape (samples, 3)")
        if excursions_mm.shape != (elapsed_s.size, len(TENDONS)):
            raise ValueError("excursions_mm must have shape (samples, 6)")
        if not np.isfinite(elapsed_s).all():
            raise ValueError("elapsed_s contains a non-finite value")
        if not np.isfinite(joint_angles_rad).all():
            raise ValueError("joint angle trajectory contains a non-finite value")
        if not np.isfinite(excursions_mm).all():
            raise ValueError("excursion trajectory contains a non-finite value")
        if np.any(np.diff(elapsed_s) <= 0.0):
            raise ValueError("elapsed_s must be strictly increasing")
        if not math.isfinite(direction_deadband_rad) or direction_deadband_rad < 0.0:
            raise ValueError("direction_deadband_rad must be finite and non-negative")

        self._elapsed_s = np.asarray(elapsed_s, dtype=np.float64)
        self._joint_angles_rad = np.asarray(joint_angles_rad, dtype=np.float64)
        self._excursions_mm = np.asarray(excursions_mm, dtype=np.float64)
        self._directions = self._build_directions(direction_deadband_rad)

    @classmethod
    def from_csv(
        cls,
        csv_path: Path,
        *,
        direction_deadband_rad: float = math.radians(0.2),
    ) -> "FeedbackTrajectory":
        path = csv_path.expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Prediction CSV not found: {path}")

        required = ("elapsed_s", *JOINT_COLUMNS, *EXCURSION_COLUMNS)
        elapsed_values: list[float] = []
        joint_rows: list[list[float]] = []
        excursion_rows: list[list[float]] = []

        with path.open("r", newline="", encoding="utf-8-sig") as handle:
            reader = csv.DictReader(handle)
            missing = [column for column in required if column not in (reader.fieldnames or ())]
            if missing:
                raise ValueError(f"Prediction CSV is missing required columns: {missing}")

            for row_index, row in enumerate(reader):
                try:
                    elapsed_values.append(float(row["elapsed_s"]))
                    joint_rows.append([float(row[column]) for column in JOINT_COLUMNS])
                    excursion_rows.append([float(row[column]) for column in EXCURSION_COLUMNS])
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Prediction CSV row {row_index} contains a non-numeric value"
                    ) from exc

        if len(elapsed_values) < 2:
            raise ValueError("Prediction CSV must contain at least two data rows")

        elapsed = np.asarray(elapsed_values, dtype=np.float64)
        elapsed = elapsed - elapsed[0]
        return cls(
            elapsed,
            np.asarray(joint_rows, dtype=np.float64),
            np.asarray(excursion_rows, dtype=np.float64),
            direction_deadband_rad=direction_deadband_rad,
        )

    @property
    def duration_s(self) -> float:
        return float(self._elapsed_s[-1])

    def sample(self, elapsed_s: float) -> TrajectorySample:
        if not math.isfinite(elapsed_s):
            raise ValueError("elapsed_s must be finite")
        sample_time = min(max(float(elapsed_s), 0.0), self.duration_s)
        joint_angles = tuple(
            float(np.interp(sample_time, self._elapsed_s, self._joint_angles_rad[:, index]))
            for index in range(len(JOINTS))
        )
        excursions = tuple(
            float(np.interp(sample_time, self._elapsed_s, self._excursions_mm[:, index]))
            for index in range(len(TENDONS))
        )
        row_index = int(np.searchsorted(self._elapsed_s, sample_time, side="right") - 1)
        row_index = min(max(row_index, 0), len(self._directions) - 1)
        return TrajectorySample(
            elapsed_s=sample_time,
            joint_angles_rad=joint_angles,
            nominal_excursions_mm=excursions,
            motion_directions=self._directions[row_index],
        )

    def _build_directions(
        self,
        deadband_rad: float,
    ) -> tuple[tuple[str, str, str], ...]:
        current = ["flexion"] * len(JOINTS)
        result: list[tuple[str, str, str]] = [tuple(current)]
        for row_index in range(1, self._joint_angles_rad.shape[0]):
            deltas = self._joint_angles_rad[row_index] - self._joint_angles_rad[row_index - 1]
            for joint_index, delta in enumerate(deltas):
                if delta > deadband_rad:
                    current[joint_index] = "flexion"
                elif delta < -deadband_rad:
                    current[joint_index] = "extension"
            result.append(tuple(current))
        return tuple(result)
