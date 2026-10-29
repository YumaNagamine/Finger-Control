"""Continuous time trajectories loaded from tendon-excursion CSV files."""

from __future__ import annotations

import bisect
import csv
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Mapping, Sequence

import numpy as np

from controller.servo_mapping import TENDONS


EXCURSION_COLUMNS = tuple(f"{tendon}_predicted_excursion_mm" for tendon in TENDONS)
TARGET_ANGLE_DEG_COLUMNS = tuple(
    f"{joint}_smoothed_angle_deg" for joint in ("DIP", "PIP", "MCP")
)


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
    position_units_per_mm: tuple[float, ...] | None = None
    time_scale: float = 1.0
    target_angles_deg: tuple[tuple[float, ...], ...] | None = None
    source_elapsed_times_s: tuple[float, ...] | None = None

    def __post_init__(self) -> None:
        if not self.elapsed_times_s or len(self.elapsed_times_s) != len(self.excursions_mm):
            raise ValueError("Trajectory must contain matching non-empty time and value rows")
        if any(
            current < previous
            for previous, current in zip(self.elapsed_times_s, self.elapsed_times_s[1:])
        ):
            raise ValueError("Trajectory elapsed times must be monotonic")
        if self.position_units_per_mm is not None:
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
        if self.target_angles_deg is not None:
            if len(self.target_angles_deg) != len(self.elapsed_times_s):
                raise ValueError(
                    "Target angles must contain one row per trajectory time"
                )
            for row_index, row in enumerate(self.target_angles_deg):
                if len(row) != len(TARGET_ANGLE_DEG_COLUMNS):
                    raise ValueError(
                        f"target angle row {row_index} requires "
                        f"{len(TARGET_ANGLE_DEG_COLUMNS)} values"
                    )
                if not all(math.isfinite(value) for value in row):
                    raise ValueError(
                        f"target angle row {row_index} contains a non-finite value"
                    )
        if self.source_elapsed_times_s is not None:
            if len(self.source_elapsed_times_s) != len(self.elapsed_times_s):
                raise ValueError("Source timestamps must match the trajectory rows")
            if any(
                not math.isfinite(value)
                for value in self.source_elapsed_times_s
            ) or any(
                current < previous
                for previous, current in zip(
                    self.source_elapsed_times_s,
                    self.source_elapsed_times_s[1:],
                )
            ):
                raise ValueError("Source timestamps must be finite and monotonic")

    @classmethod
    def from_csv(
        cls,
        csv_path: Path,
        position_units_per_mm: Sequence[float] | None = None,
        *,
        time_scale: float = 1.0,
    ) -> "ExcursionTrajectory":
        path = Path(csv_path).expanduser().resolve()
        if not path.is_file():
            raise FileNotFoundError(f"Prediction CSV not found: {path}")

        required = ("elapsed_s", *EXCURSION_COLUMNS)
        rows: list[tuple[float, ...]] = []
        target_angle_rows: list[tuple[float, ...]] | None = None
        with path.open("r", encoding="utf-8-sig", newline="") as csv_file:
            reader = csv.DictReader(csv_file)
            fieldnames = reader.fieldnames or ()
            missing = [column for column in required if column not in fieldnames]
            if missing:
                raise ValueError(f"Prediction CSV is missing columns: {missing}")
            if all(column in fieldnames for column in TARGET_ANGLE_DEG_COLUMNS):
                target_angle_rows = []
            for row_number, row in enumerate(reader, start=2):
                try:
                    values = tuple(float(row[column]) for column in required)
                    target_angles = (
                        None
                        if target_angle_rows is None
                        else tuple(
                            float(row[column])
                            for column in TARGET_ANGLE_DEG_COLUMNS
                        )
                    )
                except (TypeError, ValueError) as exc:
                    raise ValueError(
                        f"Prediction CSV row {row_number} contains a non-numeric value"
                    ) from exc
                if not all(math.isfinite(value) for value in values) or (
                    target_angles is not None
                    and not all(math.isfinite(value) for value in target_angles)
                ):
                    raise ValueError(
                        f"Prediction CSV row {row_number} contains a non-finite value"
                    )
                rows.append(values)
                if target_angle_rows is not None and target_angles is not None:
                    target_angle_rows.append(target_angles)

        if not rows:
            raise ValueError("Prediction CSV contains no rows")
        first_time = rows[0][0]
        source_elapsed = tuple(float(row[0]) for row in rows)
        elapsed = tuple(value - first_time for value in source_elapsed)
        excursions = tuple(tuple(row[1:]) for row in rows)
        scales = (
            None
            if position_units_per_mm is None
            else tuple(float(value) for value in position_units_per_mm)
        )
        return cls(
            elapsed_times_s=elapsed,
            excursions_mm=excursions,
            position_units_per_mm=scales,
            time_scale=time_scale,
            target_angles_deg=(
                None
                if target_angle_rows is None
                else tuple(target_angle_rows)
            ),
            source_elapsed_times_s=source_elapsed,
        )

    @classmethod
    def from_joint_point_to_point(
        cls,
        q0_rad: Sequence[float],
        q_goal_rad: Sequence[float],
        duration_s: float,
        moment_arm: object,
        *,
        sample_hz: float = 50.0,
    ) -> "ExcursionTrajectory":
        """Build a minimum-jerk joint reference and integrate its FF excursion."""
        q0 = np.asarray(q0_rad, dtype=np.float64)
        q_goal = np.asarray(q_goal_rad, dtype=np.float64)
        if q0.shape != (3,) or q_goal.shape != (3,):
            raise ValueError("q0_rad and q_goal_rad must contain DIP, PIP, and MCP")
        if not np.isfinite(q0).all() or not np.isfinite(q_goal).all():
            raise ValueError("Point-to-point joint angles must be finite")
        if not math.isfinite(duration_s) or duration_s <= 0.0:
            raise ValueError("duration_s must be finite and positive")
        if not math.isfinite(sample_hz) or sample_hz <= 0.0:
            raise ValueError("sample_hz must be finite and positive")

        ranges = getattr(moment_arm, "evaluation_ranges_rad", None)
        if not isinstance(ranges, Mapping) or any(
            joint not in ranges for joint in ("DIP", "PIP", "MCP")
        ):
            raise ValueError("Moment-arm evaluation ranges are required for P2P")

        interval_count = max(1, math.ceil(float(duration_s) * float(sample_hz)))
        elapsed = tuple(
            float(duration_s) * index / interval_count
            for index in range(interval_count + 1)
        )
        references: list[np.ndarray] = []
        for time_s in elapsed:
            phase = time_s / float(duration_s)
            blend = 10.0 * phase**3 - 15.0 * phase**4 + 6.0 * phase**5
            references.append(q0 + blend * (q_goal - q0))

        for joint in ("DIP", "PIP", "MCP"):
            low, high = (float(value) for value in ranges[joint])
            if not math.isfinite(low) or not math.isfinite(high) or low >= high:
                raise ValueError(f"Invalid Moment-arm evaluation range for {joint}")

        excursions: list[tuple[float, ...]] = [(0.0,) * len(TENDONS)]
        cumulative = np.zeros(len(TENDONS), dtype=np.float64)
        for previous, current in zip(references, references[1:]):
            integrator = getattr(moment_arm, "integrate_excursion", None)
            if callable(integrator):
                increment = np.asarray(
                    integrator(previous, current),
                    dtype=np.float64,
                )
            else:
                midpoint = 0.5 * (previous + current)
                matrix = np.asarray(moment_arm.matrix(midpoint), dtype=np.float64)
                if matrix.shape != (len(TENDONS), 3) or not np.isfinite(matrix).all():
                    raise ValueError("Moment-arm model returned an invalid matrix")
                increment = matrix @ (current - previous)
            if increment.shape != (len(TENDONS),) or not np.isfinite(increment).all():
                raise ValueError("Moment-arm model returned an invalid excursion")
            cumulative = cumulative + increment
            excursions.append(tuple(float(value) for value in cumulative))

        return cls(
            elapsed_times_s=elapsed,
            excursions_mm=tuple(excursions),
            target_angles_deg=tuple(
                tuple(float(value) for value in np.rad2deg(reference))
                for reference in references
            ),
            source_elapsed_times_s=elapsed,
        )

    def to_csv(self, csv_path: Path) -> Path:
        """Write the reference in the existing CSV Player / Dataset schema."""
        if self.target_angles_deg is None:
            raise ValueError("A joint reference is required to save this trajectory")
        path = Path(csv_path).expanduser().resolve()
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("w", encoding="utf-8", newline="") as handle:
            writer = csv.writer(handle)
            writer.writerow(("elapsed_s", *EXCURSION_COLUMNS, *TARGET_ANGLE_DEG_COLUMNS))
            for elapsed_s, excursions, angles in zip(
                self.elapsed_times_s,
                self.excursions_mm,
                self.target_angles_deg,
            ):
                writer.writerow((elapsed_s, *excursions, *angles))
        return path

    @property
    def recorded_elapsed_times_s(self) -> tuple[float, ...]:
        """Return source CSV timestamps, or normalized times for direct instances."""
        if self.source_elapsed_times_s is not None:
            return self.source_elapsed_times_s
        return self.elapsed_times_s

    @property
    def source_time_origin_s(self) -> float:
        """Timestamp offset removed from elapsed_times_s during CSV loading."""
        if self.source_elapsed_times_s is not None:
            return self.source_elapsed_times_s[0]
        return 0.0

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
        if self.position_units_per_mm is None:
            raise ValueError(
                "Servo positions require position_units_per_mm; use excursions_at "
                "for encoder-position playback"
            )
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


def validate_point_to_point_trajectory(
    trajectory: ExcursionTrajectory,
    *,
    model_ranges_rad: Mapping[str, Sequence[float]] | None = None,
    approved_joint_limits_rad: Mapping[str, Sequence[float]],
    min_action_mm_by_tendon: Mapping[str, float],
    max_action_mm_by_tendon: Mapping[str, float],
    max_action_delta_mm_by_tendon: Mapping[str, float],
    decision_hz: float,
) -> dict[str, float]:
    """Fail closed on physical joint limits and FF/Safety Filter limits.

    ``model_ranges_rad`` remains accepted for caller compatibility, but fitting
    ranges no longer restrict reference angles.
    """
    joints = ("DIP", "PIP", "MCP")
    if trajectory.target_angles_deg is None:
        raise ValueError("P2P requires a saved joint reference")
    if not math.isfinite(decision_hz) or decision_hz <= 0.0:
        raise ValueError("decision_hz must be finite and positive")
    decision_span = trajectory.duration_s * float(decision_hz)
    if not math.isclose(decision_span, round(decision_span), rel_tol=0.0, abs_tol=1e-8):
        raise ValueError(
            "P2P duration must end on a Decision boundary so q_goal is applied"
        )

    joint_ranges: dict[str, tuple[float, float]] = {}
    for joint in joints:
        pair = approved_joint_limits_rad.get(joint)
        if pair is None or len(pair) != 2:
            raise ValueError(f"approved angle range for {joint} is required")
        low, high = (float(value) for value in pair)
        if not math.isfinite(low) or not math.isfinite(high) or low > high:
            raise ValueError(f"approved angle range for {joint} is invalid")
        joint_ranges[joint] = (low, high)

    for row_index, row in enumerate(trajectory.target_angles_deg):
        for joint, angle_deg in zip(joints, row):
            angle = math.radians(float(angle_deg))
            low, high = joint_ranges[joint]
            if angle < low or angle > high:
                raise ValueError(
                    f"P2P reference row {row_index} exceeds approved {joint} limits"
                )

    origin = trajectory.initial_excursion_mm
    action_rows = tuple(
        tuple(value - initial for value, initial in zip(row, origin))
        for row in trajectory.excursions_mm
    )
    for index, tendon in enumerate(TENDONS):
        low = float(min_action_mm_by_tendon[tendon])
        high = float(max_action_mm_by_tendon[tendon])
        if not math.isfinite(low) or not math.isfinite(high) or low > high:
            raise ValueError(f"Safety excursion range for {tendon} is invalid")
        for row_index, action in enumerate(action_rows):
            if action[index] < low or action[index] > high:
                raise ValueError(
                    f"P2P FF excursion at row {row_index} exceeds the Safety "
                    f"Filter range for {tendon}"
                )

    decision_count = int(round(decision_span))
    decision_actions = tuple(
        tuple(
            value - initial
            for value, initial in zip(
                trajectory.excursions_at(min(index / decision_hz, trajectory.duration_s)),
                origin,
            )
        )
        for index in range(decision_count + 1)
    )
    maximum_delta = {tendon: 0.0 for tendon in TENDONS}
    for previous, current in zip(decision_actions, decision_actions[1:]):
        for index, tendon in enumerate(TENDONS):
            delta = abs(current[index] - previous[index])
            allowed = float(max_action_delta_mm_by_tendon[tendon])
            if not math.isfinite(allowed) or allowed <= 0.0:
                raise ValueError(f"Safety action delta for {tendon} is invalid")
            maximum_delta[tendon] = max(maximum_delta[tendon], delta)
            if delta > allowed + 1e-12:
                raise ValueError(
                    f"P2P FF decision delta for {tendon} is {delta:.6g} mm; "
                    f"Safety Filter allows {allowed:.6g} mm"
                )
    return maximum_delta
