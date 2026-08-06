"""Build bounded servo positions from nominal and feedback excursions."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from controller.servo_mapping import TENDONS


@dataclass(frozen=True)
class ServoCommand:
    nominal_excursions_mm: tuple[float, float, float, float, float, float]
    feedback_excursions_mm: tuple[float, float, float, float, float, float]
    total_excursions_mm: tuple[float, float, float, float, float, float]
    nominal_target_positions: tuple[int, int, int, int, int, int]
    target_positions: tuple[int, int, int, int, int, int]


class FeedbackCommandBuilder:
    def __init__(
        self,
        *,
        start_positions: Sequence[int],
        initial_excursions_mm: Sequence[float],
        position_units_per_mm: Sequence[float],
        position_limits: Sequence[tuple[int, int]],
        max_position_step: Sequence[int],
    ) -> None:
        self._start_positions = self._int_vector(start_positions, "start_positions")
        self._initial_excursions = self._float_vector(
            initial_excursions_mm,
            "initial_excursions_mm",
        )
        self._position_units_per_mm = self._float_vector(
            position_units_per_mm,
            "position_units_per_mm",
        )
        if np.any(self._position_units_per_mm == 0.0):
            raise ValueError("position_units_per_mm must not contain zero")

        if len(position_limits) != len(TENDONS):
            raise ValueError(f"position_limits must contain {len(TENDONS)} pairs")
        limits = np.asarray(position_limits, dtype=np.int64)
        if limits.shape != (len(TENDONS), 2) or np.any(limits[:, 0] > limits[:, 1]):
            raise ValueError("position_limits contains an invalid range")
        self._position_limits = limits

        self._max_position_step = self._int_vector(
            max_position_step,
            "max_position_step",
        )
        if np.any(self._max_position_step < 0):
            raise ValueError("max_position_step must be non-negative")
        self._last_nominal_positions = self._start_positions.copy()
        self._last_positions = self._start_positions.copy()

    def build(
        self,
        nominal_excursions_mm: Sequence[float],
        feedback_excursions_mm: Sequence[float],
    ) -> ServoCommand:
        nominal = self._float_vector(
            nominal_excursions_mm,
            "nominal_excursions_mm",
        )
        feedback = self._float_vector(
            feedback_excursions_mm,
            "feedback_excursions_mm",
        )
        total = nominal + feedback
        raw_nominal_positions = np.rint(
            self._start_positions
            + (nominal - self._initial_excursions) * self._position_units_per_mm
        ).astype(np.int64)
        raw_positions = np.rint(
            self._start_positions
            + (total - self._initial_excursions) * self._position_units_per_mm
        ).astype(np.int64)

        below = raw_positions < self._position_limits[:, 0]
        above = raw_positions > self._position_limits[:, 1]
        if np.any(below | above):
            tendon_index = int(np.flatnonzero(below | above)[0])
            tendon = TENDONS[tendon_index]
            raise ValueError(
                f"{tendon} target position {raw_positions[tendon_index]} is outside "
                f"{tuple(int(value) for value in self._position_limits[tendon_index])}"
            )

        nominal_lower_step = self._last_nominal_positions - self._max_position_step
        nominal_upper_step = self._last_nominal_positions + self._max_position_step
        nominal_limited_positions = np.clip(
            raw_nominal_positions,
            nominal_lower_step,
            nominal_upper_step,
        )
        self._last_nominal_positions = nominal_limited_positions

        lower_step = self._last_positions - self._max_position_step
        upper_step = self._last_positions + self._max_position_step
        limited_positions = np.clip(raw_positions, lower_step, upper_step)
        self._last_positions = limited_positions
        return ServoCommand(
            nominal_excursions_mm=tuple(float(value) for value in nominal),
            feedback_excursions_mm=tuple(float(value) for value in feedback),
            total_excursions_mm=tuple(float(value) for value in total),
            nominal_target_positions=tuple(
                int(value) for value in nominal_limited_positions
            ),
            target_positions=tuple(int(value) for value in limited_positions),
        )

    @staticmethod
    def _float_vector(values: Sequence[float], name: str) -> np.ndarray:
        vector = np.asarray(values, dtype=np.float64)
        if vector.shape != (len(TENDONS),):
            raise ValueError(f"{name} must contain {len(TENDONS)} values")
        if not np.isfinite(vector).all():
            raise ValueError(f"{name} contains a non-finite value")
        return vector

    @staticmethod
    def _int_vector(values: Sequence[int], name: str) -> np.ndarray:
        vector = np.asarray(values, dtype=np.int64)
        if vector.shape != (len(TENDONS),):
            raise ValueError(f"{name} must contain {len(TENDONS)} values")
        return vector
