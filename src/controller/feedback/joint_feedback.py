"""Joint-space proportional feedback mapped to tendon excursion."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Sequence

import numpy as np

from controller.feedback.moment_arm_runtime import JOINTS, MomentArmRuntime
from controller.servo_mapping import TENDONS


@dataclass(frozen=True)
class FeedbackResult:
    error_rad: tuple[float, float, float]
    joint_correction_rad: tuple[float, float, float]
    excursion_correction_mm: tuple[float, float, float, float, float, float]


class JointFeedbackController:
    def __init__(
        self,
        moment_arm: MomentArmRuntime,
        *,
        kp: Sequence[float],
        max_joint_correction_rad: Sequence[float],
        max_excursion_correction_mm: Sequence[float],
    ) -> None:
        self._moment_arm = moment_arm
        self._kp = self._validated_vector(kp, len(JOINTS), "kp")
        self._max_joint = self._validated_vector(
            max_joint_correction_rad,
            len(JOINTS),
            "max_joint_correction_rad",
        )
        self._max_excursion = self._validated_vector(
            max_excursion_correction_mm,
            len(TENDONS),
            "max_excursion_correction_mm",
        )
        if np.any(self._max_joint < 0.0) or np.any(self._max_excursion < 0.0):
            raise ValueError("feedback correction limits must be non-negative")

    def compute(
        self,
        reference_angles_rad: Sequence[float],
        measured_angles_rad: Sequence[float],
        motion_directions: Sequence[str],
    ) -> FeedbackResult:
        reference = self._validated_vector(
            reference_angles_rad,
            len(JOINTS),
            "reference_angles_rad",
        )
        measured = self._validated_vector(
            measured_angles_rad,
            len(JOINTS),
            "measured_angles_rad",
        )
        error = reference - measured
        joint_correction = np.clip(
            self._kp * error,
            -self._max_joint,
            self._max_joint,
        )
        excursion_correction = self._moment_arm.correction_excursion(
            reference,
            joint_correction,
            motion_directions,
        )
        excursion_correction = np.clip(
            excursion_correction,
            -self._max_excursion,
            self._max_excursion,
        )
        return FeedbackResult(
            error_rad=tuple(float(value) for value in error),
            joint_correction_rad=tuple(float(value) for value in joint_correction),
            excursion_correction_mm=tuple(float(value) for value in excursion_correction),
        )

    @staticmethod
    def _validated_vector(
        values: Sequence[float],
        expected_length: int,
        name: str,
    ) -> np.ndarray:
        vector = np.asarray(values, dtype=np.float64)
        if vector.shape != (expected_length,):
            raise ValueError(f"{name} must contain {expected_length} values")
        if not np.isfinite(vector).all():
            raise ValueError(f"{name} contains a non-finite value")
        return vector
