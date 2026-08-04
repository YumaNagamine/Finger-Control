"""Runtime evaluation of saved moment-arm polynomial models."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Sequence

import numpy as np

from controller.servo_mapping import TENDONS


JOINTS = ("DIP", "PIP", "MCP")
MOTIONS = ("flexion", "extension")


class MomentArmRuntime:
    def __init__(
        self,
        coefficients: dict[tuple[str, str], dict[str, np.ndarray]],
    ) -> None:
        self._coefficients = coefficients

    @classmethod
    def from_directory(cls, directory: Path) -> "MomentArmRuntime":
        root = directory.expanduser().resolve()
        if not root.is_dir():
            raise FileNotFoundError(f"Moment-arm directory not found: {root}")

        coefficients: dict[tuple[str, str], dict[str, np.ndarray]] = {}
        for joint in JOINTS:
            for motion in MOTIONS:
                path = root / f"moment_arm_{joint}_{motion}_average.json"
                if not path.is_file():
                    raise FileNotFoundError(f"Moment-arm JSON not found: {path}")
                with path.open("r", encoding="utf-8") as handle:
                    payload = json.load(handle)
                raw_by_tendon = payload.get("moment_arm_coefficients")
                if not isinstance(raw_by_tendon, dict):
                    raise ValueError(f"Missing moment_arm_coefficients in {path}")

                tendon_coefficients: dict[str, np.ndarray] = {}
                for tendon in TENDONS:
                    raw_values = raw_by_tendon.get(tendon)
                    if not isinstance(raw_values, list) or not raw_values:
                        raise ValueError(f"Missing coefficients for {tendon} in {path}")
                    values = np.asarray(raw_values, dtype=np.float64)
                    if values.ndim != 1 or not np.isfinite(values).all():
                        raise ValueError(f"Invalid coefficients for {tendon} in {path}")
                    tendon_coefficients[tendon] = values
                coefficients[(joint, motion)] = tendon_coefficients
        return cls(coefficients)

    def matrix(
        self,
        joint_angles_rad: Sequence[float],
        motion_directions: Sequence[str],
    ) -> np.ndarray:
        angles = np.asarray(joint_angles_rad, dtype=np.float64)
        directions = tuple(str(value) for value in motion_directions)
        if angles.shape != (len(JOINTS),):
            raise ValueError("joint_angles_rad must contain DIP, PIP, and MCP")
        if not np.isfinite(angles).all():
            raise ValueError("joint_angles_rad contains a non-finite value")
        if len(directions) != len(JOINTS) or any(
            direction not in MOTIONS for direction in directions
        ):
            raise ValueError("motion_directions must contain flexion or extension for each joint")

        result = np.empty((len(TENDONS), len(JOINTS)), dtype=np.float64)
        for joint_index, (joint, motion, angle) in enumerate(
            zip(JOINTS, directions, angles)
        ):
            by_tendon = self._coefficients[(joint, motion)]
            for tendon_index, tendon in enumerate(TENDONS):
                result[tendon_index, joint_index] = float(
                    np.polyval(by_tendon[tendon], angle)
                )
        return result

    def correction_excursion(
        self,
        joint_angles_rad: Sequence[float],
        joint_correction_rad: Sequence[float],
        motion_directions: Sequence[str],
    ) -> np.ndarray:
        correction = np.asarray(joint_correction_rad, dtype=np.float64)
        if correction.shape != (len(JOINTS),):
            raise ValueError("joint_correction_rad must contain DIP, PIP, and MCP")
        if not np.isfinite(correction).all():
            raise ValueError("joint_correction_rad contains a non-finite value")
        return self.matrix(joint_angles_rad, motion_directions) @ correction
