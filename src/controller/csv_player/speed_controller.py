"""Feedback speed control for continuous CSV trajectory playback."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Mapping


@dataclass(frozen=True)
class SpeedControlConfig:
    kp: float = 0.5
    max_speed: int = 500
    max_acceleration: float = 3000.0
    position_deadband: float = 3.0

    def __post_init__(self) -> None:
        if not math.isfinite(self.kp) or self.kp < 0.0:
            raise ValueError("kp must be finite and non-negative")
        if self.max_speed <= 0:
            raise ValueError("max_speed must be positive")
        if not math.isfinite(self.max_acceleration) or self.max_acceleration <= 0.0:
            raise ValueError("max_acceleration must be finite and positive")
        if not math.isfinite(self.position_deadband) or self.position_deadband < 0.0:
            raise ValueError("position_deadband must be finite and non-negative")


class SpeedController:
    """Convert position errors into bounded physical-ID speed commands."""

    def __init__(self, config: SpeedControlConfig | None = None) -> None:
        self.config = config or SpeedControlConfig()
        self._previous_speed: dict[int, float] = {}

    def reset(self) -> None:
        self._previous_speed.clear()

    def update(
        self,
        target_positions: Mapping[int, float],
        actual_positions: Mapping[int, float],
        dt_s: float,
    ) -> dict[int, int]:
        if not math.isfinite(dt_s) or dt_s <= 0.0:
            raise ValueError("dt_s must be finite and positive")

        max_delta = self.config.max_acceleration * dt_s
        result: dict[int, int] = {}
        for servo_id, target in target_positions.items():
            if servo_id not in actual_positions:
                raise ValueError(f"Missing actual position for servo {servo_id}")
            error = float(target) - float(actual_positions[servo_id])
            requested = 0.0 if abs(error) <= self.config.position_deadband else self.config.kp * error
            requested = max(-self.config.max_speed, min(self.config.max_speed, requested))
            previous = self._previous_speed.get(int(servo_id), 0.0)
            bounded = max(previous - max_delta, min(previous + max_delta, requested))
            bounded = max(-self.config.max_speed, min(self.config.max_speed, bounded))
            self._previous_speed[int(servo_id)] = bounded
            result[int(servo_id)] = int(round(bounded))
        return result
