"""Safety-aware force PID controller.

The controller deliberately outputs a speed command. The current
ServoHardwareAPI exposes speed commands, not arbitrary force or position
commands, so force is controlled by changing tendon speed while the measured
force approaches its target.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Optional

from controller.servo.control_common.pid import PID


class ForceControlError(RuntimeError):
    """Base class for errors that must stop force control."""


class ForceSafetyError(ForceControlError):
    """Raised when a feedback or configured safety limit is violated."""


@dataclass(frozen=True)
class ForceFeedback:
    """One synchronized force-control feedback sample."""

    force_n: float
    servo_position: float
    servo_age_s: float
    encoder_age_s: float


@dataclass(frozen=True)
class ForceControlConfig:
    target_force_n: float
    max_speed: float
    hard_force_limit_n: float
    min_target_force_n: float = 0.5
    max_feedback_age_s: float = 0.2
    min_position: Optional[float] = None
    max_position: Optional[float] = None
    control_sign: float = 1.0

    def __post_init__(self) -> None:
        values = (
            self.target_force_n,
            self.max_speed,
            self.hard_force_limit_n,
            self.min_target_force_n,
            self.max_feedback_age_s,
            self.control_sign,
        )
        if not all(math.isfinite(value) for value in values):
            raise ValueError("Force-control settings must be finite")
        if self.max_speed <= 0.0:
            raise ValueError("max_speed must be greater than zero")
        if self.hard_force_limit_n <= 0.0:
            raise ValueError("hard_force_limit_n must be greater than zero")
        if self.min_target_force_n < 0.0:
            raise ValueError("min_target_force_n must not be negative")
        if self.target_force_n < self.min_target_force_n:
            raise ValueError("target_force_n must be at least min_target_force_n")
        if abs(self.target_force_n) >= self.hard_force_limit_n:
            raise ValueError("target_force_n must be below hard_force_limit_n")
        if self.max_feedback_age_s <= 0.0:
            raise ValueError("max_feedback_age_s must be greater than zero")
        if self.control_sign not in (-1.0, 1.0):
            raise ValueError("control_sign must be either -1 or 1")
        if (
            self.min_position is not None
            and self.max_position is not None
            and self.min_position >= self.max_position
        ):
            raise ValueError("min_position must be smaller than max_position")


@dataclass(frozen=True)
class ForceControlOutput:
    force_error_n: float
    raw_speed_command: float
    speed_command: int
    limited_by_position: bool


class ForcePIDController:
    """Convert calibrated force feedback into a bounded speed command."""

    def __init__(self, pid: PID, config: ForceControlConfig):
        self.pid = pid
        self.config = config

    def reset(self) -> None:
        self.pid.reset()

    def step(
        self,
        feedback: ForceFeedback,
        *,
        now: Optional[float] = None,
    ) -> ForceControlOutput:
        self._validate_feedback(feedback)

        raw_speed = self.config.control_sign * self.pid.update(
            self.config.target_force_n,
            feedback.force_n,
            now=now,
        )
        speed = max(
            -self.config.max_speed,
            min(self.config.max_speed, raw_speed),
        )

        limited_by_position = False
        if (
            self.config.min_position is not None
            and feedback.servo_position <= self.config.min_position
            and speed < 0.0
        ):
            speed = 0.0
            limited_by_position = True
        if (
            self.config.max_position is not None
            and feedback.servo_position >= self.config.max_position
            and speed > 0.0
        ):
            speed = 0.0
            limited_by_position = True

        return ForceControlOutput(
            force_error_n=self.config.target_force_n - feedback.force_n,
            raw_speed_command=raw_speed,
            speed_command=int(round(speed)),
            limited_by_position=limited_by_position,
        )

    def _validate_feedback(self, feedback: ForceFeedback) -> None:
        values = (
            feedback.force_n,
            feedback.servo_position,
            feedback.servo_age_s,
            feedback.encoder_age_s,
        )
        if not all(math.isfinite(value) for value in values):
            raise ForceSafetyError("Force feedback contains a non-finite value")
        if feedback.servo_age_s > self.config.max_feedback_age_s:
            raise ForceSafetyError("Servo feedback is stale")
        if feedback.encoder_age_s > self.config.max_feedback_age_s:
            raise ForceSafetyError("Encoder feedback is stale")
        if abs(feedback.force_n) >= self.config.hard_force_limit_n:
            raise ForceSafetyError("Force safety limit exceeded")
