"""Safety-aware magnetic-encoder position PID control for the servo."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Optional

from controller.servo.control_common.pid import PID


class MagneticEncoderPositionControlError(RuntimeError):
    """Base class for errors that must stop encoder-position control."""


class MagneticEncoderPositionSafetyError(MagneticEncoderPositionControlError):
    """Raised when feedback or a configured position limit is unsafe."""


@dataclass(frozen=True)
class MagneticEncoderPositionFeedback:
    """One synchronized sample used by the position controller."""

    position_deg: float
    servo_age_s: float
    encoder_age_s: float
    status: str = "OK"


@dataclass(frozen=True)
class MagneticEncoderPositionControlConfig:
    """Position target, PID output limits, and feedback safety limits."""

    target_position_deg: float
    max_speed: float
    hard_position_error_deg: float
    max_feedback_age_s: float = 0.2
    min_position_deg: Optional[float] = None
    max_position_deg: Optional[float] = None
    control_sign: float = 1.0

    def __post_init__(self) -> None:
        values = (
            self.target_position_deg,
            self.max_speed,
            self.hard_position_error_deg,
            self.max_feedback_age_s,
            self.control_sign,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise ValueError("Position-control settings must be finite")
        if self.max_speed <= 0.0:
            raise ValueError("max_speed must be greater than zero")
        if self.hard_position_error_deg <= 0.0:
            raise ValueError("hard_position_error_deg must be greater than zero")
        if self.max_feedback_age_s <= 0.0:
            raise ValueError("max_feedback_age_s must be greater than zero")
        if self.control_sign not in (-1.0, 1.0):
            raise ValueError("control_sign must be either -1 or 1")
        if (
            self.min_position_deg is not None
            and self.max_position_deg is not None
            and self.min_position_deg >= self.max_position_deg
        ):
            raise ValueError("min_position_deg must be smaller than max_position_deg")


@dataclass(frozen=True)
class MagneticEncoderPositionControlOutput:
    """One bounded servo-speed result from the position controller."""

    position_error_deg: float
    raw_speed_command: float
    speed_command: int
    limited_by_position: bool


def encoder_position_from_sample(
    sample: Mapping[str, Any],
    *,
    encoder_direction: float = 1.0,
) -> float:
    """Return the continuous encoder position used by the control loop.

    The AS5048A firmware exposes ``relative_multi_turn_deg`` relative to its
    configured encoder zero.  The single-turn ``relative_deg`` field is not a
    safe fallback for position control because it wraps at the revolution
    boundary.
    """

    if encoder_direction not in (-1.0, 1.0):
        raise ValueError("encoder_direction must be either -1 or 1")
    status = str(sample.get("status", "")).strip().upper()
    if status != "OK":
        raise MagneticEncoderPositionSafetyError(
            f"Magnetic encoder status is not safe for position control: {status or 'MISSING'}"
        )
    raw_position = sample.get("relative_multi_turn_deg")
    if raw_position is None:
        raise MagneticEncoderPositionSafetyError(
            "Magnetic encoder multi-turn position is unavailable"
        )
    try:
        position = float(raw_position)
    except (TypeError, ValueError) as exc:
        raise MagneticEncoderPositionSafetyError(
            "Magnetic encoder multi-turn position is invalid"
        ) from exc
    if not math.isfinite(position):
        raise MagneticEncoderPositionSafetyError(
            "Magnetic encoder multi-turn position is not finite"
        )
    return float(encoder_direction) * position


class MagneticEncoderPositionPIDController:
    """Convert encoder-position error into a bounded servo-speed command."""

    def __init__(
        self,
        pid: PID,
        config: MagneticEncoderPositionControlConfig,
    ) -> None:
        self.pid = pid
        self.config = config

    def reset(self) -> None:
        self.pid.reset()

    def step(
        self,
        feedback: MagneticEncoderPositionFeedback,
        *,
        now: Optional[float] = None,
    ) -> MagneticEncoderPositionControlOutput:
        self._validate_feedback(feedback)
        position_error = self.config.target_position_deg - feedback.position_deg
        if abs(position_error) >= self.config.hard_position_error_deg:
            raise MagneticEncoderPositionSafetyError(
                "Magnetic encoder position error reached the hard limit: "
                f"{position_error:.6f} deg"
            )

        raw_speed = self.config.control_sign * self.pid.update(
            self.config.target_position_deg,
            feedback.position_deg,
            now=now,
        )
        speed = max(-self.config.max_speed, min(self.config.max_speed, raw_speed))

        limited_by_position = False
        if (
            self.config.min_position_deg is not None
            and feedback.position_deg <= self.config.min_position_deg
            and speed < 0.0
        ):
            speed = 0.0
            limited_by_position = True
        if (
            self.config.max_position_deg is not None
            and feedback.position_deg >= self.config.max_position_deg
            and speed > 0.0
        ):
            speed = 0.0
            limited_by_position = True

        return MagneticEncoderPositionControlOutput(
            position_error_deg=position_error,
            raw_speed_command=raw_speed,
            speed_command=int(round(speed)),
            limited_by_position=limited_by_position,
        )

    def _validate_feedback(
        self,
        feedback: MagneticEncoderPositionFeedback,
    ) -> None:
        values = (
            feedback.position_deg,
            feedback.servo_age_s,
            feedback.encoder_age_s,
        )
        if not all(math.isfinite(float(value)) for value in values):
            raise MagneticEncoderPositionSafetyError(
                "Magnetic encoder feedback contains a non-finite value"
            )
        if str(feedback.status).strip().upper() != "OK":
            raise MagneticEncoderPositionSafetyError(
                f"Magnetic encoder status is not safe: {feedback.status}"
            )
        if feedback.servo_age_s > self.config.max_feedback_age_s:
            raise MagneticEncoderPositionSafetyError("Servo feedback is stale")
        if feedback.encoder_age_s > self.config.max_feedback_age_s:
            raise MagneticEncoderPositionSafetyError(
                "Magnetic encoder feedback is stale"
            )
