"""Small deterministic PID implementation shared by servo controllers."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Optional


@dataclass(frozen=True)
class PIDConfig:
    """PID gains and output/integral limits."""

    kp: float
    ki: float
    kd: float
    output_min: float
    output_max: float
    integral_min: float = -float("inf")
    integral_max: float = float("inf")
    max_dt_s: float = 0.25

    def __post_init__(self) -> None:
        values = (
            self.kp,
            self.ki,
            self.kd,
            self.output_min,
            self.output_max,
            self.integral_min,
            self.integral_max,
            self.max_dt_s,
        )
        if not all(math.isfinite(value) for value in values[:5]):
            raise ValueError("PID gains and output limits must be finite")
        if self.output_min >= self.output_max:
            raise ValueError("output_min must be smaller than output_max")
        if self.integral_min >= self.integral_max:
            raise ValueError("integral_min must be smaller than integral_max")
        if not math.isfinite(self.max_dt_s) or self.max_dt_s <= 0.0:
            raise ValueError("max_dt_s must be greater than zero")


class PID:
    """PID controller with derivative-on-measurement and anti-windup.

    ``now`` is injectable so the controller can be tested without sleeping.
    The output units are intentionally generic; servo controllers use them as
    ServoHardwareAPI speed-command units.
    """

    def __init__(self, config: PIDConfig):
        self.config = config
        self.reset()

    def reset(self) -> None:
        self.integral = 0.0
        self._previous_measurement: Optional[float] = None
        self._previous_time: Optional[float] = None

    def update(
        self,
        target: float,
        measurement: float,
        *,
        now: Optional[float] = None,
    ) -> float:
        target = float(target)
        measurement = float(measurement)
        if not math.isfinite(target) or not math.isfinite(measurement):
            raise ValueError("PID target and measurement must be finite")

        current_time = time.monotonic() if now is None else float(now)
        if not math.isfinite(current_time):
            raise ValueError("PID time must be finite")

        dt = 0.0
        if self._previous_time is not None:
            candidate_dt = current_time - self._previous_time
            if 0.0 < candidate_dt <= self.config.max_dt_s:
                dt = candidate_dt

        error = target - measurement
        derivative = 0.0
        if self._previous_measurement is not None and dt > 0.0:
            # Differentiate the measurement to avoid a derivative kick when
            # the target changes.
            derivative = -(
                measurement - self._previous_measurement
            ) / dt

        candidate_integral = self.integral
        if dt > 0.0:
            candidate_integral = _clamp(
                self.integral + error * dt,
                self.config.integral_min,
                self.config.integral_max,
            )

        raw_output = (
            self.config.kp * error
            + self.config.ki * candidate_integral
            + self.config.kd * derivative
        )
        output = _clamp(
            raw_output,
            self.config.output_min,
            self.config.output_max,
        )

        # Do not continue integrating while the output is saturated in the
        # same direction as the error. This is the simple anti-windup rule
        # appropriate for the bounded speed command used here.
        saturated_high = raw_output > self.config.output_max and error > 0.0
        saturated_low = raw_output < self.config.output_min and error < 0.0
        if not saturated_high and not saturated_low:
            self.integral = candidate_integral

        self._previous_measurement = measurement
        self._previous_time = current_time
        return output


def _clamp(value: float, lower: float, upper: float) -> float:
    return max(lower, min(upper, value))
