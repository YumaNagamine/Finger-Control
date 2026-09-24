"""Offline onset and settling analysis for one servo step response."""

from __future__ import annotations

import math
from dataclasses import dataclass
from statistics import median
from typing import Sequence


@dataclass(frozen=True)
class TimedScalar:
    """One scalar observation with a PC monotonic timestamp."""

    at: float
    value: float


@dataclass(frozen=True)
class TimedServoSample:
    at: float
    position: float
    speed: float | None


@dataclass(frozen=True)
class Detection:
    """First sample and confirmation sample of a consecutive detection."""

    first_at: float
    confirmed_at: float
    first_index: int
    confirmed_index: int


def robust_noise(values: Sequence[float]) -> float:
    """Return a MAD-based noise estimate, or zero for an empty sequence."""

    finite = [float(value) for value in values if math.isfinite(float(value))]
    if not finite:
        return 0.0
    center = median(finite)
    return 1.4826 * median(abs(value - center) for value in finite)


def threshold_from_noise(
    configured_min: float,
    noise: float,
    noise_scale: float,
) -> float:
    if not math.isfinite(configured_min) or configured_min < 0.0:
        raise ValueError("configured_min must be finite and non-negative")
    if not math.isfinite(noise) or noise < 0.0:
        raise ValueError("noise must be finite and non-negative")
    if not math.isfinite(noise_scale) or noise_scale < 0.0:
        raise ValueError("noise_scale must be finite and non-negative")
    return max(float(configured_min), float(noise_scale) * float(noise))


def detect_onset(
    samples: Sequence[TimedScalar],
    *,
    baseline: float,
    threshold: float,
    consecutive_count: int,
    direction: int | None = None,
) -> Detection | None:
    """Detect a directed change that exceeds a threshold consecutively.

    When ``direction`` is omitted, the first threshold crossing selects the
    direction.  The returned ``first_at`` is the event estimate; the later
    ``confirmed_at`` is when the consecutive-sample rule became knowable.
    """

    if not math.isfinite(baseline):
        raise ValueError("baseline must be finite")
    if not math.isfinite(threshold) or threshold < 0.0:
        raise ValueError("threshold must be finite and non-negative")
    if consecutive_count <= 0:
        raise ValueError("consecutive_count must be positive")
    if direction not in (None, -1, 1):
        raise ValueError("direction must be -1, 1, or None")

    run_direction: int | None = None
    run_start = -1
    for index, sample in enumerate(samples):
        delta = float(sample.value) - baseline
        if not math.isfinite(delta) or abs(delta) <= threshold:
            run_direction = None
            run_start = -1
            continue
        sample_direction = 1 if delta > 0.0 else -1
        if direction is not None and sample_direction != direction:
            run_direction = None
            run_start = -1
            continue
        if sample_direction != run_direction:
            run_direction = sample_direction
            run_start = index
        elif run_start < 0:
            run_start = index
        if index - run_start + 1 >= consecutive_count:
            first = samples[run_start]
            return Detection(
                first_at=float(first.at),
                confirmed_at=float(sample.at),
                first_index=run_start,
                confirmed_index=index,
            )
    return None


def detect_servo_settling(
    samples: Sequence[TimedServoSample],
    *,
    onset_at: float | None,
    target_position: float,
    position_band: float,
    speed_threshold: float,
    consecutive_count: int,
) -> Detection | None:
    if not math.isfinite(target_position):
        raise ValueError("target_position must be finite")
    if not math.isfinite(position_band) or position_band < 0.0:
        raise ValueError("position_band must be finite and non-negative")
    if not math.isfinite(speed_threshold) or speed_threshold < 0.0:
        raise ValueError("speed_threshold must be finite and non-negative")
    if consecutive_count <= 0:
        raise ValueError("consecutive_count must be positive")

    run_start = -1
    for index, sample in enumerate(samples):
        if onset_at is not None and sample.at < onset_at:
            continue
        settled = (
            math.isfinite(sample.position)
            and abs(sample.position - target_position) <= position_band
            and sample.speed is not None
            and math.isfinite(sample.speed)
            and abs(sample.speed) <= speed_threshold
        )
        if not settled:
            run_start = -1
            continue
        if run_start < 0:
            run_start = index
        if index - run_start + 1 >= consecutive_count:
            first = samples[run_start]
            return Detection(
                first_at=float(first.at),
                confirmed_at=float(sample.at),
                first_index=run_start,
                confirmed_index=index,
            )
    return None


def detect_joint_settling(
    samples: Sequence[TimedScalar],
    *,
    onset_at: float | None,
    terminal_window: int,
    position_band: float,
    slope_threshold: float,
    consecutive_count: int,
) -> Detection | None:
    """Detect a stable terminal joint plateau using the recorded tail."""

    if terminal_window <= 0 or consecutive_count <= 0:
        raise ValueError("window and consecutive_count must be positive")
    if not math.isfinite(position_band) or position_band < 0.0:
        raise ValueError("position_band must be finite and non-negative")
    if not math.isfinite(slope_threshold) or slope_threshold < 0.0:
        raise ValueError("slope_threshold must be finite and non-negative")

    eligible = [
        (index, sample)
        for index, sample in enumerate(samples)
        if (onset_at is None or sample.at >= onset_at)
        and math.isfinite(sample.value)
    ]
    if len(eligible) < terminal_window:
        return None
    terminal_values = [sample.value for _, sample in eligible[-terminal_window:]]
    terminal_center = median(terminal_values)

    for end in range(consecutive_count - 1, len(eligible)):
        window = eligible[end - consecutive_count + 1 : end + 1]
        values = [sample.value for _, sample in window]
        if any(abs(value - terminal_center) > position_band for value in values):
            continue
        if any(
            abs(current - previous) > slope_threshold
            for previous, current in zip(values, values[1:])
        ):
            continue
        first_index, first_sample = window[0]
        last_index, last_sample = window[-1]
        return Detection(
            first_at=float(first_sample.at),
            confirmed_at=float(last_sample.at),
            first_index=first_index,
            confirmed_index=last_index,
        )
    return None
