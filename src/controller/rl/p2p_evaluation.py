"""Offline metrics for comparable FF and FF+FB point-to-point trials."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

import numpy as np


JOINTS = ("DIP", "PIP", "MCP")


@dataclass(frozen=True)
class P2PEvaluationConfig:
    success_tolerance_rad: float
    settling_tolerance_rad: float
    holding_duration_s: float
    minimum_settled_duration_s: float

    def __post_init__(self) -> None:
        for name in (
            "success_tolerance_rad",
            "settling_tolerance_rad",
            "holding_duration_s",
            "minimum_settled_duration_s",
        ):
            value = float(getattr(self, name))
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")


def evaluate_p2p_trial(
    *,
    experiment_id: str,
    controller_type: str,
    timestamps_s: Sequence[float],
    measured_joint_rad: Sequence[Sequence[float]],
    goal_joint_rad: Sequence[float],
    target_trajectory_rad: Sequence[Sequence[float]] | None,
    tendon_tension_n: Sequence[Sequence[float]] | None,
    feedback_correction_mm: Sequence[Sequence[float]] | None,
    safety_violation_count: int,
    config: P2PEvaluationConfig,
    quality_reasons: Sequence[str] = (),
) -> dict[str, Any]:
    reasons = list(str(item) for item in quality_reasons)
    times = np.asarray(timestamps_s, dtype=np.float64)
    measured = np.asarray(measured_joint_rad, dtype=np.float64)
    goal = np.asarray(goal_joint_rad, dtype=np.float64)
    if times.ndim != 1 or times.size == 0:
        reasons.append("missing_timestamps")
    if measured.shape != (times.size, len(JOINTS)):
        reasons.append("invalid_measured_joint_shape")
    if goal.shape != (len(JOINTS),):
        reasons.append("invalid_goal_shape")
    if times.size and (not np.isfinite(times).all() or np.any(np.diff(times) <= 0.0)):
        reasons.append("non_monotonic_or_non_finite_timestamps")
    if measured.size and not np.isfinite(measured).all():
        reasons.append("non_finite_measured_joint")
    if goal.size and not np.isfinite(goal).all():
        reasons.append("non_finite_goal")
    if reasons:
        return {
            "experiment_id": str(experiment_id),
            "controller_type": str(controller_type),
            "evaluation_quality": "invalid",
            "quality_reasons": list(dict.fromkeys(reasons)),
        }

    error = goal[None, :] - measured
    absolute_error = np.abs(error)
    final_error = error[-1]
    hold_start = max(times[0], times[-1] - config.holding_duration_s)
    hold_error = error[times >= hold_start]
    success = bool(
        np.all(absolute_error[-1] <= config.success_tolerance_rad)
        and np.all(np.abs(hold_error) <= config.success_tolerance_rad)
    )

    within = np.all(absolute_error <= config.settling_tolerance_rad, axis=1)
    settling_time_s: float | None = None
    for index, inside in enumerate(within):
        if not inside:
            continue
        dwell_end = times[index] + config.minimum_settled_duration_s
        dwell_mask = (times >= times[index]) & (times <= dwell_end + 1e-12)
        if dwell_end <= times[-1] and np.all(within[dwell_mask]):
            settling_time_s = float(times[index] - times[0])
            break

    tracking_rmse: dict[str, float] | None = None
    if target_trajectory_rad is not None:
        target = np.asarray(target_trajectory_rad, dtype=np.float64)
        if target.shape == measured.shape and np.isfinite(target).all():
            tracking_rmse = dict(
                zip(JOINTS, (float(v) for v in np.sqrt(np.mean(np.square(target - measured), axis=0))))
            )
        else:
            reasons.append("invalid_target_trajectory")

    max_tension_n: float | None = None
    if tendon_tension_n is not None:
        tension = np.asarray(tendon_tension_n, dtype=np.float64)
        if tension.ndim == 2 and tension.shape[0] == times.size and np.isfinite(tension).all():
            max_tension_n = float(np.max(np.abs(tension)))
        else:
            reasons.append("invalid_tension_samples")

    feedback_rms_mm: float | None = None
    feedback_max_abs_mm: float | None = None
    if feedback_correction_mm is not None:
        correction = np.asarray(feedback_correction_mm, dtype=np.float64)
        if correction.ndim == 2 and correction.shape[0] == times.size and np.isfinite(correction).all():
            feedback_rms_mm = float(np.sqrt(np.mean(np.square(correction))))
            feedback_max_abs_mm = float(np.max(np.abs(correction)))
        else:
            reasons.append("invalid_feedback_correction_samples")

    return {
        "experiment_id": str(experiment_id),
        "controller_type": str(controller_type),
        "evaluation_quality": "degraded" if reasons else "valid",
        "quality_reasons": list(dict.fromkeys(reasons)),
        "final_joint_error_rad": dict(zip(JOINTS, (float(v) for v in final_error))),
        "integrated_final_posture_error_rad": float(np.linalg.norm(final_error)),
        "goal_reached": success,
        "settling_time_s": settling_time_s,
        "holding_joint_rmse_rad": dict(
            zip(JOINTS, (float(v) for v in np.sqrt(np.mean(np.square(hold_error), axis=0))))
        ),
        "trajectory_tracking_rmse_rad": tracking_rmse,
        "max_tendon_tension_n": max_tension_n,
        "safety_violation_count": int(safety_violation_count),
        "feedback_correction_rms_mm": feedback_rms_mm,
        "feedback_correction_max_abs_mm": feedback_max_abs_mm,
    }


def summarize_p2p_trials(trials: Sequence[Mapping[str, Any]]) -> dict[str, Any]:
    grouped: dict[str, list[Mapping[str, Any]]] = {}
    rejected = 0
    for trial in trials:
        if trial.get("evaluation_quality") == "invalid":
            rejected += 1
            continue
        grouped.setdefault(str(trial.get("controller_type", "unknown")), []).append(trial)
    controllers: dict[str, Any] = {}
    for name, items in grouped.items():
        controllers[name] = {
            "trial_count": len(items),
            "success_rate": float(np.mean([bool(item.get("goal_reached")) for item in items])),
            "mean_integrated_final_posture_error_rad": float(
                np.mean([float(item["integrated_final_posture_error_rad"]) for item in items])
            ),
            "safety_violation_count": int(
                sum(int(item.get("safety_violation_count", 0)) for item in items)
            ),
        }
    return {"controllers": controllers, "invalid_trial_count": rejected}

