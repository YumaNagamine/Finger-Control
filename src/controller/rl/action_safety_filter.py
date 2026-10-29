"""Composable tendon-action safety checks for real-hardware control."""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Mapping, Sequence

from controller.servo_mapping import TENDONS


JOINTS = ("DIP", "PIP", "MCP")


def _vector(values: Sequence[float], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != len(TENDONS):
        raise ValueError(f"{name} must contain {len(TENDONS)} values")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


def _tendon_mapping(
    values: Mapping[str, float],
    name: str,
    *,
    positive: bool = False,
) -> tuple[float, ...]:
    missing = [tendon for tendon in TENDONS if tendon not in values]
    if missing:
        raise ValueError(f"{name} is missing: {', '.join(missing)}")
    result = tuple(float(values[tendon]) for tendon in TENDONS)
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values")
    if positive and not all(value > 0.0 for value in result):
        raise ValueError(f"{name} values must be positive")
    return result


@dataclass(frozen=True)
class ActionComponents:
    """Action terms before safety projection, in tendon excursion millimetres."""

    nominal_action_mm: tuple[float, ...]
    feedback_action_mm: tuple[float, ...]
    exploration_action_mm: tuple[float, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "nominal_action_mm",
            _vector(self.nominal_action_mm, "nominal_action_mm"),
        )
        object.__setattr__(
            self,
            "feedback_action_mm",
            _vector(self.feedback_action_mm, "feedback_action_mm"),
        )
        object.__setattr__(
            self,
            "exploration_action_mm",
            _vector(self.exploration_action_mm, "exploration_action_mm"),
        )

    @classmethod
    def nominal_only(cls, action_mm: Sequence[float]) -> "ActionComponents":
        zero = (0.0,) * len(TENDONS)
        return cls(_vector(action_mm, "action_mm"), zero, zero)

    @property
    def requested_action_mm(self) -> tuple[float, ...]:
        return tuple(
            nominal + feedback + exploration
            for nominal, feedback, exploration in zip(
                self.nominal_action_mm,
                self.feedback_action_mm,
                self.exploration_action_mm,
            )
        )


@dataclass(frozen=True)
class ActionSafetyContext:
    previous_applied_action_mm: tuple[float, ...] | None = None
    force_n_by_tendon: Mapping[str, float | None] | None = None
    joint_angles_rad: Mapping[str, float | None] | None = None
    servo_age_s: float | None = None
    encoder_age_s_by_tendon: Mapping[str, float | None] | None = None
    joint_age_s: float | None = None
    deadline_lateness_s: float | None = None


@dataclass(frozen=True)
class ActionSafetyConfig:
    min_action_mm_by_tendon: Mapping[str, float]
    max_action_mm_by_tendon: Mapping[str, float]
    max_action_delta_mm_by_tendon: Mapping[str, float]
    hard_force_limits_n_by_tendon: Mapping[str, float] | None = None
    joint_angle_limits_rad: Mapping[str, tuple[float, float]] | None = None
    max_servo_age_s: float | None = None
    max_encoder_age_s: float | None = None
    max_joint_age_s: float | None = None
    max_deadline_lateness_s: float | None = None
    require_force: bool = True
    reject_on_action_projection: bool = False

    def __post_init__(self) -> None:
        lower = _tendon_mapping(
            self.min_action_mm_by_tendon,
            "min_action_mm_by_tendon",
        )
        upper = _tendon_mapping(
            self.max_action_mm_by_tendon,
            "max_action_mm_by_tendon",
        )
        if any(low > high for low, high in zip(lower, upper)):
            raise ValueError("Action minimum exceeds maximum")
        _tendon_mapping(
            self.max_action_delta_mm_by_tendon,
            "max_action_delta_mm_by_tendon",
            positive=True,
        )
        if self.hard_force_limits_n_by_tendon is not None:
            _tendon_mapping(
                self.hard_force_limits_n_by_tendon,
                "hard_force_limits_n_by_tendon",
                positive=True,
            )
        if self.joint_angle_limits_rad is not None:
            missing = [joint for joint in JOINTS if joint not in self.joint_angle_limits_rad]
            if missing:
                raise ValueError(
                    "joint_angle_limits_rad is missing: " + ", ".join(missing)
                )
            for joint in JOINTS:
                limits = self.joint_angle_limits_rad[joint]
                if len(limits) != 2:
                    raise ValueError(f"joint_angle_limits_rad.{joint} must be a pair")
                low, high = (float(value) for value in limits)
                if not math.isfinite(low) or not math.isfinite(high) or low > high:
                    raise ValueError(f"joint_angle_limits_rad.{joint} is invalid")
        for name in (
            "max_servo_age_s",
            "max_encoder_age_s",
            "max_joint_age_s",
            "max_deadline_lateness_s",
        ):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ValueError(f"{name} must be finite and non-negative")


@dataclass(frozen=True)
class ActionSafetyResult:
    components: ActionComponents
    requested_action_mm: tuple[float, ...]
    applied_action_mm: tuple[float, ...]
    intervened: bool
    rejected: bool
    reasons: tuple[str, ...]

    def as_payload(self) -> dict[str, Any]:
        def tendon_map(values: Sequence[float]) -> dict[str, float]:
            return {
                tendon: float(value)
                for tendon, value in zip(TENDONS, values)
            }

        return {
            "nominal_action_excursion_mm": tendon_map(
                self.components.nominal_action_mm
            ),
            "feedback_action_excursion_mm": tendon_map(
                self.components.feedback_action_mm
            ),
            "exploration_action_excursion_mm": tendon_map(
                self.components.exploration_action_mm
            ),
            "requested_action_excursion_mm": tendon_map(
                self.requested_action_mm
            ),
            "applied_action_excursion_mm": tendon_map(self.applied_action_mm),
            "safety_intervened": self.intervened,
            "safety_rejected": self.rejected,
            "safety_reasons": list(self.reasons),
        }


class ActionSafetyError(RuntimeError):
    def __init__(self, result: ActionSafetyResult) -> None:
        self.result = result
        super().__init__(
            "Action rejected by safety filter: " + ", ".join(result.reasons)
        )


class ActionSafetyContextError(RuntimeError):
    """Safety telemetry or timing was invalid outside an action decision."""

    def __init__(self, reasons: Sequence[str]) -> None:
        self.reasons = tuple(str(reason) for reason in reasons)
        super().__init__(
            "Safety context rejected: " + ", ".join(self.reasons)
        )


class ActionSafetyFilter:
    """Project action bounds and reject unsafe sensor or plant state."""

    def __init__(self, config: ActionSafetyConfig) -> None:
        self.config = config
        self._lower = _tendon_mapping(
            config.min_action_mm_by_tendon,
            "min_action_mm_by_tendon",
        )
        self._upper = _tendon_mapping(
            config.max_action_mm_by_tendon,
            "max_action_mm_by_tendon",
        )
        self._max_delta = _tendon_mapping(
            config.max_action_delta_mm_by_tendon,
            "max_action_delta_mm_by_tendon",
            positive=True,
        )

    def evaluate(
        self,
        components: ActionComponents,
        context: ActionSafetyContext,
    ) -> ActionSafetyResult:
        requested = components.requested_action_mm
        previous = (
            None
            if context.previous_applied_action_mm is None
            else _vector(
                context.previous_applied_action_mm,
                "previous_applied_action_mm",
            )
        )
        rejection_reasons = list(self.context_rejection_reasons(context))
        applied = list(requested)
        projection_reasons: list[str] = []

        for index, tendon in enumerate(TENDONS):
            bounded = min(max(applied[index], self._lower[index]), self._upper[index])
            if bounded != applied[index]:
                projection_reasons.append(f"action_bound:{tendon}")
                applied[index] = bounded
            if previous is not None:
                low = previous[index] - self._max_delta[index]
                high = previous[index] + self._max_delta[index]
                limited = min(max(applied[index], low), high)
                if limited != applied[index]:
                    projection_reasons.append(f"action_delta:{tendon}")
                    applied[index] = limited

        if projection_reasons and self.config.reject_on_action_projection:
            rejection_reasons.extend(projection_reasons)
        reasons = tuple(dict.fromkeys((*rejection_reasons, *projection_reasons)))
        rejected = bool(rejection_reasons)
        if rejected and previous is not None:
            applied = list(previous)
        return ActionSafetyResult(
            components=components,
            requested_action_mm=requested,
            applied_action_mm=tuple(float(value) for value in applied),
            intervened=bool(reasons),
            rejected=rejected,
            reasons=reasons,
        )

    def context_rejection_reasons(
        self,
        context: ActionSafetyContext,
    ) -> tuple[str, ...]:
        """Validate sensor and timing facts independently of a new action.

        The lower control loop uses this at every tick.  It intentionally does
        not project an action or apply the per-decision excursion-delta rule.
        """

        return tuple(self._rejection_reasons(context))

    def _rejection_reasons(self, context: ActionSafetyContext) -> list[str]:
        reasons: list[str] = []
        self._check_age(
            reasons,
            "servo_stale",
            context.servo_age_s,
            self.config.max_servo_age_s,
        )
        self._check_age(
            reasons,
            "joint_stale",
            context.joint_age_s,
            self.config.max_joint_age_s,
        )
        self._check_age(
            reasons,
            "control_deadline",
            context.deadline_lateness_s,
            self.config.max_deadline_lateness_s,
        )

        if self.config.max_encoder_age_s is not None:
            if context.encoder_age_s_by_tendon is None:
                reasons.append("encoder_age_missing")
            else:
                for tendon in TENDONS:
                    self._check_age(
                        reasons,
                        f"encoder_stale:{tendon}",
                        context.encoder_age_s_by_tendon.get(tendon),
                        self.config.max_encoder_age_s,
                    )

        force_limits = self.config.hard_force_limits_n_by_tendon
        if force_limits is not None:
            if context.force_n_by_tendon is None:
                if self.config.require_force:
                    reasons.append("force_missing")
            else:
                for tendon in TENDONS:
                    raw_force = context.force_n_by_tendon.get(tendon)
                    if raw_force is None:
                        if self.config.require_force:
                            reasons.append(f"force_missing:{tendon}")
                        continue
                    force = float(raw_force)
                    if not math.isfinite(force):
                        reasons.append(f"force_non_finite:{tendon}")
                    elif abs(force) >= float(force_limits[tendon]):
                        reasons.append(f"force_limit:{tendon}")

        joint_limits = self.config.joint_angle_limits_rad
        if joint_limits is not None:
            if context.joint_angles_rad is None:
                reasons.append("joint_angles_missing")
            else:
                for joint in JOINTS:
                    raw_angle = context.joint_angles_rad.get(joint)
                    if raw_angle is None:
                        reasons.append(f"joint_angle_missing:{joint}")
                        continue
                    angle = float(raw_angle)
                    low, high = joint_limits[joint]
                    if not math.isfinite(angle):
                        reasons.append(f"joint_angle_non_finite:{joint}")
                    elif angle < float(low) or angle > float(high):
                        reasons.append(f"joint_angle_limit:{joint}")
        return reasons

    @staticmethod
    def _check_age(
        reasons: list[str],
        reason: str,
        value: float | None,
        limit: float | None,
    ) -> None:
        if limit is None:
            return
        if value is None or not math.isfinite(float(value)):
            reasons.append(reason + "_missing")
        elif float(value) > limit:
            reasons.append(reason)
