"""Shared incremental-excursion contract for direct and residual control.

Existing P2P behavior controllers produce absolute excursion targets. This
module defines the learning-controller contract: policy actions are increments,
while the lower excursion PID still receives an absolute target. A
PID-reflected target is a software fact, not evidence of physical application.
"""

from __future__ import annotations

import math
from dataclasses import dataclass, replace
from enum import Enum
from typing import Any, Sequence

from controller.rl.action_safety_filter import (
    ActionComponents,
    ActionSafetyContext,
    ActionSafetyFilter,
    ActionSafetyResult,
)
from controller.servo_mapping import TENDONS


def _six_finite(values: Sequence[float], name: str) -> tuple[float, ...]:
    result = tuple(float(value) for value in values)
    if len(result) != len(TENDONS):
        raise ValueError(f"{name} must contain {len(TENDONS)} values")
    if not all(math.isfinite(value) for value in result):
        raise ValueError(f"{name} must contain only finite values")
    return result


class ControlMode(str, Enum):
    DIRECT = "direct"
    RESIDUAL = "residual"


@dataclass(frozen=True)
class IncrementalActionResult:
    mode: ControlMode
    previous_pid_target_mm: tuple[float, ...]
    requested_delta_mm: tuple[float, ...] | None
    requested_target_mm: tuple[float, ...]
    filtered_target_mm: tuple[float, ...]
    pid_reflected_delta_mm: tuple[float, ...]
    rejected: bool
    intervened: bool
    reasons: tuple[str, ...]
    safety_result: ActionSafetyResult | None = None

    def as_payload(self) -> dict[str, Any]:
        def tendon_map(values: Sequence[float]) -> dict[str, float]:
            return {tendon: float(value) for tendon, value in zip(TENDONS, values)}

        return {
            "control_mode": self.mode.value,
            "action_semantics": "incremental_excursion_mm",
            "previous_pid_target_mm": tendon_map(self.previous_pid_target_mm),
            "requested_delta_mm": (
                None
                if self.requested_delta_mm is None
                else tendon_map(self.requested_delta_mm)
            ),
            "requested_target_mm": tendon_map(self.requested_target_mm),
            "filtered_target_mm": tendon_map(self.filtered_target_mm),
            "pid_reflected_target_mm": tendon_map(self.filtered_target_mm),
            "pid_reflected_delta_mm": tendon_map(self.pid_reflected_delta_mm),
            "action_clipped": self.intervened and not self.rejected,
            "action_rejected": self.rejected,
            "safety_intervened": self.intervened,
            "safety_reasons": list(self.reasons),
            "physical_applied_status": "unknown",
            "physical_applied_monotonic_s": None,
        }


class IncrementalExcursionController:
    """Accumulate a bounded increment into an absolute lower-PID target."""

    def __init__(
        self,
        *,
        safety_filter: ActionSafetyFilter | None,
        initial_target_mm: Sequence[float] = (0.0,) * len(TENDONS),
        mode: ControlMode | str = ControlMode.DIRECT,
    ) -> None:
        self.safety_filter = safety_filter
        self.mode = ControlMode(mode)
        self._pid_target_mm = _six_finite(initial_target_mm, "initial_target_mm")

    @property
    def pid_target_mm(self) -> tuple[float, ...]:
        return self._pid_target_mm

    def reset(self, initial_target_mm: Sequence[float]) -> None:
        self._pid_target_mm = _six_finite(initial_target_mm, "initial_target_mm")

    def step(
        self,
        requested_delta_mm: Sequence[float] | None,
        *,
        safety_context: ActionSafetyContext,
        communication_ok: bool | None,
        base_target_mm: Sequence[float] | None = None,
    ) -> IncrementalActionResult:
        previous = self._pid_target_mm
        precheck_reasons: list[str] = []
        delta: tuple[float, ...] | None
        try:
            if requested_delta_mm is None:
                raise ValueError("action_missing")
            delta = _six_finite(requested_delta_mm, "requested_delta_mm")
        except (TypeError, ValueError):
            delta = None
            precheck_reasons.append("action_missing_or_invalid")

        if communication_ok is not True:
            precheck_reasons.append(
                "communication_status_missing"
                if communication_ok is None
                else "communication_unavailable"
            )

        if self.mode is ControlMode.RESIDUAL:
            try:
                reference = _six_finite(base_target_mm, "base_target_mm")  # type: ignore[arg-type]
            except (TypeError, ValueError):
                reference = previous
                precheck_reasons.append("base_target_missing_or_invalid")
        else:
            reference = previous

        requested_target = (
            previous
            if delta is None
            else tuple(base + change for base, change in zip(reference, delta))
        )
        if precheck_reasons:
            return self._rejected_result(previous, delta, requested_target, precheck_reasons)

        components = ActionComponents.nominal_only(requested_target)
        if self.safety_filter is None:
            safety_result = ActionSafetyResult(
                components=components,
                requested_action_mm=requested_target,
                applied_action_mm=requested_target,
                intervened=False,
                rejected=False,
                reasons=(),
            )
        else:
            safety_result = self.safety_filter.evaluate(
                components,
                replace(safety_context, previous_applied_action_mm=previous),
            )

        filtered = previous if safety_result.rejected else safety_result.applied_action_mm
        reflected_delta = tuple(current - old for current, old in zip(filtered, previous))
        result = IncrementalActionResult(
            mode=self.mode,
            previous_pid_target_mm=previous,
            requested_delta_mm=delta,
            requested_target_mm=requested_target,
            filtered_target_mm=filtered,
            pid_reflected_delta_mm=reflected_delta,
            rejected=safety_result.rejected,
            intervened=safety_result.intervened,
            reasons=safety_result.reasons,
            safety_result=safety_result,
        )
        if not result.rejected:
            self._pid_target_mm = filtered
        return result

    def _rejected_result(
        self,
        previous: tuple[float, ...],
        delta: tuple[float, ...] | None,
        requested_target: tuple[float, ...],
        reasons: Sequence[str],
    ) -> IncrementalActionResult:
        return IncrementalActionResult(
            mode=self.mode,
            previous_pid_target_mm=previous,
            requested_delta_mm=delta,
            requested_target_mm=requested_target,
            filtered_target_mm=previous,
            pid_reflected_delta_mm=(0.0,) * len(TENDONS),
            rejected=True,
            intervened=True,
            reasons=tuple(dict.fromkeys(str(reason) for reason in reasons)),
        )

