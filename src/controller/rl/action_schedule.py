"""High-level action schedules layered above the existing excursion PID."""

from __future__ import annotations

import math
from dataclasses import dataclass

from controller.csv_player.trajectory import ExcursionTrajectory
from controller.servo_mapping import TENDONS


INTERPOLATED_PLAYBACK_MODE = "interpolated"
RL_FF_ZOH_PLAYBACK_MODE = "rl_ff_zoh"
PLAYBACK_MODES = frozenset(
    {INTERPOLATED_PLAYBACK_MODE, RL_FF_ZOH_PLAYBACK_MODE}
)


@dataclass(frozen=True)
class ActionDecision:
    decision_id: int
    scheduled_elapsed_s: float
    action_excursion_mm: tuple[float, ...]
    skipped_decisions: int = 0


class RLFFZOHActionSchedule:
    """Sample a continuous FF trajectory at a configurable rate and hold it."""

    def __init__(
        self,
        trajectory: ExcursionTrajectory,
        *,
        decision_hz: float = 5.0,
    ) -> None:
        if not math.isfinite(decision_hz) or decision_hz <= 0.0:
            raise ValueError("decision_hz must be finite and positive")
        self.trajectory = trajectory
        self.decision_hz = float(decision_hz)
        self._active_decision_id: int | None = None

    def reset(self) -> None:
        self._active_decision_id = None

    def advance(self, elapsed_s: float) -> ActionDecision | None:
        """Return the newest newly-active decision, or ``None`` if unchanged.

        If the lower-level loop stalls across more than one decision boundary,
        only the newest action can actually be applied.  ``skipped_decisions``
        records the unapplied intermediate decisions instead of pretending
        that they reached the PID.
        """

        if not math.isfinite(elapsed_s):
            raise ValueError("elapsed_s must be finite")
        bounded_elapsed = min(
            max(0.0, float(elapsed_s)),
            self.trajectory.duration_s,
        )
        decision_id = int(math.floor(bounded_elapsed * self.decision_hz + 1e-12))
        if decision_id == self._active_decision_id:
            return None
        previous = self._active_decision_id
        scheduled_elapsed_s = min(
            decision_id / self.decision_hz,
            self.trajectory.duration_s,
        )
        sampled = self.trajectory.excursions_at(scheduled_elapsed_s)
        origin = self.trajectory.initial_excursion_mm
        action = tuple(
            float(value - initial)
            for value, initial in zip(sampled, origin)
        )
        if len(action) != len(TENDONS):
            raise ValueError("Action must contain one excursion per tendon")
        skipped = 0 if previous is None else max(0, decision_id - previous - 1)
        self._active_decision_id = decision_id
        return ActionDecision(
            decision_id=decision_id,
            scheduled_elapsed_s=scheduled_elapsed_s,
            action_excursion_mm=action,
            skipped_decisions=skipped,
        )

