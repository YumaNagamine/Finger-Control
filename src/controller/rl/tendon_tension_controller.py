"""Vector interface between RL tendon targets and the force-control session."""

from __future__ import annotations

import math
from typing import Any, Iterable, Mapping, Protocol

from controller.servo_mapping import TENDONS


# Public vector order used by Force Control: servo IDs 1 through 6.
FORCE_CONTROL_TENDON_ORDER = ("ED", "DI", "FDS", "FDP", "LU", "PI")


class ForceControlSessionProtocol(Protocol):
    def set_target_vector(
        self,
        target_force_n_by_tendon: Mapping[str, float],
        *,
        mode: str = "Step",
        duration_s: float = 1.0,
    ) -> tuple[float, ...]: ...

    def force_feedback_snapshot(self) -> dict: ...


def _force_vector(values: Iterable[float], name: str) -> tuple[float, ...]:
    if isinstance(values, (str, bytes)):
        raise ValueError(f"{name} must contain six force values")
    try:
        vector = tuple(values)
    except TypeError as exc:
        raise ValueError(f"{name} must contain six force values") from exc
    if len(vector) != 6:
        raise ValueError(f"{name} must contain exactly six values")

    result = []
    for index, value in enumerate(vector):
        if isinstance(value, bool):
            raise ValueError(f"{name}[{index}] must be a finite number")
        try:
            number = float(value)
        except (TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"{name}[{index}] must be a finite number") from exc
        if not math.isfinite(number):
            raise ValueError(f"{name}[{index}] must be a finite number")
        result.append(number)
    return tuple(result)


def rl_targets_to_force_control_order(
    targets_n: Iterable[float],
) -> tuple[float, ...]:
    """Convert RL's canonical tendon order into Force Control's servo order."""

    rl_vector = _force_vector(targets_n, "targets_n")
    rl_by_tendon = dict(zip(TENDONS, rl_vector))
    force_control_names = {
        "LU": "LUM",
    }
    return tuple(
        rl_by_tendon[force_control_names.get(tendon, tendon)]
        for tendon in FORCE_CONTROL_TENDON_ORDER
    )


class TendonTensionVectorController:
    """Coordinate six absolute tension targets without duplicating the PID.

    ``set_targets`` accepts Newton values in Force Control order
    (ED, DI, FDS, FDP, LU, PI). ``set_rl_targets`` accepts Newton values in
    the policy order declared by ``controller.servo_mapping.TENDONS`` and
    translates LUM to Force Control's LU channel.
    """

    def __init__(self, session: ForceControlSessionProtocol):
        self._session = session

    def set_targets(
        self,
        targets_n: Iterable[float],
        *,
        mode: str = "Step",
        duration_s: float = 1.0,
    ) -> tuple[float, ...]:
        """Set all six absolute targets in Force Control tendon order."""

        vector = _force_vector(targets_n, "targets_n")
        by_tendon = dict(zip(FORCE_CONTROL_TENDON_ORDER, vector))
        self._session.set_target_vector(
            by_tendon,
            mode=mode,
            duration_s=duration_s,
        )
        return vector

    def set_rl_targets(
        self,
        targets_n: Iterable[float],
        *,
        mode: str = "Step",
        duration_s: float = 1.0,
    ) -> tuple[float, ...]:
        """Set absolute Newton targets supplied in RL's canonical tendon order."""

        vector = rl_targets_to_force_control_order(targets_n)
        return self.set_targets(
            vector,
            mode=mode,
            duration_s=duration_s,
        )

    def feedback_snapshot(self) -> dict[str, Any]:
        """Return measured force, targets, freshness, and status by tendon."""

        return self._session.force_feedback_snapshot()


__all__ = [
    "FORCE_CONTROL_TENDON_ORDER",
    "TendonTensionVectorController",
    "rl_targets_to_force_control_order",
]
