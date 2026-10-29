"""Minimal goal-conditioned environment and explicitly non-physical mock."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Mapping, Sequence

import numpy as np

from controller.rl.action_safety_filter import ActionSafetyContext
from controller.rl.control_interface import IncrementalExcursionController
from controller.servo_mapping import TENDONS


JOINTS = ("DIP", "PIP", "MCP")
OBSERVATION_DIM = 24


@dataclass(frozen=True)
class RewardConfig:
    posture_weight: float = 1.0
    action_weight: float = 0.0
    success_tolerance_rad: float = 0.05
    success_dwell_steps: int = 3
    horizon_steps: int = 100


class MockFingerDynamics:
    """Deterministic software-test dynamics; not a robot physics model."""

    def __init__(self, response_fraction: float = 0.25) -> None:
        if not 0.0 < response_fraction <= 1.0:
            raise ValueError("response_fraction must be in (0, 1]")
        self.response_fraction = float(response_fraction)
        self.joint_rad = np.zeros(3, dtype=np.float64)
        self.excursion_mm = np.zeros(6, dtype=np.float64)
        self.tension_n = np.ones(6, dtype=np.float64)
        self._software_test_map = np.asarray(
            [
                [0.020, 0.010, 0.000, -0.005, 0.000, 0.000],
                [0.010, 0.020, 0.000, 0.000, -0.005, 0.000],
                [0.005, 0.005, -0.020, 0.000, 0.000, -0.005],
            ],
            dtype=np.float64,
        )

    def reset(
        self,
        *,
        joint_rad: Sequence[float],
        excursion_mm: Sequence[float],
        tension_n: Sequence[float],
    ) -> None:
        self.joint_rad = np.asarray(joint_rad, dtype=np.float64).reshape(3)
        self.excursion_mm = np.asarray(excursion_mm, dtype=np.float64).reshape(6)
        self.tension_n = np.asarray(tension_n, dtype=np.float64).reshape(6)

    def step(self, target_excursion_mm: Sequence[float]) -> None:
        target = np.asarray(target_excursion_mm, dtype=np.float64).reshape(6)
        change = self.response_fraction * (target - self.excursion_mm)
        self.excursion_mm = self.excursion_mm + change
        self.joint_rad = self.joint_rad + self._software_test_map @ change
        self.tension_n = np.maximum(0.0, 1.0 + 0.1 * np.abs(self.excursion_mm))


class GoalConditionedMockEnv:
    """Exercise reset/step, safety, reward, boundaries, and logging without I/O."""

    def __init__(
        self,
        *,
        controller: IncrementalExcursionController,
        dynamics: MockFingerDynamics | None = None,
        reward_config: RewardConfig | None = None,
        event_sink: Callable[[str, Mapping[str, Any]], None] | None = None,
    ) -> None:
        self.controller = controller
        self.dynamics = dynamics or MockFingerDynamics()
        self.reward_config = reward_config or RewardConfig()
        self.event_sink = event_sink
        self.goal_rad = np.zeros(3, dtype=np.float64)
        self.previous_applied_delta_mm = np.zeros(6, dtype=np.float64)
        self.steps = 0
        self.success_steps = 0

    def reset(
        self,
        *,
        initial_joint_rad: Sequence[float],
        goal_joint_rad: Sequence[float],
        initial_excursion_mm: Sequence[float] = (0.0,) * 6,
        initial_tension_n: Sequence[float] = (1.0,) * 6,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        self.goal_rad = np.asarray(goal_joint_rad, dtype=np.float64).reshape(3)
        self.dynamics.reset(
            joint_rad=initial_joint_rad,
            excursion_mm=initial_excursion_mm,
            tension_n=initial_tension_n,
        )
        self.controller.reset(initial_excursion_mm)
        self.previous_applied_delta_mm = np.zeros(6, dtype=np.float64)
        self.steps = 0
        self.success_steps = 0
        observation = self._observation()
        info = {"mock_physics_not_real": True, "episode_step": 0}
        self._emit("reset", info)
        return observation, info

    def step(
        self,
        action: Sequence[float] | None,
    ) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        safety_context = ActionSafetyContext(
            force_n_by_tendon=dict(zip(TENDONS, self.dynamics.tension_n)),
            joint_angles_rad=dict(zip(JOINTS, self.dynamics.joint_rad)),
            servo_age_s=0.0,
            encoder_age_s_by_tendon={tendon: 0.0 for tendon in TENDONS},
            joint_age_s=0.0,
            deadline_lateness_s=0.0,
        )
        result = self.controller.step(
            action,
            safety_context=safety_context,
            communication_ok=True,
        )
        if not result.rejected:
            self.dynamics.step(result.filtered_target_mm)
        self.previous_applied_delta_mm = np.asarray(
            result.pid_reflected_delta_mm,
            dtype=np.float64,
        )
        self.steps += 1
        error = self.goal_rad - self.dynamics.joint_rad
        reward = -self.reward_config.posture_weight * float(np.mean(np.square(error)))
        reward -= self.reward_config.action_weight * float(
            np.mean(np.square(self.previous_applied_delta_mm))
        )
        if np.all(np.abs(error) <= self.reward_config.success_tolerance_rad):
            self.success_steps += 1
        else:
            self.success_steps = 0
        terminated = self.success_steps >= self.reward_config.success_dwell_steps
        truncated = self.steps >= self.reward_config.horizon_steps and not terminated
        observation = self._observation()
        info = {
            "mock_physics_not_real": True,
            "episode_step": self.steps,
            "termination_reason": (
                "goal_dwell" if terminated else "horizon" if truncated else None
            ),
            **result.as_payload(),
        }
        self._emit("step", info)
        return observation, reward, terminated, truncated, info

    def _observation(self) -> np.ndarray:
        error = self.goal_rad - self.dynamics.joint_rad
        observation = np.concatenate(
            (
                self.dynamics.joint_rad,
                error,
                self.dynamics.excursion_mm,
                self.dynamics.tension_n,
                self.previous_applied_delta_mm,
            )
        ).astype(np.float32)
        if observation.shape != (OBSERVATION_DIM,):
            raise RuntimeError("Mock observation schema is not 24-dimensional")
        return observation

    def _emit(self, event_type: str, payload: Mapping[str, Any]) -> None:
        if self.event_sink is not None:
            self.event_sink(event_type, payload)

