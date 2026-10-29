"""Runtime for feedback speed-controlled excursion playback."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any, Callable, Mapping

from controller.csv_player.force_estimator import CalibratedForceEstimator
from controller.csv_player.recorder import ControlTraceWriter
from controller.csv_player.speed_controller import SpeedController
from controller.csv_player.trajectory import ExcursionTrajectory
from controller.rl.action_schedule import (
    INTERPOLATED_PLAYBACK_MODE,
    PLAYBACK_MODES,
)
from controller.servo.hardware_session import ServoHardwareSession
from controller.servo_mapping import TENDONS


class PlaybackStopped(RuntimeError):
    """Raised when an operator requests a controlled stop."""


class HardForceLimitError(RuntimeError):
    """Raised after a calibrated force value reaches a configured hard limit."""

    def __init__(
        self,
        *,
        tendon: str,
        servo_id: int,
        force_n: float,
        limit_n: float,
        elapsed_s: float,
    ) -> None:
        self.tendon = str(tendon)
        self.servo_id = int(servo_id)
        self.force_n = float(force_n)
        self.limit_n = float(limit_n)
        self.elapsed_s = float(elapsed_s)
        super().__init__(
            f"{self.tendon} force hard limit reached: "
            f"{self.force_n:.3f} N >= {self.limit_n:.3f} N"
        )


class ForceTelemetryUnavailable(RuntimeError):
    """Raised when force safety cannot be evaluated for a configured tendon."""

    def __init__(self, *, tendon: str, servo_id: int, status: str) -> None:
        self.tendon = str(tendon)
        self.servo_id = int(servo_id)
        self.status = str(status)
        super().__init__(
            f"Force telemetry unavailable for {self.tendon} "
            f"(servo {self.servo_id}, status={self.status})"
        )


@dataclass(frozen=True)
class PlaybackConfig:
    control_hz: float = 50.0
    telemetry_stale_s: float = 0.2
    telemetry_wait_s: float = 3.0
    display_interval_s: float = 0.25
    return_to_start: bool = True
    return_tolerance: float = 10.0
    return_timeout_s: float = 5.0
    return_stable_frames: int = 3
    hard_force_limits_n_by_tendon: Mapping[str, float] | None = None
    playback_mode: str = INTERPOLATED_PLAYBACK_MODE
    decision_hz: float = 5.0
    control_period_tolerance_fraction: float = 0.25

    def __post_init__(self) -> None:
        if not math.isfinite(self.control_hz) or self.control_hz <= 0.0:
            raise ValueError("control_hz must be finite and positive")
        if not math.isfinite(self.telemetry_stale_s) or self.telemetry_stale_s <= 0.0:
            raise ValueError("telemetry_stale_s must be finite and positive")
        if not math.isfinite(self.telemetry_wait_s) or self.telemetry_wait_s <= 0.0:
            raise ValueError("telemetry_wait_s must be finite and positive")
        if not math.isfinite(self.display_interval_s) or self.display_interval_s <= 0.0:
            raise ValueError("display_interval_s must be finite and positive")
        if self.return_tolerance < 0.0 or self.return_timeout_s <= 0.0:
            raise ValueError("return settings are invalid")
        if self.return_stable_frames <= 0:
            raise ValueError("return_stable_frames must be positive")
        if self.playback_mode not in PLAYBACK_MODES:
            raise ValueError(
                f"playback_mode must be one of {sorted(PLAYBACK_MODES)}"
            )
        if not math.isfinite(self.decision_hz) or self.decision_hz <= 0.0:
            raise ValueError("decision_hz must be finite and positive")
        if self.decision_hz > self.control_hz:
            raise ValueError("decision_hz must not exceed control_hz")
        if (
            not math.isfinite(self.control_period_tolerance_fraction)
            or self.control_period_tolerance_fraction < 0.0
        ):
            raise ValueError(
                "control_period_tolerance_fraction must be finite and non-negative"
            )
        if self.hard_force_limits_n_by_tendon is not None:
            missing = [
                tendon
                for tendon in TENDONS
                if tendon not in self.hard_force_limits_n_by_tendon
            ]
            if missing:
                raise ValueError(
                    "hard_force_limits_n_by_tendon is missing: "
                    + ", ".join(missing)
                )
            for tendon in TENDONS:
                limit = float(self.hard_force_limits_n_by_tendon[tendon])
                if not math.isfinite(limit) or limit <= 0.0:
                    raise ValueError(
                        f"hard force limit for {tendon} must be finite and positive"
                    )


class SpeedTrajectoryRunner:
    """Run one trajectory while recording servo and calibrated force feedback."""

    def __init__(
        self,
        *,
        trajectory: ExcursionTrajectory,
        servo_ids_by_tendon: tuple[int, ...],
        hardware: ServoHardwareSession,
        speed_controller: SpeedController,
        config: PlaybackConfig,
        recorder: ControlTraceWriter | None = None,
        force_estimator: CalibratedForceEstimator | None = None,
        on_sample: Callable[[dict[str, Any]], None] | None = None,
        stop_event: Any | None = None,
    ) -> None:
        self.trajectory = trajectory
        self.servo_ids_by_tendon = tuple(servo_ids_by_tendon)
        self.hardware = hardware
        self.speed_controller = speed_controller
        self.config = config
        self.recorder = recorder
        self.force_estimator = force_estimator
        self.on_sample = on_sample
        self.stop_event = stop_event
        if len(self.servo_ids_by_tendon) != len(self.trajectory.position_units_per_mm):
            raise ValueError("servo IDs and trajectory channels must have equal lengths")
        if self.config.hard_force_limits_n_by_tendon is not None and self.force_estimator is None:
            raise ValueError("force_estimator is required when hard force limits are configured")

    def run(self, start_positions_by_id: dict[int, float]) -> None:
        start_positions = tuple(start_positions_by_id[servo_id] for servo_id in self.servo_ids_by_tendon)
        self.speed_controller.reset()
        started_at = time.monotonic()
        self._run_playback(started_at, start_positions)
        if self.config.return_to_start:
            self._run_return(start_positions, started_at)
        self.hardware.stop_servos()

    def _run_playback(self, started_at: float, start_positions: tuple[float, ...]) -> None:
        next_tick = time.monotonic()
        last_tick = next_tick
        next_display = next_tick
        sent_final = False
        while True:
            self._check_stop_requested()
            now = time.monotonic()
            self.hardware.check_health()
            elapsed_s = now - started_at
            if elapsed_s >= self.trajectory.duration_s:
                elapsed_s = self.trajectory.duration_s
                sent_final = True
            feedback = self.hardware.latest_feedback(self.config.telemetry_stale_s)
            if feedback is None:
                raise TimeoutError("Servo telemetry became unavailable during playback")
            actual = self.hardware.position_by_servo_id(feedback)
            targets = self.trajectory.positions_at(elapsed_s, start_positions)
            target_by_id = dict(zip(self.servo_ids_by_tendon, targets))
            dt_s = max(1e-6, now - last_tick)
            force_readings = (
                self.force_estimator.estimate(feedback, self._servo_index_by_id())
                if self.force_estimator is not None
                else {}
            )
            self._check_hard_force_limits(force_readings, elapsed_s)
            speeds = self.speed_controller.update(target_by_id, actual, dt_s)
            self._check_stop_requested()
            self._submit_speed_to_servos(speeds)
            self._notify_sample(
                phase="playback",
                elapsed_s=elapsed_s,
                dt_s=dt_s,
                target_positions=target_by_id,
                actual_positions=actual,
                speeds=speeds,
                force_readings=force_readings,
            )
            if self.recorder is not None:
                self.recorder.write(
                    phase="playback",
                    elapsed_s=elapsed_s,
                    dt_s=dt_s,
                    servo_telemetry_age_s=feedback.servo_age_s,
                    target_excursions_mm=self.trajectory.excursions_at(elapsed_s),
                    target_positions=target_by_id,
                    actual_positions=actual,
                    speeds=speeds,
                    force_readings=force_readings,
                )
            if now >= next_display or sent_final:
                self._print_status("PLAYBACK", elapsed_s, target_by_id, actual, speeds, force_readings)
                next_display = now + self.config.display_interval_s
            if sent_final:
                return
            last_tick = now
            next_tick += 1.0 / self.config.control_hz
            time.sleep(max(0.0, next_tick - time.monotonic()))

    def _run_return(self, start_positions: tuple[float, ...], experiment_started_at: float) -> None:
        self.speed_controller.reset()
        target_by_id = dict(zip(self.servo_ids_by_tendon, start_positions))
        deadline = time.monotonic() + self.config.return_timeout_s
        next_tick = time.monotonic()
        last_tick = next_tick
        next_display = next_tick
        stable_frames = 0
        while True:
            self._check_stop_requested()
            now = time.monotonic()
            self.hardware.check_health()
            if now >= deadline:
                raise TimeoutError("Servos did not return to their initial positions")
            feedback = self.hardware.latest_feedback(self.config.telemetry_stale_s)
            if feedback is None:
                raise TimeoutError("Servo telemetry became unavailable during return")
            actual = self.hardware.position_by_servo_id(feedback)
            dt_s = max(1e-6, now - last_tick)
            force_readings = (
                self.force_estimator.estimate(feedback, self._servo_index_by_id())
                if self.force_estimator is not None
                else {}
            )
            self._check_hard_force_limits(
                force_readings,
                now - experiment_started_at,
            )
            speeds = self.speed_controller.update(target_by_id, actual, dt_s)
            self._check_stop_requested()
            self._submit_speed_to_servos(speeds)
            reached = all(
                abs(target_by_id[servo_id] - actual[servo_id]) <= self.config.return_tolerance
                for servo_id in self.servo_ids_by_tendon
            )
            stable_frames = stable_frames + 1 if reached else 0
            self._notify_sample(
                phase="return",
                elapsed_s=now - experiment_started_at,
                dt_s=dt_s,
                target_positions=target_by_id,
                actual_positions=actual,
                speeds=speeds,
                force_readings=force_readings,
            )
            if self.recorder is not None:
                self.recorder.write(
                    phase="return",
                    elapsed_s=time.monotonic() - experiment_started_at,
                    dt_s=dt_s,
                    servo_telemetry_age_s=feedback.servo_age_s,
                    target_excursions_mm=self.trajectory.initial_excursion_mm,
                    target_positions=target_by_id,
                    actual_positions=actual,
                    speeds=speeds,
                    force_readings=force_readings,
                )
            if now >= next_display or stable_frames >= self.config.return_stable_frames:
                self._print_status("RETURN", now - experiment_started_at, target_by_id, actual, speeds, force_readings)
                next_display = now + self.config.display_interval_s
            if stable_frames >= self.config.return_stable_frames:
                return
            last_tick = now
            next_tick += 1.0 / self.config.control_hz
            time.sleep(max(0.0, next_tick - time.monotonic()))

    def _check_stop_requested(self) -> None:
        if self.stop_event is not None and self.stop_event.is_set():
            raise PlaybackStopped("Playback was stopped by the operator")

    def _check_hard_force_limits(
        self,
        force_readings: dict[int, dict[str, Any]],
        elapsed_s: float,
    ) -> None:
        limits = self.config.hard_force_limits_n_by_tendon
        if limits is None:
            return
        for tendon, servo_id in zip(TENDONS, self.servo_ids_by_tendon):
            reading = force_readings.get(servo_id, {})
            force_value = reading.get("force_n")
            if force_value is None:
                status = str(reading.get("status", "MISSING"))
                try:
                    self.hardware.stop_servos()
                finally:
                    raise ForceTelemetryUnavailable(
                        tendon=tendon,
                        servo_id=servo_id,
                        status=status,
                    )
            force_n = float(force_value)
            if not math.isfinite(force_n):
                continue
            limit_n = float(limits[tendon])
            if abs(force_n) < limit_n:
                continue
            try:
                self.hardware.stop_servos()
            finally:
                raise HardForceLimitError(
                    tendon=tendon,
                    servo_id=servo_id,
                    force_n=force_n,
                    limit_n=limit_n,
                    elapsed_s=elapsed_s,
                )

    def _notify_sample(
        self,
        *,
        phase: str,
        elapsed_s: float,
        dt_s: float,
        target_positions: dict[int, float],
        actual_positions: dict[int, float],
        speeds: dict[int, int],
        force_readings: dict[int, dict[str, Any]],
    ) -> None:
        if self.on_sample is None:
            return
        self.on_sample(
            {
                "phase": phase,
                "elapsed_s": float(elapsed_s),
                "dt_s": float(dt_s),
                "target_positions": dict(target_positions),
                "actual_positions": dict(actual_positions),
                "speeds": dict(speeds),
                "force_readings": {
                    int(servo_id): dict(reading)
                    for servo_id, reading in force_readings.items()
                },
                "received_at": time.monotonic(),
            }
        )

    def _servo_index_by_id(self) -> dict[int, int]:
        return {
            servo_id: self.hardware.system.index_for_servo_id(servo_id)
            for servo_id in self.hardware.system.servo_ids
        }

    def _submit_speed_to_servos(self, speeds: dict[int, int]) -> None:
        """Publish through the dispatcher, with a small test-double fallback."""

        submit = getattr(self.hardware, "submit_speed_to_servos", None)
        if callable(submit):
            submit(speeds)
            return
        # Keep lightweight hardware fakes and older project adapters usable;
        # real ServoHardwareSession instances always take the dispatcher path.
        self.hardware.sender.send_speed_to_servos(speeds)

    @staticmethod
    def _print_status(
        phase: str,
        elapsed_s: float,
        targets: dict[int, float],
        actual: dict[int, float],
        speeds: dict[int, int],
        force_readings: dict[int, dict[str, Any]],
    ) -> None:
        parts = []
        for servo_id in targets:
            force = force_readings.get(servo_id, {}).get("force_n")
            force_text = "N/A" if force is None else f"{float(force):.2f}N"
            parts.append(
                f"ID{servo_id}={targets[servo_id]:.0f}/{actual[servo_id]:.0f}"
                f"({targets[servo_id] - actual[servo_id]:+.0f})"
                f" v={speeds[servo_id]} F={force_text}"
            )
        print(f"{phase} t={elapsed_s:.3f}s " + " | ".join(parts), flush=True)
