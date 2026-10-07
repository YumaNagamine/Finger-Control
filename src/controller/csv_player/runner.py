"""Runtime for feedback speed-controlled excursion playback."""

from __future__ import annotations

import math
import time
from dataclasses import dataclass
from typing import Any

from controller.csv_player.force_estimator import CalibratedForceEstimator
from controller.csv_player.recorder import ControlTraceWriter
from controller.csv_player.speed_controller import SpeedController
from controller.csv_player.trajectory import ExcursionTrajectory
from controller.servo.hardware_session import ServoHardwareSession


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
    ) -> None:
        self.trajectory = trajectory
        self.servo_ids_by_tendon = tuple(servo_ids_by_tendon)
        self.hardware = hardware
        self.speed_controller = speed_controller
        self.config = config
        self.recorder = recorder
        self.force_estimator = force_estimator
        if len(self.servo_ids_by_tendon) != len(self.trajectory.position_units_per_mm):
            raise ValueError("servo IDs and trajectory channels must have equal lengths")

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
            speeds = self.speed_controller.update(target_by_id, actual, dt_s)
            self._submit_speed_to_servos(speeds)
            force_readings = (
                self.force_estimator.estimate(feedback, self._servo_index_by_id())
                if self.force_estimator is not None
                else {}
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
            now = time.monotonic()
            self.hardware.check_health()
            if now >= deadline:
                raise TimeoutError("Servos did not return to their initial positions")
            feedback = self.hardware.latest_feedback(self.config.telemetry_stale_s)
            if feedback is None:
                raise TimeoutError("Servo telemetry became unavailable during return")
            actual = self.hardware.position_by_servo_id(feedback)
            dt_s = max(1e-6, now - last_tick)
            speeds = self.speed_controller.update(target_by_id, actual, dt_s)
            self._submit_speed_to_servos(speeds)
            reached = all(
                abs(target_by_id[servo_id] - actual[servo_id]) <= self.config.return_tolerance
                for servo_id in self.servo_ids_by_tendon
            )
            stable_frames = stable_frames + 1 if reached else 0
            force_readings = (
                self.force_estimator.estimate(feedback, self._servo_index_by_id())
                if self.force_estimator is not None
                else {}
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
