"""CSV excursion playback using magnetic-encoder position PID feedback."""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Mapping

from controller.csv_player.encoder_calibration import EncoderExcursionCalibration
from controller.csv_player.force_estimator import CalibratedForceEstimator
from controller.csv_player.recorder import ControlTraceWriter
from controller.csv_player.runner import (
    ForceTelemetryUnavailable,
    HardForceLimitError,
    PlaybackConfig,
    PlaybackStopped,
)
from controller.csv_player.trajectory import ExcursionTrajectory
from controller.servo.control_common.pid import PID, PIDConfig
from controller.servo.hardware_session import (
    HardwareFeedback,
    ServoHardwareSession,
)
from controller.servo.magneticEncoderPos_control.controller import (
    MagneticEncoderPositionControlConfig,
    MagneticEncoderPositionFeedback,
    MagneticEncoderPositionPIDController,
    MagneticEncoderPositionSafetyError,
    encoder_position_from_sample,
)
from controller.servo_mapping import TENDONS


class EncoderPositionTrajectoryRunner:
    """Convert CSV excursions to encoder targets and command servo speed by PID."""

    def __init__(
        self,
        *,
        trajectory: ExcursionTrajectory,
        servo_ids_by_tendon: tuple[int, ...],
        encoder_sensor_ids_by_servo_id: Mapping[int, int],
        calibration: EncoderExcursionCalibration,
        hardware: ServoHardwareSession,
        config: PlaybackConfig,
        recorder: ControlTraceWriter | None = None,
        force_estimator: CalibratedForceEstimator | None = None,
        on_sample: Callable[[dict[str, Any]], None] | None = None,
        stop_event: Any | None = None,
    ) -> None:
        if trajectory.position_units_per_mm is not None:
            raise ValueError(
                "Encoder playback must load the trajectory without servo position calibration"
            )
        if len(servo_ids_by_tendon) != len(TENDONS):
            raise ValueError("servo_ids_by_tendon must contain one ID per tendon")
        if len(calibration.degrees_per_mm) != len(TENDONS):
            raise ValueError("Encoder calibration must contain one scale per tendon")
        self.trajectory = trajectory
        self.servo_ids_by_tendon = tuple(int(value) for value in servo_ids_by_tendon)
        self.encoder_sensor_ids_by_servo_id = {
            int(servo_id): int(sensor_id)
            for servo_id, sensor_id in encoder_sensor_ids_by_servo_id.items()
        }
        self.calibration = calibration
        self.hardware = hardware
        self.config = config
        self.recorder = recorder
        self.force_estimator = force_estimator
        self.on_sample = on_sample
        self.stop_event = stop_event
        if self.config.hard_force_limits_n_by_tendon is not None and self.force_estimator is None:
            raise ValueError("force_estimator is required when hard force limits are configured")

        initial = self.trajectory.initial_excursion_mm
        max_target_delta = tuple(
            max(abs(row[index] - initial[index]) for row in self.trajectory.excursions_mm)
            for index in range(len(TENDONS))
        )
        self.controllers: dict[int, MagneticEncoderPositionPIDController] = {}
        for tendon_index, (_tendon, servo_id) in enumerate(
            zip(TENDONS, self.servo_ids_by_tendon)
        ):
            scale = self.calibration.degrees_per_mm[tendon_index]
            hard_error = max(
                self.calibration.minimum_hard_error_deg,
                max_target_delta[tendon_index] * scale
                + self.calibration.hard_error_margin_deg,
            )
            pid = PID(
                PIDConfig(
                    kp=self.calibration.kp,
                    ki=self.calibration.ki,
                    kd=self.calibration.kd,
                    output_min=-self.calibration.max_speed,
                    output_max=self.calibration.max_speed,
                    integral_min=-self.calibration.max_speed,
                    integral_max=self.calibration.max_speed,
                )
            )
            self.controllers[servo_id] = MagneticEncoderPositionPIDController(
                pid,
                MagneticEncoderPositionControlConfig(
                    target_position_deg=0.0,
                    max_speed=self.calibration.max_speed,
                    hard_position_error_deg=hard_error,
                    max_feedback_age_s=self.config.telemetry_stale_s,
                ),
            )

        missing = [
            servo_id
            for servo_id in self.servo_ids_by_tendon
            if servo_id not in self.encoder_sensor_ids_by_servo_id
        ]
        if missing:
            raise ValueError(f"Missing encoder sensor IDs for servo IDs: {missing}")

    def run(self) -> None:
        start_feedback = self.hardware.wait_for_feedback(
            self.config.telemetry_wait_s,
            require_encoder=True,
        )
        start_positions = self._encoder_positions(start_feedback)
        self._reset_controllers()
        started_at = time.monotonic()
        self._run_playback(started_at, start_positions)
        if self.config.return_to_start:
            self._run_return(start_positions, started_at)
        self.hardware.stop_servos()

    def _run_playback(
        self,
        started_at: float,
        start_positions: dict[int, float],
    ) -> None:
        next_tick = time.monotonic()
        last_tick = next_tick
        next_display = next_tick
        last_speeds = {servo_id: 0.0 for servo_id in self.servo_ids_by_tendon}
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
            actual = self._encoder_positions(feedback)
            excursions = self.trajectory.excursions_at(elapsed_s)
            targets = self._targets_from_excursions(excursions, start_positions)
            dt_s = max(1e-6, now - last_tick)
            force_readings = self._force_readings(feedback)
            self._check_hard_force_limits(force_readings, elapsed_s)
            speeds = self._calculate_speeds(
                feedback,
                targets,
                actual,
                dt_s,
                now,
                last_speeds,
            )
            self._check_stop_requested()
            command_submitted_at, command_sequence = self._submit_speed_to_servos(speeds)
            self._notify_sample(
                phase="playback",
                elapsed_s=elapsed_s,
                dt_s=dt_s,
                target_positions=targets,
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
                    target_excursions_mm=excursions,
                    target_positions=targets,
                    actual_positions=actual,
                    speeds=speeds,
                    force_readings=force_readings,
                    control_tick_monotonic_s=now,
                    command_submitted_monotonic_s=command_submitted_at,
                    command_sequence=command_sequence,
                    servo_feedback_received_monotonic_s=feedback.servo_received_at,
                    encoder_received_at_by_servo_id={
                        servo_id: feedback.encoder_received_at.get(sensor_id)
                        for servo_id, sensor_id in self.encoder_sensor_ids_by_servo_id.items()
                    },
                )
            if now >= next_display or sent_final:
                self._print_status("PLAYBACK", elapsed_s, targets, actual, speeds, force_readings)
                next_display = now + self.config.display_interval_s
            if sent_final:
                return
            last_tick = now
            next_tick += 1.0 / self.config.control_hz
            time.sleep(max(0.0, next_tick - time.monotonic()))

    def _run_return(
        self,
        start_positions: dict[int, float],
        experiment_started_at: float,
    ) -> None:
        self._reset_controllers()
        targets = dict(start_positions)
        deadline = time.monotonic() + self.config.return_timeout_s
        next_tick = time.monotonic()
        last_tick = next_tick
        next_display = next_tick
        last_speeds = {servo_id: 0.0 for servo_id in self.servo_ids_by_tendon}
        stable_frames = 0
        while True:
            self._check_stop_requested()
            now = time.monotonic()
            self.hardware.check_health()
            if now >= deadline:
                raise TimeoutError("Encoders did not return to their initial positions")
            feedback = self.hardware.latest_feedback(self.config.telemetry_stale_s)
            if feedback is None:
                raise TimeoutError("Servo telemetry became unavailable during return")
            actual = self._encoder_positions(feedback)
            dt_s = max(1e-6, now - last_tick)
            force_readings = self._force_readings(feedback)
            self._check_hard_force_limits(
                force_readings,
                now - experiment_started_at,
            )
            speeds = self._calculate_speeds(
                feedback,
                targets,
                actual,
                dt_s,
                now,
                last_speeds,
            )
            self._check_stop_requested()
            command_submitted_at, command_sequence = self._submit_speed_to_servos(speeds)
            reached = all(
                abs(targets[servo_id] - actual[servo_id])
                <= self.calibration.return_tolerance_deg
                for servo_id in self.servo_ids_by_tendon
            )
            stable_frames = stable_frames + 1 if reached else 0
            self._notify_sample(
                phase="return",
                elapsed_s=now - experiment_started_at,
                dt_s=dt_s,
                target_positions=targets,
                actual_positions=actual,
                speeds=speeds,
                force_readings=force_readings,
            )
            if self.recorder is not None:
                self.recorder.write(
                    phase="return",
                    elapsed_s=now - experiment_started_at,
                    dt_s=dt_s,
                    servo_telemetry_age_s=feedback.servo_age_s,
                    target_excursions_mm=self.trajectory.initial_excursion_mm,
                    target_positions=targets,
                    actual_positions=actual,
                    speeds=speeds,
                    force_readings=force_readings,
                    control_tick_monotonic_s=now,
                    command_submitted_monotonic_s=command_submitted_at,
                    command_sequence=command_sequence,
                    servo_feedback_received_monotonic_s=feedback.servo_received_at,
                    encoder_received_at_by_servo_id={
                        servo_id: feedback.encoder_received_at.get(sensor_id)
                        for servo_id, sensor_id in self.encoder_sensor_ids_by_servo_id.items()
                    },
                )
            if now >= next_display or stable_frames >= self.config.return_stable_frames:
                self._print_status(
                    "RETURN",
                    now - experiment_started_at,
                    targets,
                    actual,
                    speeds,
                    force_readings,
                )
                next_display = now + self.config.display_interval_s
            if stable_frames >= self.config.return_stable_frames:
                return
            last_tick = now
            next_tick += 1.0 / self.config.control_hz
            time.sleep(max(0.0, next_tick - time.monotonic()))

    def _targets_from_excursions(
        self,
        excursions: tuple[float, ...],
        start_positions: Mapping[int, float],
    ) -> dict[int, float]:
        initial = self.trajectory.initial_excursion_mm
        targets: dict[int, float] = {}
        for tendon_index, servo_id in enumerate(self.servo_ids_by_tendon):
            targets[servo_id] = start_positions[servo_id] + (
                excursions[tendon_index] - initial[tendon_index]
            ) * self.calibration.degrees_per_mm[tendon_index]
        return targets

    def _encoder_positions(self, feedback: HardwareFeedback) -> dict[int, float]:
        if feedback.servo_age_s > self.config.telemetry_stale_s:
            raise MagneticEncoderPositionSafetyError("Servo feedback is stale")
        positions: dict[int, float] = {}
        for tendon_index, servo_id in enumerate(self.servo_ids_by_tendon):
            sensor_id = self.encoder_sensor_ids_by_servo_id[servo_id]
            sample = feedback.encoder_by_sensor.get(sensor_id)
            age_s = feedback.encoder_age_s(sensor_id)
            if sample is None or age_s is None:
                raise MagneticEncoderPositionSafetyError(
                    f"Magnetic encoder feedback is missing for servo {servo_id}"
                )
            if age_s > self.config.telemetry_stale_s:
                raise MagneticEncoderPositionSafetyError(
                    f"Magnetic encoder feedback is stale for servo {servo_id}"
                )
            positions[servo_id] = encoder_position_from_sample(
                sample,
                encoder_direction=self.calibration.encoder_directions[tendon_index],
            )
        return positions

    def _calculate_speeds(
        self,
        feedback: HardwareFeedback,
        targets: Mapping[int, float],
        actual: Mapping[int, float],
        dt_s: float,
        now: float,
        previous_speeds: dict[int, float],
    ) -> dict[int, int]:
        max_delta = self.calibration.max_acceleration * dt_s
        speeds: dict[int, int] = {}
        try:
            for tendon_index, servo_id in enumerate(self.servo_ids_by_tendon):
                sensor_id = self.encoder_sensor_ids_by_servo_id[servo_id]
                encoder_age_s = feedback.encoder_age_s(sensor_id)
                if encoder_age_s is None:
                    raise MagneticEncoderPositionSafetyError(
                        f"Magnetic encoder feedback is missing for servo {servo_id}"
                    )
                sample = feedback.encoder_by_sensor[sensor_id]
                output = self.controllers[servo_id].step(
                    MagneticEncoderPositionFeedback(
                        position_deg=actual[servo_id],
                        servo_age_s=feedback.servo_age_s,
                        encoder_age_s=encoder_age_s,
                        status=str(sample.get("status", "")),
                    ),
                    target_position_deg=targets[servo_id],
                    now=now,
                )
                requested = float(output.speed_command)
                previous = previous_speeds[servo_id]
                limited = max(previous - max_delta, min(previous + max_delta, requested))
                limited = max(-self.calibration.max_speed, min(self.calibration.max_speed, limited))
                previous_speeds[servo_id] = limited
                speeds[servo_id] = int(round(limited))
        except Exception:
            self.hardware.stop_servos()
            raise
        return speeds

    def _reset_controllers(self) -> None:
        for controller in self.controllers.values():
            controller.reset()

    def _force_readings(
        self,
        feedback: HardwareFeedback,
    ) -> dict[int, dict[str, Any]]:
        if self.force_estimator is None:
            return {}
        return self.force_estimator.estimate(feedback, self._servo_index_by_id())

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

    def _submit_speed_to_servos(
        self,
        speeds: dict[int, int],
    ) -> tuple[float, int | None]:
        submit = getattr(self.hardware, "submit_speed_to_servos", None)
        if callable(submit):
            ticket = submit(speeds)
            generated_at = getattr(ticket, "generated_at", None)
            sequence = getattr(ticket, "sequence", None)
            return (
                time.monotonic() if generated_at is None else float(generated_at),
                None if sequence is None else int(sequence),
            )
        submitted_at = time.monotonic()
        self.hardware.sender.send_speed_to_servos(speeds)
        return submitted_at, None

    def _check_stop_requested(self) -> None:
        if self.stop_event is not None and self.stop_event.is_set():
            raise PlaybackStopped("Playback was stopped by the operator")

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
                f"ID{servo_id}={targets[servo_id]:.2f}/{actual[servo_id]:.2f}deg"
                f"({targets[servo_id] - actual[servo_id]:+.2f})"
                f" v={speeds[servo_id]} F={force_text}"
            )
        print(f"{phase} t={elapsed_s:.3f}s " + " | ".join(parts), flush=True)
