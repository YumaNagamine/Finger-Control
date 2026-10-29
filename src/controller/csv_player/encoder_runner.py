"""CSV excursion playback using magnetic-encoder position PID feedback."""

from __future__ import annotations

import math
import time
from typing import Any, Callable, Mapping, Sequence

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
from controller.rl.action_schedule import (
    INTERPOLATED_PLAYBACK_MODE,
    RL_FF_ZOH_PLAYBACK_MODE,
    ActionDecision,
    RLFFZOHActionSchedule,
)
from controller.rl.raw_event_logger import AsyncRawEventLogger
from controller.rl.action_safety_filter import (
    ActionComponents,
    ActionSafetyContext,
    ActionSafetyContextError,
    ActionSafetyError,
    ActionSafetyFilter,
    ActionSafetyResult,
)
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
        on_playback_clock: Callable[[float, float, bool], None] | None = None,
        stop_event: Any | None = None,
        force_quit_event: Any | None = None,
        raw_event_logger: AsyncRawEventLogger | None = None,
        action_safety_filter: ActionSafetyFilter | None = None,
        start_state_validator: (
            Callable[[HardwareFeedback, Mapping[int, float]], Mapping[str, Any]]
            | None
        ) = None,
        feedback_action_provider: (
            Callable[[float, tuple[float, ...]], Sequence[float]] | None
        ) = None,
        joint_safety_state_provider: (
            Callable[[], tuple[Mapping[str, float | None], float | None] | None]
            | None
        ) = None,
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
        self.on_playback_clock = on_playback_clock
        self.stop_event = stop_event
        self.force_quit_event = force_quit_event
        self.raw_event_logger = raw_event_logger
        self.action_safety_filter = action_safety_filter
        self.start_state_validator = start_state_validator
        self.feedback_action_provider = feedback_action_provider
        self.joint_safety_state_provider = joint_safety_state_provider
        self._previous_relative_deg_by_servo_id: dict[int, float] = {}
        self._unwrapped_position_deg_by_servo_id: dict[int, float] = {}
        self._encoder_received_at_by_servo_id: dict[int, float] = {}
        self._last_playback_elapsed_s = 0.0
        self._record_timeline_index = 0
        self._previous_record_sample: dict[str, Any] | None = None
        self._tick_id = 0
        self._active_decision: ActionDecision | None = None
        self._active_decision_applied_at: float | None = None
        self._active_action_safety_result: ActionSafetyResult | None = None
        self._previous_applied_action_mm: tuple[float, ...] | None = None
        self._pending_command_tickets: dict[int, Any] = {}
        self._control_period_violation_count = 0
        self._max_control_period_error_s = 0.0
        self._zoh_schedule = (
            RLFFZOHActionSchedule(
                self.trajectory,
                decision_hz=self.config.decision_hz,
            )
            if self.config.playback_mode == RL_FF_ZOH_PLAYBACK_MODE
            else None
        )
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
                    allow_multiturn_uncertain=True,
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
        if not self._feedback_is_fresh(start_feedback):
            self.hardware.stop_servos()
            start_feedback = self._wait_for_fresh_feedback()
        start_positions = self._encoder_positions(start_feedback)
        start_state_validation = None
        if self.start_state_validator is not None:
            try:
                start_state_validation = self.start_state_validator(
                    start_feedback,
                    start_positions,
                )
                if not isinstance(start_state_validation, Mapping):
                    raise ValueError("start_state_validator must return a mapping")
            except Exception as exc:
                self._log_raw_event(
                    "point_to_point_start_state_rejected",
                    start_feedback.sampled_at,
                    {
                        "reason": f"{type(exc).__name__}: {exc}",
                        "runner_encoder_origin_deg": {
                            tendon: float(start_positions[servo_id])
                            for tendon, servo_id in zip(
                                TENDONS,
                                self.servo_ids_by_tendon,
                            )
                        },
                    },
                )
                raise
        initial_force_readings = self._force_readings(start_feedback)
        self._log_session_start(
            start_feedback,
            start_positions,
            initial_force_readings,
            start_state_validation=start_state_validation,
        )
        self._reset_controllers()
        if self._zoh_schedule is not None:
            self._zoh_schedule.reset()
        self._active_action_safety_result = None
        self._previous_applied_action_mm = (0.0,) * len(TENDONS)
        started_at = time.monotonic()
        try:
            try:
                self._notify_playback_clock(
                    started_at,
                    self.trajectory.source_time_origin_s,
                    finished=False,
                )
                try:
                    self._run_playback(started_at, start_positions)
                finally:
                    playback_finished_at = time.monotonic()
                    self._notify_playback_clock(
                        playback_finished_at,
                        self.trajectory.source_time_origin_s
                        + self._last_playback_elapsed_s * self.trajectory.time_scale,
                        finished=True,
                    )
                    self._log_raw_event(
                        "playback_end",
                        playback_finished_at,
                        {
                            "playback_elapsed_s": self._last_playback_elapsed_s,
                            "source_elapsed_s": (
                                self.trajectory.source_time_origin_s
                                + self._last_playback_elapsed_s
                                * self.trajectory.time_scale
                            ),
                        },
                    )
            except PlaybackStopped:
                if self._force_quit_requested():
                    raise
                if self.config.return_to_start:
                    if self.stop_event is not None:
                        self.stop_event.clear()
                    self._run_return(start_positions, started_at)
                raise
            except Exception as playback_error:
                if self._force_quit_requested() or not self.config.return_to_start:
                    raise
                if self.stop_event is not None:
                    self.stop_event.clear()
                try:
                    self._run_return(start_positions, started_at)
                except Exception as return_error:
                    raise playback_error from return_error
                raise
            if self.config.return_to_start:
                self._run_return(start_positions, started_at)
            self.hardware.stop_servos()
        finally:
            self._log_completed_command_receipts(wait_timeout_s=0.25)
            self._log_control_summary()

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
            self._log_completed_command_receipts()
            self.hardware.check_health()
            feedback_lookup_started_at = time.monotonic()
            feedback = self.hardware.latest_feedback(self.config.telemetry_stale_s)
            if not self._feedback_is_fresh(feedback):
                if self.action_safety_filter is not None:
                    self._request_latched_stop("servo_or_encoder_feedback_stale")
                    raise ActionSafetyContextError(
                        ("servo_or_encoder_feedback_stale",)
                    )
                self.hardware.stop_servos()
                pause_started_at = now
                feedback = self._wait_for_fresh_feedback()
                started_at += time.monotonic() - pause_started_at
                now = time.monotonic()
                next_tick = now
                last_tick = now
                last_speeds = {
                    servo_id: 0.0 for servo_id in self.servo_ids_by_tendon
                }
                self._reset_controllers()
            feedback_lookup_s = time.monotonic() - feedback_lookup_started_at
            elapsed_s = now - started_at
            if elapsed_s >= self.trajectory.duration_s:
                elapsed_s = self.trajectory.duration_s
                sent_final = True
            encoder_conversion_started_at = time.monotonic()
            actual = self._encoder_positions(feedback)
            encoder_conversion_s = time.monotonic() - encoder_conversion_started_at
            force_check_started_at = time.monotonic()
            force_readings = self._force_readings(feedback)
            self._check_hard_force_limits(force_readings, elapsed_s)
            force_check_s = time.monotonic() - force_check_started_at
            dt_s = max(1e-6, now - last_tick)
            self._check_fast_safety_context(
                feedback=feedback,
                force_readings=force_readings,
                scheduled_tick_monotonic_s=next_tick,
                evaluated_at_monotonic_s=now,
            )
            action_started_at = time.monotonic()
            excursions, action_excursion_mm = self._playback_action(
                elapsed_s,
                applied_at_monotonic_s=now,
                allow_new_decision=not sent_final or self._active_decision is None,
                feedback=feedback,
                force_readings=force_readings,
                scheduled_tick_monotonic_s=next_tick,
            )
            action_composition_s = time.monotonic() - action_started_at
            target_conversion_started_at = time.monotonic()
            targets = self._targets_from_excursions(excursions, start_positions)
            target_conversion_s = time.monotonic() - target_conversion_started_at
            self._record_control_period(dt_s)
            pid_started_at = time.monotonic()
            speeds, pid_details = self._calculate_speeds(
                feedback,
                targets,
                actual,
                dt_s,
                now,
                last_speeds,
            )
            pid_compute_s = time.monotonic() - pid_started_at
            self._check_stop_requested()
            dispatcher_started_at = time.monotonic()
            command_submitted_at, command_sequence = self._submit_speed_to_servos(
                speeds
            )
            dispatcher_submit_s = time.monotonic() - dispatcher_started_at
            source_elapsed_s = (
                self.trajectory.source_time_origin_s
                + elapsed_s * self.trajectory.time_scale
            )
            self._last_playback_elapsed_s = elapsed_s
            self._log_pid_tick(
                phase="playback",
                tick_monotonic_s=now,
                scheduled_tick_monotonic_s=next_tick,
                elapsed_s=elapsed_s,
                dt_s=dt_s,
                feedback=feedback,
                action_excursion_mm=action_excursion_mm,
                targets=targets,
                actual=actual,
                pid_details=pid_details,
                speeds=speeds,
                force_readings=force_readings,
                command_submitted_at=command_submitted_at,
                command_sequence=command_sequence,
                stage_durations_s={
                    "feedback_lookup": feedback_lookup_s,
                    "encoder_conversion": encoder_conversion_s,
                    "force_estimation_and_check": force_check_s,
                    "action_composition_and_filter": action_composition_s,
                    "target_conversion": target_conversion_s,
                    "pid_compute": pid_compute_s,
                    "dispatcher_submit": dispatcher_submit_s,
                    "instrumented_pipeline_total": (
                        feedback_lookup_s
                        + encoder_conversion_s
                        + force_check_s
                        + action_composition_s
                        + target_conversion_s
                        + pid_compute_s
                        + dispatcher_submit_s
                    ),
                },
            )
            self._notify_playback_clock(now, source_elapsed_s, finished=False)
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
                self._record_playback_sample(
                    source_elapsed_s=source_elapsed_s,
                    servo_telemetry_age_s=feedback.servo_age_s,
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
                    start_positions=start_positions,
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
            self._log_completed_command_receipts()
            self.hardware.check_health()
            if now >= deadline:
                raise TimeoutError("Encoders did not return to their initial positions")
            feedback_lookup_started_at = time.monotonic()
            feedback = self.hardware.latest_feedback(self.config.telemetry_stale_s)
            if not self._feedback_is_fresh(feedback):
                self.hardware.stop_servos()
                pause_started_at = now
                feedback = self._wait_for_fresh_feedback()
                deadline += time.monotonic() - pause_started_at
                now = time.monotonic()
                next_tick = now
                last_tick = now
                last_speeds = {
                    servo_id: 0.0 for servo_id in self.servo_ids_by_tendon
                }
                self._reset_controllers()
            feedback_lookup_s = time.monotonic() - feedback_lookup_started_at
            encoder_conversion_started_at = time.monotonic()
            actual = self._encoder_positions(feedback)
            encoder_conversion_s = time.monotonic() - encoder_conversion_started_at
            dt_s = max(1e-6, now - last_tick)
            self._record_control_period(dt_s)
            force_check_started_at = time.monotonic()
            force_readings = self._force_readings(feedback)
            self._check_hard_force_limits(
                force_readings,
                now - experiment_started_at,
            )
            force_check_s = time.monotonic() - force_check_started_at
            pid_started_at = time.monotonic()
            speeds, pid_details = self._calculate_speeds(
                feedback,
                targets,
                actual,
                dt_s,
                now,
                last_speeds,
            )
            pid_compute_s = time.monotonic() - pid_started_at
            self._check_stop_requested()
            dispatcher_started_at = time.monotonic()
            command_submitted_at, command_sequence = self._submit_speed_to_servos(
                speeds
            )
            dispatcher_submit_s = time.monotonic() - dispatcher_started_at
            self._log_pid_tick(
                phase="return",
                tick_monotonic_s=now,
                scheduled_tick_monotonic_s=next_tick,
                elapsed_s=now - experiment_started_at,
                dt_s=dt_s,
                feedback=feedback,
                action_excursion_mm=(0.0,) * len(TENDONS),
                targets=targets,
                actual=actual,
                pid_details=pid_details,
                speeds=speeds,
                force_readings=force_readings,
                command_submitted_at=command_submitted_at,
                command_sequence=command_sequence,
                stage_durations_s={
                    "feedback_lookup": feedback_lookup_s,
                    "encoder_conversion": encoder_conversion_s,
                    "force_estimation_and_check": force_check_s,
                    "action_composition_and_filter": 0.0,
                    "target_conversion": 0.0,
                    "pid_compute": pid_compute_s,
                    "dispatcher_submit": dispatcher_submit_s,
                    "instrumented_pipeline_total": (
                        feedback_lookup_s
                        + encoder_conversion_s
                        + force_check_s
                        + pid_compute_s
                        + dispatcher_submit_s
                    ),
                },
            )
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

    def _playback_action(
        self,
        elapsed_s: float,
        *,
        applied_at_monotonic_s: float,
        allow_new_decision: bool,
        feedback: HardwareFeedback,
        force_readings: Mapping[int, Mapping[str, object]],
        scheduled_tick_monotonic_s: float,
    ) -> tuple[tuple[float, ...], tuple[float, ...]]:
        initial = self.trajectory.initial_excursion_mm
        if self.config.playback_mode == INTERPOLATED_PLAYBACK_MODE:
            excursions = self.trajectory.excursions_at(elapsed_s)
            nominal_action = tuple(
                float(value - origin)
                for value, origin in zip(excursions, initial)
            )
            result = self._evaluate_action(
                nominal_action,
                elapsed_s=elapsed_s,
                evaluated_at_monotonic_s=applied_at_monotonic_s,
                scheduled_tick_monotonic_s=scheduled_tick_monotonic_s,
                feedback=feedback,
                force_readings=force_readings,
            )
            self._active_action_safety_result = result
            self._previous_applied_action_mm = result.applied_action_mm
            applied_excursions = tuple(
                float(origin + delta)
                for origin, delta in zip(initial, result.applied_action_mm)
            )
            return applied_excursions, result.applied_action_mm

        if self._zoh_schedule is None:
            raise RuntimeError("rl_ff_zoh mode requires an action schedule")
        decision = (
            self._zoh_schedule.advance(elapsed_s)
            if allow_new_decision
            else None
        )
        if decision is not None:
            previous_pid_target = (
                (0.0,) * len(TENDONS)
                if self._previous_applied_action_mm is None
                else self._previous_applied_action_mm
            )
            result = self._evaluate_action(
                decision.action_excursion_mm,
                elapsed_s=elapsed_s,
                evaluated_at_monotonic_s=applied_at_monotonic_s,
                scheduled_tick_monotonic_s=scheduled_tick_monotonic_s,
                feedback=feedback,
                force_readings=force_readings,
            )
            self._active_decision = decision
            self._active_decision_applied_at = applied_at_monotonic_s
            self._active_action_safety_result = result
            self._previous_applied_action_mm = result.applied_action_mm
            self._log_raw_event(
                "decision",
                applied_at_monotonic_s,
                {
                    "decision_id": decision.decision_id,
                    "decision_hz": self.config.decision_hz,
                    "scheduled_elapsed_s": decision.scheduled_elapsed_s,
                    "decided_monotonic_s": applied_at_monotonic_s,
                    "pid_target_reflected_monotonic_s": applied_at_monotonic_s,
                    "action_excursion_mm": dict(
                        zip(TENDONS, result.applied_action_mm)
                    ),
                    "previous_pid_reflected_target_mm": dict(
                        zip(TENDONS, previous_pid_target)
                    ),
                    "requested_target_mm": dict(
                        zip(TENDONS, result.requested_action_mm)
                    ),
                    "requested_delta_mm": dict(
                        zip(
                            TENDONS,
                            (
                                current - previous
                                for current, previous in zip(
                                    result.requested_action_mm,
                                    previous_pid_target,
                                )
                            ),
                        )
                    ),
                    "filtered_target_mm": dict(
                        zip(TENDONS, result.applied_action_mm)
                    ),
                    "pid_reflected_target_mm": dict(
                        zip(TENDONS, result.applied_action_mm)
                    ),
                    "pid_reflected_delta_mm": dict(
                        zip(
                            TENDONS,
                            (
                                current - previous
                                for current, previous in zip(
                                    result.applied_action_mm,
                                    previous_pid_target,
                                )
                            ),
                        )
                    ),
                    **result.as_payload(),
                    "safety_filter_enabled": self.action_safety_filter is not None,
                    "action_reference": "session_encoder_origin",
                    "action_is_incremental": False,
                    "rl_dataset_action_is_incremental": True,
                    "skipped_unapplied_decisions": decision.skipped_decisions,
                    "physical_servo_applied_monotonic_s": None,
                    "physical_servo_applied_time_status": "unobserved",
                },
            )
        if self._active_decision is None:
            raise RuntimeError("No active rl_ff_zoh decision")
        if self._active_action_safety_result is None:
            raise RuntimeError("No active action result")
        action = self._active_action_safety_result.applied_action_mm
        excursions = tuple(
            float(origin + delta)
            for origin, delta in zip(initial, action)
        )
        return excursions, action

    def _evaluate_action(
        self,
        nominal_action_mm: Sequence[float],
        *,
        elapsed_s: float,
        evaluated_at_monotonic_s: float,
        scheduled_tick_monotonic_s: float,
        feedback: HardwareFeedback,
        force_readings: Mapping[int, Mapping[str, object]],
    ) -> ActionSafetyResult:
        nominal = tuple(float(value) for value in nominal_action_mm)
        feedback_action = (
            (0.0,) * len(TENDONS)
            if self.feedback_action_provider is None
            else tuple(
                float(value)
                for value in self.feedback_action_provider(elapsed_s, nominal)
            )
        )
        components = ActionComponents(
            nominal_action_mm=nominal,
            feedback_action_mm=feedback_action,
            exploration_action_mm=(0.0,) * len(TENDONS),
        )
        if self.action_safety_filter is None:
            requested = components.requested_action_mm
            return ActionSafetyResult(
                components=components,
                requested_action_mm=requested,
                applied_action_mm=requested,
                intervened=False,
                rejected=False,
                reasons=(),
            )

        joint_angles: Mapping[str, float | None] | None = None
        joint_age_s: float | None = None
        if self.joint_safety_state_provider is not None:
            joint_state = self.joint_safety_state_provider()
            if joint_state is not None:
                joint_angles, joint_age_s = joint_state
        context = ActionSafetyContext(
            previous_applied_action_mm=self._previous_applied_action_mm,
            force_n_by_tendon={
                tendon: force_readings.get(servo_id, {}).get("force_n")
                for tendon, servo_id in zip(TENDONS, self.servo_ids_by_tendon)
            },
            joint_angles_rad=joint_angles,
            servo_age_s=feedback.servo_age_s,
            encoder_age_s_by_tendon={
                tendon: feedback.encoder_age_s(
                    self.encoder_sensor_ids_by_servo_id[servo_id]
                )
                for tendon, servo_id in zip(TENDONS, self.servo_ids_by_tendon)
            },
            joint_age_s=joint_age_s,
            deadline_lateness_s=max(
                0.0,
                evaluated_at_monotonic_s - scheduled_tick_monotonic_s,
            ),
        )
        result = self.action_safety_filter.evaluate(components, context)
        if result.intervened:
            self._log_raw_event(
                "action_safety_intervention",
                evaluated_at_monotonic_s,
                {
                    "elapsed_s": float(elapsed_s),
                    **result.as_payload(),
                },
            )
        if result.rejected:
            raise ActionSafetyError(result)
        return result

    def _check_fast_safety_context(
        self,
        *,
        feedback: HardwareFeedback,
        force_readings: Mapping[int, Mapping[str, object]],
        scheduled_tick_monotonic_s: float,
        evaluated_at_monotonic_s: float,
    ) -> None:
        """Fail closed on safety telemetry at every lower-control tick.

        Action bounds and action deltas remain decision-rate checks.  Sensor
        freshness, force validity, joint feedback freshness, and tick deadline
        are plant-state checks and must not wait for the next ZOH decision.
        """

        if self.action_safety_filter is None:
            return
        joint_angles: Mapping[str, float | None] | None = None
        joint_age_s: float | None = None
        if self.joint_safety_state_provider is not None:
            joint_state = self.joint_safety_state_provider()
            if joint_state is not None:
                joint_angles, joint_age_s = joint_state
        context = ActionSafetyContext(
            force_n_by_tendon={
                tendon: force_readings.get(servo_id, {}).get("force_n")
                for tendon, servo_id in zip(TENDONS, self.servo_ids_by_tendon)
            },
            joint_angles_rad=joint_angles,
            servo_age_s=feedback.servo_age_s,
            encoder_age_s_by_tendon={
                tendon: feedback.encoder_age_s(
                    self.encoder_sensor_ids_by_servo_id[servo_id]
                )
                for tendon, servo_id in zip(TENDONS, self.servo_ids_by_tendon)
            },
            joint_age_s=joint_age_s,
            deadline_lateness_s=max(
                0.0,
                evaluated_at_monotonic_s - scheduled_tick_monotonic_s,
            ),
        )
        reasons = self.action_safety_filter.context_rejection_reasons(context)
        if not reasons:
            return
        self._log_raw_event(
            "safety_context_rejection",
            evaluated_at_monotonic_s,
            {
                "reasons": list(reasons),
                "monitor_layer": "lower_control_tick",
                "scheduled_tick_monotonic_s": float(scheduled_tick_monotonic_s),
            },
        )
        self._request_latched_stop("safety_context_rejection")
        raise ActionSafetyContextError(reasons)

    def _record_control_period(self, dt_s: float) -> None:
        if self._tick_id == 0:
            return
        expected_s = 1.0 / self.config.control_hz
        error_s = abs(float(dt_s) - expected_s)
        self._max_control_period_error_s = max(
            self._max_control_period_error_s,
            error_s,
        )
        if error_s > expected_s * self.config.control_period_tolerance_fraction:
            self._control_period_violation_count += 1

    def _log_session_start(
        self,
        feedback: HardwareFeedback,
        start_positions: Mapping[int, float],
        force_readings: Mapping[int, Mapping[str, object]],
        *,
        start_state_validation: Mapping[str, Any] | None = None,
    ) -> None:
        self._log_raw_event(
            "control_session_start",
            feedback.sampled_at,
            {
                "playback_mode": self.config.playback_mode,
                "control_hz": self.config.control_hz,
                "decision_hz": self.config.decision_hz,
                "tendon_order": list(TENDONS),
                "action_components": [
                    "nominal_action_excursion_mm",
                    "feedback_action_excursion_mm",
                    "exploration_action_excursion_mm",
                    "requested_action_excursion_mm",
                    "applied_action_excursion_mm",
                ],
                "feedback_action_provider_enabled": (
                    self.feedback_action_provider is not None
                ),
                "safety_filter_enabled": self.action_safety_filter is not None,
                "action_definition": (
                    "six-dimensional target tendon excursion in millimetres "
                    "relative to the encoder position measured at session start"
                ),
                "session_encoder_origin_deg": {
                    tendon: float(start_positions[servo_id])
                    for tendon, servo_id in zip(
                        TENDONS,
                        self.servo_ids_by_tendon,
                    )
                },
                "source_excursion_origin_mm": dict(
                    zip(TENDONS, self.trajectory.initial_excursion_mm)
                ),
                "initial_force_n": {
                    tendon: force_readings.get(servo_id, {}).get("force_n")
                    for tendon, servo_id in zip(
                        TENDONS,
                        self.servo_ids_by_tendon,
                    )
                },
                "initial_force_status": {
                    tendon: force_readings.get(servo_id, {}).get(
                        "status",
                        "NOT_RECORDED",
                    )
                    for tendon, servo_id in zip(
                        TENDONS,
                        self.servo_ids_by_tendon,
                    )
                },
                "p2p_start_state_validation": (
                    None
                    if start_state_validation is None
                    else dict(start_state_validation)
                ),
                "calibration": {
                    "encoder_degrees_per_mm": dict(
                        zip(TENDONS, self.calibration.degrees_per_mm)
                    ),
                    "encoder_direction": dict(
                        zip(TENDONS, self.calibration.encoder_directions)
                    ),
                    "kp": self.calibration.kp,
                    "ki": self.calibration.ki,
                    "kd": self.calibration.kd,
                    "max_speed": self.calibration.max_speed,
                    "max_acceleration": self.calibration.max_acceleration,
                    "hard_error_margin_deg": self.calibration.hard_error_margin_deg,
                    "minimum_hard_error_deg": self.calibration.minimum_hard_error_deg,
                    "return_tolerance_deg": self.calibration.return_tolerance_deg,
                    "calibration_status": self.calibration.calibration_status,
                },
                "servo_feedback_received_monotonic_s": feedback.servo_received_at,
                "encoder_received_monotonic_s": {
                    tendon: feedback.encoder_received_at.get(
                        self.encoder_sensor_ids_by_servo_id[servo_id]
                    )
                    for tendon, servo_id in zip(
                        TENDONS,
                        self.servo_ids_by_tendon,
                    )
                },
            },
        )

    def _log_pid_tick(
        self,
        *,
        phase: str,
        tick_monotonic_s: float,
        scheduled_tick_monotonic_s: float,
        elapsed_s: float,
        dt_s: float,
        feedback: HardwareFeedback,
        action_excursion_mm: tuple[float, ...],
        targets: Mapping[int, float],
        actual: Mapping[int, float],
        pid_details: Mapping[int, Mapping[str, float | int | bool]],
        speeds: Mapping[int, int],
        force_readings: Mapping[int, Mapping[str, object]],
        command_submitted_at: float,
        command_sequence: int | None,
        stage_durations_s: Mapping[str, float],
    ) -> None:
        action_result = (
            self._active_action_safety_result
            if phase == "playback"
            else None
        )
        tendon_state: dict[str, dict[str, object]] = {}
        for tendon_index, (tendon, servo_id) in enumerate(
            zip(TENDONS, self.servo_ids_by_tendon)
        ):
            sensor_id = self.encoder_sensor_ids_by_servo_id[servo_id]
            force = force_readings.get(servo_id, {})
            state: dict[str, object] = {
                "servo_id": servo_id,
                "encoder_sensor_id": sensor_id,
                "action_excursion_mm": float(action_excursion_mm[tendon_index]),
                "target_encoder_deg": float(targets[servo_id]),
                "actual_encoder_deg": float(actual[servo_id]),
                "encoder_received_monotonic_s": feedback.encoder_received_at.get(
                    sensor_id
                ),
                "encoder_age_s": feedback.encoder_age_s(sensor_id),
                "speed_command": int(speeds[servo_id]),
                "force_n": force.get("force_n"),
                "force_status": force.get("status", "NOT_RECORDED"),
                "pid": dict(pid_details[servo_id]),
            }
            if action_result is not None:
                state.update(
                    {
                        "nominal_action_excursion_mm": float(
                            action_result.components.nominal_action_mm[tendon_index]
                        ),
                        "feedback_action_excursion_mm": float(
                            action_result.components.feedback_action_mm[tendon_index]
                        ),
                        "exploration_action_excursion_mm": float(
                            action_result.components.exploration_action_mm[
                                tendon_index
                            ]
                        ),
                        "requested_action_excursion_mm": float(
                            action_result.requested_action_mm[tendon_index]
                        ),
                        "applied_action_excursion_mm": float(
                            action_result.applied_action_mm[tendon_index]
                        ),
                    }
                )
            tendon_state[tendon] = state
        expected_period_s = 1.0 / self.config.control_hz
        self._log_raw_event(
            "pid_tick",
            tick_monotonic_s,
            {
                "tick_id": self._tick_id,
                "phase": phase,
                "decision_id": (
                    self._active_decision.decision_id
                    if phase == "playback" and self._active_decision is not None
                    else None
                ),
                "decision_reflected_monotonic_s": (
                    self._active_decision_applied_at
                    if phase == "playback"
                    else None
                ),
                "elapsed_s": float(elapsed_s),
                "scheduled_tick_monotonic_s": float(
                    scheduled_tick_monotonic_s
                ),
                "tick_lateness_s": max(
                    0.0,
                    tick_monotonic_s - scheduled_tick_monotonic_s,
                ),
                "dt_s": float(dt_s),
                "expected_period_s": expected_period_s,
                "period_error_s": float(dt_s) - expected_period_s,
                "period_violation": abs(dt_s - expected_period_s)
                > expected_period_s
                * self.config.control_period_tolerance_fraction,
                "stage_durations_s": {
                    str(name): float(value)
                    for name, value in stage_durations_s.items()
                },
                "sensor_snapshot_monotonic_s": feedback.sampled_at,
                "servo_feedback_received_monotonic_s": feedback.servo_received_at,
                "servo_feedback_age_s": feedback.servo_age_s,
                "dispatcher_submitted_monotonic_s": command_submitted_at,
                "command_sequence": command_sequence,
                "transport_write_started_monotonic_s": None,
                "transport_write_finished_monotonic_s": None,
                "physical_servo_applied_monotonic_s": None,
                "physical_servo_applied_time_status": "unobserved",
                **(
                    {}
                    if action_result is None
                    else action_result.as_payload()
                ),
                "tendons": tendon_state,
            },
        )
        self._tick_id += 1

    def _log_control_summary(self) -> None:
        self._log_raw_event(
            "control_loop_end",
            time.monotonic(),
            {
                "pid_tick_count": self._tick_id,
                "control_period_violation_count": (
                    self._control_period_violation_count
                ),
                "max_control_period_error_s": self._max_control_period_error_s,
                "pending_transport_receipts": len(
                    self._pending_command_tickets
                ),
            },
        )

    def _log_raw_event(
        self,
        event_type: str,
        event_monotonic_s: float,
        payload: Mapping[str, object],
    ) -> None:
        if self.raw_event_logger is None:
            return
        self.raw_event_logger.log(
            event_type,
            event_monotonic_s=event_monotonic_s,
            payload=payload,
        )

    def _record_playback_sample(
        self,
        *,
        source_elapsed_s: float,
        servo_telemetry_age_s: float,
        actual_positions: dict[int, float],
        speeds: dict[int, int],
        force_readings: Mapping[int, Mapping[str, object]],
        control_tick_monotonic_s: float,
        command_submitted_monotonic_s: float,
        command_sequence: int,
        servo_feedback_received_monotonic_s: float,
        encoder_received_at_by_servo_id: Mapping[int, float | None],
        start_positions: Mapping[int, float],
    ) -> None:
        current = {
            "source_elapsed_s": source_elapsed_s,
            "servo_telemetry_age_s": servo_telemetry_age_s,
            "actual_positions": dict(actual_positions),
            "speeds": dict(speeds),
            "force_readings": force_readings,
            "control_tick_monotonic_s": control_tick_monotonic_s,
            "command_submitted_monotonic_s": command_submitted_monotonic_s,
            "command_sequence": command_sequence,
            "servo_feedback_received_monotonic_s": servo_feedback_received_monotonic_s,
            "encoder_received_at_by_servo_id": dict(encoder_received_at_by_servo_id),
        }
        timestamps = self.trajectory.recorded_elapsed_times_s
        previous = self._previous_record_sample
        while (
            self._record_timeline_index < len(timestamps)
            and timestamps[self._record_timeline_index] <= source_elapsed_s
        ):
            timestamp_s = timestamps[self._record_timeline_index]
            if previous is None:
                selected = current
            elif timestamp_s <= previous["source_elapsed_s"]:
                selected = previous
            elif (
                timestamp_s - previous["source_elapsed_s"]
                <= source_elapsed_s - timestamp_s
            ):
                selected = previous
            else:
                selected = current

            target_excursions = self.trajectory.excursions_mm[
                self._record_timeline_index
            ]
            target_positions_at_time = self._targets_from_excursions(
                target_excursions,
                start_positions,
            )
            previous_timestamp_s = (
                timestamps[self._record_timeline_index - 1]
                if self._record_timeline_index > 0
                else timestamp_s
            )
            self.recorder.write(
                phase="playback",
                elapsed_s=timestamp_s,
                dt_s=timestamp_s - previous_timestamp_s,
                servo_telemetry_age_s=selected["servo_telemetry_age_s"],
                target_excursions_mm=target_excursions,
                target_positions=target_positions_at_time,
                actual_positions=selected["actual_positions"],
                speeds=selected["speeds"],
                force_readings=selected["force_readings"],
                control_tick_monotonic_s=selected["control_tick_monotonic_s"],
                command_submitted_monotonic_s=selected["command_submitted_monotonic_s"],
                command_sequence=selected["command_sequence"],
                servo_feedback_received_monotonic_s=selected[
                    "servo_feedback_received_monotonic_s"
                ],
                encoder_received_at_by_servo_id=selected[
                    "encoder_received_at_by_servo_id"
                ],
            )
            self._record_timeline_index += 1
        self._previous_record_sample = current

    def _notify_playback_clock(
        self,
        monotonic_s: float,
        source_elapsed_s: float,
        *,
        finished: bool,
    ) -> None:
        if self.on_playback_clock is None:
            return
        try:
            self.on_playback_clock(monotonic_s, source_elapsed_s, finished)
        except Exception:
            # Clock recording is auxiliary and must not interrupt motor control.
            return

    def _feedback_is_fresh(self, feedback: HardwareFeedback | None) -> bool:
        if feedback is None or feedback.servo_age_s > self.config.telemetry_stale_s:
            return False
        encoder_freshness_limit_s = max(
            0.0,
            self.config.telemetry_stale_s - 1.0 / self.config.control_hz,
        )
        for sensor_id in self.encoder_sensor_ids_by_servo_id.values():
            age_s = feedback.encoder_age_s(sensor_id)
            if (
                sensor_id not in feedback.encoder_by_sensor
                or age_s is None
                or age_s > encoder_freshness_limit_s
            ):
                return False
        return True

    def _wait_for_fresh_feedback(self) -> HardwareFeedback:
        deadline = time.monotonic() + self.config.telemetry_wait_s
        while True:
            self._check_stop_requested()
            self.hardware.check_health()
            feedback = self.hardware.latest_feedback(self.config.telemetry_stale_s)
            if self._feedback_is_fresh(feedback):
                return feedback
            remaining_s = deadline - time.monotonic()
            if remaining_s <= 0.0:
                raise TimeoutError("Encoder telemetry did not recover in time")
            time.sleep(min(0.005, remaining_s))

    def _encoder_positions(self, feedback: HardwareFeedback) -> dict[int, float]:
        if feedback.servo_age_s > self.config.telemetry_stale_s:
            raise MagneticEncoderPositionSafetyError("Servo feedback is stale")
        positions: dict[int, float] = {}
        # CSV excursions are local to this playback, so unwrap the single-turn
        # angle locally instead of relying on the firmware's multi-turn count.
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
            status = str(sample.get("status", "")).strip().upper()
            if status not in {"OK", "MULTITURN_UNCERTAIN"}:
                raise MagneticEncoderPositionSafetyError(
                    "Magnetic encoder status is not safe for position control: "
                    f"{status or 'MISSING'} (servo {servo_id})"
                )
            try:
                relative_deg = float(sample["relative_deg"])
            except (KeyError, TypeError, ValueError) as exc:
                raise MagneticEncoderPositionSafetyError(
                    f"Magnetic encoder single-turn position is invalid for servo {servo_id}"
                ) from exc
            if not math.isfinite(relative_deg):
                raise MagneticEncoderPositionSafetyError(
                    f"Magnetic encoder single-turn position is not finite for servo {servo_id}"
                )
            relative_deg %= 360.0

            received_at = float(feedback.encoder_received_at[sensor_id])
            previous_received_at = self._encoder_received_at_by_servo_id.get(servo_id)
            previous_relative_deg = self._previous_relative_deg_by_servo_id.get(servo_id)
            if previous_received_at is None:
                position_deg = relative_deg
            else:
                if received_at < previous_received_at:
                    raise MagneticEncoderPositionSafetyError(
                        f"Magnetic encoder sample arrived out of order for servo {servo_id}"
                    )
                if previous_relative_deg is None:
                    raise MagneticEncoderPositionSafetyError(
                        f"Magnetic encoder unwrap reference is missing for servo {servo_id}"
                    )

                delta_deg = (relative_deg - previous_relative_deg + 180.0) % 360.0 - 180.0
                if math.isclose(abs(delta_deg), 180.0, abs_tol=1e-9):
                    raise MagneticEncoderPositionSafetyError(
                        f"Magnetic encoder direction is ambiguous after a half-turn for servo {servo_id}"
                    )
                position_deg = self._unwrapped_position_deg_by_servo_id[servo_id] + (
                    self.calibration.encoder_directions[tendon_index] * delta_deg
                )

            if previous_relative_deg is None:
                position_deg *= self.calibration.encoder_directions[tendon_index]
            self._previous_relative_deg_by_servo_id[servo_id] = relative_deg
            self._unwrapped_position_deg_by_servo_id[servo_id] = position_deg
            self._encoder_received_at_by_servo_id[servo_id] = received_at
            positions[servo_id] = position_deg
        return positions

    def _force_quit_requested(self) -> bool:
        return bool(
            self.force_quit_event is not None
            and self.force_quit_event.is_set()
        )

    def _calculate_speeds(
        self,
        feedback: HardwareFeedback,
        targets: Mapping[int, float],
        actual: Mapping[int, float],
        dt_s: float,
        now: float,
        previous_speeds: dict[int, float],
    ) -> tuple[dict[int, int], dict[int, dict[str, float | int | bool]]]:
        max_delta = self.calibration.max_acceleration * dt_s
        speeds: dict[int, int] = {}
        details: dict[int, dict[str, float | int | bool]] = {}
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
                details[servo_id] = {
                    "position_error_deg": float(output.position_error_deg),
                    "pid_raw_speed_command": float(output.raw_speed_command),
                    "pid_bounded_speed_command": int(output.speed_command),
                    "acceleration_limited_speed_command": speeds[servo_id],
                    "limited_by_position": bool(output.limited_by_position),
                }
        except Exception:
            if self.action_safety_filter is not None:
                self._request_latched_stop("pid_calculation_error")
            else:
                self.hardware.stop_servos()
            raise
        return speeds, details

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
                    self._request_latched_stop("force_telemetry_unavailable")
                finally:
                    raise ForceTelemetryUnavailable(
                        tendon=tendon,
                        servo_id=servo_id,
                        status=status,
                    )
            force_n = float(force_value)
            if not math.isfinite(force_n):
                try:
                    self._request_latched_stop("force_telemetry_non_finite")
                finally:
                    raise ForceTelemetryUnavailable(
                        tendon=tendon,
                        servo_id=servo_id,
                        status="NON_FINITE",
                    )
            limit_n = float(limits[tendon])
            if abs(force_n) < limit_n:
                continue
            try:
                self._request_latched_stop("force_hard_limit")
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
            if sequence is not None and ticket is not None:
                self._pending_command_tickets[int(sequence)] = ticket
            return (
                time.monotonic() if generated_at is None else float(generated_at),
                None if sequence is None else int(sequence),
            )
        submitted_at = time.monotonic()
        self.hardware.sender.send_speed_to_servos(speeds)
        write_finished_at = time.monotonic()
        self._log_raw_event(
            "transport_write",
            write_finished_at,
            {
                "command_sequence": None,
                "dispatcher_submitted_monotonic_s": submitted_at,
                "transport_write_started_monotonic_s": submitted_at,
                "transport_write_finished_monotonic_s": write_finished_at,
                "transport_status": "sent_sync_fallback",
                "transport_error": None,
                "physical_servo_applied_monotonic_s": None,
                "physical_servo_applied_time_status": "unobserved",
            },
        )
        return submitted_at, None

    def _request_latched_stop(self, reason: str) -> None:
        """Request one non-blocking, latched zero-speed command and log facts."""

        request = getattr(self.hardware, "request_latched_stop", None)
        if not callable(request):
            self.hardware.stop_servos()
            self._log_raw_event(
                "stop_command_requested",
                time.monotonic(),
                {
                    "reason": reason,
                    "stop_latched": False,
                    "pc_write_status": "legacy_stop_requested",
                    "physical_stop_confirmed": False,
                    "physical_stop_confirmation": "unobserved",
                    "tension_safe_confirmed": False,
                    "tension_safe_confirmation": "unobserved",
                },
            )
            return
        try:
            ticket = request()
        except RuntimeError as exc:
            # Repeated fault paths share a latched dispatcher.  Preserve the
            # first request rather than attempting to clear or replace it.
            if "already stop-latched" not in str(exc):
                raise
            ticket = None
        if ticket is not None:
            sequence = getattr(ticket, "sequence", None)
            if sequence is not None:
                self._pending_command_tickets[int(sequence)] = ticket
        snapshot = getattr(self.hardware, "stop_state_snapshot", lambda: {})()
        self._log_raw_event(
            "stop_command_requested",
            time.monotonic(),
            {
                "reason": reason,
                "stop_latched": True,
                "stop_state": dict(snapshot),
            },
        )

    def _log_completed_command_receipts(
        self,
        *,
        wait_timeout_s: float = 0.0,
    ) -> None:
        for sequence, ticket in tuple(self._pending_command_tickets.items()):
            receipt = getattr(ticket, "receipt", None)
            if receipt is None and wait_timeout_s > 0.0:
                wait = getattr(ticket, "wait", None)
                if callable(wait):
                    try:
                        receipt = wait(wait_timeout_s)
                    except TimeoutError:
                        receipt = None
            if receipt is None:
                continue
            write_finished_at = getattr(receipt, "write_finished_at", None)
            generated_at = float(getattr(receipt, "generated_at"))
            self._log_raw_event(
                "transport_write",
                generated_at
                if write_finished_at is None
                else float(write_finished_at),
                {
                    "command_sequence": int(sequence),
                    "command_kind": str(
                        getattr(ticket, "command_kind", "motion")
                    ),
                    "dispatcher_submitted_monotonic_s": generated_at,
                    "transport_write_started_monotonic_s": getattr(
                        receipt,
                        "write_started_at",
                        None,
                    ),
                    "transport_write_finished_monotonic_s": write_finished_at,
                    "transport_status": str(getattr(receipt, "status", "unknown")),
                    "transport_error": getattr(receipt, "error", None),
                    "physical_servo_applied_monotonic_s": None,
                    "physical_servo_applied_time_status": "unobserved",
                },
            )
            if getattr(ticket, "command_kind", None) == "latched_stop":
                snapshot = getattr(
                    self.hardware,
                    "stop_state_snapshot",
                    lambda: {},
                )()
                self._log_raw_event(
                    "stop_command_write_result",
                    generated_at
                    if write_finished_at is None
                    else float(write_finished_at),
                    {"stop_state": dict(snapshot)},
                )
            del self._pending_command_tickets[sequence]

    def _check_stop_requested(self) -> None:
        if self.stop_event is not None and self.stop_event.is_set():
            if self.action_safety_filter is not None:
                self._request_latched_stop("operator_stop")
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
