"""Shared encoder-playback setup used by the CLI and GUI frontends."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping, Sequence

from controller.csv_player.encoder_calibration import EncoderExcursionCalibration
from controller.csv_player.encoder_runner import EncoderPositionTrajectoryRunner
from controller.csv_player.force_estimator import CalibratedForceEstimator
from controller.csv_player.recorder import ControlTraceWriter
from controller.csv_player.runner import PlaybackConfig
from controller.csv_player.trajectory import ExcursionTrajectory
from controller.rl.raw_event_logger import AsyncRawEventLogger, RawLoggerStats
from controller.rl.action_safety_filter import ActionSafetyFilter
from controller.servo.hardware_session import ServoHardwareSession
from controller.servo_mapping import (
    ENCODER_SENSOR_IDS_BY_SERVO_ID,
    SERVO_IDS,
    SERVO_IDS_BY_TENDON,
    TENDONS,
)


def build_encoder_playback_config(
    calibration: EncoderExcursionCalibration,
    *,
    control_hz: float = 50.0,
    telemetry_stale_s: float = 0.2,
    telemetry_wait_s: float = 3.0,
    display_interval_s: float = 0.25,
    return_to_start: bool = True,
    return_timeout_s: float = 5.0,
    return_stable_frames: int = 3,
    hard_force_limits_n_by_tendon: Mapping[str, float] | None = None,
    playback_mode: str = "interpolated",
    decision_hz: float = 5.0,
    control_period_tolerance_fraction: float = 0.25,
) -> PlaybackConfig:
    """Build the common trajectory settings, including calibrated return tolerance."""

    return PlaybackConfig(
        control_hz=control_hz,
        telemetry_stale_s=telemetry_stale_s,
        telemetry_wait_s=telemetry_wait_s,
        display_interval_s=display_interval_s,
        return_to_start=return_to_start,
        return_tolerance=calibration.return_tolerance_deg,
        return_timeout_s=return_timeout_s,
        return_stable_frames=return_stable_frames,
        hard_force_limits_n_by_tendon=hard_force_limits_n_by_tendon,
        playback_mode=playback_mode,
        decision_hz=decision_hz,
        control_period_tolerance_fraction=control_period_tolerance_fraction,
    )


class EncoderPlaybackService:
    """Own shared force/trace setup and trajectory execution for each frontend."""

    def __init__(
        self,
        *,
        trajectory: ExcursionTrajectory,
        calibration: EncoderExcursionCalibration,
        trace_path: Path,
        run_id: str,
        force_calibration_dir: Path,
        zero_offset_path: Path,
        config: PlaybackConfig,
        record_force: bool = True,
        on_sample: Callable[[dict[str, Any]], None] | None = None,
        on_playback_clock: Callable[[float, float, bool], None] | None = None,
        stop_event: Any | None = None,
        force_quit_event: Any | None = None,
        raw_event_logger: AsyncRawEventLogger | None = None,
        raw_event_path: Path | None = None,
        action_safety_filter: ActionSafetyFilter | None = None,
        start_state_validator: (
            Callable[[Any, Mapping[int, float]], Mapping[str, Any]] | None
        ) = None,
        feedback_action_provider: (
            Callable[[float, tuple[float, ...]], Sequence[float]] | None
        ) = None,
        joint_safety_state_provider: (
            Callable[[], tuple[Mapping[str, float | None], float | None] | None]
            | None
        ) = None,
    ) -> None:
        self.trajectory = trajectory
        self.calibration = calibration
        self.force_calibration_dir = Path(force_calibration_dir)
        self.zero_offset_path = Path(zero_offset_path)
        self.config = config
        self.record_force = bool(record_force)
        self.on_sample = on_sample
        self.on_playback_clock = on_playback_clock
        self.stop_event = stop_event
        self.force_quit_event = force_quit_event
        self.action_safety_filter = action_safety_filter
        self.start_state_validator = start_state_validator
        self.feedback_action_provider = feedback_action_provider
        self.joint_safety_state_provider = joint_safety_state_provider
        if raw_event_logger is not None and raw_event_path is not None:
            raise ValueError(
                "Provide raw_event_logger or raw_event_path, not both"
            )
        self.raw_event_logger = raw_event_logger
        self._owns_raw_event_logger = False
        if self.raw_event_logger is None and raw_event_path is not None:
            self.raw_event_logger = AsyncRawEventLogger(
                raw_event_path,
                run_id=run_id,
            )
            self._owns_raw_event_logger = True
        self.raw_logger_stats: RawLoggerStats | None = None
        self.recorder = ControlTraceWriter(
            trace_path,
            SERVO_IDS_BY_TENDON,
            position_label="encoder_deg",
            run_id=run_id,
        )
        self._closed = False
        self._started = False

    def run(self, hardware: ServoHardwareSession) -> None:
        if self._closed:
            raise RuntimeError("Encoder playback service is closed")
        if self._started:
            raise RuntimeError("Encoder playback service can only run once")
        self._started = True

        force_estimator = None
        if self.record_force:
            force_estimator = CalibratedForceEstimator(
                calibration_dir=self.force_calibration_dir,
                zero_offset_path=self.zero_offset_path,
                servo_ids=SERVO_IDS,
                tendon_by_servo_id=dict(zip(SERVO_IDS_BY_TENDON, TENDONS)),
                telemetry_stale_after_s=self.config.telemetry_stale_s,
            )
        runner = EncoderPositionTrajectoryRunner(
            trajectory=self.trajectory,
            servo_ids_by_tendon=SERVO_IDS_BY_TENDON,
            encoder_sensor_ids_by_servo_id=ENCODER_SENSOR_IDS_BY_SERVO_ID,
            calibration=self.calibration,
            hardware=hardware,
            config=self.config,
            recorder=self.recorder,
            force_estimator=force_estimator,
            on_sample=self.on_sample,
            on_playback_clock=self.on_playback_clock,
            stop_event=self.stop_event,
            force_quit_event=self.force_quit_event,
            raw_event_logger=self.raw_event_logger,
            action_safety_filter=self.action_safety_filter,
            start_state_validator=self.start_state_validator,
            feedback_action_provider=self.feedback_action_provider,
            joint_safety_state_provider=self.joint_safety_state_provider,
        )
        runner.run()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.recorder.close()
        if self._owns_raw_event_logger and self.raw_event_logger is not None:
            self.raw_logger_stats = self.raw_event_logger.close()
