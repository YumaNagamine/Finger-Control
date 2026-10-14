"""Shared encoder-playback setup used by the CLI and GUI frontends."""

from __future__ import annotations

from pathlib import Path
from typing import Any, Callable, Mapping

from controller.csv_player.encoder_calibration import EncoderExcursionCalibration
from controller.csv_player.encoder_runner import EncoderPositionTrajectoryRunner
from controller.csv_player.force_estimator import CalibratedForceEstimator
from controller.csv_player.recorder import ControlTraceWriter
from controller.csv_player.runner import PlaybackConfig
from controller.csv_player.trajectory import ExcursionTrajectory
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
        stop_event: Any | None = None,
    ) -> None:
        self.trajectory = trajectory
        self.calibration = calibration
        self.force_calibration_dir = Path(force_calibration_dir)
        self.zero_offset_path = Path(zero_offset_path)
        self.config = config
        self.record_force = bool(record_force)
        self.on_sample = on_sample
        self.stop_event = stop_event
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
            stop_event=self.stop_event,
        )
        runner.run()

    def close(self) -> None:
        if self._closed:
            return
        self._closed = True
        self.recorder.close()
