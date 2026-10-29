"""Tkinter GUI for CSV trajectory playback with DLC and force monitoring.

The GUI owns the presentation layer only.  Servo control, camera inference,
and serial readers run on worker threads so that plotting never blocks the
control loop.
"""

from __future__ import annotations

import bisect
import csv
import json
import math
import os
import queue
import shutil
import statistics
import threading
import time
import tkinter as tk
import traceback
from collections import deque
from contextlib import ExitStack
from dataclasses import dataclass
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable, Mapping, Sequence

import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

try:
    from PIL import Image, ImageTk
except ImportError:  # pragma: no cover - Pillow is normally installed with matplotlib
    Image = None
    ImageTk = None

from controller.csv_player.encoder_calibration import (
    EncoderExcursionCalibration,
    encoder_conversion_is_p2p_approved,
    load_encoder_excursion_calibration,
)
from controller.csv_player.force_estimator import CalibratedForceEstimator
from controller.csv_player.gui_support import LatestEventMailbox
from controller.csv_player.playback_service import (
    EncoderPlaybackService,
    build_encoder_playback_config,
)
from controller.csv_player.runner import (
    ForceTelemetryUnavailable,
    HardForceLimitError,
    PlaybackStopped,
)
from controller.csv_player.trajectory import (
    ExcursionTrajectory,
    validate_point_to_point_trajectory,
)
from controller.feedback.joint_feedback import JointFeedbackController
from controller.feedback.moment_arm_runtime import MomentArmRuntime
from controller.rl.action_schedule import (
    INTERPOLATED_PLAYBACK_MODE,
    RL_FF_ZOH_PLAYBACK_MODE,
)
from controller.rl.action_safety_filter import (
    ActionSafetyConfig,
    ActionSafetyContextError,
    ActionSafetyError,
    ActionSafetyFilter,
)
from controller.rl.raw_event_logger import AsyncRawEventLogger, RawLoggerStats
from controller.servo.hardware_session import (
    API_ROOT,
    HardwareFeedback,
    ServoHardwareSession,
)
from controller.servo.remembered_position import RememberedPositionController
from controller.servo.torque_calibration.storage import (
    load_encoder_calibration_positions,
)
from controller.servo_mapping import (
    ENCODER_SENSOR_IDS_BY_SERVO_ID,
    SERVO_IDS,
    SERVO_IDS_BY_TENDON,
    SERVO_NAMES_BY_ID,
    TENDONS,
)
from observation.camera.camera_utils import (
    apply_camera_settings,
    resolve_backend,
    setup_undistortion_from_config,
    undistort_frame,
)
from observation.camera.camera_param_resolver import resolve_param_path
from observation.vision.realtime_joint_angle_source import (
    JOINTS,
    RealtimeJointAngleSource,
)
from utils.path_utils import resolve_path
from utils.config_loader import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
CONTROLLER_ROOT = SRC_ROOT / "controller"
DEFAULT_CSV_DIRECTORY = (
    PROJECT_ROOT
    / "logs"
    / "dual_camera_motionBenchmark"
    / "excursion_predictions"
    / "csv_player"
)
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "logs" / "csv_player_gui"
DEFAULT_ENCODER_CALIBRATION_PATH = CONTROLLER_ROOT / "excursion_encoder_calibration.json"
DEFAULT_FORCE_CONFIG_PATH = CONTROLLER_ROOT / "config_force_control.json"
DEFAULT_FEEDBACK_CONFIG_PATH = (
    CONTROLLER_ROOT / "config_dlc_excursion_feedback.json"
)
DEFAULT_DLC_CONFIG_PATH = SRC_ROOT / "observation" / "vision" / "config_deeplabcut_angle_control.json"
DEFAULT_CAMERA_CONFIG_PATH = resolve_param_path("camera_config_dlc.json")
DEFAULT_FORCE_CALIBRATION_DIR = API_ROOT / "calibration" / "current"
DEFAULT_ZERO_OFFSET_PATH = DEFAULT_FORCE_CALIBRATION_DIR / "as5048a_unloaded_zero_offsets.json"
DEFAULT_SERVO_PORT = "COM3"
DEFAULT_ENCODER_PORT = "COM5"
DEFAULT_CONTROL_HZ = 50.0
DEFAULT_VISION_HZ = 5.0
DEFAULT_DECISION_HZ = 5.0
POINT_TO_POINT_SOURCE = "point_to_point"
CSV_TRAJECTORY_SOURCE = "csv"
PTP_Q0_FRAME_COUNT = 5
PTP_Q0_MAX_SPREAD_DEG = 2.0
PTP_Q0_ACQUIRE_TIMEOUT_S = 3.0
PLOT_HISTORY_SIZE = 1200
MAX_EVENTS_PER_POLL = 100
MAX_RESULT_ROWS = 100
CRITICAL_EVENT_KINDS = frozenset({
    "worker_error",
    "hard_limit",
    "result",
    "manual_command_done",
    "manual_command_error",
    "manual_safety_stop",
    "position_action_done",
    "position_action_error",
})
LATEST_EVENT_KINDS = frozenset({"force", "angle"})
PLOT_UPDATE_INTERVAL_S = 0.1
PLOT_AXIS_REFRESH_INTERVAL_S = 1.0
MANUAL_SPEED_PRESETS = (-300, -100, 0, 100, 300)
MANUAL_MONITOR_INTERVAL_S = 0.05
MANUAL_DISPLAY_INTERVAL_S = 0.1
MANUAL_FEEDBACK_WAIT_TIMEOUT_S = 3.0
AS5048A_COUNTS_PER_REVOLUTION = 1 << 14


def _load_hard_limits(path: Path) -> dict[str, float]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    default = payload.get("default", {}) if isinstance(payload, dict) else {}
    raw = default.get("hard_force_limits_n_by_tendon", {})
    if not isinstance(raw, dict):
        raise ValueError("default.hard_force_limits_n_by_tendon must be an object")
    limits: dict[str, float] = {}
    for tendon in TENDONS:
        key = "LU" if tendon == "LUM" and "LUM" not in raw else tendon
        value = float(raw.get(key, 0.0))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"A positive hard force limit is required for {tendon}")
        limits[tendon] = value
    return limits


def _resolve_config_path(raw_path: str, base_path: Path) -> Path:
    return resolve_path(raw_path, base_path)


class DLCFeedbackUnavailable(RuntimeError):
    """Raised before command submission when DLC feedback cannot be trusted."""


@dataclass(frozen=True)
class JointMeasurementSnapshot:
    angles_rad: tuple[float, float, float]
    captured_at: float
    inference_finished_at: float
    valid: bool
    held_last: bool
    reason: str | None


class LatestJointState:
    """Thread-safe handoff from the DLC worker to the control worker."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._snapshot: JointMeasurementSnapshot | None = None

    def update(self, measurement: Any) -> None:
        angles = tuple(float(value) for value in measurement.flexion_angles_rad)
        if len(angles) != len(JOINTS):
            raise ValueError("DLC measurement must contain DIP, PIP, and MCP")
        snapshot = JointMeasurementSnapshot(
            angles_rad=angles,
            captured_at=float(measurement.captured_at),
            inference_finished_at=float(measurement.inference_finished_at),
            valid=bool(measurement.valid) and all(
                math.isfinite(value) for value in angles
            ),
            held_last=bool(getattr(measurement, "held_last", False)),
            reason=None
            if getattr(measurement, "reason", None) is None
            else str(measurement.reason),
        )
        with self._lock:
            self._snapshot = snapshot

    def invalidate(self, reason: str) -> None:
        now = time.monotonic()
        with self._lock:
            self._snapshot = JointMeasurementSnapshot(
                angles_rad=(math.nan, math.nan, math.nan),
                captured_at=now,
                inference_finished_at=now,
                valid=False,
                held_last=False,
                reason=str(reason),
            )

    def snapshot(self) -> JointMeasurementSnapshot | None:
        with self._lock:
            return self._snapshot

    def fresh_snapshot(
        self,
        max_age_s: float,
        *,
        now: float | None = None,
    ) -> JointMeasurementSnapshot | None:
        snapshot = self.snapshot()
        current = time.monotonic() if now is None else float(now)
        if (
            snapshot is None
            or not snapshot.valid
            or snapshot.held_last
            or snapshot.inference_finished_at > current
            or current - snapshot.captured_at > max_age_s
        ):
            return None
        return snapshot

    def safety_state(
        self,
        max_age_s: float,
    ) -> tuple[Mapping[str, float | None], float | None] | None:
        now = time.monotonic()
        snapshot = self.fresh_snapshot(max_age_s, now=now)
        if snapshot is None:
            return None
        return (
            dict(zip(JOINTS, snapshot.angles_rad)),
            max(0.0, now - snapshot.captured_at),
        )


@dataclass(frozen=True)
class IntegratedControlRuntime:
    feedback_enabled: bool
    safety_filter_enabled: bool
    require_rl_ff_zoh: bool
    max_joint_age_s: float | None
    feedback_controller: JointFeedbackController | None
    action_safety_filter: ActionSafetyFilter | None
    automatic_return_after_safety_stop: bool
    config_summary: Mapping[str, Any]

    @property
    def active(self) -> bool:
        return self.feedback_enabled or self.safety_filter_enabled


@dataclass(frozen=True)
class PointToPointRequest:
    q_goal_rad: tuple[float, float, float]
    duration_s: float
    feedback_enabled: bool
    moment_arm: MomentArmRuntime


def _validate_point_to_point_request(
    q_goal_rad: Sequence[float],
    duration_s: float,
    *,
    playback_mode: str,
    decision_hz: float,
    safety_filter_enabled: bool,
    encoder_calibration: EncoderExcursionCalibration,
    config_path: Path = DEFAULT_FEEDBACK_CONFIG_PATH,
) -> MomentArmRuntime:
    goal = tuple(float(value) for value in q_goal_rad)
    if len(goal) != len(JOINTS) or not all(math.isfinite(value) for value in goal):
        raise ValueError("q_goal must contain finite DIP, PIP, and MCP angles")
    if not math.isfinite(duration_s) or duration_s <= 0.0:
        raise ValueError("運動時間Tは0より大きい有限値にしてください")
    if playback_mode != RL_FF_ZOH_PLAYBACK_MODE:
        raise ValueError("Point-to-Point requires RL FF ZOH playback")
    if not safety_filter_enabled:
        raise ValueError("Point-to-Point requires the Safety Filter")
    if not encoder_conversion_is_p2p_approved(encoder_calibration):
        raise ValueError(
            "Point-to-Point requires measured encoder calibration or an explicitly "
            "approved spool-radius geometry conversion"
        )
    if not math.isclose(
        duration_s * decision_hz,
        round(duration_s * decision_hz),
        rel_tol=0.0,
        abs_tol=1e-8,
    ):
        raise ValueError(
            "運動時間TはDecision周期に合わせてください（T × Decision Hz が整数）"
        )

    config = load_config(
        config_path,
        required_keys=("paths", "gui_integration"),
    )
    gui_config = config["gui_integration"]
    if not isinstance(gui_config, Mapping):
        raise ValueError("gui_integration must be an object")
    raw_limits = gui_config.get("joint_angle_limits_deg")
    if not isinstance(raw_limits, Mapping):
        raise ValueError(
            "Point-to-Point requires explicitly approved joint_angle_limits_deg"
        )
    moment_arm_path = _resolve_config_path(
        str(config["paths"]["moment_arm_directory"]), PROJECT_ROOT
    )
    moment_arm = MomentArmRuntime.from_directory(moment_arm_path)
    ranges = moment_arm.evaluation_ranges_rad
    if any(joint not in ranges for joint in JOINTS):
        raise ValueError("Moment-arm evaluation ranges are missing")
    for joint, angle_rad in zip(JOINTS, goal):
        raw_pair = raw_limits.get(joint)
        if not isinstance(raw_pair, list) or len(raw_pair) != 2:
            raise ValueError(f"Approved angle limits for {joint} are required")
        low_deg, high_deg = (float(value) for value in raw_pair)
        if not math.isfinite(low_deg) or not math.isfinite(high_deg) or low_deg > high_deg:
            raise ValueError(f"Approved angle limits for {joint} are invalid")
        low, high = math.radians(low_deg), math.radians(high_deg)
        if angle_rad < low or angle_rad > high:
            raise ValueError(f"q_goal exceeds approved {joint} limits")
    return moment_arm


def _connect_moment_arm_extrapolation_logger(
    moment_arm: MomentArmRuntime,
    raw_event_logger: AsyncRawEventLogger,
) -> None:
    """Route out-of-fit evaluations into the current Session JSONL log."""

    def record(payload: dict[str, object]) -> None:
        raw_event_logger.log(
            "moment_arm_constant_extrapolation",
            event_monotonic_s=time.monotonic(),
            payload=payload,
        )

    moment_arm.set_extrapolation_event_callback(record)


def _collect_stable_initial_dlc(
    latest_joint_state: LatestJointState,
    *,
    max_age_s: float,
    stop_event: threading.Event,
    frame_count: int = PTP_Q0_FRAME_COUNT,
    max_spread_deg: float = PTP_Q0_MAX_SPREAD_DEG,
    timeout_s: float = PTP_Q0_ACQUIRE_TIMEOUT_S,
) -> tuple[JointMeasurementSnapshot, tuple[JointMeasurementSnapshot, ...], tuple[float, ...], tuple[float, ...]]:
    """Collect distinct fresh DLC samples and return the latest stable sample."""
    deadline = time.monotonic() + float(timeout_s)
    snapshots: list[JointMeasurementSnapshot] = []
    last_captured_at = -math.inf
    while len(snapshots) < frame_count:
        if stop_event.is_set():
            raise PlaybackStopped("Playback stopped while acquiring initial DLC posture")
        snapshot = latest_joint_state.fresh_snapshot(max_age_s)
        if snapshot is not None and snapshot.captured_at > last_captured_at:
            snapshots.append(snapshot)
            last_captured_at = snapshot.captured_at
        if len(snapshots) >= frame_count:
            break
        if time.monotonic() >= deadline:
            raise DLCFeedbackUnavailable(
                f"Could not acquire {frame_count} distinct fresh DLC frames for P2P q0"
            )
        time.sleep(0.01)

    median_angles = tuple(
        statistics.median(snapshot.angles_rad[index] for snapshot in snapshots)
        for index in range(len(JOINTS))
    )
    spread_deg = tuple(
        math.degrees(
            max(snapshot.angles_rad[index] for snapshot in snapshots)
            - min(snapshot.angles_rad[index] for snapshot in snapshots)
        )
        for index in range(len(JOINTS))
    )
    if any(value > max_spread_deg for value in spread_deg):
        raise DLCFeedbackUnavailable(
            "Initial DLC posture is not stationary: per-joint spread exceeds "
            f"{max_spread_deg:g} deg"
        )
    return snapshots[-1], tuple(snapshots), median_angles, spread_deg


def _validate_point_to_point_encoder_snapshot(
    hardware: ServoHardwareSession,
    *,
    q0_snapshot: JointMeasurementSnapshot,
    max_servo_age_s: float,
    max_encoder_age_s: float,
    max_joint_age_s: float,
) -> dict[str, Any]:
    feedback = hardware.latest_feedback(max_servo_age_s)
    if feedback is None or feedback.servo_age_s > max_servo_age_s:
        raise TimeoutError("Fresh servo feedback is unavailable for P2P origin")
    try:
        servo_positions = hardware.position_by_servo_id(feedback)
        normalized_servo_positions = {
            int(servo_id): float(position)
            for servo_id, position in servo_positions.items()
        }
    except Exception as exc:
        raise ValueError("Servo position feedback is invalid for P2P origin") from exc
    if any(
        servo_id not in normalized_servo_positions
        or not math.isfinite(normalized_servo_positions[servo_id])
        for servo_id in SERVO_IDS_BY_TENDON
    ):
        raise ValueError("Servo position feedback is missing or non-finite for P2P origin")
    positions: dict[str, float] = {}
    raw_counts: dict[str, int] = {}
    received_at: dict[str, float] = {}
    statuses: dict[str, str] = {}
    for tendon, servo_id in zip(TENDONS, SERVO_IDS_BY_TENDON):
        sensor_id = ENCODER_SENSOR_IDS_BY_SERVO_ID[servo_id]
        sample = feedback.encoder_by_sensor.get(sensor_id)
        age_s = feedback.encoder_age_s(sensor_id)
        if not isinstance(sample, Mapping) or age_s is None:
            raise TimeoutError(f"Fresh encoder feedback is unavailable for {tendon}")
        status = str(sample.get("status", "")).strip().upper()
        if status not in {"OK", "MULTITURN_UNCERTAIN"}:
            raise ValueError(f"Encoder status for {tendon} is unsafe: {status or 'MISSING'}")
        try:
            position = float(sample["relative_deg"])
            raw_count = int(sample["raw"])
            sampled_at = float(feedback.encoder_received_at[sensor_id])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Encoder position for {tendon} is invalid") from exc
        if (
            not math.isfinite(position)
            or not math.isfinite(sampled_at)
            or not 0 <= raw_count < AS5048A_COUNTS_PER_REVOLUTION
        ):
            raise ValueError(f"Encoder position for {tendon} is invalid")
        if age_s > max_encoder_age_s:
            raise TimeoutError(f"Encoder feedback for {tendon} is stale")
        if abs(sampled_at - q0_snapshot.captured_at) > max_joint_age_s:
            raise TimeoutError(
                f"DLC q0 and encoder origin for {tendon} do not represent a fresh static state"
            )
        positions[tendon] = position
        raw_counts[tendon] = raw_count
        received_at[tendon] = sampled_at
        statuses[tendon] = status
    return {
        "encoder_feedback_sampled_at": float(feedback.sampled_at),
        "servo_feedback_received_at": float(feedback.servo_received_at),
        "servo_feedback_age_s": float(feedback.servo_age_s),
        "servo_position_by_servo_id": normalized_servo_positions,
        "encoder_relative_deg_by_tendon": positions,
        "encoder_raw_count_by_tendon": raw_counts,
        "encoder_received_at_by_tendon": received_at,
        "encoder_status_by_tendon": statuses,
    }


def _joint_snapshot_audit_payload(
    snapshot: JointMeasurementSnapshot | None,
) -> dict[str, Any] | None:
    if snapshot is None:
        return None
    return {
        "captured_at_monotonic_s": snapshot.captured_at,
        "inference_finished_at_monotonic_s": snapshot.inference_finished_at,
        "angles_rad": {
            joint: _finite_optional_float(angle)
            for joint, angle in zip(JOINTS, snapshot.angles_rad)
        },
        "angles_deg": {
            joint: _finite_optional_float(math.degrees(angle))
            for joint, angle in zip(JOINTS, snapshot.angles_rad)
        },
        "valid": snapshot.valid,
        "held_last": snapshot.held_last,
        "reason": snapshot.reason,
    }


def _finite_optional_float(value: Any) -> float | None:
    try:
        converted = float(value)
    except (TypeError, ValueError, OverflowError):
        return None
    return converted if math.isfinite(converted) else None


def _point_to_point_start_audit_payload(
    hardware: ServoHardwareSession,
    *,
    latest_joint_state: LatestJointState,
    q0_snapshot: JointMeasurementSnapshot,
    encoder_origin_snapshot: Mapping[str, Any],
    feedback: HardwareFeedback,
    runner_encoder_origin_deg_by_servo_id: Mapping[int, float],
    encoder_directions: Sequence[float],
) -> dict[str, Any]:
    """Capture the exact raw snapshots available to the P2P start validator."""

    try:
        runner_servo_positions = hardware.position_by_servo_id(feedback)
    except Exception:
        runner_servo_positions = {}
    previous_servo_positions = encoder_origin_snapshot.get(
        "servo_position_by_servo_id", {}
    )
    previous_encoder_counts = encoder_origin_snapshot.get(
        "encoder_raw_count_by_tendon", {}
    )
    previous_encoder_angles = encoder_origin_snapshot.get(
        "encoder_relative_deg_by_tendon", {}
    )
    previous_encoder_times = encoder_origin_snapshot.get(
        "encoder_received_at_by_tendon", {}
    )
    runner_encoder_samples: dict[str, Any] = {}
    encoder_count_delta: dict[str, int | None] = {}
    encoder_origin_delta_deg: dict[str, float | None] = {}
    encoder_elapsed_s: dict[str, float | None] = {}
    servo_position_delta: dict[str, float | None] = {}
    servo_elapsed_s: dict[str, float | None] = {}
    previous_servo_received_at = _finite_optional_float(
        encoder_origin_snapshot.get("servo_feedback_received_at")
    )
    runner_servo_received_at = _finite_optional_float(feedback.servo_received_at)

    for tendon_index, (tendon, servo_id) in enumerate(
        zip(TENDONS, SERVO_IDS_BY_TENDON)
    ):
        sensor_id = ENCODER_SENSOR_IDS_BY_SERVO_ID[servo_id]
        sample = feedback.encoder_by_sensor.get(sensor_id)
        received_at = feedback.encoder_received_at.get(sensor_id)
        raw_count: int | None = None
        if isinstance(sample, Mapping):
            try:
                raw_count = int(sample.get("raw"))
            except (TypeError, ValueError, OverflowError):
                pass
        current_received_at = _finite_optional_float(received_at)
        current_servo_position = _finite_optional_float(
            runner_servo_positions.get(servo_id)
        )
        previous_servo_position = _finite_optional_float(
            previous_servo_positions.get(
                servo_id,
                previous_servo_positions.get(str(servo_id)),
            )
        ) if isinstance(previous_servo_positions, Mapping) else None
        try:
            previous_count = int(previous_encoder_counts[tendon])
            if not 0 <= previous_count < AS5048A_COUNTS_PER_REVOLUTION:
                previous_count = -1
        except (KeyError, TypeError, ValueError, OverflowError):
            previous_count = -1

        if (
            raw_count is not None
            and 0 <= raw_count < AS5048A_COUNTS_PER_REVOLUTION
            and previous_count >= 0
        ):
            difference = abs(raw_count - previous_count)
            encoder_count_delta[tendon] = min(
                difference,
                AS5048A_COUNTS_PER_REVOLUTION - difference,
            )
        else:
            encoder_count_delta[tendon] = None

        try:
            previous_relative_deg = float(previous_encoder_angles[tendon])
            runner_origin_deg = float(
                runner_encoder_origin_deg_by_servo_id[servo_id]
            )
            if not math.isfinite(previous_relative_deg) or not math.isfinite(runner_origin_deg):
                raise ValueError
            direction = float(encoder_directions[tendon_index])
            expected_origin_deg = (
                direction * (previous_relative_deg % 360.0)
            ) % 360.0
            runner_origin_deg %= 360.0
            angle_delta = abs(runner_origin_deg - expected_origin_deg)
            encoder_origin_delta_deg[tendon] = min(angle_delta, 360.0 - angle_delta)
        except (KeyError, TypeError, ValueError, OverflowError):
            encoder_origin_delta_deg[tendon] = None

        previous_encoder_received_at = _finite_optional_float(
            previous_encoder_times.get(tendon)
            if isinstance(previous_encoder_times, Mapping)
            else None
        )
        encoder_elapsed_s[tendon] = (
            None
            if current_received_at is None or previous_encoder_received_at is None
            else current_received_at - previous_encoder_received_at
        )
        servo_position_delta[tendon] = (
            None
            if current_servo_position is None or previous_servo_position is None
            else abs(current_servo_position - previous_servo_position)
        )
        servo_elapsed_s[tendon] = (
            None
            if runner_servo_received_at is None or previous_servo_received_at is None
            else runner_servo_received_at - previous_servo_received_at
        )
        runner_encoder_samples[tendon] = {
            "sensor_id": sensor_id,
            "received_at_monotonic_s": current_received_at,
            "raw_count": raw_count,
            "relative_deg": (
                _finite_optional_float(sample.get("relative_deg"))
                if isinstance(sample, Mapping)
                else None
            ),
            "status": (
                str(sample.get("status", ""))
                if isinstance(sample, Mapping)
                else None
            ),
            "runner_origin_deg": _finite_optional_float(
                runner_encoder_origin_deg_by_servo_id.get(servo_id)
            ),
        }

    latest_dlc = latest_joint_state.snapshot()
    return {
        "q0_snapshot": _joint_snapshot_audit_payload(q0_snapshot),
        "pre_snapshot": {
            "sampled_at_monotonic_s": encoder_origin_snapshot.get(
                "encoder_feedback_sampled_at"
            ),
            "servo_received_at_monotonic_s": previous_servo_received_at,
            "servo_position_by_servo_id": {
                str(key): value
                for key, value in previous_servo_positions.items()
            } if isinstance(previous_servo_positions, Mapping) else {},
            "encoder_received_at_monotonic_s_by_tendon": dict(
                previous_encoder_times
            ) if isinstance(previous_encoder_times, Mapping) else {},
            "encoder_raw_count_by_tendon": dict(
                previous_encoder_counts
            ) if isinstance(previous_encoder_counts, Mapping) else {},
            "encoder_relative_deg_by_tendon": dict(
                previous_encoder_angles
            ) if isinstance(previous_encoder_angles, Mapping) else {},
        },
        "runner_snapshot": {
            "sampled_at_monotonic_s": feedback.sampled_at,
            "servo_received_at_monotonic_s": runner_servo_received_at,
            "servo_position_by_servo_id": {
                str(key): value for key, value in runner_servo_positions.items()
            },
            "servo_raw_frame": feedback.servo,
            "encoder_by_tendon": runner_encoder_samples,
            "latest_dlc_snapshot_at_audit": _joint_snapshot_audit_payload(
                latest_dlc
            ),
        },
        "time_deltas_s": {
            "servo_by_tendon": servo_elapsed_s,
            "encoder_by_tendon": encoder_elapsed_s,
            "dlc_q0_to_latest_at_audit": (
                None
                if latest_dlc is None
                else latest_dlc.captured_at - q0_snapshot.captured_at
            ),
        },
        "position_deltas": {
            "servo_count_by_tendon": servo_position_delta,
            "encoder_raw_count_circular_by_tendon": encoder_count_delta,
            "encoder_origin_deg_by_tendon": encoder_origin_delta_deg,
            "joint_abs_deg_from_latest_at_audit": (
                None
                if latest_dlc is None
                else {
                    joint: (
                        _finite_optional_float(
                            math.degrees(
                                abs(
                                    latest_dlc.angles_rad[index]
                                    - q0_snapshot.angles_rad[index]
                                )
                            )
                        )
                    )
                    for index, joint in enumerate(JOINTS)
                }
            ),
        },
        "latest_dlc_snapshot_used_for_validation": False,
    }


def _configured_encoder_start_tolerance_counts(
    gui_config: Mapping[str, Any],
) -> int:
    raw_value = gui_config.get("encoder_start_position_tolerance_counts", 1)
    if isinstance(raw_value, bool):
        raise ValueError("encoder_start_position_tolerance_counts must be an integer")
    try:
        numeric_value = float(raw_value)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError(
            "encoder_start_position_tolerance_counts must be an integer"
        ) from exc
    if (
        not math.isfinite(numeric_value)
        or not numeric_value.is_integer()
        or numeric_value < 0
        or numeric_value > AS5048A_COUNTS_PER_REVOLUTION // 2
    ):
        raise ValueError(
            "encoder_start_position_tolerance_counts must be an integer from 0 "
            f"through {AS5048A_COUNTS_PER_REVOLUTION // 2}"
        )
    return int(numeric_value)


def _validate_point_to_point_runner_start_state(
    hardware: ServoHardwareSession,
    *,
    latest_joint_state: LatestJointState,
    q0_snapshot: JointMeasurementSnapshot,
    q0_spread_deg: Sequence[float],
    encoder_origin_snapshot: Mapping[str, Any],
    feedback: HardwareFeedback,
    runner_encoder_origin_deg_by_servo_id: Mapping[int, float],
    encoder_directions: Sequence[float],
    max_joint_age_s: float,
    encoder_start_tolerance_counts: int = 1,
    audit_details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    """Reject a changed or stale start state before the Runner can command motion."""

    if not math.isfinite(max_joint_age_s) or max_joint_age_s <= 0.0:
        raise ValueError("max_joint_age_s must be finite and positive")
    if len(q0_spread_deg) != len(JOINTS):
        raise ValueError("q0_spread_deg must contain DIP, PIP, and MCP")
    if len(encoder_directions) != len(TENDONS):
        raise ValueError("encoder_directions must contain one value per tendon")
    if (
        isinstance(encoder_start_tolerance_counts, bool)
        or not isinstance(encoder_start_tolerance_counts, int)
        or not 0 <= encoder_start_tolerance_counts
        <= AS5048A_COUNTS_PER_REVOLUTION // 2
    ):
        raise ValueError("encoder_start_tolerance_counts is invalid")

    deadline = time.monotonic() + max_joint_age_s
    start_joint_snapshot = None
    while time.monotonic() <= deadline:
        candidate = latest_joint_state.fresh_snapshot(max_joint_age_s)
        if (
            candidate is not None
            and candidate.captured_at > q0_snapshot.captured_at
            and abs(candidate.captured_at - feedback.sampled_at) <= max_joint_age_s
        ):
            start_joint_snapshot = candidate
            break
        time.sleep(0.005)
    if start_joint_snapshot is None:
        raise DLCFeedbackUnavailable(
            "A new fresh DLC frame aligned with Runner start was unavailable"
        )

    if audit_details is not None:
        audit_details["runner_dlc_snapshot_used_for_validation"] = (
            _joint_snapshot_audit_payload(start_joint_snapshot)
        )
        audit_details["latest_dlc_snapshot_used_for_validation"] = True
        audit_details.setdefault("time_deltas_s", {})[
            "dlc_q0_to_runner"
        ] = start_joint_snapshot.captured_at - q0_snapshot.captured_at
        audit_details.setdefault("position_deltas", {})[
            "joint_abs_deg_from_runner_dlc"
        ] = {
            joint: math.degrees(
                abs(start_joint_snapshot.angles_rad[index] - q0_snapshot.angles_rad[index])
            )
            for index, joint in enumerate(JOINTS)
        }

    joint_delta_deg: dict[str, float] = {}
    joint_tolerance_deg: dict[str, float] = {}
    for index, joint in enumerate(JOINTS):
        delta_deg = math.degrees(
            abs(start_joint_snapshot.angles_rad[index] - q0_snapshot.angles_rad[index])
        )
        tolerance_deg = float(q0_spread_deg[index])
        if not math.isfinite(delta_deg) or not math.isfinite(tolerance_deg) or tolerance_deg < 0.0:
            raise ValueError(f"Invalid q0 repeatability data for {joint}")
        joint_delta_deg[joint] = delta_deg
        joint_tolerance_deg[joint] = tolerance_deg
        if audit_details is not None:
            audit_details.setdefault("joint_comparison", {})[joint] = {
                "delta_deg": delta_deg,
                "tolerance_deg_from_q0_spread": tolerance_deg,
            }
        if delta_deg > tolerance_deg + 1e-9:
            raise DLCFeedbackUnavailable(
                f"{joint} posture changed after P2P q0 capture "
                f"({delta_deg:.6g} deg > measured q0 spread {tolerance_deg:.6g} deg)"
            )

    try:
        current_servo_positions = hardware.position_by_servo_id(feedback)
    except Exception as exc:
        raise ValueError("Runner-start Servo positions are invalid") from exc
    previous_servo_positions = encoder_origin_snapshot.get(
        "servo_position_by_servo_id"
    )
    previous_encoder_counts = encoder_origin_snapshot.get(
        "encoder_raw_count_by_tendon"
    )
    previous_encoder_received_at = encoder_origin_snapshot.get(
        "encoder_received_at_by_tendon"
    )
    if not all(
        isinstance(value, Mapping)
        for value in (
            previous_servo_positions,
            previous_encoder_counts,
            previous_encoder_received_at,
        )
    ):
        raise ValueError("P2P Encoder origin snapshot is incomplete")

    encoder_count_delta: dict[str, int] = {}
    encoder_origin_delta_deg: dict[str, float] = {}
    servo_position_delta_counts: dict[str, float] = {}
    encoder_origin_tolerance_deg = (
        encoder_start_tolerance_counts
        * 360.0
        / AS5048A_COUNTS_PER_REVOLUTION
    )
    current_servo_received_at = float(feedback.servo_received_at)
    previous_servo_received_at = float(
        encoder_origin_snapshot["servo_feedback_received_at"]
    )
    if current_servo_received_at <= previous_servo_received_at:
        raise TimeoutError("Runner-start Servo sample is not newer than the P2P snapshot")

    for tendon_index, (tendon, servo_id) in enumerate(
        zip(TENDONS, SERVO_IDS_BY_TENDON)
    ):
        sensor_id = ENCODER_SENSOR_IDS_BY_SERVO_ID[servo_id]
        sample = feedback.encoder_by_sensor.get(sensor_id)
        received_at = feedback.encoder_received_at.get(sensor_id)
        if not isinstance(sample, Mapping) or received_at is None:
            raise TimeoutError(f"Runner-start Encoder sample is unavailable for {tendon}")
        if float(received_at) <= float(previous_encoder_received_at[tendon]):
            raise TimeoutError(f"Runner-start Encoder sample is not new for {tendon}")
        if str(sample.get("status", "")).strip().upper() not in {
            "OK",
            "MULTITURN_UNCERTAIN",
        }:
            raise ValueError(f"Runner-start Encoder status is unsafe for {tendon}")
        try:
            raw_count = int(sample["raw"])
            previous_raw_count = int(previous_encoder_counts[tendon])
            runner_origin_deg = float(
                runner_encoder_origin_deg_by_servo_id[servo_id]
            )
            previous_relative_deg = float(
                encoder_origin_snapshot["encoder_relative_deg_by_tendon"][tendon]
            )
            current_servo_position = float(current_servo_positions[servo_id])
            previous_servo_position = float(
                previous_servo_positions.get(
                    servo_id,
                    previous_servo_positions.get(str(servo_id)),
                )
            )
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Runner-start origin is invalid for {tendon}") from exc
        if (
            not 0 <= raw_count < AS5048A_COUNTS_PER_REVOLUTION
            or not 0 <= previous_raw_count < AS5048A_COUNTS_PER_REVOLUTION
            or not all(
                math.isfinite(value)
                for value in (
                    runner_origin_deg,
                    previous_relative_deg,
                    current_servo_position,
                    previous_servo_position,
                )
            )
        ):
            raise ValueError(f"Runner-start origin is non-finite for {tendon}")

        count_difference = abs(raw_count - previous_raw_count)
        count_difference = min(
            count_difference,
            AS5048A_COUNTS_PER_REVOLUTION - count_difference,
        )
        encoder_count_delta[tendon] = count_difference
        if count_difference > encoder_start_tolerance_counts:
            raise TimeoutError(
                f"Encoder origin changed for {tendon} by {count_difference} counts"
            )

        expected_origin_deg = float(encoder_directions[tendon_index]) * (
            previous_relative_deg % 360.0
        )
        expected_origin_deg %= 360.0
        runner_origin_deg %= 360.0
        angle_difference = abs(runner_origin_deg - expected_origin_deg)
        angle_difference = min(angle_difference, 360.0 - angle_difference)
        encoder_origin_delta_deg[tendon] = angle_difference
        if angle_difference > encoder_origin_tolerance_deg + 1e-9:
            raise TimeoutError(f"Runner Encoder origin changed for {tendon}")

        servo_delta = abs(current_servo_position - previous_servo_position)
        servo_position_delta_counts[tendon] = servo_delta
        if servo_delta > 1.0:
            raise TimeoutError(
                f"Servo position changed for {tendon} by {servo_delta:.6g} counts"
            )

    return {
        "status": "passed",
        "q0_capture_monotonic_s": q0_snapshot.captured_at,
        "runner_dlc_capture_monotonic_s": start_joint_snapshot.captured_at,
        "runner_feedback_sampled_at": feedback.sampled_at,
        "joint_delta_deg": joint_delta_deg,
        "joint_tolerance_deg_from_q0_sample_spread": joint_tolerance_deg,
        "encoder_raw_count_delta": encoder_count_delta,
        "encoder_count_tolerance": encoder_start_tolerance_counts,
        "encoder_origin_delta_deg": encoder_origin_delta_deg,
        "encoder_origin_tolerance_deg": encoder_origin_tolerance_deg,
        "servo_position_delta_counts": servo_position_delta_counts,
        "servo_position_count_tolerance": 1,
    }


def _validate_and_log_point_to_point_start_state(
    hardware: ServoHardwareSession,
    *,
    latest_joint_state: LatestJointState,
    q0_snapshot: JointMeasurementSnapshot,
    q0_spread_deg: Sequence[float],
    encoder_origin_snapshot: Mapping[str, Any],
    feedback: HardwareFeedback,
    runner_encoder_origin_deg_by_servo_id: Mapping[int, float],
    encoder_directions: Sequence[float],
    max_joint_age_s: float,
    raw_event_logger: AsyncRawEventLogger,
    encoder_start_tolerance_counts: int = 1,
) -> Mapping[str, Any]:
    audit = _point_to_point_start_audit_payload(
        hardware,
        latest_joint_state=latest_joint_state,
        q0_snapshot=q0_snapshot,
        encoder_origin_snapshot=encoder_origin_snapshot,
        feedback=feedback,
        runner_encoder_origin_deg_by_servo_id=runner_encoder_origin_deg_by_servo_id,
        encoder_directions=encoder_directions,
    )
    audit["q0_sample_spread_deg"] = dict(zip(JOINTS, q0_spread_deg))
    encoder_origin_tolerance_deg = (
        encoder_start_tolerance_counts
        * 360.0
        / AS5048A_COUNTS_PER_REVOLUTION
    )
    audit["comparison_thresholds"] = {
        "encoder_raw_count": encoder_start_tolerance_counts,
        "encoder_origin_deg": encoder_origin_tolerance_deg,
        "servo_position_count": 1.0,
        "joint_deg_source": "per-joint q0 5-frame max-min spread",
    }
    try:
        result = _validate_point_to_point_runner_start_state(
            hardware,
            latest_joint_state=latest_joint_state,
            q0_snapshot=q0_snapshot,
            q0_spread_deg=q0_spread_deg,
            encoder_origin_snapshot=encoder_origin_snapshot,
            feedback=feedback,
            runner_encoder_origin_deg_by_servo_id=runner_encoder_origin_deg_by_servo_id,
            encoder_directions=encoder_directions,
            max_joint_age_s=max_joint_age_s,
            encoder_start_tolerance_counts=encoder_start_tolerance_counts,
            audit_details=audit,
        )
    except Exception as exc:
        audit["status"] = "rejected"
        audit["reason"] = f"{type(exc).__name__}: {exc}"
        raw_event_logger.log(
            "point_to_point_start_alignment_check",
            event_monotonic_s=feedback.sampled_at,
            payload=audit,
        )
        raise
    audit["status"] = "passed"
    audit["reason"] = None
    audit["validation"] = dict(result)
    raw_event_logger.log(
        "point_to_point_start_alignment_check",
        event_monotonic_s=feedback.sampled_at,
        payload=audit,
    )
    return result


def _ordered_config_values(
    mapping: Mapping[str, Any],
    names: Sequence[str],
    name: str,
    *,
    positive: bool = False,
) -> tuple[float, ...]:
    missing = [item for item in names if item not in mapping]
    if missing:
        raise ValueError(f"{name} is missing: {', '.join(missing)}")
    values = tuple(float(mapping[item]) for item in names)
    if not all(math.isfinite(value) for value in values):
        raise ValueError(f"{name} must contain only finite values")
    if positive and not all(value > 0.0 for value in values):
        raise ValueError(f"{name} values must be positive")
    return values


def _target_angles_rad_at(
    trajectory: ExcursionTrajectory,
    elapsed_s: float,
) -> tuple[float, float, float]:
    if trajectory.target_angles_deg is None:
        raise ValueError("FF+FB requires target joint-angle columns in the CSV")
    source_time = min(
        max(0.0, float(elapsed_s)) * trajectory.time_scale,
        trajectory.elapsed_times_s[-1],
    )
    right = bisect.bisect_right(trajectory.elapsed_times_s, source_time)
    if right <= 0:
        values = trajectory.target_angles_deg[0]
    elif right >= len(trajectory.elapsed_times_s):
        values = trajectory.target_angles_deg[-1]
    else:
        left = right - 1
        span = trajectory.elapsed_times_s[right] - trajectory.elapsed_times_s[left]
        fraction = (
            0.0
            if span <= 0.0
            else (source_time - trajectory.elapsed_times_s[left]) / span
        )
        values = tuple(
            left_value + (right_value - left_value) * fraction
            for left_value, right_value in zip(
                trajectory.target_angles_deg[left],
                trajectory.target_angles_deg[right],
            )
        )
    return tuple(math.radians(float(value)) for value in values)


def _build_integrated_control_runtime(
    trajectory: ExcursionTrajectory,
    hard_force_limits_n_by_tendon: Mapping[str, float],
    *,
    feedback_enabled: bool,
    safety_filter_enabled: bool,
    config_path: Path = DEFAULT_FEEDBACK_CONFIG_PATH,
    moment_arm_override: MomentArmRuntime | None = None,
) -> IntegratedControlRuntime:
    if not feedback_enabled and not safety_filter_enabled:
        return IntegratedControlRuntime(
            feedback_enabled=False,
            safety_filter_enabled=False,
            require_rl_ff_zoh=False,
            max_joint_age_s=None,
            feedback_controller=None,
            action_safety_filter=None,
            automatic_return_after_safety_stop=True,
            config_summary={"active": False},
        )

    config = load_config(
        config_path,
        required_keys=("paths", "feedback", "gui_integration"),
    )
    gui_config = config["gui_integration"]
    if not isinstance(gui_config, Mapping):
        raise ValueError("gui_integration must be an object")
    require_safety = bool(
        gui_config.get("require_safety_filter_for_feedback", True)
    )
    if feedback_enabled and require_safety and not safety_filter_enabled:
        raise ValueError("FF+FB requires Safety Filter to be enabled")
    if bool(gui_config.get("exploration_enabled", False)):
        raise ValueError("GUI integrated control does not permit automatic exploration")
    automatic_return = bool(
        gui_config.get("automatic_return_after_safety_stop", False)
    )
    if automatic_return:
        raise ValueError(
            "automatic_return_after_safety_stop must remain false for GUI control"
        )
    max_joint_age_s = float(gui_config["max_joint_age_s"])
    if not math.isfinite(max_joint_age_s) or max_joint_age_s <= 0.0:
        raise ValueError("gui_integration.max_joint_age_s must be positive")

    feedback_controller = None
    moment_arm_runtime: MomentArmRuntime | None = None
    if feedback_enabled:
        _target_angles_rad_at(trajectory, 0.0)
        paths = config["paths"]
        feedback_config = config["feedback"]
        moment_arm_path = _resolve_config_path(
            str(paths["moment_arm_directory"]),
            PROJECT_ROOT,
        )
        moment_arm = moment_arm_override or MomentArmRuntime.from_directory(
            moment_arm_path
        )
        moment_arm_runtime = moment_arm
        raw_joint_limit = feedback_config.get("max_joint_correction_deg")
        max_joint_correction = (
            None
            if raw_joint_limit is None
            else tuple(
                math.radians(value)
                for value in _ordered_config_values(
                    raw_joint_limit,
                    JOINTS,
                    "feedback.max_joint_correction_deg",
                )
            )
        )
        feedback_controller = JointFeedbackController(
            moment_arm,
            kp=_ordered_config_values(
                feedback_config["kp"],
                JOINTS,
                "feedback.kp",
            ),
            max_joint_correction_rad=max_joint_correction,
            max_excursion_correction_mm=_ordered_config_values(
                feedback_config["max_excursion_correction_mm"],
                TENDONS,
                "feedback.max_excursion_correction_mm",
                positive=True,
            ),
        )

    action_safety_filter = None
    if safety_filter_enabled:
        margin = _ordered_config_values(
            gui_config["action_envelope_margin_mm"],
            TENDONS,
            "gui_integration.action_envelope_margin_mm",
            positive=True,
        )
        max_delta = _ordered_config_values(
            gui_config["max_action_delta_mm"],
            TENDONS,
            "gui_integration.max_action_delta_mm",
            positive=True,
        )
        origin = trajectory.initial_excursion_mm
        nominal_actions = tuple(
            tuple(float(value - initial) for value, initial in zip(row, origin))
            for row in trajectory.excursions_mm
        )
        lower = {
            tendon: min(row[index] for row in nominal_actions) - margin[index]
            for index, tendon in enumerate(TENDONS)
        }
        upper = {
            tendon: max(row[index] for row in nominal_actions) + margin[index]
            for index, tendon in enumerate(TENDONS)
        }
        raw_joint_limits = gui_config.get("joint_angle_limits_deg")
        if not isinstance(raw_joint_limits, Mapping):
            raise ValueError(
                "Integrated D0/D1 requires explicitly approved "
                "gui_integration.joint_angle_limits_deg; numeric limits are not inferred"
            )
        joint_limits = {}
        for joint in JOINTS:
            raw_pair = raw_joint_limits.get(joint)
            if not isinstance(raw_pair, list) or len(raw_pair) != 2:
                raise ValueError(
                    f"gui_integration.joint_angle_limits_deg.{joint} must be [min, max]"
                )
            lower_deg, upper_deg = (float(value) for value in raw_pair)
            if (
                not math.isfinite(lower_deg)
                or not math.isfinite(upper_deg)
                or lower_deg >= upper_deg
            ):
                raise ValueError(
                    f"gui_integration.joint_angle_limits_deg.{joint} must be finite [min, max]"
                )
            joint_limits[joint] = (
                math.radians(lower_deg),
                math.radians(upper_deg),
            )
        action_safety_filter = ActionSafetyFilter(
            ActionSafetyConfig(
                min_action_mm_by_tendon=lower,
                max_action_mm_by_tendon=upper,
                max_action_delta_mm_by_tendon=dict(zip(TENDONS, max_delta)),
                hard_force_limits_n_by_tendon=hard_force_limits_n_by_tendon,
                joint_angle_limits_rad=joint_limits,
                max_servo_age_s=float(gui_config["max_servo_age_s"]),
                max_encoder_age_s=float(gui_config["max_encoder_age_s"]),
                max_joint_age_s=max_joint_age_s,
                max_deadline_lateness_s=float(
                    gui_config["max_deadline_lateness_s"]
                ),
                require_force=True,
                reject_on_action_projection=bool(
                    gui_config.get("reject_on_action_projection", True)
                ),
            )
        )

    return IntegratedControlRuntime(
        feedback_enabled=bool(feedback_enabled),
        safety_filter_enabled=bool(safety_filter_enabled),
        require_rl_ff_zoh=bool(gui_config.get("require_rl_ff_zoh", True)),
        max_joint_age_s=max_joint_age_s,
        feedback_controller=feedback_controller,
        action_safety_filter=action_safety_filter,
        automatic_return_after_safety_stop=automatic_return,
        config_summary={
            "active": True,
            "config_path": str(Path(config_path).resolve()),
            "feedback_enabled": bool(feedback_enabled),
            "safety_filter_enabled": bool(safety_filter_enabled),
            "exploration_enabled": False,
            "require_rl_ff_zoh": bool(
                gui_config.get("require_rl_ff_zoh", True)
            ),
            "max_joint_age_s": max_joint_age_s,
            "automatic_return_after_safety_stop": automatic_return,
            "moment_arm_model": (
                None
                if moment_arm_runtime is None
                else moment_arm_runtime.model_info
            ),
        },
    )


def _make_feedback_action_provider(
    runtime: IntegratedControlRuntime,
    trajectory: ExcursionTrajectory,
    latest_joint_state: LatestJointState,
    force_quit_event: threading.Event,
    raw_event_logger: AsyncRawEventLogger | None,
    decision_hz: float | None = None,
) -> Callable[[float, tuple[float, ...]], Sequence[float]] | None:
    controller = runtime.feedback_controller
    if controller is None:
        return None
    if runtime.max_joint_age_s is None:
        raise ValueError("FF+FB requires max_joint_age_s")

    def provide(
        elapsed_s: float,
        _nominal_action_mm: tuple[float, ...],
    ) -> Sequence[float]:
        snapshot = latest_joint_state.fresh_snapshot(runtime.max_joint_age_s)
        if snapshot is None:
            force_quit_event.set()
            if raw_event_logger is not None:
                raw_event_logger.log(
                    "feedback_unavailable",
                    payload={
                        "elapsed_s": float(elapsed_s),
                        "reason": "missing_invalid_held_or_stale_dlc",
                    },
                )
            raise DLCFeedbackUnavailable(
                "Fresh, valid, non-held DLC feedback is unavailable"
            )
        reference_elapsed_s = float(elapsed_s)
        if decision_hz is not None:
            decision_id = math.floor(
                max(0.0, reference_elapsed_s) * decision_hz + 1e-12
            )
            reference_elapsed_s = min(
                decision_id / decision_hz,
                trajectory.duration_s,
            )
        reference = _target_angles_rad_at(trajectory, reference_elapsed_s)
        result = controller.compute(reference, snapshot.angles_rad)
        if decision_hz is not None and raw_event_logger is not None:
            raw_event_logger.log(
                "feedback_detail",
                payload={
                    "elapsed_s": float(elapsed_s),
                    "reference_elapsed_s": float(reference_elapsed_s),
                    "reference_rad": dict(zip(JOINTS, reference)),
                    "measured_rad": dict(zip(JOINTS, snapshot.angles_rad)),
                    "measurement_captured_monotonic_s": snapshot.captured_at,
                    "measurement_age_s": max(
                        0.0, time.monotonic() - snapshot.captured_at
                    ),
                    "error_rad": dict(zip(JOINTS, result.error_rad)),
                    "joint_correction_rad": dict(
                        zip(JOINTS, result.joint_correction_rad)
                    ),
                    "excursion_correction_mm": dict(
                        zip(TENDONS, result.excursion_correction_mm)
                    ),
                },
            )
        return result.excursion_correction_mm

    return provide


class VisionWorker:
    """Read and process camera frames at a bounded rate."""

    def __init__(
        self,
        source: RealtimeJointAngleSource,
        emit: Callable[[dict[str, Any]], None],
        stop_event: threading.Event,
        output_csv: Path,
        playback_started_at: list[float | None],
        playback_timeline: list[tuple[float, ...] | None],
        playback_clock: deque[tuple[float, float, bool]],
        playback_clock_lock: Any,
        latest_joint_state: LatestJointState,
        raw_event_logger: AsyncRawEventLogger | None = None,
        run_id: str = "",
        interval_s: float = 1.0 / DEFAULT_VISION_HZ,
    ) -> None:
        self.source = source
        self.emit = emit
        self.stop_event = stop_event
        self.output_csv = output_csv
        self.playback_started_at = playback_started_at
        self.playback_timeline = playback_timeline
        self.playback_clock = playback_clock
        self.playback_clock_lock = playback_clock_lock
        self.latest_joint_state = latest_joint_state
        self.raw_event_logger = raw_event_logger
        self.run_id = str(run_id)
        self.interval_s = max(0.02, float(interval_s))

    def _emit(self, event: dict[str, Any]) -> None:
        self.emit(event)

    def _playback_source_time_at(self, captured_at: float) -> float | None:
        with self.playback_clock_lock:
            snapshots = tuple(self.playback_clock)
        if not snapshots or captured_at < snapshots[0][0]:
            return None
        if captured_at > snapshots[-1][0]:
            return None
        monotonic_times = tuple(item[0] for item in snapshots)
        right = bisect.bisect_left(monotonic_times, captured_at)
        if right < len(snapshots) and monotonic_times[right] == captured_at:
            return snapshots[right][1]
        if right <= 0 or right >= len(snapshots):
            return None
        left = right - 1
        wall_span_s = monotonic_times[right] - monotonic_times[left]
        max_mapping_gap_s = 2.5 / DEFAULT_CONTROL_HZ
        if wall_span_s <= 0.0 or wall_span_s > max_mapping_gap_s:
            return None
        fraction = (captured_at - monotonic_times[left]) / wall_span_s
        return snapshots[left][1] + fraction * (
            snapshots[right][1] - snapshots[left][1]
        )

    def _write_aligned_angle_rows(
        self,
        writer: Any,
        timeline: tuple[float, ...],
        next_index: int,
        previous_sample: dict[str, Any] | None,
        current_sample: dict[str, Any],
        through_s: float,
    ) -> int:
        while next_index < len(timeline) and timeline[next_index] <= through_s:
            timestamp_s = timeline[next_index]
            if previous_sample is None:
                selected = current_sample
            elif timestamp_s <= previous_sample["source_elapsed_s"]:
                selected = previous_sample
            elif (
                timestamp_s - previous_sample["source_elapsed_s"]
                <= current_sample["source_elapsed_s"] - timestamp_s
            ):
                selected = previous_sample
            else:
                selected = current_sample
            measurement = selected["measurement"]
            writer.writerow(
                [
                    self.run_id,
                    measurement.frame_index,
                    str(timestamp_s),
                    f"{measurement.captured_at:.6f}",
                    f"{measurement.inference_finished_at:.6f}",
                    f"{measurement.inference_ms:.6f}",
                    f"{measurement.min_likelihood:.6f}",
                    int(measurement.valid),
                    int(measurement.held_last),
                    measurement.reason or "",
                    *[f"{value:.6f}" for value in selected["angles_deg"]],
                ]
            )
            next_index += 1
        return next_index

    def run(self) -> None:
        frame_index = 0
        try:
            with ExitStack() as stack:
                handle = None
                writer = None
                playback_started = False
                monitoring_started_at = time.monotonic()
                timeline = self.playback_timeline[0]
                next_angle_index = 0
                previous_angle_sample: dict[str, Any] | None = None
                latest_angle_sample: dict[str, Any] | None = None
                latest_raw_measurement: Any | None = None
                latest_raw_angles_deg: tuple[float, ...] | None = None
                while not self.stop_event.is_set():
                    started_at = time.monotonic()
                    run_started_at = self.playback_started_at[0]
                    if run_started_at is not None and not playback_started:
                        if self.raw_event_logger is not None:
                            if (
                                latest_raw_measurement is not None
                                and latest_raw_angles_deg is not None
                            ):
                                self.raw_event_logger.log(
                                    "initial_posture",
                                    event_monotonic_s=run_started_at,
                                    payload={
                                        "status": "measured",
                                        "angles_deg": dict(
                                            zip(JOINTS, latest_raw_angles_deg)
                                        ),
                                        "frame_idx": latest_raw_measurement.frame_index,
                                        "capture_monotonic_s": (
                                            latest_raw_measurement.captured_at
                                        ),
                                        "inference_finished_monotonic_s": (
                                            latest_raw_measurement.inference_finished_at
                                        ),
                                        "valid": bool(latest_raw_measurement.valid),
                                        "held_last": bool(
                                            latest_raw_measurement.held_last
                                        ),
                                        "available_before_playback": bool(
                                            latest_raw_measurement.inference_finished_at
                                            <= run_started_at
                                        ),
                                    },
                                )
                            else:
                                self.raw_event_logger.log(
                                    "initial_posture",
                                    event_monotonic_s=run_started_at,
                                    payload={
                                        "status": "missing",
                                        "reason": (
                                            "no completed vision measurement before playback"
                                        ),
                                    },
                                )
                        self.source.reset_temporal_history()
                        playback_started = True
                    measurement = self.source.read(frame_index)
                    self.latest_joint_state.update(measurement)
                    timeline = self.playback_timeline[0]
                    source_elapsed_s_at_capture = (
                        self._playback_source_time_at(measurement.captured_at)
                        if timeline
                        else None
                    )
                    angles_deg = tuple(
                        math.degrees(value)
                        for value in measurement.flexion_angles_rad
                    )
                    if self.raw_event_logger is not None:
                        self.raw_event_logger.log(
                            "vision_observation",
                            event_monotonic_s=measurement.inference_finished_at,
                            payload={
                                "frame_idx": measurement.frame_index,
                                "capture_monotonic_s": measurement.captured_at,
                                "inference_finished_monotonic_s": (
                                    measurement.inference_finished_at
                                ),
                                "inference_ms": measurement.inference_ms,
                                "angles_deg": dict(zip(JOINTS, angles_deg)),
                                "min_likelihood": measurement.min_likelihood,
                                "valid": bool(measurement.valid),
                                "held_last": bool(measurement.held_last),
                                "reason": measurement.reason,
                                "available_to_controller_at_monotonic_s": (
                                    measurement.inference_finished_at
                                ),
                                "source_elapsed_s_at_capture": (
                                    source_elapsed_s_at_capture
                                ),
                                "capture_source_time_mapping": (
                                    "linear_between_control_ticks"
                                    if source_elapsed_s_at_capture is not None
                                    else "unavailable"
                                ),
                            },
                        )
                    latest_raw_measurement = measurement
                    latest_raw_angles_deg = angles_deg
                    if timeline:
                        source_elapsed_s = source_elapsed_s_at_capture
                        if source_elapsed_s is not None:
                            current_angle_sample = {
                                "source_elapsed_s": source_elapsed_s,
                                "measurement": measurement,
                                "angles_deg": angles_deg,
                            }
                            if writer is None:
                                handle = stack.enter_context(
                                    self.output_csv.open("w", encoding="utf-8", newline="")
                                )
                                writer = csv.writer(handle)
                                writer.writerow(
                                    [
                                        "run_id",
                                        "frame_idx",
                                        "elapsed_s",
                                        "captured_at_monotonic",
                                        "inference_finished_at_monotonic",
                                        "inference_ms",
                                        "min_likelihood",
                                        "valid",
                                        "held_last",
                                        "reason",
                                        *[f"{joint}_deg" for joint in JOINTS],
                                    ]
                                )
                            next_angle_index = self._write_aligned_angle_rows(
                                writer,
                                timeline,
                                next_angle_index,
                                previous_angle_sample,
                                current_angle_sample,
                                source_elapsed_s,
                            )
                            handle.flush()
                            previous_angle_sample = current_angle_sample
                            latest_angle_sample = current_angle_sample
                    if (
                        run_started_at is not None
                        and measurement.captured_at >= run_started_at
                    ):
                        elapsed_s = measurement.captured_at - run_started_at
                        phase = "playback"
                    else:
                        elapsed_s = measurement.captured_at - monitoring_started_at
                        phase = "manual"
                    self._emit(
                        {
                            "kind": "angle",
                            "phase": phase,
                            "frame_idx": measurement.frame_index,
                            "captured_at": measurement.captured_at,
                            "elapsed_s": elapsed_s,
                            "angles_deg": angles_deg,
                            "valid": measurement.valid,
                            "held_last": measurement.held_last,
                            "min_likelihood": measurement.min_likelihood,
                            "inference_ms": measurement.inference_ms,
                            "reason": measurement.reason,
                        }
                    )
                    frame_index += 1
                    self.stop_event.wait(
                        max(0.0, self.interval_s - (time.monotonic() - started_at))
                    )
                if (
                    writer is not None
                    and timeline
                    and latest_angle_sample is not None
                ):
                    with self.playback_clock_lock:
                        final_clock = (
                            self.playback_clock[-1]
                            if self.playback_clock
                            else None
                        )
                    if final_clock is not None and final_clock[2]:
                        next_angle_index = self._write_aligned_angle_rows(
                            writer,
                            timeline,
                            next_angle_index,
                            latest_angle_sample,
                            latest_angle_sample,
                            final_clock[1],
                        )
                        handle.flush()
        except Exception as exc:
            self.latest_joint_state.invalidate(
                f"{type(exc).__name__}: {exc}"
            )
            self._emit(
                {
                    "kind": "worker_error",
                    "source": "DLC",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            self.stop_event.set()


class DLCMeasurementWorker:
    """Measure joint angles from the camera without connecting servos."""

    def __init__(
        self,
        source: RealtimeJointAngleSource,
        emit: Callable[[dict[str, Any]], None],
        *,
        interval_s: float = 1.0 / DEFAULT_VISION_HZ,
    ) -> None:
        self.source = source
        self.emit = emit
        self.interval_s = max(0.02, float(interval_s))
        self.stop_event = threading.Event()
        self.finished_event = threading.Event()
        self.error: str | None = None
        self.thread = threading.Thread(
            target=self._run,
            name="csv-player-dlc-measurement",
            daemon=True,
        )

    def start(self) -> None:
        self.thread.start()

    def stop(self, timeout_s: float = 2.0) -> bool:
        self.stop_event.set()
        self.thread.join(timeout=max(0.0, timeout_s))
        return not self.thread.is_alive()

    def _run(self) -> None:
        frame_index = 0
        started_at = time.monotonic()
        try:
            self.source.open()
            self.emit({"kind": "dlc_ready", "worker": self})
            while not self.stop_event.is_set():
                loop_started_at = time.monotonic()
                measurement = self.source.read(frame_index)
                self.emit(
                    {
                        "kind": "angle",
                        "frame_idx": measurement.frame_index,
                        "captured_at": measurement.captured_at,
                        "elapsed_s": measurement.captured_at - started_at,
                        "angles_deg": tuple(
                            math.degrees(value)
                            for value in measurement.flexion_angles_rad
                        ),
                        "valid": measurement.valid,
                        "held_last": measurement.held_last,
                        "min_likelihood": measurement.min_likelihood,
                        "inference_ms": measurement.inference_ms,
                        "reason": measurement.reason,
                    }
                )
                frame_index += 1
                self.stop_event.wait(
                    max(0.0, self.interval_s - (time.monotonic() - loop_started_at))
                )
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.emit({"kind": "dlc_error", "worker": self, "error": self.error})
        finally:
            try:
                self.source.close()
            finally:
                self.finished_event.set()
                self.emit({"kind": "dlc_stopped", "worker": self})


class CameraPreviewWorker:
    """Capture raw camera frames for preview without initializing DLC or servos."""

    def __init__(self, camera_config_path: Path) -> None:
        self.camera_config_path = Path(camera_config_path).expanduser()
        self.stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_frame: Any | None = None
        self._frame_index = -1
        self.error: str | None = None
        self.thread = threading.Thread(
            target=self._run,
            name="csv-player-camera-preview",
            daemon=True,
        )

    def start(self) -> None:
        self.thread.start()

    def stop(self, timeout_s: float = 2.0) -> bool:
        self.stop_event.set()
        self.thread.join(timeout=max(0.0, timeout_s))
        return not self.thread.is_alive()

    @property
    def latest_preview(self) -> tuple[int, bool, Any] | None:
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            return self._frame_index, False, self._latest_frame.copy()

    def _run(self) -> None:
        capture: cv2.VideoCapture | None = None
        try:
            camera_config = load_config(self.camera_config_path)
            camera_index = int(camera_config.get("index", 0))
            backend = resolve_backend(camera_config.get("backend"))
            capture = (
                cv2.VideoCapture(camera_index, backend)
                if backend is not None
                else cv2.VideoCapture(camera_index)
            )
            if not capture.isOpened():
                raise RuntimeError(f"Camera not available: index={camera_index}")
            apply_camera_settings(capture, camera_config)
            calibration = setup_undistortion_from_config(
                camera_config,
                log_prefix="[csv-player-preview]",
            )
            frame_index = 0
            while not self.stop_event.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise RuntimeError("Failed to read a frame from the camera")
                frame = undistort_frame(frame, calibration)
                with self._frame_lock:
                    self._latest_frame = frame.copy()
                    self._frame_index = frame_index
                frame_index += 1
                self.stop_event.wait(0.01)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            if capture is not None:
                capture.release()


class CsvPlayerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("CSV Player | DLC and Force Monitor")
        self.geometry("1500x980")
        self.minsize(1180, 780)
        try:
            self.state("zoomed")
        except tk.TclError:
            pass
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.events: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=400)
        self._critical_events: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=32)
        self._latest_events = LatestEventMailbox()
        self.csv_paths: list[Path] = []
        self.force_history = {
            tendon: deque(maxlen=PLOT_HISTORY_SIZE) for tendon in TENDONS
        }
        self.angle_history = {
            joint: deque(maxlen=PLOT_HISTORY_SIZE) for joint in JOINTS
        }
        self.target_angle_history: dict[str, tuple[tuple[float, float], ...]] = {
            joint: () for joint in JOINTS
        }
        self.force_lines: dict[str, Any] = {}
        self.force_limit_lines: dict[str, Any] = {}
        self.angle_lines: dict[str, Any] = {}
        self.target_angle_lines: dict[str, Any] = {}
        self._runner_thread: threading.Thread | None = None
        self._vision_thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._start_event: threading.Event | None = None
        self._force_quit_event: threading.Event | None = None
        self._resource_lock = threading.Lock()
        self._servo_command_path_lock = threading.Lock()
        self._playback_command_lock_held = False
        self._hardware: ServoHardwareSession | None = None
        self._angle_source: RealtimeJointAngleSource | None = None
        self._camera_preview_worker: CameraPreviewWorker | None = None
        self._standalone_dlc_worker: DLCMeasurementWorker | None = None
        self._session_dir: Path | None = None
        self._closing_requested = False
        self._start_in_progress = False
        self._running = False
        self._connection_ready = False
        self._playback_started = False
        self._position_controller: RememberedPositionController | None = None
        self._position_action_thread: threading.Thread | None = None
        self._position_action_cancel_event: threading.Event | None = None
        self._position_action_pending = False
        self._remembered_position_vars: dict[int, tk.StringVar] = {}
        self._remember_position_buttons: dict[int, ttk.Button] = {}
        self._return_position_buttons: dict[int, ttk.Button] = {}
        self._return_all_positions_button: ttk.Button | None = None
        self._manual_speed_lock = threading.Lock()
        self._manual_speed_commands = {servo_id: 0 for servo_id in SERVO_IDS}
        self._manual_command_pending = False
        self._manual_feedback_ready = False
        self._manual_feedback_ready_event = threading.Event()
        self._manual_window: tk.Toplevel | None = None
        self._manual_row_vars: dict[int, dict[str, tk.StringVar]] = {}
        self._manual_speed_buttons: dict[int, list[tuple[int, ttk.Button]]] = {}
        self._manual_stop_all_button: ttk.Button | None = None
        self._latest_manual_sample: dict[str, Any] | None = None
        self._target_angles_visible = True
        self._input_widgets: list[Any] = []
        self._csv_widgets: list[Any] = []
        self._pending_playback_config: tuple[
            Path | None,
            ExcursionTrajectory | None,
            dict[str, float],
            EncoderExcursionCalibration,
            str,
            float,
            IntegratedControlRuntime | None,
            PointToPointRequest | None,
        ] | None = None
        self._active_duration_s: float | None = None
        self._last_angle_frame = -1
        self._min_likelihood_seen = 1.0
        self._last_plot_update = 0.0
        self._force_plot_dirty = True
        self._angle_plot_dirty = True
        self._force_plot_backgrounds: dict[str, Any] = {}
        self._angle_plot_background = None
        self._force_last_full_draw_at = 0.0
        self._angle_last_full_draw_at = 0.0
        self._last_preview_frame = -1
        self._last_preview_is_overlay: bool | None = None
        self._last_preview_size = (0, 0)
        self._poll_after_id: str | None = None
        self._plot_resize_after_id: str | None = None
        self._photo = None
        self._preview_enabled = False
        self._dlc_enabled = False
        self._selected_csv_path: Path | None = None
        self._selected_csv_valid = False
        self._settings_expanded = False
        self._status_badges: dict[str, tk.Label] = {}
        self._header_status_vars: dict[str, tk.StringVar] = {}
        self._status_badge_values: dict[str, tuple[str, str]] = {}

        self.folder_var = tk.StringVar(value=str(DEFAULT_CSV_DIRECTORY))
        self.selected_var = tk.StringVar(value="CSV: 未選択")
        self.detail_var = tk.StringVar(value="CSVを選択してください")
        self.status_var = tk.StringVar(value="待機中")
        self.phase_var = tk.StringVar(value="未実行")
        self.camera_var = tk.StringVar(value="Camera: --")
        self.dlc_var = tk.StringVar(value="DLC: --")
        self.force_var = tk.StringVar(value="Force: --")
        self.elapsed_var = tk.StringVar(value="経過時間: --")
        self.progress_var = tk.StringVar(value="進捗: --")
        self.servo_port_var = tk.StringVar(value=DEFAULT_SERVO_PORT)
        self.encoder_port_var = tk.StringVar(value=DEFAULT_ENCODER_PORT)
        self.playback_mode_var = tk.StringVar(value=INTERPOLATED_PLAYBACK_MODE)
        self.trajectory_source_var = tk.StringVar(value=CSV_TRAJECTORY_SOURCE)
        self.decision_hz_var = tk.StringVar(value=f"{DEFAULT_DECISION_HZ:g}")
        self.point_to_point_goal_vars = {
            joint: tk.StringVar(value="") for joint in JOINTS
        }
        self.point_to_point_duration_var = tk.StringVar(value="")
        self.feedback_enabled_var = tk.BooleanVar(value=False)
        self.safety_filter_enabled_var = tk.BooleanVar(value=False)
        self.dlc_config_var = tk.StringVar(value=str(DEFAULT_DLC_CONFIG_PATH))
        self.camera_config_var = tk.StringVar(value=str(DEFAULT_CAMERA_CONFIG_PATH))
        self.limit_vars = {
            tendon: tk.StringVar(value="--") for tendon in TENDONS
        }
        for variable in self.limit_vars.values():
            variable.trace_add("write", self._mark_force_plot_dirty)

        try:
            defaults = _load_hard_limits(DEFAULT_FORCE_CONFIG_PATH)
            for tendon, value in defaults.items():
                self.limit_vars[tendon].set(f"{value:g}")
        except Exception as exc:
            self.status_var.set(f"Force設定を読めません: {exc}")

        self._build_widgets()
        self._refresh_files()
        self._schedule_event_poll()

    def _build_widgets(self) -> None:
        self._configure_styles()
        self.configure(bg="#f4f6f8")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(6, weight=1)

        header = ttk.Frame(self, padding=(14, 10, 14, 8))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)
        ttk.Label(
            header,
            text="CSV Player",
            style="AppTitle.TLabel",
        ).grid(row=0, column=0, sticky="w", padx=(0, 18))
        ttk.Label(
            header,
            textvariable=self.selected_var,
            style="HeaderFile.TLabel",
        ).grid(row=0, column=1, sticky="w")
        badges = ttk.Frame(header)
        badges.grid(row=0, column=2, sticky="e")
        for index, (key, label, initial) in enumerate(
            (
                ("servo", "Servo", "未接続"),
                ("encoder", "Encoder", "未接続"),
                ("camera", "Camera", "未接続"),
                ("dlc", "DLC", "未接続"),
                ("experiment", "実験", "待機中"),
            )
        ):
            variable = tk.StringVar(value=f"{label}: {initial}")
            self._header_status_vars[key] = variable
            badge = tk.Label(
                badges,
                textvariable=variable,
                font=("Yu Gothic UI", 9, "bold"),
                padx=9,
                pady=5,
                bd=0,
                relief="flat",
            )
            badge.grid(row=0, column=index, padx=(4 if index else 0, 0))
            self._status_badges[key] = badge
            self._set_status_badge(key, initial, "idle")

        telemetry = ttk.Frame(header, padding=(0, 8, 0, 0))
        telemetry.grid(row=1, column=0, columnspan=3, sticky="ew")
        telemetry.columnconfigure(6, weight=1)
        for index, variable in enumerate(
            (
                self.phase_var,
                self.elapsed_var,
                self.camera_var,
                self.dlc_var,
                self.force_var,
                self.progress_var,
            )
        ):
            ttk.Label(telemetry, textvariable=variable, style="Telemetry.TLabel").grid(
                row=0, column=index, sticky="w", padx=(0 if index == 0 else 14, 8)
            )
        self.progress_bar = ttk.Progressbar(
            telemetry, mode="determinate", maximum=100.0, length=180
        )
        self.progress_bar.grid(row=0, column=6, sticky="ew", padx=(8, 0))

        controls = ttk.Frame(self, padding=(14, 2, 14, 6))
        controls.grid(row=1, column=0, sticky="ew")
        controls.columnconfigure(0, weight=1)
        ttk.Label(controls, textvariable=self.selected_var, style="Selected.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 8)
        )
        self.selected_path_button = ttk.Button(
            controls,
            text="フルパス",
            command=self._show_selected_path,
            state="disabled",
            style="Small.TButton",
        )
        self.selected_path_button.grid(row=0, column=1, padx=3)
        self.preflight_button = ttk.Button(
            controls, text="事前確認", command=self._preflight, style="Action.TButton"
        )
        self.preflight_button.grid(row=0, column=2, padx=3)
        self._input_widgets.append(self.preflight_button)
        self.connect_button = ttk.Button(
            controls, text="Connect", command=self._connect, style="Primary.TButton"
        )
        self.connect_button.grid(row=0, column=3, padx=3)
        self._input_widgets.append(self.connect_button)
        self.manual_speed_button = ttk.Button(
            controls,
            text="Manual Speed",
            command=self._show_manual_speed_window,
            state="disabled",
            style="Action.TButton",
        )
        self.manual_speed_button.grid(row=0, column=4, padx=3)
        self.start_button = ttk.Button(
            controls,
            text="再生開始",
            command=self._start,
            state="disabled",
            style="Primary.TButton",
        )
        self.start_button.grid(row=0, column=5, padx=3)
        self.stop_button = ttk.Button(
            controls,
            text="ソフトウェア停止",
            command=self._request_stop,
            state="disabled",
            style="Warning.TButton",
        )
        self.stop_button.grid(row=0, column=6, padx=3)
        self.force_quit_button = ttk.Button(
            controls,
            text="Force Quit",
            command=self._force_quit,
            state="disabled",
            style="Danger.TButton",
        )
        self.force_quit_button.grid(row=0, column=7, padx=3)
        ttk.Label(controls, textvariable=self.status_var, style="OperationStatus.TLabel").grid(
            row=0, column=8, sticky="e", padx=(10, 0)
        )
        self.alert_label = ttk.Label(
            self,
            text="",
            foreground="#991b1b",
            style="Alert.TLabel",
        )
        self.alert_label.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 2))

        selection = ttk.LabelFrame(self, text="CSV選択", padding=6)
        selection.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 4))
        selection.columnconfigure(0, weight=2)
        selection.columnconfigure(2, weight=3)
        self.file_list = tk.Listbox(
            selection,
            height=3,
            exportselection=False,
            font=("Yu Gothic UI", 10),
            activestyle="none",
        )
        self.file_list.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        file_scrollbar = ttk.Scrollbar(
            selection, orient="vertical", command=self.file_list.yview
        )
        file_scrollbar.grid(row=0, column=1, sticky="ns", padx=(0, 8))
        self.file_list.configure(yscrollcommand=file_scrollbar.set)
        self._input_widgets.append(self.file_list)
        self._csv_widgets.append(self.file_list)
        self.file_list.bind("<<ListboxSelect>>", self._on_select_csv)
        ttk.Label(
            selection,
            textvariable=self.detail_var,
            anchor="nw",
            justify="left",
            style="Detail.TLabel",
        ).grid(row=0, column=2, sticky="nsew", padx=5)

        settings_toggle = ttk.Frame(self, padding=(14, 2, 14, 2))
        settings_toggle.grid(row=4, column=0, sticky="ew")
        self.settings_toggle_button = ttk.Button(
            settings_toggle,
            text="設定を表示 ▼",
            command=self._toggle_settings,
            style="SectionToggle.TButton",
        )
        self.settings_toggle_button.pack(anchor="w")

        settings = ttk.LabelFrame(self, text="接続・安全設定", padding=8)
        settings.grid(row=5, column=0, sticky="ew", padx=14, pady=(0, 5))
        settings.columnconfigure(1, weight=1)
        settings.columnconfigure(3, weight=1)
        settings.columnconfigure(5, weight=2)
        settings.columnconfigure(7, weight=2)
        self.settings_body = settings
        ttk.Label(settings, text="CSVフォルダ").grid(row=0, column=0, sticky="w", padx=4, pady=3)
        self.folder_entry = ttk.Entry(settings, textvariable=self.folder_var)
        self.folder_entry.grid(row=0, column=1, columnspan=5, sticky="ew", padx=4, pady=3)
        self.choose_folder_button = ttk.Button(
            settings, text="参照", command=self._choose_folder, style="Small.TButton"
        )
        self.choose_folder_button.grid(row=0, column=6, padx=3, pady=3)
        self.refresh_files_button = ttk.Button(
            settings, text="更新", command=self._refresh_files, style="Small.TButton"
        )
        self.refresh_files_button.grid(row=0, column=7, padx=3, pady=3)
        self._input_widgets.extend(
            [self.folder_entry, self.choose_folder_button, self.refresh_files_button]
        )
        self._csv_widgets.extend(
            [self.folder_entry, self.choose_folder_button, self.refresh_files_button]
        )

        self._input_widgets.extend(
            [
                self._entry(settings, "Servoポート", self.servo_port_var, 1, 0, width=10),
                self._entry(settings, "Encoderポート", self.encoder_port_var, 1, 2, width=10),
                self._entry(settings, "DLC設定", self.dlc_config_var, 1, 4, width=30),
                self._entry(settings, "カメラ設定", self.camera_config_var, 1, 6, width=30),
            ]
        )
        force_controls = ttk.Frame(settings)
        force_controls.grid(row=2, column=0, columnspan=8, sticky="w", padx=4, pady=3)
        ttk.Label(force_controls, text="Force上限 [N]").pack(side="left", padx=(0, 8))
        for tendon in TENDONS:
            tendon_control = ttk.Frame(force_controls)
            tendon_control.pack(side="left", padx=(0, 8))
            ttk.Label(tendon_control, text=tendon).pack(side="left", padx=(0, 2))
            limit_entry = ttk.Entry(
                tendon_control, textvariable=self.limit_vars[tendon], width=7
            )
            limit_entry.pack(side="left")
            self._input_widgets.append(limit_entry)
        position_controls = ttk.Frame(settings)
        position_controls.grid(row=3, column=0, columnspan=8, sticky="w", padx=4, pady=3)
        ttk.Label(position_controls, text="Remembered encoder").pack(
            side="left", padx=(0, 8)
        )
        for tendon, servo_id in zip(TENDONS, SERVO_IDS_BY_TENDON):
            tendon_control = ttk.Frame(position_controls)
            tendon_control.pack(side="left", padx=(0, 8))
            remembered_var = tk.StringVar(value="--")
            self._remembered_position_vars[servo_id] = remembered_var
            ttk.Label(tendon_control, text=f"{tendon}:").pack(side="left")
            ttk.Label(tendon_control, textvariable=remembered_var, width=9).pack(
                side="left", padx=(2, 2)
            )
            remember_button = ttk.Button(
                tendon_control,
                text="Remember",
                command=lambda sid=servo_id: self._remember_position(sid),
                state="disabled",
                style="Small.TButton",
            )
            remember_button.pack(side="left", padx=(0, 2))
            return_button = ttk.Button(
                tendon_control,
                text="Return",
                command=lambda sid=servo_id: self._return_position(sid),
                state="disabled",
                style="Small.TButton",
            )
            return_button.pack(side="left")
            self._remember_position_buttons[servo_id] = remember_button
            self._return_position_buttons[servo_id] = return_button
        mode_controls = ttk.Frame(settings)
        mode_controls.grid(row=4, column=0, columnspan=8, sticky="w", padx=4, pady=3)
        ttk.Label(mode_controls, text="再生モード").pack(side="left", padx=(0, 8))
        interpolated_button = ttk.Radiobutton(
            mode_controls,
            text="50 Hz補間（既定）",
            variable=self.playback_mode_var,
            value=INTERPOLATED_PLAYBACK_MODE,
        )
        interpolated_button.pack(side="left", padx=(0, 8))
        zoh_button = ttk.Radiobutton(
            mode_controls,
            text="RL FF ZOH",
            variable=self.playback_mode_var,
            value=RL_FF_ZOH_PLAYBACK_MODE,
        )
        zoh_button.pack(side="left", padx=(0, 8))
        ttk.Label(mode_controls, text="Decision Hz").pack(side="left", padx=(8, 4))
        decision_hz_entry = ttk.Entry(
            mode_controls,
            textvariable=self.decision_hz_var,
            width=7,
        )
        decision_hz_entry.pack(side="left")
        safety_filter_button = ttk.Checkbutton(
            mode_controls,
            text="Safety Filter",
            variable=self.safety_filter_enabled_var,
        )
        safety_filter_button.pack(side="left", padx=(12, 4))
        feedback_button = ttk.Checkbutton(
            mode_controls,
            text="FF+FB（Safety必須）",
            variable=self.feedback_enabled_var,
        )
        feedback_button.pack(side="left", padx=(4, 0))
        self._input_widgets.extend(
            [
                interpolated_button,
                zoh_button,
                decision_hz_entry,
                safety_filter_button,
                feedback_button,
            ]
        )
        self._csv_widgets.extend(
            [
                interpolated_button,
                zoh_button,
                decision_hz_entry,
                safety_filter_button,
                feedback_button,
            ]
        )
        point_to_point_controls = ttk.Frame(settings)
        point_to_point_controls.grid(
            row=5, column=0, columnspan=8, sticky="w", padx=4, pady=3
        )
        ttk.Label(point_to_point_controls, text="軌道入力").pack(
            side="left", padx=(0, 6)
        )
        csv_source_button = ttk.Radiobutton(
            point_to_point_controls,
            text="CSV Excursion",
            variable=self.trajectory_source_var,
            value=CSV_TRAJECTORY_SOURCE,
            command=self._on_trajectory_source_changed,
        )
        csv_source_button.pack(side="left", padx=(0, 8))
        p2p_source_button = ttk.Radiobutton(
            point_to_point_controls,
            text="Point-to-Point",
            variable=self.trajectory_source_var,
            value=POINT_TO_POINT_SOURCE,
            command=self._on_trajectory_source_changed,
        )
        p2p_source_button.pack(side="left", padx=(0, 10))
        self._input_widgets.extend([csv_source_button, p2p_source_button])
        self._csv_widgets.extend([csv_source_button, p2p_source_button])
        for joint in JOINTS:
            ttk.Label(point_to_point_controls, text=f"{joint} goal [deg]").pack(
                side="left", padx=(4, 2)
            )
            entry = ttk.Entry(
                point_to_point_controls,
                textvariable=self.point_to_point_goal_vars[joint],
                width=7,
            )
            entry.pack(side="left", padx=(0, 4))
            self._input_widgets.append(entry)
            self._csv_widgets.append(entry)
        ttk.Label(point_to_point_controls, text="T [s]").pack(
            side="left", padx=(4, 2)
        )
        duration_entry = ttk.Entry(
            point_to_point_controls,
            textvariable=self.point_to_point_duration_var,
            width=7,
        )
        duration_entry.pack(side="left", padx=(0, 4))
        self._input_widgets.append(duration_entry)
        self._csv_widgets.append(duration_entry)
        return_all_button = ttk.Button(
            position_controls,
            text="Return All",
            command=self._return_all_positions,
            state="disabled",
            style="Action.TButton",
        )
        return_all_button.pack(side="left", padx=(4, 0))
        self._return_all_positions_button = return_all_button
        self._refresh_remembered_position_display()
        self.settings_body.grid_remove()

        body = ttk.PanedWindow(self, orient="vertical")
        body.grid(row=6, column=0, sticky="nsew", padx=14, pady=(0, 10))
        monitor = ttk.PanedWindow(body, orient="horizontal")
        body.add(monitor, weight=4)
        preview_frame = ttk.LabelFrame(monitor, text="Camera / DLC Preview", padding=6)
        force_frame = ttk.LabelFrame(monitor, text="Estimated Force", padding=6)
        angle_frame = ttk.LabelFrame(monitor, text="Flexion Angles", padding=6)
        monitor.add(preview_frame, weight=3)
        monitor.add(force_frame, weight=4)
        monitor.add(angle_frame, weight=3)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(1, weight=1)
        preview_controls = ttk.Frame(preview_frame)
        preview_controls.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        self.preview_toggle_button = ttk.Button(
            preview_controls,
            text="プレビューON",
            command=self._toggle_preview,
            style="Small.TButton",
        )
        self.preview_toggle_button.pack(side="right")
        self.dlc_toggle_button = ttk.Button(
            preview_controls,
            text="DLC ON",
            command=self._toggle_dlc,
            style="Small.TButton",
        )
        self.dlc_toggle_button.pack(side="right", padx=(0, 6))
        self.plot_reset_button = ttk.Button(
            preview_controls,
            text="Plot Reset",
            command=self._reset_plots,
            style="Small.TButton",
        )
        self.plot_reset_button.pack(side="left")
        self.preview_label = tk.Label(
            preview_frame,
            text="カメラ未接続",
            anchor="center",
            bg="#e9eef3",
            fg="#475569",
            font=("Yu Gothic UI", 12),
        )
        self.preview_label.grid(row=1, column=0, sticky="nsew")
        self._build_force_plot(force_frame)
        self._build_angle_plot(angle_frame)

        results = ttk.Frame(body, padding=(0, 6, 0, 0))
        body.add(results, weight=1)
        results.columnconfigure(0, weight=1)
        results.rowconfigure(0, weight=1)
        result_table = ttk.LabelFrame(results, text="実験結果一覧", padding=5)
        result_table.grid(row=0, column=0, sticky="nsew")
        result_table.columnconfigure(0, weight=1)
        result_table.rowconfigure(0, weight=1)
        self.result_tree = ttk.Treeview(
            result_table,
            columns=("status", "max_force", "likelihood", "session"),
            show="headings",
            height=8,
        )
        headings = {
            "status": "状態",
            "max_force": "最大Force [N]",
            "likelihood": "最低DLC尤度",
            "session": "セッション",
        }
        for column, heading in headings.items():
            self.result_tree.heading(column, text=heading)
        self.result_tree.column("status", width=150, minwidth=120, stretch=False)
        self.result_tree.column("max_force", width=300, minwidth=180, stretch=True)
        self.result_tree.column("likelihood", width=140, minwidth=120, stretch=False)
        self.result_tree.column("session", width=460, minwidth=220, stretch=True)
        self.result_tree.grid(row=0, column=0, sticky="nsew")
        result_scrollbar = ttk.Scrollbar(
            result_table, orient="vertical", command=self.result_tree.yview
        )
        result_scrollbar.grid(row=0, column=1, sticky="ns")
        self.result_tree.configure(yscrollcommand=result_scrollbar.set)
        self.open_session_button = ttk.Button(
            result_table,
            text="セッションフォルダを開く",
            command=self._open_session_folder,
            state="disabled",
            style="Small.TButton",
        )
        self.open_session_button.grid(row=1, column=0, sticky="w", pady=(5, 0))

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("vista")
        except tk.TclError:
            try:
                style.theme_use("clam")
            except tk.TclError:
                pass
        style.configure("TLabel", font=("Yu Gothic UI", 10))
        style.configure("TEntry", font=("Yu Gothic UI", 10))
        style.configure("TLabelframe.Label", font=("Yu Gothic UI", 12, "bold"), foreground="#1e293b")
        style.configure("AppTitle.TLabel", font=("Yu Gothic UI", 18, "bold"), foreground="#0f172a")
        style.configure("HeaderFile.TLabel", font=("Yu Gothic UI", 11), foreground="#334155")
        style.configure("Selected.TLabel", font=("Yu Gothic UI", 11, "bold"), foreground="#1e293b")
        style.configure("Detail.TLabel", font=("Yu Gothic UI", 10), foreground="#475569")
        style.configure("Telemetry.TLabel", font=("Yu Gothic UI", 9), foreground="#475569")
        style.configure("OperationStatus.TLabel", font=("Yu Gothic UI", 10, "bold"), foreground="#334155")
        style.configure("Alert.TLabel", font=("Yu Gothic UI", 10, "bold"))
        style.configure("Small.TButton", font=("Yu Gothic UI", 10), padding=(8, 4))
        style.configure("Action.TButton", font=("Yu Gothic UI", 10, "bold"), padding=(10, 5))
        style.configure("Primary.TButton", font=("Yu Gothic UI", 10, "bold"), padding=(11, 5))
        style.configure("Warning.TButton", font=("Yu Gothic UI", 10, "bold"), foreground="#92400e", padding=(11, 5))
        style.configure("Danger.TButton", font=("Yu Gothic UI", 10, "bold"), foreground="#991b1b", padding=(11, 5))
        style.configure("SectionToggle.TButton", font=("Yu Gothic UI", 10, "bold"), padding=(4, 2))
        style.configure("Treeview", font=("Yu Gothic UI", 10), rowheight=28)
        style.configure("Treeview.Heading", font=("Yu Gothic UI", 10, "bold"))

    def _set_status_badge(self, key: str, text: str, state: str = "idle") -> None:
        palette = {
            "idle": ("#e2e8f0", "#334155"),
            "ok": ("#dcfce7", "#166534"),
            "busy": ("#dbeafe", "#1d4ed8"),
            "warn": ("#fef3c7", "#92400e"),
            "error": ("#fee2e2", "#991b1b"),
        }
        badge = self._status_badges.get(key)
        variable = self._header_status_vars.get(key)
        if badge is None or variable is None:
            return
        badge_value = (str(text), str(state))
        if self._status_badge_values.get(key) == badge_value:
            return
        self._status_badge_values[key] = badge_value
        label = {
            "servo": "Servo",
            "encoder": "Encoder",
            "camera": "Camera",
            "dlc": "DLC",
            "experiment": "実験",
        }.get(key, key)
        variable.set(f"{label}: {text}")
        background, foreground = palette.get(state, palette["idle"])
        badge.configure(background=background, foreground=foreground)

    def _toggle_settings(self) -> None:
        self._settings_expanded = not self._settings_expanded
        if self._settings_expanded:
            self.settings_body.grid()
            self.settings_toggle_button.configure(text="設定を隠す ▲")
        else:
            self.settings_body.grid_remove()
            self.settings_toggle_button.configure(text="設定を表示 ▼")

    def _show_selected_path(self) -> None:
        path = self._selected_csv_path
        if path is None:
            messagebox.showinfo("CSVフルパス", "CSVが選択されていません")
            return
        messagebox.showinfo("CSVフルパス", str(path.resolve()))

    @staticmethod
    def _shorten_path(path: Path | str, max_chars: int = 72) -> str:
        value = str(path)
        if len(value) <= max_chars:
            return value
        head = max_chars // 2 - 1
        tail = max_chars - head - 1
        return f"{value[:head]}…{value[-tail:]}"

    def _apply_status_event(self, text: str, phase: str) -> None:
        self.status_var.set(text)
        self.phase_var.set(phase)
        if "カメラとDLC" in text:
            self._set_status_badge("camera", "接続中", "busy")
            self._set_status_badge("dlc", "準備中", "busy")
            self._set_status_badge("experiment", "準備中", "busy")
        elif "DLC" in text:
            self._set_status_badge("dlc", "キャリブレーション", "busy")
            self._set_status_badge("experiment", phase, "busy")
        elif "サーボとエンコーダ" in text:
            self._set_status_badge("servo", "接続中", "busy")
            self._set_status_badge("encoder", "接続中", "busy")
            self._set_status_badge("experiment", "接続中", "busy")
        elif phase in {"実行中", "再生開始"}:
            self._set_status_badge("experiment", "実行中", "ok")
        elif phase in {"完了", "接続完了"}:
            self._set_status_badge("experiment", phase, "ok")
        elif phase in {"停止", "Force Quit"}:
            self._set_status_badge("experiment", phase, "warn")

    @staticmethod
    def _entry(parent: ttk.Widget, label: str, variable: tk.StringVar, row: int, column: int, *, width: int = 18) -> ttk.Entry:
        ttk.Label(parent, text=label).grid(row=row, column=column, padx=(4, 1), pady=4, sticky="w")
        entry = ttk.Entry(parent, textvariable=variable, width=width)
        entry.grid(
            row=row, column=column + 1, padx=(0, 4), pady=4, sticky="ew"
        )
        return entry

    def _build_force_plot(self, parent: ttk.Frame) -> None:
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)
        self.force_figure = Figure(figsize=(8, 5), dpi=100)
        self.force_axes = self.force_figure.subplots(3, 2, squeeze=False).flatten()
        force_handles: tuple[Any, Any] | None = None
        for axis, tendon in zip(self.force_axes, TENDONS):
            line, = axis.plot([], [], label="Estimated")
            limit_line, = axis.plot(
                [], [], color="#b91c1c", linestyle="--", alpha=0.55, label="Limit"
            )
            if force_handles is None:
                force_handles = (line, limit_line)
            self.force_lines[tendon] = line
            self.force_limit_lines[tendon] = limit_line
            line.set_animated(True)
            limit_line.set_animated(True)
            axis.set_title(tendon, fontsize=10, pad=4)
            axis.set_ylabel("N", fontsize=9)
            axis.tick_params(labelsize=8)
            axis.set_xlim(0.0, 20.0)
            axis.set_ylim(0.0, 1.0)
            axis.grid(True, alpha=0.3)
        for axis in self.force_axes[-2:]:
            axis.set_xlabel("Time [s]", fontsize=9)
        self.force_figure.suptitle(
            "Estimated force", fontsize=12, x=0.02, y=0.995, ha="left"
        )
        if force_handles is not None:
            self.force_figure.legend(
                force_handles,
                ("Estimated", "Limit"),
                loc="upper right",
                bbox_to_anchor=(0.98, 0.995),
                fontsize=8,
                frameon=False,
                ncol=2,
            )
        self.force_figure.subplots_adjust(
            left=0.08, right=0.98, bottom=0.09, top=0.91, hspace=0.52, wspace=0.28
        )
        self.force_canvas = FigureCanvasTkAgg(self.force_figure, master=parent)
        self.force_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.force_canvas.mpl_connect(
            "draw_event", self._capture_force_plot_backgrounds
        )
        self.force_canvas.draw()
        self._blit_force_plot()

    def _build_angle_plot(self, parent: ttk.Frame) -> None:
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)
        self.angle_figure = Figure(figsize=(6, 5), dpi=100)
        self.angle_axis = self.angle_figure.subplots()
        for joint in JOINTS:
            line, = self.angle_axis.plot([], [], label=f"{joint} flexion")
            target_line, = self.angle_axis.plot(
                [],
                [],
                color=line.get_color(),
                linestyle="--",
                alpha=0.8,
                label=f"{joint} target",
            )
            line.set_animated(True)
            target_line.set_animated(True)
            self.angle_lines[joint] = line
            self.target_angle_lines[joint] = target_line
        self.angle_axis.set_title("Flexion angle", fontsize=12, pad=8)
        self.angle_axis.set_xlabel("Time [s]", fontsize=10)
        self.angle_axis.set_ylabel("Flexion angle [deg]", fontsize=10)
        self.angle_axis.set_xlim(0.0, 20.0)
        self.angle_axis.set_ylim(0.0, 180.0)
        self.angle_axis.tick_params(labelsize=9)
        self.angle_axis.grid(True, alpha=0.3)
        self.angle_axis.legend(loc="upper left", fontsize=9, frameon=False)
        self.angle_figure.subplots_adjust(left=0.16, right=0.98, bottom=0.13, top=0.90)
        self.angle_canvas = FigureCanvasTkAgg(self.angle_figure, master=parent)
        self.angle_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.angle_canvas.mpl_connect(
            "draw_event", self._capture_angle_plot_background
        )
        self.angle_canvas.draw()
        self._blit_angle_plot()

    def _choose_folder(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.folder_var.get())
        if selected:
            self.folder_var.set(selected)
            self._refresh_files()

    def _set_input_widgets_enabled(self, enabled: bool) -> None:
        self._set_widgets_enabled(self._input_widgets, enabled)

    @staticmethod
    def _set_widgets_enabled(widgets: list[Any], enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for widget in widgets:
            widget.configure(state=state)

    def _preflight(self) -> None:
        if self._running or self._start_in_progress:
            return
        if self.trajectory_source_var.get() == POINT_TO_POINT_SOURCE:
            try:
                limits = self._parse_limits()
                calibration = self._load_encoder_calibration()
                goal_deg = tuple(
                    float(self.point_to_point_goal_vars[joint].get())
                    for joint in JOINTS
                )
                duration_s = float(self.point_to_point_duration_var.get())
                decision_hz = float(self.decision_hz_var.get())
                if not math.isfinite(decision_hz) or not 0.0 < decision_hz <= DEFAULT_CONTROL_HZ:
                    raise ValueError("Decision Hz is outside the supported range")
                model = _validate_point_to_point_request(
                    tuple(math.radians(value) for value in goal_deg),
                    duration_s,
                    playback_mode=self.playback_mode_var.get().strip(),
                    decision_hz=decision_hz,
                    safety_filter_enabled=bool(self.safety_filter_enabled_var.get()),
                    encoder_calibration=calibration,
                )
                required_files = (
                    ("DLC設定", Path(self.dlc_config_var.get()).expanduser()),
                    ("カメラ設定", Path(self.camera_config_var.get()).expanduser()),
                    ("encoder位置キャリブレーション", DEFAULT_ENCODER_CALIBRATION_PATH),
                )
                missing = [
                    f"{label}: {file_path}"
                    for label, file_path in required_files
                    if not file_path.is_file()
                ]
                if missing:
                    raise FileNotFoundError(
                        "必要なファイルがありません:\n" + "\n".join(missing)
                    )
            except Exception as exc:
                self.status_var.set(f"事前確認エラー: {exc}")
                self.phase_var.set("事前確認エラー")
                messagebox.showerror("事前確認エラー", str(exc))
                return
            self.status_var.set("Point-to-Point事前条件OK（q0取得は開始時）")
            self.phase_var.set("事前確認")
            self._set_status_badge("experiment", "事前確認OK", "ok")
            self.detail_var.set(
                f"目標: {', '.join(f'{joint}={value:g}°' for joint, value in zip(JOINTS, goal_deg))}\n"
                f"運動時間: {duration_s:g} s / Decision: {decision_hz:g} Hz\n"
                f"encoder換算: {calibration.calibration_status}\n"
                f"Force上限: {', '.join(f'{tendon}={limits[tendon]:g} N' for tendon in TENDONS)}\n"
                f"Moment Arm: {model.model_info}\n"
                "開始時にq0、承認角度範囲、FF Excursion、Decision間Δを再検証します。"
            )
            return
        selection = self.file_list.curselection()
        if not selection:
            messagebox.showwarning("CSV未選択", "事前確認するCSVを選択してください")
            return

        path = self.csv_paths[selection[0]]
        try:
            limits = self._parse_limits()
            calibration = self._load_encoder_calibration()
            trajectory = ExcursionTrajectory.from_csv(path)
            playback_mode = self.playback_mode_var.get().strip()
            integrated_runtime = self._build_selected_integrated_runtime(
                trajectory,
                limits,
                playback_mode,
            )
            required_files = (
                ("CSV", path),
                ("encoder位置キャリブレーション", DEFAULT_ENCODER_CALIBRATION_PATH),
                ("DLC設定", Path(self.dlc_config_var.get()).expanduser()),
                ("カメラ設定", Path(self.camera_config_var.get()).expanduser()),
            )
            missing = [
                f"{label}: {file_path}"
                for label, file_path in required_files
                if not file_path.is_file()
            ]
            if missing:
                raise FileNotFoundError("必要なファイルがありません:\n" + "\n".join(missing))
        except Exception as exc:
            self.status_var.set(f"事前確認エラー: {exc}")
            self.phase_var.set("事前確認エラー")
            messagebox.showerror("事前確認エラー", str(exc))
            return

        self.status_var.set("事前確認OK（接続は未確認）")
        self.phase_var.set("事前確認")
        self._set_status_badge("experiment", "事前確認OK", "ok")
        self.detail_var.set(
            f"ファイル: {path.name}\n"
            f"行数: {len(trajectory.elapsed_times_s)}\n"
            f"再生時間: {trajectory.duration_s:.3f} s\n"
            f"encoder換算: {calibration.calibration_status}\n"
            f"Force上限: {', '.join(f'{tendon}={limits[tendon]:g} N' for tendon in TENDONS)}\n"
            f"統合制御: FF+FB={integrated_runtime.feedback_enabled}, "
            f"Safety={integrated_runtime.safety_filter_enabled}, Exploration=False\n"
            "状態: 事前確認OK（camera / serial接続は未確認）"
        )

    def _open_session_folder(self) -> None:
        session_dir = self._session_dir
        if session_dir is None or not session_dir.is_dir():
            messagebox.showinfo("セッションフォルダ", "開けるセッションフォルダがありません")
            return
        try:
            startfile = getattr(os, "startfile", None)
            if not callable(startfile):
                raise RuntimeError("このOSではフォルダを開けません")
            startfile(str(session_dir))
        except Exception as exc:
            messagebox.showerror("フォルダを開けません", str(exc))

    def _refresh_files(self) -> None:
        folder = Path(self.folder_var.get()).expanduser()
        self.file_list.delete(0, tk.END)
        paths: list[tuple[float, Path]] = []
        try:
            for path in folder.glob("*.csv"):
                try:
                    if path.is_file():
                        paths.append((path.stat().st_mtime, path))
                except OSError:
                    continue
        except OSError as exc:
            self.csv_paths = []
            self._selected_csv_path = None
            self._selected_csv_valid = False
            self._set_target_angle_trajectory(None)
            self.selected_var.set("CSV: 未選択")
            self.detail_var.set(f"CSV一覧を読めません: {exc}")
            self.selected_path_button.configure(state="disabled")
            self.connect_button.configure(
                state="disabled" if self._running else "normal"
            )
            self._update_start_button_state()
            return
        self.csv_paths = [path for _, path in sorted(paths, reverse=True)]
        for path in self.csv_paths:
            self.file_list.insert(tk.END, path.name)
        if self.csv_paths:
            self.file_list.selection_set(0)
            self._on_select_csv()
        else:
            self._selected_csv_path = None
            self._selected_csv_valid = False
            self._set_target_angle_trajectory(None)
            self.selected_var.set("CSV: 未選択")
            self.detail_var.set(f"CSVがありません: {folder}")
            self.selected_path_button.configure(state="disabled")
            self.connect_button.configure(
                state="disabled" if self._running else "normal"
            )
            self._update_start_button_state()

    def _on_select_csv(self, _event: object | None = None) -> None:
        selection = self.file_list.curselection()
        if not selection:
            return
        path = self.csv_paths[selection[0]]
        self._selected_csv_path = path
        self.selected_var.set(f"CSV: {path.name}")
        self.selected_path_button.configure(state="normal")
        try:
            calibration = self._load_encoder_calibration()
            trajectory = ExcursionTrajectory.from_csv(path)
            self._set_target_angle_trajectory(trajectory)
            scale_note = (
                "幾何学換算（明示承認済み）"
                if calibration.p2p_conversion_approved
                and calibration.calibration_status.lower() != "measured"
                else "腱別実測校正"
                if calibration.calibration_status.lower() == "measured"
                else "未承認の公称値"
            )
            self.detail_var.set(
                f"ファイル: {path.name}\n"
                f"行数: {len(trajectory.elapsed_times_s)}\n"
                f"再生時間: {trajectory.duration_s:.3f} s\n"
                f"encoder換算: {scale_note}\n"
                f"状態: 実行可能"
            )
            self.connect_button.configure(
                state="normal"
                if not self._running and not self._start_in_progress
                else "disabled"
            )
            self._selected_csv_valid = True
            self._update_start_button_state()
        except Exception as exc:
            self._set_target_angle_trajectory(None)
            self.detail_var.set(f"検証エラー: {type(exc).__name__}: {exc}")
            self._selected_csv_valid = False
            self.connect_button.configure(
                state="disabled" if self._running else "normal"
            )
            self.start_button.configure(state="disabled")

    def _on_trajectory_source_changed(self) -> None:
        if self.trajectory_source_var.get() == POINT_TO_POINT_SOURCE:
            self.selected_var.set("軌道: Point-to-Point（実測q0 → 目標姿勢）")
            self._set_target_angle_trajectory(None)
        elif self._selected_csv_path is not None:
            self.selected_var.set(f"CSV: {self._selected_csv_path.name}")
            try:
                self._set_target_angle_trajectory(
                    ExcursionTrajectory.from_csv(self._selected_csv_path)
                )
            except Exception:
                self._set_target_angle_trajectory(None)
        self._update_start_button_state()

    def _start_input_ready(self) -> bool:
        if self.trajectory_source_var.get() == POINT_TO_POINT_SOURCE:
            return True
        return bool(self.file_list.curselection() and self._selected_csv_valid)

    def _update_start_button_state(self) -> None:
        ready = (
            self._running
            and self._connection_ready
            and not self._playback_started
            and not self._manual_command_pending
            and not self._manual_speed_active()
            and not self._position_action_pending
            and self._start_input_ready()
        )
        self.start_button.configure(state="normal" if ready else "disabled")

    @staticmethod
    def _load_encoder_calibration() -> EncoderExcursionCalibration:
        return load_encoder_excursion_calibration(DEFAULT_ENCODER_CALIBRATION_PATH)

    def _build_selected_integrated_runtime(
        self,
        trajectory: ExcursionTrajectory,
        limits: Mapping[str, float],
        playback_mode: str,
    ) -> IntegratedControlRuntime:
        if (
            (self.feedback_enabled_var.get() or self.safety_filter_enabled_var.get())
            and not encoder_conversion_is_p2p_approved(
                self._load_encoder_calibration()
            )
        ):
            raise ValueError(
                "Integrated D0/D1 requires measured encoder calibration or an "
                "explicitly approved spool-radius geometry conversion"
            )
        runtime = _build_integrated_control_runtime(
            trajectory,
            limits,
            feedback_enabled=bool(self.feedback_enabled_var.get()),
            safety_filter_enabled=bool(self.safety_filter_enabled_var.get()),
        )
        if (
            runtime.active
            and runtime.require_rl_ff_zoh
            and playback_mode != RL_FF_ZOH_PLAYBACK_MODE
        ):
            raise ValueError(
                "FF+FBまたはSafety Filterを使う場合はRL FF ZOHを選択してください"
            )
        return runtime

    def _set_target_angle_trajectory(
        self,
        trajectory: ExcursionTrajectory | None,
    ) -> None:
        target_angles = None if trajectory is None else trajectory.target_angles_deg
        if trajectory is None or target_angles is None:
            self.target_angle_history = {joint: () for joint in JOINTS}
        else:
            times = tuple(
                elapsed_s / trajectory.time_scale
                for elapsed_s in trajectory.elapsed_times_s
            )
            self.target_angle_history = {
                joint: tuple(
                    (elapsed_s, row[joint_index])
                    for elapsed_s, row in zip(times, target_angles)
                )
                for joint_index, joint in enumerate(JOINTS)
            }
        self._angle_plot_dirty = True

    def _reset_plots(self) -> None:
        for history in self.force_history.values():
            history.clear()
        for history in self.angle_history.values():
            history.clear()
        self._last_angle_frame = -1
        self._force_plot_backgrounds = {}
        self._angle_plot_background = None
        self._force_plot_dirty = True
        self._angle_plot_dirty = True
        self._update_force_plot()
        self._update_angle_plot()

    def _parse_limits(self) -> dict[str, float]:
        limits: dict[str, float] = {}
        for tendon in TENDONS:
            value = float(self.limit_vars[tendon].get())
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{tendon}のHard limitは正の有限値が必要です")
            limits[tendon] = value
        return limits

    def _manual_speeds_snapshot(self) -> dict[int, int]:
        with self._manual_speed_lock:
            return dict(self._manual_speed_commands)

    def _manual_speed_active(self) -> bool:
        return any(self._manual_speeds_snapshot().values())

    def _show_manual_speed_window(self) -> None:
        if self._manual_window is not None and self._manual_window.winfo_exists():
            self._manual_window.deiconify()
            self._manual_window.lift()
            return

        window = tk.Toplevel(self)
        window.title("CSV Player | Manual Speed")
        window.resizable(False, False)
        window.protocol("WM_DELETE_WINDOW", self._hide_manual_speed_window)
        self._manual_window = window

        ttk.Label(
            window,
            text=(
                "接続後・CSV再生前専用です。速度はraw ServoHardwareAPI単位で、"
                "0を押すまで継続します。"
            ),
        ).grid(row=0, column=0, columnspan=7, sticky="w", padx=10, pady=(10, 6))
        headers = ("ID", "Tendon", "Encoder [deg]", "Force [N]", "Command", "Status", "Speed")
        for column, label in enumerate(headers):
            ttk.Label(window, text=label, style="Detail.TLabel").grid(
                row=1, column=column, sticky="w", padx=5, pady=3
            )

        self._manual_row_vars = {}
        self._manual_speed_buttons = {}
        for row, (tendon, servo_id) in enumerate(
            zip(TENDONS, SERVO_IDS_BY_TENDON), start=2
        ):
            variables = {
                "encoder": tk.StringVar(value="--"),
                "force": tk.StringVar(value="--"),
                "command": tk.StringVar(value="0"),
                "status": tk.StringVar(value="waiting"),
            }
            self._manual_row_vars[servo_id] = variables
            ttk.Label(window, text=str(servo_id)).grid(row=row, column=0, padx=5, pady=3)
            ttk.Label(window, text=tendon).grid(row=row, column=1, sticky="w", padx=5, pady=3)
            ttk.Label(window, textvariable=variables["encoder"], width=14).grid(
                row=row, column=2, sticky="w", padx=5, pady=3
            )
            ttk.Label(window, textvariable=variables["force"], width=12).grid(
                row=row, column=3, sticky="w", padx=5, pady=3
            )
            ttk.Label(window, textvariable=variables["command"], width=10).grid(
                row=row, column=4, sticky="w", padx=5, pady=3
            )
            ttk.Label(window, textvariable=variables["status"], width=22).grid(
                row=row, column=5, sticky="w", padx=5, pady=3
            )
            controls = ttk.Frame(window)
            controls.grid(row=row, column=6, sticky="w", padx=5, pady=3)
            buttons: list[tuple[int, ttk.Button]] = []
            for speed in MANUAL_SPEED_PRESETS:
                button = ttk.Button(
                    controls,
                    text=str(speed),
                    width=6,
                    command=lambda sid=servo_id, value=speed: self._set_manual_speed(
                        sid, value
                    ),
                    style="Small.TButton",
                )
                button.pack(side="left", padx=1)
                buttons.append((speed, button))
            self._manual_speed_buttons[servo_id] = buttons

        self._manual_stop_all_button = ttk.Button(
            window,
            text="STOP ALL",
            command=self._stop_all_manual_speeds,
            style="Danger.TButton",
        )
        self._manual_stop_all_button.grid(
            row=2 + len(SERVO_IDS), column=0, columnspan=7, pady=(8, 10)
        )
        self._refresh_manual_speed_window()
        self._update_manual_controls()

    def _hide_manual_speed_window(self) -> None:
        if self._manual_command_pending:
            return
        if self._manual_speed_active():
            self._stop_all_manual_speeds()
        if self._manual_window is not None:
            self._manual_window.withdraw()

    def _refresh_manual_speed_window(self) -> None:
        if (
            not self._manual_row_vars
            or self._manual_window is None
            or not self._manual_window.winfo_exists()
            or not self._manual_window.winfo_viewable()
        ):
            return
        sample = self._latest_manual_sample or {}
        readings = sample.get("force_readings", {})
        positions = sample.get("actual_positions", {})
        commands = self._manual_speeds_snapshot()
        for servo_id, variables in self._manual_row_vars.items():
            reading = readings.get(servo_id, readings.get(str(servo_id), {}))
            position = positions.get(servo_id, positions.get(str(servo_id)))
            force_n = reading.get("force_n") if isinstance(reading, dict) else None
            status = (
                str(reading.get("status", "UNAVAILABLE"))
                if isinstance(reading, dict)
                else "UNAVAILABLE"
            )
            variables["encoder"].set(
                "--" if position is None else f"{float(position):.3f}"
            )
            variables["force"].set(
                "--" if force_n is None else f"{float(force_n):.3f}"
            )
            variables["command"].set(str(commands.get(servo_id, 0)))
            variables["status"].set(status)

    def _update_manual_controls(self) -> None:
        connected = self._connection_ready and not self._playback_started
        idle = connected and not self._position_action_pending and not self._manual_command_pending
        if hasattr(self, "manual_speed_button"):
            self.manual_speed_button.configure(state="normal" if connected else "disabled")
        for buttons in self._manual_speed_buttons.values():
            for speed, button in buttons:
                button.configure(state="normal" if idle else "disabled")
        if self._manual_stop_all_button is not None:
            self._manual_stop_all_button.configure(
                state="normal" if connected and not self._manual_command_pending else "disabled"
            )

    def _set_manual_speed(self, servo_id: int, speed: int) -> None:
        if speed not in MANUAL_SPEED_PRESETS:
            return
        self._submit_manual_speed_commands(
            {int(servo_id): int(speed)},
            f"Servo {servo_id} manual speed={speed}",
        )

    def _stop_all_manual_speeds(self) -> None:
        self._submit_manual_speed_commands(
            {servo_id: 0 for servo_id in SERVO_IDS},
            "Manual speed STOP ALL",
            stop_all=True,
        )

    def _submit_manual_speed_commands(
        self,
        commands: dict[int, int],
        description: str,
        *,
        stop_all: bool = False,
    ) -> None:
        if (
            not self._connection_ready
            or self._playback_started
            or self._position_action_pending
            or self._manual_command_pending
        ):
            return
        with self._resource_lock:
            hardware = self._hardware
        if hardware is None:
            return
        self._manual_command_pending = True
        self._update_manual_controls()
        self._update_position_action_buttons()

        def worker() -> None:
            command_lock_acquired = False
            try:
                command_lock_acquired = self._servo_command_path_lock.acquire(
                    blocking=False
                )
                if not command_lock_acquired:
                    raise RuntimeError("Another servo command path is active")
                stop_event = self._stop_event
                if stop_event is None or stop_event.is_set() or self._playback_started:
                    raise RuntimeError("Manual speed session is stopping")
                if stop_all:
                    hardware.stop_servos()
                else:
                    if any(int(speed) != 0 for speed in commands.values()):
                        deadline = time.monotonic() + MANUAL_FEEDBACK_WAIT_TIMEOUT_S
                        while True:
                            if stop_event.is_set() or self._playback_started:
                                raise RuntimeError(
                                    "Manual speed session is stopping"
                                )
                            remaining = deadline - time.monotonic()
                            if remaining <= 0.0:
                                raise TimeoutError(
                                    "新しいForce・encoder telemetryを"
                                    f"{MANUAL_FEEDBACK_WAIT_TIMEOUT_S:g}秒以内に"
                                    "確認できませんでした"
                                )
                            if self._manual_feedback_ready_event.wait(
                                min(MANUAL_MONITOR_INTERVAL_S, remaining)
                            ):
                                break
                    if stop_event.is_set() or self._playback_started:
                        raise RuntimeError("Manual speed session is stopping")
                    receipt = hardware.submit_speed_to_servos(commands).wait(1.0)
                    if receipt.status != "sent":
                        raise RuntimeError(
                            f"Manual speed command was not sent: {receipt.status}"
                        )
                with self._manual_speed_lock:
                    self._manual_speed_commands.update(commands)
            except Exception as exc:
                try:
                    hardware.stop_servos()
                except Exception:
                    pass
                with self._manual_speed_lock:
                    self._manual_speed_commands = {
                        servo_id: 0 for servo_id in SERVO_IDS
                    }
                self._emit(
                    {
                        "kind": "manual_command_error",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            else:
                self._emit(
                    {
                        "kind": "manual_command_done",
                        "description": description,
                    }
                )
            finally:
                if command_lock_acquired:
                    self._servo_command_path_lock.release()

        threading.Thread(
            target=worker,
            name="csv-player-manual-speed",
            daemon=True,
        ).start()

    def _refresh_remembered_position_display(self) -> None:
        try:
            controller = self._position_controller
            positions = (
                controller.positions()
                if controller is not None
                else load_encoder_calibration_positions()
            )
        except Exception as exc:
            positions = {}
            self.status_var.set(f"Remember位置を読めません: {exc}")
        for servo_id, variable in self._remembered_position_vars.items():
            record = positions.get(servo_id)
            variable.set(
                "--"
                if not record
                else f"{float(record['encoder_position_deg']):.3f}°"
            )
        self._update_position_action_buttons(positions)

    def _update_position_action_buttons(
        self,
        positions: dict[int, dict[str, Any]] | None = None,
    ) -> None:
        if positions is None:
            try:
                positions = (
                    self._position_controller.positions()
                    if self._position_controller is not None
                    else load_encoder_calibration_positions()
                )
            except Exception:
                positions = {}
        enabled = (
            self._connection_ready
            and not self._playback_started
            and not self._position_action_pending
            and not self._manual_command_pending
            and not self._manual_speed_active()
            and self._position_controller is not None
        )
        for servo_id, button in self._remember_position_buttons.items():
            button.configure(state="normal" if enabled else "disabled")
            self._return_position_buttons[servo_id].configure(
                state="normal" if enabled and servo_id in positions else "disabled"
            )
        if self._return_all_positions_button is not None:
            all_remembered = all(servo_id in positions for servo_id in SERVO_IDS)
            self._return_all_positions_button.configure(
                state="normal" if enabled and all_remembered else "disabled"
            )

    def _start_position_action(
        self,
        action: Callable[[RememberedPositionController, threading.Event], Any],
        description: str,
    ) -> None:
        controller = self._position_controller
        if (
            controller is None
            or not self._connection_ready
            or self._playback_started
            or self._position_action_pending
            or self._manual_command_pending
            or self._manual_speed_active()
        ):
            return
        cancel_event = threading.Event()
        self._position_action_pending = True
        self._position_action_cancel_event = cancel_event
        self._update_position_action_buttons()
        self._update_manual_controls()
        self.start_button.configure(state="disabled")
        self.status_var.set(description)

        def worker() -> None:
            command_lock_acquired = False
            try:
                command_lock_acquired = self._servo_command_path_lock.acquire(
                    blocking=False
                )
                if not command_lock_acquired:
                    raise RuntimeError("Another servo command path is active")
                result = action(controller, cancel_event)
            except Exception as exc:
                self._emit(
                    {
                        "kind": "position_action_error",
                        "description": description,
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            else:
                self._emit(
                    {
                        "kind": "position_action_done",
                        "description": description,
                        "result": result,
                    }
                )
            finally:
                if command_lock_acquired:
                    self._servo_command_path_lock.release()

        thread = threading.Thread(
            target=worker,
            name="csv-player-remembered-position",
            daemon=True,
        )
        self._position_action_thread = thread
        try:
            thread.start()
        except Exception:
            self._position_action_pending = False
            self._position_action_cancel_event = None
            self._position_action_thread = None
            self._update_position_action_buttons()
            raise

    def _remember_position(self, servo_id: int) -> None:
        self._start_position_action(
            lambda controller, _cancel: controller.remember(servo_id),
            f"Servo {servo_id}のencoder位置をRemember中",
        )

    def _return_position(self, servo_id: int) -> None:
        self._start_position_action(
            lambda controller, cancel: controller.return_one(
                servo_id, cancel_event=cancel
            ),
            f"Servo {servo_id}をRemember位置へ復帰中",
        )

    def _return_all_positions(self) -> None:
        if not messagebox.askyesno(
            "Return All",
            "すべてのservoを1台ずつRemember位置へ戻しますか？",
        ):
            return
        self._start_position_action(
            lambda controller, cancel: controller.return_all(cancel_event=cancel),
            "すべてのservoをRemember位置へ復帰中",
        )

    def _update_progress(self, elapsed_s: float, phase: str) -> None:
        if phase == "return":
            self.progress_bar.configure(value=100.0)
            self.progress_var.set("進捗: 再生完了 / 初期位置へ復帰中")
            return
        duration_s = self._active_duration_s
        if duration_s is None or duration_s <= 0.0:
            return
        ratio = min(max(elapsed_s / duration_s, 0.0), 1.0)
        remaining_s = max(duration_s - elapsed_s, 0.0)
        self.progress_bar.configure(value=ratio * 100.0)
        self.progress_var.set(
            f"進捗: {ratio * 100.0:.0f}% / 残り: {remaining_s:.1f} s"
        )

    def _start(self) -> None:
        if (
            not self._running
            or not self._connection_ready
            or self._start_event is None
            or self._playback_started
            or self._position_action_pending
            or self._manual_command_pending
        ):
            return
        if self._manual_speed_active():
            messagebox.showwarning(
                "Manual Speed動作中",
                "すべてのmanual speedを0にしてからCSV再生を開始してください。",
            )
            return
        source_kind = self.trajectory_source_var.get()
        path: Path | None = None
        trajectory: ExcursionTrajectory | None = None
        point_to_point_request: PointToPointRequest | None = None
        try:
            limits = self._parse_limits()
            calibration = self._load_encoder_calibration()
            playback_mode = self.playback_mode_var.get().strip()
            decision_hz = float(self.decision_hz_var.get())
            if playback_mode not in {
                INTERPOLATED_PLAYBACK_MODE,
                RL_FF_ZOH_PLAYBACK_MODE,
            }:
                raise ValueError(f"未対応の再生モードです: {playback_mode}")
            if (
                not math.isfinite(decision_hz)
                or decision_hz <= 0.0
                or decision_hz > DEFAULT_CONTROL_HZ
            ):
                raise ValueError(
                    f"Decision Hzは0より大きく{DEFAULT_CONTROL_HZ:g}以下にしてください"
                )
            if source_kind == POINT_TO_POINT_SOURCE:
                goal_deg = tuple(
                    float(self.point_to_point_goal_vars[joint].get())
                    for joint in JOINTS
                )
                duration_s = float(self.point_to_point_duration_var.get())
                moment_arm = _validate_point_to_point_request(
                    tuple(math.radians(value) for value in goal_deg),
                    duration_s,
                    playback_mode=playback_mode,
                    decision_hz=decision_hz,
                    safety_filter_enabled=bool(self.safety_filter_enabled_var.get()),
                    encoder_calibration=calibration,
                )
                point_to_point_request = PointToPointRequest(
                    q_goal_rad=tuple(math.radians(value) for value in goal_deg),
                    duration_s=duration_s,
                    feedback_enabled=bool(self.feedback_enabled_var.get()),
                    moment_arm=moment_arm,
                )
                integrated_runtime = None
            else:
                selection = self.file_list.curselection()
                if not selection:
                    messagebox.showwarning("CSV未選択", "再生するCSVを選択してください")
                    return
                path = self.csv_paths[selection[0]]
                trajectory = ExcursionTrajectory.from_csv(path)
                integrated_runtime = self._build_selected_integrated_runtime(
                    trajectory,
                    limits,
                    playback_mode,
                )
        except Exception as exc:
            messagebox.showerror("再生設定エラー", str(exc))
            return
        duration_s = (
            point_to_point_request.duration_s
            if point_to_point_request is not None
            else trajectory.duration_s
        )
        calibration_warning = (
            "\nencoder換算は未承認です。統合D0/D1では実測校正または明示承認済みの幾何換算が必要です。"
            if not encoder_conversion_is_p2p_approved(calibration)
            else ""
        )
        if point_to_point_request is None:
            assert path is not None and trajectory is not None
            assert integrated_runtime is not None
            confirmation_text = (
                f"{path.name}\n再生時間: {trajectory.duration_s:.3f} s\n"
                f"モード: {playback_mode} / Decision: {decision_hz:g} Hz\n"
                f"FF+FB: {integrated_runtime.feedback_enabled} / "
                f"Safety Filter: {integrated_runtime.safety_filter_enabled}\n"
                f"接続済みです。CSV再生を開始しますか？{calibration_warning}"
            )
        else:
            goal_text = ", ".join(
                f"{joint}={math.degrees(angle):.2f}°"
                for joint, angle in zip(JOINTS, point_to_point_request.q_goal_rad)
            )
            confirmation_text = (
                "実測q0から目標姿勢へのPoint-to-Point制御\n"
                f"目標: {goal_text}\n"
                f"運動時間: {point_to_point_request.duration_s:.3f} s\n"
                f"モード: {playback_mode} / Decision: {decision_hz:g} Hz\n"
                f"FF+FB: {point_to_point_request.feedback_enabled} / "
                "Safety Filter: True\n"
                "実行中はFF軌道を再計算しません。制御を開始しますか？"
            )
        if not messagebox.askyesno("再生開始", confirmation_text):
            return
        if not self._servo_command_path_lock.acquire(blocking=False):
            messagebox.showerror(
                "制御経路使用中",
                "別の制御経路がサーボ指令を使用中です。停止完了後に再試行してください。",
            )
            return
        self._playback_command_lock_held = True
        with self._resource_lock:
            self._pending_playback_config = (
                path,
                trajectory,
                limits,
                calibration,
                playback_mode,
                decision_hz,
                integrated_runtime,
                point_to_point_request,
            )
        self._playback_started = True
        self._manual_feedback_ready = False
        self._manual_feedback_ready_event.clear()
        self._target_angles_visible = True
        for history in self.force_history.values():
            history.clear()
        for history in self.angle_history.values():
            history.clear()
        self._last_angle_frame = -1
        self._force_plot_dirty = True
        self._angle_plot_dirty = True
        self._update_position_action_buttons()
        self._update_manual_controls()
        self._active_duration_s = duration_s
        self.progress_bar.configure(value=0.0)
        self.progress_var.set(f"進捗: 0% / 残り: {duration_s:.1f} s")
        self._set_widgets_enabled(self._csv_widgets, False)
        self.start_button.configure(state="disabled")
        self.status_var.set("再生開始")
        self.phase_var.set("再生開始")
        self._set_status_badge("experiment", "再生開始", "busy")
        self._start_event.set()

    def _connect(self) -> None:
        if self._running or self._start_in_progress:
            return
        self._start_in_progress = True
        self.start_button.configure(state="disabled")
        try:
            manual_limits = self._parse_limits()
        except Exception as exc:
            messagebox.showerror("Force設定エラー", str(exc))
            self._start_in_progress = False
            self.connect_button.configure(state="normal")
            return
        if not messagebox.askyesno(
            "接続確認",
            "カメラとサーボに接続します。\n"
            "接続後にCSVを選択し、再生開始を押すまでサーボは動かしません。接続しますか？",
        ):
            self._start_in_progress = False
            self.connect_button.configure(state="normal")
            return

        if not self._stop_dlc_measurement_worker():
            self._start_in_progress = False
            self.connect_button.configure(state="normal")
            messagebox.showwarning(
                "DLC停止待ち",
                "DLC計測中のカメラを解放できませんでした。少し待ってから再度Connectを押してください。",
            )
            return

        if not self._stop_camera_preview_worker():
            self._start_in_progress = False
            self.connect_button.configure(state="normal")
            messagebox.showwarning(
                "カメラ切替待ち",
                "プレビュー用カメラを解放できませんでした。少し待ってから再度Connectを押してください。",
            )
            return

        # Tk variables belong to the main thread.  Snapshot all run settings
        # before starting the worker instead of reading widgets from it.
        dlc_config_path = Path(self.dlc_config_var.get())
        camera_config_path = Path(self.camera_config_var.get())
        servo_port = self.servo_port_var.get()
        encoder_port = self.encoder_port_var.get()
        session_dir = DEFAULT_OUTPUT_DIRECTORY / datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            session_dir.mkdir(parents=True, exist_ok=False)
        except Exception as exc:
            messagebox.showerror("セッション作成エラー", str(exc))
            self._start_in_progress = False
            self.connect_button.configure(state="normal")
            if self._preview_enabled:
                self._start_camera_preview_worker()
            return
        self._session_dir = session_dir
        stop_event = threading.Event()
        start_event = threading.Event()
        force_quit_event = threading.Event()
        with self._resource_lock:
            self._pending_playback_config = None
        self._stop_event = stop_event
        self._start_event = start_event
        self._force_quit_event = force_quit_event
        self._running = True
        self._connection_ready = False
        self._playback_started = False
        self._manual_feedback_ready = False
        self._manual_feedback_ready_event.clear()
        self._latest_manual_sample = None
        self._target_angles_visible = False
        with self._manual_speed_lock:
            self._manual_speed_commands = {servo_id: 0 for servo_id in SERVO_IDS}
        self._active_duration_s = None
        self.progress_bar.configure(value=0.0)
        self.progress_var.set("進捗: CSV選択待ち")
        self._set_input_widgets_enabled(False)
        self._set_widgets_enabled(self._csv_widgets, True)
        self._last_angle_frame = -1
        self._last_preview_frame = -1
        self._last_preview_is_overlay = None
        self._min_likelihood_seen = 1.0
        self._latest_events.clear()
        for history in self.force_history.values():
            history.clear()
        for history in self.angle_history.values():
            history.clear()
        self._force_plot_dirty = True
        self._angle_plot_dirty = True
        self.alert_label.configure(text="")
        self.status_var.set("準備中")
        self.phase_var.set("準備中")
        self._set_status_badge("experiment", "準備中", "busy")
        self._set_status_badge("camera", "接続中", "busy")
        self._set_status_badge("dlc", "準備中", "busy")
        self._set_status_badge("servo", "接続待ち", "busy")
        self._set_status_badge("encoder", "接続待ち", "busy")
        self.connect_button.configure(state="disabled")
        self.dlc_toggle_button.configure(state="disabled", text="DLC（session）")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.force_quit_button.configure(state="normal")
        self._runner_thread = threading.Thread(
            target=self._run_session,
            args=(
                None,
                None,
                manual_limits,
                session_dir,
                dlc_config_path,
                camera_config_path,
                servo_port,
                encoder_port,
                stop_event,
                start_event,
                force_quit_event,
            ),
            name="csv-player-gui-session",
            daemon=True,
        )
        try:
            self._runner_thread.start()
        except Exception as exc:
            self._running = False
            self._stop_event = None
            self._start_event = None
            self._force_quit_event = None
            with self._resource_lock:
                self._pending_playback_config = None
            self._connection_ready = False
            self._playback_started = False
            self._start_in_progress = False
            self._set_input_widgets_enabled(True)
            self.dlc_toggle_button.configure(state="normal", text="DLC ON")
            self.stop_button.configure(state="disabled")
            self.force_quit_button.configure(state="disabled")
            messagebox.showerror("実行開始エラー", str(exc))
            return
        self._start_in_progress = False

    def _run_session(
        self,
        csv_path: Path | None,
        trajectory: ExcursionTrajectory | None,
        limits: dict[str, float] | None,
        session_dir: Path,
        dlc_config_path: Path,
        camera_config_path: Path,
        servo_port: str,
        encoder_port: str,
        stop_event: threading.Event,
        start_event: threading.Event,
        force_quit_event: threading.Event,
    ) -> None:
        hardware: ServoHardwareSession | None = None
        source: RealtimeJointAngleSource | None = None
        playback_service: EncoderPlaybackService | None = None
        raw_event_logger: AsyncRawEventLogger | None = None
        raw_logger_stats: RawLoggerStats | None = None
        vision_worker: VisionWorker | None = None
        vision_thread: threading.Thread | None = None
        playback_started_at: list[float | None] = [None]
        playback_timeline: list[tuple[float, ...] | None] = [None]
        playback_clock: deque[tuple[float, float, bool]] = deque(maxlen=1024)
        playback_clock_lock = threading.Lock()
        latest_joint_state = LatestJointState()
        run_started_at: float | None = None
        encoder_diagnostics_path = session_dir / "encoder_diagnostics.json"
        encoder_diagnostics_written = False
        error_traceback: str | None = None
        termination_monotonic_s: float | None = None
        completed = False
        selected_playback_mode: str | None = None
        selected_decision_hz: float | None = None
        selected_integrated_runtime: IntegratedControlRuntime | None = None
        point_to_point_metadata: dict[str, Any] | None = None
        termination_reason = "error"
        error_text: str | None = None
        self._emit({"kind": "status", "text": "カメラとDLCを準備中", "phase": "準備中"})
        try:
            raw_event_logger = AsyncRawEventLogger(
                session_dir / "raw_events.jsonl",
                run_id=session_dir.name,
            )
            raw_event_logger.log(
                "gui_session_created",
                payload={
                    "wall_clock_created_at": datetime.now().isoformat(
                        timespec="microseconds"
                    ),
                    "timestamp_timebase": "time.monotonic seconds",
                },
            )
            source = RealtimeJointAngleSource(
                dlc_config_path=dlc_config_path,
                camera_config_path=camera_config_path,
                min_likelihood=0.3,
                temporal_outlier_enabled=True,
                temporal_jump_threshold_deg=30.0,
                temporal_max_hold_frames=5,
            )
            source.open()
            with self._resource_lock:
                self._angle_source = source
            if stop_event.is_set():
                raise PlaybackStopped("Playback was stopped during camera setup")
            vision_worker = VisionWorker(
                source,
                self._emit,
                stop_event,
                session_dir / "angle_trace.csv",
                playback_started_at,
                playback_timeline,
                playback_clock,
                playback_clock_lock,
                latest_joint_state,
                raw_event_logger=raw_event_logger,
                run_id=session_dir.name,
            )
            vision_thread = threading.Thread(
                target=vision_worker.run,
                name="csv-player-gui-dlc",
                daemon=True,
            )
            vision_thread.start()
            if stop_event.is_set():
                raise PlaybackStopped("Playback was stopped during camera setup")
            self._emit({"kind": "status", "text": "サーボとエンコーダを接続中", "phase": "接続"})
            hardware = ServoHardwareSession(
                servo_ids=SERVO_IDS,
                servo_names=SERVO_NAMES_BY_ID,
                encoder_sensor_ids=tuple(
                    ENCODER_SENSOR_IDS_BY_SERVO_ID[servo_id] for servo_id in SERVO_IDS
                ),
                servo_port=servo_port,
                encoder_port=encoder_port,
                baudrate=921600,
                serial_timeout_s=0.05,
                encoder_rate_hz=35.0,
                capture_encoder_diagnostics=True,
                batch_encoder_reads=True,
                encoder_batch_max_lines=128,
            )
            with self._resource_lock:
                self._hardware = hardware
            if stop_event.is_set():
                raise PlaybackStopped("Playback was stopped before hardware connection")
            hardware.connect(startup_timeout_s=3.0)
            if stop_event.is_set():
                raise PlaybackStopped("Playback was stopped during hardware connection")
            feedback = hardware.wait_for_feedback(3.0, require_encoder=True)
            if stop_event.is_set():
                raise PlaybackStopped("Playback was stopped while waiting for telemetry")
            self._emit(
                {
                    "kind": "connected",
                    "text": "接続完了。再生開始を押してください",
                    "phase": "接続完了",
                }
            )
            self._monitor_before_playback(
                hardware=hardware,
                stop_event=stop_event,
                start_event=start_event,
                hard_force_limits_n_by_tendon=dict(limits or {}),
            )
            if stop_event.is_set():
                raise PlaybackStopped("Playback was stopped before playback started")
            with self._resource_lock:
                playback_config = self._pending_playback_config
            if playback_config is None:
                raise RuntimeError("Playback configuration was not selected")
            (
                csv_path,
                trajectory,
                limits,
                calibration,
                selected_playback_mode,
                selected_decision_hz,
                selected_integrated_runtime,
                point_to_point_request,
            ) = playback_config
            if point_to_point_request is not None:
                if hardware is None or raw_event_logger is None:
                    raise RuntimeError("P2P session resources are unavailable")
                config = load_config(
                    DEFAULT_FEEDBACK_CONFIG_PATH,
                    required_keys=("paths", "gui_integration"),
                )
                gui_config = config["gui_integration"]
                if not isinstance(gui_config, Mapping):
                    raise ValueError("gui_integration must be an object")
                encoder_start_tolerance_counts = (
                    _configured_encoder_start_tolerance_counts(gui_config)
                )
                q0_snapshot, q0_samples, q0_median, q0_spread_deg = (
                    _collect_stable_initial_dlc(
                        latest_joint_state,
                        max_age_s=float(gui_config["max_joint_age_s"]),
                        stop_event=stop_event,
                    )
                )
                encoder_origin_snapshot = _validate_point_to_point_encoder_snapshot(
                    hardware,
                    q0_snapshot=q0_snapshot,
                    max_servo_age_s=float(gui_config["max_servo_age_s"]),
                    max_encoder_age_s=float(gui_config["max_encoder_age_s"]),
                    max_joint_age_s=float(gui_config["max_joint_age_s"]),
                )
                moment_arm = point_to_point_request.moment_arm
                _connect_moment_arm_extrapolation_logger(
                    moment_arm,
                    raw_event_logger,
                )
                trajectory = ExcursionTrajectory.from_joint_point_to_point(
                    q0_snapshot.angles_rad,
                    point_to_point_request.q_goal_rad,
                    point_to_point_request.duration_s,
                    moment_arm,
                    sample_hz=DEFAULT_CONTROL_HZ,
                )
                selected_integrated_runtime = _build_integrated_control_runtime(
                    trajectory,
                    limits,
                    feedback_enabled=point_to_point_request.feedback_enabled,
                    safety_filter_enabled=True,
                    moment_arm_override=moment_arm,
                )
                if selected_integrated_runtime.action_safety_filter is None:
                    raise ValueError("Point-to-Point requires an active Safety Filter")
                safety_config = selected_integrated_runtime.action_safety_filter.config
                max_ff_decision_delta = validate_point_to_point_trajectory(
                    trajectory,
                    approved_joint_limits_rad=safety_config.joint_angle_limits_rad or {},
                    min_action_mm_by_tendon=safety_config.min_action_mm_by_tendon,
                    max_action_mm_by_tendon=safety_config.max_action_mm_by_tendon,
                    max_action_delta_mm_by_tendon=safety_config.max_action_delta_mm_by_tendon,
                    decision_hz=selected_decision_hz,
                )
                reference_path = trajectory.to_csv(
                    session_dir / "reference_trajectory.csv"
                )
                csv_path = reference_path
                goal_deg = tuple(
                    math.degrees(value)
                    for value in point_to_point_request.q_goal_rad
                )
                point_to_point_metadata = {
                    "mode": "point_to_point_moment_arm_ff",
                    "control_variant": (
                        "ff_fb"
                        if point_to_point_request.feedback_enabled
                        else "ff"
                    ),
                    "profile": "minimum_jerk",
                    "q0_estimator": "latest_of_5_stable_frames",
                    "q0_rad": dict(zip(JOINTS, q0_snapshot.angles_rad)),
                    "q0_median_rad": dict(zip(JOINTS, q0_median)),
                    "q0_sample_spread_deg": dict(zip(JOINTS, q0_spread_deg)),
                    "q0_samples": [
                        {
                            "captured_at": item.captured_at,
                            "inference_finished_at": item.inference_finished_at,
                            "angles_rad": dict(zip(JOINTS, item.angles_rad)),
                        }
                        for item in q0_samples
                    ],
                    "q0_capture_monotonic_s": q0_snapshot.captured_at,
                    "q_goal_rad": dict(zip(JOINTS, point_to_point_request.q_goal_rad)),
                    "q_goal_deg": dict(zip(JOINTS, goal_deg)),
                    "duration_s": point_to_point_request.duration_s,
                    "sample_hz": DEFAULT_CONTROL_HZ,
                    "decision_hz": selected_decision_hz,
                    "encoder_origin_snapshot": encoder_origin_snapshot,
                    "moment_arm_model": point_to_point_request.moment_arm.model_info,
                    "moment_arm_extrapolation_policy": "constant_endpoint_outside_fit_range",
                    "max_ff_decision_delta_mm": max_ff_decision_delta,
                    "reference_csv_path": str(reference_path),
                    "ff_recomputed_during_run": False,
                    "initial_tension_recorded_in": "control_session_start event",
                }
                raw_event_logger.log(
                    "point_to_point_reference",
                    event_monotonic_s=time.monotonic(),
                    payload=point_to_point_metadata,
                )
                self._emit(
                    {"kind": "point_to_point_trajectory", "trajectory": trajectory}
                )
            if trajectory is None or selected_integrated_runtime is None:
                raise RuntimeError("Playback trajectory or runtime is unavailable")
            playback_timeline[0] = trajectory.recorded_elapsed_times_s

            feedback_action_provider = _make_feedback_action_provider(
                selected_integrated_runtime,
                trajectory,
                latest_joint_state,
                force_quit_event,
                raw_event_logger,
                decision_hz=(
                    selected_decision_hz
                    if point_to_point_request is not None
                    and selected_integrated_runtime.feedback_enabled
                    else None
                ),
            )
            joint_safety_state_provider = (
                None
                if not selected_integrated_runtime.safety_filter_enabled
                or selected_integrated_runtime.max_joint_age_s is None
                else lambda: latest_joint_state.safety_state(
                    selected_integrated_runtime.max_joint_age_s
                )
            )
            raw_event_logger.log(
                "integrated_control_configured",
                payload=dict(selected_integrated_runtime.config_summary),
            )

            def record_playback_clock(
                monotonic_s: float,
                source_elapsed_s: float,
                finished: bool,
            ) -> None:
                with playback_clock_lock:
                    playback_clock.append(
                        (monotonic_s, source_elapsed_s, finished)
                    )

            playback_config = build_encoder_playback_config(
                calibration,
                control_hz=DEFAULT_CONTROL_HZ,
                telemetry_stale_s=0.25,
                telemetry_wait_s=3.0,
                display_interval_s=0.25,
                return_to_start=(
                    not selected_integrated_runtime.active
                    or selected_integrated_runtime.automatic_return_after_safety_stop
                ),
                return_timeout_s=15.0,
                return_stable_frames=3,
                hard_force_limits_n_by_tendon=limits,
                playback_mode=selected_playback_mode,
                decision_hz=selected_decision_hz,
            )
            p2p_start_state_validator = None
            if point_to_point_metadata is not None:
                if selected_integrated_runtime.max_joint_age_s is None:
                    raise ValueError("P2P requires a configured fresh-DLC limit")

                def validate_p2p_runner_start_state(
                    start_feedback: HardwareFeedback,
                    runner_origin: Mapping[int, float],
                ) -> Mapping[str, Any]:
                    return _validate_and_log_point_to_point_start_state(
                        hardware,
                        latest_joint_state=latest_joint_state,
                        q0_snapshot=q0_snapshot,
                        q0_spread_deg=q0_spread_deg,
                        encoder_origin_snapshot=encoder_origin_snapshot,
                        feedback=start_feedback,
                        runner_encoder_origin_deg_by_servo_id=runner_origin,
                        encoder_directions=calibration.encoder_directions,
                        max_joint_age_s=(
                            selected_integrated_runtime.max_joint_age_s
                        ),
                        raw_event_logger=raw_event_logger,
                        encoder_start_tolerance_counts=(
                            encoder_start_tolerance_counts
                        ),
                    )

                p2p_start_state_validator = validate_p2p_runner_start_state
            playback_service = EncoderPlaybackService(
                trajectory=trajectory,
                calibration=calibration,
                trace_path=session_dir / "control_trace.csv",
                run_id=session_dir.name,
                force_calibration_dir=DEFAULT_FORCE_CALIBRATION_DIR,
                zero_offset_path=DEFAULT_ZERO_OFFSET_PATH,
                config=playback_config,
                record_force=True,
                on_sample=lambda sample: self._emit({"kind": "force", "sample": sample}),
                on_playback_clock=record_playback_clock,
                stop_event=stop_event,
                force_quit_event=force_quit_event,
                raw_event_logger=raw_event_logger,
                action_safety_filter=(
                    selected_integrated_runtime.action_safety_filter
                ),
                start_state_validator=p2p_start_state_validator,
                feedback_action_provider=feedback_action_provider,
                joint_safety_state_provider=joint_safety_state_provider,
            )

            run_started_at = time.monotonic()
            video_path = session_dir / "dlc_overlay.mp4"
            source.start_overlay_recording(video_path, fps=DEFAULT_VISION_HZ, codec="mp4v", queue_size=10)
            playback_started_at[0] = run_started_at
            self._emit({"kind": "status", "text": "実行中", "phase": "実行中"})

            playback_service.run(hardware)
            completed = True
            termination_reason = "completed"
            self._emit({"kind": "status", "text": "完了", "phase": "完了"})
        except (ActionSafetyError, ActionSafetyContextError) as exc:
            termination_reason = "safety_filter_rejection"
            error_text = str(exc)
            force_quit_event.set()
            stop_event.set()
            if hardware is not None:
                self._request_latched_hardware_stop(hardware)
            self._emit(
                {
                    "kind": "hard_limit",
                    "text": f"SAFETY FILTER STOP: {exc}",
                    "phase": "Safety停止",
                }
            )
        except DLCFeedbackUnavailable as exc:
            termination_reason = "dlc_feedback_unavailable"
            error_text = str(exc)
            force_quit_event.set()
            stop_event.set()
            if hardware is not None:
                self._request_latched_hardware_stop(hardware)
            self._emit(
                {
                    "kind": "hard_limit",
                    "text": f"DLC FEEDBACK STOP: {exc}",
                    "phase": "DLC停止",
                }
            )
        except TimeoutError as exc:
            termination_reason = "communication_timeout"
            error_text = str(exc)
            force_quit_event.set()
            stop_event.set()
            if hardware is not None:
                self._request_latched_hardware_stop(hardware)
            self._emit(
                {
                    "kind": "hard_limit",
                    "text": f"COMMUNICATION TIMEOUT: {exc}",
                    "phase": "通信停止",
                }
            )
        except HardForceLimitError as exc:
            termination_reason = "force_hard_limit"
            error_text = str(exc)
            stop_event.set()
            self._emit(
                {
                    "kind": "hard_limit",
                    "text": (
                        f"FORCE HARD LIMIT: {exc.tendon} "
                        f"{exc.force_n:.3f} N / {exc.limit_n:.3f} N"
                    ),
                    "phase": "ハードリミット停止",
                }
            )
        except ForceTelemetryUnavailable as exc:
            termination_reason = "force_telemetry_unavailable"
            error_text = str(exc)
            stop_event.set()
            self._emit(
                {
                    "kind": "hard_limit",
                    "text": f"FORCE SAFETY STOP: {exc}",
                    "phase": "Force停止",
                }
            )
        except PlaybackStopped as exc:
            termination_reason = "force_quit" if force_quit_event.is_set() else "operator_stop"
            error_text = str(exc)
            self._emit(
                {
                    "kind": "status",
                    "text": "Force Quitで停止しました"
                    if termination_reason == "force_quit"
                    else "安全停止しました",
                    "phase": "Force Quit"
                    if termination_reason == "force_quit"
                    else "停止",
                }
            )
        except Exception as exc:
            termination_reason = "error"
            error_text = f"{type(exc).__name__}: {exc}"
            error_traceback = traceback.format_exc()
            self._emit({"kind": "worker_error", "source": "実行", "error": error_text})
        finally:
            termination_monotonic_s = time.monotonic()
            stop_event.set()
            if self._position_action_cancel_event is not None:
                self._position_action_cancel_event.set()
            cleanup_errors: list[str] = []
            stop_state: Mapping[str, Any] | None = None
            if hardware is not None:
                try:
                    hardware.stop_servos()
                except Exception as exc:
                    cleanup_errors.append(f"servo stop: {type(exc).__name__}: {exc}")
            if vision_thread is not None:
                # The source is shared with the vision worker.  Wait for the
                # reader to leave source.read() before closing the camera or
                # the overlay recorder from this worker.
                vision_thread.join()
            if source is not None:
                try:
                    source.stop_overlay_recording()
                except Exception as exc:
                    cleanup_errors.append(
                        f"overlay recording: {type(exc).__name__}: {exc}"
                    )
                try:
                    source.close()
                except Exception as exc:
                    cleanup_errors.append(f"camera close: {type(exc).__name__}: {exc}")
            if playback_service is not None:
                try:
                    playback_service.close()
                except Exception as exc:
                    cleanup_errors.append(
                        f"playback service close: {type(exc).__name__}: {exc}"
                    )
            if hardware is not None:
                try:
                    hardware.close()
                except Exception as exc:
                    cleanup_errors.append(f"hardware close: {type(exc).__name__}: {exc}")
                try:
                    stop_state = hardware.stop_state_snapshot()
                except Exception as exc:
                    cleanup_errors.append(
                        f"stop state: {type(exc).__name__}: {exc}"
                    )
                diagnostics = hardware.encoder_diagnostics_snapshot()
                if diagnostics is not None:
                    try:
                        encoder_diagnostics_path.write_text(
                            json.dumps(diagnostics, indent=2, ensure_ascii=False),
                            encoding="utf-8",
                        )
                        encoder_diagnostics_written = True
                    except Exception as exc:
                        cleanup_errors.append(
                            f"encoder diagnostics: {type(exc).__name__}: {exc}"
                        )
            if cleanup_errors:
                cleanup_text = "; ".join(cleanup_errors)
                error_text = (
                    f"{error_text}; cleanup: {cleanup_text}"
                    if error_text
                    else f"cleanup: {cleanup_text}"
                )
                if termination_reason in {"completed", "operator_stop"}:
                    termination_reason = "error"
                self._emit(
                    {
                        "kind": "worker_error",
                        "source": "cleanup",
                        "error": cleanup_text,
                    }
                )
            if raw_event_logger is not None:
                if stop_state is not None:
                    raw_event_logger.log(
                        "stop_state",
                        event_monotonic_s=termination_monotonic_s,
                        payload=dict(stop_state),
                    )
                raw_event_logger.log(
                    "session_end",
                    event_monotonic_s=termination_monotonic_s,
                    payload={
                        "termination_reason": termination_reason,
                        "completed_playback": completed,
                        "error": error_text,
                        "rl_boundary_classification": "deferred_to_dataset_processor",
                        "stop_state": stop_state,
                    },
                )
                raw_logger_stats = raw_event_logger.close()
                try:
                    (session_dir / "raw_event_stats.json").write_text(
                        json.dumps(
                            {
                                **raw_logger_stats.as_dict(),
                                "first_overflow_monotonic_s": (
                                    raw_event_logger.first_overflow_monotonic_s
                                ),
                            },
                            indent=2,
                            ensure_ascii=False,
                        ),
                        encoding="utf-8",
                    )
                except Exception as exc:
                    cleanup_errors.append(
                        f"raw event stats: {type(exc).__name__}: {exc}"
                    )
            manifest = {
                "run_id": session_dir.name,
                "timestamp_timebase": "time.monotonic seconds; shared by control and vision logs within this run",
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "input_csv_path": (
                    str(csv_path.resolve()) if csv_path is not None else None
                ),
                "session_dir": str(session_dir.resolve()),
                "hard_force_limits_n_by_tendon": limits,
                "encoder_calibration_path": str(
                    DEFAULT_ENCODER_CALIBRATION_PATH.resolve()
                ),
                "force_calibration_directory": str(
                    DEFAULT_FORCE_CALIBRATION_DIR.resolve()
                ),
                "force_zero_offset_path": str(DEFAULT_ZERO_OFFSET_PATH.resolve()),
                "playback_mode": selected_playback_mode,
                "control_hz": DEFAULT_CONTROL_HZ,
                "decision_hz": selected_decision_hz,
                "integrated_control": (
                    {"active": False}
                    if selected_integrated_runtime is None
                    else dict(selected_integrated_runtime.config_summary)
                ),
                "automatic_return_enabled": (
                    True
                    if selected_integrated_runtime is None
                    else (
                        not selected_integrated_runtime.active
                        or selected_integrated_runtime.automatic_return_after_safety_stop
                    )
                ),
                "control_target": point_to_point_metadata,
                "servo_port": servo_port,
                "encoder_port": encoder_port,
                "completed": completed,
                "termination_reason": termination_reason,
                "error": error_text,
                "error_traceback": error_traceback,
                "termination_monotonic_s": termination_monotonic_s,
                "rl_boundary_classification": (
                    "deferred_to_dataset_processor"
                ),
                "raw_logger_stats": (
                    None
                    if raw_logger_stats is None
                    else raw_logger_stats.as_dict()
                ),
                "outputs": {
                    "control_trace": str((session_dir / "control_trace.csv").resolve()),
                    "angle_trace": str((session_dir / "angle_trace.csv").resolve()),
                    "reference_trajectory": (
                        str(csv_path.resolve())
                        if point_to_point_metadata is not None and csv_path is not None
                        else None
                    ),
                    "raw_events": str((session_dir / "raw_events.jsonl").resolve()),
                    "raw_event_stats": str(
                        (session_dir / "raw_event_stats.json").resolve()
                    ),
                    "overlay_video": str((session_dir / "dlc_overlay.mp4").resolve()),
                    "encoder_diagnostics": (
                        str(encoder_diagnostics_path.resolve())
                        if encoder_diagnostics_written
                        else None
                    ),
                },
            }
            try:
                (session_dir / "session_manifest.json").write_text(
                    json.dumps(manifest, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception as exc:
                error_text = (
                    f"{error_text}; manifest: {type(exc).__name__}: {exc}"
                    if error_text
                    else f"manifest: {type(exc).__name__}: {exc}"
                )
                termination_reason = "error"
                self._emit(
                    {
                        "kind": "worker_error",
                        "source": "manifest",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            with self._resource_lock:
                self._angle_source = None
                self._hardware = None
            if self._playback_command_lock_held:
                self._playback_command_lock_held = False
                try:
                    self._servo_command_path_lock.release()
                except RuntimeError:
                    pass
            self._emit(
                {
                    "kind": "result",
                    "status": termination_reason,
                    "session_dir": str(session_dir),
                    "error": error_text,
                }
            )

    def _monitor_before_playback(
        self,
        *,
        hardware: ServoHardwareSession,
        stop_event: threading.Event,
        start_event: threading.Event,
        hard_force_limits_n_by_tendon: dict[str, float],
    ) -> None:
        force_estimator = CalibratedForceEstimator(
            calibration_dir=DEFAULT_FORCE_CALIBRATION_DIR,
            zero_offset_path=DEFAULT_ZERO_OFFSET_PATH,
            servo_ids=SERVO_IDS,
            tendon_by_servo_id=dict(zip(SERVO_IDS_BY_TENDON, TENDONS)),
            telemetry_stale_after_s=0.25,
        )
        calibration = load_encoder_excursion_calibration(
            DEFAULT_ENCODER_CALIBRATION_PATH
        )
        direction_by_servo_id = {
            servo_id: calibration.encoder_directions[index]
            for index, servo_id in enumerate(SERVO_IDS_BY_TENDON)
        }
        servo_index_by_id = {
            servo_id: hardware.system.index_for_servo_id(servo_id)
            for servo_id in hardware.system.servo_ids
        }
        monitoring_started_at = time.monotonic()
        next_display_at = monitoring_started_at

        while not start_event.is_set():
            if stop_event.is_set():
                raise PlaybackStopped("Playback was stopped before playback started")
            hardware.check_health()
            now = time.monotonic()
            feedback = hardware.latest_feedback(0.25)
            commands = self._manual_speeds_snapshot()
            manual_active = any(commands.values())
            force_readings: dict[int, dict[str, Any]] = {}
            positions: dict[int, float] = {}
            encoder_errors: dict[int, str] = {}

            if feedback is None:
                for tendon, servo_id in zip(TENDONS, SERVO_IDS_BY_TENDON):
                    force_readings[servo_id] = {
                        "servo_id": servo_id,
                        "tendon": tendon,
                        "force_n": None,
                        "status": "STALE_SERVO_TELEMETRY",
                    }
            else:
                force_readings = force_estimator.estimate(
                    feedback, servo_index_by_id
                )
                for servo_id in SERVO_IDS_BY_TENDON:
                    sensor_id = ENCODER_SENSOR_IDS_BY_SERVO_ID[servo_id]
                    sample = feedback.encoder_by_sensor.get(sensor_id)
                    encoder_age_s = feedback.encoder_age_s(sensor_id)
                    if sample is None or encoder_age_s is None or encoder_age_s > 0.25:
                        encoder_errors[servo_id] = "STALE_ENCODER_TELEMETRY"
                        continue
                    status = str(sample.get("status", ""))
                    if status not in {"OK", "MULTITURN_UNCERTAIN"}:
                        encoder_errors[servo_id] = f"ENCODER_STATUS_{status or 'MISSING'}"
                        continue
                    try:
                        relative_deg = float(sample["relative_deg"])
                    except (KeyError, TypeError, ValueError):
                        encoder_errors[servo_id] = "INVALID_ENCODER_POSITION"
                        continue
                    if not math.isfinite(relative_deg):
                        encoder_errors[servo_id] = "INVALID_ENCODER_POSITION"
                        continue
                    positions[servo_id] = (
                        direction_by_servo_id[servo_id] * (relative_deg % 360.0)
                    )
                for servo_id, encoder_error in encoder_errors.items():
                    reading = dict(force_readings.get(servo_id, {}))
                    reading["status"] = encoder_error
                    force_readings[servo_id] = reading

            telemetry_ready = feedback is not None and not encoder_errors
            unavailable: tuple[str, int, str] | None = None
            for tendon, servo_id in zip(TENDONS, SERVO_IDS_BY_TENDON):
                reading = force_readings.get(servo_id, {})
                force_value = reading.get("force_n")
                if force_value is None:
                    telemetry_ready = False
                    if unavailable is None:
                        unavailable = (
                            tendon,
                            servo_id,
                            str(reading.get("status", "MISSING")),
                        )
                    continue
                force_n = float(force_value)
                if not math.isfinite(force_n):
                    telemetry_ready = False
                    if unavailable is None:
                        unavailable = (tendon, servo_id, "NONFINITE_FORCE")
                    continue
                limit_n = hard_force_limits_n_by_tendon.get(tendon)
                if limit_n is not None and abs(force_n) >= float(limit_n):
                    try:
                        hardware.stop_servos()
                    finally:
                        with self._manual_speed_lock:
                            self._manual_speed_commands = {
                                item: 0 for item in SERVO_IDS
                            }
                        raise HardForceLimitError(
                            tendon=tendon,
                            servo_id=servo_id,
                            force_n=force_n,
                            limit_n=float(limit_n),
                            elapsed_s=now - monitoring_started_at,
                        )

            if encoder_errors and unavailable is None:
                first_servo_id = next(iter(encoder_errors))
                first_tendon = TENDONS[
                    SERVO_IDS_BY_TENDON.index(first_servo_id)
                ]
                unavailable = (
                    first_tendon,
                    first_servo_id,
                    encoder_errors[first_servo_id],
                )

            if telemetry_ready:
                self._manual_feedback_ready_event.set()
            else:
                self._manual_feedback_ready_event.clear()

            if manual_active and not telemetry_ready:
                try:
                    hardware.stop_servos()
                finally:
                    with self._manual_speed_lock:
                        self._manual_speed_commands = {
                            item: 0 for item in SERVO_IDS
                        }
                    tendon, servo_id, status = unavailable or (
                        TENDONS[0],
                        SERVO_IDS_BY_TENDON[0],
                        "UNAVAILABLE",
                    )
                self._emit(
                    {
                        "kind": "manual_safety_stop",
                        "text": (
                            f"Manual speedを停止しました: {tendon} / Servo "
                            f"{servo_id} ({status})。テレメトリ復帰を待っています。"
                        ),
                    }
                )

            if now >= next_display_at:
                self._emit(
                    {
                        "kind": "force",
                        "sample": {
                            "phase": "manual",
                            "elapsed_s": now - monitoring_started_at,
                            "dt_s": MANUAL_DISPLAY_INTERVAL_S,
                            "actual_positions": positions,
                            "speeds": commands,
                            "force_readings": force_readings,
                            "manual_ready": telemetry_ready,
                            "received_at": now,
                        },
                    }
                )
                next_display_at = now + MANUAL_DISPLAY_INTERVAL_S
            start_event.wait(MANUAL_MONITOR_INTERVAL_S)

        if self._manual_speed_active():
            try:
                hardware.stop_servos()
            finally:
                with self._manual_speed_lock:
                    self._manual_speed_commands = {
                        servo_id: 0 for servo_id in SERVO_IDS
                    }
            raise RuntimeError(
                "Manual speed commands must be zero before CSV playback"
            )
        hardware.stop_servos()
        self._manual_feedback_ready_event.clear()

    def _emit(self, event: dict[str, Any]) -> None:
        if event.get("kind") in LATEST_EVENT_KINDS:
            self._latest_events.publish(event)
            return
        events = (
            self._critical_events
            if event.get("kind") in CRITICAL_EVENT_KINDS
            else self.events
        )
        try:
            events.put_nowait(event)
        except queue.Full:
            try:
                events.get_nowait()
            except queue.Empty:
                return
            try:
                events.put_nowait(event)
            except queue.Full:
                pass

    def _request_stop(self) -> None:
        if not self._running or self._stop_event is None:
            return
        if self._position_action_cancel_event is not None:
            self._position_action_cancel_event.set()
        self._connection_ready = False
        self._manual_feedback_ready = False
        self._manual_feedback_ready_event.clear()
        self._update_manual_controls()
        self.status_var.set("停止処理中")
        self.phase_var.set("停止処理中")
        self._set_status_badge("experiment", "停止処理中", "warn")
        self._stop_event.set()
        self.stop_button.configure(state="disabled")

    def _force_quit(self) -> None:
        if not self._running or self._stop_event is None:
            return
        force_quit_event = self._force_quit_event
        if force_quit_event is None or force_quit_event.is_set():
            return
        force_quit_event.set()
        self._connection_ready = False
        self._manual_feedback_ready = False
        self._manual_feedback_ready_event.clear()
        self._update_manual_controls()
        if self._position_action_cancel_event is not None:
            self._position_action_cancel_event.set()
        self._stop_event.set()
        self.status_var.set("Force Quit処理中")
        self.phase_var.set("Force Quit")
        self._set_status_badge("experiment", "Force Quit", "error")
        self.stop_button.configure(state="disabled")
        self.force_quit_button.configure(state="disabled")
        with self._resource_lock:
            hardware = self._hardware
        if hardware is not None:
            threading.Thread(
                target=self._force_stop_hardware,
                args=(hardware,),
                name="csv-player-force-quit",
                daemon=True,
            ).start()

    @staticmethod
    def _request_latched_hardware_stop(hardware: ServoHardwareSession) -> None:
        try:
            hardware.request_latched_stop()
        except Exception:
            # A write already in progress cannot be cancelled by the PC.  The
            # worker records the resulting transport status; do not claim a
            # physical stop or attempt an automatic unload here.
            pass

    @staticmethod
    def _force_stop_hardware(hardware: ServoHardwareSession) -> None:
        CsvPlayerApp._request_latched_hardware_stop(hardware)

    def _schedule_event_poll(self) -> None:
        if self._poll_after_id is not None:
            return
        try:
            self._poll_after_id = self.after(100, self._poll_events)
        except tk.TclError:
            self._poll_after_id = None

    def _poll_events(self) -> None:
        self._poll_after_id = None
        for _ in range(MAX_EVENTS_PER_POLL):
            try:
                event = self._critical_events.get_nowait()
            except queue.Empty:
                try:
                    event = self.events.get_nowait()
                except queue.Empty:
                    break
            try:
                kind = event.get("kind")
                if kind == "status":
                    self._apply_status_event(
                        str(event.get("text", "")), str(event.get("phase", ""))
                    )
                elif kind == "connected":
                    self._connection_ready = True
                    try:
                        calibration = self._load_encoder_calibration()
                        directions_by_servo_id = {
                            servo_id: calibration.encoder_directions[index]
                            for index, servo_id in enumerate(SERVO_IDS_BY_TENDON)
                        }
                        with self._resource_lock:
                            hardware = self._hardware
                        if hardware is None:
                            raise RuntimeError("Hardware session is unavailable")
                        self._position_controller = RememberedPositionController(
                            hardware=hardware,
                            servo_ids=tuple(SERVO_IDS),
                            encoder_sensor_ids_by_servo_id=ENCODER_SENSOR_IDS_BY_SERVO_ID,
                            encoder_directions_by_servo_id=directions_by_servo_id,
                        )
                    except Exception as exc:
                        self._position_controller = None
                        self.status_var.set(
                            f"接続済み（Remember位置機能は使用不可: {exc}）"
                        )
                    self._refresh_remembered_position_display()
                    self.status_var.set(str(event.get("text", "接続完了")))
                    self.phase_var.set(str(event.get("phase", "接続完了")))
                    self._set_status_badge("servo", "接続済み", "ok")
                    self._set_status_badge("encoder", "接続済み", "ok")
                    self._set_status_badge("camera", "接続済み", "ok")
                    self._set_status_badge("dlc", "準備完了", "ok")
                    self._set_status_badge("experiment", "接続完了", "ok")
                    if not self._preview_enabled:
                        self.preview_label.configure(image="", text="プレビューOFF")
                        self._photo = None
                    else:
                        with self._resource_lock:
                            source = self._angle_source
                        if source is None or source.latest_overlay is None:
                            self.preview_label.configure(
                                image="", text="カメラ映像を取得中"
                            )
                    self.connect_button.configure(state="disabled")
                    self._update_start_button_state()
                    self._update_position_action_buttons()
                    self._update_manual_controls()
                elif kind in {"manual_command_done", "manual_command_error"}:
                    self._manual_command_pending = False
                    self._refresh_manual_speed_window()
                    self._update_manual_controls()
                    self._update_position_action_buttons()
                    if kind == "manual_command_error":
                        self.status_var.set(
                            f"Manual speedエラー: {event.get('error', 'unknown error')}"
                        )
                        self._set_status_badge("experiment", "Manual停止", "error")
                    else:
                        self.status_var.set(str(event.get("description", "Manual speed")))
                        self._set_status_badge("experiment", "Manual操作", "warn")
                    self._update_start_button_state()
                elif kind == "manual_safety_stop":
                    self.status_var.set(
                        str(event.get("text", "Manual speedを停止しました"))
                    )
                    self._set_status_badge("experiment", "Manual安全停止", "warn")
                    self._refresh_manual_speed_window()
                    self._update_manual_controls()
                    self._update_position_action_buttons()
                    self._update_start_button_state()
                elif kind in {"position_action_done", "position_action_error"}:
                    self._position_action_pending = False
                    self._position_action_cancel_event = None
                    self._position_action_thread = None
                    self._refresh_remembered_position_display()
                    if kind == "position_action_error":
                        self.status_var.set(
                            f"位置操作エラー: {event.get('error', 'unknown error')}"
                        )
                        self._set_status_badge("experiment", "位置操作エラー", "error")
                    else:
                        self.status_var.set(
                            f"完了: {event.get('description', '位置操作')}"
                        )
                        self._set_status_badge("experiment", "位置操作完了", "ok")
                    self._update_start_button_state()
                    self._update_manual_controls()
                elif kind == "force":
                    self._consume_force_sample(event["sample"])
                elif kind == "angle":
                    self._consume_angle_sample(event)
                elif kind == "dlc_ready":
                    if event.get("worker") is self._standalone_dlc_worker:
                        self.status_var.set("DLC角度測定中（サーボ未接続）")
                        self.phase_var.set("DLC測定")
                        self._set_status_badge("camera", "接続済み", "ok")
                        self._set_status_badge("dlc", "測定中", "ok")
                elif kind == "dlc_error":
                    if event.get("worker") is self._standalone_dlc_worker:
                        self.status_var.set(f"DLC計測エラー: {event.get('error')}")
                        self._set_status_badge("camera", "エラー", "error")
                        self._set_status_badge("dlc", "エラー", "error")
                elif kind == "dlc_stopped":
                    worker = event.get("worker")
                    if worker is self._standalone_dlc_worker:
                        with self._resource_lock:
                            self._standalone_dlc_worker = None
                        self._dlc_enabled = False
                        self.dlc_toggle_button.configure(
                            state="disabled" if self._running else "normal",
                            text="DLC（session）" if self._running else "DLC ON",
                        )
                        if worker.error is None and not self._running:
                            self._set_status_badge("dlc", "未接続", "idle")
                            if not self._preview_enabled:
                                self._set_status_badge("camera", "未接続", "idle")
                        if self._preview_enabled and not self._running:
                            self._start_camera_preview_worker()
                elif kind == "worker_error":
                    self.status_var.set(f"エラー: {event.get('source')}: {event.get('error')}")
                    self.phase_var.set("異常終了")
                    self._set_status_badge("experiment", "異常終了", "error")
                    source = str(event.get("source", ""))
                    if "DLC" in source or "カメラ" in source:
                        self._set_status_badge("camera", "エラー", "error")
                        self._set_status_badge("dlc", "エラー", "error")
                    elif "servo" in source.lower() or "接続" in source:
                        self._set_status_badge("servo", "エラー", "error")
                        self._set_status_badge("encoder", "エラー", "error")
                elif kind == "hard_limit":
                    self.alert_label.configure(text=str(event.get("text", "")))
                    self.status_var.set(str(event.get("text", "")))
                    self.phase_var.set("ハードリミット停止")
                    self._set_status_badge("experiment", "ハードリミット停止", "error")
                elif kind == "point_to_point_trajectory":
                    trajectory = event.get("trajectory")
                    if isinstance(trajectory, ExcursionTrajectory):
                        self._set_target_angle_trajectory(trajectory)
                elif kind == "result":
                    self._handle_result(event)
            except Exception as exc:
                self.status_var.set(f"GUI更新エラー: {type(exc).__name__}: {exc}")
                self.phase_var.set("GUI異常")
                self._set_status_badge("experiment", "GUI異常", "error")

        try:
            for event in self._latest_events.drain():
                kind = event.get("kind")
                if kind == "force":
                    self._consume_force_sample(event["sample"])
                elif kind == "angle":
                    self._consume_angle_sample(event)
            self._update_preview()
            if (
                (self._force_plot_dirty or self._angle_plot_dirty)
                and time.monotonic() - self._last_plot_update
                >= PLOT_UPDATE_INTERVAL_S
            ):
                self._update_plots()
                self._last_plot_update = time.monotonic()
        except Exception as exc:
            self.status_var.set(f"GUI描画エラー: {type(exc).__name__}: {exc}")
            self.phase_var.set("GUI描画異常")
            self._set_status_badge("experiment", "GUI描画異常", "error")
        self._schedule_event_poll()

    def _consume_force_sample(self, sample: dict[str, Any]) -> None:
        elapsed_s = float(sample["elapsed_s"])
        phase = str(sample["phase"])
        if phase == "manual" and self._playback_started:
            return
        self.phase_var.set(phase)
        self.elapsed_var.set(f"経過時間: {elapsed_s:.2f} s")
        self._update_progress(elapsed_s, phase)
        readings = sample.get("force_readings", {})
        values: list[str] = []
        for tendon, servo_id in zip(TENDONS, SERVO_IDS_BY_TENDON):
            reading = readings.get(servo_id, readings.get(str(servo_id), {}))
            value = reading.get("force_n") if isinstance(reading, dict) else None
            if value is None:
                continue
            value = float(value)
            if math.isfinite(value):
                self.force_history[tendon].append((elapsed_s, value))
                values.append(f"{tendon}={value:.2f}N")
        self._force_plot_dirty = True
        self.force_var.set("Force: " + (", ".join(values) if values else "--"))
        if phase == "manual":
            self._latest_manual_sample = sample
            ready = bool(sample.get("manual_ready", False))
            if ready != self._manual_feedback_ready:
                self._manual_feedback_ready = ready
                self._update_manual_controls()
            self._refresh_manual_speed_window()

    def _consume_angle_sample(self, event: dict[str, Any]) -> None:
        if event.get("phase") == "manual" and self._playback_started:
            return
        elapsed_s = float(event["elapsed_s"])
        frame_index = int(event.get("frame_idx", -1))
        if frame_index == self._last_angle_frame:
            return
        self._last_angle_frame = frame_index
        self._min_likelihood_seen = min(
            self._min_likelihood_seen,
            float(event.get("min_likelihood", 0.0)),
        )
        for joint, value in zip(JOINTS, event["angles_deg"]):
            if math.isfinite(float(value)):
                self.angle_history[joint].append((elapsed_s, float(value)))
        self._angle_plot_dirty = True
        validity = "OK" if event.get("valid") else "INVALID"
        held = " held" if event.get("held_last") else ""
        self.camera_var.set(f"Camera: {elapsed_s:.2f}s")
        self.dlc_var.set(
            f"DLC: {validity}{held}, likelihood={float(event.get('min_likelihood', 0.0)):.2f}, "
            f"{float(event.get('inference_ms', 0.0)):.1f} ms"
        )
        self._set_status_badge(
            "camera",
            "稼働中" if event.get("valid") else "フレーム無効",
            "ok" if event.get("valid") else "warn",
        )
        self._set_status_badge(
            "dlc",
            "OK" if event.get("valid") else "尤度不足",
            "ok" if event.get("valid") else "warn",
        )

    def _update_preview(self) -> None:
        if not self._preview_enabled:
            return
        with self._resource_lock:
            source = self._angle_source
            camera_preview_worker = self._camera_preview_worker
            standalone_dlc_worker = self._standalone_dlc_worker
        if Image is None or ImageTk is None:
            return
        standalone_source = (
            standalone_dlc_worker.source
            if standalone_dlc_worker is not None
            and not standalone_dlc_worker.finished_event.is_set()
            else None
        )
        preview = (
            source.latest_preview
            if source is not None
            else standalone_source.latest_preview
            if standalone_source is not None
            else camera_preview_worker.latest_preview
            if camera_preview_worker is not None
            else None
        )
        if preview is None:
            if camera_preview_worker is not None and camera_preview_worker.error:
                self.preview_label.configure(
                    image="",
                    text=f"カメラ接続エラー\n{camera_preview_worker.error}",
                )
                self._set_status_badge("camera", "エラー", "error")
            return
        frame_index, is_overlay, frame = preview
        available_width = max(160, self.preview_label.winfo_width() - 12)
        available_height = max(120, self.preview_label.winfo_height() - 12)
        preview_size = (available_width, available_height)
        if (
            frame_index == self._last_preview_frame
            and is_overlay == self._last_preview_is_overlay
            and preview_size == self._last_preview_size
        ):
            return
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = rgb.shape[:2]
            max_width, max_height = available_width, available_height
            scale = min(max_width / width, max_height / height, 1.0)
            if scale < 1.0:
                rgb = cv2.resize(
                    rgb,
                    (max(1, int(width * scale)), max(1, int(height * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            image = Image.fromarray(rgb)
            self._photo = ImageTk.PhotoImage(image=image)
            self.preview_label.configure(image=self._photo, text="")
            self._last_preview_size = preview_size
            self._last_preview_frame = frame_index
            self._last_preview_is_overlay = is_overlay
            if source is None and standalone_source is None:
                self._set_status_badge("camera", "プレビュー中", "ok")
        except Exception as exc:
            self.status_var.set(f"Preview更新エラー: {type(exc).__name__}: {exc}")
            self.preview_label.configure(image="", text="カメラ表示エラー")

    def _start_camera_preview_worker(self) -> None:
        with self._resource_lock:
            if self._angle_source is not None or self._running:
                return
            standalone_dlc_worker = self._standalone_dlc_worker
            if (
                standalone_dlc_worker is not None
                and not standalone_dlc_worker.finished_event.is_set()
            ):
                return
            current_worker = self._camera_preview_worker
            if current_worker is not None and current_worker.thread.is_alive():
                return
            worker = CameraPreviewWorker(Path(self.camera_config_var.get()))
            self._camera_preview_worker = worker
        worker.start()

    def _stop_camera_preview_worker(self, timeout_s: float = 2.0) -> bool:
        with self._resource_lock:
            worker = self._camera_preview_worker
        if worker is None:
            return True
        if not worker.stop(timeout_s):
            return False
        with self._resource_lock:
            if self._camera_preview_worker is worker:
                self._camera_preview_worker = None
        return True

    def _start_dlc_measurement_worker(self) -> bool:
        if self._running or self._angle_source is not None:
            return False
        if not self._stop_camera_preview_worker():
            messagebox.showwarning(
                "カメラ切替待ち",
                "プレビュー用カメラを解放できませんでした。少し待ってからDLC ONを押してください。",
            )
            return False
        with self._resource_lock:
            current_worker = self._standalone_dlc_worker
            if current_worker is not None and not current_worker.finished_event.is_set():
                return False
            source = RealtimeJointAngleSource(
                dlc_config_path=Path(self.dlc_config_var.get()),
                camera_config_path=Path(self.camera_config_var.get()),
                min_likelihood=0.3,
                temporal_outlier_enabled=True,
                temporal_jump_threshold_deg=30.0,
                temporal_max_hold_frames=5,
            )
            worker = DLCMeasurementWorker(source, self._emit)
            self._standalone_dlc_worker = worker
        self._last_angle_frame = -1
        self._min_likelihood_seen = 1.0
        for history in self.angle_history.values():
            history.clear()
        self._angle_plot_dirty = True
        self._last_preview_frame = -1
        self._last_preview_is_overlay = None
        self._last_preview_size = (0, 0)
        self._set_status_badge("camera", "接続中", "busy")
        self._set_status_badge("dlc", "準備中", "busy")
        if self._preview_enabled:
            self.preview_label.configure(image="", text="DLC計測を開始しています")
            self._photo = None
        try:
            worker.start()
        except Exception as exc:
            with self._resource_lock:
                if self._standalone_dlc_worker is worker:
                    self._standalone_dlc_worker = None
            self._set_status_badge("camera", "エラー", "error")
            self._set_status_badge("dlc", "エラー", "error")
            messagebox.showerror("DLC起動エラー", f"{type(exc).__name__}: {exc}")
            return False
        return True

    def _stop_dlc_measurement_worker(self, timeout_s: float = 2.0) -> bool:
        with self._resource_lock:
            worker = self._standalone_dlc_worker
        if worker is None:
            self._dlc_enabled = False
            return True
        if not worker.stop(timeout_s):
            return False
        self._dlc_enabled = False
        return True

    def _toggle_dlc(self) -> None:
        if self._running:
            return
        if self._dlc_enabled:
            self._dlc_enabled = False
            self.dlc_toggle_button.configure(text="DLC停止中", state="disabled")
            if not self._stop_dlc_measurement_worker():
                self.status_var.set("DLC停止待ち: 推論処理の終了を待っています")
            return
        if self._start_dlc_measurement_worker():
            self._dlc_enabled = True
            self.dlc_toggle_button.configure(text="DLC OFF", state="normal")

    def _toggle_preview(self) -> None:
        self._preview_enabled = not self._preview_enabled
        self._last_preview_frame = -1
        self._last_preview_is_overlay = None
        self._last_preview_size = (0, 0)
        if self._preview_enabled:
            self.preview_toggle_button.configure(text="プレビューOFF")
            with self._resource_lock:
                source = self._angle_source
                camera_preview_worker = self._camera_preview_worker
                standalone_dlc_worker = self._standalone_dlc_worker
            if (
                source is None
                and camera_preview_worker is None
                and standalone_dlc_worker is None
                and not self._running
            ):
                self._start_camera_preview_worker()
                with self._resource_lock:
                    camera_preview_worker = self._camera_preview_worker
            self.preview_label.configure(
                image="",
                text=(
                    "カメラ映像を待機中"
                    if (
                        source is not None
                        or camera_preview_worker is not None
                        or standalone_dlc_worker is not None
                        or self._running
                    )
                    else "カメラを開いています"
                ),
            )
            self._photo = None
            self._update_preview()
        else:
            self.preview_toggle_button.configure(text="プレビューON")
            self.preview_label.configure(image="", text="プレビューOFF")
            self._photo = None
            with self._resource_lock:
                source = self._angle_source
                standalone_dlc_worker = self._standalone_dlc_worker
            if (
                source is None
                and standalone_dlc_worker is None
                and not self._running
            ):
                self._stop_camera_preview_worker()
                self._set_status_badge("camera", "未接続", "idle")

    def _mark_force_plot_dirty(self, *_args) -> None:
        self._force_plot_dirty = True

    def _capture_force_plot_backgrounds(self, _event=None) -> None:
        self._force_plot_backgrounds = {
            tendon: self.force_canvas.copy_from_bbox(axis.bbox)
            for tendon, axis in zip(TENDONS, self.force_axes)
        }
        self._force_plot_dirty = True

    def _capture_angle_plot_background(self, _event=None) -> None:
        self._angle_plot_background = self.angle_canvas.copy_from_bbox(
            self.angle_axis.bbox
        )
        self._angle_plot_dirty = True

    def _blit_force_plot(self) -> bool:
        if len(self._force_plot_backgrounds) != len(TENDONS):
            return False
        for tendon, axis in zip(TENDONS, self.force_axes):
            self.force_canvas.restore_region(self._force_plot_backgrounds[tendon])
            axis.draw_artist(self.force_lines[tendon])
            axis.draw_artist(self.force_limit_lines[tendon])
            self.force_canvas.blit(axis.bbox)
        return True

    def _blit_angle_plot(self) -> bool:
        if self._angle_plot_background is None:
            return False
        self.angle_canvas.restore_region(self._angle_plot_background)
        for line in self.angle_lines.values():
            self.angle_axis.draw_artist(line)
        for line in self.target_angle_lines.values():
            self.angle_axis.draw_artist(line)
        self.angle_canvas.blit(self.angle_axis.bbox)
        return True

    def _update_plots(self) -> None:
        if self._force_plot_dirty:
            self._update_force_plot()
        if self._angle_plot_dirty:
            self._update_angle_plot()

    def _update_force_plot(self) -> None:
        latest = max(
            (history[-1][0] for history in self.force_history.values() if history),
            default=0.0,
        )
        visible_start = max(0.0, latest - 20.0)
        desired_xlim = (visible_start, max(20.0, latest))
        desired_ylims: dict[str, tuple[float, float]] = {}
        urgent_rescale = False

        for tendon, axis in zip(TENDONS, self.force_axes):
            visible = [
                item
                for item in self.force_history[tendon]
                if item[0] >= visible_start
            ]
            times = [item[0] for item in visible]
            forces = [item[1] for item in visible]
            self.force_lines[tendon].set_data(times, forces)
            low = min(0.0, min(forces) * 1.1) if forces else 0.0
            high = max(1.0, max(forces) * 1.1) if forces else 1.0
            try:
                limit = float(self.limit_vars[tendon].get())
            except (TypeError, ValueError):
                limit = None
            if limit is not None and math.isfinite(limit) and limit > 0.0:
                self.force_limit_lines[tendon].set_data(
                    [desired_xlim[0], desired_xlim[1]], [limit, limit]
                )
                high = max(high, limit * 1.1)
            else:
                self.force_limit_lines[tendon].set_data([], [])
            desired_ylims[tendon] = (low, high)
            current_low, current_high = axis.get_ylim()
            urgent_rescale = urgent_rescale or low < current_low or high > current_high

        now = time.monotonic()
        limits_changed = any(
            tuple(axis.get_xlim()) != desired_xlim
            or tuple(axis.get_ylim()) != desired_ylims[tendon]
            for tendon, axis in zip(TENDONS, self.force_axes)
        )
        full_draw = (
            len(self._force_plot_backgrounds) != len(TENDONS)
            or urgent_rescale
            or (
                limits_changed
                and now - self._force_last_full_draw_at
                >= PLOT_AXIS_REFRESH_INTERVAL_S
            )
        )
        if full_draw:
            for tendon, axis in zip(TENDONS, self.force_axes):
                axis.set_xlim(*desired_xlim)
                axis.set_ylim(*desired_ylims[tendon])
            self.force_canvas.draw()
            self._force_last_full_draw_at = now
        self._blit_force_plot()
        self._force_plot_dirty = False

    def _update_angle_plot(self) -> None:
        latest = max(
            (history[-1][0] for history in self.angle_history.values() if history),
            default=0.0,
        )
        visible_start = max(0.0, latest - 20.0)
        desired_xlim = (visible_start, max(20.0, latest))
        all_angles: list[float] = []
        for joint in JOINTS:
            visible = [
                item
                for item in self.angle_history[joint]
                if item[0] >= visible_start
            ]
            times = [item[0] for item in visible]
            angles = [item[1] for item in visible]
            self.angle_lines[joint].set_data(times, angles)
            all_angles.extend(angles)
            target_visible = (
                [
                    item
                    for item in self.target_angle_history[joint]
                    if visible_start <= item[0] <= desired_xlim[1]
                ]
                if self._target_angles_visible
                else []
            )
            target_times = [item[0] for item in target_visible]
            target_angles = [item[1] for item in target_visible]
            self.target_angle_lines[joint].set_data(target_times, target_angles)
            all_angles.extend(target_angles)
        if all_angles:
            low = min(all_angles)
            high = max(all_angles)
            margin = max(5.0, (high - low) * 0.15)
            desired_ylim = (low - margin, high + margin)
        else:
            desired_ylim = (0.0, 180.0)

        current_low, current_high = self.angle_axis.get_ylim()
        urgent_rescale = (
            min(all_angles, default=current_low) < current_low
            or max(all_angles, default=current_high) > current_high
        )
        now = time.monotonic()
        limits_changed = (
            tuple(self.angle_axis.get_xlim()) != desired_xlim
            or tuple(self.angle_axis.get_ylim()) != desired_ylim
        )
        full_draw = (
            self._angle_plot_background is None
            or urgent_rescale
            or (
                limits_changed
                and now - self._angle_last_full_draw_at
                >= PLOT_AXIS_REFRESH_INTERVAL_S
            )
        )
        if full_draw:
            self.angle_axis.set_xlim(*desired_xlim)
            self.angle_axis.set_ylim(*desired_ylim)
            self.angle_canvas.draw()
            self._angle_last_full_draw_at = now
        self._blit_angle_plot()
        self._angle_plot_dirty = False

    def _handle_result(self, event: dict[str, Any]) -> None:
        self._latest_events.clear()
        if self._position_action_cancel_event is not None:
            self._position_action_cancel_event.set()
        self._position_controller = None
        self._position_action_pending = False
        self._position_action_cancel_event = None
        self._position_action_thread = None
        self._running = False
        self._runner_thread = None
        self._manual_command_pending = False
        self._manual_feedback_ready = False
        self._manual_feedback_ready_event.clear()
        self._latest_manual_sample = None
        self._target_angles_visible = True
        with self._manual_speed_lock:
            self._manual_speed_commands = {servo_id: 0 for servo_id in SERVO_IDS}
        self._refresh_manual_speed_window()
        self._angle_plot_dirty = True
        self.stop_button.configure(state="disabled")
        self.force_quit_button.configure(state="disabled")
        self._set_input_widgets_enabled(True)
        self._connection_ready = False
        self._playback_started = False
        self._update_manual_controls()
        self._update_position_action_buttons()
        self._set_status_badge("servo", "未接続", "idle")
        self._set_status_badge("encoder", "未接続", "idle")
        self._set_status_badge("camera", "未接続", "idle")
        self._set_status_badge("dlc", "未接続", "idle")
        self.dlc_toggle_button.configure(state="normal", text="DLC ON")
        self.preview_label.configure(
            image="",
            text="カメラ映像を待機中" if self._preview_enabled else "カメラ未接続",
        )
        self._photo = None
        self._last_preview_frame = -1
        self._last_preview_is_overlay = None
        self._last_preview_size = (0, 0)
        if self._preview_enabled:
            self._start_camera_preview_worker()
        status = str(event.get("status", "error"))
        session_dir = Path(str(event["session_dir"]))
        self._session_dir = session_dir
        if status == "force_hard_limit":
            self.alert_label.configure(text="Forceハードリミットで停止しました。ログを確認してください")
        elif status == "force_telemetry_unavailable":
            self.alert_label.configure(text="Forceテレメトリ不明のため停止しました。ログを確認してください")
        max_force_parts = []
        for tendon in TENDONS:
            values = [value for _, value in self.force_history[tendon]]
            if values:
                max_force_parts.append(f"{tendon}={max(values):.2f}")
        display_status = {
            "completed": "完了",
            "operator_stop": "手動停止",
            "force_quit": "FORCE QUIT",
            "force_hard_limit": "HARD LIMIT STOP",
            "force_telemetry_unavailable": "FORCE SAFETY STOP",
            "error": "異常終了",
        }.get(status, status)
        result_state = "ok" if status == "completed" else (
            "warn" if status in {"operator_stop", "force_quit"} else "error"
        )
        self._set_status_badge("experiment", display_status, result_state)
        self.result_tree.insert(
            "",
            0,
            values=(
                display_status,
                ", ".join(max_force_parts) or "--",
                f"{self._min_likelihood_seen:.3f}",
                self._shorten_path(session_dir),
            ),
        )
        for item_id in self.result_tree.get_children()[MAX_RESULT_ROWS:]:
            self.result_tree.delete(item_id)
        self._stop_event = None
        self._start_event = None
        self._force_quit_event = None
        with self._resource_lock:
            self._pending_playback_config = None
        self._start_in_progress = False
        self.connect_button.configure(
            state="normal" if self.file_list.curselection() else "disabled"
        )
        self.start_button.configure(state="disabled")
        self.open_session_button.configure(state="normal")
        if status == "completed":
            self.progress_bar.configure(value=100.0)
            self.progress_var.set("進捗: 完了")
        elif status == "operator_stop":
            self.progress_var.set("進捗: 手動停止")
        elif status == "force_quit":
            self.progress_var.set("進捗: Force Quit")
        else:
            self.progress_var.set(f"進捗: {status}")
        if not self._closing_requested:
            keep = messagebox.askyesno(
                "結果保存",
                f"セッション結果を保存しますか？\n{session_dir}",
            )
            if not keep and status not in {
                "force_hard_limit",
                "force_telemetry_unavailable",
            }:
                try:
                    shutil.rmtree(session_dir)
                    self.status_var.set("結果を破棄しました。診断ログも削除されています")
                except Exception as exc:
                    self.status_var.set(f"結果の破棄に失敗: {exc}")
            elif not keep:
                self.status_var.set("Force保護停止の結果は安全のため保持しました")
            else:
                self.status_var.set(f"結果を保存しました: {session_dir}")
        if self._closing_requested:
            self.after_idle(self._destroy_after_cleanup)

    def _destroy_after_cleanup(self) -> None:
        self._stop_dlc_measurement_worker()
        self._stop_camera_preview_worker()
        if self._poll_after_id is not None:
            try:
                self.after_cancel(self._poll_after_id)
            except tk.TclError:
                pass
            self._poll_after_id = None
        self.destroy()

    def _on_close(self) -> None:
        if self._running:
            if not messagebox.askyesno(
                "実行中",
                "ソフトウェア停止してGUIを閉じますか？",
            ):
                return
            self._closing_requested = True
            self._request_stop()
            return
        self._destroy_after_cleanup()


def main() -> None:
    app = CsvPlayerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
