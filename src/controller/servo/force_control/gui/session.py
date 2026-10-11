"""Hardware session shared by the force-control GUI.

The session keeps serial I/O and force control away from Tkinter's event loop.
It never resets servo IDs or writes servo/encoder zero commands.  Existing
calibration files are loaded at connect time, and the AS5048A stream is started
for the current session.
"""

from __future__ import annotations

import json
import csv
import math
import queue
import sys
import threading
import time
from bisect import bisect_left
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Sequence

from controller.servo.torque_calibration.storage import (
    DEFAULT_ENCODER_POSITION_PATH,
    intercept_for_model,
    load_encoder_calibration_positions,
    load_torque_intercept_calibration,
    save_encoder_calibration_positions,
    save_torque_intercept_calibration,
)


CONTROLLER_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = CONTROLLER_ROOT.parent
PROJECT_ROOT = SRC_ROOT.parent
API_ROOT = PROJECT_ROOT / "third_party" / "ServoHardwareAPI"
DEFAULT_FORCE_CONFIG_PATH = CONTROLLER_ROOT / "config_force_control.json"
DEFAULT_CALIBRATION_DIR = API_ROOT / "calibration" / "current"

SERVO_IDS = (1, 2, 3, 4, 5, 6)
SERVO_NAMES = ("ED", "DI", "FDS", "FDP", "LU", "PI")
MANUAL_SPEED_PRESETS = (-500, -100, 0, 100, 500)
FEEDBACK_SYNC_TOLERANCE_S = 0.030
MONITORING_FEEDBACK_SYNC_TOLERANCE_S = 0.500
MONITORING_FEEDBACK_TIMEOUT_S = 0.500
FEEDBACK_LOG_COOLDOWN_S = 1.0
CALIBRATION_SAMPLE_WAIT_S = 3.0
COMMUNICATION_STATS_WINDOW_S = 5.0
COMMUNICATION_STATS_MAX_SAMPLES = 1024
VALID_ENCODER_STATUSES = {"OK", "MULTITURN_UNCERTAIN"}
FEEDBACK_HISTORY_DURATION_S = 0.5
FEEDBACK_HISTORY_MAX_SAMPLES = 128
MONITORING_FORCE_JUMP_THRESHOLD_N = 2.0
MONITORING_ANGLE_JUMP_THRESHOLD_DEG = 10.0
MONITORING_JUMP_CONFIRMATION_TIMEOUT_S = 1.0
MONITORING_JUMP_LOG_COOLDOWN_S = 1.0
DEFAULT_ZERO_OFFSET_FILE = DEFAULT_CALIBRATION_DIR / "as5048a_unloaded_zero_offsets.json"
DEFAULT_INTERCEPT_CALIBRATION_FILE = (
    DEFAULT_CALIBRATION_DIR / "torque_intercept_calibration.json"
)


class _CompleteLineTransport:
    """Keep timeout-split serial bytes until a complete newline arrives."""

    def __init__(self, transport: Any):
        self._transport = transport
        self._read_buffer = bytearray()

    def __getattr__(self, name: str) -> Any:
        return getattr(self._transport, name)

    def reset_buffers(self) -> None:
        self._read_buffer.clear()
        self._transport.reset_buffers()

    def read_raw_line(self) -> bytes:
        self._transport.require_open()
        newline_index = self._read_buffer.find(b"\n")
        if newline_index < 0:
            raw = self._transport.port.read_until(b"\n")
            if raw:
                self._read_buffer.extend(raw)
            newline_index = self._read_buffer.find(b"\n")
        if newline_index < 0:
            return b""
        line = bytes(self._read_buffer[: newline_index + 1])
        del self._read_buffer[: newline_index + 1]
        return line

    def read_line(self, encoding: str = "utf-8") -> str:
        raw = self.read_raw_line()
        if not raw:
            return ""
        return raw.decode(encoding, errors="replace")

    def close(self) -> None:
        self._read_buffer.clear()
        self._transport.close()


@dataclass(frozen=True)
class GuiForceConfig:
    min_target_force_n: float
    max_target_force_n: float
    target_force_levels_n: tuple
    target_force_levels_by_tendon: Dict[str, tuple]
    hard_force_limits_n_by_tendon: Dict[str, float]
    ramp_duration_s: float
    control_hz: float
    encoder_rate_hz: float
    feedback_timeout_s: float


def _finite_number(value: Any, field_name: str) -> float:
    if isinstance(value, bool):
        raise ValueError(f"{field_name} must be a number")
    try:
        number = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{field_name} must be a number") from exc
    if not math.isfinite(number):
        raise ValueError(f"{field_name} must be finite")
    return number


def load_gui_force_config(path: Path = DEFAULT_FORCE_CONFIG_PATH) -> GuiForceConfig:
    """Load and validate the tendon-specific target levels used by the GUI."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("default"), dict):
        raise ValueError("Force-control config must contain a default object")

    default = payload["default"]
    minimum = _finite_number(default.get("min_target_force_n"), "min_target_force_n")
    maximum = _finite_number(default.get("max_target_force_n"), "max_target_force_n")
    if minimum < 0.0 or minimum >= maximum:
        raise ValueError("min_target_force_n must be non-negative and below max_target_force_n")

    levels_by_tendon_value = default.get("target_force_levels_n_by_tendon")
    if levels_by_tendon_value is None:
        levels_value = default.get("target_force_levels_n")
        if not isinstance(levels_value, list) or not levels_value:
            raise ValueError("target_force_levels_n must be a non-empty list")
        levels_by_tendon_value = {
            tendon: levels_value for tendon in SERVO_NAMES
        }
    if not isinstance(levels_by_tendon_value, dict):
        raise ValueError("target_force_levels_n_by_tendon must be an object")

    levels_by_tendon = {}
    for tendon in SERVO_NAMES:
        levels_value = levels_by_tendon_value.get(tendon)
        if not isinstance(levels_value, list) or not levels_value:
            raise ValueError(
                f"target_force_levels_n_by_tendon[{tendon}] must be a non-empty list"
            )
        levels = tuple(
            _finite_number(value, f"target_force_levels_n_by_tendon[{tendon}]")
            for value in levels_value
        )
        if tuple(sorted(set(levels))) != levels:
            raise ValueError(
                f"target_force_levels_n_by_tendon[{tendon}] must be strictly increasing"
            )
        if levels[0] < minimum or levels[-1] > maximum:
            raise ValueError(
                f"target_force_levels_n_by_tendon[{tendon}] must be inside the target force range"
            )
        levels_by_tendon[tendon] = levels
    levels = tuple(sorted(set(value for values in levels_by_tendon.values() for value in values)))

    hard_limits_value = default.get("hard_force_limits_n_by_tendon")
    if not isinstance(hard_limits_value, dict):
        raise ValueError("hard_force_limits_n_by_tendon must be an object")
    hard_limits_by_tendon = {}
    for tendon in SERVO_NAMES:
        if tendon not in hard_limits_value:
            raise ValueError(f"hard_force_limits_n_by_tendon must define {tendon}")
        hard_limit = _finite_number(
            hard_limits_value[tendon],
            f"hard_force_limits_n_by_tendon[{tendon}]",
        )
        if hard_limit <= max(levels_by_tendon[tendon]):
            raise ValueError(
                f"hard_force_limits_n_by_tendon[{tendon}] must be above its maximum target"
            )
        hard_limits_by_tendon[tendon] = hard_limit

    ramp_duration = _finite_number(default.get("ramp_duration_s"), "ramp_duration_s")
    control_hz = _finite_number(default.get("control_hz"), "control_hz")
    encoder_rate_hz = _finite_number(default.get("encoder_rate_hz"), "encoder_rate_hz")
    feedback_timeout_s = _finite_number(
        default.get("feedback_timeout_s"), "feedback_timeout_s"
    )
    if ramp_duration <= 0.0 or control_hz <= 0.0 or encoder_rate_hz <= 0.0:
        raise ValueError("GUI timing settings must be greater than zero")
    if feedback_timeout_s <= 0.0:
        raise ValueError("feedback_timeout_s must be greater than zero")

    return GuiForceConfig(
        min_target_force_n=minimum,
        max_target_force_n=maximum,
        target_force_levels_n=levels,
        target_force_levels_by_tendon=levels_by_tendon,
        hard_force_limits_n_by_tendon=hard_limits_by_tendon,
        ramp_duration_s=ramp_duration,
        control_hz=control_hz,
        encoder_rate_hz=encoder_rate_hz,
        feedback_timeout_s=feedback_timeout_s,
    )


@dataclass(frozen=True)
class ControlSample:
    received_at: float
    servo_id: int
    tendon: str
    force_n: float
    target_force_n: float
    force_error_n: float
    speed_command: int
    position: float
    encoder_position_deg: float
    angle_delta_deg: float
    active: bool


@dataclass(frozen=True)
class DiagnosticSample:
    """One raw reader-callback value for low-latency GUI diagnostics."""

    received_at: float
    channel_id: int
    value: float


@dataclass
class _ServoState:
    servo_id: int
    tendon: str
    sensor_id: int
    model: dict
    intercept_nmm: float
    target_force_n: float = 0.5
    baseline_force_n: Optional[float] = None
    previous_snapshot: Optional[dict] = None
    controller: Any = None
    ramp_started_at: Optional[float] = None
    ramp_start_n: float = 0.5
    ramp_end_n: float = 0.5
    ramp_duration_s: float = 1.0
    active: bool = False
    latest_force_n: Optional[float] = None
    latest_position: Optional[float] = None
    pending_monitoring_jump: Optional[dict] = None
    pending_monitoring_jump_received_at: Optional[float] = None
    last_monitoring_jump_log_at: float = 0.0


class _RecentStreamStats:
    """Keep a bounded rolling window of receive timing and validity."""

    def __init__(self):
        self._lock = threading.Lock()
        self._samples: Dict[int, Deque[tuple]] = {}

    def clear(self) -> None:
        with self._lock:
            self._samples.clear()

    def record(
        self,
        stream_id: int,
        valid: bool,
        *,
        received_at: Optional[float] = None,
    ) -> None:
        timestamp = time.monotonic() if received_at is None else float(received_at)
        with self._lock:
            samples = self._samples.setdefault(
                int(stream_id),
                deque(maxlen=COMMUNICATION_STATS_MAX_SAMPLES),
            )
            samples.append((timestamp, bool(valid)))

    def snapshot(
        self,
        stream_id: int,
        *,
        now: Optional[float] = None,
        window_s: float = COMMUNICATION_STATS_WINDOW_S,
    ) -> dict:
        current_time = time.monotonic() if now is None else float(now)
        cutoff = current_time - float(window_s)
        with self._lock:
            samples = [
                sample
                for sample in self._samples.get(int(stream_id), ())
                if sample[0] >= cutoff
            ]
        intervals = [
            current[0] - previous[0]
            for previous, current in zip(samples, samples[1:])
            if current[0] >= previous[0]
        ]
        elapsed_s = samples[-1][0] - samples[0][0] if len(samples) >= 2 else 0.0
        effective_hz = (
            (len(samples) - 1) / elapsed_s
            if len(samples) >= 2 and elapsed_s > 0.0
            else None
        )
        valid_percent = (
            100.0 * sum(1 for _, valid in samples if valid) / len(samples)
            if samples
            else None
        )
        return {
            "window_s": float(window_s),
            "sample_count": len(samples),
            "effective_hz": effective_hz,
            "max_interval_s": max(intervals) if intervals else None,
            "valid_percent": valid_percent,
            "last_age_s": (
                max(0.0, current_time - samples[-1][0]) if samples else None
            ),
        }


class _EncoderFanout:
    """Route six AS5048A samples from one reader to six latest stores."""

    def __init__(self, stores: Dict[int, Any], on_sample=None):
        self.stores = stores
        self.on_sample = on_sample

    def __call__(self, parsed: Any) -> Optional[Any]:
        if not isinstance(parsed, (tuple, list)) or len(parsed) != 2:
            return None
        if parsed[0] not in ("sensor_data", "data"):
            return None
        sample = parsed[1]
        if not isinstance(sample, dict):
            return None
        try:
            sensor_id = int(sample["sensor_id"])
        except (KeyError, TypeError, ValueError):
            return None
        store = self.stores.get(sensor_id)
        if store is not None:
            received_at = time.monotonic()
            store.publish(sample, received_at=received_at)
            if self.on_sample is not None:
                self.on_sample(sensor_id, sample, received_at)
        return None


class ForceControlSession:
    """Six-servo force-control session for the GUI.

    ``connect`` starts feedback acquisition but does not send motion commands.
    ``arm`` validates controller parameters, and only ``start`` allows speed
    commands to be sent.
    """

    def __init__(
        self,
        config: Optional[GuiForceConfig] = None,
        *,
        status_callback: Optional[Callable[[str], None]] = None,
    ):
        self.config = config or load_gui_force_config()
        self.status_callback = status_callback
        self._lock = threading.RLock()
        self._command_lock = threading.Lock()
        self._feedback_lock = threading.Lock()
        self._baseline_lock = threading.Lock()
        self._calibration_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._sample_queue: queue.Queue = queue.Queue(maxsize=3000)
        self._servo_diagnostic_queue: queue.Queue = queue.Queue(maxsize=3000)
        self._encoder_diagnostic_queue: queue.Queue = queue.Queue(maxsize=3000)
        self._event_queue: queue.Queue = queue.Queue(maxsize=256)
        self._connected = False
        self._armed = False
        self._running = False
        self._faulted = False
        self._last_fault_message: Optional[str] = None
        self._servo = None
        self._encoder = None
        self._servo_transport = None
        self._encoder_transport = None
        self._servo_store = None
        self._encoder_reader_store = None
        self._encoder_stores: Dict[int, Any] = {}
        self._servo_reader = None
        self._encoder_reader = None
        self._control_thread: Optional[threading.Thread] = None
        self._states: Dict[int, _ServoState] = {}
        self._servo_index: Dict[int, int] = {}
        self._manual_speed_commands: Dict[int, int] = {
            servo_id: 0 for servo_id in SERVO_IDS
        }
        self._manual_feedback_recovering = False
        self._feedback_errors: Dict[int, str] = {}
        self._servo_stream_stats = _RecentStreamStats()
        self._encoder_stream_stats = _RecentStreamStats()
        self._feedback_history_lock = threading.Lock()
        self._servo_feedback_history: Deque[tuple] = deque(
            maxlen=FEEDBACK_HISTORY_MAX_SAMPLES
        )
        self._encoder_feedback_histories: Dict[int, Deque[tuple]] = {
            sensor_id: deque(maxlen=FEEDBACK_HISTORY_MAX_SAMPLES)
            for sensor_id in SERVO_IDS
        }
        self._last_feedback_warning_at = 0.0
        self._feedback_warning_logged = False
        self._last_parameters: Optional[dict] = None
        self._calibration_dir = DEFAULT_CALIBRATION_DIR
        self._zero_offset_path = DEFAULT_ZERO_OFFSET_FILE
        self._intercept_calibration_path = DEFAULT_INTERCEPT_CALIBRATION_FILE
        self._encoder_position_path = DEFAULT_ENCODER_POSITION_PATH
        self._zero_offsets: Dict[int, dict] = {}
        self._encoder_calibration_positions: Dict[int, dict] = {}
        self._connect_cancel_event: Optional[threading.Event] = None

    @property
    def connected(self) -> bool:
        with self._lock:
            return self._connected

    @property
    def armed(self) -> bool:
        with self._lock:
            return self._armed

    @property
    def running(self) -> bool:
        with self._lock:
            return self._running

    @property
    def faulted(self) -> bool:
        with self._lock:
            return self._faulted

    @property
    def fault_message(self) -> Optional[str]:
        with self._lock:
            return self._last_fault_message

    @property
    def manual_feedback_recovering(self) -> bool:
        with self._lock:
            return self._manual_feedback_recovering

    def feedback_errors_snapshot(self) -> Dict[int, str]:
        with self._lock:
            return dict(self._feedback_errors)

    def manual_speed_snapshot(self) -> Dict[int, int]:
        with self._lock:
            return dict(self._manual_speed_commands)

    def calibration_readiness_snapshot(self) -> dict:
        """Return user-facing reasons why torque calibration is unavailable."""

        with self._lock:
            if not self._connected:
                common_reason = "connect first"
            elif self._faulted:
                common_reason = "faulted; disconnect and reconnect"
            elif self._armed or self._running:
                common_reason = "stop and disarm force control"
            elif any(self._manual_speed_commands.values()):
                common_reason = "set every manual speed command to 0"
            else:
                common_reason = None

            by_servo = {
                servo_id: common_reason
                for servo_id in SERVO_IDS
            }
            return {"all_reason": common_reason, "by_servo": by_servo}

    def _emit(self, message: str) -> None:
        try:
            self._event_queue.put_nowait(message)
        except queue.Full:
            try:
                self._event_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._event_queue.put_nowait(message)
            except queue.Full:
                pass
        if self.status_callback is not None:
            self.status_callback(message)

    def _update_feedback_errors(
        self,
        errors: Dict[int, str],
        *,
        all_feedback_failed: bool,
    ) -> None:
        message = None
        now = time.monotonic()
        with self._lock:
            previous_ids = set(self._feedback_errors)
            self._feedback_errors = dict(errors)
            current_ids = set(errors)
            if not current_ids:
                if previous_ids and self._feedback_warning_logged:
                    self._feedback_warning_logged = False
                    message = "Feedback recovered."
            elif (
                not all_feedback_failed
                and current_ids != previous_ids
                and now - self._last_feedback_warning_at >= FEEDBACK_LOG_COOLDOWN_S
            ):
                details = "; ".join(
                    errors[servo_id] for servo_id in sorted(errors)
                )
                self._last_feedback_warning_at = now
                self._feedback_warning_logged = True
                message = f"WARNING: partial feedback unavailable: {details}"
        if message is not None:
            self._emit(message)

    @staticmethod
    def _check_cancelled(cancel_event: threading.Event) -> None:
        if cancel_event.is_set():
            raise RuntimeError("Connection cancelled")

    def drain_events(self, limit: int = 100) -> List[str]:
        events: List[str] = []
        for _ in range(max(1, int(limit))):
            try:
                events.append(self._event_queue.get_nowait())
            except queue.Empty:
                return events
        return events

    def drain_samples(self, limit: int = 300) -> List[ControlSample]:
        samples: List[ControlSample] = []
        for _ in range(max(1, int(limit))):
            try:
                samples.append(self._sample_queue.get_nowait())
            except queue.Empty:
                break
        return samples

    def drain_servo_diagnostic_samples(
        self, limit: int = 300
    ) -> List[DiagnosticSample]:
        return self._drain_queue(self._servo_diagnostic_queue, limit)

    def drain_encoder_diagnostic_samples(
        self, limit: int = 300
    ) -> List[DiagnosticSample]:
        return self._drain_queue(self._encoder_diagnostic_queue, limit)

    @staticmethod
    def _drain_queue(sample_queue: queue.Queue, limit: int) -> list:
        samples = []
        for _ in range(max(1, int(limit))):
            try:
                samples.append(sample_queue.get_nowait())
            except queue.Empty:
                break
        return samples

    def baseline_snapshot(self) -> Dict[int, Optional[float]]:
        """Return baseline forces without exposing mutable session state."""

        with self._lock:
            return {
                servo_id: state.baseline_force_n
                for servo_id, state in self._states.items()
            }

    def calibration_state_snapshot(self) -> Dict[int, dict]:
        """Return fixed model values and remembered positions for the calibration tab."""

        with self._lock:
            positions = dict(self._encoder_calibration_positions)
            return {
                servo_id: {
                    "slope_nmm_per_deg": float(state.model["slope_nmm_per_deg"]),
                    "intercept_nmm": float(state.intercept_nmm),
                    "remembered_encoder_position_deg": (
                        None
                        if servo_id not in positions
                        else float(positions[servo_id]["encoder_position_deg"])
                    ),
                }
                for servo_id, state in self._states.items()
            }

    @staticmethod
    def _format_file_time(path: Path) -> Optional[str]:
        try:
            return datetime.fromtimestamp(path.stat().st_mtime).astimezone().strftime(
                "%Y-%m-%d %H:%M:%S %Z"
            )
        except OSError:
            return None

    @staticmethod
    def _read_calibration_sensor_id(path: Path, fallback: int) -> int:
        try:
            with path.open("r", newline="", encoding="utf-8") as handle:
                row = next(csv.DictReader(handle), None)
            value = (row or {}).get("encoder_sensor_id")
            return int(float(value)) if value not in (None, "") else fallback
        except (OSError, StopIteration, TypeError, ValueError):
            return fallback

    @staticmethod
    def _read_zero_offset_records(path: Path) -> Dict[int, dict]:
        try:
            with path.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            channels = payload.get("channels", {})
            return {int(sensor_id): dict(record) for sensor_id, record in channels.items()}
        except (OSError, TypeError, ValueError, json.JSONDecodeError):
            return {}

    def diagnostics_snapshot(self) -> dict:
        """Return read-only session and calibration information for the logger tab."""

        now = time.monotonic()
        with self._lock:
            states = dict(self._states)
            calibration_dir = Path(self._calibration_dir)
            zero_offset_path = Path(self._zero_offset_path)
            offsets = dict(self._zero_offsets)
            connected = self._connected
            armed = self._armed
            running = self._running
            faulted = self._faulted
            parameters = dict(self._last_parameters or {})
            servo_store = self._servo_store
            encoder_stores = dict(self._encoder_stores)
            servo_reader = self._servo_reader
            encoder_reader = self._encoder_reader
            encoder_reader_store = self._encoder_reader_store

        if not offsets:
            offsets = self._read_zero_offset_records(zero_offset_path)

        def reader_status(reader: Any, stores: Iterable[Any]) -> dict:
            reader_thread = getattr(reader, "_thread", None)
            alive = bool(reader_thread is not None and reader_thread.is_alive())
            store_list = [store for store in stores if store is not None]
            records = [store.snapshot() for store in store_list]
            records = [record for record in records if record is not None]
            latest = max(records, key=lambda record: record.received_at) if records else None
            error = next(
                (str(store.error) for store in store_list if getattr(store, "error", None)),
                None,
            )
            age_s = None if latest is None else max(0.0, now - latest.received_at)
            if not connected:
                state = "disconnected"
            elif error is not None:
                state = "error"
            elif not alive:
                state = "stopped"
            elif latest is None:
                state = "waiting"
            elif age_s is not None and age_s > self.config.feedback_timeout_s:
                state = "timeout"
            else:
                state = "online"
            return {
                "state": state,
                "reader_alive": alive,
                "last_age_s": age_s,
                "error": error,
            }

        mapping = []
        for servo_id, tendon in zip(SERVO_IDS, SERVO_NAMES):
            state = states.get(servo_id)
            calibration_path = calibration_dir / f"spring_torque_calibration_servo_id_{servo_id}.csv"
            sensor_id = (
                state.sensor_id
                if state is not None
                else self._read_calibration_sensor_id(calibration_path, servo_id)
            )
            offset = offsets.get(sensor_id, {})
            mapping.append(
                {
                    "servo_id": servo_id,
                    "tendon": tendon,
                    "sensor_id": sensor_id,
                    "calibration_path": str(calibration_path),
                    "calibration_updated": self._format_file_time(calibration_path),
                    "zero_offset_updated": self._format_file_time(zero_offset_path),
                    "zero_sample_count": offset.get("sample_count"),
                    "zero_std_deg": offset.get("std_delta_deg"),
                    "initial_force_n": None if state is None else state.baseline_force_n,
                    "servo_stream": self._servo_stream_stats.snapshot(
                        servo_id, now=now
                    ),
                    "encoder_stream": self._encoder_stream_stats.snapshot(
                        sensor_id, now=now
                    ),
                }
            )

        return {
            "connection": {
                "connected": connected,
                "armed": armed,
                "running": running,
                "faulted": faulted,
            },
            "servo_communication": reader_status(servo_reader, (servo_store,)),
            "encoder_communication": reader_status(
                encoder_reader,
                (encoder_reader_store, *encoder_stores.values()),
            ),
            "feedback_errors": self.feedback_errors_snapshot(),
            "control_sign": parameters.get("control_sign"),
            "mapping": mapping,
        }

    def _load_models(
        self,
        calibration_dir: Path,
        zero_offset_path: Path,
        intercept_calibration_path: Path,
    ) -> Dict[int, _ServoState]:
        if str(API_ROOT / "python") not in sys.path:
            sys.path.insert(0, str(API_ROOT / "python"))
        from drivers.as5048a.calibration_store import (  # pylint: disable=import-outside-toplevel
            load_calibration_points,
            load_zero_offsets,
        )
        from drivers.as5048a.torque import (  # pylint: disable=import-outside-toplevel
            calibration_model_from_points,
        )

        offsets = {}
        if zero_offset_path.exists():
            offsets = load_zero_offsets(str(zero_offset_path))
        else:
            self._emit(f"WARNING: zero-offset file not found: {zero_offset_path}")
        intercept_records = load_torque_intercept_calibration(intercept_calibration_path)

        states: Dict[int, _ServoState] = {}
        for servo_id, tendon in zip(SERVO_IDS, SERVO_NAMES):
            calibration_path = calibration_dir / f"spring_torque_calibration_servo_id_{servo_id}.csv"
            if not calibration_path.exists():
                raise FileNotFoundError(f"Calibration file not found: {calibration_path}")
            points = load_calibration_points(
                str(calibration_path), servo_id, f"servo-{servo_id}"
            )
            model = calibration_model_from_points(points)
            if model["servo_id"] != servo_id:
                raise ValueError(
                    f"Calibration servo ID mismatch for {calibration_path}: "
                    f"{model['servo_id']} != {servo_id}"
                )
            sensor_id = int(model["sensor_id"])
            intercept_nmm = model["intercept_nmm"]
            calibrated_intercept = intercept_for_model(intercept_records, model)
            offset = offsets.get(sensor_id)
            if calibrated_intercept is not None:
                intercept_nmm = calibrated_intercept
                self._emit(
                    f"Using calibrated torque intercept for servo {servo_id}: "
                    f"{intercept_nmm:.6g} Nmm"
                )
            elif offset is not None:
                if int(offset["servo_id"]) != servo_id:
                    raise ValueError(f"Zero-offset servo ID mismatch for sensor {sensor_id}")
                intercept_nmm = -model["slope_nmm_per_deg"] * offset["mean_delta_deg"]
            else:
                self._emit(f"WARNING: no zero offset for servo {servo_id}; using calibration intercept")
            states[servo_id] = _ServoState(
                servo_id=servo_id,
                tendon=tendon,
                sensor_id=sensor_id,
                model=model,
                intercept_nmm=intercept_nmm,
                target_force_n=self.config.min_target_force_n,
                ramp_start_n=self.config.min_target_force_n,
                ramp_end_n=self.config.min_target_force_n,
                ramp_duration_s=self.config.ramp_duration_s,
            )
        with self._lock:
            self._zero_offsets = dict(offsets)
        return states

    def connect(
        self,
        servo_port: str,
        encoder_port: str,
        *,
        calibration_dir: Path = DEFAULT_CALIBRATION_DIR,
        zero_offset_path: Path = DEFAULT_ZERO_OFFSET_FILE,
        intercept_calibration_path: Path = DEFAULT_INTERCEPT_CALIBRATION_FILE,
    ) -> None:
        """Connect and initialize feedback without arming motion."""

        self.disconnect()
        connect_cancel_event = threading.Event()
        with self._lock:
            if self._connect_cancel_event is not None:
                raise RuntimeError("Another connection attempt is still stopping")
            self._connect_cancel_event = connect_cancel_event
            self._calibration_dir = Path(calibration_dir)
            self._zero_offset_path = Path(zero_offset_path)
            self._intercept_calibration_path = Path(intercept_calibration_path)
            self._encoder_position_path = (
                Path(calibration_dir) / DEFAULT_ENCODER_POSITION_PATH.name
            )
        servo_transport = None
        encoder_transport = None
        servo = None
        encoder = None
        servo_reader = None
        encoder_reader = None
        stop_event = None
        control_thread = None
        try:
            self._check_cancelled(connect_cancel_event)
            if not API_ROOT.exists():
                raise FileNotFoundError(f"ServoHardwareAPI was not found: {API_ROOT}")

            if str(API_ROOT / "python") not in sys.path:
                sys.path.insert(0, str(API_ROOT / "python"))
            from drivers import (  # pylint: disable=import-outside-toplevel
                AS5048ABridgeDriver,
                SerialTransport,
                ServoControllerDriver,
                ServoSystem,
            )
            from controller.servo.force_control.runner import (  # pylint: disable=import-outside-toplevel
                LatestSampleStore,
                ReaderTask,
            )

            self._emit("Loading force calibration files...")
            states = self._load_models(
                Path(calibration_dir),
                Path(zero_offset_path),
                Path(intercept_calibration_path),
            )
            encoder_calibration_positions = load_encoder_calibration_positions(
                self._encoder_position_path
            )
            self._check_cancelled(connect_cancel_event)
            system = ServoSystem(
                count=6,
                servo_ids=list(SERVO_IDS),
                names=list(SERVO_NAMES),
                as5048a_sensor_map=list(SERVO_IDS),
            )
            self._servo_index = {
                servo_id: system.index_for_servo_id(servo_id) for servo_id in SERVO_IDS
            }

            servo_transport = SerialTransport(
                str(servo_port),
                baudrate=921600,
                timeout=min(0.05, self.config.feedback_timeout_s),
            ).open()
            encoder_transport = _CompleteLineTransport(
                SerialTransport(
                    str(encoder_port),
                    baudrate=921600,
                    timeout=min(0.05, self.config.feedback_timeout_s),
                ).open()
            )
            servo = ServoControllerDriver(servo_transport, servo_system=system)
            encoder = AS5048ABridgeDriver(encoder_transport)
            servo_store = LatestSampleStore()
            encoder_reader_store = LatestSampleStore()
            encoder_stores = {sensor_id: LatestSampleStore() for sensor_id in SERVO_IDS}
            self._servo_stream_stats.clear()
            self._encoder_stream_stats.clear()
            with self._feedback_history_lock:
                self._servo_feedback_history.clear()
                for history in self._encoder_feedback_histories.values():
                    history.clear()
            self._drain_queue(self._servo_diagnostic_queue, 3000)
            self._drain_queue(self._encoder_diagnostic_queue, 3000)
            stop_event = threading.Event()
            servo_reader = ReaderTask(
                servo.read_parsed_line,
                self._accept_servo,
                servo_store,
                stop_event,
                name="force-gui-servo-reader",
            )
            encoder_reader = ReaderTask(
                encoder.read_parsed_line,
                _EncoderFanout(encoder_stores, self._record_encoder_sample),
                encoder_reader_store,
                stop_event,
                name="force-gui-encoder-reader",
            )

            self._check_cancelled(connect_cancel_event)
            encoder.stop_stream()
            encoder.set_rate_hz(self.config.encoder_rate_hz)
            encoder.start_all_stream()
            servo_reader.start()
            encoder_reader.start()
            servo_store.wait_for_sample(
                self.config.feedback_timeout_s * 10.0,
                stop_event=connect_cancel_event,
            )
            for store in encoder_stores.values():
                store.wait_for_sample(
                    self.config.feedback_timeout_s * 10.0,
                    stop_event=connect_cancel_event,
                )

            self._check_cancelled(connect_cancel_event)
            with self._lock:
                self._check_cancelled(connect_cancel_event)
                self._servo = servo
                self._encoder = encoder
                self._servo_transport = servo_transport
                self._encoder_transport = encoder_transport
                self._servo_store = servo_store
                self._encoder_reader_store = encoder_reader_store
                self._encoder_stores = encoder_stores
                self._servo_reader = servo_reader
                self._encoder_reader = encoder_reader
                self._stop_event = stop_event
                self._states = states
                self._encoder_calibration_positions = encoder_calibration_positions
                self._manual_speed_commands = {
                    servo_id: 0 for servo_id in SERVO_IDS
                }
                self._manual_feedback_recovering = False
                self._feedback_errors = {}
                self._connected = True
                self._armed = False
                self._running = False
                self._faulted = False
                self._last_fault_message = None
                self._last_parameters = None

            self._emit("Feedback connected; capturing stable initial force...")
            self.capture_baseline()
            self._check_cancelled(connect_cancel_event)
            control_thread = threading.Thread(
                target=self._loop,
                args=(stop_event,),
                name="force-gui-control-loop",
                daemon=True,
            )
            self._control_thread = control_thread
            control_thread.start()
            self._emit("Connected and disarmed. No motion command is being sent.")
        except Exception:
            if stop_event is not None:
                stop_event.set()
            if encoder is not None:
                try:
                    encoder.stop_stream()
                except Exception:
                    pass
            for reader in (servo_reader, encoder_reader):
                if reader is not None:
                    reader.join(1.0)
            if control_thread is not None and control_thread is not threading.current_thread():
                control_thread.join(1.0)
            if servo_transport is not None:
                try:
                    servo_transport.close()
                except Exception:
                    pass
            if encoder_transport is not None:
                try:
                    encoder_transport.close()
                except Exception:
                    pass
            with self._lock:
                self._connected = False
                self._armed = False
                self._running = False
                self._faulted = False
                self._servo = None
                self._encoder = None
                self._servo_transport = None
                self._encoder_transport = None
                self._servo_store = None
                self._encoder_reader_store = None
                self._encoder_stores = {}
                self._servo_reader = None
                self._encoder_reader = None
                self._control_thread = None
                self._states = {}
                self._encoder_calibration_positions = {}
                self._manual_speed_commands = {
                    servo_id: 0 for servo_id in SERVO_IDS
                }
                self._manual_feedback_recovering = False
                self._feedback_errors = {}
                self._last_fault_message = None
                self._stop_event = threading.Event()
            raise
        finally:
            with self._lock:
                if self._connect_cancel_event is connect_cancel_event:
                    self._connect_cancel_event = None

    def _accept_servo(self, parsed: Any) -> Optional[dict]:
        if isinstance(parsed, dict) and parsed.get("kind") == "telemetry":
            received_at = time.monotonic()
            with self._feedback_history_lock:
                self._servo_feedback_history.append((received_at, parsed))
                cutoff = received_at - FEEDBACK_HISTORY_DURATION_S
                while (
                    self._servo_feedback_history
                    and self._servo_feedback_history[0][0] < cutoff
                ):
                    self._servo_feedback_history.popleft()
            valid_values = parsed.get("valid", ())
            positions = parsed.get("positions", ())
            for servo_id in SERVO_IDS:
                servo_index = self._servo_index.get(servo_id)
                valid = bool(
                    servo_index is not None
                    and servo_index < len(valid_values)
                    and valid_values[servo_index]
                )
                self._servo_stream_stats.record(
                    servo_id, valid, received_at=received_at
                )
                if valid and servo_index < len(positions):
                    self._publish_bounded(
                        self._servo_diagnostic_queue,
                        DiagnosticSample(
                            received_at=received_at,
                            channel_id=servo_id,
                            value=float(positions[servo_index]),
                        ),
                    )
            return parsed
        return None

    def _record_encoder_sample(
        self, sensor_id: int, sample: dict, received_at: float
    ) -> None:
        status = str(sample.get("status", "")).strip().upper()
        has_multiturn = sample.get("relative_multi_turn_deg") is not None
        with self._feedback_history_lock:
            history = self._encoder_feedback_histories.get(int(sensor_id))
            if history is not None:
                history.append((received_at, sample))
                cutoff = received_at - FEEDBACK_HISTORY_DURATION_S
                while history and history[0][0] < cutoff:
                    history.popleft()
        self._encoder_stream_stats.record(
            sensor_id,
            status in VALID_ENCODER_STATUSES and has_multiturn,
            received_at=received_at,
        )
        if status in VALID_ENCODER_STATUSES and has_multiturn:
            self._publish_bounded(
                self._encoder_diagnostic_queue,
                DiagnosticSample(
                    received_at=received_at,
                    channel_id=int(sensor_id),
                    value=float(sample["relative_multi_turn_deg"]),
                ),
            )

    def _synchronized_feedback_pair(
        self,
        servo_id: int,
        sensor_id: int,
        servo_index: int,
        now: float,
        *,
        sync_tolerance_s: float = FEEDBACK_SYNC_TOLERANCE_S,
        feedback_timeout_s: Optional[float] = None,
        servo_history: Optional[Sequence[tuple]] = None,
        encoder_history: Optional[Sequence[tuple]] = None,
    ) -> tuple:
        """Return the freshest valid servo/encoder pair inside the sync limit."""

        if feedback_timeout_s is None:
            feedback_timeout_s = self.config.feedback_timeout_s

        if servo_history is None or encoder_history is None:
            with self._feedback_history_lock:
                servo_history = list(self._servo_feedback_history)
                encoder_history = list(
                    self._encoder_feedback_histories.get(sensor_id, ())
                )
        else:
            servo_history = list(servo_history)
            encoder_history = list(encoder_history)

        if not servo_history:
            return None, f"ID {servo_id}: no servo telemetry history"
        if not encoder_history:
            return None, f"ID {servo_id}: no encoder history for CS{sensor_id}"

        latest_servo_at, latest_servo = servo_history[-1]
        latest_encoder_at, latest_encoder = encoder_history[-1]
        servo_age_s = max(0.0, now - latest_servo_at)
        encoder_age_s = max(0.0, now - latest_encoder_at)
        valid_values = latest_servo.get("valid", ())
        latest_servo_valid = bool(
            servo_index < len(valid_values) and valid_values[servo_index]
        )
        latest_encoder_status = str(
            latest_encoder.get("status", "unknown")
        ).strip().upper()
        latest_encoder_has_multiturn = (
            latest_encoder.get("relative_multi_turn_deg") is not None
        )

        fresh_servos = []
        for received_at, telemetry in servo_history:
            age_s = max(0.0, now - received_at)
            valid = telemetry.get("valid", ())
            if (
                age_s <= feedback_timeout_s
                and servo_index < len(valid)
                and valid[servo_index]
            ):
                fresh_servos.append((received_at, telemetry))

        fresh_encoders = [
            (received_at, sample)
            for received_at, sample in encoder_history
            if max(0.0, now - received_at) <= feedback_timeout_s
            and str(sample.get("status", "")).strip().upper()
            in VALID_ENCODER_STATUSES
            and sample.get("relative_multi_turn_deg") is not None
        ]

        servo_times = [entry[0] for entry in fresh_servos]
        closest_gap_s = None
        for encoder_entry in reversed(fresh_encoders):
            encoder_at = encoder_entry[0]
            if not fresh_servos:
                break
            insertion_index = bisect_left(servo_times, encoder_at)
            candidate_indexes = {
                max(0, insertion_index - 1),
                min(len(fresh_servos) - 1, insertion_index),
            }
            servo_entry = min(
                (fresh_servos[index] for index in candidate_indexes),
                key=lambda item: abs(item[0] - encoder_at),
            )
            gap_s = abs(servo_entry[0] - encoder_at)
            if closest_gap_s is None or gap_s < closest_gap_s:
                closest_gap_s = gap_s
            if gap_s <= sync_tolerance_s:
                return (servo_entry, encoder_entry, gap_s), None

        if not fresh_servos:
            if not latest_servo_valid:
                reason = "latest servo telemetry is invalid"
            else:
                reason = (
                    f"servo sample is stale ({servo_age_s * 1000.0:.0f} ms > "
                    f"{feedback_timeout_s * 1000.0:.0f} ms)"
                )
        elif not fresh_encoders:
            if (
                latest_encoder_status in VALID_ENCODER_STATUSES
                and not latest_encoder_has_multiturn
            ):
                reason = "latest encoder sample has no multi-turn angle"
            elif latest_encoder_status not in VALID_ENCODER_STATUSES:
                reason = f"latest encoder status is {latest_encoder_status}"
            else:
                reason = (
                    f"encoder sample is stale ({encoder_age_s * 1000.0:.0f} ms > "
                    f"{feedback_timeout_s * 1000.0:.0f} ms)"
                )
        else:
            reason = (
                f"closest timestamp gap {closest_gap_s * 1000.0:.0f} ms exceeds "
                f"{sync_tolerance_s * 1000.0:.0f} ms"
            )
        return None, (
            f"ID {servo_id}: synchronized feedback unavailable: {reason}; "
            f"servo age {servo_age_s * 1000.0:.0f} ms, "
            f"encoder age {encoder_age_s * 1000.0:.0f} ms"
        )

    def _feedback(self) -> Dict[int, dict]:
        with self._feedback_lock:
            return self._feedback_unlocked()

    def _filter_monitoring_jump(
        self,
        state: _ServoState,
        snapshot: dict,
        *,
        now: float,
        sample_received_at: float,
    ) -> tuple:
        """Hold one abrupt monitoring sample and accept a persistent change."""

        previous = state.previous_snapshot
        if previous is None:
            state.pending_monitoring_jump = None
            state.pending_monitoring_jump_received_at = None
            return snapshot, None
        previous_force = previous.get("force_n")
        current_force = snapshot.get("force_n")
        if previous_force is None or current_force is None:
            state.pending_monitoring_jump = None
            state.pending_monitoring_jump_received_at = None
            return snapshot, None
        force_jump_n = abs(float(current_force) - float(previous_force))
        angle_jump_deg = abs(
            float(snapshot.get("angle_delta_deg", 0.0))
            - float(previous.get("angle_delta_deg", 0.0))
        )
        suspicious = (
            force_jump_n >= MONITORING_FORCE_JUMP_THRESHOLD_N
            or angle_jump_deg >= MONITORING_ANGLE_JUMP_THRESHOLD_DEG
        )
        if not suspicious:
            state.pending_monitoring_jump = None
            state.pending_monitoring_jump_received_at = None
            return snapshot, None
        if state.pending_monitoring_jump is not None:
            pending_received_at = state.pending_monitoring_jump_received_at
            if pending_received_at is None:
                state.pending_monitoring_jump = None
            elif sample_received_at <= pending_received_at:
                return dict(previous), None
            elif (
                sample_received_at - pending_received_at
                <= MONITORING_JUMP_CONFIRMATION_TIMEOUT_S
            ):
                state.pending_monitoring_jump = None
                state.pending_monitoring_jump_received_at = None
                return snapshot, None

        state.pending_monitoring_jump = snapshot
        state.pending_monitoring_jump_received_at = sample_received_at
        message = None
        if now - state.last_monitoring_jump_log_at >= MONITORING_JUMP_LOG_COOLDOWN_S:
            state.last_monitoring_jump_log_at = now
            message = (
                f"WARNING: ID {state.servo_id} monitoring jump held for confirmation: "
                f"force delta {force_jump_n:.2f} N, angle delta {angle_jump_deg:.2f} deg, "
                f"servo raw {snapshot.get('servo_raw')}, "
                f"encoder {snapshot.get('encoder_angle_deg')} deg"
            )
        return dict(previous), message

    def _feedback_unlocked(self) -> Dict[int, dict]:
        from drivers.as5048a.torque import compute_torque_snapshot  # pylint: disable=import-outside-toplevel

        with self._lock:
            servo_store = self._servo_store
            encoder_reader_store = self._encoder_reader_store
            states = list(self._states.values())
            servo = self._servo
        if servo_store is None or servo is None:
            raise RuntimeError("Feedback is not connected")
        if servo_store.error is not None:
            raise RuntimeError(
                f"Servo feedback reader failed: {servo_store.error}"
            )
        if encoder_reader_store is not None and encoder_reader_store.error is not None:
            raise RuntimeError(
                f"Encoder feedback reader failed: {encoder_reader_store.error}"
            )

        now = time.monotonic()
        result: Dict[int, dict] = {}
        feedback_errors: Dict[int, str] = {}
        with self._feedback_history_lock:
            servo_history = list(self._servo_feedback_history)
            encoder_histories = {
                sensor_id: list(history)
                for sensor_id, history in self._encoder_feedback_histories.items()
            }
        for state in states:
            with self._lock:
                force_control_active = bool(
                    self._running and self._armed and state.active
                )
                manual_speed = self._manual_speed_commands.get(state.servo_id, 0)
            stationary_monitoring = not force_control_active and manual_speed == 0
            sync_tolerance_s = (
                MONITORING_FEEDBACK_SYNC_TOLERANCE_S
                if stationary_monitoring
                else FEEDBACK_SYNC_TOLERANCE_S
            )
            feedback_timeout_s = (
                max(self.config.feedback_timeout_s, MONITORING_FEEDBACK_TIMEOUT_S)
                if stationary_monitoring
                else self.config.feedback_timeout_s
            )
            servo_index = self._servo_index[state.servo_id]
            synchronized, sync_error = self._synchronized_feedback_pair(
                state.servo_id,
                state.sensor_id,
                servo_index,
                now,
                sync_tolerance_s=sync_tolerance_s,
                feedback_timeout_s=feedback_timeout_s,
                servo_history=servo_history,
                encoder_history=encoder_histories.get(state.sensor_id, ()),
            )
            if synchronized is None:
                feedback_errors[state.servo_id] = sync_error
                continue
            servo_entry, encoder_entry, sync_age_s = synchronized
            servo_received_at, telemetry = servo_entry
            encoder_received_at, encoder_sample = encoder_entry
            servo_age_s = max(0.0, now - servo_received_at)
            encoder_age_s = max(0.0, now - encoder_received_at)
            telemetry_age_s = max(servo_age_s, encoder_age_s, sync_age_s)
            try:
                snapshot = compute_torque_snapshot(
                    sample=encoder_sample,
                    servo_id=state.servo_id,
                    motor_name=state.model["motor_name"],
                    servo_raw=telemetry["positions"][servo_index],
                    counts_per_rev=state.model["servo_counts_per_rev"],
                    servo_zero_raw=state.model["servo_zero_raw"],
                    servo_direction=state.model["servo_direction"],
                    encoder_direction=state.model["encoder_direction"],
                    torque_zero_delta_deg=state.model["torque_zero_delta_deg"],
                    calibration_slope=state.model["slope_nmm_per_deg"],
                    calibration_intercept=state.intercept_nmm,
                    calibration_angle_reference_deg=state.model["angle_reference_deg"],
                    spring_k=0.0,
                    moment_arm_mm=state.model["moment_arm_mm"],
                    pull_angle_deg=state.model["pull_angle_deg"],
                    telemetry_age_s=telemetry_age_s,
                    telemetry_stale_after_s=feedback_timeout_s,
                    # A held force spike can hide a real overload indefinitely.
                    # Monitoring has its own one-sample confirmation filter below;
                    # active control must always receive the current measurement.
                    jump_filter_enabled=False,
                    previous_snapshot=state.previous_snapshot,
                )
            except (IndexError, KeyError, TypeError, ValueError, OverflowError) as exc:
                feedback_errors[state.servo_id] = (
                    f"ID {state.servo_id}: force calculation failed "
                    f"({type(exc).__name__})"
                )
                continue
            if snapshot is None or snapshot["force_n"] is None:
                sample_status = str(encoder_sample.get("status", "unknown"))
                feedback_errors[state.servo_id] = (
                    f"ID {state.servo_id}: force unavailable "
                    f"(status={sample_status}, servo_age={servo_age_s:.3f}s, "
                    f"encoder_age={encoder_age_s:.3f}s, sync={sync_age_s:.3f}s)"
                )
                continue
            jump_message = None
            if force_control_active:
                with self._lock:
                    state.pending_monitoring_jump = None
                    state.pending_monitoring_jump_received_at = None
            else:
                with self._lock:
                    snapshot, jump_message = self._filter_monitoring_jump(
                        state,
                        snapshot,
                        now=now,
                        sample_received_at=float(encoder_received_at),
                    )
            if jump_message is not None:
                self._emit(jump_message)
            force_n = float(snapshot["force_n"])
            position = float(telemetry["positions"][servo_index])
            with self._lock:
                if self._states.get(state.servo_id) is not state:
                    raise RuntimeError("Session changed during feedback capture")
                state.previous_snapshot = snapshot
                state.latest_force_n = force_n
                state.latest_position = position
            result[state.servo_id] = {
                "force_n": force_n,
                "position": position,
                "encoder_position_deg": float(snapshot["encoder_angle_deg"]),
                "angle_delta_deg": float(snapshot["angle_delta_deg"]),
                "encoder_received_at": float(encoder_received_at),
                "multiturn_angle_model": bool(snapshot["multiturn_angle_model"]),
                "servo_age_s": servo_age_s,
                "encoder_age_s": encoder_age_s,
                "sync_age_s": sync_age_s,
            }
        self._update_feedback_errors(
            feedback_errors,
            all_feedback_failed=bool(states) and not result,
        )
        return result

    def _validate_calibration_ready(self, servo_id: int) -> _ServoState:
        servo_id = int(servo_id)
        with self._lock:
            if not self._connected:
                raise RuntimeError("Connect before torque calibration")
            if self._faulted:
                raise RuntimeError("Disconnect and reconnect after a fault")
            if self._armed or self._running:
                raise RuntimeError("Stop and disarm force control before torque calibration")
            if any(self._manual_speed_commands.values()):
                raise RuntimeError("Set every manual speed command to 0 before torque calibration")
            state = self._states.get(servo_id)
            if state is None:
                raise ValueError(f"Unknown servo ID: {servo_id}")
            return state

    def calibrate_current_force_as_zero(
        self, servo_id: int, duration_s: float = 1.0
    ) -> dict:
        """Set the current loose position to zero torque without changing slope k."""

        duration_s = float(duration_s)
        if not math.isfinite(duration_s) or duration_s <= 0.0:
            raise ValueError("Calibration duration must be greater than zero")
        if not self._calibration_lock.acquire(blocking=False):
            raise RuntimeError("Another torque calibration action is already in progress")
        try:
            state = self._validate_calibration_ready(servo_id)
            samples: List[float] = []
            encoder_positions: List[float] = []
            previous_received_at: Optional[float] = None
            started_at = time.monotonic()
            capture_deadline = started_at + duration_s
            retry_deadline = started_at + max(duration_s, CALIBRATION_SAMPLE_WAIT_S)
            last_feedback_error = None
            while (
                time.monotonic() < capture_deadline
                or (not samples and time.monotonic() < retry_deadline)
            ):
                self._validate_calibration_ready(servo_id)
                try:
                    feedback = self._feedback()
                except RuntimeError as exc:
                    last_feedback_error = str(exc)
                    time.sleep(0.01)
                    continue
                item = feedback.get(int(servo_id))
                if item is None:
                    last_feedback_error = self.feedback_errors_snapshot().get(
                        int(servo_id),
                        f"Feedback unavailable for ID {servo_id}",
                    )
                else:
                    received_at = item["encoder_received_at"]
                    if received_at != previous_received_at:
                        samples.append(float(item["angle_delta_deg"]))
                        encoder_positions.append(float(item["encoder_position_deg"]))
                        previous_received_at = received_at
                time.sleep(0.005)
            if not samples:
                detail = last_feedback_error or "no valid feedback was received"
                raise RuntimeError(
                    f"No valid magnetic-encoder samples were captured within "
                    f"{max(duration_s, CALIBRATION_SAMPLE_WAIT_S):g} s: {detail}"
                )

            if str(API_ROOT / "python") not in sys.path:
                sys.path.insert(0, str(API_ROOT / "python"))
            from drivers.as5048a.torque import (  # pylint: disable=import-outside-toplevel
                zero_intercept_from_angle_samples,
            )

            slope = float(state.model["slope_nmm_per_deg"])
            result = zero_intercept_from_angle_samples(slope, samples)
            intercept = float(result["intercept_nmm"])
            self._validate_calibration_ready(servo_id)
            records = load_torque_intercept_calibration(self._intercept_calibration_path)
            records[int(servo_id)] = {
                "servo_id": int(servo_id),
                "encoder_sensor_id": int(state.sensor_id),
                "slope_nmm_per_deg": slope,
                "intercept_nmm": intercept,
                "point_count": 1,
                "method": "loose_position_zero",
                "mean_delta_deg": float(result["mean_delta_deg"]),
                "std_delta_deg": float(result["std_delta_deg"]),
                "sample_count": int(result["sample_count"]),
                "capture_duration_s": duration_s,
                "mean_encoder_position_deg": sum(encoder_positions) / len(encoder_positions),
                "calibrated_at_unix_s": time.time(),
            }
            save_torque_intercept_calibration(self._intercept_calibration_path, records)
            with self._lock:
                current_state = self._states.get(int(servo_id))
                if current_state is not state:
                    raise RuntimeError("Session changed while saving torque calibration")
                state.intercept_nmm = intercept
                state.baseline_force_n = 0.0
            self._emit(
                f"Servo {servo_id} torque zero calibrated: b={intercept:.6g} Nmm "
                f"from {len(samples)} samples"
            )
            return dict(records[int(servo_id)])
        finally:
            self._calibration_lock.release()

    def calibrate_all_current_force_as_zero(
        self, duration_s: float = 1.0
    ) -> Dict[int, dict]:
        """Set all current loose positions to zero torque in one capture window."""

        duration_s = float(duration_s)
        if not math.isfinite(duration_s) or duration_s <= 0.0:
            raise ValueError("Calibration duration must be greater than zero")
        if not self._calibration_lock.acquire(blocking=False):
            raise RuntimeError("Another torque calibration action is already in progress")
        try:
            states = {
                servo_id: self._validate_calibration_ready(servo_id)
                for servo_id in SERVO_IDS
            }
            samples_by_servo: Dict[int, List[float]] = {
                servo_id: [] for servo_id in SERVO_IDS
            }
            encoder_positions_by_servo: Dict[int, List[float]] = {
                servo_id: [] for servo_id in SERVO_IDS
            }
            previous_received_at: Dict[int, Optional[float]] = {
                servo_id: None for servo_id in SERVO_IDS
            }
            started_at = time.monotonic()
            capture_deadline = started_at + duration_s
            retry_deadline = started_at + max(duration_s, CALIBRATION_SAMPLE_WAIT_S)
            last_feedback_errors: Dict[int, str] = {}
            while (
                time.monotonic() < capture_deadline
                or (
                    any(not samples_by_servo[servo_id] for servo_id in SERVO_IDS)
                    and time.monotonic() < retry_deadline
                )
            ):
                for servo_id in SERVO_IDS:
                    self._validate_calibration_ready(servo_id)
                try:
                    feedback = self._feedback()
                except RuntimeError as exc:
                    last_feedback_errors = {
                        servo_id: str(exc) for servo_id in SERVO_IDS
                    }
                    time.sleep(0.01)
                    continue
                last_feedback_errors.update(self.feedback_errors_snapshot())
                for servo_id in SERVO_IDS:
                    item = feedback.get(servo_id)
                    if item is None:
                        continue
                    received_at = item["encoder_received_at"]
                    if received_at != previous_received_at[servo_id]:
                        samples_by_servo[servo_id].append(
                            float(item["angle_delta_deg"])
                        )
                        encoder_positions_by_servo[servo_id].append(
                            float(item["encoder_position_deg"])
                        )
                        previous_received_at[servo_id] = received_at
                time.sleep(0.005)

            missing = [servo_id for servo_id in SERVO_IDS if not samples_by_servo[servo_id]]
            if missing:
                details = "; ".join(
                    last_feedback_errors.get(
                        servo_id, f"feedback unavailable for ID {servo_id}"
                    )
                    for servo_id in missing
                )
                raise RuntimeError(
                    "No valid magnetic-encoder samples were captured within "
                    f"{max(duration_s, CALIBRATION_SAMPLE_WAIT_S):g} s: {details}"
                )

            if str(API_ROOT / "python") not in sys.path:
                sys.path.insert(0, str(API_ROOT / "python"))
            from drivers.as5048a.torque import (  # pylint: disable=import-outside-toplevel
                zero_intercept_from_angle_samples,
            )

            records = load_torque_intercept_calibration(self._intercept_calibration_path)
            calibrated_at = time.time()
            results: Dict[int, dict] = {}
            for servo_id in SERVO_IDS:
                state = states[servo_id]
                samples = samples_by_servo[servo_id]
                result = zero_intercept_from_angle_samples(
                    float(state.model["slope_nmm_per_deg"]),
                    samples,
                )
                intercept = float(result["intercept_nmm"])
                records[servo_id] = {
                    "servo_id": int(servo_id),
                    "encoder_sensor_id": int(state.sensor_id),
                    "slope_nmm_per_deg": float(state.model["slope_nmm_per_deg"]),
                    "intercept_nmm": intercept,
                    "point_count": 1,
                    "method": "loose_position_zero",
                    "mean_delta_deg": float(result["mean_delta_deg"]),
                    "std_delta_deg": float(result["std_delta_deg"]),
                    "sample_count": int(result["sample_count"]),
                    "capture_duration_s": duration_s,
                    "mean_encoder_position_deg": (
                        sum(encoder_positions_by_servo[servo_id])
                        / len(encoder_positions_by_servo[servo_id])
                    ),
                    "calibrated_at_unix_s": calibrated_at,
                }
                results[servo_id] = dict(records[servo_id])

            for servo_id in SERVO_IDS:
                self._validate_calibration_ready(servo_id)
            save_torque_intercept_calibration(self._intercept_calibration_path, records)
            with self._lock:
                for servo_id, state in states.items():
                    if self._states.get(servo_id) is not state:
                        raise RuntimeError("Session changed while saving torque calibration")
                for servo_id, state in states.items():
                    state.intercept_nmm = records[servo_id]["intercept_nmm"]
                    state.baseline_force_n = 0.0
            self._emit(
                f"All {len(SERVO_IDS)} tendon torque zeros calibrated from a "
                f"{duration_s:g} s capture"
            )
            return results
        finally:
            self._calibration_lock.release()

    def remember_encoder_position_for_calibration(self, servo_id: int) -> dict:
        """Persist the current magnetic-encoder position for future auto calibration."""

        if not self._calibration_lock.acquire(blocking=False):
            raise RuntimeError("Another torque calibration action is already in progress")
        try:
            state = self._validate_calibration_ready(servo_id)
            feedback = self._feedback()
            item = feedback.get(int(servo_id))
            if item is None:
                reason = self.feedback_errors_snapshot().get(
                    int(servo_id),
                    f"Feedback unavailable for ID {servo_id}",
                )
                raise RuntimeError(f"Remember calibration position stopped: {reason}")
            if not item["multiturn_angle_model"]:
                raise RuntimeError("Multi-turn magnetic-encoder position is unavailable")
            record = {
                "servo_id": int(servo_id),
                "encoder_sensor_id": int(state.sensor_id),
                "encoder_position_deg": float(item["encoder_position_deg"]),
                "recorded_at_unix_s": time.time(),
            }
            with self._lock:
                positions = dict(self._encoder_calibration_positions)
            positions[int(servo_id)] = record
            save_encoder_calibration_positions(self._encoder_position_path, positions)
            with self._lock:
                self._encoder_calibration_positions = positions
            self._emit(
                f"Remembered servo {servo_id} calibration encoder position: "
                f"{record['encoder_position_deg']:.6f} deg"
            )
            return dict(record)
        finally:
            self._calibration_lock.release()

    def capture_baseline(self, duration_s: float = 0.5) -> Dict[int, float]:
        if not self._calibration_lock.acquire(blocking=False):
            raise RuntimeError("Torque calibration is already in progress")
        try:
            if not self._baseline_lock.acquire(blocking=False):
                raise RuntimeError("Baseline capture is already in progress")
            try:
                return self._capture_baseline(duration_s)
            finally:
                self._baseline_lock.release()
        finally:
            self._calibration_lock.release()

    def _capture_baseline(self, duration_s: float = 0.5) -> Dict[int, float]:
        """Capture a median baseline without moving any servo."""

        if not self.connected:
            raise RuntimeError("Connect before capturing a baseline")
        with self._lock:
            if self._running or self._armed:
                raise RuntimeError(
                    "Stop and disarm force control before capturing a new baseline"
                )
            if any(self._manual_speed_commands.values()):
                raise RuntimeError("Stop manual speed control before capturing a baseline")
        values: Dict[int, List[float]] = {servo_id: [] for servo_id in SERVO_IDS}
        deadline = time.monotonic() + max(0.1, float(duration_s))
        last_error: Optional[Exception] = None
        while time.monotonic() < deadline:
            try:
                feedback = self._feedback()
            except Exception as exc:
                last_error = exc
                time.sleep(0.01)
                continue
            for servo_id, item in feedback.items():
                values[servo_id].append(item["force_n"])
            time.sleep(0.01)
        missing = [servo_id for servo_id, samples in values.items() if not samples]
        if missing:
            servo_id = missing[0]
            if last_error is not None:
                raise RuntimeError(
                    f"Unable to capture baseline for servo {servo_id}: {last_error}"
                ) from last_error
            raise RuntimeError(
                "No baseline samples for servo "
                + ", ".join(str(servo_id) for servo_id in missing)
            )

        baselines = {
            servo_id: float(median(samples))
            for servo_id, samples in values.items()
        }
        with self._lock:
            if not self._connected:
                raise RuntimeError("Session disconnected during baseline capture")
            for servo_id, baseline in baselines.items():
                state = self._states[servo_id]
                state.baseline_force_n = baseline
                state.pending_monitoring_jump = None
                state.pending_monitoring_jump_received_at = None
                state.target_force_n = min(
                    self.config.max_target_force_n,
                    max(self.config.min_target_force_n, baseline),
                )
                state.ramp_start_n = state.target_force_n
                state.ramp_end_n = state.target_force_n
                state.ramp_started_at = None
        return baselines

    def arm(
        self,
        *,
        kp: float,
        ki: float,
        kd: float,
        max_speed: float,
        control_sign: float,
        hard_force_limit_n: Optional[float] = None,
        hard_force_limits_n: Optional[Dict[str, float]] = None,
    ) -> None:
        if not self.connected:
            raise RuntimeError("Connect before arming")
        if self.manual_feedback_recovering:
            raise RuntimeError("Wait for feedback recovery before arming")
        with self._lock:
            if self._faulted:
                raise RuntimeError("Disconnect and reconnect after a fault")
            if self._armed or self._running:
                raise RuntimeError("Stop and disarm force control before arming")
            if max_speed <= 0.0:
                raise ValueError("max_speed must be greater than zero")
            if hard_force_limits_n is None:
                if hard_force_limit_n is None or hard_force_limit_n <= 0.0:
                    raise ValueError("hard force limits must be greater than zero")
                limits_by_tendon = {
                    tendon: float(hard_force_limit_n) for tendon in SERVO_NAMES
                }
            else:
                if not isinstance(hard_force_limits_n, dict):
                    raise ValueError("hard_force_limits_n must be an object")
                limits_by_tendon = {}
                for tendon in SERVO_NAMES:
                    if tendon not in hard_force_limits_n:
                        raise ValueError(f"hard_force_limits_n must define {tendon}")
                    limit = _finite_number(
                        hard_force_limits_n[tendon],
                        f"hard_force_limits_n[{tendon}]",
                    )
                    if limit <= 0.0:
                        raise ValueError(
                            f"hard_force_limits_n[{tendon}] must be greater than zero"
                        )
                    limits_by_tendon[tendon] = limit
            for state in self._states.values():
                if limits_by_tendon[state.tendon] <= state.target_force_n:
                    raise ValueError(
                        f"hard force limit for {state.tendon} must be above its selected target"
                    )
        if control_sign not in (-1.0, 1.0):
            raise ValueError("control_sign must be either -1 or 1")

        if str(API_ROOT / "python") not in sys.path:
            sys.path.insert(0, str(API_ROOT / "python"))
        from controller.servo.force_control import PID, PIDConfig, ForceControlConfig

        with self._lock:
            if any(self._manual_speed_commands.values()):
                raise RuntimeError("Stop manual speed control before arming")
            for state in self._states.values():
                state.controller = ForcePIDControllerWithConfig(
                    PID(
                        PIDConfig(
                            kp=float(kp),
                            ki=float(ki),
                            kd=float(kd),
                            output_min=-float(max_speed),
                            output_max=float(max_speed),
                            integral_min=-float(max_speed),
                            integral_max=float(max_speed),
                        )
                    ),
                    ForceControlConfig(
                        target_force_n=state.target_force_n,
                        max_speed=float(max_speed),
                        hard_force_limit_n=limits_by_tendon[state.tendon],
                        min_target_force_n=self.config.min_target_force_n,
                        max_feedback_age_s=self.config.feedback_timeout_s,
                        control_sign=float(control_sign),
                    ),
                )
                state.active = True
            self._last_parameters = {
                "kp": float(kp),
                "ki": float(ki),
                "kd": float(kd),
                "max_speed": float(max_speed),
                "hard_force_limits_n": dict(limits_by_tendon),
                "control_sign": float(control_sign),
            }
            self._armed = True
            self._faulted = False
        self._emit("Armed. Press Start to send force-control speed commands.")

    def start(self) -> None:
        with self._lock:
            if not self._connected or not self._armed:
                raise RuntimeError("Connect and arm before starting control")
            if self._faulted:
                raise RuntimeError("Disconnect and reconnect after a fault")
            if self._running:
                raise RuntimeError("Force control is already running")
            self._running = True
        self._emit("Force control started.")

    def stop_all(self) -> None:
        with self._lock:
            self._running = False
            self._armed = False
            for state in self._states.values():
                state.active = False
            self._manual_speed_commands = {
                servo_id: 0 for servo_id in SERVO_IDS
            }
        self._send_zero_all()
        self._emit("All servos stopped and disarmed.")

    def stop_servo(self, servo_id: int) -> None:
        with self._command_lock:
            with self._lock:
                state = self._states.get(int(servo_id))
                if state is None:
                    raise ValueError(f"Unknown servo ID: {servo_id}")
                state.active = False
                self._manual_speed_commands[int(servo_id)] = 0
                servo = self._servo
            if servo is not None:
                servo.send_speed_by_id(int(servo_id), 0)

    def set_target(self, servo_id: int, target_force_n: float, mode: str, duration_s: float) -> None:
        target = float(target_force_n)
        if target < self.config.min_target_force_n or target > self.config.max_target_force_n:
            raise ValueError(
                f"Target force must be between {self.config.min_target_force_n:g} "
                f"and {self.config.max_target_force_n:g} N"
            )
        if mode not in ("Step", "Ramp"):
            raise ValueError("mode must be Step or Ramp")
        duration = float(duration_s)
        if duration <= 0.0:
            raise ValueError("Ramp duration must be greater than zero")
        with self._lock:
            state = self._states.get(int(servo_id))
            if state is None:
                raise ValueError(f"Unknown servo ID: {servo_id}")
            if mode == "Step":
                state.target_force_n = target
                state.ramp_started_at = None
                state.ramp_start_n = target
                state.ramp_end_n = target
            else:
                state.ramp_start_n = state.target_force_n
                state.ramp_end_n = target
                state.ramp_duration_s = duration
                state.ramp_started_at = time.monotonic()
        self._emit(f"Servo {servo_id} target set to {target:.3f} N ({mode}).")

    def set_manual_speed(self, servo_id: int, speed: int) -> None:
        """Send one persistent raw speed command while force control is stopped."""

        servo_id = int(servo_id)
        speed = int(speed)
        if speed not in MANUAL_SPEED_PRESETS:
            raise ValueError(
                f"Manual speed must be one of {MANUAL_SPEED_PRESETS}"
            )

        with self._command_lock:
            with self._lock:
                if not self._connected or self._servo is None:
                    raise RuntimeError("Connect before using speed control")
                if self._armed or self._running:
                    raise RuntimeError("Stop force control before using speed control")
                if servo_id not in self._states:
                    raise ValueError(f"Unknown servo ID: {servo_id}")
                servo = self._servo
                self._manual_speed_commands[servo_id] = speed
            try:
                servo.send_speed_by_id(servo_id, speed)
            except Exception as exc:
                with self._lock:
                    self._manual_speed_commands[servo_id] = 0
                send_error = exc
            else:
                send_error = None

        if send_error is not None:
            self._handle_fault(send_error)
            raise send_error

        self._emit(f"Servo {servo_id} manual speed set to {speed}.")

    def _update_ramp(self, state: _ServoState, now: float) -> None:
        if state.ramp_started_at is None:
            return
        progress = (now - state.ramp_started_at) / state.ramp_duration_s
        if progress >= 1.0:
            state.target_force_n = state.ramp_end_n
            state.ramp_started_at = None
            return
        progress = max(0.0, progress)
        state.target_force_n = state.ramp_start_n + (
            state.ramp_end_n - state.ramp_start_n
        ) * progress

    def _publish(self, sample: ControlSample) -> None:
        self._publish_bounded(self._sample_queue, sample)

    @staticmethod
    def _publish_bounded(sample_queue: queue.Queue, sample: Any) -> None:
        try:
            sample_queue.put_nowait(sample)
        except queue.Full:
            try:
                sample_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                sample_queue.put_nowait(sample)
            except queue.Full:
                pass

    def _loop(self, stop_event: threading.Event) -> None:
        period_s = 1.0 / self.config.control_hz
        next_tick = time.monotonic()
        while not stop_event.is_set():
            with self._lock:
                if (
                    self._stop_event is not stop_event
                    or self._control_thread is not threading.current_thread()
                ):
                    break
            try:
                feedback = self._feedback()
                with self._lock:
                    states = list(self._states.values())
                    feedback_errors = dict(self._feedback_errors)
                    force_control_active = self._running and self._armed
                if states and not feedback:
                    details = "; ".join(
                        feedback_errors[servo_id]
                        for servo_id in sorted(feedback_errors)
                    )
                    if force_control_active:
                        raise RuntimeError(f"All force feedback unavailable: {details}")
                    self._handle_manual_feedback_error(
                        RuntimeError(f"All force feedback unavailable: {details}")
                    )
                with self._lock:
                    self._manual_feedback_recovering = bool(feedback_errors)
                now = time.monotonic()
                commands: List[tuple] = []
                for state in states:
                    item = feedback.get(state.servo_id)
                    if item is None:
                        with self._lock:
                            state_active = bool(
                                self._running and self._armed and state.active
                            )
                        if state_active:
                            raise RuntimeError(
                                feedback_errors.get(
                                    state.servo_id,
                                    f"Feedback unavailable for ID {state.servo_id}",
                                )
                            )
                        continue
                    with self._lock:
                        if self._states.get(state.servo_id) is not state:
                            continue
                        active = bool(self._running and self._armed and state.active)
                        if active:
                            self._update_ramp(state, now)
                            controller = state.controller
                            target_force_n = state.target_force_n
                        else:
                            controller = None
                            target_force_n = state.target_force_n
                            manual_speed = self._manual_speed_commands.get(
                                state.servo_id, 0
                            )
                    speed_command = 0
                    if active:
                        if controller is None:
                            raise RuntimeError("Controller is not armed")
                        if controller.config.target_force_n != target_force_n:
                            controller.config = replace(
                                controller.config,
                                target_force_n=target_force_n,
                            )
                        output = controller.step(
                            self._feedback_object(state, item),
                            now=now,
                        )
                        speed_command = output.speed_command
                        commands.append((state.servo_id, speed_command))
                        force_error = output.force_error_n
                    else:
                        speed_command = manual_speed
                        force_error = target_force_n - item["force_n"]
                    self._publish(
                        ControlSample(
                            received_at=time.time(),
                            servo_id=state.servo_id,
                            tendon=state.tendon,
                            force_n=item["force_n"],
                            target_force_n=target_force_n,
                            force_error_n=force_error,
                            speed_command=speed_command,
                            position=item["position"],
                            encoder_position_deg=item["encoder_position_deg"],
                            angle_delta_deg=item["angle_delta_deg"],
                            active=active,
                        )
                    )
                if commands:
                    with self._command_lock:
                        with self._lock:
                            current_session = (
                                self._stop_event is stop_event
                                and self._control_thread is threading.current_thread()
                            )
                            should_send = (
                                current_session
                                and self._running
                                and self._armed
                                and not self._faulted
                            )
                            active_ids = {
                                state.servo_id
                                for state in self._states.values()
                                if state.active
                            }
                            servo = self._servo
                        if should_send and servo is not None:
                            for servo_id, command in commands:
                                if servo_id in active_ids:
                                    servo.send_speed_by_id(servo_id, command)
            except Exception as exc:
                with self._lock:
                    current_session = (
                        self._stop_event is stop_event
                        and self._control_thread is threading.current_thread()
                    )
                if not current_session:
                    break
                if self._handle_manual_feedback_error(exc):
                    next_tick = time.monotonic()
                    stop_event.wait(period_s)
                    continue
                self._handle_fault(exc)
                break

            next_tick += period_s
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0.0:
                stop_event.wait(sleep_s)
            else:
                next_tick = time.monotonic()

    def _feedback_object(self, state: _ServoState, item: dict) -> Any:
        from controller.servo.force_control import ForceFeedback

        return ForceFeedback(
            force_n=float(item["force_n"]),
            servo_position=float(item["position"]),
            servo_age_s=float(item["servo_age_s"]),
            encoder_age_s=float(item["encoder_age_s"]),
        )

    def _handle_fault(self, exc: Exception) -> None:
        message = str(exc)
        with self._lock:
            self._running = False
            self._armed = False
            self._faulted = True
            self._last_fault_message = message
            for state in self._states.values():
                state.active = False
        self._send_zero_all()
        self._emit(f"FAULT: all servos stopped: {message}")

    def _handle_manual_feedback_error(self, exc: Exception) -> bool:
        """Keep a connected, non-running session alive while feedback recovers."""

        warning_message = None
        now = time.monotonic()
        with self._lock:
            force_control_active = self._running and self._armed
            manual_speed_active = any(self._manual_speed_commands.values())
            recovering = self._manual_feedback_recovering
            if force_control_active or not self._connected:
                return False
            self._manual_feedback_recovering = True

        if manual_speed_active:
            with self._lock:
                if now - self._last_feedback_warning_at >= FEEDBACK_LOG_COOLDOWN_S:
                    self._last_feedback_warning_at = now
                    self._feedback_warning_logged = True
                    warning_message = (
                        "WARNING: manual speed is continuing without force feedback; "
                        f"use 0 or STOP ALL to stop: {exc}"
                    )
        elif not recovering:
            with self._lock:
                if now - self._last_feedback_warning_at >= FEEDBACK_LOG_COOLDOWN_S:
                    self._last_feedback_warning_at = now
                    self._feedback_warning_logged = True
                    warning_message = (
                        f"WARNING: feedback unavailable; waiting for recovery: {exc}"
                    )
        if warning_message is not None:
            self._emit(warning_message)
        return True

    def _send_zero_all(self) -> None:
        with self._command_lock:
            servo = self._servo
            if servo is None:
                return
            for servo_id in SERVO_IDS:
                try:
                    servo.send_speed_by_id(servo_id, 0)
                except Exception as exc:
                    self._emit(f"WARNING: failed to stop servo {servo_id}: {exc}")

    def disconnect(self) -> None:
        cleanup_errors: List[str] = []
        with self._lock:
            connect_cancel_event = self._connect_cancel_event
            control_thread = self._control_thread
            servo_reader = self._servo_reader
            encoder_reader = self._encoder_reader
            stop_event = self._stop_event
            encoder = self._encoder
            servo_transport = self._servo_transport
            encoder_transport = self._encoder_transport
            was_connected = self._connected or servo_transport is not None or encoder_transport is not None
            self._running = False
            self._armed = False
            stop_event.set()
            if connect_cancel_event is not None:
                connect_cancel_event.set()
        if was_connected:
            self._send_zero_all()
        if encoder is not None:
            try:
                encoder.stop_stream()
            except Exception as exc:
                cleanup_errors.append(f"encoder stop failed: {exc}")
        if control_thread is not None and control_thread is not threading.current_thread():
            control_thread.join(timeout=1.0)
            if control_thread.is_alive():
                cleanup_errors.append("control thread did not stop within 1 second")
        if servo_reader is not None:
            servo_reader.join(1.0)
            reader_thread = getattr(servo_reader, "_thread", None)
            if reader_thread is not None and reader_thread.is_alive():
                cleanup_errors.append("servo reader did not stop within 1 second")
        if encoder_reader is not None:
            encoder_reader.join(1.0)
            reader_thread = getattr(encoder_reader, "_thread", None)
            if reader_thread is not None and reader_thread.is_alive():
                cleanup_errors.append("encoder reader did not stop within 1 second")
        for transport in (encoder_transport, servo_transport):
            if transport is not None:
                try:
                    transport.close()
                except Exception as exc:
                    cleanup_errors.append(f"transport close failed: {exc}")
        with self._lock:
            self._connected = False
            self._servo = None
            self._encoder = None
            self._servo_transport = None
            self._encoder_transport = None
            self._servo_store = None
            self._encoder_reader_store = None
            self._encoder_stores = {}
            self._encoder_calibration_positions = {}
            self._servo_reader = None
            self._encoder_reader = None
            self._control_thread = None
            self._states = {}
            self._manual_speed_commands = {
                servo_id: 0 for servo_id in SERVO_IDS
            }
            self._manual_feedback_recovering = False
            self._feedback_errors = {}
            self._last_fault_message = None
            self._stop_event = threading.Event()
        if was_connected:
            self._emit("Disconnected.")
        for error in cleanup_errors:
            self._emit(f"WARNING: cleanup incomplete: {error}")

    def shutdown(self) -> None:
        self.disconnect()


class ForcePIDControllerWithConfig:
    """Small adapter to keep the GUI session independent of import timing."""

    def __init__(self, pid: Any, config: Any):
        from controller.servo.force_control import ForcePIDController

        self._controller = ForcePIDController(pid, config)

    @property
    def config(self) -> Any:
        return self._controller.config

    @config.setter
    def config(self, value: Any) -> None:
        self._controller.config = value

    def step(self, feedback: Any, *, now: Optional[float] = None) -> Any:
        return self._controller.step(feedback, now=now)
