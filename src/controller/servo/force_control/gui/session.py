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
from collections import deque
from dataclasses import dataclass, field, replace
from datetime import datetime
from pathlib import Path
from statistics import median
from typing import Any, Callable, Deque, Dict, Iterable, List, Optional, Sequence


CONTROLLER_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = CONTROLLER_ROOT.parent
PROJECT_ROOT = SRC_ROOT.parent
API_ROOT = PROJECT_ROOT / "third_party" / "ServoHardwareAPI"
DEFAULT_FORCE_CONFIG_PATH = CONTROLLER_ROOT / "config_force_control.json"
DEFAULT_CALIBRATION_DIR = API_ROOT / "calibration" / "current"

SERVO_IDS = (1, 2, 3, 4, 5, 6)
SERVO_NAMES = ("ED", "DI", "FDS", "FDP", "LU", "PI")
DEFAULT_ZERO_OFFSET_FILE = DEFAULT_CALIBRATION_DIR / "as5048a_unloaded_zero_offsets.json"


@dataclass(frozen=True)
class GuiForceConfig:
    min_target_force_n: float
    max_target_force_n: float
    target_force_levels_n: tuple
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
    """Load and validate the fixed target levels used by the GUI."""

    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict) or not isinstance(payload.get("default"), dict):
        raise ValueError("Force-control config must contain a default object")

    default = payload["default"]
    minimum = _finite_number(default.get("min_target_force_n"), "min_target_force_n")
    maximum = _finite_number(default.get("max_target_force_n"), "max_target_force_n")
    if minimum < 0.0 or minimum >= maximum:
        raise ValueError("min_target_force_n must be non-negative and below max_target_force_n")

    levels_value = default.get("target_force_levels_n")
    if not isinstance(levels_value, list) or not levels_value:
        raise ValueError("target_force_levels_n must be a non-empty list")
    levels = tuple(_finite_number(value, "target_force_levels_n") for value in levels_value)
    if tuple(sorted(set(levels))) != levels:
        raise ValueError("target_force_levels_n must be strictly increasing")
    if levels[0] < minimum or levels[-1] > maximum:
        raise ValueError("target_force_levels_n must be inside the target force range")

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
    active: bool


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


class _EncoderFanout:
    """Route six AS5048A samples from one reader to six latest stores."""

    def __init__(self, stores: Dict[int, Any]):
        self.stores = stores

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
            store.publish(sample)
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
        self._stop_event = threading.Event()
        self._sample_queue: queue.Queue = queue.Queue(maxsize=3000)
        self._event_queue: queue.Queue = queue.Queue()
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
        self._encoder_stores: Dict[int, Any] = {}
        self._servo_reader = None
        self._encoder_reader = None
        self._control_thread: Optional[threading.Thread] = None
        self._states: Dict[int, _ServoState] = {}
        self._servo_index: Dict[int, int] = {}
        self._last_parameters: Optional[dict] = None
        self._calibration_dir = DEFAULT_CALIBRATION_DIR
        self._zero_offset_path = DEFAULT_ZERO_OFFSET_FILE
        self._zero_offsets: Dict[int, dict] = {}

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

    def _emit(self, message: str) -> None:
        self._event_queue.put(message)
        if self.status_callback is not None:
            self.status_callback(message)

    def drain_events(self) -> List[str]:
        events: List[str] = []
        while True:
            try:
                events.append(self._event_queue.get_nowait())
            except queue.Empty:
                return events

    def drain_samples(self, limit: int = 300) -> List[ControlSample]:
        samples: List[ControlSample] = []
        for _ in range(max(1, int(limit))):
            try:
                samples.append(self._sample_queue.get_nowait())
            except queue.Empty:
                break
                return samples

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
            "encoder_communication": reader_status(encoder_reader, encoder_stores.values()),
            "control_sign": parameters.get("control_sign"),
            "mapping": mapping,
        }

    def _load_models(self, calibration_dir: Path, zero_offset_path: Path) -> Dict[int, _ServoState]:
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
            offset = offsets.get(sensor_id)
            if offset is not None:
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
    ) -> None:
        """Connect and initialize feedback without arming motion."""

        self.disconnect()
        with self._lock:
            self._calibration_dir = Path(calibration_dir)
            self._zero_offset_path = Path(zero_offset_path)
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
        states = self._load_models(Path(calibration_dir), Path(zero_offset_path))
        system = ServoSystem(
            count=6,
            servo_ids=list(SERVO_IDS),
            names=list(SERVO_NAMES),
            as5048a_sensor_map=list(SERVO_IDS),
        )
        self._servo_index = {
            servo_id: system.index_for_servo_id(servo_id) for servo_id in SERVO_IDS
        }

        servo_transport = None
        encoder_transport = None
        try:
            servo_transport = SerialTransport(
                str(servo_port),
                baudrate=921600,
                timeout=min(0.05, self.config.feedback_timeout_s),
            ).open()
            encoder_transport = SerialTransport(
                str(encoder_port),
                baudrate=921600,
                timeout=min(0.05, self.config.feedback_timeout_s),
            ).open()
            servo = ServoControllerDriver(servo_transport, servo_system=system)
            encoder = AS5048ABridgeDriver(encoder_transport)
            servo_store = LatestSampleStore()
            encoder_reader_store = LatestSampleStore()
            encoder_stores = {sensor_id: LatestSampleStore() for sensor_id in SERVO_IDS}
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
                _EncoderFanout(encoder_stores),
                encoder_reader_store,
                stop_event,
                name="force-gui-encoder-reader",
            )

            encoder.stop_stream()
            encoder.set_rate_hz(self.config.encoder_rate_hz)
            encoder.start_all_stream()
            servo_reader.start()
            encoder_reader.start()
            servo_store.wait_for_sample(self.config.feedback_timeout_s * 10.0)
            for store in encoder_stores.values():
                store.wait_for_sample(self.config.feedback_timeout_s * 10.0)

            with self._lock:
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
                self._connected = True
                self._armed = False
                self._running = False
                self._faulted = False
                self._last_parameters = None

            self._emit("Feedback connected; capturing stable initial force...")
            self.capture_baseline()
            self._control_thread = threading.Thread(
                target=self._loop,
                name="force-gui-control-loop",
                daemon=True,
            )
            self._control_thread.start()
            self._emit("Connected and disarmed. No motion command is being sent.")
        except Exception:
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
            raise

    @staticmethod
    def _accept_servo(parsed: Any) -> Optional[dict]:
        if isinstance(parsed, dict) and parsed.get("kind") == "telemetry":
            return parsed
        return None

    def _feedback(self) -> Dict[int, dict]:
        from drivers.as5048a.torque import compute_torque_snapshot  # pylint: disable=import-outside-toplevel

        with self._lock:
            servo_store = self._servo_store
            encoder_stores = dict(self._encoder_stores)
            states = list(self._states.values())
            servo = self._servo
        if servo_store is None or servo is None:
            raise RuntimeError("Feedback is not connected")

        servo_record = servo_store.snapshot()
        if servo_record is None:
            raise RuntimeError("Servo telemetry is unavailable")
        now = time.monotonic()
        telemetry = servo_record.value
        result: Dict[int, dict] = {}
        for state in states:
            servo_index = self._servo_index[state.servo_id]
            if not telemetry["valid"][servo_index]:
                raise RuntimeError(f"Servo telemetry is invalid for ID {state.servo_id}")
            encoder_record = encoder_stores[state.sensor_id].snapshot()
            if encoder_record is None:
                raise RuntimeError(f"AS5048A telemetry is unavailable for CS{state.sensor_id}")
            servo_age_s = max(0.0, now - servo_record.received_at)
            encoder_age_s = max(0.0, now - encoder_record.received_at)
            sync_age_s = abs(encoder_record.received_at - servo_record.received_at)
            snapshot = compute_torque_snapshot(
                sample=encoder_record.value,
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
                telemetry_age_s=sync_age_s,
                telemetry_stale_after_s=self.config.feedback_timeout_s,
                previous_snapshot=state.previous_snapshot,
            )
            if snapshot is None or snapshot["force_n"] is None:
                raise RuntimeError(f"Force estimate is unavailable for ID {state.servo_id}")
            state.previous_snapshot = snapshot
            force_n = float(snapshot["force_n"])
            position = float(telemetry["positions"][servo_index])
            state.latest_force_n = force_n
            state.latest_position = position
            result[state.servo_id] = {
                "force_n": force_n,
                "position": position,
                "servo_age_s": servo_age_s,
                "encoder_age_s": encoder_age_s,
            }
        return result

    def capture_baseline(self, duration_s: float = 0.5) -> Dict[int, float]:
        """Capture a median baseline without moving any servo."""

        if not self.connected:
            raise RuntimeError("Connect before capturing a baseline")
        with self._lock:
            if self._running:
                raise RuntimeError("Stop control before capturing a new baseline")
        values: Dict[int, List[float]] = {servo_id: [] for servo_id in SERVO_IDS}
        deadline = time.monotonic() + max(0.1, float(duration_s))
        while time.monotonic() < deadline:
            feedback = self._feedback()
            for servo_id, item in feedback.items():
                values[servo_id].append(item["force_n"])
            time.sleep(0.01)
        baselines: Dict[int, float] = {}
        with self._lock:
            for servo_id, samples in values.items():
                if not samples:
                    raise RuntimeError(f"No baseline samples for servo {servo_id}")
                baseline = float(median(samples))
                baselines[servo_id] = baseline
                state = self._states[servo_id]
                state.baseline_force_n = baseline
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
        hard_force_limit_n: float,
        control_sign: float,
    ) -> None:
        if not self.connected:
            raise RuntimeError("Connect before arming")
        if max_speed <= 0.0 or hard_force_limit_n <= 0.0:
            raise ValueError("max_speed and hard_force_limit_n must be greater than zero")
        if hard_force_limit_n <= self.config.max_target_force_n:
            raise ValueError("hard_force_limit_n must be above the maximum target force")
        if control_sign not in (-1.0, 1.0):
            raise ValueError("control_sign must be either -1 or 1")

        if str(API_ROOT / "python") not in sys.path:
            sys.path.insert(0, str(API_ROOT / "python"))
        from controller.servo.force_control import PID, PIDConfig, ForceControlConfig

        with self._lock:
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
                        hard_force_limit_n=float(hard_force_limit_n),
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
                "hard_force_limit_n": float(hard_force_limit_n),
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
            self._running = True
        self._emit("Force control started.")

    def stop_all(self) -> None:
        with self._lock:
            self._running = False
            self._armed = False
            for state in self._states.values():
                state.active = False
        self._send_zero_all()
        self._emit("All servos stopped and disarmed.")

    def stop_servo(self, servo_id: int) -> None:
        with self._lock:
            state = self._states.get(int(servo_id))
            if state is None:
                raise ValueError(f"Unknown servo ID: {servo_id}")
            state.active = False
        if self._servo is not None:
            self._servo.send_speed_by_id(int(servo_id), 0)

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
        try:
            self._sample_queue.put_nowait(sample)
        except queue.Full:
            try:
                self._sample_queue.get_nowait()
            except queue.Empty:
                pass
            try:
                self._sample_queue.put_nowait(sample)
            except queue.Full:
                pass

    def _loop(self) -> None:
        period_s = 1.0 / self.config.control_hz
        next_tick = time.monotonic()
        while not self._stop_event.is_set():
            try:
                feedback = self._feedback()
                now = time.monotonic()
                with self._lock:
                    running = self._running
                    armed = self._armed
                    states = list(self._states.values())
                commands: List[tuple] = []
                for state in states:
                    item = feedback[state.servo_id]
                    speed_command = 0
                    if running and armed and state.active:
                        self._update_ramp(state, now)
                        if state.controller is None:
                            raise RuntimeError("Controller is not armed")
                        state.controller.config = replace(
                            state.controller.config,
                            target_force_n=state.target_force_n,
                        )
                        output = state.controller.step(
                            self._feedback_object(state, item),
                            now=now,
                        )
                        speed_command = output.speed_command
                        commands.append((state.servo_id, speed_command))
                        force_error = output.force_error_n
                    else:
                        force_error = state.target_force_n - item["force_n"]
                    self._publish(
                        ControlSample(
                            received_at=time.time(),
                            servo_id=state.servo_id,
                            tendon=state.tendon,
                            force_n=item["force_n"],
                            target_force_n=state.target_force_n,
                            force_error_n=force_error,
                            speed_command=speed_command,
                            position=item["position"],
                            active=bool(running and armed and state.active),
                        )
                    )
                if running and armed and self._servo is not None:
                    for servo_id, command in commands:
                        self._servo.send_speed_by_id(servo_id, command)
            except Exception as exc:
                self._handle_fault(exc)
                break

            next_tick += period_s
            sleep_s = next_tick - time.monotonic()
            if sleep_s > 0.0:
                self._stop_event.wait(sleep_s)
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
        with self._lock:
            self._running = False
            self._armed = False
            self._faulted = True
            for state in self._states.values():
                state.active = False
        self._send_zero_all()
        self._emit(f"FAULT: all servos stopped: {exc}")

    def _send_zero_all(self) -> None:
        servo = self._servo
        if servo is None:
            return
        for servo_id in SERVO_IDS:
            try:
                servo.send_speed_by_id(servo_id, 0)
            except Exception as exc:
                self._emit(f"WARNING: failed to stop servo {servo_id}: {exc}")

    def disconnect(self) -> None:
        with self._lock:
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
        if was_connected:
            self._send_zero_all()
        if encoder is not None:
            try:
                encoder.stop_stream()
            except Exception:
                pass
        if control_thread is not None and control_thread is not threading.current_thread():
            control_thread.join(timeout=1.0)
        if servo_reader is not None:
            servo_reader.join(1.0)
        if encoder_reader is not None:
            encoder_reader.join(1.0)
        for transport in (encoder_transport, servo_transport):
            if transport is not None:
                try:
                    transport.close()
                except Exception:
                    pass
        with self._lock:
            self._connected = False
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
            self._stop_event = threading.Event()
        if was_connected:
            self._emit("Disconnected.")

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
