"""Project-owned ServoHardwareAPI session and latest telemetry access."""

from __future__ import annotations

import copy
import sys
import threading
import time
from collections import Counter, deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Optional

from controller.servo.command_sender import ServoCommandSender
from controller.servo.command_dispatcher import (
    CommandTicket,
    ServoCommandDispatcher,
)
from controller.servo.force_control.runner import LatestSampleStore, ReaderTask


PROJECT_ROOT = Path(__file__).resolve().parents[3]
API_ROOT = PROJECT_ROOT / "third_party" / "ServoHardwareAPI"


@dataclass(frozen=True)
class HardwareFeedback:
    servo: dict[str, Any]
    servo_received_at: float
    encoder_by_sensor: dict[int, Any]
    encoder_received_at: dict[int, float]
    sampled_at: float

    @property
    def servo_age_s(self) -> float:
        return max(0.0, self.sampled_at - self.servo_received_at)

    def encoder_age_s(self, sensor_id: int) -> float | None:
        received_at = self.encoder_received_at.get(int(sensor_id))
        if received_at is None:
            return None
        return max(0.0, self.sampled_at - received_at)


class _EncoderDiagnostics:
    """Keep bounded raw-line context and per-sensor receive statistics."""

    RECENT_LINE_LIMIT = 2048
    RAW_LINE_LIMIT = 512

    def __init__(
        self,
        *,
        port: str,
        baudrate: int,
        rate_hz: float,
        sensor_ids: tuple[int, ...],
    ) -> None:
        self._lock = threading.Lock()
        self._port = port
        self._baudrate = baudrate
        self._rate_hz = rate_hz
        self._sensor_ids = frozenset(sensor_ids)
        self._read_calls = 0
        self._empty_reads = 0
        self._nonempty_lines = 0
        self._unparsed_lines = 0
        self._status_lines = 0
        self._unassigned_samples = 0
        self._unknown_sensor_samples = 0
        self._parsed_kind_counts: Counter[str] = Counter()
        self._sensor_stats = {
            str(sensor_id): {
                "sample_count": 0,
                "first_received_monotonic_s": None,
                "last_received_monotonic_s": None,
                "max_receive_gap_s": 0.0,
                "last_device_ms": None,
                "last_status": None,
            }
            for sensor_id in sorted(self._sensor_ids)
        }
        self._recent_lines: deque[dict[str, Any]] = deque(
            maxlen=self.RECENT_LINE_LIMIT
        )
        self._reader_errors: dict[str, str | None] = {
            "servo": None,
            "encoder": None,
        }

    def record(
        self,
        raw_line: str,
        parsed: Any,
        received_at: float,
        *,
        parse_error: str | None = None,
    ) -> None:
        raw = str(raw_line).rstrip("\r\n")
        parsed_kind = (
            str(parsed[0])
            if isinstance(parsed, (tuple, list)) and parsed
            else None
        )
        payload = (
            parsed[1]
            if isinstance(parsed, (tuple, list)) and len(parsed) >= 2
            else None
        )
        is_sample = parsed_kind in {"data", "sensor_data"}
        sensor_id: int | None = None
        if is_sample and isinstance(payload, dict):
            try:
                sensor_id = int(payload["sensor_id"])
            except (KeyError, TypeError, ValueError):
                pass

        with self._lock:
            self._read_calls += 1
            if not raw:
                self._empty_reads += 1
                return

            self._nonempty_lines += 1
            kind_key = parsed_kind or "unparsed"
            self._parsed_kind_counts[kind_key] += 1
            if parsed_kind is None:
                self._unparsed_lines += 1
            elif parsed_kind == "status":
                self._status_lines += 1

            if is_sample:
                if sensor_id is None:
                    self._unassigned_samples += 1
                elif sensor_id not in self._sensor_ids:
                    self._unknown_sensor_samples += 1
                else:
                    stats = self._sensor_stats[str(sensor_id)]
                    previous_at = stats["last_received_monotonic_s"]
                    if previous_at is None:
                        stats["first_received_monotonic_s"] = received_at
                    else:
                        stats["max_receive_gap_s"] = max(
                            float(stats["max_receive_gap_s"]),
                            max(0.0, received_at - float(previous_at)),
                        )
                    stats["sample_count"] += 1
                    stats["last_received_monotonic_s"] = received_at
                    stats["last_device_ms"] = payload.get("device_ms")
                    stats["last_status"] = payload.get("status")

            event: dict[str, Any] = {
                "received_monotonic_s": received_at,
                "parsed_kind": parsed_kind,
                "sensor_id": sensor_id,
                "device_ms": payload.get("device_ms")
                if isinstance(payload, dict)
                else None,
                "status": payload.get("status")
                if isinstance(payload, dict)
                else None,
                "raw_line": raw[: self.RAW_LINE_LIMIT],
                "raw_line_truncated": len(raw) > self.RAW_LINE_LIMIT,
            }
            if parse_error is not None:
                event["parse_error"] = parse_error
            self._recent_lines.append(event)

    def set_reader_errors(
        self,
        *,
        servo_error: Exception | None,
        encoder_error: Exception | None,
    ) -> None:
        with self._lock:
            self._reader_errors = {
                "servo": None
                if servo_error is None
                else f"{type(servo_error).__name__}: {servo_error}",
                "encoder": None
                if encoder_error is None
                else f"{type(encoder_error).__name__}: {encoder_error}",
            }

    def snapshot(self) -> dict[str, Any]:
        with self._lock:
            return {
                "encoder_port": self._port,
                "baudrate": self._baudrate,
                "configured_rate_hz": self._rate_hz,
                "recent_line_limit": self.RECENT_LINE_LIMIT,
                "counts": {
                    "read_calls": self._read_calls,
                    "empty_reads": self._empty_reads,
                    "nonempty_lines": self._nonempty_lines,
                    "unparsed_lines": self._unparsed_lines,
                    "status_lines": self._status_lines,
                    "unassigned_sensor_samples": self._unassigned_samples,
                    "unknown_sensor_samples": self._unknown_sensor_samples,
                    "parsed_kinds": dict(self._parsed_kind_counts),
                },
                "sensor_stats": {
                    sensor_id: dict(stats)
                    for sensor_id, stats in self._sensor_stats.items()
                },
                "reader_errors": dict(self._reader_errors),
                "recent_lines": list(self._recent_lines),
            }


class _EncoderFanout:
    def __init__(self, stores: dict[int, LatestSampleStore]) -> None:
        self._stores = stores

    def __call__(self, parsed: Any) -> None:
        if not isinstance(parsed, (tuple, list)) or len(parsed) != 2:
            return
        if parsed[0] not in ("sensor_data", "data") or not isinstance(parsed[1], dict):
            return
        try:
            sensor_id = int(parsed[1]["sensor_id"])
        except (KeyError, TypeError, ValueError):
            return
        store = self._stores.get(sensor_id)
        if store is not None:
            store.publish(parsed[1])


def _accept_servo(parsed: Any) -> Optional[dict[str, Any]]:
    if isinstance(parsed, dict) and parsed.get("kind") == "telemetry":
        return parsed
    return None


class ServoHardwareSession:
    """Own both serial readers and expose project-level feedback snapshots."""

    TELEMETRY_CAPTURE_LIMIT = 50000

    def __init__(
        self,
        *,
        servo_ids: tuple[int, ...],
        servo_names: tuple[str, ...],
        encoder_sensor_ids: tuple[int, ...],
        servo_port: str,
        encoder_port: str | None = None,
        baudrate: int = 921600,
        serial_timeout_s: float = 0.05,
        encoder_rate_hz: float = 70.0,
        capture_encoder_diagnostics: bool = False,
        batch_encoder_reads: bool = False,
        encoder_batch_max_lines: int = 128,
        read_only: bool = False,
        capture_telemetry: bool = False,
    ) -> None:
        if len(servo_ids) != len(servo_names):
            raise ValueError("servo_ids and servo_names must have equal lengths")
        if len(servo_ids) != len(encoder_sensor_ids):
            raise ValueError("servo_ids and encoder_sensor_ids must have equal lengths")
        self.servo_ids = tuple(int(value) for value in servo_ids)
        self.servo_names = tuple(str(value) for value in servo_names)
        self.encoder_sensor_ids = tuple(int(value) for value in encoder_sensor_ids)
        self.servo_port = str(servo_port)
        self.encoder_port = None if encoder_port is None else str(encoder_port)
        self.baudrate = int(baudrate)
        self.serial_timeout_s = float(serial_timeout_s)
        self.encoder_rate_hz = float(encoder_rate_hz)
        self.capture_encoder_diagnostics = bool(capture_encoder_diagnostics)
        self.batch_encoder_reads = bool(batch_encoder_reads)
        self.read_only = bool(read_only)
        self.capture_telemetry = bool(capture_telemetry)
        self.encoder_batch_max_lines = int(encoder_batch_max_lines)
        if self.encoder_batch_max_lines <= 0:
            raise ValueError("encoder_batch_max_lines must be positive")
        self.system: Any = None
        self.servo_driver: Any = None
        self.encoder_driver: Any = None
        self.sender: ServoCommandSender | None = None
        self.dispatcher: ServoCommandDispatcher | None = None
        self._servo_transport: Any = None
        self._encoder_transport: Any = None
        self._servo_store: LatestSampleStore | None = None
        self._encoder_reader_store: LatestSampleStore | None = None
        self._encoder_stores: dict[int, LatestSampleStore] = {}
        self._encoder_diagnostics: _EncoderDiagnostics | None = None
        self._stop_event: threading.Event | None = None
        self._servo_reader: ReaderTask | None = None
        self._encoder_reader: ReaderTask | None = None
        self._encoder_batch_buffer = bytearray()
        self._encoder_pending_lines: deque[tuple[str, float]] = deque()
        self._telemetry_capture_lock = threading.Lock()
        self._telemetry_samples: deque[dict[str, Any]] = deque(
            maxlen=self.TELEMETRY_CAPTURE_LIMIT
        )
        self._telemetry_samples_seen = 0
        self._telemetry_samples_dropped = 0
        self._last_stop_state: dict[str, Any] = {
            "latched": False,
            "stop_requested_monotonic_s": None,
            "stop_command_sequence": None,
            "pc_write_completed_monotonic_s": None,
            "pc_write_status": "not_requested",
            "physical_stop_confirmed": False,
            "physical_stop_confirmation": "unobserved",
            "tension_safe_confirmed": False,
            "tension_safe_confirmation": "unobserved",
        }

    @property
    def connected(self) -> bool:
        return self.sender is not None

    def connect(self, *, startup_timeout_s: float = 3.0) -> None:
        self.close()
        with self._telemetry_capture_lock:
            self._telemetry_samples.clear()
            self._telemetry_samples_seen = 0
            self._telemetry_samples_dropped = 0
        self._encoder_diagnostics = (
            _EncoderDiagnostics(
                port=self.encoder_port or "",
                baudrate=self.baudrate,
                rate_hz=self.encoder_rate_hz,
                sensor_ids=self.encoder_sensor_ids,
            )
            if self.capture_encoder_diagnostics and self.encoder_port is not None
            else None
        )
        if not API_ROOT.is_dir():
            raise FileNotFoundError(f"ServoHardwareAPI was not found: {API_ROOT}")
        api_python = str(API_ROOT / "python")
        if api_python not in sys.path:
            sys.path.insert(0, api_python)
        from drivers import AS5048ABridgeDriver, SerialTransport, ServoControllerDriver, ServoSystem

        self.system = ServoSystem(
            count=len(self.servo_ids),
            servo_ids=list(self.servo_ids),
            names=list(self.servo_names),
            as5048a_sensor_map=list(self.encoder_sensor_ids),
        )
        self._servo_transport = SerialTransport(
            self.servo_port,
            baudrate=self.baudrate,
            timeout=self.serial_timeout_s,
        ).open()
        try:
            self.servo_driver = ServoControllerDriver(
                self._servo_transport,
                servo_system=self.system,
            )
            if not self.read_only:
                self.sender = ServoCommandSender(self.servo_driver)
                self.dispatcher = ServoCommandDispatcher(self.sender)
                self.dispatcher.start()
            self._stop_event = threading.Event()
            self._servo_store = LatestSampleStore()
            self._servo_reader = ReaderTask(
                self._read_servo_line
                if self.capture_telemetry
                else self.servo_driver.read_parsed_line,
                self._accept_servo_sample,
                self._servo_store,
                self._stop_event,
                name="csv-player-servo-reader",
            )
            self._servo_reader.start()

            if self.encoder_port is not None:
                self._encoder_transport = SerialTransport(
                    self.encoder_port,
                    baudrate=self.baudrate,
                    timeout=self.serial_timeout_s,
                ).open()
                self.encoder_driver = AS5048ABridgeDriver(self._encoder_transport)
                self._encoder_stores = {
                    sensor_id: LatestSampleStore()
                    for sensor_id in self.encoder_sensor_ids
                }
                self._encoder_reader_store = LatestSampleStore()
                self._encoder_reader = ReaderTask(
                    self._read_encoder_line
                    if self._encoder_diagnostics is not None
                    or self.batch_encoder_reads
                    or self.capture_telemetry
                    else self.encoder_driver.read_parsed_line,
                    _EncoderFanout(self._encoder_stores),
                    self._encoder_reader_store,
                    self._stop_event,
                    name="csv-player-encoder-reader",
                )
                self.encoder_driver.stop_stream()
                self.encoder_driver.set_rate_hz(self.encoder_rate_hz)
                self.encoder_driver.start_all_stream()
                self._encoder_reader.start()

            self.wait_for_feedback(startup_timeout_s, require_encoder=self.encoder_port is not None)
        except Exception:
            self.close()
            raise

    def latest_feedback(self, max_age_s: float) -> HardwareFeedback | None:
        if self._servo_store is None:
            return None
        servo_record = self._servo_store.snapshot()
        if servo_record is None:
            return None
        sampled_at = time.monotonic()
        if sampled_at - servo_record.received_at > max_age_s:
            return None
        encoder_by_sensor: dict[int, Any] = {}
        encoder_received_at: dict[int, float] = {}
        for sensor_id, store in self._encoder_stores.items():
            record = store.snapshot()
            if record is not None:
                encoder_by_sensor[sensor_id] = record.value
                encoder_received_at[sensor_id] = record.received_at
        return HardwareFeedback(
            servo=servo_record.value,
            servo_received_at=servo_record.received_at,
            encoder_by_sensor=encoder_by_sensor,
            encoder_received_at=encoder_received_at,
            sampled_at=sampled_at,
        )

    def check_health(self) -> None:
        """Raise when either serial reader has failed."""

        stores = [self._servo_store, self._encoder_reader_store]
        for store in stores:
            if store is not None and store.error is not None:
                raise RuntimeError("Servo hardware telemetry reader failed") from store.error
        if self.dispatcher is not None:
            self.dispatcher.check_health()

    def encoder_diagnostics_snapshot(self) -> dict[str, Any] | None:
        if self._encoder_diagnostics is None:
            return None
        return self._encoder_diagnostics.snapshot()

    def drain_telemetry_samples(self) -> list[dict[str, Any]]:
        """Return captured raw telemetry records and clear the bounded buffer."""

        with self._telemetry_capture_lock:
            records = list(self._telemetry_samples)
            self._telemetry_samples.clear()
        return records

    def telemetry_capture_stats(self) -> dict[str, int]:
        with self._telemetry_capture_lock:
            return {
                "samples_seen": self._telemetry_samples_seen,
                "samples_dropped": self._telemetry_samples_dropped,
                "pending_samples": len(self._telemetry_samples),
                "sample_limit": self.TELEMETRY_CAPTURE_LIMIT,
            }

    def _record_telemetry_sample(self, record: dict[str, Any]) -> None:
        if not self.capture_telemetry:
            return
        with self._telemetry_capture_lock:
            if len(self._telemetry_samples) >= self.TELEMETRY_CAPTURE_LIMIT:
                self._telemetry_samples_dropped += 1
            self._telemetry_samples_seen += 1
            self._telemetry_samples.append(record)

    def _accept_servo_sample(self, parsed: Any) -> Optional[dict[str, Any]]:
        return _accept_servo(parsed)

    def _read_servo_line(self) -> Any:
        if self.servo_driver is None or self._servo_transport is None:
            raise RuntimeError("Servo telemetry transport is not connected")
        raw_line = self._servo_transport.read_line()
        parsed = self.servo_driver.parse_line(raw_line)
        if self.capture_telemetry:
            if isinstance(parsed, dict) and parsed.get("kind") == "telemetry":
                accepted = parsed
                self._record_telemetry_sample(
                    {
                        "signal": "servo_frame",
                        "received_monotonic_s": time.monotonic(),
                        "device_ms": accepted.get("device_ms"),
                        "valid": list(accepted.get("valid", ())),
                        "raw": copy.deepcopy(accepted),
                    }
                )
            else:
                self._record_telemetry_sample(
                    {
                        "signal": "servo_line",
                        "received_monotonic_s": time.monotonic(),
                        "valid": False,
                        "raw_line": str(raw_line).rstrip("\r\n"),
                        "parsed": copy.deepcopy(parsed),
                    }
                )
        return parsed

    def _read_available_encoder_lines(self) -> list[str]:
        """Read one blocking line, then drain complete buffered encoder lines.

        This CSV Player opt-in was added after DLC workloads exposed 150-200 ms
        reader stalls. Other applications keep the existing one-line transport
        behavior.
        """
        transport = self.encoder_driver.transport
        transport.require_open()
        if self._encoder_batch_buffer.find(b"\n") < 0:
            raw = transport.port.read_until(b"\n")
            if raw:
                self._encoder_batch_buffer.extend(raw)

        available = int(transport.port.in_waiting)
        if available > 0:
            self._encoder_batch_buffer.extend(transport.port.read(available))

        decoded: list[str] = []
        while len(decoded) < self.encoder_batch_max_lines:
            newline_index = self._encoder_batch_buffer.find(b"\n")
            if newline_index < 0:
                break
            raw_line = bytes(self._encoder_batch_buffer[: newline_index + 1])
            del self._encoder_batch_buffer[: newline_index + 1]
            try:
                decoded.append(raw_line.decode("utf-8", errors="replace"))
            except Exception:
                decoded.append(raw_line.decode("latin1", errors="replace"))
        return decoded

    def _read_encoder_line(self) -> Any:
        if self.encoder_driver is None:
            raise RuntimeError("Magnetic encoder driver is not connected")
        try:
            if self.batch_encoder_reads:
                if not self._encoder_pending_lines:
                    raw_lines = self._read_available_encoder_lines()
                    received_at = time.monotonic()
                    self._encoder_pending_lines.extend(
                        (raw_line, received_at) for raw_line in raw_lines
                    )
                if self._encoder_pending_lines:
                    raw_line, received_at = self._encoder_pending_lines.popleft()
                else:
                    raw_line = ""
                    received_at = time.monotonic()
            else:
                raw_line = self.encoder_driver.transport.read_line()
                received_at = time.monotonic()
        except Exception as exc:
            if self._encoder_diagnostics is not None:
                self._encoder_diagnostics.set_reader_errors(
                    servo_error=None,
                    encoder_error=exc,
                )
            raise
        try:
            parsed = self.encoder_driver.parse_line(raw_line)
        except Exception as exc:
            if self.capture_telemetry:
                self._record_telemetry_sample(
                    {
                        "signal": "encoder_line",
                        "received_monotonic_s": received_at,
                        "device_ms": None,
                        "sensor_id": None,
                        "parsed_kind": None,
                        "status": None,
                        "valid": False,
                        "raw_line": raw_line.rstrip("\r\n"),
                        "parse_error": f"{type(exc).__name__}: {exc}",
                        "raw": None,
                    }
                )
            if self._encoder_diagnostics is not None:
                self._encoder_diagnostics.record(
                    raw_line,
                    None,
                    received_at,
                    parse_error=f"{type(exc).__name__}: {exc}",
                )
                self._encoder_diagnostics.set_reader_errors(
                    servo_error=None,
                    encoder_error=exc,
                )
            raise
        if self.capture_telemetry:
            parsed_kind = (
                str(parsed[0])
                if isinstance(parsed, (tuple, list)) and parsed
                else None
            )
            sample = (
                parsed[1]
                if isinstance(parsed, (tuple, list)) and len(parsed) >= 2
                else None
            )
            status = (
                str(sample.get("status", "")).strip().upper()
                if isinstance(sample, dict)
                else ""
            )
            self._record_telemetry_sample(
                {
                    "signal": "encoder_line",
                    "received_monotonic_s": received_at,
                    "device_ms": sample.get("device_ms")
                    if isinstance(sample, dict)
                    else None,
                    "sensor_id": sample.get("sensor_id")
                    if isinstance(sample, dict)
                    else None,
                    "parsed_kind": parsed_kind,
                    "status": status or None,
                    "valid": parsed_kind in {"data", "sensor_data"}
                    and status in {"OK", "MULTITURN_UNCERTAIN"},
                    "raw_line": raw_line.rstrip("\r\n"),
                    "raw": copy.deepcopy(sample),
                }
            )
        if self._encoder_diagnostics is not None:
            self._encoder_diagnostics.record(raw_line, parsed, received_at)
        return parsed

    def submit_speed_to_servos(
        self,
        speed_by_servo_id: dict[int, int],
    ) -> CommandTicket:
        if self.read_only:
            raise RuntimeError("Servo commands are disabled in read-only mode")
        if self.dispatcher is None:
            raise RuntimeError("Hardware session is not connected")
        return self.dispatcher.submit_speed_to_servos(speed_by_servo_id)

    def wait_for_feedback(
        self,
        timeout_s: float,
        *,
        require_encoder: bool = False,
    ) -> HardwareFeedback:
        if self._servo_store is None:
            raise RuntimeError("Hardware session is not connected")
        self.check_health()
        self._servo_store.wait_for_sample(timeout_s)
        deadline = time.monotonic() + timeout_s
        if require_encoder:
            while time.monotonic() < deadline:
                self.check_health()
                if all(store.snapshot() is not None for store in self._encoder_stores.values()):
                    break
                time.sleep(0.005)
            else:
                raise TimeoutError("Not all AS5048A telemetry channels became available")
        feedback = self.latest_feedback(timeout_s)
        if feedback is None:
            raise TimeoutError("Servo telemetry is stale")
        return feedback

    def position_by_servo_id(self, feedback: HardwareFeedback) -> dict[int, float]:
        if self.system is None:
            raise RuntimeError("Hardware session is not connected")
        return {
            servo_id: float(feedback.servo["positions"][self.system.index_for_servo_id(servo_id)])
            for servo_id in self.servo_ids
        }

    def stop_servos(self) -> None:
        """Request ordinary zero speed without latching a new session out."""

        if self.dispatcher is not None:
            self.dispatcher.stop_servos()
        elif self.sender is not None:
            self.sender.stop_servos()

    def request_latched_stop(self) -> CommandTicket:
        """Atomically block future motion commands and request zero speed.

        The returned ticket reports only the PC-to-transport write attempt.  It
        must not be interpreted as physical motion or tendon-tension feedback.
        """

        if self.dispatcher is None:
            raise RuntimeError("Hardware session is not connected")
        ticket = self.dispatcher.request_latched_stop()
        self._last_stop_state = self.dispatcher.stop_state_snapshot()
        return ticket

    def stop_state_snapshot(self) -> dict[str, Any]:
        """Expose stop-request evidence without claiming physical confirmation."""

        if self.dispatcher is not None:
            self._last_stop_state = self.dispatcher.stop_state_snapshot()
        return dict(self._last_stop_state)

    def close(self) -> None:
        sender = self.sender
        dispatcher = self.dispatcher
        encoder = self.encoder_driver
        stop_event = self._stop_event
        if dispatcher is not None:
            if not self.read_only:
                try:
                    dispatcher.stop_servos()
                except Exception:
                    pass
            self._last_stop_state = dispatcher.stop_state_snapshot()
            dispatcher.close()
        elif sender is not None:
            if not self.read_only:
                try:
                    sender.stop_servos()
                except Exception:
                    pass
        if encoder is not None:
            try:
                encoder.stop_stream()
            except Exception:
                pass
        if stop_event is not None:
            stop_event.set()
        if self._servo_reader is not None:
            self._servo_reader.join(1.0)
        if self._encoder_reader is not None:
            self._encoder_reader.join(1.0)
        if self._encoder_diagnostics is not None:
            self._encoder_diagnostics.set_reader_errors(
                servo_error=None
                if self._servo_store is None
                else self._servo_store.error,
                encoder_error=None
                if self._encoder_reader_store is None
                else self._encoder_reader_store.error,
            )
        for transport in (self._encoder_transport, self._servo_transport):
            if transport is not None:
                try:
                    transport.close()
                except Exception:
                    pass
        self.system = None
        self.servo_driver = None
        self.encoder_driver = None
        self.sender = None
        self.dispatcher = None
        self._servo_transport = None
        self._encoder_transport = None
        self._servo_store = None
        self._encoder_reader_store = None
        self._encoder_stores = {}
        self._encoder_batch_buffer.clear()
        self._encoder_pending_lines.clear()
        self._stop_event = None
        self._servo_reader = None
        self._encoder_reader = None
