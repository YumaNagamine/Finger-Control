"""Project-owned ServoHardwareAPI session and latest telemetry access."""

from __future__ import annotations

import sys
import threading
import time
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
        self._stop_event: threading.Event | None = None
        self._servo_reader: ReaderTask | None = None
        self._encoder_reader: ReaderTask | None = None

    @property
    def connected(self) -> bool:
        return self.sender is not None

    def connect(self, *, startup_timeout_s: float = 3.0) -> None:
        self.close()
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
            self.sender = ServoCommandSender(self.servo_driver)
            self.dispatcher = ServoCommandDispatcher(self.sender)
            self.dispatcher.start()
            self._stop_event = threading.Event()
            self._servo_store = LatestSampleStore()
            self._servo_reader = ReaderTask(
                self.servo_driver.read_parsed_line,
                _accept_servo,
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
                    self.encoder_driver.read_parsed_line,
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

    def submit_speed_to_servos(
        self,
        speed_by_servo_id: dict[int, int],
    ) -> CommandTicket:
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
        if self.dispatcher is not None:
            self.dispatcher.stop_servos()
        elif self.sender is not None:
            self.sender.stop_servos()

    def close(self) -> None:
        sender = self.sender
        dispatcher = self.dispatcher
        encoder = self.encoder_driver
        stop_event = self._stop_event
        if dispatcher is not None:
            try:
                dispatcher.stop_servos()
            except Exception:
                pass
            dispatcher.close()
        elif sender is not None:
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
        self._stop_event = None
        self._servo_reader = None
        self._encoder_reader = None
