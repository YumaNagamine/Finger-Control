from __future__ import annotations

import threading
import time
from typing import Optional

from servo.servoGUI_api.app_state import AppState
from servo.servoGUI_api.event_bus import EventBus
from servo.servoGUI_api.infra.line_parser import parse_line
from servo.servoGUI_api.infra.mock_backend import MockServoAPI


class ServoSession:
    def __init__(
        self,
        app_state: AppState,
        event_bus: EventBus,
        port: str,
        baud_rate: int,
        timeout: float,
        use_mock: bool,
    ) -> None:
        self._state = app_state
        self._bus = event_bus
        self._port = port
        self._baud_rate = baud_rate
        self._timeout = timeout
        self._use_mock = use_mock

        self._backend = None
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._call_lock = threading.RLock()

    @property
    def use_mock(self) -> bool:
        return self._use_mock

    def start(self) -> None:
        self._open_backend()
        self._running = True
        self._thread = threading.Thread(target=self._read_loop, daemon=True)
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread and self._thread.is_alive():
            self._thread.join(timeout=1.0)
        self._thread = None

        with self._call_lock:
            if self._backend is not None:
                self._backend.close()
                self._backend = None

        self._state.set_connected(False)
        self._bus.publish("connection", {"connected": False, "mock": self._use_mock})

    def reconnect(self, use_mock: Optional[bool] = None) -> None:
        if use_mock is not None:
            self._use_mock = use_mock
        self.stop()
        self.start()

    def _open_backend(self) -> None:
        with self._call_lock:
            if self._use_mock:
                backend = MockServoAPI(num_motors=self._state.num_motors, timeout=min(self._timeout, 0.05))
                backend.open()
                self._backend = backend
            else:
                try:
                    from servo.servo_APIs import ServoAPI
                except Exception as exc:
                    raise RuntimeError(
                        "ServoAPI backend requires pyserial. "
                        "Install pyserial or set SERVOGUI_MOCK=1."
                    ) from exc
                api = ServoAPI(
                    port=self._port,
                    baud_rate=self._baud_rate,
                    timeout=self._timeout,
                    auto_open=True,
                )
                self._backend = api

        self._state.set_connected(True)
        self._state.set_error(None)
        self._bus.publish("connection", {"connected": True, "mock": self._use_mock})

    def _read_loop(self) -> None:
        while self._running:
            try:
                with self._call_lock:
                    if self._backend is None:
                        time.sleep(0.05)
                        continue
                    line = self._backend.read_line()
            except Exception as exc:
                msg = f"Read loop error: {exc}"
                self._state.set_error(msg)
                self._bus.publish("error", msg)
                time.sleep(0.1)
                continue

            if not line:
                continue

            parsed = parse_line(line, self._state.num_motors)
            if parsed.kind == "telemetry" and parsed.telemetry is not None:
                frame = parsed.telemetry
                self._state.update_telemetry(frame.timestamp_ms, frame.positions, frame.loads, frame.speeds)
                self._bus.publish("telemetry", frame)
            elif parsed.kind == "id_event":
                self._bus.publish("id_event", parsed.line)
            else:
                self._bus.publish("raw_line", parsed.line)

    def _call(self, method_name: str, *args, **kwargs) -> None:
        with self._call_lock:
            if self._backend is None:
                self._bus.publish("error", "Backend is not initialized")
                self._bus.publish(
                    "command",
                    {
                        "method": method_name,
                        "args": args,
                        "kwargs": kwargs,
                        "ok": False,
                        "error": "Backend is not initialized",
                    },
                )
                return
            method = getattr(self._backend, method_name)
            try:
                method(*args, **kwargs)
            except Exception as exc:
                self._bus.publish("error", f"Command failed ({method_name}): {exc}")
                self._bus.publish(
                    "command",
                    {
                        "method": method_name,
                        "args": args,
                        "kwargs": kwargs,
                        "ok": False,
                        "error": str(exc),
                    },
                )
                return

            self._bus.publish(
                "command",
                {
                    "method": method_name,
                    "args": args,
                    "kwargs": kwargs,
                    "ok": True,
                },
            )

    def set_speed(self, servo_id: int, speed: int, force_init: bool = False) -> None:
        self._call("set_speed", servo_id, speed, force_init)

    def timed_run(self, servo_id: int, speed: int, time_ms: int) -> None:
        self._call("timed_run", servo_id, speed, time_ms)

    def go_to_zero(self, servo_id: int) -> None:
        self._call("go_to_zero", servo_id)

    def set_zero(self, servo_id: int) -> None:
        self._call("set_zero", servo_id)

    def set_position(self, servo_id: int, position: int, time_ms: int = 0) -> None:
        self._call("set_position", servo_id, position, time_ms)

    def stop_all(self) -> None:
        self._call("stop_all")

    def reset_system(self) -> None:
        self._call("reset_system")

    def scan_ids(self, start_id: int, end_id: int) -> None:
        self._call("scan_ids", start_id, end_id)

    def change_id(self, old_id: int, new_id: int) -> None:
        self._call("change_id", old_id, new_id)

    def reset_ids(self) -> None:
        self._call("reset_ids")

    def list_ids(self) -> None:
        self._call("list_ids")
