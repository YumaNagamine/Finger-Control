from __future__ import annotations

from dataclasses import dataclass
from typing import List, Optional

import serial


NUM_SERVOS = 6


@dataclass
class TelemetryFrame:
    timestamp_ms: int
    positions: List[int]
    loads: List[int]
    speeds: Optional[List[int]]
    raw: str


class ServoAPI:
    """
    Thin serial-command client for st_control3_2_experimental_V3 firmware.
    """

    def __init__(
        self,
        port: str = "COM7",
        baud_rate: int = 921600,
        timeout: float = 1.0,
        auto_open: bool = True,
    ) -> None:
        self.port = port
        self.baud_rate = baud_rate
        self.timeout = timeout
        self._serial: Optional[serial.Serial] = None
        if auto_open:
            self.open()

    def open(self) -> None:
        if self._serial and self._serial.is_open:
            return
        self._serial = serial.Serial(self.port, self.baud_rate, timeout=self.timeout)

    def close(self) -> None:
        if self._serial and self._serial.is_open:
            self._serial.close()

    def __enter__(self) -> "ServoAPI":
        self.open()
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()

    def _require_open(self) -> serial.Serial:
        if not self._serial or not self._serial.is_open:
            raise RuntimeError("Serial port is not open. Call open() first.")
        return self._serial

    def _validate_servo_id(self, servo_id: int) -> None:
        if not (0 <= servo_id < NUM_SERVOS):
            raise ValueError(f"servo_id must be 0-{NUM_SERVOS - 1}, got {servo_id}")

    def send_raw(self, command: str) -> None:
        port = self._require_open()
        text = command if command.endswith("\n") else f"{command}\n"
        port.write(text.encode("utf-8"))

    def read_line(self) -> str:
        port = self._require_open()
        return port.readline().decode("utf-8", errors="replace").strip()

    def try_read_telemetry(self) -> Optional[TelemetryFrame]:
        line = self.read_line()
        if not line:
            return None

        parts = line.split(",")
        if not parts or not parts[0].isdigit():
            return None

        if len(parts) == 13:
            timestamp = int(parts[0])
            positions: List[int] = []
            loads: List[int] = []
            for i in range(NUM_SERVOS):
                base = 1 + (i * 2)
                positions.append(int(parts[base]))
                loads.append(int(parts[base + 1]))
            return TelemetryFrame(
                timestamp_ms=timestamp,
                positions=positions,
                loads=loads,
                speeds=None,
                raw=line,
            )

        if len(parts) == 19:
            timestamp = int(parts[0])
            positions = []
            loads = []
            speeds: List[int] = []
            for i in range(NUM_SERVOS):
                base = 1 + (i * 3)
                positions.append(int(parts[base]))
                loads.append(int(parts[base + 1]))
                speeds.append(int(parts[base + 2]))
            return TelemetryFrame(
                timestamp_ms=timestamp,
                positions=positions,
                loads=loads,
                speeds=speeds,
                raw=line,
            )

        return None

    def set_speed(self, servo_id: int, speed: int, force_init: bool = False) -> None:
        self._validate_servo_id(servo_id)
        force = 1 if force_init else 0
        self.send_raw(f"{servo_id},{speed},{force}")

    def timed_run(self, servo_id: int, speed: int, time_ms: int) -> None:
        self._validate_servo_id(servo_id)
        self.send_raw(f"d,{servo_id},{speed},{time_ms}")

    def go_to_zero(self, servo_id: int) -> None:
        self._validate_servo_id(servo_id)
        self.send_raw(f"g,{servo_id}")

    def set_zero(self, servo_id: int) -> None:
        self._validate_servo_id(servo_id)
        self.send_raw(f"p,{servo_id}")

    def enter_zeroing_mode(self, servo_id: int) -> None:
        self._validate_servo_id(servo_id)
        self.send_raw(f"z,{servo_id}")

    def zeroing_jog(self, command: str) -> None:
        allowed = {"+", "++", "-", "--"}
        if command not in allowed:
            raise ValueError(f"zeroing jog must be one of {sorted(allowed)}")
        self.send_raw(command)

    def zeroing_confirm(self) -> None:
        self.send_raw("c")

    def zeroing_cancel(self) -> None:
        self.send_raw("q")

    def set_position(self, servo_id: int, position: int, time_ms: int = 0) -> None:
        self._validate_servo_id(servo_id)
        self.send_raw(f"x,{servo_id},{position},{time_ms}")
    def set_multiturn_position(
        self,
        servo_id: int,
        position: int,
        time_ms: int = 0,
    ) -> None:
        self._validate_servo_id(servo_id)
        if not -(2**31) <= position <= (2**31 - 1):
            raise ValueError("position must fit in a signed 32-bit integer")
        if not 0 <= time_ms <= 65535:
            raise ValueError("time_ms must be in the range 0-65535")
        self.send_raw(f"ma,{servo_id},{position},{time_ms}")

    def move_relative(
        self,
        servo_id: int,
        delta: int,
        time_ms: int = 0,
    ) -> None:
        self._validate_servo_id(servo_id)
        if not -32767 <= delta <= 32767:
            raise ValueError("delta must be in the range -32767-32767")
        if not 0 <= time_ms <= 65535:
            raise ValueError("time_ms must be in the range 0-65535")
        self.send_raw(f"mr,{servo_id},{delta},{time_ms}")


    def stop_all(self) -> None:
        self.send_raw("s")

    def reset_system(self) -> None:
        self.send_raw("r")

    def scan_ids(self, start_id: int, end_id: int) -> None:
        if start_id < 0 or end_id < 0 or start_id > end_id:
            raise ValueError("Invalid scan range.")
        self.send_raw(f"S,{start_id},{end_id}")

    def change_id(self, old_id: int, new_id: int) -> None:
        self.send_raw(f"W,{old_id},{new_id}")

    def reset_ids(self) -> None:
        self.send_raw("RESET_IDS")

    def list_ids(self) -> None:
        self.send_raw("LIST_IDS")
