from __future__ import annotations

import time
from collections import deque
from threading import RLock
from typing import Deque, List


class MockServoAPI:
    """Mock backend with the same high-level method surface as ServoAPI."""

    def __init__(self, num_motors: int = 6, timeout: float = 0.03) -> None:
        self.num_motors = num_motors
        self.timeout = timeout
        self._lock = RLock()
        self._is_open = False

        self.positions = [0] * num_motors
        self.loads = [0] * num_motors
        self.speeds = [0] * num_motors
        self._stop_deadlines = [0.0] * num_motors
        self._servo_ids = list(range(num_motors))

        self._boot_t = time.time()
        self._last_update_t = self._boot_t
        self._line_queue: Deque[str] = deque()

    @property
    def is_open(self) -> bool:
        return self._is_open

    def open(self) -> None:
        with self._lock:
            self._is_open = True
            self._boot_t = time.time()
            self._last_update_t = self._boot_t

    def close(self) -> None:
        with self._lock:
            self._is_open = False

    def _validate_servo_id(self, servo_id: int) -> None:
        if not (0 <= servo_id < self.num_motors):
            raise ValueError(f"servo_id must be 0-{self.num_motors - 1}, got {servo_id}")

    def _update_state(self) -> None:
        now = time.time()
        dt = max(0.0, now - self._last_update_t)
        self._last_update_t = now

        for idx in range(self.num_motors):
            deadline = self._stop_deadlines[idx]
            if deadline > 0 and now >= deadline:
                self.speeds[idx] = 0
                self._stop_deadlines[idx] = 0.0

            self.positions[idx] += int(self.speeds[idx] * dt * 0.25)
            self.positions[idx] = max(0, min(4095, self.positions[idx]))

            base = abs(self.speeds[idx]) // 10
            ripple = ((int((now - self._boot_t) * 20) + (idx * 7)) % 18) - 9
            self.loads[idx] = max(0, min(1023, base + ripple))

    def _enqueue(self, line: str) -> None:
        self._line_queue.append(line)

    def read_line(self) -> str:
        with self._lock:
            if not self._is_open:
                return ""
            self._update_state()
            if self._line_queue:
                return self._line_queue.popleft()

            if self.timeout > 0:
                time.sleep(min(self.timeout, 0.03))

            timestamp_ms = int((time.time() - self._boot_t) * 1000)
            parts: List[str] = [str(timestamp_ms)]
            for idx in range(self.num_motors):
                parts.append(str(int(self.positions[idx])))
                parts.append(str(int(self.loads[idx])))
                parts.append(str(int(self.speeds[idx])))
            return ",".join(parts)

    def set_speed(self, servo_id: int, speed: int, force_init: bool = False) -> None:
        del force_init
        with self._lock:
            self._validate_servo_id(servo_id)
            self.speeds[servo_id] = int(speed)
            self._stop_deadlines[servo_id] = 0.0

    def timed_run(self, servo_id: int, speed: int, time_ms: int) -> None:
        with self._lock:
            self._validate_servo_id(servo_id)
            self.speeds[servo_id] = int(speed)
            self._stop_deadlines[servo_id] = time.time() + (int(time_ms) / 1000.0)

    def go_to_zero(self, servo_id: int) -> None:
        with self._lock:
            self._validate_servo_id(servo_id)
            self.positions[servo_id] = 0
            self.speeds[servo_id] = 0
            self._stop_deadlines[servo_id] = 0.0

    def set_zero(self, servo_id: int) -> None:
        self.go_to_zero(servo_id)

    def enter_zeroing_mode(self, servo_id: int) -> None:
        self._validate_servo_id(servo_id)

    def zeroing_jog(self, command: str) -> None:
        del command

    def zeroing_confirm(self) -> None:
        pass

    def zeroing_cancel(self) -> None:
        pass

    def set_position(self, servo_id: int, position: int, time_ms: int = 0) -> None:
        del time_ms
        with self._lock:
            self._validate_servo_id(servo_id)
            self.positions[servo_id] = max(0, min(4095, int(position)))
            self.speeds[servo_id] = 0
            self._stop_deadlines[servo_id] = 0.0

    def stop_all(self) -> None:
        with self._lock:
            self.speeds = [0] * self.num_motors
            self._stop_deadlines = [0.0] * self.num_motors

    def reset_system(self) -> None:
        with self._lock:
            self.positions = [0] * self.num_motors
            self.loads = [0] * self.num_motors
            self.speeds = [0] * self.num_motors
            self._stop_deadlines = [0.0] * self.num_motors

    def scan_ids(self, start_id: int, end_id: int) -> None:
        with self._lock:
            self._enqueue("SCAN_START")
            for servo_id in self._servo_ids:
                if start_id <= servo_id <= end_id:
                    self._enqueue(f"FOUND_ID:{servo_id}")
            self._enqueue("SCAN_END")

    def change_id(self, old_id: int, new_id: int) -> None:
        with self._lock:
            self._enqueue("EXECUTING: CHANGE_ID")
            if old_id in self._servo_ids and 0 <= new_id <= 253 and new_id not in self._servo_ids:
                idx = self._servo_ids.index(old_id)
                self._servo_ids[idx] = new_id
                self._enqueue("VERIFY_SUCCESS")
                self._enqueue("NVS_UPDATED")
            else:
                self._enqueue("VERIFY_FAIL")

    def reset_ids(self) -> None:
        with self._lock:
            self._servo_ids = list(range(self.num_motors))
            self._enqueue("NVS_UPDATED")

    def list_ids(self) -> None:
        with self._lock:
            for servo_id in self._servo_ids:
                self._enqueue(f"FOUND_ID:{servo_id}")
