from __future__ import annotations

from dataclasses import dataclass
from threading import RLock
from typing import List, Optional


@dataclass
class TelemetrySnapshot:
    timestamp_ms: int
    positions: List[int]
    loads: List[int]
    speeds: List[int]


class AppState:
    def __init__(self, num_motors: int) -> None:
        self.num_motors = num_motors
        self._lock = RLock()
        self.connected = False
        self.last_error: Optional[str] = None
        self.timestamp_ms = 0
        self.current_positions = [0] * num_motors
        self.current_loads = [0] * num_motors
        self.current_speeds = [0] * num_motors

    def set_connected(self, connected: bool) -> None:
        with self._lock:
            self.connected = connected

    def set_error(self, message: Optional[str]) -> None:
        with self._lock:
            self.last_error = message

    def update_telemetry(
        self,
        timestamp_ms: int,
        positions: List[int],
        loads: List[int],
        speeds: Optional[List[int]],
    ) -> None:
        with self._lock:
            self.timestamp_ms = timestamp_ms
            self.current_positions = list(positions)
            self.current_loads = list(loads)
            self.current_speeds = list(speeds) if speeds is not None else [0] * self.num_motors

    def snapshot(self) -> TelemetrySnapshot:
        with self._lock:
            return TelemetrySnapshot(
                timestamp_ms=self.timestamp_ms,
                positions=list(self.current_positions),
                loads=list(self.current_loads),
                speeds=list(self.current_speeds),
            )
