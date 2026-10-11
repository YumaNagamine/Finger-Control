"""Thread-safe helpers shared by the CSV Player GUI and its tests."""

from __future__ import annotations

import threading
from typing import Any


class LatestEventMailbox:
    """Keep only the newest high-rate GUI event of each live-data kind."""

    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._events: dict[str, dict[str, Any]] = {}

    def publish(self, event: dict[str, Any]) -> None:
        with self._lock:
            self._events[str(event["kind"])] = event

    def drain(self) -> list[dict[str, Any]]:
        with self._lock:
            events = list(self._events.values())
            self._events.clear()
        return events

    def clear(self) -> None:
        with self._lock:
            self._events.clear()
