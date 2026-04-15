from __future__ import annotations

from collections import defaultdict
from threading import RLock
from typing import Any, Callable, DefaultDict, List


Callback = Callable[[Any], None]


class EventBus:
    def __init__(self) -> None:
        self._lock = RLock()
        self._subscribers: DefaultDict[str, List[Callback]] = defaultdict(list)

    def subscribe(self, event_name: str, callback: Callback) -> None:
        with self._lock:
            self._subscribers[event_name].append(callback)

    def publish(self, event_name: str, payload: Any) -> None:
        with self._lock:
            callbacks = list(self._subscribers.get(event_name, []))

        for callback in callbacks:
            try:
                callback(payload)
            except Exception as exc:  # pragma: no cover
                print(f"EventBus callback error ({event_name}): {exc}")
