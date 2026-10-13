"""Thread-safe latest-sample storage for serial reader threads."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Any, Callable, Optional


@dataclass(frozen=True)
class LatestSample:
    value: Any
    received_at: float


class LatestSampleStore:
    """Publish the newest accepted sample while retaining reader errors."""

    def __init__(self):
        self._condition = threading.Condition()
        self._sample: Optional[LatestSample] = None
        self._error: Optional[Exception] = None

    def publish(self, value: Any, *, received_at: Optional[float] = None) -> None:
        timestamp = time.monotonic() if received_at is None else float(received_at)
        with self._condition:
            self._sample = LatestSample(value=value, received_at=timestamp)
            self._condition.notify_all()

    def snapshot(self) -> Optional[LatestSample]:
        with self._condition:
            return self._sample

    def wait_for_sample(
        self,
        timeout_s: float,
        *,
        stop_event: Optional[threading.Event] = None,
    ) -> LatestSample:
        deadline = time.monotonic() + timeout_s
        with self._condition:
            while self._sample is None:
                if stop_event is not None and stop_event.is_set():
                    raise RuntimeError("Waiting for feedback was cancelled")
                if self._error is not None:
                    raise RuntimeError("Feedback reader failed") from self._error
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise TimeoutError("No feedback sample was received")
                wait_timeout = remaining
                if stop_event is not None:
                    wait_timeout = min(wait_timeout, 0.05)
                self._condition.wait(timeout=wait_timeout)
            return self._sample

    def set_error(self, error: Exception) -> None:
        with self._condition:
            self._error = error
            self._condition.notify_all()

    @property
    def error(self) -> Optional[Exception]:
        with self._condition:
            return self._error


class ReaderTask:
    """Own one serial reader and publish only accepted parsed records."""

    def __init__(
        self,
        read_fn: Callable[[], Any],
        accept_fn: Callable[[Any], Optional[Any]],
        store: LatestSampleStore,
        stop_event: threading.Event,
        *,
        name: str,
    ):
        self._read_fn = read_fn
        self._accept_fn = accept_fn
        self._store = store
        self._stop_event = stop_event
        self._thread = threading.Thread(target=self._run, name=name, daemon=True)

    def start(self) -> None:
        self._thread.start()

    def join(self, timeout_s: float) -> None:
        self._thread.join(timeout=timeout_s)

    def _run(self) -> None:
        try:
            while not self._stop_event.is_set():
                parsed = self._read_fn()
                accepted = self._accept_fn(parsed)
                if accepted is not None:
                    self._store.publish(accepted)
        except Exception as exc:
            self._store.set_error(exc)
