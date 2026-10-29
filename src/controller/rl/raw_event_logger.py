"""Non-blocking raw event logging for control and vision timelines."""

from __future__ import annotations

import json
import math
import numbers
import queue
import threading
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping


RAW_EVENT_SCHEMA_VERSION = 1


def _json_safe(value: Any) -> Any:
    if value is None or isinstance(value, (str, bool)):
        return value
    if isinstance(value, numbers.Integral):
        return int(value)
    if isinstance(value, numbers.Real):
        numeric = float(value)
        return numeric if math.isfinite(numeric) else None
    if isinstance(value, Mapping):
        return {str(key): _json_safe(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [_json_safe(item) for item in value]
    raise TypeError(f"Raw event value is not JSON-compatible: {type(value).__name__}")


@dataclass(frozen=True)
class RawLoggerStats:
    enqueued_events: int
    written_events: int
    dropped_events: int
    max_queue_depth: int
    max_write_lag_s: float
    writer_error: str | None
    closed: bool

    @property
    def complete(self) -> bool:
        return (
            self.closed
            and self.dropped_events == 0
            and self.writer_error is None
            and self.enqueued_events == self.written_events
        )

    def as_dict(self) -> dict[str, Any]:
        return {
            "enqueued_events": self.enqueued_events,
            "written_events": self.written_events,
            "dropped_events": self.dropped_events,
            "max_queue_depth": self.max_queue_depth,
            "max_write_lag_s": self.max_write_lag_s,
            "writer_error": self.writer_error,
            "closed": self.closed,
            "complete": self.complete,
        }


class AsyncRawEventLogger:
    """Append JSONL events from a background writer thread.

    Producers perform only a bounded, non-blocking queue insertion.  JSON
    encoding, file I/O, and flushing happen on the writer thread so logging
    does not extend the lower-level PID critical path.  Queue overflow is not
    hidden: dropped events and the first overflow time are included in stats.
    """

    _STOP = object()

    def __init__(
        self,
        path: str | Path,
        *,
        run_id: str,
        queue_capacity: int = 8192,
        flush_interval_s: float = 0.1,
    ) -> None:
        if queue_capacity <= 0:
            raise ValueError("queue_capacity must be positive")
        if flush_interval_s <= 0.0:
            raise ValueError("flush_interval_s must be positive")
        self.path = Path(path)
        self.path.parent.mkdir(parents=True, exist_ok=True)
        self.run_id = str(run_id)
        self.queue_capacity = int(queue_capacity)
        self.flush_interval_s = float(flush_interval_s)
        self._queue: queue.Queue[object] = queue.Queue(maxsize=self.queue_capacity)
        self._lock = threading.Lock()
        self._next_sequence = 0
        self._enqueued_events = 0
        self._written_events = 0
        self._dropped_events = 0
        self._max_queue_depth = 0
        self._max_write_lag_s = 0.0
        self._first_overflow_monotonic_s: float | None = None
        self._writer_error: str | None = None
        self._closed = False
        self._thread = threading.Thread(
            target=self._run,
            name="raw-event-writer",
            daemon=True,
        )
        self._thread.start()

    def log(
        self,
        event_type: str,
        *,
        event_monotonic_s: float | None = None,
        payload: Mapping[str, Any] | None = None,
    ) -> bool:
        """Queue one event without waiting for file I/O.

        ``event_monotonic_s`` is the time of the represented event, while
        ``enqueued_monotonic_s`` records when the producer handed it to the
        logger.  They are intentionally kept distinct.
        """

        event_name = str(event_type).strip()
        if not event_name:
            raise ValueError("event_type must not be empty")
        now = time.monotonic()
        event = {
            "schema_version": RAW_EVENT_SCHEMA_VERSION,
            "run_id": self.run_id,
            "event_sequence": None,
            "event_type": event_name,
            "event_monotonic_s": now
            if event_monotonic_s is None
            else float(event_monotonic_s),
            "enqueued_monotonic_s": now,
            "payload": dict(payload or {}),
        }
        with self._lock:
            if self._closed or self._writer_error is not None:
                self._dropped_events += 1
                return False
            sequence = self._next_sequence
            self._next_sequence += 1
            event["event_sequence"] = sequence
            try:
                self._queue.put_nowait(event)
            except queue.Full:
                self._dropped_events += 1
                if self._first_overflow_monotonic_s is None:
                    self._first_overflow_monotonic_s = now
                return False
            self._enqueued_events += 1
            self._max_queue_depth = max(
                self._max_queue_depth,
                self._queue.qsize(),
            )
        return True

    def stats(self) -> RawLoggerStats:
        with self._lock:
            return RawLoggerStats(
                enqueued_events=self._enqueued_events,
                written_events=self._written_events,
                dropped_events=self._dropped_events,
                max_queue_depth=self._max_queue_depth,
                max_write_lag_s=self._max_write_lag_s,
                writer_error=self._writer_error,
                closed=self._closed,
            )

    @property
    def first_overflow_monotonic_s(self) -> float | None:
        with self._lock:
            return self._first_overflow_monotonic_s

    def close(self, *, timeout_s: float = 5.0) -> RawLoggerStats:
        with self._lock:
            if self._closed:
                already_closed = True
            else:
                already_closed = False
                self._closed = True
        if already_closed:
            return self.stats()
        try:
            self._queue.put(self._STOP, timeout=max(0.0, float(timeout_s)))
        except queue.Full:
            with self._lock:
                self._writer_error = self._writer_error or (
                    "Raw logger queue did not drain during close"
                )
            return self.stats()
        self._thread.join(max(0.0, float(timeout_s)))
        if self._thread.is_alive():
            with self._lock:
                self._writer_error = self._writer_error or (
                    "Raw logger writer did not stop before timeout"
                )
        return self.stats()

    def _run(self) -> None:
        last_flush = time.monotonic()
        try:
            with self.path.open("w", encoding="utf-8", newline="\n") as handle:
                while True:
                    item = self._queue.get()
                    if item is self._STOP:
                        break
                    assert isinstance(item, dict)
                    written_at = time.monotonic()
                    item["written_monotonic_s"] = written_at
                    lag_s = max(
                        0.0,
                        written_at - float(item["enqueued_monotonic_s"]),
                    )
                    handle.write(
                        json.dumps(
                            _json_safe(item),
                            ensure_ascii=False,
                            separators=(",", ":"),
                            allow_nan=False,
                        )
                        + "\n"
                    )
                    with self._lock:
                        self._written_events += 1
                        self._max_write_lag_s = max(
                            self._max_write_lag_s,
                            lag_s,
                        )
                    if written_at - last_flush >= self.flush_interval_s:
                        handle.flush()
                        last_flush = written_at
                handle.flush()
        except Exception as exc:  # pragma: no cover - exercised through stats
            with self._lock:
                self._writer_error = f"{type(exc).__name__}: {exc}"

