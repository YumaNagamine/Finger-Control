"""Single-reader telemetry monitor shared by higher-level controllers."""

from __future__ import annotations

import threading
import time
from queue import Full, Queue
from dataclasses import dataclass
from typing import TypeAlias


@dataclass(frozen=True)
class TelemetrySnapshot:
    received_at: float
    timestamp_ms: int
    positions: tuple[int, ...]
    loads: tuple[int, ...]
    speeds: tuple[int, ...] | None
    sequence: int


TelemetrySubscription: TypeAlias = Queue[TelemetrySnapshot]


class TelemetryMonitor:
    """Continuously reads telemetry and exposes the latest complete snapshot.

    This class must be the only consumer of ``api.try_read_telemetry()`` while
    it is running.  Command-side code reads snapshots from this monitor instead
    of reading the serial stream directly.
    """

    def __init__(self, api, *, num_servos: int, read_timeout_s: float) -> None:
        if num_servos <= 0:
            raise ValueError("num_servos must be greater than zero")
        if read_timeout_s <= 0.0:
            raise ValueError("read_timeout_s must be greater than zero")

        self._api = api
        self._num_servos = num_servos
        self._read_timeout_s = read_timeout_s
        self._stop_event = threading.Event()
        self._condition = threading.Condition()
        self._thread: threading.Thread | None = None
        self._latest: TelemetrySnapshot | None = None
        self._error: Exception | None = None
        self._sequence = 0
        self._subscriptions: set[TelemetrySubscription] = set()

    def subscribe(self, *, maxsize: int = 0) -> TelemetrySubscription:
        """Return a queue receiving every complete snapshot from this monitor.

        The monitor remains the only serial reader.  Subscribers receive the
        immutable snapshots after the monitor has accepted and published them.
        An unbounded queue (the default) is appropriate for short recording
        trials and prevents a slow DLC thread from dropping telemetry samples.
        """
        if maxsize < 0:
            raise ValueError("maxsize must be non-negative")
        subscription: TelemetrySubscription = Queue(maxsize=maxsize)
        with self._condition:
            if self._stop_event.is_set():
                raise RuntimeError("Telemetry monitor is stopped")
            self._subscriptions.add(subscription)
        return subscription

    def unsubscribe(self, subscription: TelemetrySubscription) -> None:
        with self._condition:
            self._subscriptions.discard(subscription)

    def start(self) -> None:
        if self._thread is not None:
            raise RuntimeError("Telemetry monitor has already been started")
        self._thread = threading.Thread(
            target=self._read_loop,
            name="servo-telemetry",
            daemon=True,
        )
        self._thread.start()

    def stop(self) -> None:
        self._stop_event.set()
        with self._condition:
            self._condition.notify_all()
        if self._thread is not None:
            self._thread.join(timeout=max(1.0, self._read_timeout_s + 1.0))

    def raise_if_failed(self) -> None:
        with self._condition:
            error = self._error
        if error is not None:
            raise RuntimeError("Telemetry reader failed") from error

    def latest(self, max_age_s: float) -> TelemetrySnapshot | None:
        if max_age_s <= 0.0:
            raise ValueError("max_age_s must be greater than zero")
        self.raise_if_failed()
        with self._condition:
            snapshot = self._latest
        if snapshot is None:
            return None
        if time.monotonic() - snapshot.received_at > max_age_s:
            return None
        return snapshot

    def wait_for_newer(
        self,
        sequence: int,
        timeout_s: float,
    ) -> TelemetrySnapshot:
        if sequence < 0:
            raise ValueError("sequence must be non-negative")
        if timeout_s <= 0.0:
            raise ValueError("timeout_s must be greater than zero")

        deadline = time.monotonic() + timeout_s
        with self._condition:
            while True:
                if self._error is not None:
                    raise RuntimeError("Telemetry reader failed") from self._error
                if self._latest is not None and self._latest.sequence > sequence:
                    return self._latest
                if self._stop_event.is_set():
                    raise RuntimeError("Telemetry monitor stopped")

                remaining_s = deadline - time.monotonic()
                if remaining_s <= 0.0:
                    raise TimeoutError(
                        f"No telemetry newer than sequence {sequence} received "
                        f"within {timeout_s:.3f} s"
                    )
                self._condition.wait(timeout=remaining_s)

    def _read_loop(self) -> None:
        try:
            while not self._stop_event.is_set():
                frame = self._api.try_read_telemetry()
                if frame is None:
                    continue
                if len(frame.positions) != self._num_servos:
                    continue
                if len(frame.loads) != self._num_servos:
                    continue
                if frame.speeds is not None and len(frame.speeds) != self._num_servos:
                    continue

                with self._condition:
                    self._sequence += 1
                    snapshot = TelemetrySnapshot(
                        received_at=time.monotonic(),
                        timestamp_ms=int(frame.timestamp_ms),
                        positions=tuple(int(value) for value in frame.positions),
                        loads=tuple(int(value) for value in frame.loads),
                        speeds=(
                            None
                            if frame.speeds is None
                            else tuple(int(value) for value in frame.speeds)
                        ),
                        sequence=self._sequence,
                    )
                    self._latest = snapshot
                    subscriptions = tuple(self._subscriptions)
                    self._condition.notify_all()
                for subscription in subscriptions:
                    try:
                        subscription.put_nowait(snapshot)
                    except Full as exc:
                        with self._condition:
                            self._error = RuntimeError(
                                "Telemetry subscription queue is full"
                            )
                            self._condition.notify_all()
                        raise exc
        except Exception as exc:
            if not self._stop_event.is_set():
                with self._condition:
                    self._error = exc
                    self._condition.notify_all()
