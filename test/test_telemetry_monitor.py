from __future__ import annotations

import sys
from pathlib import Path
from queue import Empty
from types import SimpleNamespace
from unittest import TestCase

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from servo.control.telemetry_monitor import TelemetryMonitor


class FakeTelemetryAPI:
    def __init__(self, frames):
        self.frames = list(frames)

    def try_read_telemetry(self):
        if self.frames:
            return self.frames.pop(0)
        return None


class TelemetryMonitorTest(TestCase):
    def test_subscription_receives_each_complete_snapshot(self):
        frames = [
            SimpleNamespace(
                timestamp_ms=index,
                positions=[index] * 6,
                loads=[0] * 6,
                speeds=[0] * 6,
            )
            for index in range(3)
        ]
        monitor = TelemetryMonitor(
            FakeTelemetryAPI(frames), num_servos=6, read_timeout_s=0.01
        )
        subscription = monitor.subscribe()
        monitor.start()
        try:
            snapshots = [subscription.get(timeout=1.0) for _ in range(3)]
        finally:
            monitor.unsubscribe(subscription)
            monitor.stop()
        self.assertEqual([snapshot.sequence for snapshot in snapshots], [1, 2, 3])
        self.assertEqual([snapshot.timestamp_ms for snapshot in snapshots], [0, 1, 2])

    def test_unsubscribe_stops_delivery(self):
        frame = SimpleNamespace(
            timestamp_ms=1,
            positions=[1] * 6,
            loads=[0] * 6,
            speeds=[0] * 6,
        )
        monitor = TelemetryMonitor(
            FakeTelemetryAPI([frame]), num_servos=6, read_timeout_s=0.01
        )
        subscription = monitor.subscribe()
        monitor.unsubscribe(subscription)
        monitor.start()
        try:
            with self.assertRaises(Empty):
                subscription.get(timeout=0.05)
        finally:
            monitor.stop()
