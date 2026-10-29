from __future__ import annotations

import threading
import unittest

from controller.servo.command_dispatcher import ServoCommandDispatcher


class _FakeSystem:
    servo_ids = (1, 2)


class _FakeDriver:
    system = _FakeSystem()


class _FakeSender:
    driver = _FakeDriver()

    def __init__(self, *, block_first: bool = False, fail: bool = False) -> None:
        self.calls: list[dict[int, int]] = []
        self.block_first = block_first
        self.fail = fail
        self.first_started = threading.Event()
        self.release_first = threading.Event()

    def send_speed_to_servos(self, commands: dict[int, int]) -> None:
        if self.fail:
            raise OSError("serial write failed")
        self.calls.append(dict(commands))
        if self.block_first and len(self.calls) == 1:
            self.first_started.set()
            self.release_first.wait(1.0)


class ServoCommandDispatcherTest(unittest.TestCase):
    def test_submit_returns_receipt_after_background_write(self):
        sender = _FakeSender()
        dispatcher = ServoCommandDispatcher(sender)
        dispatcher.start()
        self.addCleanup(dispatcher.close)

        ticket = dispatcher.submit_speed_to_servos({1: 120, 2: -30})
        receipt = ticket.wait(1.0)

        self.assertEqual(receipt.status, "sent")
        self.assertIsNotNone(receipt.write_finished_at)
        self.assertEqual(sender.calls, [{1: 120, 2: -30}])

    def test_pending_commands_are_superseded_by_the_latest_command(self):
        sender = _FakeSender(block_first=True)
        dispatcher = ServoCommandDispatcher(sender)
        dispatcher.start()
        self.addCleanup(dispatcher.close)

        first = dispatcher.submit_speed_to_servos({1: 10})
        self.assertTrue(sender.first_started.wait(1.0))
        second = dispatcher.submit_speed_to_servos({1: 20})
        third = dispatcher.submit_speed_to_servos({1: 30})

        self.assertEqual(second.wait(1.0).status, "superseded")
        sender.release_first.set()
        self.assertEqual(first.wait(1.0).status, "sent")
        self.assertEqual(third.wait(1.0).status, "sent")
        self.assertEqual(sender.calls, [{1: 10}, {1: 30}])

    def test_transport_error_is_reported_without_blocking_submit(self):
        sender = _FakeSender(fail=True)
        dispatcher = ServoCommandDispatcher(sender)
        dispatcher.start()
        self.addCleanup(dispatcher.close)

        receipt = dispatcher.submit_speed_to_servos({1: 100}).wait(1.0)

        self.assertEqual(receipt.status, "error")
        with self.assertRaises(RuntimeError):
            dispatcher.check_health()

    def test_stop_servos_is_sent_and_waited_for(self):
        sender = _FakeSender()
        dispatcher = ServoCommandDispatcher(sender)
        dispatcher.start()
        self.addCleanup(dispatcher.close)

        receipt = dispatcher.stop_servos()

        self.assertEqual(receipt.status, "sent")
        self.assertEqual(sender.calls[-1], {1: 0, 2: 0})

    def test_latched_stop_supersedes_pending_motion_and_rejects_later_motion(self):
        sender = _FakeSender(block_first=True)
        dispatcher = ServoCommandDispatcher(sender)
        dispatcher.start()
        self.addCleanup(dispatcher.close)

        first = dispatcher.submit_speed_to_servos({1: 50, 2: 50})
        self.assertTrue(sender.first_started.wait(1.0))
        pending_motion = dispatcher.submit_speed_to_servos({1: 80, 2: 80})
        stop_ticket = dispatcher.request_latched_stop()

        self.assertEqual(pending_motion.wait(1.0).status, "superseded")
        with self.assertRaisesRegex(RuntimeError, "stop-latched"):
            dispatcher.submit_speed_to_servos({1: 10, 2: 10})
        sender.release_first.set()
        self.assertEqual(first.wait(1.0).status, "sent")
        self.assertEqual(stop_ticket.wait(1.0).status, "sent")
        self.assertEqual(sender.calls, [{1: 50, 2: 50}, {1: 0, 2: 0}])

        state = dispatcher.stop_state_snapshot()
        self.assertTrue(state["latched"])
        self.assertEqual(state["pc_write_status"], "sent")
        self.assertFalse(state["physical_stop_confirmed"])
        self.assertFalse(state["tension_safe_confirmed"])

    def test_latched_stop_does_not_claim_blocked_write_or_survive_reinitialization(self):
        sender = _FakeSender(block_first=True)
        dispatcher = ServoCommandDispatcher(sender)
        dispatcher.start()
        self.addCleanup(dispatcher.close)

        dispatcher.submit_speed_to_servos({1: 20, 2: 20})
        self.assertTrue(sender.first_started.wait(1.0))
        stop_ticket = dispatcher.request_latched_stop()
        with self.assertRaises(TimeoutError):
            stop_ticket.wait(0.01)
        state = dispatcher.stop_state_snapshot()
        self.assertEqual(state["pc_write_status"], "requested")
        self.assertFalse(state["physical_stop_confirmed"])

        sender.release_first.set()
        self.assertEqual(stop_ticket.wait(1.0).status, "sent")
        dispatcher.close()

        replacement_sender = _FakeSender()
        replacement = ServoCommandDispatcher(replacement_sender)
        replacement.start()
        self.addCleanup(replacement.close)
        replacement.submit_speed_to_servos({1: 25, 2: -25}).wait(1.0)

        # A new dispatcher is the only latch reset; no command from the old
        # pending queue is replayed into the replacement session.
        self.assertEqual(replacement_sender.calls, [{1: 25, 2: -25}])


if __name__ == "__main__":
    unittest.main()
