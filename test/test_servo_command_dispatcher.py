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


if __name__ == "__main__":
    unittest.main()
