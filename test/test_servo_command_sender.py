from __future__ import annotations

import unittest
from unittest.mock import Mock

from controller.servo.command_sender import ServoCommandSender


class _System:
    servo_ids = (1, 2, 3, 4, 5, 6)


class _Driver:
    def __init__(self, transport):
        self.system = _System()
        self.transport = transport

    def send_speed_by_id(self, servo_id, speed):
        raise AssertionError("single-command forwarding was not expected")

    @staticmethod
    def speed_command_by_id(servo_id, speed):
        return f"{servo_id},{speed}\n"


class ServoCommandSenderTest(unittest.TestCase):
    def test_sends_configured_order_in_one_write(self):
        transport = Mock()
        sender = ServoCommandSender(_Driver(transport))

        sender.send_speed_to_servos({4: -5, 1: 10})

        transport.write.assert_called_once_with("1,10\n4,-5\n")

    def test_rejects_invalid_id_before_writing(self):
        transport = Mock()
        sender = ServoCommandSender(_Driver(transport))

        with self.assertRaises(ValueError):
            sender.send_speed_to_servos({99: 1})

        transport.write.assert_not_called()

    def test_stop_servos_sends_zero_speed(self):
        transport = Mock()
        sender = ServoCommandSender(_Driver(transport))

        sender.stop_servos([4, 1])

        transport.write.assert_called_once_with("1,0\n4,0\n")


if __name__ == "__main__":
    unittest.main()
