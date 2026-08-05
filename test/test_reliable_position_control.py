from __future__ import annotations

import threading
import time
import unittest
from dataclasses import dataclass, replace

from servo.control import (
    PositionControlConfig,
    PositionControlState,
    PositionStartTimeoutError,
    ReliablePositionController,
    RetryPolicy,
    TelemetryMonitor,
)


@dataclass
class FakeTelemetryFrame:
    timestamp_ms: int
    positions: list[int]
    loads: list[int]
    speeds: list[int]


class FakeServoAPI:
    def __init__(self, num_servos: int = 6) -> None:
        self._lock = threading.Lock()
        self._started_at = time.monotonic()
        self.positions = [100] * num_servos
        self.loads = [0] * num_servos
        self.speeds = [0] * num_servos
        self.position_commands: list[tuple[int, int, int]] = []
        self.speed_commands: list[tuple[int, int, bool]] = []
        self.ignore_next_position_commands = 0
        self.multiturn_position_commands: list[tuple[int, int, int]] = []
        self.offset_next_position_commands = 0
        self.reverse_next_position_commands = 0
        self.stop_count = 0
        self.reset_count = 0

    def try_read_telemetry(self):
        time.sleep(0.002)
        with self._lock:
            return FakeTelemetryFrame(
                timestamp_ms=int((time.monotonic() - self._started_at) * 1000),
                positions=list(self.positions),
                loads=list(self.loads),
                speeds=list(self.speeds),
            )

    def reset_ids(self) -> None:
        with self._lock:
            self.reset_count += 1

    def set_speed(self, servo_id: int, speed: int, force_init: bool = False) -> None:
        with self._lock:
            self.speeds[servo_id] = speed
            self.speed_commands.append((servo_id, speed, force_init))

    def set_position(self, servo_id: int, position: int, time_ms: int = 0) -> None:
        with self._lock:
            self.position_commands.append((servo_id, position, time_ms))
            if self.ignore_next_position_commands > 0:
                self.ignore_next_position_commands -= 1
                return
            if self.reverse_next_position_commands > 0:
                self.reverse_next_position_commands -= 1
                direction = 1 if position >= self.positions[servo_id] else -1
                self.positions[servo_id] -= direction * 10
                return
            self.positions[servo_id] = position
            if self.offset_next_position_commands:
                self.positions[servo_id] = position + self.offset_next_position_commands
                self.offset_next_position_commands = 0
                return
            self.speeds[servo_id] = 0

    def set_multiturn_position(
        self,
        servo_id: int,
        position: int,
        time_ms: int = 0,
    ) -> None:
        with self._lock:
            self.multiturn_position_commands.append(
                (servo_id, position, time_ms)
            )
            self.positions[servo_id] = position
            self.speeds[servo_id] = 0

    def stop_all(self) -> None:
        with self._lock:
            self.stop_count += 1
            self.speeds = [0] * len(self.speeds)


def test_config() -> PositionControlConfig:
    return PositionControlConfig(
        telemetry_stale_s=0.1,
        telemetry_wait_s=0.2,
        id_map_reset_wait_s=0.0,
        speed_init_wait_s=0.0,
        prime_command_count=2,
        prime_interval_s=0.0,
        start_observation_s=0.03,
        start_min_delta=2,
        position_tolerance=1,
        speed_tolerance=0,
        stable_frame_count=2,
        arrival_timeout_s=0.2,
        max_start_retries=1,
        reset_id_map_on_prepare=True,
    )


class ReliablePositionControllerTest(unittest.TestCase):
    def setUp(self) -> None:
        self.api = FakeServoAPI()
        self.telemetry = TelemetryMonitor(
            self.api,
            num_servos=6,
            read_timeout_s=0.01,
        )
        self.controller = ReliablePositionController(
            self.api,
            self.telemetry,
            test_config(),
        )
        self.telemetry.start()

    def tearDown(self) -> None:
        self.telemetry.stop()

    def prepare_servo(self) -> None:
        self.controller.prepare((5,), force_init_servo_ids=tuple(range(6)))

    def test_prepare_primes_only_requested_position_servo(self) -> None:
        self.prepare_servo()

        self.assertEqual(self.api.reset_count, 1)
        self.assertEqual(self.controller.state(5), PositionControlState.READY)
        self.assertEqual(
            [command[0] for command in self.api.position_commands],
            [5, 5],
        )
        self.assertEqual(self.controller.state(0), PositionControlState.UNPREPARED)

    def test_prepare_keeps_one_fixed_prime_target(self) -> None:
        self.api.offset_next_position_commands = 5

        self.prepare_servo()

        self.assertEqual(
            self.api.position_commands,
            [(5, 100, 0), (5, 100, 0)],
        )

    def test_multiturn_prepare_does_not_force_wheel_mode(self) -> None:
        controller = ReliablePositionController(
            self.api,
            self.telemetry,
            replace(
                test_config(),
                multiturn=True,
                position_min=-28672,
                position_max=28672,
            ),
        )

        controller.prepare((5,), force_init_servo_ids=tuple(range(6)))

        self.assertEqual(self.api.speed_commands, [])
        self.assertEqual(
            self.api.multiturn_position_commands,
            [(5, 100, 0), (5, 100, 0)],
        )

    def test_move_retries_once_when_first_real_command_is_ignored(self) -> None:
        self.prepare_servo()
        prime_command_count = len(self.api.position_commands)
        self.api.ignore_next_position_commands = 1

        result = self.controller.move_and_wait(
            5,
            140,
            retry_policy=RetryPolicy.ON_NO_START,
        )

        self.assertEqual(result.final_position, 140)
        self.assertEqual(result.retries, 1)
        self.assertEqual(
            len(self.api.position_commands) - prime_command_count,
            2,
        )
        self.assertEqual(self.controller.state(5), PositionControlState.READY)

    def test_move_stops_and_fails_after_retry_limit(self) -> None:
        self.prepare_servo()
        self.api.ignore_next_position_commands = 10

        with self.assertRaises(PositionStartTimeoutError):
            self.controller.move_and_wait(
                5,
                140,
                retry_policy=RetryPolicy.ON_NO_START,
            )

        self.assertGreaterEqual(self.api.stop_count, 1)
        self.assertEqual(self.controller.state(5), PositionControlState.FAILED)

    def test_move_retries_when_first_command_only_moves_in_reverse(self) -> None:
        self.prepare_servo()
        self.api.reverse_next_position_commands = 1

        result = self.controller.move_and_wait(
            5,
            140,
            retry_policy=RetryPolicy.ON_NO_START,
        )

        self.assertEqual(result.final_position, 140)
        self.assertEqual(result.retries, 1)

    def test_never_retry_policy_sends_only_once(self) -> None:
        self.prepare_servo()
        prime_command_count = len(self.api.position_commands)
        self.api.ignore_next_position_commands = 1

        with self.assertRaises(PositionStartTimeoutError):
            self.controller.move_and_wait(
                5,
                140,
                retry_policy=RetryPolicy.NEVER,
            )

        self.assertEqual(
            len(self.api.position_commands) - prime_command_count,
            1,
        )


    def test_stream_positions_requires_prepared_servos(self) -> None:
        with self.assertRaisesRegex(Exception, "not ready"):
            self.controller.stream_positions((5,), (140,))

        self.assertEqual(self.api.position_commands, [])

    def test_stream_positions_sends_all_targets_without_waiting(self) -> None:
        self.controller.prepare(
            (0, 1, 2, 3, 4, 5),
            force_init_servo_ids=tuple(range(6)),
        )
        prime_command_count = len(self.api.position_commands)

        result = self.controller.stream_positions(
            (0, 1, 2, 3, 4, 5),
            (110, 120, 130, 140, 150, 160),
            time_ms=0,
        )

        self.assertEqual(
            self.api.position_commands[prime_command_count:],
            [
                (0, 110, 0),
                (1, 120, 0),
                (2, 130, 0),
                (3, 140, 0),
                (4, 150, 0),
                (5, 160, 0),
            ],
        )
        self.assertEqual(result.target_positions, (110, 120, 130, 140, 150, 160))
        self.assertTrue(
            all(
                self.controller.state(servo_id) is PositionControlState.READY
                for servo_id in range(6)
            )
        )

    def test_stream_positions_rejects_out_of_range_target(self) -> None:
        self.prepare_servo()
        prime_command_count = len(self.api.position_commands)

        with self.assertRaises(ValueError):
            self.controller.stream_positions((5,), (5000,))

        self.assertEqual(len(self.api.position_commands), prime_command_count)

    def test_multiturn_stream_accepts_extended_position(self) -> None:
        controller = ReliablePositionController(
            self.api,
            self.telemetry,
            replace(
                test_config(),
                multiturn=True,
                position_min=-28672,
                position_max=28672,
            ),
        )
        controller.prepare((5,), force_init_servo_ids=tuple(range(6)))
        prime_command_count = len(self.api.multiturn_position_commands)

        result = controller.stream_positions((5,), (5000,), time_ms=0)

        self.assertEqual(
            self.api.multiturn_position_commands[prime_command_count:],
            [(5, 5000, 0)],
        )
        self.assertEqual(result.target_positions, (5000,))

    def test_multiturn_stream_rejects_configured_out_of_range_target(self) -> None:
        controller = ReliablePositionController(
            self.api,
            self.telemetry,
            replace(test_config(), multiturn=True, position_max=6000),
        )
        controller.prepare((5,), force_init_servo_ids=tuple(range(6)))

        with self.assertRaises(ValueError):
            controller.stream_positions((5,), (6001,))

if __name__ == "__main__":
    unittest.main()
