from __future__ import annotations

import threading
import time
import unittest
from dataclasses import dataclass

from servo.control import (
    AccumulatedPositionControlConfig,
    PositionControlConfig,
    PositionControlError,
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
        self.timed_run_commands: list[tuple[int, int, int]] = []
        self.ignore_next_position_commands = 0
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
        del force_init
        with self._lock:
            self.speeds[servo_id] = speed

    def timed_run(self, servo_id: int, speed: int, time_ms: int) -> None:
        with self._lock:
            self.timed_run_commands.append((servo_id, speed, time_ms))
            self.speeds[servo_id] = speed
            if speed == 0:
                return
            direction = 1 if speed > 0 else -1
            step = min(40, max(1, abs(speed) // 4))
            self.positions[servo_id] = (
                self.positions[servo_id] + direction * step
            ) % 4096

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
            AccumulatedPositionControlConfig(
                switch_to_position_threshold=20,
                wheel_kp=1.0,
                wheel_kd=0.01,
                wheel_min_speed=40,
                wheel_max_speed=400,
                wheel_command_lifetime_ms=100,
                wheel_telemetry_timeout_s=0.1,
                wheel_arrival_timeout_s=2.0,
                wheel_stop_timeout_s=0.2,
                wheel_stop_stable_frames=2,
            ),
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


    def test_accumulated_position_unwraps_forward_boundary(self) -> None:
        self.api.positions[5] = 4090
        time.sleep(0.01)
        self.controller.set_accumulated_reference(5, 4090)

        self.api.positions[5] = 5
        time.sleep(0.01)

        self.assertEqual(self.controller.current_accumulated_position(5), 4101)

    def test_accumulated_position_unwraps_reverse_boundary(self) -> None:
        self.api.positions[5] = 5
        time.sleep(0.01)
        self.controller.set_accumulated_reference(5, 5)

        self.api.positions[5] = 4090
        time.sleep(0.01)

        self.assertEqual(self.controller.current_accumulated_position(5), -6)

    def test_move_accumulated_crosses_boundary_and_finishes_in_position_mode(
        self,
    ) -> None:
        self.prepare_servo()
        self.controller.set_accumulated_reference(5, 100)

        result = self.controller.move_accumulated_and_wait(5, 4300)

        self.assertEqual(result.target_position, 4300)
        self.assertLessEqual(abs(result.final_position - 4300), 1)
        self.assertEqual(result.final_raw_position, 4300 % 4096)
        self.assertTrue(
            any(speed > 0 for _, speed, _ in self.api.timed_run_commands)
        )
        self.assertEqual(self.controller.state(5), PositionControlState.READY)

    def test_streaming_accumulated_control_returns_to_position_mode(self) -> None:
        self.prepare_servo()
        self.controller.set_accumulated_reference(5, 100)
        self.controller.begin_accumulated_control((5,))

        command = self.controller.command_accumulated_positions((5,), (4300,))
        self.assertGreater(command.speed_commands[0], 0)

        final = self.controller.end_accumulated_control((5,), (140,))
        self.assertEqual(final, (140,))
        self.assertEqual(self.controller.state(5), PositionControlState.READY)


    def test_streaming_rejects_unsafe_position_mode_transition(self) -> None:
        self.prepare_servo()
        self.controller.set_accumulated_reference(5, 100)
        self.controller.begin_accumulated_control((5,))
        self.controller.command_accumulated_positions((5,), (4300,))

        with self.assertRaises(PositionControlError):
            self.controller.end_accumulated_control((5,), (4300,))

        self.assertGreaterEqual(self.api.stop_count, 1)
        self.assertEqual(self.controller.state(5), PositionControlState.FAILED)


    def test_move_accumulated_crosses_reverse_boundary(self) -> None:
        self.prepare_servo()
        self.controller.set_accumulated_reference(5, 100)

        result = self.controller.move_accumulated_and_wait(5, -100)

        self.assertLessEqual(abs(result.final_position - (-100)), 1)
        self.assertEqual(result.final_raw_position, (-100) % 4096)
        self.assertTrue(
            any(speed < 0 for _, speed, _ in self.api.timed_run_commands)
        )
        self.assertEqual(self.controller.state(5), PositionControlState.READY)


if __name__ == "__main__":
    unittest.main()
