from __future__ import annotations

from servo.core.session import ServoSession


class ControlService:
    def __init__(self, session: ServoSession, num_motors: int) -> None:
        self._session = session
        self._num_motors = num_motors

    def set_speed(self, motor_id: int, speed: int, force_init: bool = False) -> None:
        self._session.set_speed(motor_id, int(speed), force_init)

    def set_position(self, motor_id: int, position: int, time_ms: int = 0) -> None:
        self._session.set_position(motor_id, int(position), int(time_ms))

    def timed_run(self, motor_id: int, speed: int, time_ms: int) -> None:
        self._session.timed_run(motor_id, int(speed), int(time_ms))

    def stop_all(self) -> None:
        self._session.stop_all()

    def reset_system(self) -> None:
        self._session.reset_system()

    def go_to_zero(self, motor_id: int) -> None:
        self._session.go_to_zero(motor_id)

    def go_all_to_zero(self) -> None:
        for motor_id in range(self._num_motors):
            self._session.go_to_zero(motor_id)

    def set_zero(self, motor_id: int) -> None:
        self._session.set_zero(motor_id)

    def set_all_zero(self) -> None:
        for motor_id in range(self._num_motors):
            self._session.set_zero(motor_id)
