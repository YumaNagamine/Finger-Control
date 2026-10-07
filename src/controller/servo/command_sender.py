"""Project-owned command helpers layered on top of ServoHardwareAPI."""

from __future__ import annotations

from collections.abc import Iterable, Mapping
from typing import Any


class ServoCommandSender:
    """Send project-level servo command groups through the current driver.

    The third-party ``ServoControllerDriver`` remains responsible for the
    firmware protocol.  This adapter owns project conventions such as sending
    a group of physical-ID speed commands in one transport write, so trajectory
    control, force control, and future reinforcement-learning policies can use
    the same interface.
    """

    def __init__(self, servo_driver: Any) -> None:
        self._driver = servo_driver

    @property
    def driver(self) -> Any:
        """Return the wrapped third-party driver for telemetry and commands."""

        return self._driver

    def send_speed_by_id(self, servo_id: int, speed: int) -> None:
        """Forward one physical-ID speed command to the current driver."""

        self._driver.send_speed_by_id(servo_id, speed)

    def send_speed_to_servos(
        self,
        speed_by_servo_id: Mapping[int, int],
    ) -> None:
        """Send speed commands for multiple configured physical servo IDs.

        All IDs and speeds are validated before the first write.  Commands are
        ordered by ``ServoSystem.servo_ids``.  With the current transport this
        is one serial write containing newline-delimited firmware commands; the
        firmware still handles those lines sequentially.
        """

        if not isinstance(speed_by_servo_id, Mapping):
            raise TypeError("speed_by_servo_id must be a mapping")
        if not speed_by_servo_id:
            raise ValueError("speed_by_servo_id must not be empty")

        system = getattr(self._driver, "system", None)
        configured_ids = tuple(getattr(system, "servo_ids", ()))
        if not configured_ids:
            raise ValueError("The wrapped driver has no configured servo IDs")

        normalized: dict[int, int] = {}
        for raw_servo_id, raw_speed in speed_by_servo_id.items():
            try:
                servo_id = int(raw_servo_id)
                speed = int(raw_speed)
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    "servo IDs and speeds must be integer-compatible"
                ) from exc
            if servo_id not in configured_ids:
                raise ValueError("Servo ID {} is not configured.".format(servo_id))
            normalized[servo_id] = speed

        commands = [
            self._driver.speed_command_by_id(servo_id, normalized[servo_id])
            for servo_id in configured_ids
            if servo_id in normalized
        ]
        if not commands:
            raise ValueError("speed_by_servo_id contains no configured servo IDs")

        transport = getattr(self._driver, "transport", None)
        if transport is None:
            raise RuntimeError("No transport configured for ServoControllerDriver.")

        write = getattr(transport, "write", None)
        if callable(write):
            write("".join(commands))
            return

        write_line = getattr(transport, "write_line", None)
        if not callable(write_line):
            raise RuntimeError(
                "Servo transport must expose write or write_line for speed commands."
            )
        for command in commands:
            write_line(command.rstrip("\n"))

    def stop_servos(self, servo_ids: Iterable[int] | None = None) -> None:
        """Send zero-speed commands to selected or all configured servos."""

        configured_ids = tuple(self._driver.system.servo_ids)
        selected_ids = configured_ids if servo_ids is None else tuple(servo_ids)
        self.send_speed_to_servos({servo_id: 0 for servo_id in selected_ids})
