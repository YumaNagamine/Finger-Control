"""Asynchronous latest-command dispatch for project-owned servo control."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass
from typing import Mapping

from controller.servo.command_sender import ServoCommandSender


@dataclass(frozen=True)
class CommandReceipt:
    """Result of one attempted transport write.

    ``write_finished_at`` means that the command was written to the PC-side
    serial transport.  It is not an acknowledgement that a physical servo
    applied the command.
    """

    sequence: int
    generated_at: float
    write_started_at: float | None
    write_finished_at: float | None
    status: str
    error: str | None = None

    @property
    def write_duration_s(self) -> float | None:
        if self.write_started_at is None or self.write_finished_at is None:
            return None
        return self.write_finished_at - self.write_started_at


class CommandTicket:
    """A non-blocking handle that can be waited on when a send time matters."""

    def __init__(self, sequence: int, generated_at: float) -> None:
        self.sequence = int(sequence)
        self.generated_at = float(generated_at)
        self._event = threading.Event()
        self._receipt: CommandReceipt | None = None

    @property
    def done(self) -> bool:
        return self._event.is_set()

    @property
    def receipt(self) -> CommandReceipt | None:
        return self._receipt

    def wait(self, timeout_s: float | None = None) -> CommandReceipt:
        if not self._event.wait(timeout_s):
            raise TimeoutError(
                f"Timed out waiting for servo command {self.sequence}"
            )
        assert self._receipt is not None
        return self._receipt

    def _complete(self, receipt: CommandReceipt) -> None:
        self._receipt = receipt
        self._event.set()


class ServoCommandDispatcher:
    """Send only the newest pending speed command from one worker thread.

    The dispatcher deliberately does not retry indefinitely.  A command that
    is still pending may be superseded by a newer control-cycle command, and a
    transport error is latched for the owner to handle through ``check_health``.
    """

    def __init__(self, sender: ServoCommandSender) -> None:
        self._sender = sender
        self._condition = threading.Condition()
        self._pending: tuple[CommandTicket, dict[int, int]] | None = None
        self._sequence = 0
        self._thread: threading.Thread | None = None
        self._started = False
        self._stopping = False
        self._last_error: Exception | None = None

    @property
    def last_error(self) -> Exception | None:
        with self._condition:
            return self._last_error

    def start(self) -> None:
        with self._condition:
            if self._started:
                raise RuntimeError("ServoCommandDispatcher has already started")
            if self._stopping:
                raise RuntimeError("ServoCommandDispatcher is closed")
            self._started = True
            self._thread = threading.Thread(
                target=self._run,
                name="servo-command-dispatcher",
                daemon=True,
            )
            self._thread.start()

    def submit_speed_to_servos(
        self,
        speed_by_servo_id: Mapping[int, int],
    ) -> CommandTicket:
        if not isinstance(speed_by_servo_id, Mapping):
            raise TypeError("speed_by_servo_id must be a mapping")
        if not speed_by_servo_id:
            raise ValueError("speed_by_servo_id must not be empty")
        try:
            normalized = {
                int(servo_id): int(speed)
                for servo_id, speed in speed_by_servo_id.items()
            }
        except (TypeError, ValueError) as exc:
            raise ValueError(
                "servo IDs and speeds must be integer-compatible"
            ) from exc

        generated_at = time.monotonic()
        with self._condition:
            if not self._started or self._stopping:
                raise RuntimeError("ServoCommandDispatcher is not running")
            self._sequence += 1
            ticket = CommandTicket(self._sequence, generated_at)
            previous = self._pending
            self._pending = (ticket, normalized)
            if previous is not None:
                previous_ticket, _ = previous
                previous_ticket._complete(
                    CommandReceipt(
                        sequence=previous_ticket.sequence,
                        generated_at=previous_ticket.generated_at,
                        write_started_at=None,
                        write_finished_at=None,
                        status="superseded",
                    )
                )
            self._condition.notify()
            return ticket

    def stop_servos(
        self,
        servo_ids: tuple[int, ...] | None = None,
        *,
        timeout_s: float = 1.0,
    ) -> CommandReceipt:
        configured_ids = tuple(self._sender.driver.system.servo_ids)
        selected_ids = configured_ids if servo_ids is None else tuple(servo_ids)
        ticket = self.submit_speed_to_servos(
            {int(servo_id): 0 for servo_id in selected_ids}
        )
        receipt = ticket.wait(timeout_s)
        if receipt.status != "sent":
            raise RuntimeError(
                f"Unable to send stop command: {receipt.status}"
            )
        return receipt

    def check_health(self) -> None:
        error = self.last_error
        if error is not None:
            raise RuntimeError("Servo command dispatcher failed") from error

    def close(self, *, timeout_s: float = 1.0) -> None:
        with self._condition:
            if not self._started or self._stopping:
                return
            self._stopping = True
            pending = self._pending
            self._pending = None
            if pending is not None:
                pending_ticket, _ = pending
                pending_ticket._complete(
                    CommandReceipt(
                        sequence=pending_ticket.sequence,
                        generated_at=pending_ticket.generated_at,
                        write_started_at=None,
                        write_finished_at=None,
                        status="stopped",
                    )
                )
            thread = self._thread
            self._condition.notify_all()
        if thread is not None:
            thread.join(timeout=timeout_s)
        with self._condition:
            self._thread = None
            self._started = False

    def _run(self) -> None:
        while True:
            with self._condition:
                while self._pending is None and not self._stopping:
                    self._condition.wait()
                if self._pending is None and self._stopping:
                    return
                assert self._pending is not None
                ticket, commands = self._pending
                self._pending = None

            write_started_at = time.monotonic()
            try:
                self._sender.send_speed_to_servos(commands)
            except Exception as exc:
                write_finished_at = time.monotonic()
                with self._condition:
                    self._last_error = exc
                ticket._complete(
                    CommandReceipt(
                        sequence=ticket.sequence,
                        generated_at=ticket.generated_at,
                        write_started_at=write_started_at,
                        write_finished_at=write_finished_at,
                        status="error",
                        error=f"{type(exc).__name__}: {exc}",
                    )
                )
            else:
                ticket._complete(
                    CommandReceipt(
                        sequence=ticket.sequence,
                        generated_at=ticket.generated_at,
                        write_started_at=write_started_at,
                        write_finished_at=time.monotonic(),
                        status="sent",
                    )
                )
