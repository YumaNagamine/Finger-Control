"""Telemetry-aware position control built on top of the thin ServoAPI."""

from __future__ import annotations

import math
import threading
import time
from dataclasses import dataclass
from enum import Enum
from typing import Callable, Sequence

from servo.control.telemetry_monitor import TelemetryMonitor, TelemetrySnapshot


class RetryPolicy(Enum):
    NEVER = "never"
    ON_NO_START = "on_no_start"


class PositionControlState(Enum):
    UNPREPARED = "unprepared"
    PRIMING = "priming"
    READY = "ready"
    MOVING = "moving"
    FAILED = "failed"


class PositionControlError(RuntimeError):
    """Base class for reliable position-control failures."""


class TelemetryUnavailableError(PositionControlError):
    pass


class PositionControlNotPreparedError(PositionControlError):
    pass


class PositionStartTimeoutError(PositionControlError):
    pass


class PositionArrivalTimeoutError(PositionControlError):
    pass


class PositionControlCancelledError(PositionControlError):
    pass


@dataclass(frozen=True)
class PositionControlConfig:
    telemetry_stale_s: float = 0.5
    telemetry_wait_s: float = 3.0
    id_map_reset_wait_s: float = 0.1
    speed_init_wait_s: float = 0.2
    prime_command_count: int = 2
    prime_interval_s: float = 0.2
    start_observation_s: float = 0.3
    start_min_delta: int = 3
    position_tolerance: int = 10
    speed_tolerance: int = 5
    stable_frame_count: int = 3
    arrival_timeout_s: float = 3.0
    max_start_retries: int = 1
    reset_id_map_on_prepare: bool = True
    position_min: int = 0
    position_max: int = 4095

    def __post_init__(self) -> None:
        positive_floats = {
            "telemetry_stale_s": self.telemetry_stale_s,
            "telemetry_wait_s": self.telemetry_wait_s,
            "start_observation_s": self.start_observation_s,
            "arrival_timeout_s": self.arrival_timeout_s,
        }
        for name, value in positive_floats.items():
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{name} must be finite and greater than zero")

        non_negative_floats = {
            "id_map_reset_wait_s": self.id_map_reset_wait_s,
            "speed_init_wait_s": self.speed_init_wait_s,
            "prime_interval_s": self.prime_interval_s,
        }
        for name, value in non_negative_floats.items():
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be finite and non-negative")

        non_negative_ints = {
            "start_min_delta": self.start_min_delta,
            "position_tolerance": self.position_tolerance,
            "speed_tolerance": self.speed_tolerance,
            "max_start_retries": self.max_start_retries,
        }
        for name, value in non_negative_ints.items():
            if value < 0:
                raise ValueError(f"{name} must be non-negative")

        if self.prime_command_count <= 0:
            raise ValueError("prime_command_count must be greater than zero")
        if self.stable_frame_count <= 0:
            raise ValueError("stable_frame_count must be greater than zero")
        if self.position_min > self.position_max:
            raise ValueError("position_min must not exceed position_max")


@dataclass(frozen=True)
class PositionProgress:
    phase: str
    servo_id: int
    target_position: int
    actual_position: int
    speed: int
    load: int
    stable_frames: int
    attempt: int


@dataclass(frozen=True)
class MoveResult:
    servo_id: int
    start_position: int
    target_position: int
    final_position: int
    final_speed: int
    retries: int
    elapsed_s: float


ProgressCallback = Callable[[PositionProgress], None]
CancelCheck = Callable[[], bool]


class ReliablePositionController:
    """Adds priming, start detection, bounded retries, and arrival checks."""

    def __init__(
        self,
        api,
        telemetry: TelemetryMonitor,
        config: PositionControlConfig | None = None,
    ) -> None:
        self._api = api
        self._telemetry = telemetry
        self.config = config or PositionControlConfig()
        self._states: dict[int, PositionControlState] = {}
        self._command_lock = threading.RLock()

    def state(self, servo_id: int) -> PositionControlState:
        return self._states.get(servo_id, PositionControlState.UNPREPARED)

    def invalidate(self) -> None:
        for servo_id in tuple(self._states):
            self._states[servo_id] = PositionControlState.UNPREPARED

    def current_position(self, servo_id: int) -> int:
        snapshot = self._fresh_snapshot()
        self._validate_servo_index(snapshot, servo_id)
        return snapshot.positions[servo_id]

    def prepare(
        self,
        servo_ids: Sequence[int],
        *,
        force_init_servo_ids: Sequence[int] | None = None,
    ) -> None:
        position_servo_ids = self._unique_servo_ids(servo_ids)
        init_servo_ids = self._unique_servo_ids(
            position_servo_ids
            if force_init_servo_ids is None
            else force_init_servo_ids
        )
        if not position_servo_ids:
            raise ValueError("servo_ids must not be empty")

        with self._command_lock:
            for servo_id in position_servo_ids:
                self._states[servo_id] = PositionControlState.PRIMING

            try:
                initial = self._wait_for_any_snapshot()
                self._validate_servo_ids(initial, position_servo_ids)
                self._validate_servo_ids(initial, init_servo_ids)

                if self.config.reset_id_map_on_prepare:
                    sequence = initial.sequence
                    self._api.reset_ids()
                    self._sleep(self.config.id_map_reset_wait_s)
                    initial = self._wait_for_newer(sequence)

                for servo_id in init_servo_ids:
                    self._api.set_speed(servo_id, 0, force_init=True)
                self._sleep(self.config.speed_init_wait_s)
                snapshot = self._wait_for_newer(initial.sequence)

                for servo_id in position_servo_ids:
                    for _ in range(self.config.prime_command_count):
                        current = snapshot.positions[servo_id]
                        sequence = snapshot.sequence
                        self._api.set_position(servo_id, current, time_ms=0)
                        self._sleep(self.config.prime_interval_s)
                        snapshot = self._wait_for_newer(sequence)

                    snapshot = self._wait_until_stable(
                        servo_id,
                        snapshot.positions[servo_id],
                        snapshot,
                        deadline=time.monotonic() + self.config.telemetry_wait_s,
                        attempt=0,
                        phase="priming",
                    )
                    self._states[servo_id] = PositionControlState.READY
            except Exception:
                self._fail_safe_stop(position_servo_ids)
                raise

    def move_and_wait(
        self,
        servo_id: int,
        target_position: int,
        *,
        time_ms: int = 0,
        retry_policy: RetryPolicy = RetryPolicy.NEVER,
        progress_callback: ProgressCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> MoveResult:
        if self.state(servo_id) is not PositionControlState.READY:
            raise PositionControlNotPreparedError(
                f"Servo {servo_id} is not ready for position control"
            )
        if not self.config.position_min <= target_position <= self.config.position_max:
            raise ValueError(
                f"target_position must be {self.config.position_min}-"
                f"{self.config.position_max}, got {target_position}"
            )
        if not 0 <= time_ms <= 65535:
            raise ValueError("time_ms must be in the range 0-65535")
        if self.config.arrival_timeout_s < time_ms / 1000.0:
            raise ValueError("arrival_timeout_s must cover time_ms")

        with self._command_lock:
            started_at = time.monotonic()
            deadline = started_at + self.config.arrival_timeout_s
            start_snapshot = self._fresh_snapshot()
            self._validate_servo_index(start_snapshot, servo_id)
            self._require_speeds(start_snapshot)
            start_position = start_snapshot.positions[servo_id]
            self._states[servo_id] = PositionControlState.MOVING

            try:
                retries = 0
                snapshot = start_snapshot
                while True:
                    self._check_cancel(cancel_check)
                    sequence = snapshot.sequence
                    baseline_position = snapshot.positions[servo_id]
                    self._api.set_position(
                        servo_id,
                        target_position,
                        time_ms=time_ms,
                    )
                    outcome, snapshot = self._observe_start(
                        servo_id,
                        target_position,
                        baseline_position,
                        sequence,
                        attempt=retries + 1,
                        deadline=min(
                            deadline,
                            time.monotonic() + self.config.start_observation_s,
                        ),
                        progress_callback=progress_callback,
                        cancel_check=cancel_check,
                    )
                    if outcome == "arrived":
                        break
                    if outcome == "started":
                        snapshot = self._wait_until_stable(
                            servo_id,
                            target_position,
                            snapshot,
                            deadline=deadline,
                            attempt=retries + 1,
                            phase="moving",
                            progress_callback=progress_callback,
                            cancel_check=cancel_check,
                        )
                        break

                    can_retry = (
                        retry_policy is RetryPolicy.ON_NO_START
                        and retries < self.config.max_start_retries
                    )
                    if not can_retry:
                        raise PositionStartTimeoutError(
                            f"Servo {servo_id} did not start toward target "
                            f"{target_position} after {retries + 1} attempt(s)"
                        )
                    retries += 1

                final_speed = self._speed(snapshot, servo_id)
                self._states[servo_id] = PositionControlState.READY
                return MoveResult(
                    servo_id=servo_id,
                    start_position=start_position,
                    target_position=target_position,
                    final_position=snapshot.positions[servo_id],
                    final_speed=final_speed,
                    retries=retries,
                    elapsed_s=time.monotonic() - started_at,
                )
            except PositionControlError:
                self._fail_safe_stop((servo_id,))
                raise
            except TimeoutError as exc:
                self._fail_safe_stop((servo_id,))
                raise TelemetryUnavailableError(str(exc)) from exc
            except Exception:
                self._fail_safe_stop((servo_id,))
                raise

    def stop_all(self) -> None:
        with self._command_lock:
            try:
                self._api.stop_all()
            finally:
                self.invalidate()

    def _observe_start(
        self,
        servo_id: int,
        target_position: int,
        baseline_position: int,
        sequence: int,
        *,
        attempt: int,
        deadline: float,
        progress_callback: ProgressCallback | None,
        cancel_check: CancelCheck | None,
    ) -> tuple[str, TelemetrySnapshot]:
        stable_frames = 0
        snapshot = self._fresh_snapshot()
        direction = 1 if target_position >= baseline_position else -1

        while time.monotonic() < deadline:
            self._check_cancel(cancel_check)
            try:
                snapshot = self._wait_for_newer_until(sequence, deadline)
            except TelemetryUnavailableError:
                if (
                    time.monotonic() >= deadline
                    and self._telemetry.latest(self.config.telemetry_stale_s)
                    is not None
                ):
                    break
                raise
            sequence = snapshot.sequence
            self._require_speeds(snapshot)
            actual = snapshot.positions[servo_id]
            speed = self._speed(snapshot, servo_id)

            if self._is_settled(actual, speed, target_position):
                stable_frames += 1
            else:
                stable_frames = 0
            self._report_progress(
                progress_callback,
                "starting",
                servo_id,
                target_position,
                snapshot,
                stable_frames,
                attempt,
            )
            if stable_frames >= self.config.stable_frame_count:
                return "arrived", snapshot
            directed_progress = direction * (actual - baseline_position)
            if directed_progress >= self.config.start_min_delta:
                return "started", snapshot

        return "not_started", snapshot

    def _wait_until_stable(
        self,
        servo_id: int,
        target_position: int,
        snapshot: TelemetrySnapshot,
        *,
        deadline: float,
        attempt: int,
        phase: str,
        progress_callback: ProgressCallback | None = None,
        cancel_check: CancelCheck | None = None,
    ) -> TelemetrySnapshot:
        stable_frames = 0
        sequence = snapshot.sequence

        while time.monotonic() < deadline:
            self._check_cancel(cancel_check)
            try:
                snapshot = self._wait_for_newer_until(sequence, deadline)
            except TelemetryUnavailableError:
                if (
                    time.monotonic() >= deadline
                    and self._telemetry.latest(self.config.telemetry_stale_s)
                    is not None
                ):
                    break
                raise
            sequence = snapshot.sequence
            self._require_speeds(snapshot)
            actual = snapshot.positions[servo_id]
            speed = self._speed(snapshot, servo_id)
            if self._is_settled(actual, speed, target_position):
                stable_frames += 1
            else:
                stable_frames = 0
            self._report_progress(
                progress_callback,
                phase,
                servo_id,
                target_position,
                snapshot,
                stable_frames,
                attempt,
            )
            if stable_frames >= self.config.stable_frame_count:
                return snapshot

        raise PositionArrivalTimeoutError(
            f"Servo {servo_id} did not arrive at target {target_position} within "
            f"the configured timeout"
        )

    def _wait_for_any_snapshot(self) -> TelemetrySnapshot:
        snapshot = self._telemetry.latest(self.config.telemetry_stale_s)
        if snapshot is not None:
            return snapshot
        try:
            return self._telemetry.wait_for_newer(0, self.config.telemetry_wait_s)
        except (RuntimeError, TimeoutError) as exc:
            raise TelemetryUnavailableError(
                "Fresh servo telemetry is not available"
            ) from exc

    def _fresh_snapshot(self) -> TelemetrySnapshot:
        snapshot = self._telemetry.latest(self.config.telemetry_stale_s)
        if snapshot is None:
            raise TelemetryUnavailableError("Fresh servo telemetry is not available")
        return snapshot

    def _wait_for_newer(self, sequence: int) -> TelemetrySnapshot:
        try:
            return self._telemetry.wait_for_newer(
                sequence,
                self.config.telemetry_wait_s,
            )
        except (RuntimeError, TimeoutError) as exc:
            raise TelemetryUnavailableError(
                f"Fresh telemetry did not arrive after sequence {sequence}"
            ) from exc

    def _wait_for_newer_until(
        self,
        sequence: int,
        deadline: float,
    ) -> TelemetrySnapshot:
        remaining_s = deadline - time.monotonic()
        if remaining_s <= 0.0:
            raise TelemetryUnavailableError("Position-control deadline expired")
        try:
            return self._telemetry.wait_for_newer(sequence, remaining_s)
        except TimeoutError as exc:
            raise TelemetryUnavailableError(
                "No telemetry arrived before the position-control deadline"
            ) from exc
        except RuntimeError as exc:
            raise TelemetryUnavailableError("Telemetry monitor failed") from exc

    def _is_settled(self, actual: int, speed: int, target: int) -> bool:
        return (
            abs(target - actual) <= self.config.position_tolerance
            and abs(speed) <= self.config.speed_tolerance
        )

    def _require_speeds(self, snapshot: TelemetrySnapshot) -> None:
        if snapshot.speeds is None:
            raise TelemetryUnavailableError(
                "Position control requires 19-field telemetry with speed values"
            )

    @staticmethod
    def _speed(snapshot: TelemetrySnapshot, servo_id: int) -> int:
        assert snapshot.speeds is not None
        return snapshot.speeds[servo_id]

    def _report_progress(
        self,
        callback: ProgressCallback | None,
        phase: str,
        servo_id: int,
        target_position: int,
        snapshot: TelemetrySnapshot,
        stable_frames: int,
        attempt: int,
    ) -> None:
        if callback is None:
            return
        callback(
            PositionProgress(
                phase=phase,
                servo_id=servo_id,
                target_position=target_position,
                actual_position=snapshot.positions[servo_id],
                speed=self._speed(snapshot, servo_id),
                load=snapshot.loads[servo_id],
                stable_frames=stable_frames,
                attempt=attempt,
            )
        )

    @staticmethod
    def _check_cancel(cancel_check: CancelCheck | None) -> None:
        if cancel_check is not None and cancel_check():
            raise PositionControlCancelledError("Position movement was cancelled")

    @staticmethod
    def _sleep(duration_s: float) -> None:
        if duration_s > 0.0:
            time.sleep(duration_s)

    @staticmethod
    def _unique_servo_ids(servo_ids: Sequence[int]) -> tuple[int, ...]:
        result: list[int] = []
        for servo_id in servo_ids:
            value = int(servo_id)
            if value < 0:
                raise ValueError(f"servo_id must be non-negative, got {value}")
            if value not in result:
                result.append(value)
        return tuple(result)

    @staticmethod
    def _validate_servo_index(snapshot: TelemetrySnapshot, servo_id: int) -> None:
        if not 0 <= servo_id < len(snapshot.positions):
            raise ValueError(
                f"servo_id must be 0-{len(snapshot.positions) - 1}, got {servo_id}"
            )

    def _validate_servo_ids(
        self,
        snapshot: TelemetrySnapshot,
        servo_ids: Sequence[int],
    ) -> None:
        for servo_id in servo_ids:
            self._validate_servo_index(snapshot, servo_id)

    def _fail_safe_stop(self, servo_ids: Sequence[int]) -> None:
        for servo_id in servo_ids:
            self._states[servo_id] = PositionControlState.FAILED
        self._api.stop_all()
