"""Remember and safely restore persisted magnetic-encoder positions."""

from __future__ import annotations

import math
import threading
import time
from pathlib import Path
from typing import Any, Mapping

from controller.servo.control_common.pid import PID, PIDConfig
from controller.servo.hardware_session import HardwareFeedback, ServoHardwareSession
from controller.servo.magneticEncoderPos_control.controller import (
    MagneticEncoderPositionControlConfig,
    MagneticEncoderPositionFeedback,
    MagneticEncoderPositionPIDController,
    encoder_position_from_sample,
)
from controller.servo.torque_calibration.storage import (
    DEFAULT_ENCODER_POSITION_PATH,
    load_encoder_calibration_positions,
    save_encoder_calibration_positions,
)


RETURN_KP = 1.0
RETURN_MAX_SPEED = 100.0
RETURN_HARD_ERROR_DEG = 720.0
RETURN_TOLERANCE_DEG = 1.0
RETURN_SETTLE_S = 0.25
RETURN_TIMEOUT_S = 60.0
FEEDBACK_MAX_AGE_S = 0.25
CONTROL_HZ = 20.0


class RememberedPositionController:
    """Operate remembered positions while CSV playback is not running."""

    def __init__(
        self,
        *,
        hardware: ServoHardwareSession,
        servo_ids: tuple[int, ...],
        encoder_sensor_ids_by_servo_id: Mapping[int, int],
        encoder_directions_by_servo_id: Mapping[int, float],
        position_path: Path = DEFAULT_ENCODER_POSITION_PATH,
    ) -> None:
        self.hardware = hardware
        self.servo_ids = tuple(int(value) for value in servo_ids)
        self.encoder_sensor_ids_by_servo_id = {
            int(servo_id): int(sensor_id)
            for servo_id, sensor_id in encoder_sensor_ids_by_servo_id.items()
        }
        self.encoder_directions_by_servo_id = {
            int(servo_id): self._sign(direction, "encoder_direction")
            for servo_id, direction in encoder_directions_by_servo_id.items()
        }
        self.position_path = Path(position_path)

    @staticmethod
    def _sign(value: Any, name: str) -> float:
        number = float(value)
        if number not in (-1.0, 1.0):
            raise ValueError(f"{name} must be either -1 or 1")
        return number

    def positions(self) -> dict[int, dict[str, Any]]:
        return load_encoder_calibration_positions(self.position_path)

    def remember(self, servo_id: int) -> dict[str, Any]:
        servo_id = self._validate_servo_id(servo_id)
        feedback = self._fresh_feedback()
        position_deg, _ = self._position_feedback(servo_id, feedback)
        sensor_id = self.encoder_sensor_ids_by_servo_id[servo_id]
        encoder_direction = self.encoder_directions_by_servo_id[servo_id]
        record = {
            "servo_id": servo_id,
            "encoder_sensor_id": sensor_id,
            "encoder_position_deg": position_deg,
            "encoder_direction": encoder_direction,
            "control_sign": 1.0,
            "recorded_at_unix_s": time.time(),
        }
        positions = self.positions()
        positions[servo_id] = record
        save_encoder_calibration_positions(self.position_path, positions)
        return dict(record)

    def return_one(
        self,
        servo_id: int,
        *,
        cancel_event: threading.Event | None = None,
    ) -> dict[str, Any]:
        servo_id = self._validate_servo_id(servo_id)
        record = self._validated_record(servo_id, self.positions())
        event = cancel_event or threading.Event()
        controller = self._controller_for_record(servo_id, record)
        deadline = time.monotonic() + RETURN_TIMEOUT_S
        reached_since: float | None = None
        period_s = 1.0 / CONTROL_HZ
        next_tick = time.monotonic()
        try:
            while time.monotonic() < deadline:
                if event.is_set():
                    return {**record, "cancelled": True}
                self.hardware.check_health()
                now = time.monotonic()
                feedback = self._fresh_feedback()
                position_deg, position_feedback = self._position_feedback(
                    servo_id, feedback
                )
                output = controller.step(position_feedback, now=now)
                within_tolerance = abs(output.position_error_deg) <= RETURN_TOLERANCE_DEG
                speed = 0 if within_tolerance else output.speed_command
                if event.is_set():
                    return {**record, "cancelled": True}
                self.hardware.submit_speed_to_servos({servo_id: speed})
                if within_tolerance:
                    if reached_since is None:
                        reached_since = now
                    elif now - reached_since >= RETURN_SETTLE_S:
                        return {
                            **record,
                            "actual_position_deg": position_deg,
                            "position_error_deg": output.position_error_deg,
                        }
                else:
                    reached_since = None
                next_tick += period_s
                event.wait(max(0.0, next_tick - time.monotonic()))
            raise TimeoutError(
                f"Servo {servo_id} did not reach its remembered position within "
                f"{RETURN_TIMEOUT_S:g} s"
            )
        except Exception:
            self.hardware.stop_servos()
            raise
        finally:
            try:
                self.hardware.submit_speed_to_servos({servo_id: 0})
            except Exception:
                self.hardware.stop_servos()

    def return_all(
        self,
        *,
        cancel_event: threading.Event | None = None,
    ) -> dict[int, dict[str, Any]]:
        event = cancel_event or threading.Event()
        positions = self.positions()
        for servo_id in self.servo_ids:
            self._validated_record(servo_id, positions)
        results: dict[int, dict[str, Any]] = {}
        try:
            for servo_id in self.servo_ids:
                if event.is_set():
                    break
                result = self.return_one(servo_id, cancel_event=event)
                results[servo_id] = result
                if result.get("cancelled"):
                    break
            return results
        finally:
            if event.is_set() or len(results) != len(self.servo_ids):
                self.hardware.stop_servos()

    def _validate_servo_id(self, servo_id: int) -> int:
        servo_id = int(servo_id)
        if servo_id not in self.servo_ids:
            raise ValueError(f"Unknown servo ID: {servo_id}")
        if servo_id not in self.encoder_sensor_ids_by_servo_id:
            raise ValueError(f"Missing encoder sensor mapping for servo {servo_id}")
        if servo_id not in self.encoder_directions_by_servo_id:
            raise ValueError(f"Missing encoder direction for servo {servo_id}")
        return servo_id

    def _validated_record(
        self,
        servo_id: int,
        positions: Mapping[int, Mapping[str, Any]],
    ) -> dict[str, Any]:
        record = dict(positions.get(servo_id, {}))
        if not record:
            raise RuntimeError(f"No remembered position for servo {servo_id}")
        expected_sensor_id = self.encoder_sensor_ids_by_servo_id[servo_id]
        if int(record.get("encoder_sensor_id", expected_sensor_id)) != expected_sensor_id:
            raise RuntimeError(
                f"Remembered encoder channel does not match servo {servo_id}"
            )
        position_deg = float(record["encoder_position_deg"])
        if not math.isfinite(position_deg):
            raise ValueError(f"Remembered position is invalid for servo {servo_id}")
        self._sign(record.get("encoder_direction", 1.0), "encoder_direction")
        self._sign(record.get("control_sign", 1.0), "control_sign")
        return record

    def _controller_for_record(
        self,
        servo_id: int,
        record: Mapping[str, Any],
    ) -> MagneticEncoderPositionPIDController:
        saved_direction = self._sign(
            record.get("encoder_direction", 1.0), "encoder_direction"
        )
        active_direction = self.encoder_directions_by_servo_id[servo_id]
        target_position_deg = (
            float(record["encoder_position_deg"])
            * active_direction
            / saved_direction
        )
        return MagneticEncoderPositionPIDController(
            PID(
                PIDConfig(
                    kp=RETURN_KP,
                    ki=0.0,
                    kd=0.0,
                    output_min=-RETURN_MAX_SPEED,
                    output_max=RETURN_MAX_SPEED,
                    integral_min=-RETURN_MAX_SPEED,
                    integral_max=RETURN_MAX_SPEED,
                )
            ),
            MagneticEncoderPositionControlConfig(
                target_position_deg=target_position_deg,
                max_speed=RETURN_MAX_SPEED,
                hard_position_error_deg=RETURN_HARD_ERROR_DEG,
                max_feedback_age_s=FEEDBACK_MAX_AGE_S,
                control_sign=self._sign(
                    record.get("control_sign", 1.0), "control_sign"
                ),
                allow_multiturn_uncertain=True,
            ),
        )

    def _fresh_feedback(self) -> HardwareFeedback:
        feedback = self.hardware.latest_feedback(FEEDBACK_MAX_AGE_S)
        if feedback is None:
            raise RuntimeError("Fresh servo feedback is unavailable")
        return feedback

    def _position_feedback(
        self,
        servo_id: int,
        feedback: HardwareFeedback,
    ) -> tuple[float, MagneticEncoderPositionFeedback]:
        sensor_id = self.encoder_sensor_ids_by_servo_id[servo_id]
        sample = feedback.encoder_by_sensor.get(sensor_id)
        encoder_age_s = feedback.encoder_age_s(sensor_id)
        if sample is None or encoder_age_s is None:
            raise RuntimeError(f"Encoder feedback is unavailable for servo {servo_id}")
        if encoder_age_s > FEEDBACK_MAX_AGE_S:
            raise RuntimeError(f"Encoder feedback is stale for servo {servo_id}")
        if self.hardware.system is None:
            raise RuntimeError("Hardware session is not connected")
        servo_index = self.hardware.system.index_for_servo_id(servo_id)
        valid = feedback.servo.get("valid", ())
        if (
            not isinstance(valid, (tuple, list))
            or servo_index >= len(valid)
            or not valid[servo_index]
        ):
            raise RuntimeError(f"Servo telemetry is invalid for servo {servo_id}")
        position_deg = encoder_position_from_sample(
            sample,
            encoder_direction=self.encoder_directions_by_servo_id[servo_id],
            allow_multiturn_uncertain=True,
        )
        status = str(sample.get("status", "")).strip().upper()
        position_feedback = MagneticEncoderPositionFeedback(
            position_deg=position_deg,
            servo_age_s=feedback.servo_age_s,
            encoder_age_s=encoder_age_s,
            status=status,
        )
        return position_deg, position_feedback
