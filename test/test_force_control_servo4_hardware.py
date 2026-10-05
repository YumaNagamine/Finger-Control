"""Guarded hardware integration test for servo ID 4 (FDP).

This test is skipped during normal test discovery.  It only connects to the
hardware when both of these environment variables are set:

    RUN_SERVO_HARDWARE_TESTS=1
    SERVO4_FORCE_TEST_CONFIRM=SERVO4_FORCE

The servo and encoder ports, PID proportional gain, and hard force limit must
also be supplied through ``SERVO_PORT``, ``ENCODER_PORT``, ``KP``, and
``HARD_FORCE_LIMIT_N``.  The default target is the configured minimum of 0.5 N.

Example from an activated ``FingerControl`` environment::

    set RUN_SERVO_HARDWARE_TESTS=1
    set SERVO4_FORCE_TEST_CONFIRM=SERVO4_FORCE
    set SERVO_PORT=COM6
    set ENCODER_PORT=COM5
    set KP=5
    set HARD_FORCE_LIMIT_N=2
    python -m unittest discover -s test -p test_force_control_servo4_hardware.py

Start with the tendon unloaded, use an emergency stop, and verify the control
sign before running this test on the mechanism.
"""

from __future__ import annotations

import os
import unittest
from argparse import Namespace
from pathlib import Path

from controller.servo.run_force_pid import (
    API_ROOT,
    DEFAULT_FORCE_CONFIG_PATH,
    _load_force_control_config,
    _validate_args,
    run,
)


SERVO_ID = 4
DEFAULT_CALIBRATION_FILE = (
    API_ROOT / "calibration" / "current" / "spring_torque_calibration_servo_id_4.csv"
)
DEFAULT_ZERO_OFFSET_FILE = (
    API_ROOT / "calibration" / "current" / "as5048a_unloaded_zero_offsets.json"
)


def _required_env(name: str) -> str:
    value = os.environ.get(name)
    if not value:
        raise RuntimeError(f"{name} must be set for the servo 4 hardware test")
    return value


def _float_env(name: str, default: float | None = None) -> float:
    value = os.environ.get(name)
    if value is None:
        if default is None:
            raise RuntimeError(f"{name} must be set for the servo 4 hardware test")
        return default
    return float(value)


def _bool_env(name: str) -> bool:
    return os.environ.get(name, "").strip().lower() in {"1", "true", "yes"}


@unittest.skipUnless(
    os.environ.get("RUN_SERVO_HARDWARE_TESTS") == "1",
    "set RUN_SERVO_HARDWARE_TESTS=1 to enable hardware tests",
)
class Servo4ForceHardwareTest(unittest.TestCase):
    def test_servo_4_fdp_force_pid(self):
        self.assertEqual(
            os.environ.get("SERVO4_FORCE_TEST_CONFIRM"),
            "SERVO4_FORCE",
            "set SERVO4_FORCE_TEST_CONFIRM=SERVO4_FORCE to confirm hardware motion",
        )

        force_config = Path(
            os.environ.get("FORCE_CONFIG_FILE", str(DEFAULT_FORCE_CONFIG_PATH))
        )
        args = Namespace(
            servo_port=_required_env("SERVO_PORT"),
            encoder_port=_required_env("ENCODER_PORT"),
            servo_id=SERVO_ID,
            calibration_file=Path(
                os.environ.get(
                    "SERVO4_CALIBRATION_FILE",
                    str(DEFAULT_CALIBRATION_FILE),
                )
            ),
            zero_offset_file=Path(
                os.environ.get(
                    "SERVO4_ZERO_OFFSET_FILE",
                    str(DEFAULT_ZERO_OFFSET_FILE),
                )
            ),
            no_zero_offset=_bool_env("NO_ZERO_OFFSET"),
            target_force_n=_float_env("TARGET_FORCE_N", 0.5),
            kp=_float_env("KP"),
            ki=_float_env("KI", 0.0),
            kd=_float_env("KD", 0.0),
            max_speed=_float_env("MAX_SPEED", 20.0),
            hard_force_limit_n=_float_env("HARD_FORCE_LIMIT_N"),
            force_config=force_config,
            control_sign=_float_env("CONTROL_SIGN", 1.0),
            min_position=None,
            max_position=None,
            control_hz=_float_env("CONTROL_HZ", 20.0),
            encoder_rate_hz=_float_env("ENCODER_RATE_HZ", 70.0),
            feedback_timeout_s=_float_env("FEEDBACK_TIMEOUT_S", 0.2),
            startup_timeout_s=_float_env("STARTUP_TIMEOUT_S", 3.0),
            seconds=_float_env("SECONDS", 5.0),
            print_period_s=_float_env("PRINT_PERIOD_S", 0.2),
        )
        args.min_target_force_n = _load_force_control_config(args.force_config)
        _validate_args(args)

        self.assertEqual(run(args), 0)


if __name__ == "__main__":
    unittest.main()
