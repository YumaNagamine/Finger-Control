"""Backward-compatible exports for the shared servo PID implementation."""

from controller.servo.control_common.pid import PID, PIDConfig

__all__ = ["PID", "PIDConfig"]
