"""Verify host-side accumulated position control on one servo.

Set EXECUTE=True only with the configured servo connected and an emergency-stop
path available.  The controller treats the position observed at startup as
accumulated position zero.
"""

from __future__ import annotations

import sys
from pathlib import Path


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from servo.control import (
    AccumulatedPositionControlConfig,
    PositionControlConfig,
    PositionProgress,
    ReliablePositionController,
    TelemetryMonitor,
)


EXECUTE = False
SERIAL_PORT = "COM3"
BAUD_RATE = 921600
SERIAL_TIMEOUT_S = 0.2
SERVO_ID = 5

# Test a small move first.  Add a value beyond +/-4096 only after confirming
# direction, stop behavior, and tendon safety with the mechanism unloaded.
TARGET_POSITIONS = (200, 0)


def print_progress(progress: PositionProgress) -> None:
    print(
        f"Telemetry: phase={progress.phase}, servo={progress.servo_id}, "
        f"target={progress.target_position}, actual={progress.actual_position}, "
        f"speed={progress.speed}, load={progress.load}"
    )


def main() -> None:
    print(
        f"Plan: servo={SERVO_ID}, accumulated reference=0, "
        f"targets={TARGET_POSITIONS}"
    )
    if not EXECUTE:
        print("DRY RUN: set EXECUTE=True to connect to hardware")
        return

    from servo.servo_APIs import ServoAPI

    with ServoAPI(
        port=SERIAL_PORT,
        baud_rate=BAUD_RATE,
        timeout=SERIAL_TIMEOUT_S,
    ) as api:
        telemetry = TelemetryMonitor(
            api,
            num_servos=6,
            read_timeout_s=SERIAL_TIMEOUT_S,
        )
        controller = ReliablePositionController(
            api,
            telemetry,
            PositionControlConfig(
                telemetry_wait_s=3.0,
                arrival_timeout_s=3.0,
                reset_id_map_on_prepare=True,
            ),
            AccumulatedPositionControlConfig(
                switch_to_position_threshold=150,
                wheel_min_speed=30,
                wheel_max_speed=300,
                wheel_command_lifetime_ms=100,
                wheel_telemetry_timeout_s=0.15,
                wheel_arrival_timeout_s=15.0,
            ),
        )
        telemetry.start()
        try:
            controller.prepare((SERVO_ID,), force_init_servo_ids=tuple(range(6)))
            controller.set_accumulated_reference(SERVO_ID, 0)
            for target in TARGET_POSITIONS:
                result = controller.move_accumulated_and_wait(
                    SERVO_ID,
                    target,
                    progress_callback=print_progress,
                )
                print(
                    f"Stopped: accumulated={result.final_position}, "
                    f"raw={result.final_raw_position}, "
                    f"target={result.target_position}, "
                    f"wheel_commands={result.wheel_commands}"
                )
        except KeyboardInterrupt:
            print("Interrupted; stopping all servos")
            raise
        finally:
            controller.stop_all()
            telemetry.stop()


if __name__ == "__main__":
    main()

