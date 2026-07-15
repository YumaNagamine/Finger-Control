"""Move one tendon by a fixed signed excursion from the terminal.

This is an interactive terminal tool for the physical servo system.  Hardware
mode currently supports Windows.  Dry-run mode supports Windows and macOS and
prints the serial command without opening a port.  In hardware mode, each
movement starts from the latest measured absolute servo position.  Press Esc
at any time to stop and exit.  Enter moves in the configured direction;
Backspace moves by the same distance in the opposite direction.

The signed position-units-per-mm calibration is shared with the excursion CSV
players.  Its sign defines which servo-position direction corresponds to a
positive tendon excursion.
"""

from __future__ import annotations

import math
import sys
import time
from contextlib import contextmanager
from pathlib import Path
from typing import Callable, Iterator


SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from controller.servo_mapping import TENDONS, TENDON_TO_SERVO_ID
from controller.csv_player.excursion_player import (
    SERVO_POSITION_MAX,
    SERVO_POSITION_MIN,
    load_position_calibration,
)


# ---------------------------------------------------------------------------
# User settings
# GUI-compatible mapping: FDP->5, FDS->4, EI(ED)->2, DI->3, PI->1, LUM(LU)->0.
# Change TENDON only; SERVO_ID is derived from the shared mapping.
# ---------------------------------------------------------------------------
TENDON = "FDP"
SERVO_ID = TENDON_TO_SERVO_ID.get(TENDON, -1)
EXCURSION_MM = +5.0
DRY_RUN = False
SIMULATION_START_POSITION = 2048

CALIBRATION_PATH = SRC_ROOT / "controller" / "excursion_servo_calibration.json"

MOVE_TIME_MS = 1000
POSITION_TOLERANCE = 10
SPEED_TOLERANCE = 5
STABLE_FRAME_COUNT = 3
ARRIVAL_TIMEOUT_S = 3.0
TELEMETRY_WAIT_S = 3.0
TELEMETRY_DISPLAY_INTERVAL_S = 0.1

SERIAL_PORT = "COM3"
BAUD_RATE = 921600
SERIAL_TIMEOUT_S = 0.05

ENTER_KEYS = {"\r", "\n"}
BACKSPACE_KEYS = {"\x08", "\x7f"}
ESC_KEY = "\x1b"


class ExitRequested(Exception):
    """Raised when Esc requests an orderly emergency stop."""


def validate_settings() -> None:
    if TENDON not in TENDONS:
        raise ValueError(f"TENDON must be one of {TENDONS}, got {TENDON!r}")
    if not 0 <= SERVO_ID < len(TENDONS):
        raise ValueError(f"SERVO_ID must be 0-{len(TENDONS) - 1}, got {SERVO_ID}")
    expected_servo_id = TENDON_TO_SERVO_ID[TENDON]
    if SERVO_ID != expected_servo_id:
        raise ValueError(
            f"{TENDON} is mapped to servo {expected_servo_id}, not servo {SERVO_ID}"
        )
    if not math.isfinite(EXCURSION_MM) or EXCURSION_MM == 0.0:
        raise ValueError("EXCURSION_MM must be finite and non-zero")
    if not SERVO_POSITION_MIN <= SIMULATION_START_POSITION <= SERVO_POSITION_MAX:
        raise ValueError(
            f"SIMULATION_START_POSITION must be in the range "
            f"{SERVO_POSITION_MIN}-{SERVO_POSITION_MAX}"
        )
    if not 1 <= MOVE_TIME_MS <= 65535:
        raise ValueError("MOVE_TIME_MS must be in the range 1-65535")
    if POSITION_TOLERANCE < 0:
        raise ValueError("POSITION_TOLERANCE must be non-negative")
    if SPEED_TOLERANCE < 0:
        raise ValueError("SPEED_TOLERANCE must be non-negative")
    if STABLE_FRAME_COUNT <= 0:
        raise ValueError("STABLE_FRAME_COUNT must be greater than zero")
    if not math.isfinite(ARRIVAL_TIMEOUT_S) or ARRIVAL_TIMEOUT_S <= 0.0:
        raise ValueError("ARRIVAL_TIMEOUT_S must be finite and greater than zero")
    if ARRIVAL_TIMEOUT_S < MOVE_TIME_MS / 1000.0:
        raise ValueError("ARRIVAL_TIMEOUT_S must cover MOVE_TIME_MS")
    if not math.isfinite(TELEMETRY_WAIT_S) or TELEMETRY_WAIT_S <= 0.0:
        raise ValueError("TELEMETRY_WAIT_S must be finite and greater than zero")
    if (
        not math.isfinite(TELEMETRY_DISPLAY_INTERVAL_S)
        or TELEMETRY_DISPLAY_INTERVAL_S <= 0.0
    ):
        raise ValueError(
            "TELEMETRY_DISPLAY_INTERVAL_S must be finite and greater than zero"
        )
    if BAUD_RATE <= 0:
        raise ValueError("BAUD_RATE must be greater than zero")
    if SERIAL_TIMEOUT_S <= 0.0:
        raise ValueError("SERIAL_TIMEOUT_S must be greater than zero")


def calculate_target_position(
    actual_position: int,
    excursion_mm: float,
    position_units_per_mm: float,
) -> tuple[int, int]:
    position_delta = round(excursion_mm * position_units_per_mm)
    target_position = actual_position + position_delta
    if not SERVO_POSITION_MIN <= target_position <= SERVO_POSITION_MAX:
        raise ValueError(
            f"Target position {target_position} is outside "
            f"{SERVO_POSITION_MIN}-{SERVO_POSITION_MAX}. Current position is "
            f"{actual_position} and requested delta is {position_delta:+d} units."
        )
    return target_position, position_delta


def excursion_for_key(key: str | None) -> float | None:
    if key in ENTER_KEYS:
        return EXCURSION_MM
    if key in BACKSPACE_KEYS:
        return -EXCURSION_MM
    return None


def load_windows_key_poller() -> Callable[[], str | None]:
    import msvcrt

    def poll_key() -> str | None:
        if not msvcrt.kbhit():
            return None
        return msvcrt.getwch()

    return poll_key


@contextmanager
def open_key_poller(*, dry_run: bool) -> Iterator[Callable[[], str | None]]:
    if sys.platform == "win32":
        yield load_windows_key_poller()
        return

    if sys.platform == "darwin" and dry_run:
        import select
        import termios
        import tty

        if not sys.stdin.isatty():
            raise RuntimeError("macOS dry-run mode must be run in an interactive terminal")

        stdin_fd = sys.stdin.fileno()
        original_settings = termios.tcgetattr(stdin_fd)

        def poll_key() -> str | None:
            readable, _, _ = select.select([sys.stdin], [], [], 0.0)
            if not readable:
                return None
            return sys.stdin.read(1)

        try:
            tty.setcbreak(stdin_fd)
            yield poll_key
        finally:
            termios.tcsetattr(stdin_fd, termios.TCSADRAIN, original_settings)
        return

    if sys.platform == "darwin":
        raise RuntimeError(
            "Hardware mode currently supports only Windows. Set DRY_RUN = True "
            "when running on macOS."
        )

    raise RuntimeError(
        "Interactive key input currently supports Windows, plus macOS in dry-run mode"
    )


def read_fresh_telemetry(api, timeout_s: float):
    deadline = time.monotonic() + timeout_s
    while time.monotonic() < deadline:
        frame = api.try_read_telemetry()
        if frame is not None and len(frame.positions) == len(TENDONS):
            return frame
    raise TimeoutError(f"No valid six-servo telemetry received within {timeout_s:.1f} s")


def wait_until_stopped(
    api,
    target_position: int,
    poll_key: Callable[[], str | None],
):
    deadline = time.monotonic() + ARRIVAL_TIMEOUT_S
    stable_frames = 0
    next_display_at = 0.0

    while time.monotonic() < deadline:
        if poll_key() == ESC_KEY:
            raise ExitRequested

        frame = api.try_read_telemetry()
        if frame is None:
            continue
        if len(frame.positions) != len(TENDONS):
            continue
        if frame.speeds is None or len(frame.speeds) != len(TENDONS):
            raise RuntimeError(
                "Servo-stop detection requires the 19-field telemetry format "
                "with speed values from the current V3 firmware"
            )

        actual_position = frame.positions[SERVO_ID]
        actual_load = frame.loads[SERVO_ID]
        actual_speed = frame.speeds[SERVO_ID]
        position_reached = (
            abs(target_position - actual_position) <= POSITION_TOLERANCE
        )
        speed_settled = abs(actual_speed) <= SPEED_TOLERANCE

        if position_reached and speed_settled:
            stable_frames += 1
            if stable_frames >= STABLE_FRAME_COUNT:
                return frame
        else:
            stable_frames = 0

        now = time.monotonic()
        if now >= next_display_at:
            print(
                f"Telemetry: servo={SERVO_ID}, target={target_position}, "
                f"actual={actual_position}, "
                f"error={target_position - actual_position:+d}, "
                f"speed={actual_speed}, load={actual_load}, "
                f"stable={stable_frames}/{STABLE_FRAME_COUNT}",
                flush=True,
            )
            next_display_at = now + TELEMETRY_DISPLAY_INTERVAL_S

    raise TimeoutError(
        f"Servo {SERVO_ID} did not stop at target {target_position} within "
        f"{ARRIVAL_TIMEOUT_S:.1f} s"
    )


def run_dry_run(units_per_mm: float, poll_key: Callable[[], str | None]) -> None:
    simulated_position = SIMULATION_START_POSITION
    position_delta = round(EXCURSION_MM * units_per_mm)

    print(
        f"DRY RUN: {TENDON} (servo {SERVO_ID}), "
        f"excursion={EXCURSION_MM:+.3f} mm, delta={position_delta:+d} units, "
        f"simulated_absolute_position={simulated_position}"
    )
    print("No serial port will be opened and no command will be sent.")
    print(
        "Press Enter for the configured direction, Backspace for the opposite "
        "direction, or Esc to exit."
    )

    while True:
        key = poll_key()
        if key == ESC_KEY:
            print("DRY RUN: Esc pressed; exiting without sending a command.")
            return
        requested_excursion_mm = excursion_for_key(key)
        if requested_excursion_mm is None:
            time.sleep(0.01)
            continue

        target_position, actual_delta = calculate_target_position(
            simulated_position,
            requested_excursion_mm,
            units_per_mm,
        )
        command = f"x,{SERVO_ID},{target_position},{MOVE_TIME_MS}"
        print(
            f"DRY RUN command: {command}  "
            f"# {simulated_position} -> {target_position} "
            f"({actual_delta:+d} units)"
        )
        simulated_position = target_position


def run_manual_control(api, units_per_mm: float, poll_key: Callable[[], str | None]) -> None:
    initial_frame = read_fresh_telemetry(api, TELEMETRY_WAIT_S)
    initial_position = initial_frame.positions[SERVO_ID]
    position_delta = round(EXCURSION_MM * units_per_mm)

    print(
        f"Ready: {TENDON} (servo {SERVO_ID}), excursion={EXCURSION_MM:+.3f} mm, "
        f"delta={position_delta:+d} units, absolute_position={initial_position}"
    )
    print(
        "Press Enter for the configured direction, Backspace for the opposite "
        "direction, or Esc to stop all servos and exit."
    )

    latest_frame = initial_frame
    while True:
        frame = api.try_read_telemetry()
        if frame is not None and len(frame.positions) == len(TENDONS):
            latest_frame = frame

        key = poll_key()
        if key == ESC_KEY:
            raise ExitRequested
        requested_excursion_mm = excursion_for_key(key)
        if requested_excursion_mm is None:
            continue

        before_position = latest_frame.positions[SERVO_ID]
        target_position, actual_delta = calculate_target_position(
            before_position,
            requested_excursion_mm,
            units_per_mm,
        )
        print(
            f"Moving: servo={SERVO_ID}, {before_position} -> {target_position} "
            f"({actual_delta:+d} units)"
        )
        api.set_position(SERVO_ID, target_position, time_ms=MOVE_TIME_MS)

        final_frame = wait_until_stopped(api, target_position, poll_key)
        latest_frame = final_frame
        final_position = final_frame.positions[SERVO_ID]
        final_speed = final_frame.speeds[SERVO_ID]
        print(
            f"Stopped: servo={SERVO_ID}, absolute_position={final_position}, "
            f"target={target_position}, error={target_position - final_position:+d}, "
            f"speed={final_speed}"
        )
        print(
            "Press Enter for the configured direction, Backspace for the opposite "
            "direction, or Esc to stop all servos and exit."
        )


def main() -> None:
    validate_settings()
    calibration = load_position_calibration(CALIBRATION_PATH)
    units_per_mm = calibration[TENDONS.index(TENDON)]

    with open_key_poller(dry_run=DRY_RUN) as poll_key:
        if DRY_RUN:
            run_dry_run(units_per_mm, poll_key)
            return

        from servo.servo_APIs import ServoAPI

        with ServoAPI(
            port=SERIAL_PORT,
            baud_rate=BAUD_RATE,
            timeout=SERIAL_TIMEOUT_S,
        ) as api:
            try:
                run_manual_control(api, units_per_mm, poll_key)
            except ExitRequested:
                print("Esc pressed; stopping all servos.")
            except KeyboardInterrupt:
                print("\nCtrl-C received; stopping all servos.")
            finally:
                api.stop_all()


if __name__ == "__main__":
    main()
