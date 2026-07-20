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
from servo.control import (
    PositionControlCancelledError,
    PositionControlConfig,
    PositionProgress,
    ReliablePositionController,
    RetryPolicy,
    TelemetryMonitor,
)


# ---------------------------------------------------------------------------
# User settings
# GUI-compatible mapping: FDP->5, FDS->4, EI(ED)->2, DI->3, PI->1, LUM(LU)->0.
# Change TENDON only; SERVO_ID is derived from the shared mapping.
# ---------------------------------------------------------------------------
TENDON = "LUM"
SERVO_ID = TENDON_TO_SERVO_ID.get(TENDON, -1)
EXCURSION_MM = +5.0
DRY_RUN = False
SIMULATION_START_POSITION = 2048

CALIBRATION_PATH = SRC_ROOT / "controller" / "excursion_servo_calibration.json"

MOVE_TIME_MS = 0
POSITION_TOLERANCE = 10
SPEED_TOLERANCE = 5
STABLE_FRAME_COUNT = 3
ARRIVAL_TIMEOUT_S = 3.0
TELEMETRY_WAIT_S = 3.0
TELEMETRY_DISPLAY_INTERVAL_S = 0.1
ID_MAP_RESET_WAIT_S = 0.1
POSITION_MODE_PREPARE_WAIT_S = 0.2
POSITION_MODE_PRIME_WAIT_S = 0.2
POSITION_MODE_PRIME_COMMAND_COUNT = 2
START_OBSERVATION_S = 0.3
START_MIN_DELTA = 3
MAX_START_RETRIES = 1

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
    if not 0 <= MOVE_TIME_MS <= 65535:
        raise ValueError("MOVE_TIME_MS must be in the range 0-65535")
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
    if (
        not math.isfinite(ID_MAP_RESET_WAIT_S)
        or ID_MAP_RESET_WAIT_S <= 0.0
    ):
        raise ValueError(
            "ID_MAP_RESET_WAIT_S must be finite and greater than zero"
        )
    if (
        not math.isfinite(POSITION_MODE_PREPARE_WAIT_S)
        or POSITION_MODE_PREPARE_WAIT_S <= 0.0
    ):
        raise ValueError(
            "POSITION_MODE_PREPARE_WAIT_S must be finite and greater than zero"
        )
    if (
        not math.isfinite(POSITION_MODE_PRIME_WAIT_S)
        or POSITION_MODE_PRIME_WAIT_S < 0.0
    ):
        raise ValueError(
            "POSITION_MODE_PRIME_WAIT_S must be finite and non-negative"
        )
    if POSITION_MODE_PRIME_COMMAND_COUNT <= 0:
        raise ValueError("POSITION_MODE_PRIME_COMMAND_COUNT must be greater than zero")
    if not math.isfinite(START_OBSERVATION_S) or START_OBSERVATION_S <= 0.0:
        raise ValueError("START_OBSERVATION_S must be finite and greater than zero")
    if START_MIN_DELTA < 0:
        raise ValueError("START_MIN_DELTA must be non-negative")
    if MAX_START_RETRIES < 0:
        raise ValueError("MAX_START_RETRIES must be non-negative")

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


def build_position_control_config() -> PositionControlConfig:
    return PositionControlConfig(
        telemetry_stale_s=max(0.5, SERIAL_TIMEOUT_S * 2.0),
        telemetry_wait_s=TELEMETRY_WAIT_S,
        id_map_reset_wait_s=ID_MAP_RESET_WAIT_S,
        speed_init_wait_s=POSITION_MODE_PREPARE_WAIT_S,
        prime_command_count=POSITION_MODE_PRIME_COMMAND_COUNT,
        prime_interval_s=POSITION_MODE_PRIME_WAIT_S,
        start_observation_s=START_OBSERVATION_S,
        start_min_delta=START_MIN_DELTA,
        position_tolerance=POSITION_TOLERANCE,
        speed_tolerance=SPEED_TOLERANCE,
        stable_frame_count=STABLE_FRAME_COUNT,
        arrival_timeout_s=ARRIVAL_TIMEOUT_S,
        max_start_retries=MAX_START_RETRIES,
        reset_id_map_on_prepare=True,
        position_min=SERVO_POSITION_MIN,
        position_max=SERVO_POSITION_MAX,
    )


def prepare_position_control(controller: ReliablePositionController) -> None:
    print(
        "Preparing position control: "
        f"reset_ids=True, speed_init_servos=0-{len(TENDONS) - 1}, "
        f"position_servo={SERVO_ID}, "
        f"prime_commands={POSITION_MODE_PRIME_COMMAND_COUNT}"
    )
    controller.prepare(
        (SERVO_ID,),
        force_init_servo_ids=tuple(range(len(TENDONS))),
    )
    print(
        "Position mode primed: "
        f"servo={SERVO_ID}, actual_position={controller.current_position(SERVO_ID)}"
    )


def run_manual_control(
    controller: ReliablePositionController,
    units_per_mm: float,
    poll_key: Callable[[], str | None],
) -> None:
    initial_position = controller.current_position(SERVO_ID)
    position_delta = round(EXCURSION_MM * units_per_mm)

    print(
        f"Ready: {TENDON} (servo {SERVO_ID}), excursion={EXCURSION_MM:+.3f} mm, "
        f"delta={position_delta:+d} units, absolute_position={initial_position}"
    )
    print(
        "Press Enter for the configured direction, Backspace for the opposite "
        "direction, or Esc to stop all servos and exit."
    )

    while True:
        key = poll_key()
        if key == ESC_KEY:
            raise ExitRequested
        requested_excursion_mm = excursion_for_key(key)
        if requested_excursion_mm is None:
            continue

        before_position = controller.current_position(SERVO_ID)
        target_position, actual_delta = calculate_target_position(
            before_position,
            requested_excursion_mm,
            units_per_mm,
        )
        print(
            f"Moving: servo={SERVO_ID}, {before_position} -> {target_position} "
            f"({actual_delta:+d} units)"
        )
        next_display_at = 0.0

        def display_progress(progress: PositionProgress) -> None:
            nonlocal next_display_at
            now = time.monotonic()
            if now < next_display_at:
                return
            print(
                f"Telemetry: phase={progress.phase}, attempt={progress.attempt}, "
                f"servo={progress.servo_id}, target={progress.target_position}, "
                f"actual={progress.actual_position}, "
                f"error={progress.target_position - progress.actual_position:+d}, "
                f"speed={progress.speed}, load={progress.load}, "
                f"stable={progress.stable_frames}/{STABLE_FRAME_COUNT}",
                flush=True,
            )
            next_display_at = now + TELEMETRY_DISPLAY_INTERVAL_S

        try:
            result = controller.move_and_wait(
                SERVO_ID,
                target_position,
                time_ms=MOVE_TIME_MS,
                retry_policy=RetryPolicy.ON_NO_START,
                progress_callback=display_progress,
                cancel_check=lambda: poll_key() == ESC_KEY,
            )
        except PositionControlCancelledError as exc:
            raise ExitRequested from exc

        print(
            f"Stopped: servo={SERVO_ID}, absolute_position={result.final_position}, "
            f"target={target_position}, "
            f"error={target_position - result.final_position:+d}, "
            f"speed={result.final_speed}, retries={result.retries}"
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
            telemetry = TelemetryMonitor(
                api,
                num_servos=len(TENDONS),
                read_timeout_s=SERIAL_TIMEOUT_S,
            )
            controller = ReliablePositionController(
                api,
                telemetry,
                build_position_control_config(),
            )
            telemetry.start()
            try:
                prepare_position_control(controller)
                run_manual_control(controller, units_per_mm, poll_key)
            except ExitRequested:
                print("Esc pressed; stopping all servos.")
            except KeyboardInterrupt:
                print("\nCtrl-C received; stopping all servos.")
            finally:
                try:
                    controller.stop_all()
                finally:
                    telemetry.stop()


if __name__ == "__main__":
    main()
