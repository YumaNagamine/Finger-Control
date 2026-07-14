"""Record one USB camera while playing a tendon-excursion CSV on real servos.

Set ``EXECUTE = True`` to open the camera and serial port.  With ``False``, the
script only prints the servo commands that would be sent.
"""

from __future__ import annotations

import csv
import datetime
import json
import math
import sys
import threading
import time
from pathlib import Path

import cv2


SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from controller.csv_player.excursion_player import (
    ExcursionPlayer,
    PlaybackStatus,
    TENDONS,
    load_position_calibration,
)
from observation.camera.camera_param_resolver import resolve_param_path
from observation.camera.camera_utils import (
    apply_camera_settings,
    fourcc_from_str,
    resolve_backend,
    setup_undistortion_from_config,
    undistort_frame,
)
from utils.config_loader import load_config


# ---------------------------------------------------------------------------
# User settings
# All six-value tuples use this order: FDP, FDS, EI, DI, PI, LUM.
# ---------------------------------------------------------------------------
PROJECT_ROOT = SRC_ROOT.parent
PREDICTION_CSV_PATH = (
    PROJECT_ROOT
    / "logs"
    / "dual_camera"
    / "excursion_predictions"
    / "dual_processed_controlTest_20260708_111151_prediction_20260708.csv"
)

CALIBRATION_PATH = SRC_ROOT / "controller" / "excursion_servo_calibration.json"
SERVO_IDS = (0, 1, 2, 3, 4, 5)
START_POSITIONS: tuple[int, ...] | None = (2048, 2048, 2048, 2048, 2048, 2048)

TIME_SCALE = 1.0
EXECUTE = False
SERIAL_PORT = "COM7"
BAUD_RATE = 921600
SERIAL_TIMEOUT_S = 0.2
TELEMETRY_WAIT_S = 3.0
MAX_LAG_S = 0.5
LIVE_DISPLAY_INTERVAL_S = 0.1
TELEMETRY_STALE_S = 0.5

CAMERA_CONFIG_PATH = resolve_param_path("camera_config_dlc.json")
CAMERA_WARMUP_FRAMES = 30
SHOW_CAMERA_PREVIEW = False
CAMERA_POST_ROLL_S = 0.5

OUTPUT_ROOT = PROJECT_ROOT / "logs" / "play_excursion_recording"
RETURN_TO_START_TIME_MS = 2000
RETURN_POSITION_TOLERANCE = 10
RETURN_TIMEOUT_S = 4.0


class CameraRecorder:
    def __init__(
        self,
        camera_config: dict,
        video_path: Path,
        timestamp_csv_path: Path,
        *,
        warmup_frames: int,
        show_preview: bool,
    ) -> None:
        self.camera_config = dict(camera_config)
        self.video_path = video_path
        self.timestamp_csv_path = timestamp_csv_path
        self.warmup_frames = warmup_frames
        self.show_preview = show_preview

        self.cap = None
        self.writer = None
        self.calibration = None
        self.timestamp_file = None
        self.timestamp_writer = None
        self.thread: threading.Thread | None = None
        self.stop_event = threading.Event()
        self.lock = threading.Lock()
        self.error: Exception | None = None
        self.started_at: float | None = None
        self.frame_count = 0
        self.actual_size: tuple[int, int] | None = None

    def prepare(self) -> None:
        camera_index = int(self.camera_config.get("index", 0))
        backend = resolve_backend(self.camera_config.get("backend"))
        self.cap = (
            cv2.VideoCapture(camera_index, backend)
            if backend is not None
            else cv2.VideoCapture(camera_index)
        )
        if not self.cap.isOpened():
            raise RuntimeError(f"Camera {camera_index} is not available")

        apply_camera_settings(self.cap, self.camera_config)
        self.calibration = setup_undistortion_from_config(
            self.camera_config,
            log_prefix="[play-excursion-recording]",
        )

        for _ in range(self.warmup_frames):
            self.cap.read()
        ok, initial_frame = self.cap.read()
        if not ok or initial_frame is None:
            raise RuntimeError("Failed to read an initial camera frame")
        initial_frame = undistort_frame(initial_frame, self.calibration)
        height, width = initial_frame.shape[:2]
        self.actual_size = (width, height)

        writer_fourcc = fourcc_from_str(self.camera_config.get("writer_fourcc", "mp4v"))
        if writer_fourcc is None:
            raise ValueError("Camera writer_fourcc must contain exactly four characters")
        target_fps = float(self.camera_config.get("target_fps", 90.0))
        if target_fps <= 0.0 or not math.isfinite(target_fps):
            raise ValueError("Camera target_fps must be finite and greater than zero")
        self.writer = cv2.VideoWriter(
            str(self.video_path),
            writer_fourcc,
            target_fps,
            (width, height),
        )
        if not self.writer.isOpened():
            raise RuntimeError(f"Failed to open video writer: {self.video_path}")

        self.timestamp_file = self.timestamp_csv_path.open(
            "w",
            encoding="utf-8",
            newline="",
        )
        self.timestamp_writer = csv.writer(self.timestamp_file)
        self.timestamp_writer.writerow(["frame_idx", "timestamp_iso", "elapsed_s"])

        if self.show_preview:
            cv2.namedWindow("Excursion Recording", cv2.WINDOW_NORMAL)
        print(
            f"Camera ready: index={camera_index}, actual={width}x{height}, "
            f"target_fps={target_fps:.1f}"
        )

    def start(self, started_at: float) -> None:
        if self.cap is None or self.writer is None or self.timestamp_writer is None:
            raise RuntimeError("Call prepare() before starting camera recording")
        if self.thread is not None:
            raise RuntimeError("Camera recorder has already been started")
        self.started_at = started_at
        self.thread = threading.Thread(
            target=self._record_loop,
            name="usb-camera-recorder",
            daemon=True,
        )
        self.thread.start()

    def raise_if_failed(self) -> None:
        with self.lock:
            error = self.error
        if error is not None:
            raise RuntimeError("Camera recording failed") from error

    def stop(self) -> None:
        self.stop_event.set()
        if self.thread is not None:
            self.thread.join(timeout=2.0)
            if self.thread.is_alive() and self.cap is not None:
                self.cap.release()
                self.thread.join(timeout=1.0)

    def close(self) -> None:
        self.stop()
        if self.timestamp_file is not None:
            self.timestamp_file.flush()
            self.timestamp_file.close()
            self.timestamp_file = None
        if self.writer is not None:
            self.writer.release()
            self.writer = None
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        if self.show_preview:
            cv2.destroyAllWindows()

    def _record_loop(self) -> None:
        try:
            while not self.stop_event.is_set():
                ok, frame = self.cap.read()
                captured_at = time.monotonic()
                if not ok or frame is None:
                    if self.stop_event.is_set():
                        break
                    raise RuntimeError("Camera returned no frame")

                frame = undistort_frame(frame, self.calibration)
                self.writer.write(frame)
                timestamp_iso = datetime.datetime.now().isoformat(timespec="milliseconds")
                self.timestamp_writer.writerow(
                    [
                        self.frame_count,
                        timestamp_iso,
                        f"{captured_at - self.started_at:.6f}",
                    ]
                )
                self.frame_count += 1
                if self.frame_count % 30 == 0:
                    self.timestamp_file.flush()

                if self.show_preview:
                    cv2.imshow("Excursion Recording", frame)
                    if cv2.waitKey(1) & 0xFF == 27:
                        raise RuntimeError("Camera preview was stopped with ESC")
        except Exception as exc:
            if not self.stop_event.is_set():
                with self.lock:
                    self.error = exc


class ServoTraceWriter:
    def __init__(self, csv_path: Path) -> None:
        self.file = csv_path.open("w", encoding="utf-8", newline="")
        self.writer = csv.writer(self.file)
        header = [
            "phase",
            "elapsed_s",
            "scheduled_s",
            "row_index",
            "move_time_ms",
            "telemetry_age_s",
        ]
        for tendon in TENDONS:
            header.extend(
                [
                    f"{tendon}_target_position",
                    f"{tendon}_actual_position",
                    f"{tendon}_position_error",
                ]
            )
        self.writer.writerow(header)

    def write(self, status: PlaybackStatus) -> None:
        row: list[object] = [
            status.phase,
            f"{status.elapsed_s:.6f}",
            "" if status.scheduled_s is None else f"{status.scheduled_s:.6f}",
            "" if status.row_index is None else status.row_index,
            status.move_time_ms,
            "" if status.telemetry_age_s is None else f"{status.telemetry_age_s:.6f}",
        ]
        if status.actual_positions is None:
            for target in status.target_positions:
                row.extend([target, "", ""])
        else:
            for target, actual in zip(status.target_positions, status.actual_positions):
                row.extend([target, actual, target - actual])
        self.writer.writerow(row)

    def close(self) -> None:
        self.file.flush()
        self.file.close()


def _build_player() -> ExcursionPlayer:
    position_units_per_mm = load_position_calibration(CALIBRATION_PATH)
    return ExcursionPlayer(
        servo_ids=SERVO_IDS,
        position_units_per_mm=position_units_per_mm,
        time_scale=TIME_SCALE,
        max_lag_s=MAX_LAG_S,
        live_display_interval_s=LIVE_DISPLAY_INTERVAL_S,
        telemetry_stale_s=TELEMETRY_STALE_S,
    )


def _wait_post_roll(camera_recorder: CameraRecorder, duration_s: float) -> None:
    deadline = time.monotonic() + duration_s
    while time.monotonic() < deadline:
        camera_recorder.raise_if_failed()
        time.sleep(min(0.05, max(0.0, deadline - time.monotonic())))


def main() -> None:
    player = _build_player()
    csv_path = PREDICTION_CSV_PATH.expanduser().resolve()

    if not EXECUTE:
        if START_POSITIONS is None:
            raise ValueError("Dry-run requires START_POSITIONS")
        frames = player.load_and_build(csv_path, START_POSITIONS)
        print("DRY RUN: camera and serial port will not be opened")
        player.print_summary(csv_path, frames, START_POSITIONS)
        player.print_simulation_commands(frames)
        return

    camera_config_path = CAMERA_CONFIG_PATH.expanduser().resolve()
    camera_config = load_config(str(camera_config_path))
    session_name = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    session_dir = OUTPUT_ROOT / session_name
    session_dir.mkdir(parents=True, exist_ok=False)

    video_path = session_dir / "recording.mp4"
    frame_csv_path = session_dir / "frame_timestamps.csv"
    servo_csv_path = session_dir / "servo_trace.csv"
    manifest_path = session_dir / "session_manifest.json"
    camera_recorder = CameraRecorder(
        camera_config,
        video_path,
        frame_csv_path,
        warmup_frames=CAMERA_WARMUP_FRAMES,
        show_preview=SHOW_CAMERA_PREVIEW,
    )
    trace_writer: ServoTraceWriter | None = None
    initial_positions: tuple[int, ...] | None = None
    experiment_started_at: float | None = None
    playback_completed = False
    returned_to_start = False
    error_text: str | None = None

    try:
        camera_recorder.prepare()
        from servo.servo_APIs import ServoAPI

        with ServoAPI(
            port=SERIAL_PORT,
            baud_rate=BAUD_RATE,
            timeout=SERIAL_TIMEOUT_S,
        ) as api:
            initial_positions = player.read_start_positions(api, TELEMETRY_WAIT_S)
            frames = player.load_and_build(csv_path, initial_positions)
            player.print_summary(csv_path, frames, initial_positions)
            telemetry_monitor = player.create_telemetry_monitor(api, SERIAL_TIMEOUT_S)
            telemetry_monitor.start()
            trace_writer = ServoTraceWriter(servo_csv_path)
            try:
                experiment_started_at = time.monotonic()
                camera_recorder.start(experiment_started_at)
                print("Recording camera and executing servo trajectory. Press Ctrl-C to stop.")
                player.play(
                    api,
                    frames,
                    telemetry_monitor,
                    started_at=experiment_started_at,
                    status_callback=trace_writer.write,
                    health_check=camera_recorder.raise_if_failed,
                )
                playback_completed = True

                print("Returning all servos to their measured initial positions.")
                player.return_to_start(
                    api,
                    initial_positions,
                    telemetry_monitor,
                    experiment_started_at=experiment_started_at,
                    move_time_ms=RETURN_TO_START_TIME_MS,
                    tolerance=RETURN_POSITION_TOLERANCE,
                    timeout_s=RETURN_TIMEOUT_S,
                    status_callback=trace_writer.write,
                    health_check=camera_recorder.raise_if_failed,
                )
                returned_to_start = True
                api.stop_all()
                _wait_post_roll(camera_recorder, CAMERA_POST_ROLL_S)
            finally:
                try:
                    api.stop_all()
                finally:
                    telemetry_monitor.stop()
    except BaseException as exc:
        error_text = f"{type(exc).__name__}: {exc}"
        raise
    finally:
        if trace_writer is not None:
            trace_writer.close()
        camera_recorder.close()
        manifest = {
            "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
            "session_dir": str(session_dir),
            "input_csv_path": str(csv_path),
            "servo_calibration_path": str(CALIBRATION_PATH.resolve()),
            "position_units_per_mm": {
                tendon: value
                for tendon, value in zip(TENDONS, player.position_units_per_mm)
            },
            "camera_config_path": str(camera_config_path),
            "camera_config": camera_config,
            "camera_actual_size": camera_recorder.actual_size,
            "saved_frames": camera_recorder.frame_count,
            "servo_ids": list(SERVO_IDS),
            "initial_positions": initial_positions,
            "experiment_started_at_monotonic": experiment_started_at,
            "playback_completed": playback_completed,
            "returned_to_start": returned_to_start,
            "error": error_text,
            "outputs": {
                "video": str(video_path),
                "frame_timestamps": str(frame_csv_path),
                "servo_trace": str(servo_csv_path),
            },
        }
        manifest_path.write_text(json.dumps(manifest, indent=2), encoding="utf-8")
        print(f"Saved recording session: {session_dir}")


if __name__ == "__main__":
    main()
