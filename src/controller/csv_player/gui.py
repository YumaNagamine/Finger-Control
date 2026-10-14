"""Tkinter GUI for CSV trajectory playback with DLC and force monitoring.

The GUI owns the presentation layer only.  Servo control, camera inference,
and serial readers run on worker threads so that plotting never blocks the
control loop.
"""

from __future__ import annotations

import csv
import json
import math
import os
import queue
import shutil
import threading
import time
import tkinter as tk
from collections import deque
from contextlib import ExitStack
from datetime import datetime
from pathlib import Path
from tkinter import filedialog, messagebox, ttk
from typing import Any, Callable

import cv2
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

try:
    from PIL import Image, ImageTk
except ImportError:  # pragma: no cover - Pillow is normally installed with matplotlib
    Image = None
    ImageTk = None

from controller.csv_player.encoder_calibration import (
    EncoderExcursionCalibration,
    load_encoder_excursion_calibration,
)
from controller.csv_player.gui_support import LatestEventMailbox
from controller.csv_player.playback_service import (
    EncoderPlaybackService,
    build_encoder_playback_config,
)
from controller.csv_player.runner import (
    ForceTelemetryUnavailable,
    HardForceLimitError,
    PlaybackStopped,
)
from controller.csv_player.trajectory import ExcursionTrajectory
from controller.servo.hardware_session import API_ROOT, ServoHardwareSession
from controller.servo_mapping import (
    ENCODER_SENSOR_IDS_BY_SERVO_ID,
    SERVO_IDS,
    SERVO_IDS_BY_TENDON,
    SERVO_NAMES_BY_ID,
    TENDONS,
)
from observation.camera.camera_utils import (
    apply_camera_settings,
    resolve_backend,
    setup_undistortion_from_config,
    undistort_frame,
)
from observation.camera.camera_param_resolver import resolve_param_path
from observation.vision.realtime_joint_angle_source import (
    JOINTS,
    RealtimeJointAngleSource,
)
from utils.path_utils import resolve_path
from utils.config_loader import load_config


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
CONTROLLER_ROOT = SRC_ROOT / "controller"
DEFAULT_CSV_DIRECTORY = (
    PROJECT_ROOT
    / "logs"
    / "dual_camera_motionBenchmark"
    / "excursion_predictions"
    / "csv_player"
)
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "logs" / "csv_player_gui"
DEFAULT_ENCODER_CALIBRATION_PATH = CONTROLLER_ROOT / "excursion_encoder_calibration.json"
DEFAULT_FORCE_CONFIG_PATH = CONTROLLER_ROOT / "config_force_control.json"
DEFAULT_DLC_CONFIG_PATH = SRC_ROOT / "observation" / "vision" / "config_deeplabcut_angle_control.json"
DEFAULT_CAMERA_CONFIG_PATH = resolve_param_path("camera_config_dlc.json")
DEFAULT_FORCE_CALIBRATION_DIR = API_ROOT / "calibration" / "current"
DEFAULT_ZERO_OFFSET_PATH = DEFAULT_FORCE_CALIBRATION_DIR / "as5048a_unloaded_zero_offsets.json"
DEFAULT_SERVO_PORT = "COM3"
DEFAULT_ENCODER_PORT = "COM5"
DEFAULT_CONTROL_HZ = 50.0
DEFAULT_VISION_HZ = 5.0
PLOT_HISTORY_SIZE = 1200
MAX_EVENTS_PER_POLL = 100
MAX_RESULT_ROWS = 100
CRITICAL_EVENT_KINDS = frozenset({"worker_error", "hard_limit", "result"})
LATEST_EVENT_KINDS = frozenset({"force", "angle"})
PLOT_UPDATE_INTERVAL_S = 0.1
PLOT_AXIS_REFRESH_INTERVAL_S = 1.0


def _load_hard_limits(path: Path) -> dict[str, float]:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    default = payload.get("default", {}) if isinstance(payload, dict) else {}
    raw = default.get("hard_force_limits_n_by_tendon", {})
    if not isinstance(raw, dict):
        raise ValueError("default.hard_force_limits_n_by_tendon must be an object")
    limits: dict[str, float] = {}
    for tendon in TENDONS:
        key = "LU" if tendon == "LUM" and "LUM" not in raw else tendon
        value = float(raw.get(key, 0.0))
        if not math.isfinite(value) or value <= 0.0:
            raise ValueError(f"A positive hard force limit is required for {tendon}")
        limits[tendon] = value
    return limits


def _resolve_config_path(raw_path: str, base_path: Path) -> Path:
    return resolve_path(raw_path, base_path)


class VisionWorker:
    """Read and process camera frames at a bounded rate."""

    def __init__(
        self,
        source: RealtimeJointAngleSource,
        emit: Callable[[dict[str, Any]], None],
        stop_event: threading.Event,
        output_csv: Path,
        playback_started_at: list[float | None],
        run_id: str = "",
        interval_s: float = 1.0 / DEFAULT_VISION_HZ,
    ) -> None:
        self.source = source
        self.emit = emit
        self.stop_event = stop_event
        self.output_csv = output_csv
        self.playback_started_at = playback_started_at
        self.run_id = str(run_id)
        self.interval_s = max(0.02, float(interval_s))

    def _emit(self, event: dict[str, Any]) -> None:
        self.emit(event)

    def run(self) -> None:
        frame_index = 0
        try:
            with ExitStack() as stack:
                handle = None
                writer = None
                playback_started = False
                while not self.stop_event.is_set():
                    started_at = time.monotonic()
                    run_started_at = self.playback_started_at[0]
                    if run_started_at is not None and not playback_started:
                        self.source.reset_temporal_history()
                        playback_started = True
                    measurement = self.source.read(frame_index)
                    if (
                        run_started_at is not None
                        and measurement.captured_at >= run_started_at
                    ):
                        if writer is None:
                            handle = stack.enter_context(
                                self.output_csv.open("w", encoding="utf-8", newline="")
                            )
                            writer = csv.writer(handle)
                            writer.writerow(
                                [
                                    "run_id",
                                    "frame_idx",
                                    "elapsed_s",
                                    "captured_at_monotonic",
                                    "inference_finished_at_monotonic",
                                    "inference_ms",
                                    "min_likelihood",
                                    "valid",
                                    "held_last",
                                    "reason",
                                    *[f"{joint}_deg" for joint in JOINTS],
                                ]
                            )
                        angles_deg = tuple(
                            math.degrees(value)
                            for value in measurement.flexion_angles_rad
                        )
                        elapsed_s = measurement.captured_at - run_started_at
                        writer.writerow(
                            [
                                self.run_id,
                                measurement.frame_index,
                                f"{elapsed_s:.6f}",
                                f"{measurement.captured_at:.6f}",
                                f"{measurement.inference_finished_at:.6f}",
                                f"{measurement.inference_ms:.6f}",
                                f"{measurement.min_likelihood:.6f}",
                                int(measurement.valid),
                                int(measurement.held_last),
                                measurement.reason or "",
                                *[f"{value:.6f}" for value in angles_deg],
                            ]
                        )
                        handle.flush()
                        self._emit(
                            {
                                "kind": "angle",
                                "frame_idx": measurement.frame_index,
                                "captured_at": measurement.captured_at,
                                "elapsed_s": elapsed_s,
                                "angles_deg": angles_deg,
                                "valid": measurement.valid,
                                "held_last": measurement.held_last,
                                "min_likelihood": measurement.min_likelihood,
                                "inference_ms": measurement.inference_ms,
                                "reason": measurement.reason,
                            }
                        )
                    frame_index += 1
                    self.stop_event.wait(
                        max(0.0, self.interval_s - (time.monotonic() - started_at))
                    )
        except Exception as exc:
            self._emit(
                {
                    "kind": "worker_error",
                    "source": "DLC",
                    "error": f"{type(exc).__name__}: {exc}",
                }
            )
            self.stop_event.set()


class DLCMeasurementWorker:
    """Measure joint angles from the camera without connecting servos."""

    def __init__(
        self,
        source: RealtimeJointAngleSource,
        emit: Callable[[dict[str, Any]], None],
        *,
        interval_s: float = 1.0 / DEFAULT_VISION_HZ,
    ) -> None:
        self.source = source
        self.emit = emit
        self.interval_s = max(0.02, float(interval_s))
        self.stop_event = threading.Event()
        self.finished_event = threading.Event()
        self.error: str | None = None
        self.thread = threading.Thread(
            target=self._run,
            name="csv-player-dlc-measurement",
            daemon=True,
        )

    def start(self) -> None:
        self.thread.start()

    def stop(self, timeout_s: float = 2.0) -> bool:
        self.stop_event.set()
        self.thread.join(timeout=max(0.0, timeout_s))
        return not self.thread.is_alive()

    def _run(self) -> None:
        frame_index = 0
        started_at = time.monotonic()
        try:
            self.source.open()
            self.emit({"kind": "dlc_ready", "worker": self})
            while not self.stop_event.is_set():
                loop_started_at = time.monotonic()
                measurement = self.source.read(frame_index)
                self.emit(
                    {
                        "kind": "angle",
                        "frame_idx": measurement.frame_index,
                        "captured_at": measurement.captured_at,
                        "elapsed_s": measurement.captured_at - started_at,
                        "angles_deg": tuple(
                            math.degrees(value)
                            for value in measurement.flexion_angles_rad
                        ),
                        "valid": measurement.valid,
                        "held_last": measurement.held_last,
                        "min_likelihood": measurement.min_likelihood,
                        "inference_ms": measurement.inference_ms,
                        "reason": measurement.reason,
                    }
                )
                frame_index += 1
                self.stop_event.wait(
                    max(0.0, self.interval_s - (time.monotonic() - loop_started_at))
                )
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
            self.emit({"kind": "dlc_error", "worker": self, "error": self.error})
        finally:
            try:
                self.source.close()
            finally:
                self.finished_event.set()
                self.emit({"kind": "dlc_stopped", "worker": self})


class CameraPreviewWorker:
    """Capture raw camera frames for preview without initializing DLC or servos."""

    def __init__(self, camera_config_path: Path) -> None:
        self.camera_config_path = Path(camera_config_path).expanduser()
        self.stop_event = threading.Event()
        self._frame_lock = threading.Lock()
        self._latest_frame: Any | None = None
        self._frame_index = -1
        self.error: str | None = None
        self.thread = threading.Thread(
            target=self._run,
            name="csv-player-camera-preview",
            daemon=True,
        )

    def start(self) -> None:
        self.thread.start()

    def stop(self, timeout_s: float = 2.0) -> bool:
        self.stop_event.set()
        self.thread.join(timeout=max(0.0, timeout_s))
        return not self.thread.is_alive()

    @property
    def latest_preview(self) -> tuple[int, bool, Any] | None:
        with self._frame_lock:
            if self._latest_frame is None:
                return None
            return self._frame_index, False, self._latest_frame.copy()

    def _run(self) -> None:
        capture: cv2.VideoCapture | None = None
        try:
            camera_config = load_config(self.camera_config_path)
            camera_index = int(camera_config.get("index", 0))
            backend = resolve_backend(camera_config.get("backend"))
            capture = (
                cv2.VideoCapture(camera_index, backend)
                if backend is not None
                else cv2.VideoCapture(camera_index)
            )
            if not capture.isOpened():
                raise RuntimeError(f"Camera not available: index={camera_index}")
            apply_camera_settings(capture, camera_config)
            calibration = setup_undistortion_from_config(
                camera_config,
                log_prefix="[csv-player-preview]",
            )
            frame_index = 0
            while not self.stop_event.is_set():
                ok, frame = capture.read()
                if not ok or frame is None:
                    raise RuntimeError("Failed to read a frame from the camera")
                frame = undistort_frame(frame, calibration)
                with self._frame_lock:
                    self._latest_frame = frame.copy()
                    self._frame_index = frame_index
                frame_index += 1
                self.stop_event.wait(0.01)
        except Exception as exc:
            self.error = f"{type(exc).__name__}: {exc}"
        finally:
            if capture is not None:
                capture.release()


class CsvPlayerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("CSV Player | DLC and Force Monitor")
        self.geometry("1500x980")
        self.minsize(1180, 780)
        try:
            self.state("zoomed")
        except tk.TclError:
            pass
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.events: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=400)
        self._critical_events: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=32)
        self._latest_events = LatestEventMailbox()
        self.csv_paths: list[Path] = []
        self.force_history = {
            tendon: deque(maxlen=PLOT_HISTORY_SIZE) for tendon in TENDONS
        }
        self.angle_history = {
            joint: deque(maxlen=PLOT_HISTORY_SIZE) for joint in JOINTS
        }
        self.force_lines: dict[str, Any] = {}
        self.force_limit_lines: dict[str, Any] = {}
        self.angle_lines: dict[str, Any] = {}
        self._runner_thread: threading.Thread | None = None
        self._vision_thread: threading.Thread | None = None
        self._stop_event: threading.Event | None = None
        self._start_event: threading.Event | None = None
        self._force_quit_event: threading.Event | None = None
        self._resource_lock = threading.Lock()
        self._hardware: ServoHardwareSession | None = None
        self._angle_source: RealtimeJointAngleSource | None = None
        self._camera_preview_worker: CameraPreviewWorker | None = None
        self._standalone_dlc_worker: DLCMeasurementWorker | None = None
        self._session_dir: Path | None = None
        self._closing_requested = False
        self._start_in_progress = False
        self._running = False
        self._connection_ready = False
        self._playback_started = False
        self._input_widgets: list[Any] = []
        self._csv_widgets: list[Any] = []
        self._pending_playback_config: tuple[
            Path, ExcursionTrajectory, dict[str, float], EncoderExcursionCalibration
        ] | None = None
        self._active_duration_s: float | None = None
        self._last_angle_frame = -1
        self._min_likelihood_seen = 1.0
        self._last_plot_update = 0.0
        self._force_plot_dirty = True
        self._angle_plot_dirty = True
        self._force_plot_backgrounds: dict[str, Any] = {}
        self._angle_plot_background = None
        self._force_last_full_draw_at = 0.0
        self._angle_last_full_draw_at = 0.0
        self._last_preview_frame = -1
        self._last_preview_is_overlay: bool | None = None
        self._last_preview_size = (0, 0)
        self._poll_after_id: str | None = None
        self._plot_resize_after_id: str | None = None
        self._photo = None
        self._preview_enabled = False
        self._dlc_enabled = False
        self._selected_csv_path: Path | None = None
        self._selected_csv_valid = False
        self._settings_expanded = False
        self._status_badges: dict[str, tk.Label] = {}
        self._header_status_vars: dict[str, tk.StringVar] = {}
        self._status_badge_values: dict[str, tuple[str, str]] = {}

        self.folder_var = tk.StringVar(value=str(DEFAULT_CSV_DIRECTORY))
        self.selected_var = tk.StringVar(value="CSV: 未選択")
        self.detail_var = tk.StringVar(value="CSVを選択してください")
        self.status_var = tk.StringVar(value="待機中")
        self.phase_var = tk.StringVar(value="未実行")
        self.camera_var = tk.StringVar(value="Camera: --")
        self.dlc_var = tk.StringVar(value="DLC: --")
        self.force_var = tk.StringVar(value="Force: --")
        self.elapsed_var = tk.StringVar(value="経過時間: --")
        self.progress_var = tk.StringVar(value="進捗: --")
        self.servo_port_var = tk.StringVar(value=DEFAULT_SERVO_PORT)
        self.encoder_port_var = tk.StringVar(value=DEFAULT_ENCODER_PORT)
        self.dlc_config_var = tk.StringVar(value=str(DEFAULT_DLC_CONFIG_PATH))
        self.camera_config_var = tk.StringVar(value=str(DEFAULT_CAMERA_CONFIG_PATH))
        self.limit_vars = {
            tendon: tk.StringVar(value="--") for tendon in TENDONS
        }
        for variable in self.limit_vars.values():
            variable.trace_add("write", self._mark_force_plot_dirty)

        try:
            defaults = _load_hard_limits(DEFAULT_FORCE_CONFIG_PATH)
            for tendon, value in defaults.items():
                self.limit_vars[tendon].set(f"{value:g}")
        except Exception as exc:
            self.status_var.set(f"Force設定を読めません: {exc}")

        self._build_widgets()
        self._refresh_files()
        self._schedule_event_poll()

    def _build_widgets(self) -> None:
        self._configure_styles()
        self.configure(bg="#f4f6f8")
        self.columnconfigure(0, weight=1)
        self.rowconfigure(6, weight=1)

        header = ttk.Frame(self, padding=(14, 10, 14, 8))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(1, weight=1)
        ttk.Label(
            header,
            text="CSV Player",
            style="AppTitle.TLabel",
        ).grid(row=0, column=0, sticky="w", padx=(0, 18))
        ttk.Label(
            header,
            textvariable=self.selected_var,
            style="HeaderFile.TLabel",
        ).grid(row=0, column=1, sticky="w")
        badges = ttk.Frame(header)
        badges.grid(row=0, column=2, sticky="e")
        for index, (key, label, initial) in enumerate(
            (
                ("servo", "Servo", "未接続"),
                ("encoder", "Encoder", "未接続"),
                ("camera", "Camera", "未接続"),
                ("dlc", "DLC", "未接続"),
                ("experiment", "実験", "待機中"),
            )
        ):
            variable = tk.StringVar(value=f"{label}: {initial}")
            self._header_status_vars[key] = variable
            badge = tk.Label(
                badges,
                textvariable=variable,
                font=("Yu Gothic UI", 9, "bold"),
                padx=9,
                pady=5,
                bd=0,
                relief="flat",
            )
            badge.grid(row=0, column=index, padx=(4 if index else 0, 0))
            self._status_badges[key] = badge
            self._set_status_badge(key, initial, "idle")

        telemetry = ttk.Frame(header, padding=(0, 8, 0, 0))
        telemetry.grid(row=1, column=0, columnspan=3, sticky="ew")
        telemetry.columnconfigure(6, weight=1)
        for index, variable in enumerate(
            (
                self.phase_var,
                self.elapsed_var,
                self.camera_var,
                self.dlc_var,
                self.force_var,
                self.progress_var,
            )
        ):
            ttk.Label(telemetry, textvariable=variable, style="Telemetry.TLabel").grid(
                row=0, column=index, sticky="w", padx=(0 if index == 0 else 14, 8)
            )
        self.progress_bar = ttk.Progressbar(
            telemetry, mode="determinate", maximum=100.0, length=180
        )
        self.progress_bar.grid(row=0, column=6, sticky="ew", padx=(8, 0))

        controls = ttk.Frame(self, padding=(14, 2, 14, 6))
        controls.grid(row=1, column=0, sticky="ew")
        controls.columnconfigure(0, weight=1)
        ttk.Label(controls, textvariable=self.selected_var, style="Selected.TLabel").grid(
            row=0, column=0, sticky="w", padx=(0, 8)
        )
        self.selected_path_button = ttk.Button(
            controls,
            text="フルパス",
            command=self._show_selected_path,
            state="disabled",
            style="Small.TButton",
        )
        self.selected_path_button.grid(row=0, column=1, padx=3)
        self.preflight_button = ttk.Button(
            controls, text="事前確認", command=self._preflight, style="Action.TButton"
        )
        self.preflight_button.grid(row=0, column=2, padx=3)
        self._input_widgets.append(self.preflight_button)
        self.connect_button = ttk.Button(
            controls, text="Connect", command=self._connect, style="Primary.TButton"
        )
        self.connect_button.grid(row=0, column=3, padx=3)
        self._input_widgets.append(self.connect_button)
        self.start_button = ttk.Button(
            controls,
            text="再生開始",
            command=self._start,
            state="disabled",
            style="Primary.TButton",
        )
        self.start_button.grid(row=0, column=4, padx=3)
        self.stop_button = ttk.Button(
            controls,
            text="ソフトウェア停止",
            command=self._request_stop,
            state="disabled",
            style="Warning.TButton",
        )
        self.stop_button.grid(row=0, column=5, padx=3)
        self.force_quit_button = ttk.Button(
            controls,
            text="Force Quit",
            command=self._force_quit,
            state="disabled",
            style="Danger.TButton",
        )
        self.force_quit_button.grid(row=0, column=6, padx=3)
        ttk.Label(controls, textvariable=self.status_var, style="OperationStatus.TLabel").grid(
            row=0, column=7, sticky="e", padx=(10, 0)
        )
        self.alert_label = ttk.Label(
            self,
            text="",
            foreground="#991b1b",
            style="Alert.TLabel",
        )
        self.alert_label.grid(row=2, column=0, sticky="ew", padx=14, pady=(0, 2))

        selection = ttk.LabelFrame(self, text="CSV選択", padding=6)
        selection.grid(row=3, column=0, sticky="ew", padx=14, pady=(0, 4))
        selection.columnconfigure(0, weight=2)
        selection.columnconfigure(2, weight=3)
        self.file_list = tk.Listbox(
            selection,
            height=3,
            exportselection=False,
            font=("Yu Gothic UI", 10),
            activestyle="none",
        )
        self.file_list.grid(row=0, column=0, sticky="ew", padx=(0, 5))
        file_scrollbar = ttk.Scrollbar(
            selection, orient="vertical", command=self.file_list.yview
        )
        file_scrollbar.grid(row=0, column=1, sticky="ns", padx=(0, 8))
        self.file_list.configure(yscrollcommand=file_scrollbar.set)
        self._input_widgets.append(self.file_list)
        self._csv_widgets.append(self.file_list)
        self.file_list.bind("<<ListboxSelect>>", self._on_select_csv)
        ttk.Label(
            selection,
            textvariable=self.detail_var,
            anchor="nw",
            justify="left",
            style="Detail.TLabel",
        ).grid(row=0, column=2, sticky="nsew", padx=5)

        settings_toggle = ttk.Frame(self, padding=(14, 2, 14, 2))
        settings_toggle.grid(row=4, column=0, sticky="ew")
        self.settings_toggle_button = ttk.Button(
            settings_toggle,
            text="設定を表示 ▼",
            command=self._toggle_settings,
            style="SectionToggle.TButton",
        )
        self.settings_toggle_button.pack(anchor="w")

        settings = ttk.LabelFrame(self, text="接続・安全設定", padding=8)
        settings.grid(row=5, column=0, sticky="ew", padx=14, pady=(0, 5))
        settings.columnconfigure(1, weight=1)
        settings.columnconfigure(3, weight=1)
        settings.columnconfigure(5, weight=2)
        settings.columnconfigure(7, weight=2)
        self.settings_body = settings
        ttk.Label(settings, text="CSVフォルダ").grid(row=0, column=0, sticky="w", padx=4, pady=3)
        self.folder_entry = ttk.Entry(settings, textvariable=self.folder_var)
        self.folder_entry.grid(row=0, column=1, columnspan=5, sticky="ew", padx=4, pady=3)
        self.choose_folder_button = ttk.Button(
            settings, text="参照", command=self._choose_folder, style="Small.TButton"
        )
        self.choose_folder_button.grid(row=0, column=6, padx=3, pady=3)
        self.refresh_files_button = ttk.Button(
            settings, text="更新", command=self._refresh_files, style="Small.TButton"
        )
        self.refresh_files_button.grid(row=0, column=7, padx=3, pady=3)
        self._input_widgets.extend(
            [self.folder_entry, self.choose_folder_button, self.refresh_files_button]
        )
        self._csv_widgets.extend(
            [self.folder_entry, self.choose_folder_button, self.refresh_files_button]
        )

        self._input_widgets.extend(
            [
                self._entry(settings, "Servoポート", self.servo_port_var, 1, 0, width=10),
                self._entry(settings, "Encoderポート", self.encoder_port_var, 1, 2, width=10),
                self._entry(settings, "DLC設定", self.dlc_config_var, 1, 4, width=30),
                self._entry(settings, "カメラ設定", self.camera_config_var, 1, 6, width=30),
            ]
        )
        force_controls = ttk.Frame(settings)
        force_controls.grid(row=2, column=0, columnspan=8, sticky="w", padx=4, pady=3)
        ttk.Label(force_controls, text="Force上限 [N]").pack(side="left", padx=(0, 8))
        for tendon in TENDONS:
            tendon_control = ttk.Frame(force_controls)
            tendon_control.pack(side="left", padx=(0, 8))
            ttk.Label(tendon_control, text=tendon).pack(side="left", padx=(0, 2))
            limit_entry = ttk.Entry(
                tendon_control, textvariable=self.limit_vars[tendon], width=7
            )
            limit_entry.pack(side="left")
            self._input_widgets.append(limit_entry)
        self.settings_body.grid_remove()

        body = ttk.PanedWindow(self, orient="vertical")
        body.grid(row=6, column=0, sticky="nsew", padx=14, pady=(0, 10))
        monitor = ttk.PanedWindow(body, orient="horizontal")
        body.add(monitor, weight=4)
        preview_frame = ttk.LabelFrame(monitor, text="Camera / DLC Preview", padding=6)
        force_frame = ttk.LabelFrame(monitor, text="Estimated Force", padding=6)
        angle_frame = ttk.LabelFrame(monitor, text="Flexion Angles", padding=6)
        monitor.add(preview_frame, weight=3)
        monitor.add(force_frame, weight=4)
        monitor.add(angle_frame, weight=3)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(1, weight=1)
        preview_controls = ttk.Frame(preview_frame)
        preview_controls.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        self.preview_toggle_button = ttk.Button(
            preview_controls,
            text="プレビューON",
            command=self._toggle_preview,
            style="Small.TButton",
        )
        self.preview_toggle_button.pack(side="right")
        self.dlc_toggle_button = ttk.Button(
            preview_controls,
            text="DLC ON",
            command=self._toggle_dlc,
            style="Small.TButton",
        )
        self.dlc_toggle_button.pack(side="right", padx=(0, 6))
        self.preview_label = tk.Label(
            preview_frame,
            text="カメラ未接続",
            anchor="center",
            bg="#e9eef3",
            fg="#475569",
            font=("Yu Gothic UI", 12),
        )
        self.preview_label.grid(row=1, column=0, sticky="nsew")
        self._build_force_plot(force_frame)
        self._build_angle_plot(angle_frame)

        results = ttk.Frame(body, padding=(0, 6, 0, 0))
        body.add(results, weight=1)
        results.columnconfigure(0, weight=1)
        results.rowconfigure(0, weight=1)
        result_table = ttk.LabelFrame(results, text="実験結果一覧", padding=5)
        result_table.grid(row=0, column=0, sticky="nsew")
        result_table.columnconfigure(0, weight=1)
        result_table.rowconfigure(0, weight=1)
        self.result_tree = ttk.Treeview(
            result_table,
            columns=("status", "max_force", "likelihood", "session"),
            show="headings",
            height=8,
        )
        headings = {
            "status": "状態",
            "max_force": "最大Force [N]",
            "likelihood": "最低DLC尤度",
            "session": "セッション",
        }
        for column, heading in headings.items():
            self.result_tree.heading(column, text=heading)
        self.result_tree.column("status", width=150, minwidth=120, stretch=False)
        self.result_tree.column("max_force", width=300, minwidth=180, stretch=True)
        self.result_tree.column("likelihood", width=140, minwidth=120, stretch=False)
        self.result_tree.column("session", width=460, minwidth=220, stretch=True)
        self.result_tree.grid(row=0, column=0, sticky="nsew")
        result_scrollbar = ttk.Scrollbar(
            result_table, orient="vertical", command=self.result_tree.yview
        )
        result_scrollbar.grid(row=0, column=1, sticky="ns")
        self.result_tree.configure(yscrollcommand=result_scrollbar.set)
        self.open_session_button = ttk.Button(
            result_table,
            text="セッションフォルダを開く",
            command=self._open_session_folder,
            state="disabled",
            style="Small.TButton",
        )
        self.open_session_button.grid(row=1, column=0, sticky="w", pady=(5, 0))

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        try:
            style.theme_use("vista")
        except tk.TclError:
            try:
                style.theme_use("clam")
            except tk.TclError:
                pass
        style.configure("TLabel", font=("Yu Gothic UI", 10))
        style.configure("TEntry", font=("Yu Gothic UI", 10))
        style.configure("TLabelframe.Label", font=("Yu Gothic UI", 12, "bold"), foreground="#1e293b")
        style.configure("AppTitle.TLabel", font=("Yu Gothic UI", 18, "bold"), foreground="#0f172a")
        style.configure("HeaderFile.TLabel", font=("Yu Gothic UI", 11), foreground="#334155")
        style.configure("Selected.TLabel", font=("Yu Gothic UI", 11, "bold"), foreground="#1e293b")
        style.configure("Detail.TLabel", font=("Yu Gothic UI", 10), foreground="#475569")
        style.configure("Telemetry.TLabel", font=("Yu Gothic UI", 9), foreground="#475569")
        style.configure("OperationStatus.TLabel", font=("Yu Gothic UI", 10, "bold"), foreground="#334155")
        style.configure("Alert.TLabel", font=("Yu Gothic UI", 10, "bold"))
        style.configure("Small.TButton", font=("Yu Gothic UI", 10), padding=(8, 4))
        style.configure("Action.TButton", font=("Yu Gothic UI", 10, "bold"), padding=(10, 5))
        style.configure("Primary.TButton", font=("Yu Gothic UI", 10, "bold"), padding=(11, 5))
        style.configure("Warning.TButton", font=("Yu Gothic UI", 10, "bold"), foreground="#92400e", padding=(11, 5))
        style.configure("Danger.TButton", font=("Yu Gothic UI", 10, "bold"), foreground="#991b1b", padding=(11, 5))
        style.configure("SectionToggle.TButton", font=("Yu Gothic UI", 10, "bold"), padding=(4, 2))
        style.configure("Treeview", font=("Yu Gothic UI", 10), rowheight=28)
        style.configure("Treeview.Heading", font=("Yu Gothic UI", 10, "bold"))

    def _set_status_badge(self, key: str, text: str, state: str = "idle") -> None:
        palette = {
            "idle": ("#e2e8f0", "#334155"),
            "ok": ("#dcfce7", "#166534"),
            "busy": ("#dbeafe", "#1d4ed8"),
            "warn": ("#fef3c7", "#92400e"),
            "error": ("#fee2e2", "#991b1b"),
        }
        badge = self._status_badges.get(key)
        variable = self._header_status_vars.get(key)
        if badge is None or variable is None:
            return
        badge_value = (str(text), str(state))
        if self._status_badge_values.get(key) == badge_value:
            return
        self._status_badge_values[key] = badge_value
        label = {
            "servo": "Servo",
            "encoder": "Encoder",
            "camera": "Camera",
            "dlc": "DLC",
            "experiment": "実験",
        }.get(key, key)
        variable.set(f"{label}: {text}")
        background, foreground = palette.get(state, palette["idle"])
        badge.configure(background=background, foreground=foreground)

    def _toggle_settings(self) -> None:
        self._settings_expanded = not self._settings_expanded
        if self._settings_expanded:
            self.settings_body.grid()
            self.settings_toggle_button.configure(text="設定を隠す ▲")
        else:
            self.settings_body.grid_remove()
            self.settings_toggle_button.configure(text="設定を表示 ▼")

    def _show_selected_path(self) -> None:
        path = self._selected_csv_path
        if path is None:
            messagebox.showinfo("CSVフルパス", "CSVが選択されていません")
            return
        messagebox.showinfo("CSVフルパス", str(path.resolve()))

    @staticmethod
    def _shorten_path(path: Path | str, max_chars: int = 72) -> str:
        value = str(path)
        if len(value) <= max_chars:
            return value
        head = max_chars // 2 - 1
        tail = max_chars - head - 1
        return f"{value[:head]}…{value[-tail:]}"

    def _apply_status_event(self, text: str, phase: str) -> None:
        self.status_var.set(text)
        self.phase_var.set(phase)
        if "カメラとDLC" in text:
            self._set_status_badge("camera", "接続中", "busy")
            self._set_status_badge("dlc", "準備中", "busy")
            self._set_status_badge("experiment", "準備中", "busy")
        elif "DLC" in text:
            self._set_status_badge("dlc", "キャリブレーション", "busy")
            self._set_status_badge("experiment", phase, "busy")
        elif "サーボとエンコーダ" in text:
            self._set_status_badge("servo", "接続中", "busy")
            self._set_status_badge("encoder", "接続中", "busy")
            self._set_status_badge("experiment", "接続中", "busy")
        elif phase in {"実行中", "再生開始"}:
            self._set_status_badge("experiment", "実行中", "ok")
        elif phase in {"完了", "接続完了"}:
            self._set_status_badge("experiment", phase, "ok")
        elif phase in {"停止", "Force Quit"}:
            self._set_status_badge("experiment", phase, "warn")

    @staticmethod
    def _entry(parent: ttk.Widget, label: str, variable: tk.StringVar, row: int, column: int, *, width: int = 18) -> ttk.Entry:
        ttk.Label(parent, text=label).grid(row=row, column=column, padx=(4, 1), pady=4, sticky="w")
        entry = ttk.Entry(parent, textvariable=variable, width=width)
        entry.grid(
            row=row, column=column + 1, padx=(0, 4), pady=4, sticky="ew"
        )
        return entry

    def _build_force_plot(self, parent: ttk.Frame) -> None:
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)
        self.force_figure = Figure(figsize=(8, 5), dpi=100)
        self.force_axes = self.force_figure.subplots(3, 2, squeeze=False).flatten()
        force_handles: tuple[Any, Any] | None = None
        for axis, tendon in zip(self.force_axes, TENDONS):
            line, = axis.plot([], [], label="Estimated")
            limit_line, = axis.plot(
                [], [], color="#b91c1c", linestyle="--", alpha=0.55, label="Limit"
            )
            if force_handles is None:
                force_handles = (line, limit_line)
            self.force_lines[tendon] = line
            self.force_limit_lines[tendon] = limit_line
            line.set_animated(True)
            limit_line.set_animated(True)
            axis.set_title(tendon, fontsize=10, pad=4)
            axis.set_ylabel("N", fontsize=9)
            axis.tick_params(labelsize=8)
            axis.set_xlim(0.0, 20.0)
            axis.set_ylim(0.0, 1.0)
            axis.grid(True, alpha=0.3)
        for axis in self.force_axes[-2:]:
            axis.set_xlabel("Time [s]", fontsize=9)
        self.force_figure.suptitle(
            "Estimated force", fontsize=12, x=0.02, y=0.995, ha="left"
        )
        if force_handles is not None:
            self.force_figure.legend(
                force_handles,
                ("Estimated", "Limit"),
                loc="upper right",
                bbox_to_anchor=(0.98, 0.995),
                fontsize=8,
                frameon=False,
                ncol=2,
            )
        self.force_figure.subplots_adjust(
            left=0.08, right=0.98, bottom=0.09, top=0.91, hspace=0.52, wspace=0.28
        )
        self.force_canvas = FigureCanvasTkAgg(self.force_figure, master=parent)
        self.force_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.force_canvas.mpl_connect(
            "draw_event", self._capture_force_plot_backgrounds
        )
        self.force_canvas.draw()
        self._blit_force_plot()

    def _build_angle_plot(self, parent: ttk.Frame) -> None:
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)
        self.angle_figure = Figure(figsize=(6, 5), dpi=100)
        self.angle_axis = self.angle_figure.subplots()
        for joint in JOINTS:
            line, = self.angle_axis.plot([], [], label=f"{joint} flexion")
            line.set_animated(True)
            self.angle_lines[joint] = line
        self.angle_axis.set_title("Flexion angle", fontsize=12, pad=8)
        self.angle_axis.set_xlabel("Time [s]", fontsize=10)
        self.angle_axis.set_ylabel("Flexion angle [deg]", fontsize=10)
        self.angle_axis.set_xlim(0.0, 20.0)
        self.angle_axis.set_ylim(0.0, 180.0)
        self.angle_axis.tick_params(labelsize=9)
        self.angle_axis.grid(True, alpha=0.3)
        self.angle_axis.legend(loc="upper left", fontsize=9, frameon=False)
        self.angle_figure.subplots_adjust(left=0.16, right=0.98, bottom=0.13, top=0.90)
        self.angle_canvas = FigureCanvasTkAgg(self.angle_figure, master=parent)
        self.angle_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")
        self.angle_canvas.mpl_connect(
            "draw_event", self._capture_angle_plot_background
        )
        self.angle_canvas.draw()
        self._blit_angle_plot()

    def _choose_folder(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.folder_var.get())
        if selected:
            self.folder_var.set(selected)
            self._refresh_files()

    def _set_input_widgets_enabled(self, enabled: bool) -> None:
        self._set_widgets_enabled(self._input_widgets, enabled)

    @staticmethod
    def _set_widgets_enabled(widgets: list[Any], enabled: bool) -> None:
        state = "normal" if enabled else "disabled"
        for widget in widgets:
            widget.configure(state=state)

    def _preflight(self) -> None:
        if self._running or self._start_in_progress:
            return
        selection = self.file_list.curselection()
        if not selection:
            messagebox.showwarning("CSV未選択", "事前確認するCSVを選択してください")
            return

        path = self.csv_paths[selection[0]]
        try:
            limits = self._parse_limits()
            calibration = self._load_encoder_calibration()
            trajectory = ExcursionTrajectory.from_csv(path)
            required_files = (
                ("CSV", path),
                ("encoder位置キャリブレーション", DEFAULT_ENCODER_CALIBRATION_PATH),
                ("DLC設定", Path(self.dlc_config_var.get()).expanduser()),
                ("カメラ設定", Path(self.camera_config_var.get()).expanduser()),
            )
            missing = [
                f"{label}: {file_path}"
                for label, file_path in required_files
                if not file_path.is_file()
            ]
            if missing:
                raise FileNotFoundError("必要なファイルがありません:\n" + "\n".join(missing))
        except Exception as exc:
            self.status_var.set(f"事前確認エラー: {exc}")
            self.phase_var.set("事前確認エラー")
            messagebox.showerror("事前確認エラー", str(exc))
            return

        self.status_var.set("事前確認OK（接続は未確認）")
        self.phase_var.set("事前確認")
        self._set_status_badge("experiment", "事前確認OK", "ok")
        self.detail_var.set(
            f"ファイル: {path.name}\n"
            f"行数: {len(trajectory.elapsed_times_s)}\n"
            f"再生時間: {trajectory.duration_s:.3f} s\n"
            f"encoder換算: {calibration.calibration_status}\n"
            f"Force上限: {', '.join(f'{tendon}={limits[tendon]:g} N' for tendon in TENDONS)}\n"
            "状態: 事前確認OK（camera / serial接続は未確認）"
        )

    def _open_session_folder(self) -> None:
        session_dir = self._session_dir
        if session_dir is None or not session_dir.is_dir():
            messagebox.showinfo("セッションフォルダ", "開けるセッションフォルダがありません")
            return
        try:
            startfile = getattr(os, "startfile", None)
            if not callable(startfile):
                raise RuntimeError("このOSではフォルダを開けません")
            startfile(str(session_dir))
        except Exception as exc:
            messagebox.showerror("フォルダを開けません", str(exc))

    def _refresh_files(self) -> None:
        folder = Path(self.folder_var.get()).expanduser()
        self.file_list.delete(0, tk.END)
        paths: list[tuple[float, Path]] = []
        try:
            for path in folder.glob("*.csv"):
                try:
                    if path.is_file():
                        paths.append((path.stat().st_mtime, path))
                except OSError:
                    continue
        except OSError as exc:
            self.csv_paths = []
            self._selected_csv_path = None
            self._selected_csv_valid = False
            self.selected_var.set("CSV: 未選択")
            self.detail_var.set(f"CSV一覧を読めません: {exc}")
            self.selected_path_button.configure(state="disabled")
            self.connect_button.configure(
                state="disabled" if self._running else "normal"
            )
            self.start_button.configure(state="disabled")
            return
        self.csv_paths = [path for _, path in sorted(paths, reverse=True)]
        for path in self.csv_paths:
            self.file_list.insert(tk.END, path.name)
        if self.csv_paths:
            self.file_list.selection_set(0)
            self._on_select_csv()
        else:
            self._selected_csv_path = None
            self._selected_csv_valid = False
            self.selected_var.set("CSV: 未選択")
            self.detail_var.set(f"CSVがありません: {folder}")
            self.selected_path_button.configure(state="disabled")
            self.connect_button.configure(
                state="disabled" if self._running else "normal"
            )
            self.start_button.configure(state="disabled")

    def _on_select_csv(self, _event: object | None = None) -> None:
        selection = self.file_list.curselection()
        if not selection:
            return
        path = self.csv_paths[selection[0]]
        self._selected_csv_path = path
        self.selected_var.set(f"CSV: {path.name}")
        self.selected_path_button.configure(state="normal")
        try:
            calibration = self._load_encoder_calibration()
            trajectory = ExcursionTrajectory.from_csv(path)
            scale_note = (
                "公称値（張力下で要校正）"
                if calibration.calibration_status.lower() == "nominal"
                else "tendon別校正値"
            )
            self.detail_var.set(
                f"ファイル: {path.name}\n"
                f"行数: {len(trajectory.elapsed_times_s)}\n"
                f"再生時間: {trajectory.duration_s:.3f} s\n"
                f"encoder換算: {scale_note}\n"
                f"状態: 実行可能"
            )
            self.connect_button.configure(
                state="normal"
                if not self._running and not self._start_in_progress
                else "disabled"
            )
            self._selected_csv_valid = True
            self.start_button.configure(
                state="normal"
                if self._running and self._connection_ready and self._selected_csv_valid
                else "disabled"
            )
        except Exception as exc:
            self.detail_var.set(f"検証エラー: {type(exc).__name__}: {exc}")
            self._selected_csv_valid = False
            self.connect_button.configure(
                state="disabled" if self._running else "normal"
            )
            self.start_button.configure(state="disabled")

    @staticmethod
    def _load_encoder_calibration() -> EncoderExcursionCalibration:
        return load_encoder_excursion_calibration(DEFAULT_ENCODER_CALIBRATION_PATH)

    def _parse_limits(self) -> dict[str, float]:
        limits: dict[str, float] = {}
        for tendon in TENDONS:
            value = float(self.limit_vars[tendon].get())
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{tendon}のHard limitは正の有限値が必要です")
            limits[tendon] = value
        return limits

    def _update_progress(self, elapsed_s: float, phase: str) -> None:
        if phase == "return":
            self.progress_bar.configure(value=100.0)
            self.progress_var.set("進捗: 再生完了 / 初期位置へ復帰中")
            return
        duration_s = self._active_duration_s
        if duration_s is None or duration_s <= 0.0:
            return
        ratio = min(max(elapsed_s / duration_s, 0.0), 1.0)
        remaining_s = max(duration_s - elapsed_s, 0.0)
        self.progress_bar.configure(value=ratio * 100.0)
        self.progress_var.set(
            f"進捗: {ratio * 100.0:.0f}% / 残り: {remaining_s:.1f} s"
        )

    def _start(self) -> None:
        if (
            not self._running
            or not self._connection_ready
            or self._start_event is None
            or self._playback_started
        ):
            return
        selection = self.file_list.curselection()
        if not selection:
            messagebox.showwarning("CSV未選択", "再生するCSVを選択してください")
            return
        path = self.csv_paths[selection[0]]
        try:
            limits = self._parse_limits()
            calibration = self._load_encoder_calibration()
            trajectory = ExcursionTrajectory.from_csv(path)
        except Exception as exc:
            messagebox.showerror("再生設定エラー", str(exc))
            return
        calibration_warning = (
            "\nencoder換算は半径20 mmからの公称値です。張力下の換算係数と引く方向の符号を確認してください。"
            if calibration.calibration_status.lower() == "nominal"
            else ""
        )
        if not messagebox.askyesno(
            "再生開始",
            f"{path.name}\n再生時間: {trajectory.duration_s:.3f} s\n"
            f"接続済みです。CSV再生を開始しますか？{calibration_warning}",
        ):
            return
        with self._resource_lock:
            self._pending_playback_config = (path, trajectory, limits, calibration)
        self._playback_started = True
        self._active_duration_s = trajectory.duration_s
        self.progress_bar.configure(value=0.0)
        self.progress_var.set(f"進捗: 0% / 残り: {trajectory.duration_s:.1f} s")
        self._set_widgets_enabled(self._csv_widgets, False)
        self.start_button.configure(state="disabled")
        self.status_var.set("再生開始")
        self.phase_var.set("再生開始")
        self._set_status_badge("experiment", "再生開始", "busy")
        self._start_event.set()

    def _connect(self) -> None:
        if self._running or self._start_in_progress:
            return
        self._start_in_progress = True
        self.start_button.configure(state="disabled")
        if not messagebox.askyesno(
            "接続確認",
            "カメラとサーボに接続します。\n"
            "接続後にCSVを選択し、再生開始を押すまでサーボは動かしません。接続しますか？",
        ):
            self._start_in_progress = False
            self.connect_button.configure(state="normal")
            return

        if not self._stop_dlc_measurement_worker():
            self._start_in_progress = False
            self.connect_button.configure(state="normal")
            messagebox.showwarning(
                "DLC停止待ち",
                "DLC計測中のカメラを解放できませんでした。少し待ってから再度Connectを押してください。",
            )
            return

        if not self._stop_camera_preview_worker():
            self._start_in_progress = False
            self.connect_button.configure(state="normal")
            messagebox.showwarning(
                "カメラ切替待ち",
                "プレビュー用カメラを解放できませんでした。少し待ってから再度Connectを押してください。",
            )
            return

        # Tk variables belong to the main thread.  Snapshot all run settings
        # before starting the worker instead of reading widgets from it.
        dlc_config_path = Path(self.dlc_config_var.get())
        camera_config_path = Path(self.camera_config_var.get())
        servo_port = self.servo_port_var.get()
        encoder_port = self.encoder_port_var.get()
        session_dir = DEFAULT_OUTPUT_DIRECTORY / datetime.now().strftime("%Y%m%d_%H%M%S")
        try:
            session_dir.mkdir(parents=True, exist_ok=False)
        except Exception as exc:
            messagebox.showerror("セッション作成エラー", str(exc))
            self._start_in_progress = False
            self.connect_button.configure(state="normal")
            if self._preview_enabled:
                self._start_camera_preview_worker()
            return
        self._session_dir = session_dir
        stop_event = threading.Event()
        start_event = threading.Event()
        force_quit_event = threading.Event()
        with self._resource_lock:
            self._pending_playback_config = None
        self._stop_event = stop_event
        self._start_event = start_event
        self._force_quit_event = force_quit_event
        self._running = True
        self._connection_ready = False
        self._playback_started = False
        self._active_duration_s = None
        self.progress_bar.configure(value=0.0)
        self.progress_var.set("進捗: CSV選択待ち")
        self._set_input_widgets_enabled(False)
        self._set_widgets_enabled(self._csv_widgets, True)
        self._last_angle_frame = -1
        self._last_preview_frame = -1
        self._last_preview_is_overlay = None
        self._min_likelihood_seen = 1.0
        self._latest_events.clear()
        for history in self.force_history.values():
            history.clear()
        for history in self.angle_history.values():
            history.clear()
        self._force_plot_dirty = True
        self._angle_plot_dirty = True
        self.alert_label.configure(text="")
        self.status_var.set("準備中")
        self.phase_var.set("準備中")
        self._set_status_badge("experiment", "準備中", "busy")
        self._set_status_badge("camera", "接続中", "busy")
        self._set_status_badge("dlc", "準備中", "busy")
        self._set_status_badge("servo", "接続待ち", "busy")
        self._set_status_badge("encoder", "接続待ち", "busy")
        self.connect_button.configure(state="disabled")
        self.dlc_toggle_button.configure(state="disabled", text="DLC（session）")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self.force_quit_button.configure(state="normal")
        self._runner_thread = threading.Thread(
            target=self._run_session,
            args=(
                None,
                None,
                None,
                session_dir,
                dlc_config_path,
                camera_config_path,
                servo_port,
                encoder_port,
                stop_event,
                start_event,
                force_quit_event,
            ),
            name="csv-player-gui-session",
            daemon=True,
        )
        try:
            self._runner_thread.start()
        except Exception as exc:
            self._running = False
            self._stop_event = None
            self._start_event = None
            self._force_quit_event = None
            with self._resource_lock:
                self._pending_playback_config = None
            self._connection_ready = False
            self._playback_started = False
            self._start_in_progress = False
            self._set_input_widgets_enabled(True)
            self.dlc_toggle_button.configure(state="normal", text="DLC ON")
            self.stop_button.configure(state="disabled")
            self.force_quit_button.configure(state="disabled")
            messagebox.showerror("実行開始エラー", str(exc))
            return
        self._start_in_progress = False

    def _run_session(
        self,
        csv_path: Path | None,
        trajectory: ExcursionTrajectory | None,
        limits: dict[str, float] | None,
        session_dir: Path,
        dlc_config_path: Path,
        camera_config_path: Path,
        servo_port: str,
        encoder_port: str,
        stop_event: threading.Event,
        start_event: threading.Event,
        force_quit_event: threading.Event,
    ) -> None:
        hardware: ServoHardwareSession | None = None
        source: RealtimeJointAngleSource | None = None
        playback_service: EncoderPlaybackService | None = None
        vision_worker: VisionWorker | None = None
        vision_thread: threading.Thread | None = None
        playback_started_at: list[float | None] = [None]
        run_started_at: float | None = None
        completed = False
        termination_reason = "error"
        error_text: str | None = None
        self._emit({"kind": "status", "text": "カメラとDLCを準備中", "phase": "準備中"})
        try:
            source = RealtimeJointAngleSource(
                dlc_config_path=dlc_config_path,
                camera_config_path=camera_config_path,
                min_likelihood=0.3,
                temporal_outlier_enabled=True,
                temporal_jump_threshold_deg=30.0,
                temporal_max_hold_frames=5,
            )
            source.open()
            with self._resource_lock:
                self._angle_source = source
            if stop_event.is_set():
                raise PlaybackStopped("Playback was stopped during camera setup")
            vision_worker = VisionWorker(
                source,
                self._emit,
                stop_event,
                session_dir / "angle_trace.csv",
                playback_started_at,
                run_id=session_dir.name,
            )
            vision_thread = threading.Thread(
                target=vision_worker.run,
                name="csv-player-gui-dlc",
                daemon=True,
            )
            vision_thread.start()
            if stop_event.is_set():
                raise PlaybackStopped("Playback was stopped during camera setup")
            self._emit({"kind": "status", "text": "サーボとエンコーダを接続中", "phase": "接続"})
            hardware = ServoHardwareSession(
                servo_ids=SERVO_IDS,
                servo_names=SERVO_NAMES_BY_ID,
                encoder_sensor_ids=tuple(
                    ENCODER_SENSOR_IDS_BY_SERVO_ID[servo_id] for servo_id in SERVO_IDS
                ),
                servo_port=servo_port,
                encoder_port=encoder_port,
                baudrate=921600,
                serial_timeout_s=0.05,
                encoder_rate_hz=70.0,
            )
            with self._resource_lock:
                self._hardware = hardware
            if stop_event.is_set():
                raise PlaybackStopped("Playback was stopped before hardware connection")
            hardware.connect(startup_timeout_s=3.0)
            if stop_event.is_set():
                raise PlaybackStopped("Playback was stopped during hardware connection")
            feedback = hardware.wait_for_feedback(3.0, require_encoder=True)
            if stop_event.is_set():
                raise PlaybackStopped("Playback was stopped while waiting for telemetry")
            self._emit(
                {
                    "kind": "connected",
                    "text": "接続完了。再生開始を押してください",
                    "phase": "接続完了",
                }
            )
            while not start_event.wait(0.1):
                if stop_event.is_set():
                    raise PlaybackStopped("Playback was stopped before playback started")
            if stop_event.is_set():
                raise PlaybackStopped("Playback was stopped before playback started")
            with self._resource_lock:
                playback_config = self._pending_playback_config
            if playback_config is None:
                raise RuntimeError("Playback configuration was not selected")
            csv_path, trajectory, limits, calibration = playback_config
            playback_config = build_encoder_playback_config(
                calibration,
                control_hz=DEFAULT_CONTROL_HZ,
                telemetry_stale_s=0.2,
                telemetry_wait_s=3.0,
                display_interval_s=0.25,
                return_to_start=True,
                return_timeout_s=5.0,
                return_stable_frames=3,
                hard_force_limits_n_by_tendon=limits,
            )
            playback_service = EncoderPlaybackService(
                trajectory=trajectory,
                calibration=calibration,
                trace_path=session_dir / "control_trace.csv",
                run_id=session_dir.name,
                force_calibration_dir=DEFAULT_FORCE_CALIBRATION_DIR,
                zero_offset_path=DEFAULT_ZERO_OFFSET_PATH,
                config=playback_config,
                record_force=True,
                on_sample=lambda sample: self._emit({"kind": "force", "sample": sample}),
                stop_event=stop_event,
            )

            run_started_at = time.monotonic()
            video_path = session_dir / "dlc_overlay.mp4"
            source.start_overlay_recording(video_path, fps=DEFAULT_VISION_HZ, codec="mp4v", queue_size=10)
            playback_started_at[0] = run_started_at
            self._emit({"kind": "status", "text": "実行中", "phase": "実行中"})

            playback_service.run(hardware)
            completed = True
            termination_reason = "completed"
            self._emit({"kind": "status", "text": "完了", "phase": "完了"})
        except HardForceLimitError as exc:
            termination_reason = "force_hard_limit"
            error_text = str(exc)
            stop_event.set()
            self._emit(
                {
                    "kind": "hard_limit",
                    "text": (
                        f"FORCE HARD LIMIT: {exc.tendon} "
                        f"{exc.force_n:.3f} N / {exc.limit_n:.3f} N"
                    ),
                    "phase": "ハードリミット停止",
                }
            )
        except ForceTelemetryUnavailable as exc:
            termination_reason = "force_telemetry_unavailable"
            error_text = str(exc)
            stop_event.set()
            self._emit(
                {
                    "kind": "hard_limit",
                    "text": f"FORCE SAFETY STOP: {exc}",
                    "phase": "Force停止",
                }
            )
        except PlaybackStopped as exc:
            termination_reason = "force_quit" if force_quit_event.is_set() else "operator_stop"
            error_text = str(exc)
            self._emit(
                {
                    "kind": "status",
                    "text": "Force Quitで停止しました"
                    if termination_reason == "force_quit"
                    else "安全停止しました",
                    "phase": "Force Quit"
                    if termination_reason == "force_quit"
                    else "停止",
                }
            )
        except Exception as exc:
            termination_reason = "error"
            error_text = f"{type(exc).__name__}: {exc}"
            self._emit({"kind": "worker_error", "source": "実行", "error": error_text})
        finally:
            stop_event.set()
            cleanup_errors: list[str] = []
            if hardware is not None:
                try:
                    hardware.stop_servos()
                except Exception as exc:
                    cleanup_errors.append(f"servo stop: {type(exc).__name__}: {exc}")
            if vision_thread is not None:
                # The source is shared with the vision worker.  Wait for the
                # reader to leave source.read() before closing the camera or
                # the overlay recorder from this worker.
                vision_thread.join()
            if source is not None:
                try:
                    source.stop_overlay_recording()
                except Exception as exc:
                    cleanup_errors.append(
                        f"overlay recording: {type(exc).__name__}: {exc}"
                    )
                try:
                    source.close()
                except Exception as exc:
                    cleanup_errors.append(f"camera close: {type(exc).__name__}: {exc}")
            if playback_service is not None:
                try:
                    playback_service.close()
                except Exception as exc:
                    cleanup_errors.append(
                        f"playback service close: {type(exc).__name__}: {exc}"
                    )
            if hardware is not None:
                try:
                    hardware.close()
                except Exception as exc:
                    cleanup_errors.append(f"hardware close: {type(exc).__name__}: {exc}")
            if cleanup_errors:
                cleanup_text = "; ".join(cleanup_errors)
                error_text = (
                    f"{error_text}; cleanup: {cleanup_text}"
                    if error_text
                    else f"cleanup: {cleanup_text}"
                )
                if termination_reason in {"completed", "operator_stop"}:
                    termination_reason = "error"
                self._emit(
                    {
                        "kind": "worker_error",
                        "source": "cleanup",
                        "error": cleanup_text,
                    }
                )
            manifest = {
                "run_id": session_dir.name,
                "timestamp_timebase": "time.monotonic seconds; shared by control and vision logs within this run",
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "input_csv_path": (
                    str(csv_path.resolve()) if csv_path is not None else None
                ),
                "session_dir": str(session_dir.resolve()),
                "hard_force_limits_n_by_tendon": limits,
                "servo_port": servo_port,
                "encoder_port": encoder_port,
                "completed": completed,
                "termination_reason": termination_reason,
                "error": error_text,
                "outputs": {
                    "control_trace": str((session_dir / "control_trace.csv").resolve()),
                    "angle_trace": str((session_dir / "angle_trace.csv").resolve()),
                    "overlay_video": str((session_dir / "dlc_overlay.mp4").resolve()),
                },
            }
            try:
                (session_dir / "session_manifest.json").write_text(
                    json.dumps(manifest, indent=2, ensure_ascii=False),
                    encoding="utf-8",
                )
            except Exception as exc:
                error_text = (
                    f"{error_text}; manifest: {type(exc).__name__}: {exc}"
                    if error_text
                    else f"manifest: {type(exc).__name__}: {exc}"
                )
                termination_reason = "error"
                self._emit(
                    {
                        "kind": "worker_error",
                        "source": "manifest",
                        "error": f"{type(exc).__name__}: {exc}",
                    }
                )
            with self._resource_lock:
                self._angle_source = None
                self._hardware = None
            self._emit(
                {
                    "kind": "result",
                    "status": termination_reason,
                    "session_dir": str(session_dir),
                    "error": error_text,
                }
            )

    def _emit(self, event: dict[str, Any]) -> None:
        if event.get("kind") in LATEST_EVENT_KINDS:
            self._latest_events.publish(event)
            return
        events = (
            self._critical_events
            if event.get("kind") in CRITICAL_EVENT_KINDS
            else self.events
        )
        try:
            events.put_nowait(event)
        except queue.Full:
            try:
                events.get_nowait()
            except queue.Empty:
                return
            try:
                events.put_nowait(event)
            except queue.Full:
                pass

    def _request_stop(self) -> None:
        if not self._running or self._stop_event is None:
            return
        self.status_var.set("停止処理中")
        self.phase_var.set("停止処理中")
        self._set_status_badge("experiment", "停止処理中", "warn")
        self._stop_event.set()
        self.stop_button.configure(state="disabled")

    def _force_quit(self) -> None:
        if not self._running or self._stop_event is None:
            return
        force_quit_event = self._force_quit_event
        if force_quit_event is None or force_quit_event.is_set():
            return
        force_quit_event.set()
        self._stop_event.set()
        self.status_var.set("Force Quit処理中")
        self.phase_var.set("Force Quit")
        self._set_status_badge("experiment", "Force Quit", "error")
        self.stop_button.configure(state="disabled")
        self.force_quit_button.configure(state="disabled")
        with self._resource_lock:
            hardware = self._hardware
        if hardware is not None:
            threading.Thread(
                target=self._force_stop_hardware,
                args=(hardware,),
                name="csv-player-force-quit",
                daemon=True,
            ).start()

    @staticmethod
    def _force_stop_hardware(hardware: ServoHardwareSession) -> None:
        try:
            hardware.stop_servos()
        except Exception:
            # The session worker reports cleanup/transport errors through the
            # normal result path; the GUI callback must remain non-blocking.
            pass

    def _schedule_event_poll(self) -> None:
        if self._poll_after_id is not None:
            return
        try:
            self._poll_after_id = self.after(100, self._poll_events)
        except tk.TclError:
            self._poll_after_id = None

    def _poll_events(self) -> None:
        self._poll_after_id = None
        for _ in range(MAX_EVENTS_PER_POLL):
            try:
                event = self._critical_events.get_nowait()
            except queue.Empty:
                try:
                    event = self.events.get_nowait()
                except queue.Empty:
                    break
            try:
                kind = event.get("kind")
                if kind == "status":
                    self._apply_status_event(
                        str(event.get("text", "")), str(event.get("phase", ""))
                    )
                elif kind == "connected":
                    self._connection_ready = True
                    self.status_var.set(str(event.get("text", "接続完了")))
                    self.phase_var.set(str(event.get("phase", "接続完了")))
                    self._set_status_badge("servo", "接続済み", "ok")
                    self._set_status_badge("encoder", "接続済み", "ok")
                    self._set_status_badge("camera", "接続済み", "ok")
                    self._set_status_badge("dlc", "準備完了", "ok")
                    self._set_status_badge("experiment", "接続完了", "ok")
                    if not self._preview_enabled:
                        self.preview_label.configure(image="", text="プレビューOFF")
                        self._photo = None
                    else:
                        with self._resource_lock:
                            source = self._angle_source
                        if source is None or source.latest_overlay is None:
                            self.preview_label.configure(
                                image="", text="カメラ映像を取得中"
                            )
                    self.connect_button.configure(state="disabled")
                    self.start_button.configure(
                        state="normal"
                        if self.file_list.curselection() and self._selected_csv_valid
                        else "disabled"
                    )
                elif kind == "force":
                    self._consume_force_sample(event["sample"])
                elif kind == "angle":
                    self._consume_angle_sample(event)
                elif kind == "dlc_ready":
                    if event.get("worker") is self._standalone_dlc_worker:
                        self.status_var.set("DLC角度測定中（サーボ未接続）")
                        self.phase_var.set("DLC測定")
                        self._set_status_badge("camera", "接続済み", "ok")
                        self._set_status_badge("dlc", "測定中", "ok")
                elif kind == "dlc_error":
                    if event.get("worker") is self._standalone_dlc_worker:
                        self.status_var.set(f"DLC計測エラー: {event.get('error')}")
                        self._set_status_badge("camera", "エラー", "error")
                        self._set_status_badge("dlc", "エラー", "error")
                elif kind == "dlc_stopped":
                    worker = event.get("worker")
                    if worker is self._standalone_dlc_worker:
                        with self._resource_lock:
                            self._standalone_dlc_worker = None
                        self._dlc_enabled = False
                        self.dlc_toggle_button.configure(
                            state="disabled" if self._running else "normal",
                            text="DLC（session）" if self._running else "DLC ON",
                        )
                        if worker.error is None and not self._running:
                            self._set_status_badge("dlc", "未接続", "idle")
                            if not self._preview_enabled:
                                self._set_status_badge("camera", "未接続", "idle")
                        if self._preview_enabled and not self._running:
                            self._start_camera_preview_worker()
                elif kind == "worker_error":
                    self.status_var.set(f"エラー: {event.get('source')}: {event.get('error')}")
                    self.phase_var.set("異常終了")
                    self._set_status_badge("experiment", "異常終了", "error")
                    source = str(event.get("source", ""))
                    if "DLC" in source or "カメラ" in source:
                        self._set_status_badge("camera", "エラー", "error")
                        self._set_status_badge("dlc", "エラー", "error")
                    elif "servo" in source.lower() or "接続" in source:
                        self._set_status_badge("servo", "エラー", "error")
                        self._set_status_badge("encoder", "エラー", "error")
                elif kind == "hard_limit":
                    self.alert_label.configure(text=str(event.get("text", "")))
                    self.status_var.set(str(event.get("text", "")))
                    self.phase_var.set("ハードリミット停止")
                    self._set_status_badge("experiment", "ハードリミット停止", "error")
                elif kind == "result":
                    self._handle_result(event)
            except Exception as exc:
                self.status_var.set(f"GUI更新エラー: {type(exc).__name__}: {exc}")
                self.phase_var.set("GUI異常")
                self._set_status_badge("experiment", "GUI異常", "error")

        try:
            for event in self._latest_events.drain():
                kind = event.get("kind")
                if kind == "force":
                    self._consume_force_sample(event["sample"])
                elif kind == "angle":
                    self._consume_angle_sample(event)
            self._update_preview()
            if (
                (self._force_plot_dirty or self._angle_plot_dirty)
                and time.monotonic() - self._last_plot_update
                >= PLOT_UPDATE_INTERVAL_S
            ):
                self._update_plots()
                self._last_plot_update = time.monotonic()
        except Exception as exc:
            self.status_var.set(f"GUI描画エラー: {type(exc).__name__}: {exc}")
            self.phase_var.set("GUI描画異常")
            self._set_status_badge("experiment", "GUI描画異常", "error")
        self._schedule_event_poll()

    def _consume_force_sample(self, sample: dict[str, Any]) -> None:
        elapsed_s = float(sample["elapsed_s"])
        phase = str(sample["phase"])
        self.phase_var.set(phase)
        self.elapsed_var.set(f"経過時間: {elapsed_s:.2f} s")
        self._update_progress(elapsed_s, phase)
        readings = sample.get("force_readings", {})
        values: list[str] = []
        for tendon, servo_id in zip(TENDONS, SERVO_IDS_BY_TENDON):
            reading = readings.get(servo_id, readings.get(str(servo_id), {}))
            value = reading.get("force_n") if isinstance(reading, dict) else None
            if value is None:
                continue
            value = float(value)
            if math.isfinite(value):
                self.force_history[tendon].append((elapsed_s, value))
                values.append(f"{tendon}={value:.2f}N")
        self._force_plot_dirty = True
        self.force_var.set("Force: " + (", ".join(values) if values else "--"))

    def _consume_angle_sample(self, event: dict[str, Any]) -> None:
        elapsed_s = float(event["elapsed_s"])
        frame_index = int(event.get("frame_idx", -1))
        if frame_index == self._last_angle_frame:
            return
        self._last_angle_frame = frame_index
        self._min_likelihood_seen = min(
            self._min_likelihood_seen,
            float(event.get("min_likelihood", 0.0)),
        )
        for joint, value in zip(JOINTS, event["angles_deg"]):
            if math.isfinite(float(value)):
                self.angle_history[joint].append((elapsed_s, float(value)))
        self._angle_plot_dirty = True
        validity = "OK" if event.get("valid") else "INVALID"
        held = " held" if event.get("held_last") else ""
        self.camera_var.set(f"Camera: {elapsed_s:.2f}s")
        self.dlc_var.set(
            f"DLC: {validity}{held}, likelihood={float(event.get('min_likelihood', 0.0)):.2f}, "
            f"{float(event.get('inference_ms', 0.0)):.1f} ms"
        )
        self._set_status_badge(
            "camera",
            "稼働中" if event.get("valid") else "フレーム無効",
            "ok" if event.get("valid") else "warn",
        )
        self._set_status_badge(
            "dlc",
            "OK" if event.get("valid") else "尤度不足",
            "ok" if event.get("valid") else "warn",
        )

    def _update_preview(self) -> None:
        if not self._preview_enabled:
            return
        with self._resource_lock:
            source = self._angle_source
            camera_preview_worker = self._camera_preview_worker
            standalone_dlc_worker = self._standalone_dlc_worker
        if Image is None or ImageTk is None:
            return
        standalone_source = (
            standalone_dlc_worker.source
            if standalone_dlc_worker is not None
            and not standalone_dlc_worker.finished_event.is_set()
            else None
        )
        preview = (
            source.latest_preview
            if source is not None
            else standalone_source.latest_preview
            if standalone_source is not None
            else camera_preview_worker.latest_preview
            if camera_preview_worker is not None
            else None
        )
        if preview is None:
            if camera_preview_worker is not None and camera_preview_worker.error:
                self.preview_label.configure(
                    image="",
                    text=f"カメラ接続エラー\n{camera_preview_worker.error}",
                )
                self._set_status_badge("camera", "エラー", "error")
            return
        frame_index, is_overlay, frame = preview
        available_width = max(160, self.preview_label.winfo_width() - 12)
        available_height = max(120, self.preview_label.winfo_height() - 12)
        preview_size = (available_width, available_height)
        if (
            frame_index == self._last_preview_frame
            and is_overlay == self._last_preview_is_overlay
            and preview_size == self._last_preview_size
        ):
            return
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = rgb.shape[:2]
            max_width, max_height = available_width, available_height
            scale = min(max_width / width, max_height / height, 1.0)
            if scale < 1.0:
                rgb = cv2.resize(
                    rgb,
                    (max(1, int(width * scale)), max(1, int(height * scale))),
                    interpolation=cv2.INTER_AREA,
                )
            image = Image.fromarray(rgb)
            self._photo = ImageTk.PhotoImage(image=image)
            self.preview_label.configure(image=self._photo, text="")
            self._last_preview_size = preview_size
            self._last_preview_frame = frame_index
            self._last_preview_is_overlay = is_overlay
            if source is None and standalone_source is None:
                self._set_status_badge("camera", "プレビュー中", "ok")
        except Exception as exc:
            self.status_var.set(f"Preview更新エラー: {type(exc).__name__}: {exc}")
            self.preview_label.configure(image="", text="カメラ表示エラー")

    def _start_camera_preview_worker(self) -> None:
        with self._resource_lock:
            if self._angle_source is not None or self._running:
                return
            standalone_dlc_worker = self._standalone_dlc_worker
            if (
                standalone_dlc_worker is not None
                and not standalone_dlc_worker.finished_event.is_set()
            ):
                return
            current_worker = self._camera_preview_worker
            if current_worker is not None and current_worker.thread.is_alive():
                return
            worker = CameraPreviewWorker(Path(self.camera_config_var.get()))
            self._camera_preview_worker = worker
        worker.start()

    def _stop_camera_preview_worker(self, timeout_s: float = 2.0) -> bool:
        with self._resource_lock:
            worker = self._camera_preview_worker
        if worker is None:
            return True
        if not worker.stop(timeout_s):
            return False
        with self._resource_lock:
            if self._camera_preview_worker is worker:
                self._camera_preview_worker = None
        return True

    def _start_dlc_measurement_worker(self) -> bool:
        if self._running or self._angle_source is not None:
            return False
        if not self._stop_camera_preview_worker():
            messagebox.showwarning(
                "カメラ切替待ち",
                "プレビュー用カメラを解放できませんでした。少し待ってからDLC ONを押してください。",
            )
            return False
        with self._resource_lock:
            current_worker = self._standalone_dlc_worker
            if current_worker is not None and not current_worker.finished_event.is_set():
                return False
            source = RealtimeJointAngleSource(
                dlc_config_path=Path(self.dlc_config_var.get()),
                camera_config_path=Path(self.camera_config_var.get()),
                min_likelihood=0.3,
                temporal_outlier_enabled=True,
                temporal_jump_threshold_deg=30.0,
                temporal_max_hold_frames=5,
            )
            worker = DLCMeasurementWorker(source, self._emit)
            self._standalone_dlc_worker = worker
        self._last_angle_frame = -1
        self._min_likelihood_seen = 1.0
        for history in self.angle_history.values():
            history.clear()
        self._angle_plot_dirty = True
        self._last_preview_frame = -1
        self._last_preview_is_overlay = None
        self._last_preview_size = (0, 0)
        self._set_status_badge("camera", "接続中", "busy")
        self._set_status_badge("dlc", "準備中", "busy")
        if self._preview_enabled:
            self.preview_label.configure(image="", text="DLC計測を開始しています")
            self._photo = None
        try:
            worker.start()
        except Exception as exc:
            with self._resource_lock:
                if self._standalone_dlc_worker is worker:
                    self._standalone_dlc_worker = None
            self._set_status_badge("camera", "エラー", "error")
            self._set_status_badge("dlc", "エラー", "error")
            messagebox.showerror("DLC起動エラー", f"{type(exc).__name__}: {exc}")
            return False
        return True

    def _stop_dlc_measurement_worker(self, timeout_s: float = 2.0) -> bool:
        with self._resource_lock:
            worker = self._standalone_dlc_worker
        if worker is None:
            self._dlc_enabled = False
            return True
        if not worker.stop(timeout_s):
            return False
        self._dlc_enabled = False
        return True

    def _toggle_dlc(self) -> None:
        if self._running:
            return
        if self._dlc_enabled:
            self._dlc_enabled = False
            self.dlc_toggle_button.configure(text="DLC停止中", state="disabled")
            if not self._stop_dlc_measurement_worker():
                self.status_var.set("DLC停止待ち: 推論処理の終了を待っています")
            return
        if self._start_dlc_measurement_worker():
            self._dlc_enabled = True
            self.dlc_toggle_button.configure(text="DLC OFF", state="normal")

    def _toggle_preview(self) -> None:
        self._preview_enabled = not self._preview_enabled
        self._last_preview_frame = -1
        self._last_preview_is_overlay = None
        self._last_preview_size = (0, 0)
        if self._preview_enabled:
            self.preview_toggle_button.configure(text="プレビューOFF")
            with self._resource_lock:
                source = self._angle_source
                camera_preview_worker = self._camera_preview_worker
                standalone_dlc_worker = self._standalone_dlc_worker
            if (
                source is None
                and camera_preview_worker is None
                and standalone_dlc_worker is None
                and not self._running
            ):
                self._start_camera_preview_worker()
                with self._resource_lock:
                    camera_preview_worker = self._camera_preview_worker
            self.preview_label.configure(
                image="",
                text=(
                    "カメラ映像を待機中"
                    if (
                        source is not None
                        or camera_preview_worker is not None
                        or standalone_dlc_worker is not None
                        or self._running
                    )
                    else "カメラを開いています"
                ),
            )
            self._photo = None
            self._update_preview()
        else:
            self.preview_toggle_button.configure(text="プレビューON")
            self.preview_label.configure(image="", text="プレビューOFF")
            self._photo = None
            with self._resource_lock:
                source = self._angle_source
                standalone_dlc_worker = self._standalone_dlc_worker
            if (
                source is None
                and standalone_dlc_worker is None
                and not self._running
            ):
                self._stop_camera_preview_worker()
                self._set_status_badge("camera", "未接続", "idle")

    def _mark_force_plot_dirty(self, *_args) -> None:
        self._force_plot_dirty = True

    def _capture_force_plot_backgrounds(self, _event=None) -> None:
        self._force_plot_backgrounds = {
            tendon: self.force_canvas.copy_from_bbox(axis.bbox)
            for tendon, axis in zip(TENDONS, self.force_axes)
        }
        self._force_plot_dirty = True

    def _capture_angle_plot_background(self, _event=None) -> None:
        self._angle_plot_background = self.angle_canvas.copy_from_bbox(
            self.angle_axis.bbox
        )
        self._angle_plot_dirty = True

    def _blit_force_plot(self) -> bool:
        if len(self._force_plot_backgrounds) != len(TENDONS):
            return False
        for tendon, axis in zip(TENDONS, self.force_axes):
            self.force_canvas.restore_region(self._force_plot_backgrounds[tendon])
            axis.draw_artist(self.force_lines[tendon])
            axis.draw_artist(self.force_limit_lines[tendon])
            self.force_canvas.blit(axis.bbox)
        return True

    def _blit_angle_plot(self) -> bool:
        if self._angle_plot_background is None:
            return False
        self.angle_canvas.restore_region(self._angle_plot_background)
        for line in self.angle_lines.values():
            self.angle_axis.draw_artist(line)
        self.angle_canvas.blit(self.angle_axis.bbox)
        return True

    def _update_plots(self) -> None:
        if self._force_plot_dirty:
            self._update_force_plot()
        if self._angle_plot_dirty:
            self._update_angle_plot()

    def _update_force_plot(self) -> None:
        latest = max(
            (history[-1][0] for history in self.force_history.values() if history),
            default=0.0,
        )
        visible_start = max(0.0, latest - 20.0)
        desired_xlim = (visible_start, max(20.0, latest))
        desired_ylims: dict[str, tuple[float, float]] = {}
        urgent_rescale = False

        for tendon, axis in zip(TENDONS, self.force_axes):
            visible = [
                item
                for item in self.force_history[tendon]
                if item[0] >= visible_start
            ]
            times = [item[0] for item in visible]
            forces = [item[1] for item in visible]
            self.force_lines[tendon].set_data(times, forces)
            low = min(0.0, min(forces) * 1.1) if forces else 0.0
            high = max(1.0, max(forces) * 1.1) if forces else 1.0
            try:
                limit = float(self.limit_vars[tendon].get())
            except (TypeError, ValueError):
                limit = None
            if limit is not None and math.isfinite(limit) and limit > 0.0:
                self.force_limit_lines[tendon].set_data(
                    [desired_xlim[0], desired_xlim[1]], [limit, limit]
                )
                high = max(high, limit * 1.1)
            else:
                self.force_limit_lines[tendon].set_data([], [])
            desired_ylims[tendon] = (low, high)
            current_low, current_high = axis.get_ylim()
            urgent_rescale = urgent_rescale or low < current_low or high > current_high

        now = time.monotonic()
        limits_changed = any(
            tuple(axis.get_xlim()) != desired_xlim
            or tuple(axis.get_ylim()) != desired_ylims[tendon]
            for tendon, axis in zip(TENDONS, self.force_axes)
        )
        full_draw = (
            len(self._force_plot_backgrounds) != len(TENDONS)
            or urgent_rescale
            or (
                limits_changed
                and now - self._force_last_full_draw_at
                >= PLOT_AXIS_REFRESH_INTERVAL_S
            )
        )
        if full_draw:
            for tendon, axis in zip(TENDONS, self.force_axes):
                axis.set_xlim(*desired_xlim)
                axis.set_ylim(*desired_ylims[tendon])
            self.force_canvas.draw()
            self._force_last_full_draw_at = now
        self._blit_force_plot()
        self._force_plot_dirty = False

    def _update_angle_plot(self) -> None:
        latest = max(
            (history[-1][0] for history in self.angle_history.values() if history),
            default=0.0,
        )
        visible_start = max(0.0, latest - 20.0)
        desired_xlim = (visible_start, max(20.0, latest))
        all_angles: list[float] = []
        for joint in JOINTS:
            visible = [
                item
                for item in self.angle_history[joint]
                if item[0] >= visible_start
            ]
            times = [item[0] for item in visible]
            angles = [item[1] for item in visible]
            self.angle_lines[joint].set_data(times, angles)
            all_angles.extend(angles)
        if all_angles:
            low = min(all_angles)
            high = max(all_angles)
            margin = max(5.0, (high - low) * 0.15)
            desired_ylim = (low - margin, high + margin)
        else:
            desired_ylim = (0.0, 180.0)

        current_low, current_high = self.angle_axis.get_ylim()
        urgent_rescale = (
            min(all_angles, default=current_low) < current_low
            or max(all_angles, default=current_high) > current_high
        )
        now = time.monotonic()
        limits_changed = (
            tuple(self.angle_axis.get_xlim()) != desired_xlim
            or tuple(self.angle_axis.get_ylim()) != desired_ylim
        )
        full_draw = (
            self._angle_plot_background is None
            or urgent_rescale
            or (
                limits_changed
                and now - self._angle_last_full_draw_at
                >= PLOT_AXIS_REFRESH_INTERVAL_S
            )
        )
        if full_draw:
            self.angle_axis.set_xlim(*desired_xlim)
            self.angle_axis.set_ylim(*desired_ylim)
            self.angle_canvas.draw()
            self._angle_last_full_draw_at = now
        self._blit_angle_plot()
        self._angle_plot_dirty = False

    def _handle_result(self, event: dict[str, Any]) -> None:
        self._latest_events.clear()
        self._running = False
        self._runner_thread = None
        self.stop_button.configure(state="disabled")
        self.force_quit_button.configure(state="disabled")
        self._set_input_widgets_enabled(True)
        self._connection_ready = False
        self._playback_started = False
        self._set_status_badge("servo", "未接続", "idle")
        self._set_status_badge("encoder", "未接続", "idle")
        self._set_status_badge("camera", "未接続", "idle")
        self._set_status_badge("dlc", "未接続", "idle")
        self.dlc_toggle_button.configure(state="normal", text="DLC ON")
        self.preview_label.configure(
            image="",
            text="カメラ映像を待機中" if self._preview_enabled else "カメラ未接続",
        )
        self._photo = None
        self._last_preview_frame = -1
        self._last_preview_is_overlay = None
        self._last_preview_size = (0, 0)
        if self._preview_enabled:
            self._start_camera_preview_worker()
        status = str(event.get("status", "error"))
        session_dir = Path(str(event["session_dir"]))
        self._session_dir = session_dir
        if status == "force_hard_limit":
            self.alert_label.configure(text="Forceハードリミットで停止しました。ログを確認してください")
        elif status == "force_telemetry_unavailable":
            self.alert_label.configure(text="Forceテレメトリ不明のため停止しました。ログを確認してください")
        max_force_parts = []
        for tendon in TENDONS:
            values = [value for _, value in self.force_history[tendon]]
            if values:
                max_force_parts.append(f"{tendon}={max(values):.2f}")
        display_status = {
            "completed": "完了",
            "operator_stop": "手動停止",
            "force_quit": "FORCE QUIT",
            "force_hard_limit": "HARD LIMIT STOP",
            "force_telemetry_unavailable": "FORCE SAFETY STOP",
            "error": "異常終了",
        }.get(status, status)
        result_state = "ok" if status == "completed" else (
            "warn" if status in {"operator_stop", "force_quit"} else "error"
        )
        self._set_status_badge("experiment", display_status, result_state)
        self.result_tree.insert(
            "",
            0,
            values=(
                display_status,
                ", ".join(max_force_parts) or "--",
                f"{self._min_likelihood_seen:.3f}",
                self._shorten_path(session_dir),
            ),
        )
        for item_id in self.result_tree.get_children()[MAX_RESULT_ROWS:]:
            self.result_tree.delete(item_id)
        self._stop_event = None
        self._start_event = None
        self._force_quit_event = None
        with self._resource_lock:
            self._pending_playback_config = None
        self._start_in_progress = False
        self.connect_button.configure(
            state="normal" if self.file_list.curselection() else "disabled"
        )
        self.start_button.configure(state="disabled")
        self.open_session_button.configure(state="normal")
        if status == "completed":
            self.progress_bar.configure(value=100.0)
            self.progress_var.set("進捗: 完了")
        elif status == "operator_stop":
            self.progress_var.set("進捗: 手動停止")
        elif status == "force_quit":
            self.progress_var.set("進捗: Force Quit")
        else:
            self.progress_var.set(f"進捗: {status}")
        if not self._closing_requested:
            keep = messagebox.askyesno(
                "結果保存",
                f"セッション結果を保存しますか？\n{session_dir}",
            )
            if not keep and status not in {
                "force_hard_limit",
                "force_telemetry_unavailable",
            }:
                try:
                    shutil.rmtree(session_dir)
                    self.status_var.set("結果を破棄しました。診断ログも削除されています")
                except Exception as exc:
                    self.status_var.set(f"結果の破棄に失敗: {exc}")
            elif not keep:
                self.status_var.set("Force保護停止の結果は安全のため保持しました")
            else:
                self.status_var.set(f"結果を保存しました: {session_dir}")
        if self._closing_requested:
            self.after_idle(self._destroy_after_cleanup)

    def _destroy_after_cleanup(self) -> None:
        self._stop_dlc_measurement_worker()
        self._stop_camera_preview_worker()
        if self._poll_after_id is not None:
            try:
                self.after_cancel(self._poll_after_id)
            except tk.TclError:
                pass
            self._poll_after_id = None
        self.destroy()

    def _on_close(self) -> None:
        if self._running:
            if not messagebox.askyesno(
                "実行中",
                "ソフトウェア停止してGUIを閉じますか？",
            ):
                return
            self._closing_requested = True
            self._request_stop()
            return
        self._destroy_after_cleanup()


def main() -> None:
    app = CsvPlayerApp()
    app.mainloop()


if __name__ == "__main__":
    main()
