"""Tkinter GUI for CSV trajectory playback with DLC and force monitoring.

The GUI owns the presentation layer only.  Servo control, camera inference,
and serial readers run on worker threads so that plotting never blocks the
control loop.
"""

from __future__ import annotations

import csv
import json
import math
import queue
import shutil
import threading
import time
import tkinter as tk
from collections import deque
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

from controller.csv_player.force_estimator import CalibratedForceEstimator
from controller.csv_player.recorder import ControlTraceWriter
from controller.csv_player.runner import (
    ForceTelemetryUnavailable,
    HardForceLimitError,
    PlaybackConfig,
    PlaybackStopped,
    SpeedTrajectoryRunner,
)
from controller.csv_player.speed_controller import SpeedControlConfig, SpeedController
from controller.csv_player.trajectory import ExcursionTrajectory
from controller.servo.hardware_session import API_ROOT, ServoHardwareSession
from controller.servo_mapping import (
    ENCODER_SENSOR_IDS_BY_SERVO_ID,
    SERVO_IDS,
    SERVO_IDS_BY_TENDON,
    SERVO_NAMES_BY_ID,
    TENDONS,
)
from observation.camera.camera_param_resolver import resolve_param_path
from observation.vision.realtime_joint_angle_source import (
    JOINTS,
    Meta1CalibrationAbortedError,
    RealtimeJointAngleSource,
)
from utils.path_utils import resolve_path


PROJECT_ROOT = Path(__file__).resolve().parents[3]
SRC_ROOT = PROJECT_ROOT / "src"
CONTROLLER_ROOT = SRC_ROOT / "controller"
DEFAULT_CSV_DIRECTORY = PROJECT_ROOT / "logs" / "dual_camera" / "excursion_predictions"
DEFAULT_OUTPUT_DIRECTORY = PROJECT_ROOT / "logs" / "csv_player_gui"
DEFAULT_CALIBRATION_PATH = CONTROLLER_ROOT / "excursion_servo_calibration.json"
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
        run_started_at: float,
        interval_s: float = 1.0 / DEFAULT_VISION_HZ,
    ) -> None:
        self.source = source
        self.emit = emit
        self.stop_event = stop_event
        self.output_csv = output_csv
        self.run_started_at = run_started_at
        self.interval_s = max(0.02, float(interval_s))

    def _emit(self, event: dict[str, Any]) -> None:
        self.emit(event)

    def run(self) -> None:
        frame_index = 0
        try:
            with self.output_csv.open("w", encoding="utf-8", newline="") as handle:
                writer = csv.writer(handle)
                writer.writerow(
                    [
                        "frame_idx",
                        "elapsed_s",
                        "captured_at_monotonic",
                        "inference_ms",
                        "min_likelihood",
                        "valid",
                        "held_last",
                        "reason",
                        *[f"{joint}_deg" for joint in JOINTS],
                    ]
                )
                while not self.stop_event.is_set():
                    started_at = time.monotonic()
                    measurement = self.source.read(frame_index)
                    angles_deg = tuple(
                        math.degrees(value)
                        for value in measurement.flexion_angles_rad
                    )
                    writer.writerow(
                        [
                            measurement.frame_index,
                            f"{measurement.captured_at - self.run_started_at:.6f}",
                            f"{measurement.captured_at:.6f}",
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
                            "elapsed_s": measurement.captured_at - self.run_started_at,
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


class CsvPlayerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.title("CSV Player | DLC and Force Monitor")
        self.geometry("1500x980")
        self.minsize(1180, 780)
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.events: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=400)
        self._critical_events: queue.Queue[dict[str, Any]] = queue.Queue(maxsize=32)
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
        self._resource_lock = threading.Lock()
        self._angle_source: RealtimeJointAngleSource | None = None
        self._session_dir: Path | None = None
        self._closing_requested = False
        self._start_in_progress = False
        self._running = False
        self._last_angle_frame = -1
        self._min_likelihood_seen = 1.0
        self._last_plot_update = 0.0
        self._last_preview_frame = -1
        self._poll_after_id: str | None = None
        self._photo = None

        self.folder_var = tk.StringVar(value=str(DEFAULT_CSV_DIRECTORY))
        self.selected_var = tk.StringVar(value="CSV未選択")
        self.detail_var = tk.StringVar(value="CSVを選択してください")
        self.status_var = tk.StringVar(value="待機中")
        self.phase_var = tk.StringVar(value="未実行")
        self.camera_var = tk.StringVar(value="Camera: --")
        self.dlc_var = tk.StringVar(value="DLC: --")
        self.force_var = tk.StringVar(value="Force: --")
        self.elapsed_var = tk.StringVar(value="経過時間: --")
        self.servo_port_var = tk.StringVar(value=DEFAULT_SERVO_PORT)
        self.encoder_port_var = tk.StringVar(value=DEFAULT_ENCODER_PORT)
        self.dlc_config_var = tk.StringVar(value=str(DEFAULT_DLC_CONFIG_PATH))
        self.camera_config_var = tk.StringVar(value=str(DEFAULT_CAMERA_CONFIG_PATH))
        self.limit_vars = {
            tendon: tk.StringVar(value="--") for tendon in TENDONS
        }

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
        self.columnconfigure(0, weight=1)

        files = ttk.LabelFrame(self, text="CSV選択")
        files.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        files.columnconfigure(1, weight=1)
        ttk.Label(files, text="配置ディレクトリ").grid(row=0, column=0, padx=4, pady=4)
        ttk.Entry(files, textvariable=self.folder_var).grid(
            row=0, column=1, sticky="ew", padx=4, pady=4
        )
        ttk.Button(files, text="参照", command=self._choose_folder).grid(
            row=0, column=2, padx=4, pady=4
        )
        ttk.Button(files, text="更新", command=self._refresh_files).grid(
            row=0, column=3, padx=4, pady=4
        )

        selection = ttk.Frame(self)
        selection.grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        selection.columnconfigure(0, weight=1)
        selection.columnconfigure(1, weight=2)
        self.file_list = tk.Listbox(selection, height=5, exportselection=False)
        self.file_list.grid(row=0, column=0, sticky="ew", padx=(0, 6))
        self.file_list.bind("<<ListboxSelect>>", self._on_select_csv)
        ttk.Label(
            selection,
            textvariable=self.detail_var,
            anchor="nw",
            justify="left",
        ).grid(row=0, column=1, sticky="nsew", padx=6)

        config = ttk.LabelFrame(self, text="接続と安全上限")
        config.grid(row=2, column=0, sticky="ew", padx=8, pady=4)
        for column in range(12):
            config.columnconfigure(column, weight=1 if column in (1, 3, 5, 7, 9, 11) else 0)
        self._entry(config, "Servo", self.servo_port_var, 0, 0, width=9)
        self._entry(config, "Encoder", self.encoder_port_var, 0, 2, width=9)
        self._entry(config, "DLC config", self.dlc_config_var, 0, 4, width=38)
        self._entry(config, "Camera config", self.camera_config_var, 0, 6, width=38)
        ttk.Label(config, text="Hard limit [N]").grid(
            row=1, column=0, padx=4, pady=4, sticky="w"
        )
        for index, tendon in enumerate(TENDONS):
            column = 1 + index * 2
            ttk.Label(config, text=tendon).grid(row=1, column=column, padx=(4, 1), pady=4)
            ttk.Entry(config, textvariable=self.limit_vars[tendon], width=7).grid(
                row=1, column=column + 1, padx=(0, 4), pady=4
            )

        controls = ttk.Frame(self)
        controls.grid(row=3, column=0, sticky="ew", padx=8, pady=4)
        ttk.Label(controls, textvariable=self.selected_var).pack(side="left", padx=4)
        self.start_button = ttk.Button(controls, text="実行", command=self._start)
        self.start_button.pack(side="left", padx=4)
        self.stop_button = ttk.Button(
            controls, text="安全停止", command=self._request_stop, state="disabled"
        )
        self.stop_button.pack(side="left", padx=4)
        ttk.Label(controls, textvariable=self.status_var).pack(side="left", padx=12)
        self.alert_label = ttk.Label(controls, text="", foreground="#b00020")
        self.alert_label.pack(side="right", padx=8)

        status = ttk.LabelFrame(self, text="実行状態")
        status.grid(row=4, column=0, sticky="ew", padx=8, pady=4)
        for index, variable in enumerate(
            (self.phase_var, self.elapsed_var, self.camera_var, self.dlc_var, self.force_var)
        ):
            ttk.Label(status, textvariable=variable).grid(
                row=0, column=index, sticky="w", padx=8, pady=4
            )

        body = ttk.PanedWindow(self, orient="horizontal")
        body.grid(row=5, column=0, sticky="nsew", padx=8, pady=(4, 8))
        self.rowconfigure(5, weight=1)
        preview_frame = ttk.LabelFrame(body, text="Camera / DLC preview")
        plots_frame = ttk.Notebook(body)
        body.add(preview_frame, weight=1)
        body.add(plots_frame, weight=3)
        preview_frame.columnconfigure(0, weight=1)
        preview_frame.rowconfigure(0, weight=1)
        self.preview_label = ttk.Label(preview_frame, text="カメラ未接続", anchor="center")
        self.preview_label.grid(row=0, column=0, sticky="nsew", padx=4, pady=4)

        force_tab = ttk.Frame(plots_frame)
        angle_tab = ttk.Frame(plots_frame)
        plots_frame.add(force_tab, text="推定Force")
        plots_frame.add(angle_tab, text="DLC角度")
        self._build_force_plot(force_tab)
        self._build_angle_plot(angle_tab)

        results = ttk.LabelFrame(self, text="実行結果一覧")
        results.grid(row=6, column=0, sticky="ew", padx=8, pady=(0, 8))
        results.columnconfigure(0, weight=1)
        self.result_tree = ttk.Treeview(
            results,
            columns=("status", "max_force", "likelihood", "session"),
            show="headings",
            height=3,
        )
        headings = {
            "status": "状態",
            "max_force": "最大Force [N]",
            "likelihood": "最低DLC尤度",
            "session": "セッション",
        }
        for column, heading in headings.items():
            self.result_tree.heading(column, text=heading)
            self.result_tree.column(column, width=150 if column != "session" else 420)
        self.result_tree.grid(row=0, column=0, sticky="ew", padx=4, pady=4)

    @staticmethod
    def _entry(parent: ttk.Widget, label: str, variable: tk.StringVar, row: int, column: int, *, width: int = 18) -> None:
        ttk.Label(parent, text=label).grid(row=row, column=column, padx=(4, 1), pady=4, sticky="w")
        ttk.Entry(parent, textvariable=variable, width=width).grid(
            row=row, column=column + 1, padx=(0, 4), pady=4, sticky="ew"
        )

    def _build_force_plot(self, parent: ttk.Frame) -> None:
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)
        self.force_figure = Figure(figsize=(10, 6), dpi=100)
        self.force_axes = self.force_figure.subplots(2, 3).flatten()
        for axis, tendon in zip(self.force_axes, TENDONS):
            line, = axis.plot([], [], label="推定Force")
            limit_line = axis.axhline(0.0, color="#b00020", linestyle="--", label="Hard limit")
            self.force_lines[tendon] = line
            self.force_limit_lines[tendon] = limit_line
            axis.set_title(tendon)
            axis.set_xlabel("Time [s]")
            axis.set_ylabel("N")
            axis.grid(True, alpha=0.3)
        self.force_canvas = FigureCanvasTkAgg(self.force_figure, master=parent)
        self.force_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def _build_angle_plot(self, parent: ttk.Frame) -> None:
        parent.rowconfigure(0, weight=1)
        parent.columnconfigure(0, weight=1)
        self.angle_figure = Figure(figsize=(10, 6), dpi=100)
        self.angle_axes = self.angle_figure.subplots(3, 1, sharex=True)
        for axis, joint in zip(self.angle_axes, JOINTS):
            line, = axis.plot([], [], label="DLC measured")
            self.angle_lines[joint] = line
            axis.set_title(joint)
            axis.set_ylabel("deg")
            axis.grid(True, alpha=0.3)
            axis.legend(loc="best")
        self.angle_axes[-1].set_xlabel("Time [s]")
        self.angle_canvas = FigureCanvasTkAgg(self.angle_figure, master=parent)
        self.angle_canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

    def _choose_folder(self) -> None:
        selected = filedialog.askdirectory(initialdir=self.folder_var.get())
        if selected:
            self.folder_var.set(selected)
            self._refresh_files()

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
            self.selected_var.set("CSV未選択")
            self.detail_var.set(f"CSV一覧を読めません: {exc}")
            self.start_button.configure(state="disabled")
            return
        self.csv_paths = [path for _, path in sorted(paths, reverse=True)]
        for path in self.csv_paths:
            self.file_list.insert(tk.END, path.name)
        if self.csv_paths:
            self.file_list.selection_set(0)
            self._on_select_csv()
        else:
            self.selected_var.set("CSV未選択")
            self.detail_var.set(f"CSVがありません: {folder}")

    def _on_select_csv(self, _event: object | None = None) -> None:
        selection = self.file_list.curselection()
        if not selection:
            return
        path = self.csv_paths[selection[0]]
        self.selected_var.set(str(path))
        try:
            calibration = self._load_position_calibration()
            trajectory = ExcursionTrajectory.from_csv(path, calibration)
            self.detail_var.set(
                f"ファイル: {path.name}\n"
                f"行数: {len(trajectory.elapsed_times_s)}\n"
                f"再生時間: {trajectory.duration_s:.3f} s\n"
                f"状態: 実行可能"
            )
            self.start_button.configure(
                state="normal"
                if not self._running and not self._start_in_progress
                else "disabled"
            )
        except Exception as exc:
            self.detail_var.set(f"検証エラー: {type(exc).__name__}: {exc}")
            self.start_button.configure(state="disabled")

    def _load_position_calibration(self) -> tuple[float, ...]:
        from controller.csv_player.excursion_player import load_position_calibration

        return load_position_calibration(DEFAULT_CALIBRATION_PATH)

    def _parse_limits(self) -> dict[str, float]:
        limits: dict[str, float] = {}
        for tendon in TENDONS:
            value = float(self.limit_vars[tendon].get())
            if not math.isfinite(value) or value <= 0.0:
                raise ValueError(f"{tendon}のHard limitは正の有限値が必要です")
            limits[tendon] = value
        return limits

    def _start(self) -> None:
        if self._running or self._start_in_progress:
            return
        self._start_in_progress = True
        self.start_button.configure(state="disabled")
        selection = self.file_list.curselection()
        if not selection:
            messagebox.showwarning("CSV未選択", "再生するCSVを選択してください")
            self._start_in_progress = False
            self.start_button.configure(state="disabled")
            return
        path = self.csv_paths[selection[0]]
        try:
            limits = self._parse_limits()
            trajectory = ExcursionTrajectory.from_csv(path, self._load_position_calibration())
        except Exception as exc:
            messagebox.showerror("事前検証エラー", str(exc))
            self._start_in_progress = False
            self.start_button.configure(state="normal")
            return
        if not messagebox.askyesno(
            "実行確認",
            f"{path.name}\n再生時間: {trajectory.duration_s:.3f} s\n"
            "サーボを動かします。実行しますか？",
        ):
            self._start_in_progress = False
            self.start_button.configure(state="normal")
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
            self.start_button.configure(state="normal")
            return
        self._session_dir = session_dir
        stop_event = threading.Event()
        self._stop_event = stop_event
        self._running = True
        self._last_angle_frame = -1
        self._last_preview_frame = -1
        self._min_likelihood_seen = 1.0
        for history in self.force_history.values():
            history.clear()
        for history in self.angle_history.values():
            history.clear()
        self.alert_label.configure(text="")
        self.status_var.set("準備中")
        self.phase_var.set("準備中")
        self.start_button.configure(state="disabled")
        self.stop_button.configure(state="normal")
        self._runner_thread = threading.Thread(
            target=self._run_session,
            args=(
                path,
                trajectory,
                limits,
                session_dir,
                dlc_config_path,
                camera_config_path,
                servo_port,
                encoder_port,
                stop_event,
            ),
            name="csv-player-gui-session",
            daemon=True,
        )
        self._runner_thread.start()
        self._start_in_progress = False

    def _run_session(
        self,
        csv_path: Path,
        trajectory: ExcursionTrajectory,
        limits: dict[str, float],
        session_dir: Path,
        dlc_config_path: Path,
        camera_config_path: Path,
        servo_port: str,
        encoder_port: str,
        stop_event: threading.Event,
    ) -> None:
        hardware: ServoHardwareSession | None = None
        source: RealtimeJointAngleSource | None = None
        trace: ControlTraceWriter | None = None
        vision_worker: VisionWorker | None = None
        vision_thread: threading.Thread | None = None
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
            self._emit({"kind": "status", "text": "DLCキャリブレーション中", "phase": "DLC準備"})
            source.calibrate_meta1(stop_event=stop_event)
            if stop_event.is_set():
                raise PlaybackStopped("Playback was stopped during DLC calibration")

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
            if stop_event.is_set():
                raise PlaybackStopped("Playback was stopped before hardware connection")
            hardware.connect(startup_timeout_s=3.0)
            if stop_event.is_set():
                raise PlaybackStopped("Playback was stopped during hardware connection")
            feedback = hardware.wait_for_feedback(3.0, require_encoder=True)
            if stop_event.is_set():
                raise PlaybackStopped("Playback was stopped while waiting for telemetry")
            start_positions = hardware.position_by_servo_id(feedback)
            trace = ControlTraceWriter(session_dir / "control_trace.csv", SERVO_IDS_BY_TENDON)
            force_estimator = CalibratedForceEstimator(
                calibration_dir=DEFAULT_FORCE_CALIBRATION_DIR,
                zero_offset_path=DEFAULT_ZERO_OFFSET_PATH,
                servo_ids=SERVO_IDS_BY_TENDON,
                tendon_by_servo_id=dict(zip(SERVO_IDS_BY_TENDON, TENDONS)),
                telemetry_stale_after_s=0.2,
            )

            run_started_at = time.monotonic()
            video_path = session_dir / "dlc_overlay.mp4"
            source.start_overlay_recording(video_path, fps=DEFAULT_VISION_HZ, codec="mp4v", queue_size=10)
            vision_worker = VisionWorker(
                source,
                self._emit,
                stop_event,
                session_dir / "angle_trace.csv",
                run_started_at,
            )
            vision_thread = threading.Thread(
                target=vision_worker.run,
                name="csv-player-gui-dlc",
                daemon=True,
            )
            vision_thread.start()
            self._emit({"kind": "status", "text": "実行中", "phase": "実行中"})

            runner = SpeedTrajectoryRunner(
                trajectory=trajectory,
                servo_ids_by_tendon=SERVO_IDS_BY_TENDON,
                hardware=hardware,
                speed_controller=SpeedController(
                    SpeedControlConfig(
                        kp=0.5,
                        max_speed=500,
                        max_acceleration=3000.0,
                        position_deadband=3.0,
                    )
                ),
                config=PlaybackConfig(
                    control_hz=DEFAULT_CONTROL_HZ,
                    telemetry_stale_s=0.2,
                    telemetry_wait_s=3.0,
                    display_interval_s=0.25,
                    return_to_start=True,
                    return_tolerance=10.0,
                    return_timeout_s=5.0,
                    return_stable_frames=3,
                    hard_force_limits_n_by_tendon=limits,
                ),
                recorder=trace,
                force_estimator=force_estimator,
                on_sample=lambda sample: self._emit({"kind": "force", "sample": sample}),
                stop_event=stop_event,
            )
            runner.run(start_positions)
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
        except (PlaybackStopped, Meta1CalibrationAbortedError) as exc:
            termination_reason = "operator_stop"
            error_text = str(exc)
            self._emit({"kind": "status", "text": "安全停止しました", "phase": "停止"})
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
            if trace is not None:
                try:
                    trace.close()
                except Exception as exc:
                    cleanup_errors.append(f"trace close: {type(exc).__name__}: {exc}")
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
                "created_at": datetime.now().isoformat(timespec="seconds"),
                "input_csv_path": str(csv_path.resolve()),
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
            self._emit(
                {
                    "kind": "result",
                    "status": termination_reason,
                    "session_dir": str(session_dir),
                    "error": error_text,
                }
            )

    def _emit(self, event: dict[str, Any]) -> None:
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
        self._stop_event.set()
        self.stop_button.configure(state="disabled")

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
                    self.status_var.set(str(event.get("text", "")))
                    self.phase_var.set(str(event.get("phase", "")))
                elif kind == "force":
                    self._consume_force_sample(event["sample"])
                elif kind == "angle":
                    self._consume_angle_sample(event)
                elif kind == "worker_error":
                    self.status_var.set(f"エラー: {event.get('source')}: {event.get('error')}")
                    self.phase_var.set("異常終了")
                elif kind == "hard_limit":
                    self.alert_label.configure(text=str(event.get("text", "")))
                    self.status_var.set(str(event.get("text", "")))
                    self.phase_var.set("ハードリミット停止")
                elif kind == "result":
                    self._handle_result(event)
            except Exception as exc:
                self.status_var.set(f"GUI更新エラー: {type(exc).__name__}: {exc}")
                self.phase_var.set("GUI異常")

        try:
            self._update_preview()
            if time.monotonic() - self._last_plot_update >= 0.2:
                self._update_plots()
                self._last_plot_update = time.monotonic()
        except Exception as exc:
            self.status_var.set(f"GUI描画エラー: {type(exc).__name__}: {exc}")
            self.phase_var.set("GUI描画異常")
        self._schedule_event_poll()

    def _consume_force_sample(self, sample: dict[str, Any]) -> None:
        elapsed_s = float(sample["elapsed_s"])
        self.phase_var.set(str(sample["phase"]))
        self.elapsed_var.set(f"経過時間: {elapsed_s:.2f} s")
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
        validity = "OK" if event.get("valid") else "INVALID"
        held = " held" if event.get("held_last") else ""
        self.camera_var.set(f"Camera: {elapsed_s:.2f}s")
        self.dlc_var.set(
            f"DLC: {validity}{held}, likelihood={float(event.get('min_likelihood', 0.0)):.2f}, "
            f"{float(event.get('inference_ms', 0.0)):.1f} ms"
        )

    def _update_preview(self) -> None:
        with self._resource_lock:
            source = self._angle_source
        if source is None or Image is None or ImageTk is None:
            return
        measurement = source.latest_measurement
        if measurement is not None and measurement.frame_index == self._last_preview_frame:
            return
        frame = source.latest_overlay
        if frame is None:
            return
        try:
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            height, width = rgb.shape[:2]
            max_width, max_height = 760, 560
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
            if measurement is not None:
                self._last_preview_frame = measurement.frame_index
        except Exception as exc:
            self.status_var.set(f"Preview更新エラー: {type(exc).__name__}: {exc}")
            self.preview_label.configure(image="", text="カメラ表示エラー")

    def _update_plots(self) -> None:
        for tendon, axis in zip(TENDONS, self.force_axes):
            values = list(self.force_history[tendon])
            if not values:
                continue
            times = [item[0] for item in values]
            forces = [item[1] for item in values]
            self.force_lines[tendon].set_data(times, forces)
            try:
                limit = float(self.limit_vars[tendon].get())
            except (TypeError, ValueError):
                continue
            if not math.isfinite(limit) or limit <= 0.0:
                continue
            self.force_limit_lines[tendon].set_ydata([limit, limit])
            axis.set_xlim(max(0.0, times[-1] - 20.0), max(20.0, times[-1]))
            low = min(0.0, min(forces) * 1.1)
            high = max(limit * 1.1, max(forces) * 1.1, 1.0)
            axis.set_ylim(low, high)
        self.force_canvas.draw_idle()

        for joint, axis in zip(JOINTS, self.angle_axes):
            values = list(self.angle_history[joint])
            if not values:
                continue
            times = [item[0] for item in values]
            angles = [item[1] for item in values]
            self.angle_lines[joint].set_data(times, angles)
            axis.set_xlim(max(0.0, times[-1] - 20.0), max(20.0, times[-1]))
            low = min(angles)
            high = max(angles)
            margin = max(5.0, (high - low) * 0.15)
            axis.set_ylim(low - margin, high + margin)
        self.angle_canvas.draw_idle()

    def _handle_result(self, event: dict[str, Any]) -> None:
        self._running = False
        self._runner_thread = None
        self.stop_button.configure(state="disabled")
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
            "force_hard_limit": "HARD LIMIT STOP",
            "force_telemetry_unavailable": "FORCE SAFETY STOP",
            "error": "異常終了",
        }.get(status, status)
        self.result_tree.insert(
            "",
            0,
            values=(
                display_status,
                ", ".join(max_force_parts) or "--",
                f"{self._min_likelihood_seen:.3f}",
                str(session_dir),
            ),
        )
        for item_id in self.result_tree.get_children()[MAX_RESULT_ROWS:]:
            self.result_tree.delete(item_id)
        self._stop_event = None
        self._start_in_progress = False
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
                self.status_var.set("Force安全停止の結果は安全のため保持しました")
            else:
                self.status_var.set(f"結果を保存しました: {session_dir}")
        if self._closing_requested:
            self.after_idle(self._destroy_after_cleanup)
        else:
            self.start_button.configure(state="normal" if self.file_list.curselection() else "disabled")

    def _destroy_after_cleanup(self) -> None:
        if self._poll_after_id is not None:
            try:
                self.after_cancel(self._poll_after_id)
            except tk.TclError:
                pass
            self._poll_after_id = None
        self.destroy()

    def _on_close(self) -> None:
        if self._running:
            if not messagebox.askyesno("実行中", "安全停止してGUIを閉じますか？"):
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
