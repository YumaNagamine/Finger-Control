from __future__ import annotations

import logging
import os
import time
import tkinter as tk
from pathlib import Path
from tkinter import ttk

from servo.servoGUI_api.app_state import AppState
from servo.servoGUI_api.event_bus import EventBus
from servo.core.session import ServoSession
from servo.servoGUI_api.services.control_service import ControlService
from servo.servoGUI_api.services.id_service import IDService
from servo.servoGUI_api.services.record_service import RecordService
from servo.servoGUI_api.ui.tabs.batch_control_tab import BatchControlTab
from servo.servoGUI_api.ui.tabs.high_accuracy_pos_tab import HighAccuracyPositionTab
from servo.servoGUI_api.ui.tabs.id_manager_tab import IDManagerTab
from servo.servoGUI_api.ui.tabs.manual_control_tab import ManualControlTab
from servo.servoGUI_api.ui.tabs.monitor_all_tab import MonitorAllTab
from servo.servoGUI_api.ui.tabs.motion_recorder_tab import MotionRecorderTab
from servo.servoGUI_api.ui.tabs.plotter_tab import PlotterTab


def _env_truthy(name: str) -> bool:
    value = os.getenv(name, "").strip().lower()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1].strip()
    return value in {"1", "true", "yes", "on"}


def _build_logger() -> logging.Logger:
    logger = logging.getLogger("servoGUI_api")
    if logger.handlers:
        return logger

    logger.setLevel(logging.INFO)
    log_dir = Path("logs")
    log_dir.mkdir(parents=True, exist_ok=True)
    log_path = log_dir / "servo_gui_api.log"

    formatter = logging.Formatter("%(asctime)s %(levelname)s %(message)s")

    file_handler = logging.FileHandler(log_path, encoding="utf-8")
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(formatter)
    logger.addHandler(stream_handler)

    logger.info("Logger initialized: %s", log_path)
    return logger


class ServoControlGUIAPI:
    NUM_MOTORS = 6
    SPEED_PRESETS = [-2000, -1000, -500, 0, 500, 1000, 2000]
    GLOBAL_SPEED_PRESETS = [-2000, -1000, 0, 1000, 2000]
    TIMED_RUN_SPEEDS = [-2000, -1000, 1000, 2000]
    PLOTTER_SPEED_PRESETS = [-2000, -1000, -500, 0, 500, 1000, 2000]
    MOTOR_NAMES = ["LU", "PI", "ED", "DI", "FDS", "FDP"]

    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Robot Control Application (servoAPI)")
        self.root.geometry("900x950")
        self.logger = _build_logger()

        self.port = os.getenv("SERVOGUI_PORT", "COM4")
        self.baud_rate = int(os.getenv("SERVOGUI_BAUD", "921600"))
        self.timeout = float(os.getenv("SERVOGUI_TIMEOUT", "0.03"))
        self.mock_mode = _env_truthy("SERVOGUI_MOCK")

        self.sliders, self.pos_labels, self.load_labels = [], [], []
        self.timed_run_id, self.timed_run_ms = None, None
        self.pos_control_id = None
        self.user_zero_offsets = [0] * self.NUM_MOTORS

        self.current_positions = [0] * self.NUM_MOTORS
        self.current_loads = [0] * self.NUM_MOTORS
        self.current_speeds = [0] * self.NUM_MOTORS

        self.state = AppState(self.NUM_MOTORS)
        self.event_bus = EventBus()
        self.session = ServoSession(
            app_state=self.state,
            event_bus=self.event_bus,
            port=self.port,
            baud_rate=self.baud_rate,
            timeout=self.timeout,
            use_mock=self.mock_mode,
        )
        self.control_service = ControlService(self.session, self.NUM_MOTORS)
        self.id_service = IDService(self.session)
        self.record_service = RecordService()

        self.batch_window = None
        self.batch_control_tab = None
        self.id_manager_window = None
        self.id_manager_tab = None

        self.create_top_bar()
        self.create_menu()
        self.create_main_tabs()

        self.event_bus.subscribe("telemetry", self._on_telemetry)
        self.event_bus.subscribe("id_event", self._on_id_event)
        self.event_bus.subscribe("connection", self._on_connection)
        self.event_bus.subscribe("error", self._on_error)
        self.event_bus.subscribe("command", self._on_command)

        try:
            self.session.start()
            if not self.mock_mode:
                time.sleep(0.1)
                self.id_service.reset_ids()
        except Exception as exc:
            if self.mock_mode:
                raise
            self.status_value.config(text=f"Connection failed: {exc}")
            self.logger.exception("Connection failed")

        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)

    def create_top_bar(self):
        top_bar = ttk.Frame(self.root, padding="5", relief=tk.RAISED, borderwidth=1)
        top_bar.pack(fill=tk.X, side=tk.TOP)

        ttk.Label(top_bar, text="Global Controls:", font=("Helvetica", 10, "bold")).pack(side=tk.LEFT, padx=5)

        stop_style = ttk.Style()
        stop_style.configure("Red.TButton", foreground="red")
        ttk.Button(top_bar, text="STOP ALL", command=self.emergency_stop, style="Red.TButton").pack(side=tk.LEFT, padx=5)
        ttk.Button(top_bar, text="All Go Zero", command=self.all_go_to_zero).pack(side=tk.LEFT, padx=5)
        ttk.Button(top_bar, text="Reset System", command=self.reset_system).pack(side=tk.LEFT, padx=5)

        ttk.Button(top_bar, text="Reconnect", command=self.reconnect_backend).pack(side=tk.RIGHT, padx=5)
        ttk.Button(top_bar, text="Reset IDs (Force 1:1)", command=self.force_reset_ids).pack(side=tk.RIGHT, padx=5)

        status_frame = ttk.Frame(top_bar)
        status_frame.pack(side=tk.RIGHT, padx=10)
        ttk.Label(status_frame, text="Backend:").pack(side=tk.LEFT)
        self.status_value = ttk.Label(status_frame, text="-", foreground="blue")
        self.status_value.pack(side=tk.LEFT, padx=4)

    def create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)

        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)

        tools_menu.add_command(label="Batch Control", command=self.open_batch_control)
        tools_menu.add_command(label="ID Manager", command=self.open_id_manager)

    def create_main_tabs(self):
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(pady=10, padx=10, fill="both", expand=True)

        manual_tab_frame = ttk.Frame(self.notebook)
        plotter_tab_frame = ttk.Frame(self.notebook)
        monitor_all_frame = ttk.Frame(self.notebook)
        motion_recorder_frame = ttk.Frame(self.notebook)
        high_acc_pos_frame = ttk.Frame(self.notebook)

        self.notebook.add(manual_tab_frame, text="Manual Control")
        self.notebook.add(plotter_tab_frame, text="Real-time Plotter")
        self.notebook.add(monitor_all_frame, text="All Servos Monitor")
        self.notebook.add(motion_recorder_frame, text="Motion Recorder")
        self.notebook.add(high_acc_pos_frame, text="High Acc. Pos. Control")

        self.manual_tab = ManualControlTab(manual_tab_frame, self)
        self.plotter_tab = PlotterTab(plotter_tab_frame, self)
        self.monitor_all_tab = MonitorAllTab(monitor_all_frame, self)
        self.motion_recorder_tab = MotionRecorderTab(motion_recorder_frame, self)
        self.high_acc_pos_tab = HighAccuracyPositionTab(high_acc_pos_frame, self)

    def open_batch_control(self):
        if self.batch_window is None or not tk.Toplevel.winfo_exists(self.batch_window):
            self.batch_window = tk.Toplevel(self.root)
            self.batch_window.title("Batch Control")
            self.batch_window.geometry("600x800")
            self.batch_control_tab = BatchControlTab(self.batch_window, self)
        else:
            self.batch_window.lift()

    def open_id_manager(self):
        if self.id_manager_window is None or not tk.Toplevel.winfo_exists(self.id_manager_window):
            self.id_manager_window = tk.Toplevel(self.root)
            self.id_manager_window.title("ID Manager")
            self.id_manager_window.geometry("400x500")
            self.id_manager_tab = IDManagerTab(self.id_manager_window, self)
        else:
            self.id_manager_window.lift()

    def _on_connection(self, payload):
        self.root.after(0, lambda: self._update_connection_status(payload))

    def _update_connection_status(self, payload):
        connected = payload.get("connected", False)
        mock = payload.get("mock", False)
        if connected:
            mode = "MOCK" if mock else "SERIAL"
            self.status_value.config(text=f"Connected ({mode})", foreground="green")
        else:
            self.status_value.config(text="Disconnected", foreground="red")

    def _on_error(self, message):
        self.logger.error("%s", message)
        self.root.after(0, lambda: self.status_value.config(text=str(message), foreground="red"))

    def _on_id_event(self, line):
        self.logger.info("ID_EVENT %s", line)
        self.root.after(0, lambda: self._apply_id_event(line))

    def _on_command(self, payload):
        method = payload.get("method")
        args = payload.get("args", ())
        kwargs = payload.get("kwargs", {})
        ok = payload.get("ok", False)
        if ok:
            self.logger.info("CMD method=%s args=%s kwargs=%s", method, args, kwargs)
        else:
            self.logger.error(
                "CMD_FAILED method=%s args=%s kwargs=%s error=%s",
                method,
                args,
                kwargs,
                payload.get("error"),
            )

    def _apply_id_event(self, line):
        if self.id_manager_tab:
            self.id_manager_tab.process_scan_result(line)

    def _on_telemetry(self, frame):
        self.root.after(0, lambda f=frame: self._apply_telemetry(f))

    def _apply_telemetry(self, frame):
        self.current_positions = list(frame.positions)
        self.current_loads = list(frame.loads)
        if frame.speeds is None:
            self.current_speeds = [0] * self.NUM_MOTORS
        else:
            self.current_speeds = list(frame.speeds)

        self.update_manual_tab_labels()
        self.high_acc_pos_tab.update_telemetry()
        self.monitor_all_tab.process_frame(frame.timestamp_ms, self.current_loads)

        if self.batch_control_tab:
            self.batch_control_tab.update_telemetry(self.current_positions, self.current_loads)

        try:
            motor_id_plot_0_indexed = int(self.plotter_tab.plotting_motor_id.get())
            if 0 <= motor_id_plot_0_indexed < self.NUM_MOTORS:
                pos = self.current_positions[motor_id_plot_0_indexed]
                torque = self.current_loads[motor_id_plot_0_indexed]
                self.plotter_tab.process_data(motor_id_plot_0_indexed, pos, torque)
        except Exception:
            pass

    def update_manual_tab_labels(self):
        for i in range(self.NUM_MOTORS):
            raw_pos = self.current_positions[i]
            load_val = self.current_loads[i]
            calibrated_pos = raw_pos - self.user_zero_offsets[i]
            if i < len(self.pos_labels):
                self.pos_labels[i].config(text=f"Pos: {calibrated_pos}")
            if i < len(self.load_labels):
                self.load_labels[i].config(text=f"Load: {load_val}")

    def force_reset_ids(self):
        self.id_service.reset_ids()

    def emergency_stop(self):
        self.control_service.stop_all()

    def all_go_to_zero(self):
        self.control_service.go_all_to_zero()

    def reset_system(self):
        self.control_service.reset_system()

    def reconnect_backend(self):
        self.session.reconnect(use_mock=self.mock_mode)

    def on_preset_button_click(self, motor_id, speed):
        self.control_service.set_speed(motor_id, speed, force_init=True)
        if self.manual_tab:
            self.manual_tab.update_slider(motor_id, speed)

    def on_global_preset_button_click(self, speed):
        mode = "speed"
        if self.manual_tab:
            try:
                mode = self.manual_tab.control_mode.get()
            except Exception:
                pass

        if mode == "position":
            mapped_val = speed
            if abs(speed) == 2000:
                mapped_val = 2048 if speed > 0 else -2048
            elif abs(speed) == 1000:
                mapped_val = 1024 if speed > 0 else -1024
            elif abs(speed) == 500:
                mapped_val = 512 if speed > 0 else -512
            elif speed == 0:
                mapped_val = 0

            for slider in self.sliders:
                slider.set(mapped_val)
        else:
            for i in range(self.NUM_MOTORS):
                self.control_service.set_speed(i, speed, force_init=True)
                if self.manual_tab:
                    self.manual_tab.update_slider(i, speed)

    def on_timed_run_click(self, speed):
        motor_id_str = self.timed_run_id.get()
        duration = self.timed_run_ms.get()
        if motor_id_str.isdigit() and duration.isdigit():
            self.control_service.timed_run(int(motor_id_str), speed, int(duration))

    def on_set_zero_click(self):
        motor_id_str = self.pos_control_id.get()
        if motor_id_str.isdigit():
            motor_id = int(motor_id_str)
            self.user_zero_offsets[motor_id] = 0
            self.control_service.set_zero(motor_id)

    def on_set_all_zero_click(self):
        for i in range(self.NUM_MOTORS):
            self.user_zero_offsets[i] = 0
        self.control_service.set_all_zero()

    def on_go_to_zero_click(self):
        motor_id_str = self.pos_control_id.get()
        if motor_id_str.isdigit():
            self.control_service.go_to_zero(int(motor_id_str))

    def on_closing(self):
        try:
            self.control_service.stop_all()
        except Exception:
            pass
        self.session.stop()
        self.root.destroy()
