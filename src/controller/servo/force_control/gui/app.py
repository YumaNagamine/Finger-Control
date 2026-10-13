"""Tkinter GUI for six-servo calibrated tendon-force control."""

from __future__ import annotations

import threading
import time
import tkinter as tk
from collections import defaultdict, deque
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Deque, Dict

from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure

from controller.servo.force_control.gui.session import (
    DEFAULT_CALIBRATION_DIR,
    DEFAULT_FORCE_CONFIG_PATH,
    DEFAULT_ZERO_OFFSET_FILE,
    MANUAL_SPEED_PRESETS,
    SERVO_IDS,
    SERVO_NAMES,
    ControlSample,
    DiagnosticSample,
    ForceControlSession,
    GuiForceConfig,
    load_gui_force_config,
)


ACTUAL_PLOT_COLOR = "#1565c0"
TARGET_PLOT_COLOR = "#d84315"
POLL_INTERVAL_MS = 33
PLOT_INTERVAL_MS = 100
PLOT_AXIS_REFRESH_INTERVAL_S = 1.0


class SharedPlotPanel(ttk.LabelFrame):
    """Persistent six-tendon force plot shared by every control tab."""

    def __init__(self, parent, config: GuiForceConfig, on_refresh):
        super().__init__(parent, text="Realtime tendon force", padding=(8, 6))
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(self)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        toolbar.columnconfigure(0, weight=1)
        ttk.Button(
            toolbar,
            text="Refresh plot",
            command=on_refresh,
            style="Control.TButton",
        ).grid(row=0, column=1, sticky="e")

        self.figure = Figure(figsize=(12, 5.4), dpi=100)
        self.axes = self.figure.subplots(2, 3).flatten()
        self.plot_lines = {}
        self.value_texts = {}
        self._tendon_max_by_servo = {}
        for axis, servo_id, tendon in zip(self.axes, SERVO_IDS, SERVO_NAMES):
            tendon_max = max(config.target_force_levels_by_tendon[tendon])
            self._tendon_max_by_servo[servo_id] = tendon_max
            axis.set_title(f"ID{servo_id}  {tendon}", fontsize=10, fontweight="bold")
            axis.set_xlabel("time (s)", fontsize=9)
            axis.set_ylabel("force (N)", fontsize=9)
            axis.tick_params(labelsize=8)
            axis.grid(True, alpha=0.25, linewidth=0.7)
            axis.set_xlim(0.0, 10.0)
            axis.set_ylim(0.0, max(2.2, tendon_max * 1.2))
            actual_line, = axis.plot(
                [],
                [],
                color=ACTUAL_PLOT_COLOR,
                linewidth=1.8,
                label="Actual",
            )
            target_line, = axis.plot(
                [],
                [],
                color=TARGET_PLOT_COLOR,
                linestyle="--",
                linewidth=1.6,
                label="Target",
            )
            axis.legend(loc="upper left", fontsize=8, framealpha=0.85)
            value_text = axis.text(
                0.98,
                0.96,
                "A --  |  T --",
                transform=axis.transAxes,
                ha="right",
                va="top",
                fontsize=8,
                color="#263238",
                bbox={"facecolor": "white", "alpha": 0.75, "edgecolor": "none"},
            )
            self.plot_lines[servo_id] = (actual_line, target_line)
            self.value_texts[servo_id] = value_text
            actual_line.set_animated(True)
            target_line.set_animated(True)
            value_text.set_animated(True)

        self.figure.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        self._backgrounds = {}
        self._last_full_draw_at = 0.0
        self._full_redraw_needed = True
        self.canvas.mpl_connect("draw_event", self._capture_backgrounds)
        self.canvas.draw()
        self._blit_artists()

    def _capture_backgrounds(self, _event=None) -> None:
        self._backgrounds = {
            servo_id: self.canvas.copy_from_bbox(axis.bbox)
            for axis, servo_id in zip(self.axes, SERVO_IDS)
        }
        self._full_redraw_needed = False

    def _blit_artists(self) -> None:
        if len(self._backgrounds) != len(SERVO_IDS):
            self._full_redraw_needed = True
            return
        for axis, servo_id in zip(self.axes, SERVO_IDS):
            self.canvas.restore_region(self._backgrounds[servo_id])
            actual_line, target_line = self.plot_lines[servo_id]
            axis.draw_artist(actual_line)
            axis.draw_artist(target_line)
            axis.draw_artist(self.value_texts[servo_id])
            self.canvas.blit(axis.bbox)

    def update(self, history: Dict[int, Deque[ControlSample]], start_time: float) -> None:
        desired_limits = {}
        urgent_rescale = False
        for axis, servo_id, tendon in zip(self.axes, SERVO_IDS, SERVO_NAMES):
            samples = list(history[servo_id])
            if not samples:
                continue
            visible_start = samples[-1].received_at - 10.0
            samples = [
                sample for sample in samples if sample.received_at >= visible_start
            ]
            times = [sample.received_at - start_time for sample in samples]
            actual = [sample.force_n for sample in samples]
            target = [sample.target_force_n for sample in samples]
            actual_line, target_line = self.plot_lines[servo_id]
            actual_line.set_data(times, actual)
            target_line.set_data(times, target)

            end_time = max(times[-1], 10.0)
            axis.set_xlim(max(0.0, end_time - 10.0), max(10.0, end_time))
            low = min(0.0, min(actual) * 1.1)
            tendon_max = self._tendon_max_by_servo[servo_id]
            high = max(tendon_max * 1.1, max(actual + target + [0.0]) * 1.2)
            if high <= low:
                high = low + 1.0
            desired_limits[servo_id] = (
                (max(0.0, end_time - 10.0), max(10.0, end_time)),
                (low, high),
            )
            current_low, current_high = axis.get_ylim()
            urgent_rescale = urgent_rescale or min(actual) < current_low or max(
                actual + target
            ) > current_high
            self.value_texts[servo_id].set_text(
                f"A {actual[-1]:.2f} N  |  T {target[-1]:.2f} N"
            )
        now = time.monotonic()
        axis_refresh_due = (
            now - self._last_full_draw_at >= PLOT_AXIS_REFRESH_INTERVAL_S
        )
        limits_changed = any(
            tuple(axis.get_xlim()) != desired_limits[servo_id][0]
            or tuple(axis.get_ylim()) != desired_limits[servo_id][1]
            for axis, servo_id in zip(self.axes, SERVO_IDS)
            if servo_id in desired_limits
        )
        if (
            self._full_redraw_needed
            or urgent_rescale
            or (axis_refresh_due and limits_changed)
        ):
            for axis, servo_id in zip(self.axes, SERVO_IDS):
                limits = desired_limits.get(servo_id)
                if limits is not None:
                    axis.set_xlim(*limits[0])
                    axis.set_ylim(*limits[1])
            self.canvas.draw()
            self._last_full_draw_at = now
        self._blit_artists()

    def clear(self, *, redraw: bool = True) -> None:
        """Clear all plotted samples and restore the initial axis view."""

        for axis, servo_id, tendon in zip(self.axes, SERVO_IDS, SERVO_NAMES):
            actual_line, target_line = self.plot_lines[servo_id]
            actual_line.set_data([], [])
            target_line.set_data([], [])
            axis.set_xlim(0.0, 10.0)
            axis.set_ylim(0.0, max(2.2, self._tendon_max_by_servo[servo_id] * 1.2))
            self.value_texts[servo_id].set_text("A --  |  T --")
        if redraw:
            self.canvas.draw()
            self._last_full_draw_at = time.monotonic()
            self._blit_artists()
        else:
            self._backgrounds = {}
            self._full_redraw_needed = True


class MultiSignalPlotPanel(ttk.LabelFrame):
    """Lightweight shared-axis plot for six diagnostic signals."""

    def __init__(self, parent, *, title: str, ylabel: str, sample_field: str, on_refresh):
        super().__init__(parent, text=title, padding=(8, 6))
        self.sample_field = sample_field
        self.columnconfigure(0, weight=1)
        self.rowconfigure(1, weight=1)

        toolbar = ttk.Frame(self)
        toolbar.grid(row=0, column=0, sticky="ew", pady=(0, 4))
        toolbar.columnconfigure(0, weight=1)
        ttk.Button(
            toolbar,
            text="Refresh plot",
            command=on_refresh,
            style="Control.TButton",
        ).grid(row=0, column=1, sticky="e")

        self.figure = Figure(figsize=(12, 5.4), dpi=100)
        self.axis = self.figure.subplots()
        self.axis.set_xlabel("time (s)", fontsize=9)
        self.axis.set_ylabel(ylabel, fontsize=9)
        self.axis.tick_params(labelsize=8)
        self.axis.grid(True, alpha=0.25, linewidth=0.7)
        self.axis.set_xlim(0.0, 10.0)
        self.lines = {}
        for servo_id, tendon in zip(SERVO_IDS, SERVO_NAMES):
            line, = self.axis.plot([], [], linewidth=1.5, label=f"ID{servo_id} {tendon}")
            line.set_animated(True)
            self.lines[servo_id] = line
        self.axis.legend(loc="upper left", ncol=3, fontsize=8, framealpha=0.85)
        self.figure.tight_layout()
        self.canvas = FigureCanvasTkAgg(self.figure, master=self)
        self.canvas.get_tk_widget().grid(row=1, column=0, sticky="nsew")
        self._background = None
        self._last_full_draw_at = 0.0
        self._full_redraw_needed = True
        self.canvas.mpl_connect("draw_event", self._capture_background)
        self.canvas.draw()
        self._blit_lines()

    def _capture_background(self, _event=None) -> None:
        self._background = self.canvas.copy_from_bbox(self.axis.bbox)
        self._full_redraw_needed = False

    def _blit_lines(self) -> None:
        if self._background is None:
            self._full_redraw_needed = True
            return
        self.canvas.restore_region(self._background)
        for line in self.lines.values():
            self.axis.draw_artist(line)
        self.canvas.blit(self.axis.bbox)

    def update(self, history: Dict[int, Deque[DiagnosticSample]], start_time: float) -> None:
        all_values = []
        latest_received_at = max(
            (samples[-1].received_at for samples in history.values() if samples),
            default=None,
        )
        if latest_received_at is None:
            return
        visible_start = latest_received_at - 10.0
        for servo_id in SERVO_IDS:
            samples = [
                sample
                for sample in history[servo_id]
                if sample.received_at >= visible_start
            ]
            if not samples:
                self.lines[servo_id].set_data([], [])
                continue
            times = [sample.received_at - start_time for sample in samples]
            values = [float(getattr(sample, self.sample_field)) for sample in samples]
            self.lines[servo_id].set_data(times, values)
            all_values.extend(values)

        end_time = max(latest_received_at - start_time, 10.0)
        desired_xlim = (max(0.0, end_time - 10.0), max(10.0, end_time))
        desired_ylim = None
        urgent_rescale = False
        if all_values:
            low = min(all_values)
            high = max(all_values)
            padding = max(1.0, (high - low) * 0.1)
            desired_ylim = (low - padding, high + padding)
            current_low, current_high = self.axis.get_ylim()
            urgent_rescale = low < current_low or high > current_high
        now = time.monotonic()
        limits_changed = tuple(self.axis.get_xlim()) != desired_xlim or (
            desired_ylim is not None
            and tuple(self.axis.get_ylim()) != desired_ylim
        )
        if (
            self._full_redraw_needed
            or urgent_rescale
            or (
                limits_changed
                and now - self._last_full_draw_at
                >= PLOT_AXIS_REFRESH_INTERVAL_S
            )
        ):
            self.axis.set_xlim(*desired_xlim)
            if desired_ylim is not None:
                self.axis.set_ylim(*desired_ylim)
            self.canvas.draw()
            self._last_full_draw_at = now
        self._blit_lines()

    def clear(self, *, redraw: bool = True) -> None:
        for line in self.lines.values():
            line.set_data([], [])
        self.axis.set_xlim(0.0, 10.0)
        if redraw:
            self.canvas.draw()
            self._last_full_draw_at = time.monotonic()
            self._blit_lines()
        else:
            self._background = None
            self._full_redraw_needed = True


class ForceControlApp(tk.Tk):
    def __init__(self, config: GuiForceConfig):
        super().__init__()
        self.config_model = config
        self.title("Tendon Force Control")
        self.geometry("1500x950")
        self.minsize(1180, 760)
        self.state("zoomed")
        self.protocol("WM_DELETE_WINDOW", self._on_close)

        self.session = ForceControlSession(config)
        self._history: Dict[int, Deque[ControlSample]] = {
            servo_id: deque(maxlen=800) for servo_id in SERVO_IDS
        }
        self._servo_diagnostic_history: Dict[int, Deque[DiagnosticSample]] = {
            servo_id: deque(maxlen=800) for servo_id in SERVO_IDS
        }
        self._encoder_diagnostic_history: Dict[int, Deque[DiagnosticSample]] = {
            servo_id: deque(maxlen=800) for servo_id in SERVO_IDS
        }
        self._plot_lines = {}
        self._plot_start_time = None
        self._servo_plot_start_time = None
        self._encoder_plot_start_time = None
        self._last_plot_update = 0.0
        self._last_logger_update = 0.0
        self._log_history = deque(maxlen=500)
        self._live_log_text = None
        self._last_connection_state = None
        self._shown_fault_message = None
        self._connection_pending = False
        self._baseline_pending = False
        self._closing = False
        self._connect_thread = None
        self._baseline_thread = None
        self._command_thread = None
        self._stop_all_thread = None
        self._disconnect_thread = None
        self._calibration_thread = None
        self._command_pending = False
        self._calibration_pending = False
        self._plot_clear_pending = False
        self._stop_all_pending = False
        self._disconnect_pending = False
        self._poll_after_id = None
        self._row_vars = {}
        self._speed_row_vars = {}
        self._speed_buttons = {}
        self._calibration_row_vars = {}
        self._calibrate_all_button = None
        self._calibrate_buttons = {}
        self._remember_calibration_buttons = {}
        self._target_buttons = []
        self._stop_servo_buttons = []
        self._build_variables()
        self._build_widgets()
        self._poll_after_id = self.after(POLL_INTERVAL_MS, self._poll_session)

    def _build_variables(self) -> None:
        self.status_var = tk.StringVar(value="Disconnected")
        self.speed_status_var = tk.StringVar(value="Speed control: connect first")
        self.calibration_status_var = tk.StringVar(
            value="Torque calibration: connect first"
        )
        self.servo_port_var = tk.StringVar(value="COM3")
        self.encoder_port_var = tk.StringVar(value="COM5")
        self.calibration_dir_var = tk.StringVar(value=str(DEFAULT_CALIBRATION_DIR))
        self.zero_offset_var = tk.StringVar(value=str(DEFAULT_ZERO_OFFSET_FILE))
        self.mode_var = tk.StringVar(value="Step")
        self.ramp_duration_var = tk.StringVar(
            value=f"{self.config_model.ramp_duration_s:g}"
        )
        self.kp_var = tk.StringVar(value="5.0")
        self.ki_var = tk.StringVar(value="0.0")
        self.kd_var = tk.StringVar(value="0.0")
        self.max_speed_var = tk.StringVar(value="500.0")
        self.hard_limit_summary_var = tk.StringVar(
            value=self._format_hard_limits(self.config_model.hard_force_limits_n_by_tendon)
        )
        self.control_sign_var = tk.StringVar(value="1.0")
        for servo_id, tendon in zip(SERVO_IDS, SERVO_NAMES):
            self._row_vars[servo_id] = {
                "actual": tk.StringVar(value="--"),
                "target": tk.StringVar(value="--"),
                "error": tk.StringVar(value="--"),
                "speed": tk.StringVar(value="--"),
                "state": tk.StringVar(value="disconnected"),
                "baseline": tk.StringVar(value="--"),
            }
            self._speed_row_vars[servo_id] = {
                "force": tk.StringVar(value="--"),
                "position": tk.StringVar(value="--"),
                "encoder": tk.StringVar(value="--"),
                "delta": tk.StringVar(value="--"),
                "speed": tk.StringVar(value="--"),
                "state": tk.StringVar(value="disconnected"),
            }
            self._calibration_row_vars[servo_id] = {
                "servo_position": tk.StringVar(value="--"),
                "encoder_position": tk.StringVar(value="--"),
                "delta": tk.StringVar(value="--"),
                "slope": tk.StringVar(value="--"),
                "intercept": tk.StringVar(value="--"),
                "remembered": tk.StringVar(value="--"),
            }

    def _build_widgets(self) -> None:
        self._configure_styles()
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=0)
        self.rowconfigure(2, weight=1)

        header = ttk.Frame(self, padding=(16, 12, 16, 8))
        header.grid(row=0, column=0, sticky="ew")
        header.columnconfigure(0, weight=1)
        ttk.Label(header, text="Tendon Force Control", style="AppTitle.TLabel").grid(
            row=0, column=0, sticky="w"
        )
        ttk.Label(
            header,
            text="Six-servo experimental monitor and controller",
            style="Subtitle.TLabel",
        ).grid(row=1, column=0, sticky="w", pady=(2, 0))
        ttk.Label(header, textvariable=self.status_var, style="Status.TLabel").grid(
            row=0, column=1, rowspan=2, padx=(16, 18), sticky="e"
        )
        self.stop_button = ttk.Button(
            header,
            text="STOP ALL",
            command=self._stop_all,
            state="disabled",
            style="Danger.TButton",
        )
        self.stop_button.grid(row=0, column=2, rowspan=2, sticky="e")

        connection = ttk.LabelFrame(self, text="Connection", padding=(10, 6))
        connection.grid(row=1, column=0, sticky="ew", padx=12, pady=(0, 8))
        connection.columnconfigure(1, weight=0)
        connection.columnconfigure(3, weight=0)
        connection.columnconfigure(5, weight=1)
        connection.columnconfigure(7, weight=1)
        ttk.Label(connection, text="Servo port").grid(
            row=0, column=0, padx=(0, 4), pady=2, sticky="w"
        )
        ttk.Entry(connection, textvariable=self.servo_port_var, width=10).grid(
            row=0, column=1, padx=(0, 12), pady=2, sticky="w"
        )
        ttk.Label(connection, text="Encoder port").grid(
            row=0, column=2, padx=(0, 4), pady=2, sticky="w"
        )
        ttk.Entry(connection, textvariable=self.encoder_port_var, width=10).grid(
            row=0, column=3, padx=(0, 12), pady=2, sticky="w"
        )
        self.calibration_summary_var = tk.StringVar()
        self.zero_offset_summary_var = tk.StringVar()
        self._refresh_path_summaries()
        ttk.Label(connection, textvariable=self.calibration_summary_var).grid(
            row=0, column=4, columnspan=2, padx=(0, 10), pady=2, sticky="w"
        )
        ttk.Label(connection, textvariable=self.zero_offset_summary_var).grid(
            row=0, column=6, columnspan=2, padx=(0, 10), pady=2, sticky="w"
        )
        self.path_details_button = ttk.Button(
            connection,
            text="Path details...",
            command=self._show_path_settings,
            style="Control.TButton",
        )
        self.path_details_button.grid(row=0, column=8, padx=(0, 6), pady=2)
        self.connect_button = ttk.Button(
            connection, text="Connect", command=self._connect, style="Control.TButton"
        )
        self.connect_button.grid(row=0, column=9, padx=3, pady=2)
        self.disconnect_button = ttk.Button(
            connection,
            text="Disconnect",
            command=self._disconnect,
            state="disabled",
            style="Control.TButton",
        )
        self.disconnect_button.grid(row=0, column=10, padx=3, pady=2)
        self.baseline_button = ttk.Button(
            connection,
            text="Refresh baseline",
            command=self._refresh_baseline,
            state="disabled",
            style="Control.TButton",
        )
        self.baseline_button.grid(row=0, column=11, padx=(3, 0), pady=2)

        main_area = ttk.Frame(self, padding=(12, 0, 12, 12))
        main_area.grid(row=2, column=0, sticky="nsew")
        main_area.columnconfigure(0, weight=3)
        main_area.columnconfigure(1, weight=1)
        main_area.rowconfigure(0, weight=2)
        main_area.rowconfigure(1, weight=3)

        self.notebook = ttk.Notebook(main_area)
        self.notebook.grid(
            row=0, column=0, sticky="nsew", padx=(0, 4), pady=(0, 8)
        )
        control_tab = self._control_tab = ttk.Frame(self.notebook)
        speed_tab = self._speed_tab = ttk.Frame(self.notebook)
        calibration_tab = self._calibration_tab = ttk.Frame(self.notebook)
        logger_tab = self._logger_tab = ttk.Frame(self.notebook)
        self.notebook.add(control_tab, text="Force control")
        self.notebook.add(speed_tab, text="Speed control")
        self.notebook.add(calibration_tab, text="Torque calibration")
        self.notebook.add(logger_tab, text="Logger / diagnostics")
        parameters = ttk.LabelFrame(control_tab, text="Controller parameters")
        control_tab.columnconfigure(0, weight=1)
        control_tab.rowconfigure(1, weight=1)
        parameters.grid(
            row=0,
            column=0,
            sticky="ew",
            padx=10,
            pady=(10, 4),
        )
        self._entry(parameters, "Kp", self.kp_var, 0, 0, width=8)
        self._entry(parameters, "Ki", self.ki_var, 0, 2, width=8)
        self._entry(parameters, "Kd", self.kd_var, 0, 4, width=8)
        self._entry(parameters, "Max speed", self.max_speed_var, 0, 6, width=8)
        ttk.Label(parameters, text="Hard limits N").grid(
            row=0, column=8, padx=(4, 2), pady=4, sticky="w"
        )
        ttk.Label(
            parameters,
            textvariable=self.hard_limit_summary_var,
            width=30,
        ).grid(row=0, column=9, padx=(0, 6), pady=4, sticky="w")
        self._entry(parameters, "Control sign", self.control_sign_var, 0, 10, width=8)
        self.arm_button = ttk.Button(
            parameters, text="Arm", command=self._arm, state="disabled", style="Control.TButton"
        )
        self.arm_button.grid(row=1, column=0, padx=4, pady=6, sticky="w")
        self.start_button = ttk.Button(
            parameters,
            text="Start control",
            command=self._start,
            state="disabled",
            style="Control.TButton",
        )
        self.start_button.grid(row=1, column=1, columnspan=2, padx=4, pady=6, sticky="w")
        ttk.Label(parameters, text="Target mode").grid(row=1, column=6, padx=(12, 2), pady=4)
        self.mode_combo = ttk.Combobox(
            parameters, textvariable=self.mode_var, values=("Step", "Ramp"), state="readonly", width=8
        )
        self.mode_combo.grid(row=1, column=7, padx=2, pady=4)
        ttk.Label(parameters, text="Ramp seconds").grid(row=1, column=8, padx=(12, 2), pady=4)
        ttk.Entry(parameters, textvariable=self.ramp_duration_var, width=8).grid(
            row=1, column=9, padx=2, pady=4
        )
        ttk.Label(
            parameters,
            text="Target presets: tendon-specific",
        ).grid(row=1, column=10, columnspan=2, padx=(12, 4), pady=4, sticky="w")

        controls = ttk.LabelFrame(control_tab, text="Per-tendon force control", padding=(8, 6))
        controls.grid(row=1, column=0, sticky="nsew", padx=10, pady=(4, 10))
        controls.columnconfigure(8, weight=1)
        headers = ["ID", "Tendon", "Actual N", "Target N", "Error N", "Speed", "Baseline N", "State", "Targets"]
        for column, header in enumerate(headers):
            ttk.Label(controls, text=header, style="TableHeader.TLabel").grid(
                row=0, column=column, padx=5, pady=4, sticky="w"
            )
        for column, width in enumerate((44, 74, 82, 82, 82, 70, 92, 108, 420)):
            controls.columnconfigure(column, minsize=width)
        for row, (servo_id, tendon) in enumerate(zip(SERVO_IDS, SERVO_NAMES), start=1):
            controls.rowconfigure(row, minsize=36)
            variables = self._row_vars[servo_id]
            ttk.Label(controls, text=str(servo_id), width=4).grid(row=row, column=0, padx=5, pady=3)
            ttk.Label(controls, text=tendon, width=8).grid(row=row, column=1, padx=5, pady=3, sticky="w")
            ttk.Label(controls, textvariable=variables["actual"], width=10, style="Value.TLabel").grid(row=row, column=2, padx=5, pady=3)
            ttk.Label(controls, textvariable=variables["target"], width=10, style="Value.TLabel").grid(row=row, column=3, padx=5, pady=3)
            ttk.Label(controls, textvariable=variables["error"], width=10).grid(row=row, column=4, padx=5, pady=3)
            ttk.Label(controls, textvariable=variables["speed"], width=8).grid(row=row, column=5, padx=5, pady=3)
            ttk.Label(controls, textvariable=variables["baseline"], width=10).grid(row=row, column=6, padx=5, pady=3)
            ttk.Label(controls, textvariable=variables["state"], width=14).grid(row=row, column=7, padx=5, pady=3, sticky="w")
            target_frame = ttk.Frame(controls)
            target_frame.grid(row=row, column=8, padx=5, pady=3, sticky="w")
            for value in self.config_model.target_force_levels_by_tendon[tendon]:
                target_button = ttk.Button(
                    target_frame,
                    text=f"{value:g}",
                    width=5,
                    style="Small.TButton",
                    command=lambda sid=servo_id, target=value: self._set_target(sid, target),
                )
                target_button.pack(side="left", padx=1)
                self._target_buttons.append(target_button)
            stop_button = ttk.Button(
                target_frame,
                text="Stop",
                width=5,
                style="DangerSmall.TButton",
                command=lambda sid=servo_id: self._stop_servo(sid),
            )
            stop_button.pack(side="left", padx=(4, 1))
            self._stop_servo_buttons.append(stop_button)

        self._build_speed_widgets(speed_tab)
        self._build_calibration_widgets(calibration_tab)
        self._build_logger_widgets(logger_tab)
        self._build_live_log_widgets(main_area)

        self.plot_notebook = ttk.Notebook(main_area)
        self.plot_notebook.grid(row=1, column=0, columnspan=2, sticky="nsew")
        force_plot_tab = ttk.Frame(self.plot_notebook)
        servo_plot_tab = ttk.Frame(self.plot_notebook)
        encoder_plot_tab = ttk.Frame(self.plot_notebook)
        for tab in (force_plot_tab, servo_plot_tab, encoder_plot_tab):
            tab.columnconfigure(0, weight=1)
            tab.rowconfigure(0, weight=1)
        self.plot_notebook.add(force_plot_tab, text="Force")
        self.plot_notebook.add(servo_plot_tab, text="Servo position")
        self.plot_notebook.add(encoder_plot_tab, text="Encoder position")

        self.shared_plot_panel = SharedPlotPanel(
            force_plot_tab,
            self.config_model,
            self._refresh_plot,
        )
        self.shared_plot_panel.grid(row=0, column=0, sticky="nsew")
        self.servo_plot_panel = MultiSignalPlotPanel(
            servo_plot_tab,
            title="Realtime servo position",
            ylabel="servo position (counts)",
            sample_field="value",
            on_refresh=self._refresh_plot,
        )
        self.servo_plot_panel.grid(row=0, column=0, sticky="nsew")
        self.encoder_plot_panel = MultiSignalPlotPanel(
            encoder_plot_tab,
            title="Realtime magnetic encoder position",
            ylabel="encoder position (deg)",
            sample_field="value",
            on_refresh=self._refresh_plot,
        )
        self.encoder_plot_panel.grid(row=0, column=0, sticky="nsew")
        self._plot_panels = (
            self.shared_plot_panel,
            self.servo_plot_panel,
            self.encoder_plot_panel,
        )
        self.plot_notebook.bind("<<NotebookTabChanged>>", self._on_plot_tab_changed)
        self.figure = self.shared_plot_panel.figure
        self.axes = self.shared_plot_panel.axes
        self.canvas = self.shared_plot_panel.canvas
        self._plot_lines = self.shared_plot_panel.plot_lines

    def _configure_styles(self) -> None:
        style = ttk.Style(self)
        style.configure("AppTitle.TLabel", font=("Segoe UI", 18, "bold"))
        style.configure("Subtitle.TLabel", font=("Segoe UI", 10), foreground="#546e7a")
        style.configure("Status.TLabel", font=("Segoe UI", 11, "bold"), foreground="#263238")
        style.configure("TableHeader.TLabel", font=("Segoe UI", 10, "bold"), foreground="#37474f")
        style.configure("Value.TLabel", font=("Segoe UI", 11, "bold"))
        style.configure("TNotebook.Tab", padding=(14, 7), font=("Segoe UI", 10))
        style.configure("Treeview", rowheight=28, font=("Segoe UI", 9))
        style.configure("Treeview.Heading", font=("Segoe UI", 9, "bold"))
        style.configure("Control.TButton", padding=(10, 5), font=("Segoe UI", 10))
        style.configure("Small.TButton", padding=(7, 3), font=("Segoe UI", 9))
        style.configure(
            "Danger.TButton",
            padding=(18, 8),
            font=("Segoe UI", 11, "bold"),
            foreground="#b71c1c",
        )
        style.configure(
            "DangerSmall.TButton",
            padding=(7, 3),
            font=("Segoe UI", 9, "bold"),
            foreground="#b71c1c",
        )

    @staticmethod
    def _compact_path(value: str, max_length: int = 34) -> str:
        path = Path(value)
        text = str(path)
        if len(text) <= max_length:
            return text
        name = path.name or text
        if len(name) >= max_length - 4:
            return f"...{name[-(max_length - 3):]}"
        return f"...\\{name}"

    def _refresh_path_summaries(self) -> None:
        if hasattr(self, "calibration_summary_var"):
            self.calibration_summary_var.set(
                f"Calibration: {self._compact_path(self.calibration_dir_var.get())}"
            )
            self.zero_offset_summary_var.set(
                f"Zero offset: {self._compact_path(self.zero_offset_var.get())}"
            )

    def _show_path_settings(self) -> None:
        existing = getattr(self, "_path_settings_window", None)
        if existing is not None and existing.winfo_exists():
            existing.lift()
            existing.focus_force()
            return
        dialog = tk.Toplevel(self)
        self._path_settings_window = dialog
        dialog.title("Connection path details")
        dialog.transient(self)
        dialog.resizable(True, False)
        dialog.columnconfigure(1, weight=1)
        ttk.Label(
            dialog,
            text="These paths are used on the next connection.",
            style="Subtitle.TLabel",
        ).grid(row=0, column=0, columnspan=2, padx=14, pady=(12, 8), sticky="w")
        calibration_var = tk.StringVar(value=self.calibration_dir_var.get())
        zero_offset_var = tk.StringVar(value=self.zero_offset_var.get())
        self._entry(dialog, "Calibration directory", calibration_var, 1, 0, width=58)
        self._entry(dialog, "Zero offset file", zero_offset_var, 2, 0, width=58)
        buttons = ttk.Frame(dialog)
        buttons.grid(row=3, column=0, columnspan=2, padx=14, pady=(8, 12), sticky="e")

        def apply_paths() -> None:
            self.calibration_dir_var.set(calibration_var.get().strip())
            self.zero_offset_var.set(zero_offset_var.get().strip())
            self._refresh_path_summaries()
            dialog.destroy()

        ttk.Button(
            buttons,
            text="Apply",
            command=apply_paths,
            style="Control.TButton",
        ).pack(side="right")
        ttk.Button(buttons, text="Cancel", command=dialog.destroy).pack(side="right", padx=(0, 6))

    @staticmethod
    def _format_hard_limits(limits: dict) -> str:
        grouped = {}
        for tendon, limit in limits.items():
            grouped.setdefault(float(limit), []).append(tendon)
        return " | ".join(
            f"{'/'.join(tendons)} {limit:g}N"
            for limit, tendons in grouped.items()
        )

    def _build_speed_widgets(self, parent) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)

        ttk.Label(
            parent,
            text="Manual speed commands use raw ServoHardwareAPI speed units. "
            "Stop force control before using this tab.",
        ).grid(row=0, column=0, sticky="w", padx=10, pady=(10, 4))
        ttk.Label(parent, textvariable=self.speed_status_var).grid(
            row=1, column=0, sticky="w", padx=10, pady=(0, 4)
        )

        table = ttk.LabelFrame(
            parent,
            text="Per-servo speed control and realtime force",
            padding=(8, 6),
        )
        table.grid(row=2, column=0, sticky="nsew", padx=10, pady=(4, 8))
        for column, width in enumerate((44, 68, 82, 86, 100, 100, 126, 110, 370)):
            table.columnconfigure(column, minsize=width)
        headers = [
            "ID", "Tendon", "Force (N)", "Servo pos", "Encoder deg",
            "Delta theta", "Commanded speed", "State", "Speed",
        ]
        for column, header in enumerate(headers):
            ttk.Label(table, text=header, style="TableHeader.TLabel").grid(
                row=0, column=column, padx=5, pady=4, sticky="w"
            )

        for row, (servo_id, tendon) in enumerate(zip(SERVO_IDS, SERVO_NAMES), start=1):
            table.rowconfigure(row, minsize=36)
            variables = self._speed_row_vars[servo_id]
            ttk.Label(table, text=str(servo_id), width=5).grid(
                row=row, column=0, padx=5, pady=3
            )
            ttk.Label(table, text=tendon, width=8).grid(
                row=row, column=1, padx=5, pady=3, sticky="w"
            )
            ttk.Label(table, textvariable=variables["force"], width=10, style="Value.TLabel").grid(
                row=row, column=2, padx=5, pady=3, sticky="w"
            )
            ttk.Label(table, textvariable=variables["position"], width=10).grid(
                row=row, column=3, padx=5, pady=3, sticky="w"
            )
            ttk.Label(table, textvariable=variables["encoder"], width=12).grid(
                row=row, column=4, padx=5, pady=3, sticky="w"
            )
            ttk.Label(table, textvariable=variables["delta"], width=12).grid(
                row=row, column=5, padx=5, pady=3, sticky="w"
            )
            ttk.Label(table, textvariable=variables["speed"], width=16).grid(
                row=row, column=6, padx=5, pady=3, sticky="w"
            )
            ttk.Label(table, textvariable=variables["state"], width=14).grid(
                row=row, column=7, padx=5, pady=3, sticky="w"
            )
            button_frame = ttk.Frame(table)
            button_frame.grid(row=row, column=8, padx=5, pady=3, sticky="w")
            self._speed_buttons[servo_id] = []
            for speed in MANUAL_SPEED_PRESETS:
                button = ttk.Button(
                    button_frame,
                    text=str(speed),
                    width=6,
                    style="Small.TButton",
                    command=lambda sid=servo_id, value=speed: self._set_manual_speed(
                        sid, value
                    ),
                )
                button.pack(side="left", padx=1)
                self._speed_buttons[servo_id].append(button)

        ttk.Label(
            parent,
            text="STOP ALL is always available in the fixed header button.",
            style="Subtitle.TLabel",
        ).grid(row=3, column=0, sticky="w", padx=10, pady=(0, 8))

    def _build_calibration_widgets(self, parent) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(2, weight=1)
        ttk.Label(
            parent,
            text=(
                "Use Speed control to place the tendon(s) at the repeatable loose position, "
                "set every speed to 0, then calibrate. The 1 s mean is treated as zero "
                "torque; the fixed slope k is never changed."
            ),
            wraplength=1350,
        ).grid(row=0, column=0, sticky="w", padx=10, pady=(10, 4))

        actions = ttk.Frame(parent)
        actions.grid(row=1, column=0, sticky="w", padx=10, pady=(4, 0))
        all_calibration_button = ttk.Button(
            actions,
            text="Calibrate all tendons (set current force as zero)",
            command=self._calibrate_all_zero_force,
            style="Control.TButton",
        )
        all_calibration_button.pack(side="left")
        self._calibrate_all_button = all_calibration_button
        ttk.Label(
            actions,
            textvariable=self.calibration_status_var,
            style="Subtitle.TLabel",
        ).pack(side="left", padx=(12, 0))

        table = ttk.LabelFrame(parent, text="Per-servo zero-torque calibration", padding=(8, 6))
        table.grid(row=2, column=0, sticky="nsew", padx=10, pady=(4, 8))
        headers = (
            "ID", "Tendon", "Servo pos", "Encoder deg", "Delta theta deg",
            "k Nmm/deg", "b Nmm", "Remembered deg", "Actions",
        )
        widths = (44, 74, 90, 110, 120, 110, 120, 120, 470)
        for column, (header, width) in enumerate(zip(headers, widths)):
            table.columnconfigure(column, minsize=width)
            ttk.Label(table, text=header, style="TableHeader.TLabel").grid(
                row=0, column=column, padx=5, pady=4, sticky="w"
            )
        for row, (servo_id, tendon) in enumerate(zip(SERVO_IDS, SERVO_NAMES), start=1):
            variables = self._calibration_row_vars[servo_id]
            ttk.Label(table, text=str(servo_id), width=5).grid(row=row, column=0, padx=5, pady=3)
            ttk.Label(table, text=tendon, width=8).grid(row=row, column=1, padx=5, pady=3)
            for column, key, width in (
                (2, "servo_position", 10), (3, "encoder_position", 12),
                (4, "delta", 13), (5, "slope", 12), (6, "intercept", 13),
                (7, "remembered", 13),
            ):
                ttk.Label(table, textvariable=variables[key], width=width).grid(
                    row=row, column=column, padx=5, pady=3, sticky="w"
                )
            actions = ttk.Frame(table)
            actions.grid(row=row, column=8, padx=5, pady=3, sticky="w")
            calibrate_button = ttk.Button(
                actions,
                text="Calibrate (set current force as zero)",
                command=lambda sid=servo_id: self._calibrate_zero_force(sid),
                style="Small.TButton",
            )
            calibrate_button.pack(side="left", padx=(0, 4))
            remember_button = ttk.Button(
                actions,
                text="Remember encoder position for calibration",
                command=lambda sid=servo_id: self._remember_calibration_position(sid),
                style="Small.TButton",
            )
            remember_button.pack(side="left")
            self._calibrate_buttons[servo_id] = calibrate_button
            self._remember_calibration_buttons[servo_id] = remember_button

    def _build_logger_widgets(self, parent) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        summary = ttk.LabelFrame(parent, text="Session diagnostics", padding=(8, 6))
        summary.grid(row=0, column=0, sticky="ew", padx=10, pady=(10, 4))
        summary.columnconfigure(1, weight=1)
        summary.columnconfigure(3, weight=1)
        self.logger_session_var = tk.StringVar(value="disconnected")
        self.logger_servo_comm_var = tk.StringVar(value="--")
        self.logger_encoder_comm_var = tk.StringVar(value="--")
        self.logger_control_sign_var = tk.StringVar(value="not armed")
        ttk.Label(summary, text="Session").grid(row=0, column=0, padx=4, pady=4, sticky="w")
        ttk.Label(summary, textvariable=self.logger_session_var).grid(
            row=0, column=1, padx=4, pady=4, sticky="w"
        )
        ttk.Label(summary, text="Control sign").grid(row=0, column=2, padx=4, pady=4, sticky="w")
        ttk.Label(summary, textvariable=self.logger_control_sign_var).grid(
            row=0, column=3, padx=4, pady=4, sticky="w"
        )
        ttk.Label(summary, text="Servo communication").grid(
            row=1, column=0, padx=4, pady=4, sticky="w"
        )
        ttk.Label(summary, textvariable=self.logger_servo_comm_var).grid(
            row=1, column=1, padx=4, pady=4, sticky="w"
        )
        ttk.Label(summary, text="Encoder communication").grid(
            row=1, column=2, padx=4, pady=4, sticky="w"
        )
        ttk.Label(summary, textvariable=self.logger_encoder_comm_var).grid(
            row=1, column=3, padx=4, pady=4, sticky="w"
        )
        ttk.Button(summary, text="Refresh", command=self._update_logger).grid(
            row=0, column=4, rowspan=2, padx=8, pady=4
        )

        table_frame = ttk.LabelFrame(
            parent,
            text="Servo / tendon / calibration mapping",
            padding=(8, 6),
        )
        table_frame.grid(row=1, column=0, sticky="nsew", padx=10, pady=(4, 8))
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        columns = (
            "servo_id",
            "tendon",
            "cs",
            "servo_link",
            "encoder_link",
            "calibration_path",
            "calibration_updated",
            "zero_updated",
            "zero_samples",
            "zero_std",
            "initial_force",
        )
        self.logger_tree = ttk.Treeview(
            table_frame, columns=columns, show="headings", selectmode="none"
        )
        headings = {
            "servo_id": "Servo ID",
            "tendon": "Tendon",
            "cs": "AS5048A CS",
            "servo_link": "Servo 5 s: Hz / max gap / valid",
            "encoder_link": "Encoder 5 s: Hz / max gap / valid",
            "calibration_path": "Calibration CSV",
            "calibration_updated": "CSV updated",
            "zero_updated": "Zero offset updated",
            "zero_samples": "Zero samples",
            "zero_std": "Zero std (deg)",
            "initial_force": "Initial force (N)",
        }
        widths = {
            "servo_id": 70,
            "tendon": 90,
            "cs": 100,
            "servo_link": 220,
            "encoder_link": 220,
            "calibration_path": 420,
            "calibration_updated": 165,
            "zero_updated": 165,
            "zero_samples": 100,
            "zero_std": 110,
            "initial_force": 125,
        }
        for column in columns:
            self.logger_tree.heading(column, text=headings[column])
            self.logger_tree.column(column, width=widths[column], stretch=column == "calibration_path")
        self.logger_tree.grid(row=0, column=0, sticky="nsew")
        y_scroll = ttk.Scrollbar(table_frame, orient="vertical", command=self.logger_tree.yview)
        y_scroll.grid(row=0, column=1, sticky="ns")
        x_scroll = ttk.Scrollbar(table_frame, orient="horizontal", command=self.logger_tree.xview)
        x_scroll.grid(row=1, column=0, sticky="ew")
        self.logger_tree.configure(yscrollcommand=y_scroll.set, xscrollcommand=x_scroll.set)

    def _build_live_log_widgets(self, parent) -> None:
        log_frame = ttk.LabelFrame(
            parent,
            text="Logger / event history",
            padding=(8, 6),
        )
        log_frame.grid(row=0, column=1, sticky="nsew", padx=(4, 0), pady=(0, 8))
        log_frame.columnconfigure(0, weight=1)
        log_frame.rowconfigure(0, weight=1)
        self._live_log_text = tk.Text(
            log_frame,
            width=70,
            height=12,
            wrap="word",
            state="disabled",
            background="#fafafa",
            foreground="#263238",
            relief="flat",
            padx=6,
            pady=6,
        )
        self._live_log_text.grid(row=0, column=0, sticky="nsew")
        scrollbar = ttk.Scrollbar(
            log_frame,
            orient="vertical",
            command=self._live_log_text.yview,
        )
        scrollbar.grid(row=0, column=1, sticky="ns")
        self._live_log_text.configure(yscrollcommand=scrollbar.set)

    @staticmethod
    def _entry(parent, label, variable, row, column, *, width=18):
        ttk.Label(parent, text=label).grid(row=row, column=column, padx=(4, 2), pady=4, sticky="w")
        ttk.Entry(parent, textvariable=variable, width=width).grid(
            row=row, column=column + 1, padx=(0, 6), pady=4, sticky="ew"
        )

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _append_log(self, text: str) -> None:
        entry = f"[{time.strftime('%H:%M:%S')}] {text}"
        remove_oldest = (
            self._log_history.maxlen is not None
            and len(self._log_history) == self._log_history.maxlen
        )
        self._log_history.append(entry)
        if self._live_log_text is None:
            return
        self._live_log_text.configure(state="normal")
        if remove_oldest:
            self._live_log_text.delete("1.0", "2.0")
        self._live_log_text.insert("end", entry + "\n")
        self._live_log_text.configure(state="disabled")
        self._live_log_text.see("end")

    def _connect(self) -> None:
        servo_port = self.servo_port_var.get().strip()
        encoder_port = self.encoder_port_var.get().strip()
        calibration_dir = Path(self.calibration_dir_var.get().strip())
        zero_offset_path = Path(self.zero_offset_var.get().strip())
        if not servo_port or not encoder_port:
            messagebox.showerror("Connection", "Servo and encoder ports are required.")
            return
        if self._closing or self._disconnect_pending:
            return
        self._connection_pending = True
        self._plot_clear_pending = True
        self.connect_button.configure(state="disabled")
        self._set_status("Connecting; no motion command is sent...")

        def worker():
            try:
                if self._closing:
                    return
                self.session.connect(
                    servo_port,
                    encoder_port,
                    calibration_dir=calibration_dir,
                    zero_offset_path=zero_offset_path,
                )
            except Exception as exc:
                self.session._emit(f"ERROR: {exc}")
            finally:
                self._connection_pending = False

        thread = threading.Thread(target=worker, name="force-gui-connect", daemon=True)
        self._connect_thread = thread
        thread.start()

    def _disconnect(self) -> None:
        if self._disconnect_pending or self._closing:
            return
        self._disconnect_pending = True
        self._plot_clear_pending = True
        self._set_status("Disconnecting...")

        def worker():
            try:
                self.session.disconnect()
            except Exception as exc:
                self.session._emit(f"ERROR: disconnect failed: {exc}")
            finally:
                self._disconnect_pending = False

        thread = threading.Thread(target=worker, name="force-gui-disconnect", daemon=True)
        self._disconnect_thread = thread
        thread.start()

    def _refresh_baseline(self) -> None:
        if not self.session.connected or self._baseline_pending or self._closing:
            return
        self._baseline_pending = True
        self.baseline_button.configure(state="disabled")

        def worker():
            try:
                self.session.capture_baseline()
                self._plot_clear_pending = True
                self.session._emit("Baseline refreshed.")
            except Exception as exc:
                self.session._emit(f"ERROR: baseline capture failed: {exc}")
            finally:
                self._baseline_pending = False

        thread = threading.Thread(target=worker, name="force-gui-baseline", daemon=True)
        self._baseline_thread = thread
        thread.start()

    def _parse_parameters(self) -> dict:
        return {
            "kp": float(self.kp_var.get()),
            "ki": float(self.ki_var.get()),
            "kd": float(self.kd_var.get()),
            "max_speed": float(self.max_speed_var.get()),
            "hard_force_limits_n": dict(self.config_model.hard_force_limits_n_by_tendon),
            "control_sign": float(self.control_sign_var.get()),
        }

    def _arm(self) -> None:
        try:
            self.session.arm(**self._parse_parameters())
        except Exception as exc:
            messagebox.showerror("Arm", str(exc))

    def _start(self) -> None:
        try:
            self.session.start()
        except Exception as exc:
            messagebox.showerror("Start", str(exc))

    def _stop_all(self) -> None:
        if self._stop_all_pending or self._closing:
            return
        self._stop_all_pending = True

        def worker():
            try:
                self.session.stop_all()
            except Exception as exc:
                self.session._emit(f"ERROR: stop all failed: {exc}")
            finally:
                self._stop_all_pending = False

        thread = threading.Thread(target=worker, name="force-gui-stop-all", daemon=True)
        self._stop_all_thread = thread
        thread.start()

    def _start_command_action(self, action, error_label: str) -> None:
        if self._command_pending or self._closing or self._disconnect_pending:
            return
        self._command_pending = True

        def worker():
            try:
                action()
            except Exception as exc:
                self.session._emit(f"ERROR: {error_label}: {exc}")
            finally:
                self._command_pending = False

        thread = threading.Thread(target=worker, name="force-gui-command", daemon=True)
        self._command_thread = thread
        thread.start()

    def _set_manual_speed(self, servo_id: int, speed: int) -> None:
        self._start_command_action(
            lambda: self.session.set_manual_speed(servo_id, speed),
            "speed control",
        )

    def _start_calibration_action(
        self,
        action,
        error_label: str,
        *,
        clear_plot: bool = False,
        refresh_baseline: bool = False,
    ) -> None:
        if self._calibration_pending or self._closing or self._disconnect_pending:
            self.session._emit(f"INFO: {error_label} was not started; another GUI action is pending")
            return
        self._calibration_pending = True
        self.session._emit(f"START: {error_label}")

        def worker():
            try:
                action()
                if refresh_baseline:
                    try:
                        self.session.capture_baseline()
                        self.session._emit(
                            "Baseline refreshed automatically after torque calibration."
                        )
                    except Exception as exc:
                        self.session._emit(
                            "WARNING: torque calibration was saved, but automatic "
                            f"baseline refresh failed: {exc}"
                        )
                if clear_plot:
                    self._plot_clear_pending = True
            except Exception as exc:
                self.session._emit(f"ERROR: {error_label}: {exc}")
            finally:
                self._calibration_pending = False

        thread = threading.Thread(target=worker, name="force-gui-calibration", daemon=True)
        self._calibration_thread = thread
        thread.start()

    def _calibrate_zero_force(self, servo_id: int) -> None:
        self._start_calibration_action(
            lambda: self.session.calibrate_current_force_as_zero(servo_id, duration_s=1.0),
            "torque zero calibration",
            clear_plot=True,
            refresh_baseline=True,
        )

    def _calibrate_all_zero_force(self) -> None:
        self._start_calibration_action(
            lambda: self.session.calibrate_all_current_force_as_zero(duration_s=1.0),
            "all-tendon torque zero calibration",
            clear_plot=True,
            refresh_baseline=True,
        )

    def _remember_calibration_position(self, servo_id: int) -> None:
        self._start_calibration_action(
            lambda: self.session.remember_encoder_position_for_calibration(servo_id),
            "remember encoder calibration position",
        )

    def _stop_servo(self, servo_id: int) -> None:
        self._start_command_action(
            lambda: self.session.stop_servo(servo_id),
            "stop servo",
        )

    def _set_target(self, servo_id: int, target: float) -> None:
        try:
            self.session.set_target(
                servo_id,
                target,
                self.mode_var.get(),
                float(self.ramp_duration_var.get()),
            )
        except Exception as exc:
            messagebox.showerror("Target", str(exc))

    @staticmethod
    def _communication_text(info: dict) -> str:
        state = info.get("state", "unknown")
        age_s = info.get("last_age_s")
        if age_s is None:
            latest = "no sample"
        else:
            latest = f"last sample {age_s:.2f}s ago"
        error = info.get("error")
        if error:
            return f"{state} ({latest}; {error})"
        return f"{state} ({latest})"

    @staticmethod
    def _stream_metrics_text(info: dict) -> str:
        if not info or not info.get("sample_count"):
            return "--"
        effective_hz = info.get("effective_hz")
        max_interval_s = info.get("max_interval_s")
        valid_percent = info.get("valid_percent")
        hz_text = "--" if effective_hz is None else f"{effective_hz:.1f} Hz"
        gap_text = (
            "--"
            if max_interval_s is None
            else f"{1000.0 * max_interval_s:.0f} ms"
        )
        valid_text = (
            "--" if valid_percent is None else f"{valid_percent:.1f}%"
        )
        return f"{hz_text} / {gap_text} / {valid_text}"

    def _update_logger(self) -> None:
        diagnostics = self.session.diagnostics_snapshot()
        connection = diagnostics["connection"]
        session_state = "connected" if connection["connected"] else "disconnected"
        if connection["connected"]:
            modes = []
            if connection["armed"]:
                modes.append("armed")
            if connection["running"]:
                modes.append("running")
            if connection["faulted"]:
                modes.append("faulted")
            if modes:
                session_state += " / " + " / ".join(modes)
        self.logger_session_var.set(session_state)
        self.logger_servo_comm_var.set(
            self._communication_text(diagnostics["servo_communication"])
        )
        self.logger_encoder_comm_var.set(
            self._communication_text(diagnostics["encoder_communication"])
        )
        control_sign = diagnostics.get("control_sign")
        if control_sign is None:
            try:
                input_sign = float(self.control_sign_var.get())
                self.logger_control_sign_var.set(f"not armed (input {input_sign:+g})")
            except ValueError:
                self.logger_control_sign_var.set("not armed")
        else:
            self.logger_control_sign_var.set(f"{control_sign:+g} (armed)")

        for item in self.logger_tree.get_children():
            self.logger_tree.delete(item)
        for row in diagnostics["mapping"]:
            self.logger_tree.insert(
                "",
                "end",
                values=(
                    row["servo_id"],
                    row["tendon"],
                    f"CS{row['sensor_id']}",
                    self._stream_metrics_text(row["servo_stream"]),
                    self._stream_metrics_text(row["encoder_stream"]),
                    row["calibration_path"],
                    row["calibration_updated"] or "not found",
                    row["zero_offset_updated"] or "not found",
                    "--" if row["zero_sample_count"] is None else row["zero_sample_count"],
                    "--"
                    if row["zero_std_deg"] is None
                    else f"{float(row['zero_std_deg']):.5f}",
                    "--"
                    if row["initial_force_n"] is None
                    else f"{float(row['initial_force_n']):.3f}",
                ),
            )

    def _set_connection_states(self, connected: bool) -> None:
        feedback_recovering = self.session.manual_feedback_recovering
        feedback_errors = self.session.feedback_errors_snapshot()
        calibration_readiness = self.session.calibration_readiness_snapshot()
        faulted = self.session.faulted
        connection_state = (
            connected,
            feedback_recovering,
            tuple(sorted(feedback_errors)),
            calibration_readiness["all_reason"],
            tuple(
                (servo_id, calibration_readiness["by_servo"][servo_id])
                for servo_id in SERVO_IDS
            ),
            faulted,
            self.session.armed,
            self.session.running,
            self._connection_pending,
            self._baseline_pending,
            self._command_pending,
            self._calibration_pending,
            self._stop_all_pending,
            self._disconnect_pending,
            self._closing,
        )
        if connection_state == self._last_connection_state:
            return
        self._last_connection_state = connection_state
        self.connect_button.configure(
            state=(
                "disabled"
                if connected
                or self._connection_pending
                or self._disconnect_pending
                or self._closing
                else "normal"
            )
        )
        self.disconnect_button.configure(
            state="normal" if connected and not self._disconnect_pending else "disabled"
        )
        self.baseline_button.configure(
            state=(
                "normal"
                if connected
                and not feedback_recovering
                and not faulted
                and not self.session.armed
                and not self._baseline_pending
                and not self._disconnect_pending
                and not self.session.running
                and not self._calibration_pending
                else "disabled"
            )
        )
        self.arm_button.configure(
            state=(
                "normal"
                if connected
                and not self.session.armed
                and not self.session.running
                and not feedback_recovering
                and not faulted
                and not self._baseline_pending
                and not self._disconnect_pending
                and not self._calibration_pending
                else "disabled"
            )
        )
        self.start_button.configure(
            state=(
                "normal"
                if connected
                and self.session.armed
                and not self.session.running
                and not feedback_recovering
                and not faulted
                and not self._disconnect_pending
                and not self._calibration_pending
                else "disabled"
            )
        )
        self.stop_button.configure(
            state=(
                "normal"
                if connected and not self._stop_all_pending and not self._disconnect_pending
                else "disabled"
            )
        )
        speed_enabled = (
            connected
            and not self.session.armed
            and not self.session.running
            and not self.session.faulted
            and not self._command_pending
            and not self._stop_all_pending
            and not self._disconnect_pending
            and not self._calibration_pending
        )
        if not connected:
            speed_reason = "connect first"
        elif faulted:
            speed_reason = "faulted; disconnect and reconnect"
        elif self.session.armed or self.session.running:
            speed_reason = "stop force control or press STOP ALL"
        elif feedback_recovering or feedback_errors:
            speed_reason = (
                "ready; force feedback unavailable, manual speed has no force protection"
            )
        else:
            speed_reason = "ready"
        self.speed_status_var.set(f"Speed control: {speed_reason}")
        for servo_id, buttons in self._speed_buttons.items():
            for button in buttons:
                button.configure(state="normal" if speed_enabled else "disabled")
        gui_calibration_reason = None
        if self._calibration_pending:
            gui_calibration_reason = "calibration in progress"
        elif self._command_pending:
            gui_calibration_reason = "wait for the speed command to finish"
        elif self._baseline_pending:
            gui_calibration_reason = "wait for baseline refresh"
        elif self._disconnect_pending:
            gui_calibration_reason = "disconnecting"

        all_calibration_reason = (
            gui_calibration_reason or calibration_readiness["all_reason"]
        )
        self._calibrate_all_button.configure(
            state="normal" if all_calibration_reason is None else "disabled"
        )
        ready_ids = []
        for servo_id in SERVO_IDS:
            row_reason = (
                gui_calibration_reason
                or calibration_readiness["by_servo"][servo_id]
            )
            row_state = "normal" if row_reason is None else "disabled"
            self._calibrate_buttons[servo_id].configure(state=row_state)
            self._remember_calibration_buttons[servo_id].configure(state=row_state)
            if row_reason is None:
                ready_ids.append(servo_id)

        if all_calibration_reason is None:
            if feedback_errors:
                calibration_status = (
                    "ready; unstable feedback for "
                    + ", ".join(f"ID{servo_id}" for servo_id in sorted(feedback_errors))
                    + " (calibration will wait for valid samples)"
                )
            else:
                calibration_status = "ready"
        elif ready_ids:
            calibration_status = (
                "ready for " + ", ".join(f"ID{servo_id}" for servo_id in ready_ids)
                + f"; all unavailable: {all_calibration_reason}"
            )
        else:
            calibration_status = all_calibration_reason
        self.calibration_status_var.set(
            f"Torque calibration: {calibration_status}"
        )
        target_enabled = (
            connected
            and not faulted
            and not feedback_recovering
            and not self._disconnect_pending
        )
        for button in self._target_buttons:
            button.configure(state="normal" if target_enabled else "disabled")
        for button in self._stop_servo_buttons:
            button.configure(
                state="normal"
                if connected and not self._disconnect_pending
                else "disabled"
            )
        if not connected:
            for variables in self._row_vars.values():
                variables["actual"].set("--")
                variables["target"].set("--")
                variables["error"].set("--")
                variables["speed"].set("--")
                variables["state"].set("disconnected")
                variables["baseline"].set("--")
            for variables in self._speed_row_vars.values():
                variables["force"].set("--")
                variables["position"].set("--")
                variables["encoder"].set("--")
                variables["delta"].set("--")
                variables["speed"].set("--")
                variables["state"].set("disconnected")
            for variables in self._calibration_row_vars.values():
                for variable in variables.values():
                    variable.set("--")

    def _poll_session(self) -> None:
        if self._closing:
            return
        try:
            events = self.session.drain_events(limit=100)
            for event in events:
                self._append_log(event)
                self._set_status(event)
            samples = self.session.drain_samples()
            servo_diagnostic_samples = self.session.drain_servo_diagnostic_samples()
            encoder_diagnostic_samples = (
                self.session.drain_encoder_diagnostic_samples()
            )
            if self._plot_clear_pending:
                self._clear_plot()
                self._plot_clear_pending = False
                samples = []
                servo_diagnostic_samples = []
                encoder_diagnostic_samples = []
            latest_servo_diagnostics = {}
            for sample in servo_diagnostic_samples:
                self._servo_diagnostic_history[sample.channel_id].append(sample)
                if self._servo_plot_start_time is None:
                    self._servo_plot_start_time = sample.received_at
                latest_servo_diagnostics[sample.channel_id] = sample
            latest_encoder_diagnostics = {}
            for sample in encoder_diagnostic_samples:
                self._encoder_diagnostic_history[sample.channel_id].append(sample)
                if self._encoder_plot_start_time is None:
                    self._encoder_plot_start_time = sample.received_at
                latest_encoder_diagnostics[sample.channel_id] = sample
            latest_samples = {}
            for sample in samples:
                self._history[sample.servo_id].append(sample)
                if self._plot_start_time is None:
                    self._plot_start_time = sample.received_at
                latest_samples[sample.servo_id] = sample
            selected_tab = self.notebook.select()
            force_tab_visible = selected_tab == str(self._control_tab)
            speed_tab_visible = selected_tab == str(self._speed_tab)
            calibration_tab_visible = selected_tab == str(self._calibration_tab)
            logger_tab_visible = selected_tab == str(self._logger_tab)
            if speed_tab_visible:
                for sample in latest_servo_diagnostics.values():
                    self._speed_row_vars[sample.channel_id]["position"].set(
                        f"{sample.value:.0f}"
                    )
                for sample in latest_encoder_diagnostics.values():
                    self._speed_row_vars[sample.channel_id]["encoder"].set(
                        f"{sample.value:.4f}"
                    )
            for sample in latest_samples.values():
                if force_tab_visible:
                    variables = self._row_vars[sample.servo_id]
                    variables["actual"].set(f"{sample.force_n:.3f}")
                    variables["target"].set(f"{sample.target_force_n:.3f}")
                    variables["error"].set(f"{sample.force_error_n:+.3f}")
                    variables["speed"].set(str(sample.speed_command))
                    variables["state"].set(
                        "running" if sample.active else "monitoring"
                    )
                if speed_tab_visible:
                    speed_variables = self._speed_row_vars[sample.servo_id]
                    speed_variables["force"].set(f"{sample.force_n:.3f}")
                    speed_variables["delta"].set(f"{sample.angle_delta_deg:.4f}")
                    speed_variables["speed"].set(str(sample.speed_command))
                    speed_variables["state"].set(
                        "force PID" if sample.active
                        else ("manual speed" if sample.speed_command else "monitoring")
                    )
                if calibration_tab_visible:
                    calibration_variables = self._calibration_row_vars[sample.servo_id]
                    calibration_variables["servo_position"].set(
                        f"{sample.position:.0f}"
                    )
                    calibration_variables["encoder_position"].set(
                        f"{sample.encoder_position_deg:.4f}"
                    )
                    calibration_variables["delta"].set(
                        f"{sample.angle_delta_deg:.4f}"
                    )
            feedback_errors = self.session.feedback_errors_snapshot()
            manual_speeds = self.session.manual_speed_snapshot()
            for servo_id in feedback_errors:
                if force_tab_visible:
                    row_variables = self._row_vars[servo_id]
                    row_variables["actual"].set("--")
                    row_variables["error"].set("--")
                    row_variables["speed"].set("--")
                    row_variables["state"].set(
                        "sync unavailable"
                        if "synchronized feedback unavailable"
                        in feedback_errors[servo_id]
                        else "feedback unavailable"
                    )
                if speed_tab_visible:
                    speed_variables = self._speed_row_vars[servo_id]
                    speed_variables["force"].set("--")
                    speed_variables["delta"].set("--")
                    speed_variables["speed"].set(str(manual_speeds.get(servo_id, 0)))
                    speed_variables["state"].set(
                        "manual speed; force unavailable"
                        if manual_speeds.get(servo_id, 0)
                        else (
                            "sync unavailable"
                            if "synchronized feedback unavailable"
                            in feedback_errors[servo_id]
                            else "feedback unavailable"
                        )
                    )
            if calibration_tab_visible:
                calibration_state = self.session.calibration_state_snapshot()
                for servo_id, state in calibration_state.items():
                    variables = self._calibration_row_vars[servo_id]
                    variables["slope"].set(f"{state['slope_nmm_per_deg']:.8g}")
                    variables["intercept"].set(f"{state['intercept_nmm']:.8g}")
                    remembered = state["remembered_encoder_position_deg"]
                    variables["remembered"].set(
                        "--" if remembered is None else f"{remembered:.4f}"
                    )
            if force_tab_visible:
                for servo_id, baseline in self.session.baseline_snapshot().items():
                    if baseline is not None:
                        self._row_vars[servo_id]["baseline"].set(f"{baseline:.3f}")
            self._set_connection_states(self.session.connected)
            if self.session.faulted:
                fault_message = self.session.fault_message
                if fault_message:
                    self._set_status(f"FAULT: {fault_message}")
                    if fault_message != self._shown_fault_message:
                        self._shown_fault_message = fault_message
                        messagebox.showerror(
                            "Force control safety stop",
                            "All servos were stopped.\n\n" + fault_message,
                        )
                elif not events:
                    self._set_status("FAULT: disconnect and reconnect")
            else:
                self._shown_fault_message = None
            if (
                logger_tab_visible
                and time.monotonic() - self._last_logger_update >= 0.5
            ):
                self._update_logger()
                self._last_logger_update = time.monotonic()
            selected_plot_index = self.plot_notebook.index(
                self.plot_notebook.select()
            )
            plot_samples_received = bool(
                samples
                if selected_plot_index == 0
                else (
                    servo_diagnostic_samples
                    if selected_plot_index == 1
                    else encoder_diagnostic_samples
                )
            )
            if plot_samples_received and (
                time.monotonic() - self._last_plot_update
                >= PLOT_INTERVAL_MS / 1000.0
            ):
                self._update_plot()
                self._last_plot_update = time.monotonic()
        except Exception as exc:
            self._set_status(f"GUI update error: {exc}")
        finally:
            if not self._closing:
                self._poll_after_id = self.after(POLL_INTERVAL_MS, self._poll_session)

    def _clear_plot(self) -> None:
        for history in self._history.values():
            history.clear()
        for history in self._servo_diagnostic_history.values():
            history.clear()
        for history in self._encoder_diagnostic_history.values():
            history.clear()
        self._plot_start_time = None
        self._servo_plot_start_time = None
        self._encoder_plot_start_time = None
        self._last_plot_update = 0.0
        selected_panel = self._selected_plot_panel()
        for panel in self._plot_panels:
            panel.clear(redraw=panel is selected_panel)

    def _refresh_plot(self) -> None:
        self._clear_plot()
        self._set_status("Plot refreshed. Waiting for the next sample from time 0.")

    def _update_plot(self) -> None:
        panel, history, start_time = self._selected_plot_data()
        if start_time is None:
            return
        panel.update(history, start_time)

    def _selected_plot_panel(self):
        selected_index = self.plot_notebook.index(self.plot_notebook.select())
        return self._plot_panels[selected_index]

    def _selected_plot_data(self):
        selected_index = self.plot_notebook.index(self.plot_notebook.select())
        if selected_index == 0:
            return self.shared_plot_panel, self._history, self._plot_start_time
        if selected_index == 1:
            return (
                self.servo_plot_panel,
                self._servo_diagnostic_history,
                self._servo_plot_start_time,
            )
        return (
            self.encoder_plot_panel,
            self._encoder_diagnostic_history,
            self._encoder_plot_start_time,
        )

    def _on_plot_tab_changed(self, _event=None) -> None:
        if self._selected_plot_data()[2] is not None:
            self._last_plot_update = 0.0
            self._update_plot()

    def _on_close(self) -> None:
        self._closing = True
        if self._poll_after_id is not None:
            try:
                self.after_cancel(self._poll_after_id)
            except tk.TclError:
                pass
            self._poll_after_id = None
        try:
            if (
                self._disconnect_thread is not None
                and self._disconnect_thread.is_alive()
                and self._disconnect_thread is not threading.current_thread()
            ):
                self._disconnect_thread.join(timeout=2.0)
            self.session.shutdown()
        finally:
            for thread in (
                self._baseline_thread,
                self._connect_thread,
                self._command_thread,
                self._stop_all_thread,
                self._disconnect_thread,
                self._calibration_thread,
            ):
                if thread is not None and thread is not threading.current_thread():
                    thread.join(timeout=2.0)
            self.destroy()


def main() -> None:
    config = load_gui_force_config(DEFAULT_FORCE_CONFIG_PATH)
    app = ForceControlApp(config)
    app.mainloop()
