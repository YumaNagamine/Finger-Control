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
    ForceControlSession,
    GuiForceConfig,
    load_gui_force_config,
)


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
        self._plot_lines = {}
        self._plot_start_time = None
        self._last_plot_update = 0.0
        self._last_logger_update = 0.0
        self._connection_pending = False
        self._row_vars = {}
        self._speed_row_vars = {}
        self._speed_buttons = {}
        self._build_variables()
        self._build_widgets()
        self.after(100, self._poll_session)

    def _build_variables(self) -> None:
        self.status_var = tk.StringVar(value="Disconnected")
        self.speed_status_var = tk.StringVar(value="Speed control: connect first")
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
        self.max_speed_var = tk.StringVar(value="20.0")
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
                "speed": tk.StringVar(value="--"),
                "state": tk.StringVar(value="disconnected"),
            }

    def _build_widgets(self) -> None:
        self.columnconfigure(0, weight=1)
        self.rowconfigure(0, weight=0)
        self.rowconfigure(1, weight=3)
        self.rowconfigure(2, weight=2)

        self.notebook = ttk.Notebook(self)
        self.notebook.grid(row=1, column=0, sticky="nsew")
        control_tab = ttk.Frame(self.notebook)
        speed_tab = ttk.Frame(self.notebook)
        logger_tab = ttk.Frame(self.notebook)
        self.notebook.add(control_tab, text="Force control")
        self.notebook.add(speed_tab, text="Speed control")
        self.notebook.add(logger_tab, text="Logger / diagnostics")
        control_tab.columnconfigure(0, weight=1)
        control_tab.rowconfigure(1, weight=1)

        connection = ttk.LabelFrame(self, text="Connection and session")
        connection.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
        for column in range(12):
            connection.columnconfigure(column, weight=1 if column in (1, 3, 5, 7) else 0)
        self._entry(connection, "Servo port", self.servo_port_var, 0, 0)
        self._entry(connection, "Encoder port", self.encoder_port_var, 0, 2)
        self._entry(connection, "Calibration dir", self.calibration_dir_var, 0, 4, width=42)
        self._entry(connection, "Zero offset", self.zero_offset_var, 0, 6, width=42)
        self.connect_button = ttk.Button(connection, text="Connect", command=self._connect)
        self.connect_button.grid(row=0, column=8, padx=4, pady=4)
        self.disconnect_button = ttk.Button(
            connection, text="Disconnect", command=self._disconnect, state="disabled"
        )
        self.disconnect_button.grid(row=0, column=9, padx=4, pady=4)
        self.baseline_button = ttk.Button(
            connection, text="Refresh baseline", command=self._refresh_baseline, state="disabled"
        )
        self.baseline_button.grid(row=0, column=10, padx=4, pady=4)
        ttk.Label(connection, textvariable=self.status_var).grid(
            row=1, column=0, columnspan=12, sticky="w", padx=4, pady=(0, 4)
        )

        parameters = ttk.LabelFrame(control_tab, text="Controller parameters")
        parameters.grid(row=0, column=0, sticky="ew", padx=8, pady=4)
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
        self.arm_button = ttk.Button(parameters, text="Arm", command=self._arm, state="disabled")
        self.arm_button.grid(row=1, column=0, padx=4, pady=4, sticky="w")
        self.start_button = ttk.Button(
            parameters, text="Start control", command=self._start, state="disabled"
        )
        self.start_button.grid(row=1, column=1, columnspan=2, padx=4, pady=4, sticky="w")
        self.stop_button = ttk.Button(
            parameters, text="STOP ALL", command=self._stop_all, state="disabled"
        )
        self.stop_button.grid(row=1, column=3, columnspan=2, padx=4, pady=4, sticky="w")
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

        controls = ttk.LabelFrame(control_tab, text="Per-tendon force control")
        controls.grid(row=1, column=0, sticky="ew", padx=8, pady=4)
        headers = ["ID", "Tendon", "Actual N", "Target N", "Error N", "Speed", "Baseline N", "State", "Targets"]
        for column, header in enumerate(headers):
            ttk.Label(controls, text=header).grid(row=0, column=column, padx=4, pady=2, sticky="w")
        for row, (servo_id, tendon) in enumerate(zip(SERVO_IDS, SERVO_NAMES), start=1):
            variables = self._row_vars[servo_id]
            ttk.Label(controls, text=str(servo_id)).grid(row=row, column=0, padx=4, pady=2)
            ttk.Label(controls, text=tendon, width=8).grid(row=row, column=1, padx=4, pady=2, sticky="w")
            ttk.Label(controls, textvariable=variables["actual"], width=10).grid(row=row, column=2, padx=4, pady=2)
            ttk.Label(controls, textvariable=variables["target"], width=10).grid(row=row, column=3, padx=4, pady=2)
            ttk.Label(controls, textvariable=variables["error"], width=10).grid(row=row, column=4, padx=4, pady=2)
            ttk.Label(controls, textvariable=variables["speed"], width=8).grid(row=row, column=5, padx=4, pady=2)
            ttk.Label(controls, textvariable=variables["baseline"], width=10).grid(row=row, column=6, padx=4, pady=2)
            ttk.Label(controls, textvariable=variables["state"], width=14).grid(row=row, column=7, padx=4, pady=2)
            target_frame = ttk.Frame(controls)
            target_frame.grid(row=row, column=8, padx=4, pady=2, sticky="w")
            for value in self.config_model.target_force_levels_by_tendon[tendon]:
                ttk.Button(
                    target_frame,
                    text=f"{value:g}",
                    width=5,
                    command=lambda sid=servo_id, target=value: self._set_target(sid, target),
                ).pack(side="left", padx=1)
            ttk.Button(
                target_frame,
                text="Stop",
                width=5,
                command=lambda sid=servo_id: self._stop_servo(sid),
            ).pack(side="left", padx=(4, 1))

        self._build_speed_widgets(speed_tab)
        self._build_logger_widgets(logger_tab)

        plot_frame = ttk.LabelFrame(self, text="Realtime force")
        plot_frame.grid(row=2, column=0, sticky="nsew", padx=8, pady=(4, 8))
        plot_frame.rowconfigure(0, weight=1)
        plot_frame.columnconfigure(0, weight=1)
        self.figure = Figure(figsize=(12, 6), dpi=100)
        self.axes = self.figure.subplots(2, 3).flatten()
        for axis, servo_id, tendon in zip(self.axes, SERVO_IDS, SERVO_NAMES):
            axis.set_title(f"ID{servo_id} {tendon}")
            axis.set_xlabel("time (s)")
            axis.set_ylabel("force (N)")
            tendon_max = max(self.config_model.target_force_levels_by_tendon[tendon])
            axis.set_ylim(0.0, max(2.2, tendon_max * 1.2))
            actual_line, = axis.plot([], [], "-", label="actual")
            target_line, = axis.plot([], [], "--", label="target")
            axis.legend(loc="upper left", fontsize=8)
            self._plot_lines[servo_id] = (actual_line, target_line)
        self.canvas = FigureCanvasTkAgg(self.figure, master=plot_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().grid(row=0, column=0, sticky="nsew")

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
        ).grid(row=0, column=0, sticky="w", padx=8, pady=(8, 4))
        ttk.Label(parent, textvariable=self.speed_status_var).grid(
            row=1, column=0, sticky="w", padx=8, pady=(0, 4)
        )

        table = ttk.LabelFrame(parent, text="Per-servo speed control and realtime force")
        table.grid(row=2, column=0, sticky="nsew", padx=8, pady=(4, 8))
        headers = ["ID", "Tendon", "Force (N)", "Position", "Commanded speed", "State", "Speed"]
        for column, header in enumerate(headers):
            ttk.Label(table, text=header).grid(
                row=0, column=column, padx=4, pady=4, sticky="w"
            )

        for row, (servo_id, tendon) in enumerate(zip(SERVO_IDS, SERVO_NAMES), start=1):
            variables = self._speed_row_vars[servo_id]
            ttk.Label(table, text=str(servo_id), width=5).grid(
                row=row, column=0, padx=4, pady=4
            )
            ttk.Label(table, text=tendon, width=8).grid(
                row=row, column=1, padx=4, pady=4, sticky="w"
            )
            ttk.Label(table, textvariable=variables["force"], width=10).grid(
                row=row, column=2, padx=4, pady=4, sticky="w"
            )
            ttk.Label(table, textvariable=variables["position"], width=10).grid(
                row=row, column=3, padx=4, pady=4, sticky="w"
            )
            ttk.Label(table, textvariable=variables["speed"], width=16).grid(
                row=row, column=4, padx=4, pady=4, sticky="w"
            )
            ttk.Label(table, textvariable=variables["state"], width=14).grid(
                row=row, column=5, padx=4, pady=4, sticky="w"
            )
            button_frame = ttk.Frame(table)
            button_frame.grid(row=row, column=6, padx=4, pady=4, sticky="w")
            self._speed_buttons[servo_id] = []
            for speed in MANUAL_SPEED_PRESETS:
                button = ttk.Button(
                    button_frame,
                    text=str(speed),
                    width=6,
                    command=lambda sid=servo_id, value=speed: self._set_manual_speed(
                        sid, value
                    ),
                )
                button.pack(side="left", padx=1)
                self._speed_buttons[servo_id].append(button)

        ttk.Button(
            parent,
            text="STOP ALL",
            command=self._stop_all,
        ).grid(row=3, column=0, sticky="w", padx=8, pady=(0, 8))

    def _build_logger_widgets(self, parent) -> None:
        parent.columnconfigure(0, weight=1)
        parent.rowconfigure(1, weight=1)

        summary = ttk.LabelFrame(parent, text="Session diagnostics")
        summary.grid(row=0, column=0, sticky="ew", padx=8, pady=(8, 4))
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

        table_frame = ttk.LabelFrame(parent, text="Servo / tendon / calibration mapping")
        table_frame.grid(row=1, column=0, sticky="nsew", padx=8, pady=(4, 8))
        table_frame.columnconfigure(0, weight=1)
        table_frame.rowconfigure(0, weight=1)
        columns = (
            "servo_id",
            "tendon",
            "cs",
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

    @staticmethod
    def _entry(parent, label, variable, row, column, *, width=18):
        ttk.Label(parent, text=label).grid(row=row, column=column, padx=(4, 2), pady=4, sticky="w")
        ttk.Entry(parent, textvariable=variable, width=width).grid(
            row=row, column=column + 1, padx=(0, 6), pady=4, sticky="ew"
        )

    def _set_status(self, text: str) -> None:
        self.status_var.set(text)

    def _connect(self) -> None:
        if not self.servo_port_var.get().strip() or not self.encoder_port_var.get().strip():
            messagebox.showerror("Connection", "Servo and encoder ports are required.")
            return
        self._connection_pending = True
        self.connect_button.configure(state="disabled")
        self._set_status("Connecting; no motion command is sent...")

        def worker():
            try:
                self.session.connect(
                    self.servo_port_var.get().strip(),
                    self.encoder_port_var.get().strip(),
                    calibration_dir=Path(self.calibration_dir_var.get().strip()),
                    zero_offset_path=Path(self.zero_offset_var.get().strip()),
                )
            except Exception as exc:
                self.session._event_queue.put(f"ERROR: {exc}")
            finally:
                self._connection_pending = False

        thread = threading.Thread(target=worker, name="force-gui-connect", daemon=True)
        thread.start()

    def _disconnect(self) -> None:
        self.session.disconnect()
        self._set_status("Disconnected")
        self._set_connection_states(False)

    def _refresh_baseline(self) -> None:
        if not self.session.connected:
            return
        self.baseline_button.configure(state="disabled")

        def worker():
            try:
                self.session.capture_baseline()
                self.session._event_queue.put("Baseline refreshed.")
            except Exception as exc:
                self.session._event_queue.put(f"ERROR: baseline capture failed: {exc}")

        threading.Thread(target=worker, name="force-gui-baseline", daemon=True).start()

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
        self.session.stop_all()

    def _set_manual_speed(self, servo_id: int, speed: int) -> None:
        try:
            self.session.set_manual_speed(servo_id, speed)
        except Exception as exc:
            messagebox.showerror("Speed control", str(exc))

    def _stop_servo(self, servo_id: int) -> None:
        try:
            self.session.stop_servo(servo_id)
        except Exception as exc:
            messagebox.showerror("Stop servo", str(exc))

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
        self.connect_button.configure(
            state="disabled" if connected or self._connection_pending else "normal"
        )
        self.disconnect_button.configure(state="normal" if connected else "disabled")
        self.baseline_button.configure(
            state="normal" if connected and not feedback_recovering else "disabled"
        )
        self.arm_button.configure(
            state="normal" if connected and not feedback_recovering else "disabled"
        )
        self.start_button.configure(
            state="normal" if connected and not feedback_recovering else "disabled"
        )
        self.stop_button.configure(state="normal" if connected else "disabled")
        speed_enabled = (
            connected
            and not self.session.armed
            and not self.session.running
            and not self.session.faulted
            and not feedback_recovering
        )
        if not connected:
            speed_reason = "connect first"
        elif self.session.faulted:
            speed_reason = "faulted; disconnect and reconnect"
        elif self.session.armed or self.session.running:
            speed_reason = "stop force control or press STOP ALL"
        elif self.session.manual_feedback_recovering:
            speed_reason = "feedback recovering; motion commands blocked"
        else:
            speed_reason = "ready"
        self.speed_status_var.set(f"Speed control: {speed_reason}")
        for buttons in self._speed_buttons.values():
            for button in buttons:
                button.configure(state="normal" if speed_enabled else "disabled")
        if not connected:
            for variables in self._speed_row_vars.values():
                variables["force"].set("--")
                variables["position"].set("--")
                variables["speed"].set("--")
                variables["state"].set("disconnected")

    def _poll_session(self) -> None:
        events = self.session.drain_events()
        for event in events:
            self._set_status(event)
        samples = self.session.drain_samples()
        for sample in samples:
            self._history[sample.servo_id].append(sample)
            variables = self._row_vars[sample.servo_id]
            variables["actual"].set(f"{sample.force_n:.3f}")
            variables["target"].set(f"{sample.target_force_n:.3f}")
            variables["error"].set(f"{sample.force_error_n:+.3f}")
            variables["speed"].set(str(sample.speed_command))
            variables["state"].set("running" if sample.active else "monitoring")
            speed_variables = self._speed_row_vars[sample.servo_id]
            speed_variables["force"].set(f"{sample.force_n:.3f}")
            speed_variables["position"].set(f"{sample.position:.0f}")
            speed_variables["speed"].set(str(sample.speed_command))
            speed_variables["state"].set(
                "force PID" if sample.active
                else ("manual speed" if sample.speed_command else "monitoring")
            )
        for servo_id, baseline in self.session.baseline_snapshot().items():
            if baseline is not None:
                self._row_vars[servo_id]["baseline"].set(f"{baseline:.3f}")
        self._set_connection_states(self.session.connected)
        if self.session.faulted:
            fault_message = self.session.fault_message
            if fault_message:
                self._set_status(f"FAULT: {fault_message}")
            elif not events:
                self._set_status("FAULT: disconnect and reconnect")
        if time.monotonic() - self._last_logger_update >= 0.5:
            self._update_logger()
            self._last_logger_update = time.monotonic()
        if samples:
            self._update_plot()
        self.after(100, self._poll_session)

    def _update_plot(self) -> None:
        all_samples = [sample for history in self._history.values() for sample in history]
        if not all_samples:
            return
        if self._plot_start_time is None:
            self._plot_start_time = min(sample.received_at for sample in all_samples)
        for axis, servo_id in zip(self.axes, SERVO_IDS):
            history = list(self._history[servo_id])
            if not history:
                continue
            times = [sample.received_at - self._plot_start_time for sample in history]
            actual = [sample.force_n for sample in history]
            target = [sample.target_force_n for sample in history]
            actual_line, target_line = self._plot_lines[servo_id]
            actual_line.set_data(times, actual)
            target_line.set_data(times, target)
            end_time = max(times[-1], 10.0)
            axis.set_xlim(max(0.0, end_time - 10.0), max(10.0, end_time))
            low = min(0.0, min(actual) * 1.1)
            tendon = SERVO_NAMES[servo_id - 1]
            tendon_max = max(self.config_model.target_force_levels_by_tendon[tendon])
            high = max(tendon_max * 1.1, max(actual + target + [0.0]) * 1.2)
            if high <= low:
                high = low + 1.0
            axis.set_ylim(low, high)
        self.canvas.draw_idle()

    def _on_close(self) -> None:
        try:
            self.session.shutdown()
        finally:
            self.destroy()


def main() -> None:
    config = load_gui_force_config(DEFAULT_FORCE_CONFIG_PATH)
    app = ForceControlApp(config)
    app.mainloop()
