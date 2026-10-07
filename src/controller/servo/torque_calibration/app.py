"""Tkinter GUI for fixed-slope force-gauge torque-intercept calibration."""

from __future__ import annotations

import queue
import math
import threading
import tkinter as tk
from pathlib import Path
from tkinter import messagebox, ttk
from typing import Any, Callable

from controller.servo.hardware_session import API_ROOT, ServoHardwareSession
from controller.servo.torque_calibration.calibration import (
    calculate_calibration_point,
    fit_fixed_slope_intercept,
    load_servo_calibration_models,
)
from controller.servo.torque_calibration.hardware import capture_angle_window
from controller.servo.torque_calibration.storage import (
    DEFAULT_INTERCEPT_CALIBRATION_PATH,
    load_torque_intercept_calibration,
    save_torque_intercept_calibration,
)


DEFAULT_CALIBRATION_DIR = API_ROOT / "calibration" / "current"
SERVO_IDS = tuple(range(1, 7))
SERVO_NAMES_BY_ID = ("ED", "DI", "FDS", "FDP", "LU", "PI")
MAX_POINTS_PER_SERVO = 1000
SPEED_PRESETS = (-500, -300, -100, 100, 300, 500)


class TorqueCalibrationApp:
    def __init__(self, root: tk.Tk) -> None:
        self.root = root
        self.root.title("Servo torque intercept calibration")
        self.root.geometry("1040x760")
        self.root.minsize(900, 650)

        self.models = load_servo_calibration_models(DEFAULT_CALIBRATION_DIR)
        self.existing_records = load_torque_intercept_calibration(
            DEFAULT_INTERCEPT_CALIBRATION_PATH
        )
        self.points_by_servo: dict[int, list[dict[str, Any]]] = {
            servo_id: [] for servo_id in SERVO_IDS
        }
        self.fits_by_servo: dict[int, dict[str, Any]] = {}
        self.session: ServoHardwareSession | None = None
        self._pending_session: ServoHardwareSession | None = None
        self._closing_session: ServoHardwareSession | None = None
        self.busy = False
        self._jobs: queue.Queue[tuple[str, Any]] = queue.Queue(maxsize=8)
        self._worker: threading.Thread | None = None
        self._worker_stop: threading.Event | None = None
        self._poll_after_id: str | None = None
        self._close_requested = False
        self._destroyed = False
        self._health_error: Exception | None = None

        self.servo_port_var = tk.StringVar(value="COM3")
        self.encoder_port_var = tk.StringVar(value="COM5")
        self.servo_id_var = tk.StringVar(value="1")
        self.force_var = tk.StringVar()
        self.direction_var = tk.StringVar(value="+1")
        self.duration_var = tk.StringVar(value="1.0")
        self.status_var = tk.StringVar(value="Disconnected")
        self.model_var = tk.StringVar()
        self.current_b_var = tk.StringVar()
        self.pending_b_var = tk.StringVar(value="not calculated")
        self.pending_count_var = tk.StringVar(value="0 / 6 servos ready")
        self.speed_status_var = tk.StringVar(value="Disconnected")
        self.speed_command_vars = {
            servo_id: tk.StringVar(value="0") for servo_id in SERVO_IDS
        }
        self._speed_command_values = {servo_id: 0 for servo_id in SERVO_IDS}
        self._speed_buttons: dict[int, list[ttk.Button]] = {}
        self.stop_all_speed_button: ttk.Button | None = None

        self._build_ui()
        self._update_speed_controls()
        self._refresh_selected_servo()
        self._refresh_pending_table()
        self.root.protocol("WM_DELETE_WINDOW", self.close)
        self._schedule_job_poll()

    def _build_ui(self) -> None:
        root = self.root
        root.columnconfigure(0, weight=1)
        root.rowconfigure(1, weight=1)

        connection = ttk.LabelFrame(root, text="Connection")
        connection.grid(row=0, column=0, sticky="ew", padx=10, pady=8)
        connection.columnconfigure(7, weight=1)
        ttk.Label(connection, text="Servo port").grid(row=0, column=0, padx=5, pady=5)
        ttk.Entry(connection, textvariable=self.servo_port_var, width=12).grid(
            row=0, column=1, padx=5, pady=5
        )
        ttk.Label(connection, text="Encoder port").grid(row=0, column=2, padx=5, pady=5)
        ttk.Entry(connection, textvariable=self.encoder_port_var, width=12).grid(
            row=0, column=3, padx=5, pady=5
        )
        self.connect_button = ttk.Button(
            connection, text="Connect", command=self._connect
        )
        self.connect_button.grid(row=0, column=4, padx=5, pady=5)
        self.disconnect_button = ttk.Button(
            connection, text="Disconnect", command=self._disconnect, state="disabled"
        )
        self.disconnect_button.grid(row=0, column=5, padx=5, pady=5)
        ttk.Label(connection, textvariable=self.status_var).grid(
            row=0, column=6, columnspan=2, sticky="w", padx=10, pady=5
        )

        notebook = ttk.Notebook(root)
        notebook.grid(row=1, column=0, sticky="nsew", padx=10, pady=(0, 8))
        calibration_tab = ttk.Frame(notebook)
        speed_tab = ttk.Frame(notebook)
        notebook.add(calibration_tab, text="Calibration")
        notebook.add(speed_tab, text="Motor speed")
        calibration_tab.columnconfigure(0, weight=1)
        calibration_tab.rowconfigure(3, weight=1)

        selection = ttk.LabelFrame(calibration_tab, text="Selected motor")
        selection.grid(row=1, column=0, sticky="ew", padx=10, pady=4)
        ttk.Label(selection, text="Servo ID").grid(row=0, column=0, padx=5, pady=5)
        self.servo_id_combo = ttk.Combobox(
            selection,
            textvariable=self.servo_id_var,
            values=[str(value) for value in SERVO_IDS],
            state="readonly",
            width=8,
        )
        self.servo_id_combo.grid(row=0, column=1, padx=5, pady=5)
        self.servo_id_combo.bind("<<ComboboxSelected>>", self._on_servo_selected)
        ttk.Label(selection, textvariable=self.model_var).grid(
            row=0, column=2, sticky="w", padx=12, pady=5
        )
        ttk.Label(selection, text="Current b").grid(row=0, column=3, padx=5, pady=5)
        ttk.Label(selection, textvariable=self.current_b_var).grid(
            row=0, column=4, sticky="w", padx=5, pady=5
        )
        ttk.Label(selection, text="New b").grid(row=0, column=5, padx=5, pady=5)
        ttk.Label(selection, textvariable=self.pending_b_var).grid(
            row=0, column=6, sticky="w", padx=5, pady=5
        )

        measurement = ttk.LabelFrame(calibration_tab, text="Add force-gauge point")
        measurement.grid(row=2, column=0, sticky="ew", padx=10, pady=4)
        ttk.Label(measurement, text="Force gauge [N]").grid(
            row=0, column=0, padx=5, pady=5
        )
        ttk.Entry(measurement, textvariable=self.force_var, width=12).grid(
            row=0, column=1, padx=5, pady=5
        )
        ttk.Label(measurement, text="Torque direction").grid(
            row=0, column=2, padx=5, pady=5
        )
        ttk.Combobox(
            measurement,
            textvariable=self.direction_var,
            values=("+1", "-1"),
            state="readonly",
            width=6,
        ).grid(row=0, column=3, padx=5, pady=5)
        ttk.Label(measurement, text="Capture [s]").grid(
            row=0, column=4, padx=5, pady=5
        )
        ttk.Entry(measurement, textvariable=self.duration_var, width=8).grid(
            row=0, column=5, padx=5, pady=5
        )
        self.capture_button = ttk.Button(
            measurement, text="Capture point", command=self._capture_point
        )
        self.capture_button.grid(row=0, column=6, padx=8, pady=5)
        self.calculate_button = ttk.Button(
            measurement, text="Calculate b", command=self._calculate_selected
        )
        self.calculate_button.grid(row=0, column=7, padx=5, pady=5)
        self.clear_button = ttk.Button(
            measurement, text="Clear selected", command=self._clear_selected
        )
        self.clear_button.grid(row=0, column=8, padx=5, pady=5)

        points_frame = ttk.LabelFrame(calibration_tab, text="Selected servo points")
        points_frame.grid(row=3, column=0, sticky="nsew", padx=10, pady=4)
        points_frame.rowconfigure(0, weight=1)
        points_frame.columnconfigure(0, weight=1)
        self.points_tree = ttk.Treeview(
            points_frame,
            columns=("number", "force", "direction", "delta", "torque", "b"),
            show="headings",
            height=10,
        )
        headings = {
            "number": "#",
            "force": "Force [N]",
            "direction": "Dir",
            "delta": "Delta [deg]",
            "torque": "Torque [Nmm]",
            "b": "b [Nmm]",
        }
        for column, heading in headings.items():
            self.points_tree.heading(column, text=heading)
            self.points_tree.column(column, width=120, anchor="center")
        self.points_tree.grid(row=0, column=0, sticky="nsew")
        points_scroll = ttk.Scrollbar(
            points_frame, orient="vertical", command=self.points_tree.yview
        )
        points_scroll.grid(row=0, column=1, sticky="ns")
        self.points_tree.configure(yscrollcommand=points_scroll.set)

        bottom = ttk.Frame(calibration_tab)
        bottom.grid(row=4, column=0, sticky="ew", padx=10, pady=8)
        bottom.columnconfigure(0, weight=1)
        ttk.Label(bottom, textvariable=self.pending_count_var).grid(
            row=0, column=0, sticky="w"
        )
        self.update_button = ttk.Button(
            bottom,
            text="Update calibrated intercepts for all six servos",
            command=self._update_all,
        )
        self.update_button.grid(row=0, column=1, padx=5)

        self.pending_tree = ttk.Treeview(
            calibration_tab,
            columns=("servo", "tendon", "points", "b", "spread"),
            show="headings",
            height=6,
        )
        calibration_tab.rowconfigure(5, weight=0)
        self.pending_tree.grid(row=5, column=0, sticky="ew", padx=10, pady=4)
        for column, heading in {
            "servo": "Servo ID",
            "tendon": "Tendon",
            "points": "Points",
            "b": "Calculated b [Nmm]",
            "spread": "Std [Nmm]",
        }.items():
            self.pending_tree.heading(column, text=heading)
            self.pending_tree.column(column, width=160, anchor="center")

        self._build_speed_tab(speed_tab)

    def _build_speed_tab(self, parent: ttk.Frame) -> None:
        parent.columnconfigure(0, weight=1)
        ttk.Label(
            parent,
            text=(
                "Raw motor speed commands. Positive and negative values select "
                "the motor direction. Stop all motors before calibration capture."
            ),
        ).grid(row=0, column=0, sticky="w", padx=8, pady=(8, 4))
        ttk.Label(parent, textvariable=self.speed_status_var).grid(
            row=1, column=0, sticky="w", padx=8, pady=(0, 4)
        )

        table = ttk.LabelFrame(parent, text="Per-motor speed control")
        table.grid(row=2, column=0, sticky="nsew", padx=8, pady=4)
        headers = ("Servo ID", "Motor", "Current command", "Speed")
        for column, heading in enumerate(headers):
            ttk.Label(table, text=heading).grid(
                row=0, column=column, padx=5, pady=5, sticky="w"
            )

        for row, (servo_id, motor_name) in enumerate(
            zip(SERVO_IDS, SERVO_NAMES_BY_ID), start=1
        ):
            ttk.Label(table, text=str(servo_id), width=8).grid(
                row=row, column=0, padx=5, pady=4
            )
            ttk.Label(table, text=motor_name, width=10).grid(
                row=row, column=1, padx=5, pady=4, sticky="w"
            )
            ttk.Label(
                table,
                textvariable=self.speed_command_vars[servo_id],
                width=16,
            ).grid(row=row, column=2, padx=5, pady=4, sticky="w")
            button_frame = ttk.Frame(table)
            button_frame.grid(row=row, column=3, padx=5, pady=4, sticky="w")
            buttons: list[ttk.Button] = []
            for speed in SPEED_PRESETS:
                button = ttk.Button(
                    button_frame,
                    text=f"{speed:+d}",
                    width=6,
                    command=lambda sid=servo_id, value=speed: self._set_motor_speed(
                        sid, value
                    ),
                )
                button.pack(side="left", padx=1)
                buttons.append(button)
            stop_button = ttk.Button(
                button_frame,
                text="STOP",
                width=6,
                command=lambda sid=servo_id: self._set_motor_speed(sid, 0),
            )
            stop_button.pack(side="left", padx=(5, 1))
            buttons.append(stop_button)
            self._speed_buttons[servo_id] = buttons

        self.stop_all_speed_button = ttk.Button(
            parent,
            text="STOP ALL MOTORS",
            command=self._stop_all_motors,
        )
        self.stop_all_speed_button.grid(row=3, column=0, sticky="w", padx=8, pady=8)

    def _selected_servo_id(self) -> int:
        return int(self.servo_id_var.get())

    def _refresh_selected_servo(self) -> None:
        servo_id = self._selected_servo_id()
        model = self.models[servo_id]
        tendon = SERVO_NAMES_BY_ID[servo_id - 1]
        self.model_var.set(
            f"{tendon} | k={float(model['slope_nmm_per_deg']):.8g} Nmm/deg | "
            f"CS{int(model['sensor_id'])} | arm={float(model['moment_arm_mm']):g} mm"
        )
        existing = self.existing_records.get(servo_id)
        self.current_b_var.set(
            "--" if existing is None else f"{float(existing['intercept_nmm']):.6g} Nmm"
        )
        fit = self.fits_by_servo.get(servo_id)
        self.pending_b_var.set(
            "not calculated" if fit is None else f"{float(fit['intercept_nmm']):.6g} Nmm"
        )
        self._refresh_points_tree()

    def _refresh_points_tree(self) -> None:
        for item in self.points_tree.get_children():
            self.points_tree.delete(item)
        servo_id = self._selected_servo_id()
        for index, point in enumerate(self.points_by_servo[servo_id], start=1):
            self.points_tree.insert(
                "",
                "end",
                values=(
                    index,
                    f"{float(point['force_n']):.5g}",
                    int(point["torque_direction"]),
                    f"{float(point['angle_delta_deg']):.6f}",
                    f"{float(point['torque_nmm']):.6f}",
                    f"{float(point['intercept_nmm']):.6f}",
                ),
            )

    def _refresh_pending_table(self) -> None:
        for item in self.pending_tree.get_children():
            self.pending_tree.delete(item)
        ready = 0
        for servo_id in SERVO_IDS:
            model = self.models[servo_id]
            fit = self.fits_by_servo.get(servo_id)
            if fit is not None:
                ready += 1
                values = (
                    servo_id,
                    SERVO_NAMES_BY_ID[servo_id - 1],
                    fit["point_count"],
                    f"{float(fit['intercept_nmm']):.6f}",
                    f"{float(fit['intercept_std_nmm']):.6f}",
                )
            else:
                values = (servo_id, SERVO_NAMES_BY_ID[servo_id - 1], "--", "--", "--")
            self.pending_tree.insert("", "end", values=values)
        self.pending_count_var.set(f"{ready} / 6 servos ready")
        self.update_button.configure(state="normal" if ready == 6 else "disabled")

    def _on_servo_selected(self, _event: Any = None) -> None:
        self._refresh_selected_servo()

    def _set_busy(self, busy: bool, status: str | None = None) -> None:
        self.busy = busy
        disabled = busy or self._close_requested
        state = "disabled" if disabled else "normal"
        self.capture_button.configure(state=state)
        self.calculate_button.configure(state=state)
        self.clear_button.configure(state=state)
        self.servo_id_combo.configure(state="disabled" if disabled else "readonly")
        self.connect_button.configure(state=state)
        self.update_button.configure(
            state=(
                "disabled"
                if disabled or len(self.fits_by_servo) != len(SERVO_IDS)
                else "normal"
            )
        )
        self.disconnect_button.configure(
            state=(
                "disabled"
                if disabled or self.session is None
                else "normal"
            )
        )
        if status is not None:
            self.status_var.set(status)
        self._update_speed_controls()

    def _update_speed_controls(self) -> None:
        enabled = (
            not self.busy
            and not self._close_requested
            and self.session is not None
            and self.session.connected
        )
        if self.session is None or not self.session.connected:
            status = "Disconnected"
        elif self.busy:
            status = "Unavailable while another operation is running"
        elif any(self._speed_command_values.values()):
            status = "Ready; one or more motors are moving"
        else:
            status = "Ready"
        self.speed_status_var.set(status)
        state = "normal" if enabled else "disabled"
        for buttons in self._speed_buttons.values():
            for button in buttons:
                button.configure(state=state)
        if self.stop_all_speed_button is not None:
            self.stop_all_speed_button.configure(state=state)

    def _reset_speed_commands(self) -> None:
        for servo_id in SERVO_IDS:
            self._speed_command_values[servo_id] = 0
            self.speed_command_vars[servo_id].set("0")

    def _schedule_job_poll(self, delay_ms: int = 100) -> None:
        if self._destroyed or self._poll_after_id is not None:
            return
        self._poll_after_id = self.root.after(delay_ms, self._poll_jobs)

    def _run_worker(
        self,
        work: Callable[[threading.Event], Any],
        success_message: str,
        *,
        kind: str,
        cleanup_on_error: Callable[[], None] | None = None,
    ) -> None:
        if self._worker is not None:
            raise RuntimeError("A background operation is already running")
        stop_event = threading.Event()
        self._worker_stop = stop_event

        def worker() -> None:
            try:
                result = work(stop_event)
            except Exception as exc:  # pragma: no cover - hardware errors are environment-specific
                if cleanup_on_error is not None:
                    try:
                        cleanup_on_error()
                    except Exception as cleanup_exc:
                        exc = RuntimeError(
                            f"{exc}; cleanup failed: {cleanup_exc}"
                        )
                self._jobs.put(("error", (kind, exc)))
            else:
                self._jobs.put(("success", (kind, success_message, result)))

        self._worker = threading.Thread(
            target=worker,
            name=f"torque-calibration-{kind}",
            daemon=True,
        )
        self._worker.start()

    def _poll_jobs(self) -> None:
        self._poll_after_id = None
        for _ in range(32):
            try:
                kind, payload = self._jobs.get_nowait()
            except queue.Empty:
                break

            # The worker has produced its only result.  Clear the reference
            # before handling it so shutdown can schedule cleanup safely.
            self._worker = None
            self._worker_stop = None
            if kind == "error":
                job_kind, error = payload
                self._handle_job_error(job_kind, error)
            else:
                job_kind, message, result = payload
                self._handle_job_success(job_kind, message, result)

        if not self._destroyed:
            self._poll_session_health()
            self._schedule_job_poll(0 if not self._jobs.empty() else 100)

    def _poll_session_health(self) -> None:
        if (
            self._close_requested
            or self.busy
            or self._worker is not None
            or self.session is None
        ):
            return
        session = self.session
        try:
            session.check_health()
            if session.latest_feedback(max_age_s=1.0) is None:
                raise TimeoutError("Servo telemetry became stale")
        except Exception as exc:
            self.session = None
            self._reset_speed_commands()
            self._health_error = exc
            self._set_busy(True, "Hardware communication lost; disconnecting...")
            self._run_worker(
                lambda _stop_event: session.close(),
                "Hardware disconnected; reconnect required",
                kind="recovery",
            )

    def _handle_job_error(self, kind: str, error: Exception) -> None:
        if kind in {"connect", "capture"}:
            self.session = None
            self._pending_session = None
            self._reset_speed_commands()

        if self._close_requested:
            self._begin_shutdown_cleanup()
            return

        self._set_busy(False, f"Error: {error}")
        messagebox.showerror("Torque calibration", str(error))

    def _handle_job_success(self, kind: str, message: str, result: Any) -> None:
        if self._close_requested:
            if isinstance(result, ServoHardwareSession):
                self._closing_session = result
            self._begin_shutdown_cleanup()
            return

        self._set_busy(False, message)
        if kind == "recovery":
            error = self._health_error
            self._health_error = None
            if error is not None:
                messagebox.showerror(
                    "Hardware connection",
                    f"{error}\n\nReconnect before capturing another point.",
                )
            return
        if kind == "connect" and isinstance(result, ServoHardwareSession):
            self._handle_connected(result)
            return
        if kind == "capture" and isinstance(result, tuple) and len(result) == 2:
            servo_id, capture_result = result
            point, capture = capture_result
            point["capture_sample_count"] = capture.sample_count
            point["capture_std_delta_deg"] = capture.std_delta_deg
            self.points_by_servo[int(servo_id)].append(point)
            self._refresh_selected_servo()
            return
        if kind == "save" and isinstance(result, tuple) and len(result) == 2:
            path, records = result
            self.existing_records = records
            self._refresh_selected_servo()
            self.status_var.set(f"Saved calibrated intercepts: {path}")
            messagebox.showinfo(
                "Update calibrated intercepts",
                f"Saved all six servo intercepts.\n\n{path}\n\n"
                "The previous file was preserved as .bak when it existed.",
            )

    def _connect(self) -> None:
        if self.busy or self._close_requested or self.session is not None:
            return
        servo_port = self.servo_port_var.get().strip()
        encoder_port = self.encoder_port_var.get().strip()
        if not servo_port or not encoder_port:
            messagebox.showerror("Connection", "Enter both serial ports")
            return
        session = ServoHardwareSession(
            servo_ids=SERVO_IDS,
            servo_names=("ED", "DI", "FDS", "FDP", "LU", "PI"),
            encoder_sensor_ids=SERVO_IDS,
            servo_port=servo_port,
            encoder_port=encoder_port,
            encoder_rate_hz=70.0,
        )
        self._pending_session = session
        self._set_busy(True, "Connecting; no motion command is sent...")

        def connect(_stop_event: threading.Event) -> ServoHardwareSession:
            try:
                session.connect(startup_timeout_s=5.0)
            except Exception:
                session.close()
                raise
            return session

        self._run_worker(
            connect,
            "Connected; no servo motion has been commanded",
            kind="connect",
        )

    def _disconnect(self) -> None:
        if self.busy or self._close_requested:
            return
        session = self.session
        self.session = None
        self._pending_session = None
        self._reset_speed_commands()
        if session is None:
            self._set_busy(False, "Disconnected")
            return

        self._set_busy(True, "Disconnecting...")
        self._run_worker(
            lambda _stop_event: session.close(),
            "Disconnected",
            kind="disconnect",
        )

    def _capture_point(self) -> None:
        if self.busy or self._close_requested:
            return
        session = self.session
        if session is None or not session.connected:
            messagebox.showerror("Capture point", "Connect both devices first")
            return
        if any(self._speed_command_values.values()):
            messagebox.showerror(
                "Capture point",
                "Stop all motors before capturing a calibration point",
            )
            return
        try:
            force_n = float(self.force_var.get())
            direction = int(self.direction_var.get())
            duration_s = float(self.duration_var.get())
        except ValueError as exc:
            messagebox.showerror("Capture point", f"Invalid numeric input: {exc}")
            return
        if not math.isfinite(force_n) or not math.isfinite(duration_s):
            messagebox.showerror(
                "Capture point",
                "Force and capture duration must be finite numbers",
            )
            return
        servo_id = self._selected_servo_id()
        if len(self.points_by_servo[servo_id]) >= MAX_POINTS_PER_SERVO:
            messagebox.showerror(
                "Capture point",
                f"At most {MAX_POINTS_PER_SERVO} points can be retained for one servo",
            )
            return
        model = self.models[servo_id]
        self._set_busy(True, f"Capturing servo {servo_id}; keep the load steady...")

        def capture(stop_event: threading.Event) -> tuple[int, tuple[dict[str, Any], Any]]:
            angle_capture = capture_angle_window(
                session,
                model,
                servo_id,
                duration_s,
                stop_event=stop_event,
            )
            point = calculate_calibration_point(
                model,
                force_n=force_n,
                angle_delta_deg=angle_capture.mean_delta_deg,
                torque_direction=direction,
            )
            return (servo_id, (point, angle_capture))

        self._run_worker(
            capture,
            f"Added point for servo {servo_id}",
            kind="capture",
            cleanup_on_error=session.close,
        )

    def _calculate_selected(self) -> None:
        servo_id = self._selected_servo_id()
        try:
            fit = fit_fixed_slope_intercept(self.models[servo_id], self.points_by_servo[servo_id])
        except (KeyError, TypeError, ValueError) as exc:
            messagebox.showerror("Calculate b", str(exc))
            return
        self.fits_by_servo[servo_id] = fit
        self._refresh_selected_servo()
        self._refresh_pending_table()
        self.status_var.set(f"Calculated fixed-slope b for servo {servo_id}")

    def _clear_selected(self) -> None:
        servo_id = self._selected_servo_id()
        self.points_by_servo[servo_id].clear()
        self.fits_by_servo.pop(servo_id, None)
        self._refresh_selected_servo()
        self._refresh_pending_table()

    def _update_all(self) -> None:
        if self.busy or self._close_requested:
            return
        missing = [servo_id for servo_id in SERVO_IDS if servo_id not in self.fits_by_servo]
        if missing:
            messagebox.showerror(
                "Update calibrated intercepts",
                "Calculate b for every servo first: " + ", ".join(map(str, missing)),
            )
            return
        records = {}
        for servo_id in SERVO_IDS:
            model = self.models[servo_id]
            fit = self.fits_by_servo[servo_id]
            records[servo_id] = {
                "servo_id": servo_id,
                "encoder_sensor_id": int(model["sensor_id"]),
                "slope_nmm_per_deg": float(model["slope_nmm_per_deg"]),
                "intercept_nmm": float(fit["intercept_nmm"]),
                "intercept_std_nmm": float(fit["intercept_std_nmm"]),
                "point_count": int(fit["point_count"]),
                "moment_arm_mm": float(model["moment_arm_mm"]),
                "pull_angle_deg": float(model["pull_angle_deg"]),
                "method": "fixed_slope_force_gauge",
                "points": fit["points"],
            }
        self._set_busy(True, "Saving calibrated intercepts...")

        def save(_stop_event: threading.Event) -> tuple[Path, dict[int, dict[str, Any]]]:
            path = save_torque_intercept_calibration(
                DEFAULT_INTERCEPT_CALIBRATION_PATH,
                records,
            )
            return path, load_torque_intercept_calibration(path)

        self._run_worker(
            save,
            "Saved calibrated intercepts",
            kind="save",
        )

    def _set_motor_speed(self, servo_id: int, speed: int) -> None:
        if self.busy or self._close_requested:
            return
        session = self.session
        if session is None or not session.connected:
            messagebox.showerror("Motor speed", "Connect both devices first")
            return
        try:
            session.check_health()
            session.submit_speed_to_servos({int(servo_id): int(speed)})
        except Exception as exc:
            messagebox.showerror("Motor speed", str(exc))
            return
        self._speed_command_values[int(servo_id)] = int(speed)
        self.speed_command_vars[int(servo_id)].set(str(int(speed)))
        self._update_speed_controls()

    def _stop_all_motors(self) -> None:
        if self.busy or self._close_requested:
            return
        session = self.session
        if session is None or not session.connected:
            messagebox.showerror("Motor speed", "Connect both devices first")
            return
        try:
            session.check_health()
            session.submit_speed_to_servos({servo_id: 0 for servo_id in SERVO_IDS})
        except Exception as exc:
            messagebox.showerror("Motor speed", str(exc))
            return
        self._reset_speed_commands()
        self._update_speed_controls()

    def _handle_connected(self, session: ServoHardwareSession) -> None:
        self.session = session
        self._pending_session = None
        self._set_busy(False)

    def _begin_shutdown_cleanup(self) -> None:
        if not self._close_requested or self._worker is not None:
            return
        session = self._closing_session
        self._closing_session = None
        if session is None:
            self._destroy_root()
            return
        self._set_busy(True, "Closing hardware connections...")
        self._run_worker(
            lambda _stop_event: session.close(),
            "Closed",
            kind="shutdown",
        )

    def _destroy_root(self) -> None:
        if self._destroyed:
            return
        self._destroyed = True
        if self._poll_after_id is not None:
            try:
                self.root.after_cancel(self._poll_after_id)
            except tk.TclError:
                pass
            self._poll_after_id = None
        self.root.destroy()

    def close(self) -> None:
        if self._destroyed or self._close_requested:
            return
        self._close_requested = True
        if self._worker_stop is not None:
            self._worker_stop.set()
        self._closing_session = self.session or self._pending_session
        self.session = None
        self._pending_session = None
        self._reset_speed_commands()
        self._set_busy(True, "Closing...")
        self._begin_shutdown_cleanup()


def main() -> None:
    root = tk.Tk()
    try:
        app = TorqueCalibrationApp(root)
    except Exception as exc:
        messagebox.showerror("Torque calibration", str(exc))
        root.destroy()
        raise
    root.mainloop()


if __name__ == "__main__":
    main()
