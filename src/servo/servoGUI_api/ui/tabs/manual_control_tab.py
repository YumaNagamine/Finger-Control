from __future__ import annotations

import time
import tkinter as tk
from tkinter import ttk


class ManualControlTab:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.control_mode = tk.StringVar(value="speed")
        self._suspend_slider_events = True

        main_frame = ttk.Frame(parent, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        mode_frame = ttk.LabelFrame(main_frame, text="Control Mode")
        mode_frame.pack(fill=tk.X, pady=5)
        ttk.Radiobutton(
            mode_frame,
            text="Speed Control (Wheel Mode)",
            variable=self.control_mode,
            value="speed",
            command=lambda: self.set_mode("speed"),
        ).pack(side=tk.LEFT, padx=10)
        ttk.Radiobutton(
            mode_frame,
            text="Position Control (Servo Mode)",
            variable=self.control_mode,
            value="position",
            command=lambda: self.set_mode("position"),
        ).pack(side=tk.LEFT, padx=10)

        self.motor_container = ttk.Frame(main_frame)
        self.motor_container.pack(fill=tk.BOTH, expand=True)

        for i in range(self.app.NUM_MOTORS):
            self.create_motor_ui(self.motor_container, i)

        self.create_global_controls(main_frame)

        new_features_frame = ttk.Frame(main_frame)
        new_features_frame.pack(fill=tk.X, expand=True, padx=10, pady=10)
        self.create_timed_run_box(new_features_frame)
        self.create_position_control_box(new_features_frame)

        self.set_mode(self.control_mode.get())
        self._suspend_slider_events = False

    def update_slider(self, motor_id, value):
        if motor_id < len(self.app.sliders):
            try:
                self.app.sliders[motor_id].set(int(value))
            except ValueError:
                pass

    def create_motor_ui(self, parent, motor_id):
        motor_name = self.app.MOTOR_NAMES[motor_id] if motor_id < len(self.app.MOTOR_NAMES) else ""
        motor_frame = ttk.LabelFrame(parent, text=f"Motor {motor_id} ({motor_name})")
        motor_frame.pack(fill=tk.X, expand=True, padx=10, pady=2)

        slider_frame = ttk.Frame(motor_frame)
        slider_frame.pack(fill=tk.X, expand=True)

        slider = ttk.Scale(
            slider_frame,
            from_=-2000,
            to=2000,
            orient=tk.HORIZONTAL,
            length=400,
            command=lambda val, mid=motor_id: self.on_slider_moved(val, mid),
        )
        slider.set(0)
        slider.pack(side=tk.LEFT, padx=10, pady=5)
        self.app.sliders.append(slider)

        pos_label = ttk.Label(slider_frame, text="Pos: N/A", width=20)
        pos_label.pack(side=tk.LEFT, padx=5)
        self.app.pos_labels.append(pos_label)

        load_label = ttk.Label(slider_frame, text="Load: N/A", width=15)
        load_label.pack(side=tk.LEFT, padx=5)
        self.app.load_labels.append(load_label)

        button_frame = ttk.Frame(motor_frame)
        button_frame.pack(fill=tk.X, expand=True, pady=2)

        for speed in self.app.SPEED_PRESETS:
            btn = ttk.Button(
                button_frame,
                text=str(speed),
                width=5,
                command=lambda mid=motor_id, s=speed: self.on_preset_click(mid, s),
            )
            btn.pack(side=tk.LEFT, padx=12, expand=True)

    def set_mode(self, mode):
        if mode == "speed":
            self._suspend_slider_events = True
            for slider in self.app.sliders:
                slider.config(from_=-2000, to=2000, state=tk.NORMAL)
                slider.set(0)
            self._suspend_slider_events = False
            return

        for i in range(self.app.NUM_MOTORS):
            self.app.control_service.set_speed(i, 0, force_init=True)
        time.sleep(0.2)

        self._suspend_slider_events = True
        for slider in self.app.sliders:
            slider.config(from_=-2048, to=2047, state=tk.NORMAL)
            slider.set(0)
        self._suspend_slider_events = False

    def on_preset_click(self, motor_id, val):
        if self.control_mode.get() == "speed":
            self.app.on_preset_button_click(motor_id, val)
            return

        mapped_val = val
        if abs(val) == 2000:
            mapped_val = 2048 if val > 0 else -2048
        elif abs(val) == 1000:
            mapped_val = 1024 if val > 0 else -1024
        elif abs(val) == 500:
            mapped_val = 512 if val > 0 else -512
        elif val == 0:
            mapped_val = 0

        self.app.sliders[motor_id].set(mapped_val)

    def on_slider_moved(self, value, motor_id):
        if self._suspend_slider_events:
            return

        val = int(float(value))
        mode = self.control_mode.get()

        if mode == "speed":
            self.app.control_service.set_speed(motor_id, val, force_init=True)
        else:
            target_pos = 2048 + val
            target_pos = max(0, min(4095, target_pos))
            self.app.control_service.set_position(motor_id, target_pos, time_ms=0)

    def create_global_controls(self, parent):
        global_frame = ttk.LabelFrame(parent, text="Global Controls")
        global_frame.pack(fill=tk.X, expand=True, padx=10, pady=10)

        global_button_frame = ttk.Frame(global_frame)
        global_button_frame.pack(fill=tk.X, expand=True, pady=5)

        for speed in self.app.GLOBAL_SPEED_PRESETS:
            btn = ttk.Button(
                global_button_frame,
                text=str(speed),
                command=lambda s=speed: self.app.on_global_preset_button_click(s),
            )
            btn.pack(side=tk.LEFT, padx=20, expand=True)

    def create_timed_run_box(self, parent):
        box_frame = ttk.LabelFrame(parent, text="Timed Run (Speed Mode)")
        box_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        ttk.Label(box_frame, text="Motor ID:").pack(side=tk.LEFT, padx=5)
        self.app.timed_run_id = ttk.Combobox(
            box_frame,
            values=[str(i) for i in range(self.app.NUM_MOTORS)],
            width=3,
        )
        self.app.timed_run_id.set("0")
        self.app.timed_run_id.pack(side=tk.LEFT)

        ttk.Label(box_frame, text="Time(ms):").pack(side=tk.LEFT, padx=5)
        self.app.timed_run_ms = ttk.Entry(box_frame, width=5)
        self.app.timed_run_ms.insert(0, "50")
        self.app.timed_run_ms.pack(side=tk.LEFT)

        button_container = ttk.Frame(box_frame)
        button_container.pack(pady=5, fill=tk.X)
        for speed in self.app.TIMED_RUN_SPEEDS:
            btn = ttk.Button(
                button_container,
                text=f"Run {speed}",
                command=lambda s=speed: self.app.on_timed_run_click(s),
            )
            btn.pack(side=tk.LEFT, padx=5, expand=True)

    def create_position_control_box(self, parent):
        box_frame = ttk.LabelFrame(parent, text="Zeroing Tools")
        box_frame.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=5)

        ttk.Label(box_frame, text="Motor ID:").pack(side=tk.LEFT, padx=5)
        self.app.pos_control_id = ttk.Combobox(
            box_frame,
            values=[str(i) for i in range(self.app.NUM_MOTORS)],
            width=3,
        )
        self.app.pos_control_id.set("0")
        self.app.pos_control_id.pack(side=tk.LEFT, padx=10)

        ttk.Button(
            box_frame,
            text="Set Current as Zero",
            command=self.app.on_set_zero_click,
        ).pack(side=tk.LEFT, padx=5, expand=True)
        ttk.Button(
            box_frame,
            text="Set ALL as Zero",
            command=self.app.on_set_all_zero_click,
        ).pack(side=tk.LEFT, padx=5, expand=True)
        ttk.Button(
            box_frame,
            text="Go to Zero (Soft PID)",
            command=self.app.on_go_to_zero_click,
        ).pack(side=tk.LEFT, padx=5, expand=True)
