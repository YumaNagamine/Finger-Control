from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class BatchControlTab:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.target_values = [0] * self.app.NUM_MOTORS
        self.sliders = []
        self.labels_pos = []
        self.labels_load = []
        self.labels_target = []

        main_frame = ttk.Frame(parent, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        btn_frame = ttk.LabelFrame(main_frame, text="Batch Execution")
        btn_frame.pack(fill=tk.X, pady=10)

        ttk.Button(
            btn_frame,
            text="DRIVE (Send All Targets)",
            command=self.send_all,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, pady=10)
        ttk.Button(
            btn_frame,
            text="STOP ALL (Reset 0)",
            command=self.stop_all,
        ).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, pady=10)

        global_frame = ttk.Frame(main_frame)
        global_frame.pack(fill=tk.X, pady=5)
        ttk.Label(global_frame, text="Set All Sliders:").pack(side=tk.LEFT)
        for s in [-1000, 0, 1000]:
            ttk.Button(
                global_frame,
                text=str(s),
                command=lambda val=s: self.set_all_targets(val),
            ).pack(side=tk.LEFT, padx=5)

        self.motor_container = ttk.Frame(main_frame)
        self.motor_container.pack(fill=tk.BOTH, expand=True)

        for i in range(self.app.NUM_MOTORS):
            self.create_motor_row(i)

    def create_motor_row(self, motor_id):
        motor_name = self.app.MOTOR_NAMES[motor_id] if motor_id < len(self.app.MOTOR_NAMES) else ""
        frame = ttk.LabelFrame(self.motor_container, text=f"Motor {motor_id} ({motor_name})")
        frame.pack(fill=tk.X, pady=2, padx=5)

        row1 = ttk.Frame(frame)
        row1.pack(fill=tk.X)

        l_target = ttk.Label(row1, text="Target: 0", width=12, foreground="blue")
        self.labels_target.append(l_target)

        slider = ttk.Scale(
            row1,
            from_=-2000,
            to=2000,
            orient=tk.HORIZONTAL,
            command=lambda val, mid=motor_id: self.on_slider(mid, val),
        )
        slider.set(0)
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        self.sliders.append(slider)

        l_target.pack(side=tk.LEFT, padx=5)

        row2 = ttk.Frame(frame)
        row2.pack(fill=tk.X, padx=10, pady=2)

        l_pos = ttk.Label(row2, text="Pos: --", width=15)
        l_pos.pack(side=tk.LEFT)
        self.labels_pos.append(l_pos)

        l_load = ttk.Label(row2, text="Load: --", width=15)
        l_load.pack(side=tk.LEFT)
        self.labels_load.append(l_load)

        for s in [-500, 0, 500]:
            ttk.Button(
                row2,
                text=str(s),
                width=5,
                command=lambda val=s, mid=motor_id: self.set_single_target(mid, val),
            ).pack(side=tk.LEFT, padx=2)

    def on_slider(self, motor_id, value):
        val = int(float(value))
        self.target_values[motor_id] = val
        self.labels_target[motor_id].config(text=f"Target: {val}")

    def set_single_target(self, motor_id, val):
        self.sliders[motor_id].set(val)

    def set_all_targets(self, val):
        for i in range(self.app.NUM_MOTORS):
            self.sliders[i].set(val)

    def send_all(self):
        for i in range(self.app.NUM_MOTORS):
            self.app.control_service.set_speed(i, self.target_values[i])

    def stop_all(self):
        for i in range(self.app.NUM_MOTORS):
            self.sliders[i].set(0)
            self.app.control_service.set_speed(i, 0)

    def update_telemetry(self, positions, loads):
        for i in range(self.app.NUM_MOTORS):
            self.labels_pos[i].config(text=f"Pos: {positions[i]}")
            self.labels_load[i].config(text=f"Load: {loads[i]}")
