from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class HighAccuracyPositionTab:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.last_commanded_positions = [None] * self.app.NUM_MOTORS
        self.custom_step_entries = []
        self.pos_labels = []

        self.create_ui()

    def create_ui(self):
        main_frame = ttk.Frame(self.parent, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        for i in range(self.app.NUM_MOTORS):
            self.create_motor_row(main_frame, i)

    def create_motor_row(self, parent, motor_id):
        motor_name = self.app.MOTOR_NAMES[motor_id] if motor_id < len(self.app.MOTOR_NAMES) else ""
        row_frame = ttk.LabelFrame(parent, text=f"Motor {motor_id} ({motor_name})")
        row_frame.pack(fill=tk.X, pady=5, padx=5)

        status_frame = ttk.Frame(row_frame)
        status_frame.pack(fill=tk.X, padx=5, pady=2)

        pos_lbl = ttk.Label(status_frame, text="Pos: --", font=("Consolas", 10))
        pos_lbl.pack(side=tk.LEFT)
        self.pos_labels.append(pos_lbl)

        ctrl_frame = ttk.Frame(row_frame)
        ctrl_frame.pack(fill=tk.X, padx=5, pady=5)

        ttk.Button(ctrl_frame, text="-100", width=6, command=lambda: self.move_step(motor_id, -100)).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl_frame, text="-10", width=5, command=lambda: self.move_step(motor_id, -10)).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl_frame, text="-1", width=4, command=lambda: self.move_step(motor_id, -1)).pack(side=tk.LEFT, padx=2)

        entry = ttk.Entry(ctrl_frame, width=6, justify="center")
        entry.insert(0, "50")
        entry.pack(side=tk.LEFT, padx=10)
        self.custom_step_entries.append(entry)

        ttk.Button(ctrl_frame, text="< Custom", width=8, command=lambda: self.move_custom(motor_id, -1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl_frame, text="Custom >", width=8, command=lambda: self.move_custom(motor_id, 1)).pack(side=tk.LEFT, padx=2)

        ttk.Button(ctrl_frame, text="+1", width=4, command=lambda: self.move_step(motor_id, 1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl_frame, text="+10", width=5, command=lambda: self.move_step(motor_id, 10)).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl_frame, text="+100", width=6, command=lambda: self.move_step(motor_id, 100)).pack(side=tk.LEFT, padx=2)

    def move_custom(self, motor_id, direction):
        try:
            val = int(self.custom_step_entries[motor_id].get())
            self.move_step(motor_id, val * direction)
        except ValueError:
            print(f"Invalid custom step for Motor {motor_id}")

    def move_step(self, motor_id, step):
        if self.last_commanded_positions[motor_id] is None:
            self.last_commanded_positions[motor_id] = self.app.current_positions[motor_id]

        new_target = self.last_commanded_positions[motor_id] + step
        new_target = max(0, min(4095, new_target))

        self.last_commanded_positions[motor_id] = new_target
        self.app.control_service.set_position(motor_id, new_target, time_ms=0)

    def update_telemetry(self):
        for i in range(self.app.NUM_MOTORS):
            if i < len(self.pos_labels):
                self.pos_labels[i].config(text=f"Pos: {self.app.current_positions[i]}")
