from __future__ import annotations

import tkinter as tk
from collections import deque
from tkinter import ttk

import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class MonitorAllTab:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.is_monitoring = False

        self.y_axis_mode = tk.StringVar(value="raw")
        self.motor_visibility = [tk.BooleanVar(value=True) for _ in range(self.app.NUM_MOTORS)]
        self.plot_data = [{"time": deque(maxlen=200), "torque": deque(maxlen=200)} for _ in range(self.app.NUM_MOTORS)]

        self.create_widgets()

    def create_widgets(self):
        top_frame = ttk.Frame(self.parent, padding="10")
        top_frame.pack(fill=tk.X)

        left_controls = ttk.Frame(top_frame)
        left_controls.pack(side=tk.LEFT, fill=tk.X, expand=True)

        control_frame = ttk.LabelFrame(left_controls, text="Controls", padding="10")
        control_frame.pack(fill=tk.X)
        self.start_button = ttk.Button(control_frame, text="Start Monitoring", command=self.on_monitoring_start)
        self.start_button.pack(side=tk.LEFT, padx=5)
        self.stop_button = ttk.Button(
            control_frame,
            text="Stop Monitoring",
            command=self.on_monitoring_stop,
            state=tk.DISABLED,
        )
        self.stop_button.pack(side=tk.LEFT, padx=5)
        ttk.Button(control_frame, text="Clear Plot", command=self.on_monitoring_clear).pack(side=tk.LEFT, padx=5)

        yaxis_frame = ttk.LabelFrame(left_controls, text="Y-Axis Unit", padding="10")
        yaxis_frame.pack(fill=tk.X, pady=5)
        ttk.Radiobutton(yaxis_frame, text="Raw Integer (-1000~1000)", variable=self.y_axis_mode, value="raw").pack(anchor=tk.W)
        ttk.Radiobutton(yaxis_frame, text="Torque (kg.cm)", variable=self.y_axis_mode, value="kgcm").pack(anchor=tk.W)

        visibility_frame = ttk.LabelFrame(top_frame, text="Visible Motors", padding="10")
        visibility_frame.pack(side=tk.LEFT, fill=tk.X, padx=10)
        for i in range(self.app.NUM_MOTORS):
            ttk.Checkbutton(visibility_frame, text=f"Motor {i}", variable=self.motor_visibility[i]).pack(anchor=tk.W)

        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.colors = ["#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b"]

        self.canvas = FigureCanvasTkAgg(self.fig, master=self.parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

        self.ani = animation.FuncAnimation(self.fig, self.update_plot, blit=False, interval=100, cache_frame_data=False)

    def process_frame(self, timestamp_ms, loads):
        if not self.is_monitoring:
            return
        for i in range(self.app.NUM_MOTORS):
            self.plot_data[i]["time"].append(timestamp_ms)
            self.plot_data[i]["torque"].append(loads[i])

    def on_monitoring_start(self):
        self.on_monitoring_clear()
        self.is_monitoring = True
        self.start_button.config(state=tk.DISABLED)
        self.stop_button.config(state=tk.NORMAL)

    def on_monitoring_stop(self):
        self.is_monitoring = False
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)

    def on_monitoring_clear(self):
        for i in range(self.app.NUM_MOTORS):
            self.plot_data[i]["time"].clear()
            self.plot_data[i]["torque"].clear()

    def update_plot(self, frame):
        self.ax.clear()

        y_mode = self.y_axis_mode.get()
        if y_mode == "kgcm":
            self.ax.set_ylabel("Torque (kg.cm)")
        else:
            self.ax.set_ylabel("Torque (Raw Integer)")
        self.ax.set_title("Torque vs. Time (All Servos)")
        self.ax.set_xlabel("Time (ms)")
        self.ax.grid(True)

        for i in range(self.app.NUM_MOTORS):
            if self.motor_visibility[i].get():
                y_data = self.plot_data[i]["torque"]
                if y_mode == "kgcm":
                    y_data = [(val / 1000.0) * 20.0 for val in y_data]
                self.ax.plot(self.plot_data[i]["time"], y_data, color=self.colors[i], label=f"Motor {i}")

        self.ax.relim()
        self.ax.autoscale_view(True, True, True)

        if any(v.get() for v in self.motor_visibility):
            self.ax.legend()

        return []
