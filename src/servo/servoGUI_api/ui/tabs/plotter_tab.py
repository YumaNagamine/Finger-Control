from __future__ import annotations

import csv
import tkinter as tk
from collections import deque
from tkinter import filedialog, ttk

import matplotlib.animation as animation
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
from matplotlib.figure import Figure


class PlotterTab:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app

        self.is_plotting = False
        self.plotting_motor_id = tk.IntVar(value=0)

        self.plot_data_deque = {"pos": deque(maxlen=200), "torque": deque(maxlen=200)}
        self.full_plot_data = {"pos": [], "torque": []}

        self.create_widgets()

    def create_widgets(self):
        top_frame = ttk.Frame(self.parent, padding="10")
        top_frame.pack(fill=tk.X)

        control_frame = ttk.LabelFrame(top_frame, text="Controls")
        control_frame.pack(fill=tk.X, pady=5)

        ttk.Label(control_frame, text="Motor ID:").pack(side=tk.LEFT, padx=5)
        ttk.Combobox(
            control_frame,
            textvariable=self.plotting_motor_id,
            values=[str(i) for i in range(self.app.NUM_MOTORS)],
            width=3,
        ).pack(side=tk.LEFT, padx=5)

        self.plot_start_button = ttk.Button(control_frame, text="Start Run & Plot", command=self.on_plot_start)
        self.plot_start_button.pack(side=tk.LEFT, padx=5)

        self.plot_stop_button = ttk.Button(control_frame, text="Stop", command=self.on_plot_stop, state=tk.DISABLED)
        self.plot_stop_button.pack(side=tk.LEFT, padx=5)

        ttk.Button(control_frame, text="Clear Plot", command=self.on_plot_clear).pack(side=tk.LEFT, padx=5)

        ttk.Button(control_frame, text="Save Plot Image", command=self.on_plot_save).pack(side=tk.RIGHT, padx=5)
        ttk.Button(control_frame, text="Save Data (CSV)", command=self.on_data_save).pack(side=tk.RIGHT, padx=5)

        speed_frame = ttk.LabelFrame(top_frame, text="Speed Control")
        speed_frame.pack(fill=tk.X, pady=5)

        self.plot_speed_slider = ttk.Scale(speed_frame, from_=-2000, to=2000, orient=tk.HORIZONTAL, length=300)
        self.plot_speed_slider.set(1000)
        self.plot_speed_slider.pack(side=tk.LEFT, padx=10)

        button_container = ttk.Frame(speed_frame)
        button_container.pack(side=tk.LEFT, fill=tk.X, expand=True)
        for speed in self.app.PLOTTER_SPEED_PRESETS:
            btn = ttk.Button(
                button_container,
                text=str(speed),
                width=5,
                command=lambda s=speed: self.on_plot_preset_click(s),
            )
            btn.pack(side=tk.LEFT, padx=6, expand=True)

        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.line, = self.ax.plot([], [], "r-", animated=True)

        self.ani = animation.FuncAnimation(self.fig, self.update_plot, blit=True, interval=100, cache_frame_data=False)

    def process_data(self, motor_id, pos, torque):
        if self.is_plotting and motor_id == self.plotting_motor_id.get():
            self.plot_data_deque["pos"].append(pos)
            self.plot_data_deque["torque"].append(torque)
            self.full_plot_data["pos"].append(pos)
            self.full_plot_data["torque"].append(torque)

    def on_plot_start(self):
        self.on_plot_clear()
        self.is_plotting = True
        self.plot_start_button.config(state=tk.DISABLED)
        self.plot_stop_button.config(state=tk.NORMAL)
        speed = int(self.plot_speed_slider.get())
        self.app.control_service.set_speed(self.plotting_motor_id.get(), speed)

    def on_plot_stop(self):
        self.is_plotting = False
        self.plot_start_button.config(state=tk.NORMAL)
        self.plot_stop_button.config(state=tk.DISABLED)
        self.app.control_service.set_speed(self.plotting_motor_id.get(), 0)

    def on_plot_clear(self):
        self.plot_data_deque["pos"].clear()
        self.plot_data_deque["torque"].clear()
        self.full_plot_data["pos"].clear()
        self.full_plot_data["torque"].clear()

    def on_plot_save(self):
        filepath = filedialog.asksaveasfilename(
            defaultextension=".png",
            filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.*")],
        )
        if filepath:
            self.fig.savefig(filepath)

    def on_data_save(self):
        filepath = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV files", "*.csv"), ("All files", "*.*")],
        )
        if not filepath:
            return

        with open(filepath, "w", newline="", encoding="utf-8") as f:
            writer = csv.writer(f)
            writer.writerow(["Position", "Torque"])
            writer.writerows(zip(self.full_plot_data["pos"], self.full_plot_data["torque"]))

    def on_plot_preset_click(self, speed):
        self.plot_speed_slider.set(speed)
        if self.is_plotting:
            self.app.control_service.set_speed(self.plotting_motor_id.get(), speed)

    def update_plot(self, frame):
        self.line.set_data(self.plot_data_deque["pos"], self.plot_data_deque["torque"])
        self.ax.relim()
        self.ax.autoscale_view()

        self.ax.set_title("Position vs. Torque")
        self.ax.set_xlabel("Position")
        self.ax.set_ylabel("Torque")

        return (self.line,)
