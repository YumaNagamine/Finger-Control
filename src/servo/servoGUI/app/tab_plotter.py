import tkinter as tk
from tkinter import ttk, filedialog
from collections import deque
import csv

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation

class PlotterTab:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app

        # --- 状態変数の初期化 ---
        self.is_plotting = False
        self.plotting_motor_id = tk.IntVar(value=0) # Default to 0-indexed Motor 0
        
        # --- データストレージの初期化 ---
        # グラフ描画用のデータ (常に最新)
        self.plot_data_deque = {'pos': deque(maxlen=200), 'torque': deque(maxlen=200)}
        # CSV保存用の全データ
        self.full_plot_data = {'pos': [], 'torque': []}

        # --- UIの作成 ---
        self.create_widgets()

    def create_widgets(self):
        top_frame = ttk.Frame(self.parent, padding="10")
        top_frame.pack(fill=tk.X)
        
        # --- 上部コントロールフレーム ---
        control_frame = ttk.LabelFrame(top_frame, text="Controls")
        control_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(control_frame, text="Motor ID:").pack(side=tk.LEFT, padx=5)
        ttk.Combobox(control_frame, textvariable=self.plotting_motor_id, 
                     values=[str(i) for i in range(self.app.NUM_MOTORS)], width=3).pack(side=tk.LEFT, padx=5)
        
        self.plot_start_button = ttk.Button(control_frame, text="Start Run & Plot", command=self.on_plot_start)
        self.plot_start_button.pack(side=tk.LEFT, padx=5)
        
        self.plot_stop_button = ttk.Button(control_frame, text="Stop", command=self.on_plot_stop, state=tk.DISABLED)
        self.plot_stop_button.pack(side=tk.LEFT, padx=5)
        
        ttk.Button(control_frame, text="Clear Plot", command=self.on_plot_clear).pack(side=tk.LEFT, padx=5)

        ttk.Button(control_frame, text="Save Plot Image", command=self.on_plot_save).pack(side=tk.RIGHT, padx=5)
        ttk.Button(control_frame, text="Save Data (CSV)", command=self.on_data_save).pack(side=tk.RIGHT, padx=5)

        # --- 速度設定フレーム ---
        speed_frame = ttk.LabelFrame(top_frame, text="Speed Control")
        speed_frame.pack(fill=tk.X, pady=5)
        
        self.plot_speed_slider = ttk.Scale(speed_frame, from_=-2000, to=2000, orient=tk.HORIZONTAL, length=300)
        self.plot_speed_slider.set(1000)
        self.plot_speed_slider.pack(side=tk.LEFT, padx=10)
        
        button_container = ttk.Frame(speed_frame)
        button_container.pack(side=tk.LEFT, fill=tk.X, expand=True)
        for speed in self.app.PLOTTER_SPEED_PRESETS:
            btn = ttk.Button(button_container, text=str(speed), width=5, command=lambda s=speed: self.on_plot_preset_click(s))
            btn.pack(side=tk.LEFT, padx=6, expand=True)

        # --- Matplotlib グラフ ---
        self.fig = Figure(figsize=(8, 6), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)
        self.line, = self.ax.plot([], [], 'r-', animated=True)
        
        self.ani = animation.FuncAnimation(self.fig, self.update_plot, blit=True, interval=100, cache_frame_data=False)

    def process_data(self, motor_id, pos, torque):
        """メインアプリからデータを受け取るメソッド"""
        if self.is_plotting and motor_id == self.plotting_motor_id.get():
            self.plot_data_deque['pos'].append(pos)
            self.plot_data_deque['torque'].append(torque)
            self.full_plot_data['pos'].append(pos)
            self.full_plot_data['torque'].append(torque)

    def on_plot_start(self):
        self.on_plot_clear()
        self.is_plotting = True
        self.plot_start_button.config(state=tk.DISABLED)
        self.plot_stop_button.config(state=tk.NORMAL)
        speed = int(self.plot_speed_slider.get())
        phy_id = self.app.get_physical_id(self.plotting_motor_id.get())
        if phy_id >= 0:
            self.app.send_serial_command(f"{phy_id},{speed}\n")

    def on_plot_stop(self):
        self.is_plotting = False
        self.plot_start_button.config(state=tk.NORMAL)
        self.plot_stop_button.config(state=tk.DISABLED)
        phy_id = self.app.get_physical_id(self.plotting_motor_id.get())
        if phy_id >= 0:
            self.app.send_serial_command(f"{phy_id},0\n")

    def on_plot_clear(self):
        self.plot_data_deque['pos'].clear()
        self.plot_data_deque['torque'].clear()
        self.full_plot_data['pos'].clear()
        self.full_plot_data['torque'].clear()
        print("Plot data cleared.")

    def on_plot_save(self):
        filepath = filedialog.asksaveasfilename(defaultextension=".png", filetypes=[("PNG files", "*.png"), ("JPEG files", "*.jpg"), ("All files", "*.* पहन")])
        if filepath:
            self.fig.savefig(filepath)
            print(f"Plot saved to {filepath}")

    def on_data_save(self):
        filepath = filedialog.asksaveasfilename(defaultextension=".csv", filetypes=[("CSV files", "*.csv"), ("All files", "*.* पहन")])
        if not filepath: return
        try:
            with open(filepath, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Position", "Torque"])
                writer.writerows(zip(self.full_plot_data['pos'], self.full_plot_data['torque']))
            print(f"Data saved to {filepath}")
        except IOError as e:
            print(f"Error saving data: {e}")

    def on_plot_preset_click(self, speed):
        self.plot_speed_slider.set(speed)
        if self.is_plotting:
            phy_id = self.app.get_physical_id(self.plotting_motor_id.get())
            if phy_id >= 0:
                self.app.send_serial_command(f"{phy_id},{speed}\n")

    def update_plot(self, frame):
        # Update data for the existing line artist
        self.line.set_data(self.plot_data_deque['pos'], self.plot_data_deque['torque'])
        
        # Adjust axis limits based on new data
        self.ax.relim()
        self.ax.autoscale_view()

        # Set title and labels
        self.ax.set_title("Position vs. Torque")
        self.ax.set_xlabel("Position")
        self.ax.set_ylabel("Torque")

        return self.line,