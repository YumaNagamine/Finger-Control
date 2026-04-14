import tkinter as tk
from tkinter import ttk
import time
import csv
from collections import deque

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import matplotlib.animation as animation

class SpeedTestTab:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        
        self.is_testing = False
        self.test_motor_id = tk.IntVar(value=0) # Default to 0-indexed Motor 0
        self.start_speed = 1900 # Updated Start
        self.max_test_speed = 4000 # Updated Max
        self.step_size = 50 # Finer resolution
        self.step_duration = 0.5 # Seconds to hold each step
        
        self.measured_data = {'commanded': [], 'measured_speed': [], 'load': []}
        self.plot_data = {'x': [], 'y': []} # x=Commanded, y=Measured

        self.create_widgets()

    def create_widgets(self):
        top_frame = ttk.Frame(self.parent, padding="10")
        top_frame.pack(fill=tk.X)
        
        # Controls
        control_frame = ttk.LabelFrame(top_frame, text="Max Speed Finder", padding="10")
        control_frame.pack(fill=tk.X)
        
        ttk.Label(control_frame, text="Motor ID:").pack(side=tk.LEFT, padx=5)
        ttk.Combobox(control_frame, textvariable=self.test_motor_id, 
                     values=[str(i) for i in range(self.app.NUM_MOTORS)], width=3).pack(side=tk.LEFT, padx=5)
        
        self.fwd_btn = ttk.Button(control_frame, text="Test Forward", command=lambda: self.on_start_test(1))
        self.fwd_btn.pack(side=tk.LEFT, padx=10)

        self.rev_btn = ttk.Button(control_frame, text="Test Reverse", command=lambda: self.on_start_test(-1))
        self.rev_btn.pack(side=tk.LEFT, padx=10)
        
        self.stop_btn = ttk.Button(control_frame, text="Stop", command=self.on_stop_test, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=10)
        
        self.status_label = ttk.Label(control_frame, text="Ready", foreground="blue")
        self.status_label.pack(side=tk.LEFT, padx=20)

        # Results
        res_frame = ttk.Frame(top_frame)
        res_frame.pack(fill=tk.X, pady=5)
        self.result_label = ttk.Label(res_frame, text="Estimated Max Speed: N/A", font=('Helvetica', 12, 'bold'))
        self.result_label.pack()

        # Plot
        self.fig = Figure(figsize=(6, 5), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.parent)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, padx=10, pady=10)

    def on_start_test(self, direction):
        self.is_testing = True
        self.fwd_btn.config(state=tk.DISABLED)
        self.rev_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.measured_data = {'commanded': [], 'measured_speed': [], 'load': []}
        self.plot_data = {'x': [], 'y': []}
        self.result_label.config(text="Testing...")
        
        # Start Thread for the test sequence
        import threading
        self.test_thread = threading.Thread(target=self.run_test_sequence, args=(direction,), daemon=True)
        self.test_thread.start()

    def on_stop_test(self):
        self.is_testing = False
        self.status_label.config(text="Stopping...")

    def run_test_sequence(self, direction):
        motor_id_0_indexed = self.test_motor_id.get()
        current_cmd_mag = self.start_speed
        
        try:
            while self.is_testing and current_cmd_mag <= self.max_test_speed:
                current_cmd_signed = current_cmd_mag * direction
                self.status_label.config(text=f"Testing Speed: {current_cmd_signed}")
                
                # 1. Send Command
                phy_id = self.app.get_physical_id(motor_id_0_indexed)
                if phy_id >= 0:
                    self.app.send_serial_command(f"{phy_id},{current_cmd_signed}\n")
                
                # 2. Wait for stabilization
                time.sleep(self.step_duration)
                
                # 3. Sample Data
                measured_spd = self.app.current_speeds[motor_id_0_indexed] 
                measured_load = self.app.current_loads[motor_id_0_indexed]

                self.measured_data['commanded'].append(current_cmd_mag) # Plot against magnitude
                self.measured_data['measured_speed'].append(abs(measured_spd)) # Use abs for magnitude
                self.measured_data['load'].append(measured_load)
                
                # Update Plot Live
                self.plot_data['x'].append(current_cmd_mag)
                self.plot_data['y'].append(abs(measured_spd))
                self.update_plot()

                current_cmd_mag += self.step_size
            
            # End of Test
            phy_id = self.app.get_physical_id(motor_id_0_indexed)
            if phy_id >= 0:
                self.app.send_serial_command(f"{phy_id},0\n")
            self.analyze_results()
            
        except Exception as e:
            print(f"Test Error: {e}")
        
        self.is_testing = False
        self.parent.after(0, self.reset_ui)

    def update_plot(self):
        # Must run on main thread usually, but simple redraw might work
        # Or use after()
        self.parent.after(0, self._draw_plot)

    def _draw_plot(self):
        self.ax.clear()
        self.ax.plot(self.plot_data['x'], self.plot_data['y'], 'b-o')
        self.ax.set_title("Commanded Magnitude vs Measured Speed")
        self.ax.set_xlabel("Commanded Speed (Magnitude)")
        self.ax.set_ylabel("Measured Speed (Abs Steps/s)")
        self.ax.grid(True)
        self.canvas.draw()

    def analyze_results(self):
        # simple max finding
        if not self.measured_data['measured_speed']: return
        
        max_spd = max(self.measured_data['measured_speed'])
        
        # Find saturation point: where slope decreases significantly?
        # Or just report the max achieved.
        self.parent.after(0, lambda: self.result_label.config(text=f"Max Measured Speed: {max_spd} steps/s"))

    def reset_ui(self):
        self.fwd_btn.config(state=tk.NORMAL)
        self.rev_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Test Complete")
