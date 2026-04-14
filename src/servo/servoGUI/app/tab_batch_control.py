import tkinter as tk
from tkinter import ttk
import time

class BatchControlTab:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.target_values = [0] * self.app.NUM_MOTORS
        self.sliders = []
        self.labels_pos = [] 
        self.labels_load = []
        self.labels_target = []

        # Main UI
        main_frame = ttk.Frame(parent, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Batch Controls
        btn_frame = ttk.LabelFrame(main_frame, text="Batch Execution")
        btn_frame.pack(fill=tk.X, pady=10)
        
        ttk.Button(btn_frame, text="DRIVE (Send All Targets)", command=self.send_all).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, pady=10)
        ttk.Button(btn_frame, text="STOP ALL (Reset 0)", command=self.stop_all).pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10, pady=10)

        # Global Presets for Batch Target
        global_frame = ttk.Frame(main_frame)
        global_frame.pack(fill=tk.X, pady=5)
        ttk.Label(global_frame, text="Set All Sliders:").pack(side=tk.LEFT)
        for s in [-1000, 0, 1000]:
            ttk.Button(global_frame, text=str(s), command=lambda val=s: self.set_all_targets(val)).pack(side=tk.LEFT, padx=5)

        # Motors Container (Scrollable if needed, but 6 motors fit)
        self.motor_container = ttk.Frame(main_frame)
        self.motor_container.pack(fill=tk.BOTH, expand=True)

        for i in range(self.app.NUM_MOTORS):
            self.create_motor_row(i)

    def create_motor_row(self, motor_id):
        motor_name = self.app.MOTOR_NAMES[motor_id] if motor_id < len(self.app.MOTOR_NAMES) else ""
        frame = ttk.LabelFrame(self.motor_container, text=f"Motor {motor_id} ({motor_name})")
        frame.pack(fill=tk.X, pady=2, padx=5)
        
        # Top row: Slider
        row1 = ttk.Frame(frame)
        row1.pack(fill=tk.X)
        
        # Target Display (Create first to prevent IndexError on slider set)
        l_target = ttk.Label(row1, text="Target: 0", width=12, foreground="blue", font=('Helvetica', 10, 'bold'))
        self.labels_target.append(l_target)

        # Slider
        slider = ttk.Scale(row1, from_=-2000, to=2000, orient=tk.HORIZONTAL,
                           command=lambda val, mid=motor_id: self.on_slider(mid, val))
        slider.set(0)
        slider.pack(side=tk.LEFT, fill=tk.X, expand=True, padx=10)
        self.sliders.append(slider)
        
        # Pack Label after slider
        l_target.pack(side=tk.LEFT, padx=5)

        # Bottom row: Telemetry
        row2 = ttk.Frame(frame)
        row2.pack(fill=tk.X, padx=10, pady=2)
        
        l_pos = ttk.Label(row2, text="Pos: --", width=15)
        l_pos.pack(side=tk.LEFT)
        self.labels_pos.append(l_pos)
        
        l_load = ttk.Label(row2, text="Load: --", width=15)
        l_load.pack(side=tk.LEFT)
        self.labels_load.append(l_load)

        # Individual Presets (Similar to Page 1)
        for s in [-500, 0, 500]:
             ttk.Button(row2, text=str(s), width=5, 
                        command=lambda val=s, mid=motor_id: self.set_single_target(mid, val)).pack(side=tk.LEFT, padx=2)

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
        print("Batch Sending...")
        for i in range(self.app.NUM_MOTORS):
            val = self.target_values[i]
            # Send command
            phy_id = self.app.get_physical_id(i)
            if phy_id >= 0:
                self.app.send_serial_command(f"{phy_id},{val}\n")
            time.sleep(0.01)

    def stop_all(self):
        for i in range(self.app.NUM_MOTORS):
            self.sliders[i].set(0)
            phy_id = self.app.get_physical_id(i)
            if phy_id >= 0:
                self.app.send_serial_command(f"{phy_id},0\n")

    def update_telemetry(self, data_parts):
        # data_parts format: time, pos0, load0, spd0, pos1, load1, spd1...
        # Called from GUIMain read loop
        try:
            for i in range(self.app.NUM_MOTORS):
                base_idx = 1 + i*3
                p_idx = base_idx
                l_idx = base_idx + 1
                if p_idx < len(data_parts):
                    self.labels_pos[i].config(text=f"Pos: {data_parts[p_idx]}")
                if l_idx < len(data_parts):
                    self.labels_load[i].config(text=f"Load: {data_parts[l_idx]}")
        except Exception as e:
            print(f"Error in BatchControlTab update_telemetry: {e}")
