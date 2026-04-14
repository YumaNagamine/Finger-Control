import tkinter as tk
from tkinter import ttk

class HighAccuracyPositionTab:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        
        # Track the last commanded target to allow for rapid clicking (Option A)
        # Initialize with None, will be synced to current position on first move
        self.last_commanded_positions = [None] * self.app.NUM_MOTORS 
        self.custom_step_entries = []
        self.pos_labels = []
        
        self.create_ui()

    def create_ui(self):
        main_frame = ttk.Frame(self.parent, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)
        
        # Scrollable container in case screen is small, though 6 motors fit fine usually.
        # For simplicity, just a frame.
        
        for i in range(self.app.NUM_MOTORS):
            self.create_motor_row(main_frame, i)

    def create_motor_row(self, parent, motor_id):
        motor_name = self.app.MOTOR_NAMES[motor_id] if motor_id < len(self.app.MOTOR_NAMES) else ""
        
        row_frame = ttk.LabelFrame(parent, text=f"Motor {motor_id} ({motor_name})")
        row_frame.pack(fill=tk.X, pady=5, padx=5)
        
        # Top part: Status
        status_frame = ttk.Frame(row_frame)
        status_frame.pack(fill=tk.X, padx=5, pady=2)
        
        pos_lbl = ttk.Label(status_frame, text="Pos: --", font=('Consolas', 10))
        pos_lbl.pack(side=tk.LEFT)
        self.pos_labels.append(pos_lbl)

        # Bottom part: Controls
        ctrl_frame = ttk.Frame(row_frame)
        ctrl_frame.pack(fill=tk.X, padx=5, pady=5)

        # Negative Steps
        ttk.Button(ctrl_frame, text="-100", width=6, command=lambda: self.move_step(motor_id, -100)).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl_frame, text="-10", width=5, command=lambda: self.move_step(motor_id, -10)).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl_frame, text="-1", width=4, command=lambda: self.move_step(motor_id, -1)).pack(side=tk.LEFT, padx=2)

        # Custom
        entry = ttk.Entry(ctrl_frame, width=6, justify='center')
        entry.insert(0, "50")
        entry.pack(side=tk.LEFT, padx=10)
        self.custom_step_entries.append(entry)
        
        # Custom Move Buttons (Negative/Positive for the value in entry? Or just one button?)
        # Requirement: "custom steps. when we press it, it should move those many steps in positive or negative direction"
        # Let's add two buttons for the custom value: "Custom -" and "Custom +"
        
        ttk.Button(ctrl_frame, text="< Custom", width=8, 
                   command=lambda: self.move_custom(motor_id, -1)).pack(side=tk.LEFT, padx=2)
        ttk.Button(ctrl_frame, text="Custom >", width=8, 
                   command=lambda: self.move_custom(motor_id, 1)).pack(side=tk.LEFT, padx=2)

        # Positive Steps
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
        # Option A: Last Commanded Target logic
        
        # 1. Sync if first run or invalid
        if self.last_commanded_positions[motor_id] is None:
            # Sync to current actual position
            # If firmware just booted, it might be anywhere.
            # We accept whatever the servo reports.
            self.last_commanded_positions[motor_id] = self.app.current_positions[motor_id]

        # 2. Update Target
        new_target = self.last_commanded_positions[motor_id] + step
        
        # 3. Hardware Clamp (0-4095) for Servo PID Mode
        if new_target < 0: new_target = 0
        if new_target > 4095: new_target = 4095
        
        self.last_commanded_positions[motor_id] = new_target
        
        print(f"Motor {motor_id} Step {step} -> Target {new_target}")

        phy_id = self.app.get_physical_id(motor_id)
        if phy_id >= 0:
            # Using 'x' command: x,ID,Pos,Time
            # Firmware 'x' command is now Hardware Servo Mode (0-4095)
            self.app.send_serial_command(f"x,{phy_id},{new_target},0\n")

    def update_telemetry(self, data_parts):
        # Called from GUIMain
        # New Format: Time, Pos0, Load0, Spd0, Pos1 ...
        for i in range(self.app.NUM_MOTORS):
            if i < len(self.pos_labels):
                raw_pos = self.app.current_positions[i]
                self.pos_labels[i].config(text=f"Pos: {raw_pos}")