import tkinter as tk
from tkinter import ttk
import time
import threading
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class PositionCalibrationTab:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        
        self.calibration_motor_id = tk.IntVar(value=0) # 0-indexed
        
        self.create_widgets()

    def create_widgets(self):
        top_frame = ttk.Frame(self.parent, padding="10")
        top_frame.pack(fill=tk.X)

        # Motor Selection
        motor_select_frame = ttk.LabelFrame(top_frame, text="Motor Selection")
        motor_select_frame.pack(fill=tk.X, pady=5)
        ttk.Label(motor_select_frame, text="Motor ID:").pack(side=tk.LEFT, padx=5)
        ttk.Combobox(motor_select_frame, textvariable=self.calibration_motor_id, 
                     values=[str(i) for i in range(self.app.NUM_MOTORS)], width=3).pack(side=tk.LEFT, padx=5)

        # Position Control (Manual Move)
        pos_control_frame = ttk.LabelFrame(top_frame, text="Position Control")
        pos_control_frame.pack(fill=tk.X, pady=5)
        
        self.pos_slider = ttk.Scale(pos_control_frame, from_=-2048, to=2047, orient=tk.HORIZONTAL, length=300,
                                     command=self.on_pos_slider_moved)
        self.pos_slider.set(0) # Center
        self.pos_slider.pack(side=tk.LEFT, padx=10)
        
        self.pos_value_label = ttk.Label(pos_control_frame, text="Pos: 0", width=10)
        self.pos_value_label.pack(side=tk.LEFT)

        # Quick Position Buttons
        btn_frame = ttk.Frame(pos_control_frame)
        btn_frame.pack(side=tk.LEFT, padx=10)
        ttk.Button(btn_frame, text="Go to Center (0)", command=lambda: self.go_to_pos(0)).pack(side=tk.TOP, pady=2)
        ttk.Button(btn_frame, text="Go to Max CW (+2047)", command=lambda: self.go_to_pos(2047)).pack(side=tk.TOP, pady=2)
        ttk.Button(btn_frame, text="Go to Max CCW (-2048)", command=lambda: self.go_to_pos(-2048)).pack(side=tk.TOP, pady=2)

        # Calibration Actions
        cal_action_frame = ttk.LabelFrame(top_frame, text="Calibration Actions")
        cal_action_frame.pack(fill=tk.X, pady=5)
        
        ttk.Button(cal_action_frame, text="Set Current as User Zero", command=self.set_user_zero).pack(fill=tk.X, pady=5)
        ttk.Button(cal_action_frame, text="Reset User Zero", command=self.reset_user_zero).pack(fill=tk.X, pady=5)

        ttk.Separator(cal_action_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        self.find_min_step_btn = ttk.Button(cal_action_frame, text="Find Min Step (for Selected Motor)", command=self.start_find_min_step)
        self.find_min_step_btn.pack(fill=tk.X, pady=5)
        self.min_step_result_label = ttk.Label(cal_action_frame, text="Effective Min Step: N/A", font=('Helvetica', 10, 'bold'))
        self.min_step_result_label.pack(anchor=tk.W, padx=5, pady=2)

        # Current Status Display
        status_frame = ttk.LabelFrame(top_frame, text="Current Status")
        status_frame.pack(fill=tk.X, pady=5)
        
        self.current_pos_label = ttk.Label(status_frame, text="Current Pos (Raw): N/A")
        self.current_pos_label.pack(anchor=tk.W)
        self.calibrated_pos_label = ttk.Label(status_frame, text="Calibrated Pos: N/A")
        self.calibrated_pos_label.pack(anchor=tk.W)
        self.user_zero_label = ttk.Label(status_frame, text="User Zero Offset: N/A")
        self.user_zero_label.pack(anchor=tk.W)

        # Matplotlib Graph for Min Step Test
        self.fig = Figure(figsize=(5, 3), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=self.parent)
        self.canvas_widget = self.canvas.get_tk_widget()
        self.canvas_widget.pack(side=tk.BOTTOM, fill=tk.BOTH, expand=True, padx=5, pady=5)

    def on_pos_slider_moved(self, value):
        val = int(float(value))
        self.pos_value_label.config(text=f"Pos: {val}")
        self.go_to_pos(val)

    def go_to_pos(self, pos_value):
        motor_id_0_indexed = int(self.calibration_motor_id.get())
        # Convert -2048 to +2047 to 0-4095 for firmware
        converted_val = pos_value + 2048
        if converted_val < 0: converted_val = 0
        if converted_val > 4095: converted_val = 4095
        phy_id = self.app.get_physical_id(motor_id_0_indexed)
        if phy_id >= 0:
            self.app.send_serial_command(f"x,{phy_id},{converted_val},0\n")

    def set_user_zero(self):
        motor_id_0_indexed = int(self.calibration_motor_id.get())
        current_accumulated_pos = self.app.current_positions[motor_id_0_indexed]
        self.app.user_zero_offsets[motor_id_0_indexed] = current_accumulated_pos
        self.update_status_display()
        
    def reset_user_zero(self):
        motor_id_0_indexed = int(self.calibration_motor_id.get())
        self.app.user_zero_offsets[motor_id_0_indexed] = 0
        self.update_status_display()

    def start_find_min_step(self):
        self.find_min_step_btn.config(state=tk.DISABLED)
        self.min_step_result_label.config(text="Finding...")
        threading.Thread(target=self._find_min_step_thread, daemon=True).start()

    def _find_min_step_thread(self):
        motor_id_0_indexed = int(self.calibration_motor_id.get())
        
        # 1. Go to Center
        self.go_to_pos(0) # Center (maps to 2048 physical)
        time.sleep(1.0) # Wait to settle
        
        # Ensure we are in Position Mode for this motor in firmware
        phy_id = self.app.get_physical_id(motor_id_0_indexed)
        if phy_id >= 0:
            self.app.send_serial_command(f"x,{phy_id},2048,0\n") # Send physical 2048
        time.sleep(0.5)

        initial_raw_pos = self.app.current_positions[motor_id_0_indexed]
        
        commanded_steps_list = []
        measured_movement_list = []
        min_step_found = -1
        
        # Test steps from 1 to 10
        for step_size in range(1, 11): 
            # Command target slightly offset
            self.parent.after(0, lambda s=step_size: self.min_step_result_label.config(text=f"Testing {s} steps..."))
            
            target_pos_calibrated = step_size # Test positive direction
            
            self.go_to_pos(target_pos_calibrated) # Send command
            time.sleep(0.5) # Wait for servo to move and settle

            measured_raw_pos = self.app.current_positions[motor_id_0_indexed]
            actual_movement = measured_raw_pos - initial_raw_pos
            
            commanded_steps_list.append(step_size)
            measured_movement_list.append(actual_movement)
            
            if min_step_found == -1 and abs(actual_movement) > 0: # Check for first non-zero movement
                 min_step_found = step_size
        
        # Plot Results
        self.parent.after(0, lambda: self._draw_min_step_plot(commanded_steps_list, measured_movement_list))
        
        self.go_to_pos(0) # Return to center
        self.parent.after(0, lambda: self.min_step_result_label.config(text=f"Effective Min Step: {min_step_found}"))
        self.parent.after(0, lambda: self.find_min_step_btn.config(state=tk.NORMAL))

    def _draw_min_step_plot(self, commanded, measured):
        self.ax.clear()
        self.ax.plot(commanded, measured, 'b-o', label='Measured Movement')
        self.ax.set_title("Commanded vs Measured Step")
        self.ax.set_xlabel("Commanded Steps (from Center)")
        self.ax.set_ylabel("Measured Movement (Raw Steps)")
        self.ax.axhline(0, color='grey', linestyle='--', linewidth=0.8)
        self.ax.set_xticks(commanded)
        self.ax.grid(True)
        self.ax.legend()
        self.canvas.draw()

    def update_status_display(self):
        motor_id_0_indexed = int(self.calibration_motor_id.get())
        raw_pos = self.app.current_positions[motor_id_0_indexed]
        user_offset = self.app.user_zero_offsets[motor_id_0_indexed]
        
        calibrated_pos = raw_pos - user_offset
        
        self.current_pos_label.config(text=f"Current Pos (Raw): {raw_pos}")
        self.calibrated_pos_label.config(text=f"Calibrated Pos: {calibrated_pos}")
        self.user_zero_label.config(text=f"User Zero Offset: {user_offset}")
