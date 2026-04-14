import tkinter as tk
from tkinter import ttk
import os
import csv
import time

from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import numpy as np

class CalibrationTab:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.is_recording = False
        self.recording_motor_id = tk.IntVar(value=0) # Default to 0-indexed Motor 0
        self.run_data = []
        self.measured_peak_torque = tk.StringVar(value="N/A")
        self.user_reference_peak = tk.StringVar(value="")
        self.calibration_status = tk.StringVar(value="Ready. Click 'Start Recording' to begin.")
        self.create_widgets()

    def create_widgets(self):
        top_frame = ttk.Frame(self.parent)
        top_frame.pack(fill=tk.X, padx=10, pady=5)

        control_frame = ttk.LabelFrame(top_frame, text="Recording Controls", padding="10")
        control_frame.pack(fill=tk.X)
        ttk.Label(control_frame, text="Motor ID:").pack(side=tk.LEFT, padx=5)
        self.motor_id_combo = ttk.Combobox(control_frame, textvariable=self.recording_motor_id, 
                                           values=[str(i) for i in range(self.app.NUM_MOTORS)], width=3)
        self.motor_id_combo.pack(side=tk.LEFT, padx=5)
        self.start_button = ttk.Button(control_frame, text="Start Recording", command=self.on_recording_start)
        self.start_button.pack(side=tk.LEFT, padx=10)
        self.stop_button = ttk.Button(control_frame, text="Stop Recording", command=self.on_recording_stop, state=tk.DISABLED)
        self.stop_button.pack(side=tk.LEFT, padx=10)

        motor_control_frame = ttk.LabelFrame(top_frame, text="Motor Control (for selected ID)", padding="10")
        motor_control_frame.pack(fill=tk.X, pady=5)
        for speed in self.app.SPEED_PRESETS:
            btn = ttk.Button(motor_control_frame, text=str(speed), width=5,
                             command=lambda s=speed: self.on_cal_speed_button_click(s))
            btn.pack(side=tk.LEFT, padx=12, expand=True)

        result_frame = ttk.LabelFrame(top_frame, text="Results & Saving", padding="10")
        result_frame.pack(fill=tk.X, pady=5)
        res_left_frame = ttk.Frame(result_frame); res_left_frame.pack(side=tk.LEFT, padx=10)
        ttk.Label(res_left_frame, text="Measured Peak Torque:").pack(anchor=tk.W)
        ttk.Label(res_left_frame, textvariable=self.measured_peak_torque, font=('Helvetica', 12, 'bold')).pack(anchor=tk.W)
        res_right_frame = ttk.Frame(result_frame); res_right_frame.pack(side=tk.LEFT, padx=10)
        ttk.Label(res_right_frame, text="Your Reference Peak:").pack(anchor=tk.W)
        ttk.Entry(res_right_frame, textvariable=self.user_reference_peak, width=15).pack(anchor=tk.W)
        self.save_button = ttk.Button(result_frame, text="Save Results to CSV", command=self.on_save_results, state=tk.DISABLED)
        self.save_button.pack(side=tk.RIGHT, padx=20, pady=10)
        
        status_frame = ttk.LabelFrame(top_frame, text="Status", padding="10")
        status_frame.pack(fill=tk.X, pady=5)
        ttk.Label(status_frame, textvariable=self.calibration_status).pack()

        plot_frame = ttk.LabelFrame(self.parent, text="Calibration Curve", padding="10")
        plot_frame.pack(fill=tk.BOTH, expand=True, padx=10, pady=10)
        
        plot_button_frame = ttk.Frame(plot_frame)
        plot_button_frame.pack(pady=5)
        ttk.Button(plot_button_frame, text="Load & Plot for Selected Motor", command=self.on_load_and_draw_plot).pack(side=tk.LEFT, padx=5)
        ttk.Button(plot_button_frame, text="Load & Plot for ALL Motors", command=self.on_load_all_and_draw_plot).pack(side=tk.LEFT, padx=5)

        self.cal_fig = Figure(figsize=(7, 4), dpi=100)
        self.cal_ax = self.cal_fig.add_subplot(111)
        self.cal_canvas = FigureCanvasTkAgg(self.cal_fig, master=plot_frame)
        self.cal_canvas.draw(); self.cal_canvas.get_tk_widget().pack(side=tk.TOP, fill=tk.BOTH, expand=True)

    def on_cal_speed_button_click(self, speed):
        motor_id = self.recording_motor_id.get()
        phy_id = self.app.get_physical_id(motor_id)
        if phy_id >= 0:
            self.app.send_serial_command(f"{phy_id},{speed}\n")

    def on_load_and_draw_plot(self):
        motor_id = self.recording_motor_id.get()
        filename = f"calibration_motor_{motor_id}.csv"
        if not os.path.exists(filename):
            self.calibration_status.set(f"Error: File '{filename}' not found.")
            return
        
        ref_peaks, meas_peaks = self.read_calibration_file(filename)
        if ref_peaks is None: return

        title = f'Motor {motor_id} Calibration Curve'
        self.draw_calibration_plot(ref_peaks, meas_peaks, title)
        self.calibration_status.set(f"Plot updated from {filename}")

    def on_load_all_and_draw_plot(self):
        all_ref_peaks, all_meas_peaks = [], []
        files_read = 0
        for i in range(self.app.NUM_MOTORS): # Iterate from 0 to NUM_MOTORS - 1
            filename = f"calibration_motor_{i}.csv"
            if os.path.exists(filename):
                ref_peaks, meas_peaks = self.read_calibration_file(filename)
                if ref_peaks is not None:
                    all_ref_peaks.extend(ref_peaks)
                    all_meas_peaks.extend(meas_peaks)
                    files_read += 1
        
        if not all_ref_peaks:
            self.calibration_status.set("No calibration files found to plot.")
            return
            
        title = f'All Servos ({files_read} files) - Common Calibration Curve'
        self.draw_calibration_plot(all_ref_peaks, all_meas_peaks, title)
        self.calibration_status.set(f"Plotted aggregate data from {files_read} files.")

    def read_calibration_file(self, filename):
        ref_peaks, meas_peaks = [], []
        try:
            with open(filename, 'r', newline='') as f:
                reader = csv.DictReader(f)
                for row in reader:
                    ref_peaks.append(float(row['Reference_Peak']))
                    meas_peaks.append(float(row['Measured_Peak']))
            return ref_peaks, meas_peaks
        except (IOError, ValueError, KeyError) as e:
            self.calibration_status.set(f"Error reading {os.path.basename(filename)}: {e}")
            return None, None

    def draw_calibration_plot(self, x_data, y_data, title):
        self.cal_ax.clear()
        self.cal_ax.scatter(x_data, y_data, label='Data Points')
        if len(x_data) >= 2:
            try:
                x, y = np.array(x_data), np.array(y_data)
                m, b = np.polyfit(x, y, 1)
                self.cal_ax.plot(x, m*x + b, 'r-', label=f'Fit: y={m:.2f}x + {b:.2f}')
            except np.linalg.LinAlgError:
                self.calibration_status.set("Could not calculate line of best fit.")
        
        self.cal_ax.set_title(title)
        self.cal_ax.set_xlabel('Reference Peak (User Input)')
        self.cal_ax.set_ylabel('Measured Peak (Sensor Reading)')
        self.cal_ax.grid(True)
        self.cal_ax.legend()
        self.cal_canvas.draw()

    def process_torque_reading(self, torque):
        if self.is_recording: self.run_data.append(torque)

    def on_recording_start(self):
        self.run_data = []
        self.measured_peak_torque.set("N/A")
        self.start_button.config(state=tk.DISABLED); self.stop_button.config(state=tk.NORMAL)
        self.save_button.config(state=tk.DISABLED); self.motor_id_combo.config(state=tk.DISABLED)
        self.calibration_status.set(f"Recording torque for Motor {self.recording_motor_id.get()}...")
        self.is_recording = True

    def on_recording_stop(self):
        self.is_recording = False
        
        # Get the motor ID that was being recorded
        motor_id = self.recording_motor_id.get()

        if not self.run_data:
            self.calibration_status.set("Recording stopped. No data was captured.")
            self.measured_peak_torque.set("Error")
        else:
            peak_torque = max(self.run_data)
            self.measured_peak_torque.set(str(peak_torque))
            self.calibration_status.set("Recording stopped. Enter reference peak and save.")
        
        # Update button and combo box states
        self.start_button.config(state=tk.NORMAL)
        self.stop_button.config(state=tk.DISABLED)
        self.save_button.config(state=tk.NORMAL)
        self.motor_id_combo.config(state=tk.NORMAL)

        # Automatically stop the motor
        phy_id = self.app.get_physical_id(motor_id)
        if phy_id >= 0:
            self.app.send_serial_command(f"{phy_id},0\n")
        print(f"Stopped motor {motor_id} automatically after recording.")

    def on_save_results(self):
        user_peak = self.user_reference_peak.get()
        measured_peak = self.measured_peak_torque.get()
        motor_id = self.recording_motor_id.get()
        if not user_peak or measured_peak in ["N/A", "Error"]:
            self.calibration_status.set("Error: Cannot save. Record data and enter a reference peak."); return
        filename = f"calibration_motor_{motor_id}.csv"
        file_exists = os.path.exists(filename)
        try:
            with open(filename, 'a', newline='') as f:
                writer = csv.writer(f)
                header = ["Timestamp", "Reference_Peak", "Measured_Peak"]
                if not file_exists or f.tell() == 0: writer.writerow(header)
                timestamp = time.strftime("%Y-%m-%d %H:%M:%S")
                writer.writerow([timestamp, user_peak, measured_peak])
            self.calibration_status.set(f"Result saved to {filename}")
            self.save_button.config(state=tk.DISABLED)
        except IOError as e:
            self.calibration_status.set(f"Error: Could not save file. {e}")