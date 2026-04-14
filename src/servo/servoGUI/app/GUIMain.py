import tkinter as tk
from tkinter import ttk
import os
import threading
import time

try:
    import serial
except Exception:
    serial = None

# 作成した各タブのクラスをインポート
from tab_manual_control import ManualControlTab
from tab_plotter import PlotterTab
from tab_calibration import CalibrationTab
from tab_monitor_all import MonitorAllTab
from tab_motion_recorder import MotionRecorderTab
from tab_batch_control import BatchControlTab
from tab_speed_test import SpeedTestTab # Import New Tab
from tab_id_manager import IDManagerTab # Import ID Manager
from tab_serial_monitor import SerialMonitorTab # Import Serial Monitor
from tab_position_calibration import PositionCalibrationTab # Import New Tab
from tab_high_accuracy_pos import HighAccuracyPositionTab # Import High Accuracy Position Tab
from tab_fidelity_test import FidelityTestTab # Import Fidelity Test
from tab_system_check import SystemCheckTab # Import System Check

# --- グローバル設定 ---
COM_PORT = "COM7"
BAUD_RATE = 921600
NUM_MOTORS = 6

# --- プリセットの定義 (各タブで共有) ---
SPEED_PRESETS = [-2000, -1000, -500, 0, 500, 1000, 2000]
POSITION_PRESETS = [-2048, -1024, -512, 0, 512, 1024, 2048] # New for Position Mode
GLOBAL_SPEED_PRESETS = [-2000, -1000, 0, 1000, 2000]
TIMED_RUN_SPEEDS = [-2000, -1000, 1000, 2000]
PLOTTER_SPEED_PRESETS = [-2000, -1000, -500, 0, 500, 1000, 2000]


# set environmenyt variable SERVOGUI_MOCK=1 to enable mock serial mode for testing without hardware
def _env_truthy(name):
    value = os.getenv(name, "").strip().lower()
    if len(value) >= 2 and value[0] == value[-1] and value[0] in {"'", '"'}:
        value = value[1:-1].strip()
    return value in {"1", "true", "yes", "on"}


class MockSerialPort:
    """Minimal serial-like backend to run GUI without physical hardware."""

    def __init__(self, timeout=1.0, num_motors=6):
        self.timeout = timeout
        self.num_motors = num_motors
        self.is_open = True
        self.positions = [0] * num_motors
        self.loads = [0] * num_motors
        self.speeds = [0] * num_motors
        self._stop_deadlines = [0.0] * num_motors
        self._boot_t = time.time()
        self._last_update_t = self._boot_t

    def close(self):
        self.is_open = False

    def write(self, data):
        if not self.is_open:
            return
        text = data.decode("utf-8", errors="replace")
        for raw in text.splitlines():
            cmd = raw.strip()
            if cmd:
                self._handle_command(cmd)

    def readline(self):
        if not self.is_open:
            return b""
        self._update_state()
        if self.timeout and self.timeout > 0:
            time.sleep(min(self.timeout, 0.03))
        timestamp_ms = int((time.time() - self._boot_t) * 1000)
        parts = [str(timestamp_ms)]
        for idx in range(self.num_motors):
            parts.append(str(int(self.positions[idx])))
            parts.append(str(int(self.loads[idx])))
            parts.append(str(int(self.speeds[idx])))
        return (",".join(parts) + "\n").encode("utf-8")

    @staticmethod
    def _is_int(value):
        if not value:
            return False
        if value[0] in {"+", "-"}:
            return value[1:].isdigit()
        return value.isdigit()

    def _safe_motor(self, value):
        if not value.isdigit():
            return None
        motor = int(value)
        if 0 <= motor < self.num_motors:
            return motor
        return None

    def _handle_command(self, cmd):
        if cmd in {"RESET_IDS", "LIST_IDS"}:
            return
        if cmd == "s":
            self.speeds = [0] * self.num_motors
            self._stop_deadlines = [0.0] * self.num_motors
            return
        if cmd == "r":
            self.positions = [0] * self.num_motors
            self.loads = [0] * self.num_motors
            self.speeds = [0] * self.num_motors
            self._stop_deadlines = [0.0] * self.num_motors
            return

        parts = cmd.split(",")
        if len(parts) >= 2 and parts[0].isdigit() and self._is_int(parts[1]):
            motor = self._safe_motor(parts[0])
            if motor is not None:
                self.speeds[motor] = int(parts[1])
                self._stop_deadlines[motor] = 0.0
            return

        if len(parts) == 4 and parts[0] == "d":
            motor = self._safe_motor(parts[1])
            if motor is not None and self._is_int(parts[2]) and parts[3].isdigit():
                self.speeds[motor] = int(parts[2])
                self._stop_deadlines[motor] = time.time() + (int(parts[3]) / 1000.0)
            return

        if len(parts) >= 3 and parts[0] == "x":
            motor = self._safe_motor(parts[1])
            if motor is not None and self._is_int(parts[2]):
                self.positions[motor] = int(parts[2])
                self.speeds[motor] = 0
                self._stop_deadlines[motor] = 0.0
            return

        if len(parts) == 2 and parts[0] in {"g", "p"}:
            motor = self._safe_motor(parts[1])
            if motor is not None:
                self.positions[motor] = 0
                self.speeds[motor] = 0
                self._stop_deadlines[motor] = 0.0

    def _update_state(self):
        now = time.time()
        dt = max(0.0, now - self._last_update_t)
        self._last_update_t = now

        for idx in range(self.num_motors):
            deadline = self._stop_deadlines[idx]
            if deadline > 0 and now >= deadline:
                self.speeds[idx] = 0
                self._stop_deadlines[idx] = 0.0

            self.positions[idx] += int(self.speeds[idx] * dt * 0.25)
            self.positions[idx] = max(-4096, min(4096, self.positions[idx]))

            base = abs(self.speeds[idx]) // 10
            ripple = ((int((now - self._boot_t) * 20) + (idx * 7)) % 18) - 9
            self.loads[idx] = max(0, min(1023, base + ripple))


class ControlGUI:
    def __init__(self, root):
        self.root = root
        self.root.title("Robot Control Application")
        self.root.geometry("900x950")

        # --- アプリケーション全体で共有するリソースと状態 ---
        self.serial_port = None
        self.is_running = True
        
        self.NUM_MOTORS = NUM_MOTORS
        self.SPEED_PRESETS = SPEED_PRESETS
        self.POSITION_PRESETS = POSITION_PRESETS # Add to instance
        self.GLOBAL_SPEED_PRESETS = GLOBAL_SPEED_PRESETS
        self.TIMED_RUN_SPEEDS = TIMED_RUN_SPEEDS
        self.PLOTTER_SPEED_PRESETS = PLOTTER_SPEED_PRESETS
        
        self.MOTOR_NAMES = ["LU", "PI", "ED", "DI", "FDS", "FDP"]

        self.sliders, self.pos_labels, self.load_labels = [], [], []
        self.timed_run_id, self.timed_run_ms = None, None
        self.pos_control_id = None
        
        # Store latest telemetry for other tabs (like Recorder)
        self.current_positions = [0] * self.NUM_MOTORS
        self.current_loads = [0] * self.NUM_MOTORS
        self.current_speeds = [0] * self.NUM_MOTORS # Initialize Speed Storage
        self.mock_mode = _env_truthy("SERVOGUI_MOCK")

        try:
            if serial is None:
                raise RuntimeError("pyserial is not available.")
            self.serial_port = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
            print(f"Successfully connected to {COM_PORT}")
            # Force Firmware to reset its internal ID map to defaults (0,1,2,3,4,5)
            # This fixes the issue where firmware RAM maps multiple indices to the same ID.
            time.sleep(2.5) # Wait for boot
            self.serial_port.write(b"RESET_IDS\n")
            print("Sent RESET_IDS to firmware to force strict 1:1 mapping.")
        except Exception as e:
            if self.mock_mode:
                print(
                    f"Could not connect to {COM_PORT} ({e}). "
                    "Starting mock serial mode (SERVOGUI_MOCK=1)."
                )
                self.serial_port = MockSerialPort(timeout=0.03, num_motors=self.NUM_MOTORS)
            else:
                error_label = ttk.Label(
                    self.root,
                    text=f"Error: Could not connect to {COM_PORT}.\n"
                    "Please check port or set SERVOGUI_MOCK=1.",
                    foreground="red",
                )
                error_label.pack(pady=20, padx=20)
                return

        # --- Top Control Bar ---
        self.create_top_bar()
        self.create_menu()

        # --- タブUIの作成 ---
        self.notebook = ttk.Notebook(self.root)
        self.notebook.pack(pady=10, padx=10, fill="both", expand=True)
        
        # Frames for tabs
        manual_tab_frame = ttk.Frame(self.notebook)
        plotter_tab_frame = ttk.Frame(self.notebook)
        monitor_all_frame = ttk.Frame(self.notebook)
        motion_recorder_frame = ttk.Frame(self.notebook)
        high_acc_pos_frame = ttk.Frame(self.notebook) # New frame for high accuracy position tab
        
        # Add Main Tabs
        self.notebook.add(manual_tab_frame, text="Manual Control")
        self.notebook.add(plotter_tab_frame, text="Real-time Plotter")
        self.notebook.add(monitor_all_frame, text="All Servos Monitor")
        self.notebook.add(motion_recorder_frame, text="Motion Recorder")
        self.notebook.add(high_acc_pos_frame, text="High Acc. Pos. Control") # New tab

        # --- Initialize Tab Instances ---
        self.manual_tab = ManualControlTab(manual_tab_frame, self)
        self.plotter_tab = PlotterTab(plotter_tab_frame, self)
        self.monitor_all_tab = MonitorAllTab(monitor_all_frame, self)
        self.motion_recorder_tab = MotionRecorderTab(motion_recorder_frame, self)
        self.high_acc_pos_tab = HighAccuracyPositionTab(high_acc_pos_frame, self) # New tab instance

        # --- Hidden Tabs (Initialized but not packed in main notebook initially) ---
        self.batch_window = None
        self.calibration_window = None
        self.speed_test_window = None
        self.id_manager_window = None
        self.serial_monitor_window = None
        self.position_calibration_window = None # New
        self.fidelity_test_window = None # New
        self.system_check_window = None # New
        
        self.batch_control_tab = None
        self.calibration_tab = None
        self.speed_test_tab = None
        self.id_manager_tab = None
        self.serial_monitor_tab = None
        self.position_calibration_tab = None # New
        self.fidelity_test_tab = None # New
        self.system_check_tab = None # New

        self.user_zero_offsets = [0] * self.NUM_MOTORS # New
        self.root.protocol("WM_DELETE_WINDOW", self.on_closing)
        self.read_thread = threading.Thread(target=self.read_from_arduino, daemon=True)
        self.read_thread.start()

    def create_menu(self):
        menubar = tk.Menu(self.root)
        self.root.config(menu=menubar)
        
        tools_menu = tk.Menu(menubar, tearoff=0)
        menubar.add_cascade(label="Tools", menu=tools_menu)
        
        tools_menu.add_command(label="Batch Control", command=self.open_batch_control)
        tools_menu.add_command(label="Torque Calibration", command=self.open_calibration)
        tools_menu.add_command(label="Max Speed Test", command=self.open_speed_test)
        tools_menu.add_command(label="ID Manager", command=self.open_id_manager)
        tools_menu.add_command(label="Serial Monitor", command=self.open_serial_monitor)
        tools_menu.add_command(label="Position Calibration", command=self.open_position_calibration)
        tools_menu.add_command(label="Servo Fidelity Test", command=self.open_fidelity_test)
        tools_menu.add_command(label="System Diagnostic Suite", command=self.open_system_check)

    def open_batch_control(self):
        if self.batch_window is None or not tk.Toplevel.winfo_exists(self.batch_window):
            self.batch_window = tk.Toplevel(self.root)
            self.batch_window.title("Batch Control")
            self.batch_window.geometry("600x800")
            self.batch_control_tab = BatchControlTab(self.batch_window, self)
        else:
            self.batch_window.lift()

    def open_calibration(self):
        if self.calibration_window is None or not tk.Toplevel.winfo_exists(self.calibration_window):
            self.calibration_window = tk.Toplevel(self.root)
            self.calibration_window.title("Torque Calibration")
            self.calibration_window.geometry("800x600")
            self.calibration_tab = CalibrationTab(self.calibration_window, self)
        else:
            self.calibration_window.lift()

    def open_speed_test(self):
        if self.speed_test_window is None or not tk.Toplevel.winfo_exists(self.speed_test_window):
            self.speed_test_window = tk.Toplevel(self.root)
            self.speed_test_window.title("Max Speed Test")
            self.speed_test_window.geometry("800x600")
            self.speed_test_tab = SpeedTestTab(self.speed_test_window, self)
        else:
            self.speed_test_window.lift()

    def open_id_manager(self):
        if self.id_manager_window is None or not tk.Toplevel.winfo_exists(self.id_manager_window):
            self.id_manager_window = tk.Toplevel(self.root)
            self.id_manager_window.title("ID Manager")
            self.id_manager_window.geometry("400x500")
            self.id_manager_tab = IDManagerTab(self.id_manager_window, self)
        else:
            self.id_manager_window.lift()

    def open_serial_monitor(self):
        if self.serial_monitor_window is None or not tk.Toplevel.winfo_exists(self.serial_monitor_window):
            self.serial_monitor_window = tk.Toplevel(self.root)
            self.serial_monitor_window.title("Serial Monitor")
            self.serial_monitor_window.geometry("600x400")
            self.serial_monitor_tab = SerialMonitorTab(self.serial_monitor_window, self)
        else:
            self.serial_monitor_window.lift()

    def open_position_calibration(self):
        if self.position_calibration_window is None or not tk.Toplevel.winfo_exists(self.position_calibration_window):
            self.position_calibration_window = tk.Toplevel(self.root)
            self.position_calibration_window.title("Position Calibration")
            self.position_calibration_window.geometry("800x600")
            self.position_calibration_tab = PositionCalibrationTab(self.position_calibration_window, self)
        else:
            self.position_calibration_window.lift()

    def open_fidelity_test(self):
        if self.fidelity_test_window is None or not tk.Toplevel.winfo_exists(self.fidelity_test_window):
            self.fidelity_test_window = tk.Toplevel(self.root)
            self.fidelity_test_window.title("Servo Fidelity Test")
            self.fidelity_test_window.geometry("800x600")
            self.fidelity_test_tab = FidelityTestTab(self.fidelity_test_window, self)
        else:
            self.fidelity_test_window.lift()

    def open_system_check(self):
        if self.system_check_window is None or not tk.Toplevel.winfo_exists(self.system_check_window):
            self.system_check_window = tk.Toplevel(self.root)
            self.system_check_window.title("Comprehensive System Check")
            self.system_check_window.geometry("900x800")
            self.system_check_tab = SystemCheckTab(self.system_check_window, self)
        else:
            self.system_check_window.lift()

    def create_top_bar(self):
        top_bar = ttk.Frame(self.root, padding="5", relief=tk.RAISED, borderwidth=1)
        top_bar.pack(fill=tk.X, side=tk.TOP)
        
        ttk.Label(top_bar, text="Global Controls:", font=('Helvetica', 10, 'bold')).pack(side=tk.LEFT, padx=5)
        
        # STOP Button (Red)
        stop_style = ttk.Style()
        stop_style.configure("Red.TButton", foreground="red")
        ttk.Button(top_bar, text="STOP ALL", command=self.emergency_stop, style="Red.TButton").pack(side=tk.LEFT, padx=5)
        
        # All Go Zero
        ttk.Button(top_bar, text="All Go Zero", command=self.all_go_to_zero).pack(side=tk.LEFT, padx=5)
        
        # Reset System
        ttk.Button(top_bar, text="Reset System", command=self.reset_system).pack(side=tk.LEFT, padx=5)
        
        # Reconnect
        ttk.Button(top_bar, text="Reconnect Serial", command=self.reconnect_serial).pack(side=tk.RIGHT, padx=5)

        # Reset IDs (Force 1:1)
        ttk.Button(top_bar, text="Reset IDs (Force 1:1)", command=self.force_reset_ids).pack(side=tk.RIGHT, padx=5)

    def force_reset_ids(self):
        print("Sending RESET_IDS to firmware...")
        self.send_serial_command("RESET_IDS\n")

    def emergency_stop(self):
        print("EMERGENCY STOP!")
        self.send_serial_command("s\n") # 's' command stops all motors in st_control3.ino

    def all_go_to_zero(self):
        print("Commanding all servos to go to zero...")
        for i in range(self.NUM_MOTORS):
            # Use mapped ID
            phy_id = self.get_physical_id(i)
            if phy_id >= 0:
                self.send_serial_command(f"g,{phy_id}\n")
            time.sleep(0.005) 

    def reset_system(self):
        print("Resetting system...")
        self.send_serial_command("r\n")

    def reconnect_serial(self):
        print("Reconnecting serial...")
        if self.serial_port and self.serial_port.is_open:
            self.serial_port.close()
        
        time.sleep(0.5)
        if self.mock_mode:
            self.serial_port = MockSerialPort(timeout=0.03, num_motors=self.NUM_MOTORS)
            print("Mock serial reconnected.")
            return
        try:
            if serial is None:
                raise RuntimeError("pyserial is not available.")
            self.serial_port = serial.Serial(COM_PORT, BAUD_RATE, timeout=1)
            print(f"Successfully reconnected to {COM_PORT}")
        except Exception as e:
            print(f"Error reconnecting: {e}")

    def slider_moved(self, value, motor_id):
        pass 

    def on_preset_button_click(self, motor_id, speed):
        phy_id = self.get_physical_id(motor_id)
        if phy_id >= 0:
            self.send_serial_command(f"{phy_id},{speed},1\n") 
        if self.manual_tab:
             self.manual_tab.update_slider(motor_id, speed)

    def on_global_preset_button_click(self, speed):
        mode = "speed"
        if self.manual_tab:
            try:
                mode = self.manual_tab.control_mode.get()
            except: pass

        if mode == "position":
            # Position Mode Logic: Map speed-like presets to 2^N values
            val = speed
            mapped_val = val
            if abs(val) == 2000: mapped_val = 2048 if val > 0 else -2048
            elif abs(val) == 1000: mapped_val = 1024 if val > 0 else -1024
            elif abs(val) == 500: mapped_val = 512 if val > 0 else -512
            elif val == 0: mapped_val = 0
            
            # Update Sliders (this triggers the 'x' command in manual_tab)
            if self.manual_tab:
                for i in range(self.NUM_MOTORS):
                    # Check bounds to avoid index error
                    if i < len(self.manual_tab.sliders):
                        self.manual_tab.sliders[i].set(mapped_val)
        else:
            # Speed Mode Logic (Original)
            for i in range(self.NUM_MOTORS):
                phy_id = self.get_physical_id(i)
                if phy_id >= 0:
                    self.send_serial_command(f"{phy_id},{speed},1\n")
                if self.manual_tab:
                    self.manual_tab.update_slider(i, speed)

    def on_timed_run_click(self, speed):
        motor_id_str = self.timed_run_id.get() 
        duration = self.timed_run_ms.get()
        if motor_id_str.isdigit() and duration.isdigit():
            motor_id_0_indexed = int(motor_id_str)
            phy_id = self.get_physical_id(motor_id_0_indexed)
            if phy_id >= 0:
                self.send_serial_command(f"d,{phy_id},{speed},{duration}\n")

    def on_set_zero_click(self):
        motor_id_str = self.pos_control_id.get()
        if motor_id_str.isdigit():
            motor_id_0_indexed = int(motor_id_str)
            self.user_zero_offsets[motor_id_0_indexed] = 0 
            phy_id = self.get_physical_id(motor_id_0_indexed)
            if phy_id >= 0:
                self.send_serial_command(f"p,{phy_id}\n")

    def on_set_all_zero_click(self):
        print("Setting ALL motors to Zero Position...")
        for i in range(self.NUM_MOTORS):
            self.user_zero_offsets[i] = 0
            phy_id = self.get_physical_id(i)
            if phy_id >= 0:
                self.send_serial_command(f"p,{phy_id}\n")
            time.sleep(0.01)

    def on_go_to_zero_click(self):
        motor_id_str = self.pos_control_id.get()
        if motor_id_str.isdigit():
            motor_id_0_indexed = int(motor_id_str)
            phy_id = self.get_physical_id(motor_id_0_indexed)
            if phy_id >= 0:
                self.send_serial_command(f"g,{phy_id}\n")

    def send_serial_command(self, command):
        if self.serial_port and self.serial_port.is_open:
            try:
                print(f"Sending: {command.strip()}") # Debugging enabled
                self.serial_port.write(command.encode('utf-8'))
            except Exception as e:
                print(f"Error writing to serial port: {e}")

    def get_physical_id(self, gui_index):
        # User requested logic: 1:1 mapping
        # Motor N (GUI) -> ID N
        return gui_index

    def read_from_arduino(self):
        time.sleep(2)
        while self.is_running:
            try:
                if self.serial_port and self.serial_port.is_open:
                    try:
                        line = self.serial_port.readline().decode('utf-8', errors='replace').strip()
                    except UnicodeDecodeError:
                        continue 
                        
                    if not line: continue
                    
                    if self.serial_monitor_tab:
                        self.serial_monitor_tab.add_line(line)
                    
                    if any(k in line for k in ["FOUND_ID", "SCAN_", "CHANGED_ID", "VERIFY", "NVS_", "EXECUTING", "CMD_", "SUCCESS", "FAIL"]):
                        print(f"DEBUG: ID Manager Message Intercepted: '{line}'")
                        if self.id_manager_tab:
                            self.id_manager_tab.process_scan_result(line)
                        continue

                    # Parsing Logic
                    parts = line.split(',')
                    num_parts = len(parts)
                    
                    # Format 1: Time, Pos0, Load0, Spd0... (1 + 6*3 = 19 parts)
                    expected_3val = 1 + (self.NUM_MOTORS * 3)
                    # Format 2: Time, Pos0, Load0... (1 + 6*2 = 13 parts)
                    expected_2val = 1 + (self.NUM_MOTORS * 2)

                    if (num_parts == expected_3val or num_parts == expected_2val) and parts[0].isdigit():
                        try:
                            for i in range(self.NUM_MOTORS):
                                phy_id = self.get_physical_id(i)
                                if phy_id >= 0:
                                    if num_parts == expected_3val:
                                        base_idx = 1 + (phy_id * 3)
                                        p_idx, l_idx, s_idx = base_idx, base_idx+1, base_idx+2
                                    else:
                                        base_idx = 1 + (phy_id * 2)
                                        p_idx, l_idx, s_idx = base_idx, base_idx+1, -1

                                    if base_idx + (2 if num_parts==expected_3val else 1) < len(parts):
                                        self.current_positions[i] = int(parts[p_idx])
                                        self.current_loads[i] = int(parts[l_idx])
                                        if s_idx != -1:
                                            self.current_speeds[i] = int(parts[s_idx])
                                        else:
                                            self.current_speeds[i] = 0
                                else:
                                    self.current_positions[i] = 0
                                    self.current_loads[i] = 0
                                    self.current_speeds[i] = 0
                        except ValueError:
                            pass
                        
                        self.update_manual_tab_labels(parts)
                        
                        if self.batch_control_tab:
                            self.batch_control_tab.update_telemetry(parts)
                        
                        # Plotter Update using parsed data
                        try:
                            motor_id_plot_0_indexed = int(self.plotter_tab.plotting_motor_id.get())
                            if 0 <= motor_id_plot_0_indexed < self.NUM_MOTORS:
                                pos = self.current_positions[motor_id_plot_0_indexed]
                                torque_plot = self.current_loads[motor_id_plot_0_indexed]
                                self.plotter_tab.process_data(motor_id_plot_0_indexed, pos, torque_plot)
                        except: pass

                        # Calibration Update using parsed data
                        if self.calibration_tab:
                            try:
                                motor_id_cal_0_indexed = int(self.calibration_tab.recording_motor_id.get())
                                if 0 <= motor_id_cal_0_indexed < self.NUM_MOTORS:
                                    torque_cal = self.current_loads[motor_id_cal_0_indexed]
                                    self.calibration_tab.process_torque_reading(torque_cal)
                            except: pass

                        self.monitor_all_tab.process_data(parts)
                        self.high_acc_pos_tab.update_telemetry(parts)
                else:
                    time.sleep(0.1) 

            except Exception as e:
                print(f"Error in read_from_arduino: {e}")

    def update_manual_tab_labels(self, data_parts):
        try:
            for i in range(self.NUM_MOTORS):
                raw_pos = self.current_positions[i]
                load_val = self.current_loads[i]
                
                calibrated_pos = raw_pos - self.user_zero_offsets[i] 
                
                self.pos_labels[i].config(text=f"Pos: {calibrated_pos}")
                self.load_labels[i].config(text=f"Load: {load_val}")
        except (IndexError, tk.TclError) as e:
            print(f"Error in update_manual_tab_labels: {e}")

    def on_closing(self):
        self.is_running = False
        time.sleep(0.2)
        if self.serial_port and self.serial_port.is_open:
            print("Closing application... Stopping motors.")
            for i in range(self.NUM_MOTORS):
                phy_id = self.get_physical_id(i)
                if phy_id >= 0:
                    self.send_serial_command(f"{phy_id},0\n")
                time.sleep(0.01)
            self.serial_port.close()
            print("Serial port closed.")
        self.root.destroy()

if __name__ == "__main__":
    root = tk.Tk()
    app = ControlGUI(root)
    root.mainloop()
