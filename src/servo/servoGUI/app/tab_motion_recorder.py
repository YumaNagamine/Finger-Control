import tkinter as tk
from tkinter import ttk
import time
import threading
import json
import os

class MotionRecorderTab:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        
        self.is_recording = False
        self.is_playing = False
        self.record_start_time = 0
        self.recorded_frames = [] # List of {"t": time, "positions": [p0...p5]}
        self.motion_files_dir = "recorded_motions"
        
        # Playback parameters
        self.playback_kp = 8.923 # P-gain
        self.playback_ki = 2.15  # I-gain
        self.playback_kd = 1.537 # D-gain
        self.playback_speed_limit = 1000 # Safety cap
        self.integral_max = 5000 # Anti-windup limit for integral term
        
        # PID state variables for each motor (dictionaries keyed by motor_id)
        self.prev_errors = {}
        self.integrals = {}
        self.last_pid_times = {}
        
        # Initialize PID state for all motors
        for i in range(self.app.NUM_MOTORS):
            self.prev_errors[i] = 0
            self.integrals[i] = 0
            self.last_pid_times[i] = time.time()


        self.create_ui()
        self.update_motion_list()

        # Start the recording loop (runs in main thread via .after)
        self.record_loop()

    def create_ui(self):
        main_layout = ttk.Frame(self.parent, padding="10")
        main_layout.pack(fill=tk.BOTH, expand=True)

        # --- Left Panel: Controls ---
        control_frame = ttk.LabelFrame(main_layout, text="Recorder Controls")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)

        self.status_label = ttk.Label(control_frame, text="Status: Idle", font=("Helvetica", 12, "bold"))
        self.status_label.pack(pady=20)

        self.record_btn = ttk.Button(control_frame, text="Record Motion", command=self.toggle_recording)
        self.record_btn.pack(pady=10, fill=tk.X)

        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        self.play_btn = ttk.Button(control_frame, text="Play Selected", command=self.play_motion)
        self.play_btn.pack(pady=10, fill=tk.X)
        
        self.stop_play_btn = ttk.Button(control_frame, text="Stop Playback", command=self.stop_playback, state=tk.DISABLED)
        self.stop_play_btn.pack(pady=10, fill=tk.X)
        
        # Torque Controls
        torque_frame = ttk.Frame(control_frame)
        torque_frame.pack(pady=10, fill=tk.X)
        ttk.Button(torque_frame, text="Torque ON (Stop)", command=self.torque_on).pack(side=tk.LEFT, expand=True, fill=tk.X)
        
        # Gain tuning for playback (since we do P-control in Python)
        gain_frame = ttk.LabelFrame(control_frame, text="Playback Tuning")
        gain_frame.pack(pady=20, fill=tk.X)
        
        ttk.Label(gain_frame, text="P-Gain:").pack(side=tk.LEFT)
        self.gain_var = tk.DoubleVar(value=self.playback_kp)
        tk.Entry(gain_frame, textvariable=self.gain_var, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Button(gain_frame, text="Set", command=self.update_gain).pack(side=tk.LEFT)

        # Delete functionality
        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        self.delete_btn = ttk.Button(control_frame, text="Delete Selected", command=self.delete_motion)
        self.delete_btn.pack(pady=10, fill=tk.X)

        # --- Right Panel: Motion List ---
        list_frame = ttk.LabelFrame(main_layout, text="Saved Motions")
        list_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)

        self.motion_listbox = tk.Listbox(list_frame, font=("Courier", 10))
        self.motion_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)
        
        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.motion_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.motion_listbox.config(yscrollcommand=scrollbar.set)

    def update_gain(self):
        self.playback_kp = self.gain_var.get()
        print(f"Playback Gain set to: {self.playback_kp}")

    def torque_on(self):
        self.app.send_serial_command("s\n")
        print("Sent Stop command (Torque ON)")

    def delete_motion(self):
        selection = self.motion_listbox.curselection()
        if not selection:
            return
        
        motion_name = self.motion_listbox.get(selection[0])
        filename = os.path.join(self.motion_files_dir, f"{motion_name}.json")
        
        if os.path.exists(filename):
            try:
                os.remove(filename)
                print(f"Deleted {filename}")
                self.update_motion_list()
            except Exception as e:
                print(f"Error deleting file: {e}")

    def toggle_recording(self):
        if not self.is_recording:
            self.start_recording()
        else:
            self.stop_recording()

    def start_recording(self):
        self.is_recording = True
        self.record_btn.config(text="Stop Recording")
        self.status_label.config(text="Status: Recording...", foreground="red")
        self.play_btn.config(state=tk.DISABLED)
        self.delete_btn.config(state=tk.DISABLED)
        
        self.recorded_frames = []
        self.record_start_time = time.time()
        print("Recording started...")

    def stop_recording(self):
        self.is_recording = False
        self.record_btn.config(text="Record Motion")
        self.status_label.config(text="Status: Idle", foreground="black")
        self.play_btn.config(state=tk.NORMAL)
        self.delete_btn.config(state=tk.NORMAL)
        
        print(f"Recording stopped. {len(self.recorded_frames)} frames captured.")
        self.save_motion()

    def record_loop(self):
        if self.is_recording:
            # Deep copy current positions
            current_pos = list(self.app.current_positions)
            t = round(time.time() - self.record_start_time, 3)
            
            frame = {
                "t": t,
                "positions": current_pos
            }
            self.recorded_frames.append(frame)

        # Record at ~20Hz (50ms)
        self.parent.after(50, self.record_loop)

    def save_motion(self):
        if not self.recorded_frames:
            return

        # Create directory if it doesn't exist
        if not os.path.exists(self.motion_files_dir):
            os.makedirs(self.motion_files_dir)

        idx = 0
        while True:
            name = f"Motion {chr(65 + idx)}" if idx < 26 else f"Motion {idx}"
            filename = os.path.join(self.motion_files_dir, f"{name}.json")
            if not os.path.exists(filename):
                break
            idx += 1
        
        with open(filename, 'w') as f:
            json.dump(self.recorded_frames, f)
        
        print(f"Saved to {filename}")
        self.update_motion_list()

    def update_motion_list(self):
        self.motion_listbox.delete(0, tk.END)
        if not os.path.exists(self.motion_files_dir):
            return
            
        files = sorted([f for f in os.listdir(self.motion_files_dir) if f.endswith(".json")])
        for f in files:
            self.motion_listbox.insert(tk.END, f.replace(".json", ""))

    def play_motion(self):
        selection = self.motion_listbox.curselection()
        if not selection:
            return
        
        motion_name = self.motion_listbox.get(selection[0])
        filename = os.path.join(self.motion_files_dir, f"{motion_name}.json")
        
        try:
            with open(filename, 'r') as f:
                frames = json.load(f)
            
            if not frames:
                return

            threading.Thread(target=self.playback_thread, args=(frames,), daemon=True).start()
            
        except Exception as e:
            print(f"Error loading motion: {e}")

    def stop_playback(self):
        self.is_playing = False

    def playback_thread(self, frames):
        self.is_playing = True
        self.play_btn.config(state=tk.DISABLED)
        self.record_btn.config(state=tk.DISABLED)
        self.delete_btn.config(state=tk.DISABLED)
        self.stop_play_btn.config(state=tk.NORMAL)
        
        # 1. Move to Start Position (Closed Loop via Python)
        self.status_label.config(text="Status: Moving to Start...", foreground="blue")
        start_pos = frames[0]["positions"]
        
        if not self.move_to_target(start_pos, timeout=5.0):
            print("Failed to reach start position or stopped.")
            self.finish_playback()
            return

        # 2. Replay Motion (Feed-Forward + P-Control)
        self.status_label.config(text="Status: Playing...", foreground="green")
        start_time = time.time()
        
        for i in range(1, len(frames)):
            if not self.is_playing:
                break
            
            target_frame = frames[i]
            target_t = target_frame["t"]
            target_pos = target_frame["positions"]
            
            # Sync time
            now = time.time() - start_time
            wait = target_t - now
            if wait > 0:
                time.sleep(wait)
            
            # Control Loop
            for m_id in range(self.app.NUM_MOTORS):
                current = self.app.current_positions[m_id]
                goal = target_pos[m_id]
                
                error = goal - current

                # PID calculation
                dt = time.time() - self.last_pid_times[m_id]
                if dt == 0: dt = 0.000001 # Avoid division by zero
                self.last_pid_times[m_id] = time.time()

                # P Term
                p_term = self.playback_kp * error

                # I Term
                self.integrals[m_id] += error * dt
                # Anti-windup
                if self.integrals[m_id] > self.integral_max: self.integrals[m_id] = self.integral_max
                elif self.integrals[m_id] < -self.integral_max: self.integrals[m_id] = -self.integral_max
                i_term = self.playback_ki * self.integrals[m_id]

                # D Term
                d_term = self.playback_kd * (error - self.prev_errors[m_id]) / dt
                self.prev_errors[m_id] = error # Store current error for next iteration

                # Total PID output
                output = p_term + i_term + d_term
                speed = int(output)
                
                # Clamp speed
                if speed > self.playback_speed_limit: speed = self.playback_speed_limit
                if speed < -self.playback_speed_limit: speed = -self.playback_speed_limit
                
                # Deadband to prevent chatter
                if abs(error) < 10:
                    speed = 0
                    # Keep integral term, allow it to accumulate normally
                
                phy_id = self.app.get_physical_id(m_id)
                if phy_id >= 0:
                    self.app.send_serial_command(f"{phy_id},{speed}\n")
        
        # Stop all motors at end
        for i in range(self.app.NUM_MOTORS):
            phy_id = self.app.get_physical_id(i)
            if phy_id >= 0:
                self.app.send_serial_command(f"{phy_id},0\n")

        self.finish_playback()

    def move_to_target(self, target_positions, timeout=5.0):
        """
        Blocks until motors reach target positions or timeout.
        Uses Python-side P-control to drive motors.
        """
        start = time.time()
        loop_count = 0
        
        # PID state initialization for move_to_target
        for i in range(self.app.NUM_MOTORS):
            self.prev_errors[i] = 0
            self.integrals[i] = 0
            self.last_pid_times[i] = time.time() # Ensure dt is calculated correctly

        while time.time() - start < timeout and self.is_playing:
            all_reached = True
            loop_count += 1
            
            for i in range(self.app.NUM_MOTORS):
                current = self.app.current_positions[i]
                target = target_positions[i]
                error = target - current
                
                if i == 0 and loop_count % 20 == 0:
                    print(f"DEBUG: M0 Cur={current} Tgt={target} Err={error}")

                if abs(error) > 30: # Tolerance
                    all_reached = False
                    
                    # PID calculation for move_to_target
                    dt = time.time() - self.last_pid_times[i]
                    if dt == 0: dt = 0.000001
                    self.last_pid_times[i] = time.time()

                    # P Term
                    p_term = self.playback_kp * error

                    # I Term
                    self.integrals[i] += error * dt
                    if self.integrals[i] > self.integral_max: self.integrals[i] = self.integral_max
                    elif self.integrals[i] < -self.integral_max: self.integrals[i] = -self.integral_max
                    i_term = self.playback_ki * self.integrals[i]

                    # D Term
                    d_term = self.playback_kd * (error - self.prev_errors[i]) / dt
                    self.prev_errors[i] = error

                    # Total PID output
                    output = p_term + i_term + d_term
                    speed = int(output)
                    
                    # Min speed to overcome friction
                    min_speed = 100
                    if abs(speed) < min_speed:
                        speed = min_speed if speed > 0 else -min_speed
                        
                    # Clamp
                    if speed > self.playback_speed_limit: speed = self.playback_speed_limit
                    if speed < -self.playback_speed_limit: speed = -self.playback_speed_limit
                    
                    phy_id = self.app.get_physical_id(i)
                    if phy_id >= 0:
                        self.app.send_serial_command(f"{phy_id},{speed}\n")
                else:
                    phy_id = self.app.get_physical_id(i)
                    if phy_id >= 0:
                        self.app.send_serial_command(f"{phy_id},0\n")
                    # Keep integral term, allow it to accumulate normally
            
            if all_reached:
                return True
                
            time.sleep(0.05) # 20Hz Control Loop
            
        return False

    def finish_playback(self):
        self.is_playing = False
        self.play_btn.config(state=tk.NORMAL)
        self.record_btn.config(state=tk.NORMAL)
        self.delete_btn.config(state=tk.NORMAL)
        self.stop_play_btn.config(state=tk.DISABLED)
        self.status_label.config(text="Status: Idle", foreground="black")
        print("Playback finished.")