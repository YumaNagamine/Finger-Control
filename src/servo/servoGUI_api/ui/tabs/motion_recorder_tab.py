from __future__ import annotations

import threading
import time
import tkinter as tk
from tkinter import ttk


class MotionRecorderTab:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app

        self.is_recording = False
        self.is_playing = False
        self.record_start_time = 0.0
        self.recorded_frames = []

        self.playback_kp = 8.923
        self.playback_ki = 2.15
        self.playback_kd = 1.537
        self.playback_speed_limit = 1000
        self.integral_max = 5000

        self.prev_errors = {}
        self.integrals = {}
        self.last_pid_times = {}

        for i in range(self.app.NUM_MOTORS):
            self.prev_errors[i] = 0
            self.integrals[i] = 0
            self.last_pid_times[i] = time.time()

        self.create_ui()
        self.update_motion_list()
        self.record_loop()

    def create_ui(self):
        main_layout = ttk.Frame(self.parent, padding="10")
        main_layout.pack(fill=tk.BOTH, expand=True)

        control_frame = ttk.LabelFrame(main_layout, text="Recorder Controls")
        control_frame.pack(side=tk.LEFT, fill=tk.Y, padx=10)

        self.status_label = ttk.Label(control_frame, text="Status: Idle", font=("Helvetica", 12, "bold"))
        self.status_label.pack(pady=20)

        self.record_btn = ttk.Button(control_frame, text="Record Motion", command=self.toggle_recording)
        self.record_btn.pack(pady=10, fill=tk.X)

        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)

        self.play_btn = ttk.Button(control_frame, text="Play Selected", command=self.play_motion)
        self.play_btn.pack(pady=10, fill=tk.X)

        self.stop_play_btn = ttk.Button(
            control_frame,
            text="Stop Playback",
            command=self.stop_playback,
            state=tk.DISABLED,
        )
        self.stop_play_btn.pack(pady=10, fill=tk.X)

        torque_frame = ttk.Frame(control_frame)
        torque_frame.pack(pady=10, fill=tk.X)
        ttk.Button(torque_frame, text="Torque ON (Stop)", command=self.torque_on).pack(side=tk.LEFT, expand=True, fill=tk.X)

        gain_frame = ttk.LabelFrame(control_frame, text="Playback Tuning")
        gain_frame.pack(pady=20, fill=tk.X)

        ttk.Label(gain_frame, text="P-Gain:").pack(side=tk.LEFT)
        self.gain_var = tk.DoubleVar(value=self.playback_kp)
        tk.Entry(gain_frame, textvariable=self.gain_var, width=5).pack(side=tk.LEFT, padx=5)
        ttk.Button(gain_frame, text="Set", command=self.update_gain).pack(side=tk.LEFT)

        ttk.Separator(control_frame, orient=tk.HORIZONTAL).pack(fill=tk.X, pady=10)
        self.delete_btn = ttk.Button(control_frame, text="Delete Selected", command=self.delete_motion)
        self.delete_btn.pack(pady=10, fill=tk.X)

        list_frame = ttk.LabelFrame(main_layout, text="Saved Motions")
        list_frame.pack(side=tk.RIGHT, fill=tk.BOTH, expand=True, padx=10)

        self.motion_listbox = tk.Listbox(list_frame, font=("Courier", 10))
        self.motion_listbox.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5, pady=5)

        scrollbar = ttk.Scrollbar(list_frame, orient=tk.VERTICAL, command=self.motion_listbox.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.motion_listbox.config(yscrollcommand=scrollbar.set)

    def update_gain(self):
        self.playback_kp = self.gain_var.get()

    def torque_on(self):
        self.app.control_service.stop_all()

    def delete_motion(self):
        selection = self.motion_listbox.curselection()
        if not selection:
            return
        motion_name = self.motion_listbox.get(selection[0])
        if self.app.record_service.delete_motion(motion_name):
            self.update_motion_list()

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

    def stop_recording(self):
        self.is_recording = False
        self.record_btn.config(text="Record Motion")
        self.status_label.config(text="Status: Idle", foreground="black")
        self.play_btn.config(state=tk.NORMAL)
        self.delete_btn.config(state=tk.NORMAL)

        self.save_motion()

    def record_loop(self):
        if self.is_recording:
            current_pos = list(self.app.current_positions)
            t = round(time.time() - self.record_start_time, 3)
            self.recorded_frames.append({"t": t, "positions": current_pos})

        self.parent.after(50, self.record_loop)

    def save_motion(self):
        if not self.recorded_frames:
            return
        self.app.record_service.save_motion_auto(self.recorded_frames)
        self.update_motion_list()

    def update_motion_list(self):
        self.motion_listbox.delete(0, tk.END)
        for name in self.app.record_service.list_motions():
            self.motion_listbox.insert(tk.END, name)

    def play_motion(self):
        selection = self.motion_listbox.curselection()
        if not selection:
            return

        motion_name = self.motion_listbox.get(selection[0])
        frames = self.app.record_service.load_motion(motion_name)
        if not frames:
            return

        threading.Thread(target=self.playback_thread, args=(frames,), daemon=True).start()

    def stop_playback(self):
        self.is_playing = False

    def _set_ui_playing(self, playing: bool, status_text: str, color: str):
        def _apply():
            self.play_btn.config(state=tk.DISABLED if playing else tk.NORMAL)
            self.record_btn.config(state=tk.DISABLED if playing else tk.NORMAL)
            self.delete_btn.config(state=tk.DISABLED if playing else tk.NORMAL)
            self.stop_play_btn.config(state=tk.NORMAL if playing else tk.DISABLED)
            self.status_label.config(text=status_text, foreground=color)

        self.parent.after(0, _apply)

    def playback_thread(self, frames):
        self.is_playing = True
        self._set_ui_playing(True, "Status: Moving to Start...", "blue")

        start_pos = frames[0]["positions"]
        if not self.move_to_target(start_pos, timeout=5.0):
            self.finish_playback()
            return

        self._set_ui_playing(True, "Status: Playing...", "green")
        start_time = time.time()

        for i in range(1, len(frames)):
            if not self.is_playing:
                break

            target_frame = frames[i]
            target_t = target_frame["t"]
            target_pos = target_frame["positions"]

            now = time.time() - start_time
            wait = target_t - now
            if wait > 0:
                time.sleep(wait)

            for m_id in range(self.app.NUM_MOTORS):
                current = self.app.current_positions[m_id]
                goal = target_pos[m_id]
                error = goal - current

                dt = time.time() - self.last_pid_times[m_id]
                if dt == 0:
                    dt = 0.000001
                self.last_pid_times[m_id] = time.time()

                p_term = self.playback_kp * error

                self.integrals[m_id] += error * dt
                if self.integrals[m_id] > self.integral_max:
                    self.integrals[m_id] = self.integral_max
                elif self.integrals[m_id] < -self.integral_max:
                    self.integrals[m_id] = -self.integral_max
                i_term = self.playback_ki * self.integrals[m_id]

                d_term = self.playback_kd * (error - self.prev_errors[m_id]) / dt
                self.prev_errors[m_id] = error

                speed = int(p_term + i_term + d_term)
                speed = max(-self.playback_speed_limit, min(self.playback_speed_limit, speed))

                if abs(error) < 10:
                    speed = 0

                self.app.control_service.set_speed(m_id, speed)

        for i in range(self.app.NUM_MOTORS):
            self.app.control_service.set_speed(i, 0)

        self.finish_playback()

    def move_to_target(self, target_positions, timeout=5.0):
        start = time.time()

        for i in range(self.app.NUM_MOTORS):
            self.prev_errors[i] = 0
            self.integrals[i] = 0
            self.last_pid_times[i] = time.time()

        while time.time() - start < timeout and self.is_playing:
            all_reached = True

            for i in range(self.app.NUM_MOTORS):
                current = self.app.current_positions[i]
                target = target_positions[i]
                error = target - current

                if abs(error) > 30:
                    all_reached = False

                    dt = time.time() - self.last_pid_times[i]
                    if dt == 0:
                        dt = 0.000001
                    self.last_pid_times[i] = time.time()

                    p_term = self.playback_kp * error

                    self.integrals[i] += error * dt
                    if self.integrals[i] > self.integral_max:
                        self.integrals[i] = self.integral_max
                    elif self.integrals[i] < -self.integral_max:
                        self.integrals[i] = -self.integral_max
                    i_term = self.playback_ki * self.integrals[i]

                    d_term = self.playback_kd * (error - self.prev_errors[i]) / dt
                    self.prev_errors[i] = error

                    speed = int(p_term + i_term + d_term)
                    min_speed = 100
                    if abs(speed) < min_speed:
                        speed = min_speed if speed > 0 else -min_speed

                    speed = max(-self.playback_speed_limit, min(self.playback_speed_limit, speed))
                    self.app.control_service.set_speed(i, speed)
                else:
                    self.app.control_service.set_speed(i, 0)

            if all_reached:
                return True

            time.sleep(0.05)

        return False

    def finish_playback(self):
        self.is_playing = False
        self._set_ui_playing(False, "Status: Idle", "black")
