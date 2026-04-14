import tkinter as tk
from tkinter import ttk, messagebox
import time
import csv
import threading
import statistics
import math
import os
import numpy as np
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Portable log path (inside this servoGUI package)
LOGS_DIR = os.path.join(os.path.dirname(__file__), "Diagnostic_Logs")

class SystemCheckTab:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.is_running = False
        self.stop_event = threading.Event()
        
        # Configuration
        self.servo_id = tk.IntVar(value=0)
        
        # Results Data
        self.friction_data = {'cw': [], 'ccw': []} 
        self.step_data = {} 
        self.jitter_data = [] 
        self.max_speed_data = {'cmd': [], 'act': [], 'load': []} # Now stores ramp up data
        self.fidelity_data = {'micro_tgt': [], 'micro_act': [], 'macro_tgt': [], 'macro_act': []}
        self.torque_speed_data = {'cmd_spd': [], 'act_spd': [], 'load': []}
        
        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.parent, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Top Control Bar
        ctrl_frame = ttk.LabelFrame(main_frame, text="Comprehensive System Check")
        ctrl_frame.pack(fill=tk.X, pady=5)
        
        ttk.Label(ctrl_frame, text="Target Servo ID:").pack(side=tk.LEFT, padx=5)
        ttk.Combobox(ctrl_frame, textvariable=self.servo_id, 
                     values=[str(i) for i in range(self.app.NUM_MOTORS)], width=3).pack(side=tk.LEFT, padx=5)
        
        self.start_btn = ttk.Button(ctrl_frame, text="START DIAGNOSTIC SUITE", command=self.start_tests)
        self.start_btn.pack(side=tk.LEFT, padx=20, fill=tk.X, expand=True)
        
        self.stop_btn = ttk.Button(ctrl_frame, text="ABORT", command=self.stop_tests, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # Status / Progress
        self.status_var = tk.StringVar(value="Ready.")
        ttk.Label(main_frame, textvariable=self.status_var, font=("Helvetica", 10, "italic")).pack(anchor=tk.W, pady=2)
        
        self.progress = ttk.Progressbar(main_frame, orient=tk.HORIZONTAL, mode='determinate')
        self.progress.pack(fill=tk.X, pady=5)

        # Split View: Graphs (Top) / Log (Bottom)
        paned_win = ttk.PanedWindow(main_frame, orient=tk.VERTICAL)
        paned_win.pack(fill=tk.BOTH, expand=True, pady=5)
        
        # 1. Graphs Frame
        plot_frame = ttk.Frame(paned_win)
        paned_win.add(plot_frame, weight=4)
        
        self.fig = Figure(figsize=(10, 8), dpi=100)
        # 2x3 Grid (Friction, Step/Jitter, Speed, Fidelity, Torque-Speed, Empty)
        self.ax1 = self.fig.add_subplot(231) # Friction
        self.ax2 = self.fig.add_subplot(232) # Step & Jitter
        self.ax3 = self.fig.add_subplot(233) # Max Speed (Continuous)
        self.ax4 = self.fig.add_subplot(234) # Fidelity
        self.ax5 = self.fig.add_subplot(235) # Torque-Speed
        #self.ax6 = self.fig.add_subplot(236) # Reserved for future or bigger jitter view
        self.fig.tight_layout(pad=3.0)
        
        self.canvas = FigureCanvasTkAgg(self.fig, master=plot_frame)
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True)

        # 2. Report/Log Frame
        log_frame = ttk.LabelFrame(paned_win, text="Live Diagnostic Log")
        paned_win.add(log_frame, weight=1)
        
        scrollbar = ttk.Scrollbar(log_frame)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        
        self.report_text = tk.Text(log_frame, font=("Consolas", 9), height=10, yscrollcommand=scrollbar.set)
        self.report_text.pack(fill=tk.BOTH, expand=True)
        scrollbar.config(command=self.report_text.yview)

    def log(self, msg):
        self.status_var.set(msg)
        self.report_text.insert(tk.END, f"[{time.strftime('%H:%M:%S')}] {msg}\n")
        self.report_text.see(tk.END)
        print(f"[SystemCheck] {msg}")

    def start_tests(self):
        if self.is_running: return
        self.is_running = True
        self.stop_event.clear()
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.report_text.delete(1.0, tk.END)
        self.progress['value'] = 0
        
        # Clear plots
        for ax in [self.ax1, self.ax2, self.ax3, self.ax4, self.ax5]: ax.clear()
        self.ax1.set_title("Friction (Load vs Pos)"); self.ax1.set_ylabel("Load")
        self.ax2.set_title("Dynamics (Step & Jitter)"); self.ax2.set_ylabel("Pos")
        self.ax3.set_title("Max Speed Linearity"); self.ax3.set_ylabel("Actual Speed")
        self.ax4.set_title("Fidelity (Micro vs Macro)"); self.ax4.set_ylabel("Pos")
        self.ax5.set_title("Torque-Speed Curve"); self.ax5.set_ylabel("Load")
        self.canvas.draw()

        threading.Thread(target=self.run_sequence, daemon=True).start()

    def stop_tests(self):
        self.stop_event.set()
        self.is_running = False
        self.log("Test Aborted by User. Stopping motor.")
        phy_id = self.app.get_physical_id(self.servo_id.get())
        if phy_id >= 0:
            self.app.send_serial_command(f"{phy_id},0,1\n") # Ensure motor stops
        self.parent.after(0, lambda: self.start_btn.config(state=tk.NORMAL))
        self.parent.after(0, lambda: self.stop_btn.config(state=tk.DISABLED))

    def run_sequence(self):
        try:
            motor_idx = self.servo_id.get()
            phy_id = self.app.get_physical_id(motor_idx)
            
            if phy_id < 0: 
                self.log("Error: Invalid Servo ID.")
                return

            # ================= TEST 1/6: FRICTION CARTOGRAPHY =================
            if self.stop_event.is_set(): return
            self.log("=== Test 1/6: Friction Cartography (Health Check) ===")
            self.progress['value'] = 5
            
            self.log("Moving to Home (0)...")
            self.send_pos(phy_id, 0)
            time.sleep(2.0)
            
            self.log("Sweeping CW (0 -> 4000)...")
            # Initial speed command to ensure wheel mode is set
            self.app.send_serial_command(f"{phy_id},150,1\n") 
            
            self.friction_data['cw'] = []
            start_time = time.time()
            while time.time() - start_time < 10: 
                if self.stop_event.is_set(): break
                pos = self.app.current_positions[motor_idx]
                load = self.app.current_loads[motor_idx]
                self.friction_data['cw'].append((pos, load))
                if pos >= 4000: break
                time.sleep(0.05)
            
            self.app.send_serial_command(f"{phy_id},0\n") # Stop, no ForceInit
            time.sleep(1.0)
            
            self.log("Sweeping CCW (4000 -> 0)...")
            self.app.send_serial_command(f"{phy_id},-150\n") # No ForceInit, assumes already in Wheel Mode
            
            self.friction_data['ccw'] = []
            start_time = time.time()
            while time.time() - start_time < 10:
                if self.stop_event.is_set(): break
                pos = self.app.current_positions[motor_idx]
                load = self.app.current_loads[motor_idx]
                self.friction_data['ccw'].append((pos, load))
                if pos <= 100: break
                time.sleep(0.05)

            self.app.send_serial_command(f"{phy_id},0\n") # Stop
            self.update_plot_1()
            
            # ================= TEST 2/6: SETTLING TIME =================
            if self.stop_event.is_set(): return
            self.log("=== Test 2/6: Step Response Dynamics ===")
            self.progress['value'] = 20
            
            base_pos = 1000
            self.send_pos(phy_id, base_pos)
            time.sleep(1.5)
            
            steps_to_test = [50, 500, 2000]
            self.step_data = {}
            
            for step_sz in steps_to_test:
                if self.stop_event.is_set(): break
                self.log(f"Testing Step Size: {step_sz}...")
                self.send_pos(phy_id, base_pos)
                time.sleep(1.0)
                
                target = base_pos + step_sz
                self.send_pos(phy_id, target)
                t0 = time.time()
                
                trace = []
                while time.time() - t0 < 1.5: 
                    now = time.time() - t0
                    p = self.app.current_positions[motor_idx]
                    trace.append((now, p))
                    time.sleep(0.02) 
                
                self.step_data[str(step_sz)] = trace
                
                positions = [p for t,p in trace]
                final_pos = positions[-1]
                overshoot = max(positions) - target if max(positions) > target else 0
                
                start_val = positions[0] if positions else base_pos
                range_val = target - start_val
                t_10, t_90 = 0, 0
                
                for t,p in trace:
                    if p >= start_val + 0.1*range_val and t_10 == 0: t_10 = t
                    if p >= start_val + 0.9*range_val: t_90 = t
                rise_time = t_90 - t_10

                self.log(f"  -> Step {step_sz}: Rise={rise_time:.3f}s, Overshoot={overshoot}, Final Err={target-final_pos}")

            self.update_plot_2()

            # ================= TEST 3/6: JITTER & HOLDING =================
            if self.stop_event.is_set(): return
            self.log("=== Test 3/6: Jitter & Stability ===")
            self.progress['value'] = 35
            
            self.send_pos(phy_id, 2048)
            time.sleep(1.0)
            self.log("Holding at 2048 for 5 seconds...")
            
            self.jitter_data = []
            start_time = time.time()
            while time.time() - start_time < 5:
                if self.stop_event.is_set(): break
                p = self.app.current_positions[motor_idx]
                self.jitter_data.append(p)
                time.sleep(0.05)
            
            if len(self.jitter_data) > 1:
                sigma = statistics.stdev(self.jitter_data)
                self.log(f"Stability: Sigma={sigma:.3f} (Low is better)")
            
            self.update_plot_2()

            # ================= TEST 4/6: MAX SPEED LINEARITY & SATURATION =================
            if self.stop_event.is_set(): return
            self.log("=== Test 4/6: Max Speed Linearity & Saturation ===")
            self.progress['value'] = 55
            
            self.max_speed_data = {'cmd': [], 'act': [], 'load': []}
            max_commanded_speed = 4000 
            speed_ramp_increment = 50 
            ramp_dwell_time = 0.1 
            
            self.log("Ramping up speed from 0 to max_commanded_speed...")
            # Ensure Wheel Mode is set initially for continuous speed control
            self.app.send_serial_command(f"{phy_id},0,1\n") 
            time.sleep(1.0) # Give time for mode change

            for cmd_spd in range(0, max_commanded_speed + speed_ramp_increment, speed_ramp_increment):
                if self.stop_event.is_set(): break
                
                self.app.send_serial_command(f"{phy_id},{cmd_spd}\n") # Send speed command without force flag
                time.sleep(ramp_dwell_time) 
                
                act_spd = abs(self.app.current_speeds[motor_idx])
                load_val = self.app.current_loads[motor_idx]
                
                self.max_speed_data['cmd'].append(cmd_spd)
                self.max_speed_data['act'].append(act_spd)
                self.max_speed_data['load'].append(load_val)
                
                self.update_plot_3() 
            
            self.app.send_serial_command(f"{phy_id},0\n") # Stop, no ForceInit
            self.log("Max Speed Linearity data collected.")


            # ================= TEST 5/6: FIDELITY (MICRO VS MACRO) =================
            if self.stop_event.is_set(): return
            self.log("=== Test 5/6: Fidelity (Micro vs Macro) ===")
            self.progress['value'] = 75
            
            # Fidelity Params (Hardcoded for compilation, can be user configurable)
            total_delta = 500
            micro_step = 5
            dwell = 0.1
            
            self.send_pos(phy_id, 1000) 
            time.sleep(1.5)
            start_p = self.app.current_positions[motor_idx]
            
            self.fidelity_data['micro_tgt'] = []
            self.fidelity_data['micro_act'] = []
            
            curr = start_p
            steps_count = total_delta // micro_step
            
            for i in range(steps_count):
                if self.stop_event.is_set(): break
                curr += micro_step
                self.send_pos(phy_id, curr)
                time.sleep(dwell)
                act = self.app.current_positions[motor_idx]
                self.fidelity_data['micro_tgt'].append(curr)
                self.fidelity_data['micro_act'].append(act)
                if i % 5 == 0: self.update_plot_4()
                
            end_micro = self.app.current_positions[motor_idx]
            delta_micro = end_micro - start_p
            
            # Macro
            self.send_pos(phy_id, 1000)
            time.sleep(1.5)
            start_macro = self.app.current_positions[motor_idx]
            self.send_pos(phy_id, 1000 + total_delta)
            time.sleep(1.0)
            end_macro = self.app.current_positions[motor_idx]
            delta_macro = end_macro - start_macro
            
            hysteresis = abs(delta_macro - delta_micro)
            efficiency = (delta_micro / total_delta) * 100.0 if total_delta != 0 else 0
            
            self.log(f"Fidelity Result: Efficiency={efficiency:.1f}%, Hysteresis={hysteresis}")
            
            self.update_plot_4()

            # ================= TEST 6/6: TORQUE-SPEED CURVE =================
            if self.stop_event.is_set(): return
            self.log("=== Test 6/6: Torque-Speed Curve ===")
            self.progress['value'] = 90

            self.torque_speed_data = {'cmd_spd': [], 'act_spd': [], 'load': []}
            self.send_pos(phy_id, 2048) # Go to center to avoid hitting limits during spin-up
            time.sleep(1.0)
            
            max_cmd_speed_ts = 4000 
            speed_increment_ts = 50
            ts_dwell_time = 0.1
            
            self.log("Ramping up speed for Torque-Speed curve...")
            # Ensure Wheel Mode is set initially
            self.app.send_serial_command(f"{phy_id},0,1\n") 
            time.sleep(1.0) # Give time for mode change

            for cmd_spd in range(0, max_cmd_speed_ts + speed_increment_ts, speed_increment_ts):
                if self.stop_event.is_set(): break
                self.app.send_serial_command(f"{phy_id},{cmd_spd}\n") # Command Speed without force flag
                time.sleep(ts_dwell_time) # Allow to stabilize
                
                act_spd = abs(self.app.current_speeds[motor_idx])
                load_val = self.app.current_loads[motor_idx]
                
                self.torque_speed_data['cmd_spd'].append(cmd_spd)
                self.torque_speed_data['act_spd'].append(act_spd)
                self.torque_speed_data['load'].append(load_val)
                self.update_plot_5()
            
            self.app.send_serial_command(f"{phy_id},0\n") # Stop
            self.log("Torque-Speed curve data collected.")


            self.progress['value'] = 100
            self.log("ALL TESTS COMPLETED SUCCESSFULLY.")
            self.save_full_report(motor_idx)

        except Exception as e:
            self.log(f"CRITICAL ERROR: {e}")
            import traceback
            traceback.print_exc()
        finally:
            self.is_running = False
            self.app.send_serial_command(f"{phy_id},0,1\n") # Ensure motor stops on exit
            self.parent.after(0, lambda: self.start_btn.config(state=tk.NORMAL))
            self.parent.after(0, lambda: self.stop_btn.config(state=tk.DISABLED))

    def send_pos(self, phy_id, pos):
        p = int(pos)
        self.app.send_serial_command(f"x,{phy_id},{p},0\n")

    def update_plot_1(self): self.parent.after(0, self._draw_plot_1)
    def update_plot_2(self): self.parent.after(0, self._draw_plot_2)
    def update_plot_3(self): self.parent.after(0, self._draw_plot_3)
    def update_plot_4(self): self.parent.after(0, self._draw_plot_4)
    def update_plot_5(self): self.parent.after(0, self._draw_plot_5)


    def _draw_plot_1(self):
        self.ax1.clear()
        self.ax1.set_title("Friction (Load vs Pos)")
        self.ax1.set_xlabel("Position")
        self.ax1.set_ylabel("Load")
        cx = [p for p,l in self.friction_data['cw']]
        cy = [l for p,l in self.friction_data['cw']]
        self.ax1.plot(cx, cy, 'b.', markersize=1, label='CW')
        ccx = [p for p,l in self.friction_data['ccw']]
        ccy = [l for p,l in self.friction_data['ccw']]
        self.ax1.plot(ccx, ccy, 'r.', markersize=1, label='CCW')
        self.ax1.grid(True)
        self.ax1.legend()
        self.canvas.draw()

    def _draw_plot_2(self):
        self.ax2.clear()
        self.ax2.set_title("Dynamics (Step & Jitter)")
        self.ax2.set_ylabel("Pos")
        self.ax2.set_xlabel("Time (s)")
        for lbl, trace in self.step_data.items():
            t = [x[0] for x in trace]
            p = [x[1] for x in trace]
            self.ax2.plot(t, p, label=f'Step {lbl}')
        if self.jitter_data:
            t_jit = [i*0.05 for i in range(len(self.jitter_data))]
            self.ax2.plot(t_jit, self.jitter_data, 'k-', alpha=0.5, label='Jitter Trace')
        self.ax2.legend(fontsize='small')
        self.ax2.grid(True)
        self.canvas.draw()

    def _draw_plot_3(self):
        self.ax3.clear()
        self.ax3.set_title("Max Speed Linearity")
        self.ax3.set_ylabel("Actual Speed")
        self.ax3.set_xlabel("Commanded Speed")
        max_cmd_val = max(self.max_speed_data['cmd']) if self.max_speed_data['cmd'] else 1
        
        self.ax3.plot(self.max_speed_data['cmd'], self.max_speed_data['act'], 'b-o', markersize=3, label='Actual Speed')
        self.ax3.plot([0, max_cmd_val], [0, max_cmd_val], 'g--', label='Ideal 1:1')
        self.ax3.grid(True)
        self.ax3.legend(fontsize='small')
        self.canvas.draw()

    def _draw_plot_4(self):
        self.ax4.clear()
        self.ax4.set_title("Fidelity (Micro Step)")
        self.ax4.set_ylabel("Position")
        self.ax4.set_xlabel("Micro-Step Index")
        if self.fidelity_data['micro_tgt']:
            self.ax4.plot(range(len(self.fidelity_data['micro_tgt'])), self.fidelity_data['micro_tgt'], 'g--', label='Target Pos')
        if self.fidelity_data['micro_act']:
            self.ax4.plot(range(len(self.fidelity_data['micro_act'])), self.fidelity_data['micro_act'], 'b-', label='Actual Pos')
        self.ax4.legend(fontsize='small')
        self.ax4.grid(True)
        self.canvas.draw()

    def _draw_plot_5(self):
        self.ax5.clear()
        self.ax5.set_title("Torque-Speed Curve")
        self.ax5.set_xlabel("Measured Speed (Abs)")
        self.ax5.set_ylabel("Measured Load")
        self.ax5.plot(self.torque_speed_data['act_spd'], self.torque_speed_data['load'], 'ro-')
        self.ax5.grid(True)
        self.canvas.draw()

    def save_full_report(self, motor_idx):
        # Create directory if not exists
        if not os.path.exists(LOGS_DIR):
            os.makedirs(LOGS_DIR)
            
        filename = os.path.join(LOGS_DIR, f"SystemCheck_Full_Motor{motor_idx}_{int(time.time())}.csv")
        
        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Test", "Metric", "X_Value", "Y_Value", "Z_Value"]) # Added Z_Value for some tests
                
                # Friction
                for p, l in self.friction_data['cw']: writer.writerow(["Friction_CW", "Pos_Load", p, l, ""])
                for p, l in self.friction_data['ccw']: writer.writerow(["Friction_CCW", "Pos_Load", p, l, ""])
                
                # Step
                for step_sz, trace in self.step_data.items():
                    for t, p in trace: writer.writerow([f"Step_{step_sz}", "Time_Pos", t, p, ""])
                
                # Jitter
                for p in self.jitter_data: writer.writerow(["Jitter", "Pos", p, "", ""])
                
                # Max Speed (Cmd_Act_Load)
                for c, a, l in zip(self.max_speed_data['cmd'], self.max_speed_data['act'], self.max_speed_data['load']):
                    writer.writerow(["MaxSpeed", "Cmd_Act_Load", c, a, l])
                    
                # Fidelity
                for i in range(len(self.fidelity_data['micro_tgt'])):
                    writer.writerow(["Fidelity_Micro", "Tgt_Act", self.fidelity_data['micro_tgt'][i], self.fidelity_data['micro_act'][i], ""])
                
                # Torque-Speed (Cmd_Act_Load)
                for cs, asd, l in zip(self.torque_speed_data['cmd_spd'], self.torque_speed_data['act_spd'], self.torque_speed_data['load']):
                    writer.writerow(["TorqueSpeed", "Cmd_Act_Load", cs, asd, l])
                    
            self.log(f"Full data successfully saved to: {filename}")
        except Exception as e:
            self.log(f"Save Error: {e}")
