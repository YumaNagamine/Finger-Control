import tkinter as tk
from tkinter import ttk, filedialog
import time
import csv
import threading
import os
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

class FidelityTestTab:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.is_running = False
        
        # Parameters
        self.servo_id = tk.IntVar(value=0)
        self.total_target_delta = tk.IntVar(value=500)
        self.micro_step_size = tk.IntVar(value=5)
        self.dwell_time = tk.DoubleVar(value=0.1) # Seconds
        
        # Data Storage
        self.log_data = [] # List of [Timestamp, Phase, Step_Idx, Commanded_Pos, Actual_Pos]
        self.results_text = tk.StringVar(value="Ready to test.")
        
        self.create_widgets()

    def create_widgets(self):
        main_frame = ttk.Frame(self.parent, padding="10")
        main_frame.pack(fill=tk.BOTH, expand=True)

        # Controls
        ctrl_frame = ttk.LabelFrame(main_frame, text="Test Configuration")
        ctrl_frame.pack(fill=tk.X, pady=5)
        
        # Row 1: Inputs
        f1 = ttk.Frame(ctrl_frame)
        f1.pack(fill=tk.X, pady=2)
        
        ttk.Label(f1, text="Servo ID:").pack(side=tk.LEFT, padx=5)
        ttk.Combobox(f1, textvariable=self.servo_id, 
                     values=[str(i) for i in range(self.app.NUM_MOTORS)], width=3).pack(side=tk.LEFT)
        
        ttk.Label(f1, text="Total Delta:").pack(side=tk.LEFT, padx=10)
        ttk.Entry(f1, textvariable=self.total_target_delta, width=6).pack(side=tk.LEFT)

        ttk.Label(f1, text="Micro Step:").pack(side=tk.LEFT, padx=10)
        ttk.Entry(f1, textvariable=self.micro_step_size, width=5).pack(side=tk.LEFT)

        ttk.Label(f1, text="Dwell (s):").pack(side=tk.LEFT, padx=10)
        ttk.Entry(f1, textvariable=self.dwell_time, width=5).pack(side=tk.LEFT)

        # Row 2: Buttons
        f2 = ttk.Frame(ctrl_frame)
        f2.pack(fill=tk.X, pady=5)
        
        self.start_btn = ttk.Button(f2, text="RUN TEST", command=self.start_test)
        self.start_btn.pack(side=tk.LEFT, padx=5, fill=tk.X, expand=True)
        
        self.stop_btn = ttk.Button(f2, text="STOP", command=self.stop_test, state=tk.DISABLED)
        self.stop_btn.pack(side=tk.LEFT, padx=5)

        # Status / Results
        res_frame = ttk.LabelFrame(main_frame, text="Live Results")
        res_frame.pack(fill=tk.X, pady=5)
        ttk.Label(res_frame, textvariable=self.results_text, font=("Consolas", 10), justify=tk.LEFT).pack(anchor=tk.W, padx=5, pady=5)

        # Plot
        self.fig = Figure(figsize=(6, 4), dpi=100)
        self.ax = self.fig.add_subplot(111)
        self.canvas = FigureCanvasTkAgg(self.fig, master=main_frame)
        self.canvas.draw()
        self.canvas.get_tk_widget().pack(fill=tk.BOTH, expand=True, pady=5)

    def start_test(self):
        if self.is_running: return
        self.is_running = True
        self.start_btn.config(state=tk.DISABLED)
        self.stop_btn.config(state=tk.NORMAL)
        self.log_data = []
        self.results_text.set("Initializing...")
        
        # Clear Plot
        self.ax.clear()
        self.ax.set_title("Micro vs Macro Step Performance")
        self.ax.set_xlabel("Sample Index")
        self.ax.set_ylabel("Position (Steps)")
        self.ax.grid(True)
        self.canvas.draw()

        threading.Thread(target=self.run_test_sequence, daemon=True).start()

    def stop_test(self):
        self.is_running = False
        self.results_text.set("Test Aborted.")
        self.start_btn.config(state=tk.NORMAL)
        self.stop_btn.config(state=tk.DISABLED)

    def run_test_sequence(self):
        try:
            motor_idx = self.servo_id.get()
            phy_id = self.app.get_physical_id(motor_idx)
            total = self.total_target_delta.get()
            step = self.micro_step_size.get()
            dwell = self.dwell_time.get()

            if phy_id < 0: 
                self.log("Invalid Motor ID")
                return

            # --- PHASE 1: MICRO STEPS ---
            self.log("Moving to Start (Zero)...")
            self.send_pos(phy_id, 0)
            time.sleep(1.5) # Settle

            start_pos_micro = self.read_pos(motor_idx)
            self.log(f"Start Pos Micro: {start_pos_micro}")
            
            current_target = start_pos_micro
            steps_count = abs(total) // step
            
            x_vals = []
            y_tgt = []
            y_act = []

            self.log(f"Starting Micro-Steps ({steps_count} steps of {step})...")
            
            for i in range(steps_count):
                if not self.is_running: break
                
                current_target += step
                self.send_pos(phy_id, current_target)
                time.sleep(dwell) # Dwell to force friction break
                
                act = self.read_pos(motor_idx)
                
                # Log
                self.log_data.append([time.time(), "MICRO", i, current_target, act])
                x_vals.append(i)
                y_tgt.append(current_target)
                y_act.append(act)
                
                if i % 5 == 0: # Update plot periodically
                    self.parent.after(0, lambda x=list(x_vals), yt=list(y_tgt), ya=list(y_act): self.update_plot(x, yt, ya))

            time.sleep(0.5)
            end_pos_micro = self.read_pos(motor_idx)
            delta_micro = end_pos_micro - start_pos_micro
            self.log(f"End Pos Micro: {end_pos_micro}. Delta: {delta_micro}")

            # --- PHASE 2: MACRO STEP ---
            if self.is_running:
                self.log("Resetting to Start for Macro Test...")
                self.send_pos(phy_id, 0) # Back to zero
                time.sleep(2.0) # Allow full return and settle
                
                start_pos_macro = self.read_pos(motor_idx)
                self.log(f"Start Pos Macro: {start_pos_macro}")
                
                target_macro = start_pos_macro + total
                self.log(f"Executing Macro Step (+{total})...")
                self.send_pos(phy_id, target_macro)
                time.sleep(1.5) # Allow full move
                
                end_pos_macro = self.read_pos(motor_idx)
                delta_macro = end_pos_macro - start_pos_macro
                
                # Log Macro Point
                self.log_data.append([time.time(), "MACRO", 0, target_macro, end_pos_macro])
                
                # --- ANALYSIS ---
                hysteresis = abs(delta_macro - delta_micro)
                efficiency = (delta_micro / total) * 100.0 if total != 0 else 0
                
                report = (
                    f"""--- TEST COMPLETE ---
Target Delta: {total}
Micro Delta : {delta_micro} (Efficiency: {efficiency:.1f}%)
Macro Delta : {delta_macro}
Hysteresis (Slip/Slop): {hysteresis} steps
"""
                )
                self.log(report)
                self.save_csv()

        except Exception as e:
            self.log(f"Error: {e}")
        finally:
            self.is_running = False
            self.parent.after(0, lambda: self.start_btn.config(state=tk.NORMAL))
            self.parent.after(0, lambda: self.stop_btn.config(state=tk.DISABLED))

    def send_pos(self, phy_id, pos):
        # Uses Software PID 'x' command (unbounded)
        self.app.send_serial_command(f"x,{phy_id},{pos},0\n")

    def read_pos(self, motor_idx):
        return self.app.current_positions[motor_idx]

    def log(self, msg):
        print(f"[Fidelity] {msg}")
        self.results_text.set(msg) # Show last message
        # In a real implementation, maybe append to a text box, but StringVar is simple

    def update_plot(self, x, y_tgt, y_act):
        self.ax.clear()
        self.ax.set_title("Micro-Step Response")
        self.ax.plot(x, y_tgt, 'g--', label='Target')
        self.ax.plot(x, y_act, 'b-', label='Actual')
        self.ax.legend()
        self.canvas.draw()

    def save_csv(self):
        if not self.log_data: return
        
        filename = f"fidelity_test_motor{self.servo_id.get()}_{int(time.time())}.csv"
        try:
            with open(filename, 'w', newline='') as f:
                writer = csv.writer(f)
                writer.writerow(["Timestamp", "Phase", "Step_Idx", "Commanded", "Actual"])
                writer.writerows(self.log_data)
            self.log(f"Data saved to {filename}")
        except Exception as e:
            self.log(f"Save failed: {e}")

