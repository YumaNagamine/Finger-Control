import tkinter as tk
from tkinter import ttk, scrolledtext
import time

class SerialMonitorTab:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.is_paused = False
        self.max_lines = 1000

        self.create_widgets()

    def create_widgets(self):
        top_frame = ttk.Frame(self.parent, padding="5")
        top_frame.pack(fill=tk.X)
        
        self.clear_btn = ttk.Button(top_frame, text="Clear", command=self.clear_text)
        self.clear_btn.pack(side=tk.LEFT, padx=5)
        
        self.pause_btn = ttk.Button(top_frame, text="Pause", command=self.toggle_pause)
        self.pause_btn.pack(side=tk.LEFT, padx=5)
        
        self.status_label = ttk.Label(top_frame, text="Running")
        self.status_label.pack(side=tk.LEFT, padx=10)

        # Scrolled Text Area
        self.text_area = scrolledtext.ScrolledText(self.parent, wrap=tk.WORD, font=("Consolas", 9))
        self.text_area.pack(fill=tk.BOTH, expand=True, padx=5, pady=5)

    def toggle_pause(self):
        self.is_paused = not self.is_paused
        if self.is_paused:
            self.pause_btn.config(text="Resume")
            self.status_label.config(text="Paused")
        else:
            self.pause_btn.config(text="Pause")
            self.status_label.config(text="Running")

    def clear_text(self):
        self.text_area.delete(1.0, tk.END)

    def add_line(self, line):
        if self.is_paused: return
        
        try:
            if not self.text_area.winfo_exists(): return
            
            self.text_area.insert(tk.END, line + "\n")
            
            # Implement line limiting
            num_lines = int(self.text_area.index('end-1c').split('.')[0])
            if num_lines > self.max_lines:
                self.text_area.delete('1.0', '2.0') # Delete the first line

            self.text_area.see(tk.END)
        except Exception:
            pass # Ignore errors during shutdown/closing
