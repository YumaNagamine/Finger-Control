from __future__ import annotations

import tkinter as tk
from tkinter import ttk


class IDManagerTab:
    def __init__(self, parent, app):
        self.parent = parent
        self.app = app
        self.found_ids = []

        self.create_widgets()

    def create_widgets(self):
        top_frame = ttk.Frame(self.parent, padding="10")
        top_frame.pack(fill=tk.X)

        search_frame = ttk.LabelFrame(top_frame, text="ID Search Scanner", padding="10")
        search_frame.pack(fill=tk.X, pady=5)

        ttk.Label(search_frame, text="Start ID:").pack(side=tk.LEFT, padx=5)
        self.start_id_entry = ttk.Entry(search_frame, width=5)
        self.start_id_entry.insert(0, "0")
        self.start_id_entry.pack(side=tk.LEFT, padx=5)

        ttk.Label(search_frame, text="End ID:").pack(side=tk.LEFT, padx=5)
        self.end_id_entry = ttk.Entry(search_frame, width=5)
        self.end_id_entry.insert(0, "10")
        self.end_id_entry.pack(side=tk.LEFT, padx=5)

        self.scan_btn = ttk.Button(search_frame, text="SCAN", command=self.start_scan)
        self.scan_btn.pack(side=tk.LEFT, padx=15)

        self.scan_status = ttk.Label(search_frame, text="Ready", foreground="blue")
        self.scan_status.pack(side=tk.LEFT, padx=10)

        result_frame = ttk.LabelFrame(top_frame, text="Found IDs", padding="10")
        result_frame.pack(fill=tk.BOTH, expand=True, pady=5)

        self.result_list = tk.Listbox(result_frame, height=6)
        self.result_list.pack(side=tk.LEFT, fill=tk.BOTH, expand=True, padx=5)
        self.result_list.bind("<<ListboxSelect>>", self.on_select_id)

        scrollbar = ttk.Scrollbar(result_frame, orient=tk.VERTICAL, command=self.result_list.yview)
        scrollbar.pack(side=tk.RIGHT, fill=tk.Y)
        self.result_list.config(yscrollcommand=scrollbar.set)

        alloc_frame = ttk.LabelFrame(top_frame, text="ID Allocation (Change ID)", padding="10")
        alloc_frame.pack(fill=tk.X, pady=5)

        ttk.Label(alloc_frame, text="Selected Old ID:").pack(side=tk.LEFT, padx=5)
        self.selected_old_id = ttk.Label(alloc_frame, text="-", font=("Helvetica", 10, "bold"))
        self.selected_old_id.pack(side=tk.LEFT, padx=5)

        ttk.Label(alloc_frame, text="New ID:").pack(side=tk.LEFT, padx=10)
        self.new_id_entry = ttk.Entry(alloc_frame, width=5)
        self.new_id_entry.pack(side=tk.LEFT, padx=5)

        self.set_btn = ttk.Button(alloc_frame, text="SET NEW ID", command=self.set_new_id, state=tk.DISABLED)
        self.set_btn.pack(side=tk.LEFT, padx=15)

        seq_frame = ttk.LabelFrame(top_frame, text="Sequential Daisy-Chain Allocator", padding="10")
        seq_frame.pack(fill=tk.X, pady=5)

        ttk.Label(seq_frame, text="Default ID (New Motor):").pack(side=tk.LEFT, padx=5)
        self.seq_default_id = ttk.Entry(seq_frame, width=5)
        self.seq_default_id.insert(0, "1")
        self.seq_default_id.pack(side=tk.LEFT, padx=5)

        ttk.Label(seq_frame, text="Assign Next ID:").pack(side=tk.LEFT, padx=5)
        self.seq_next_id = ttk.Entry(seq_frame, width=5)
        self.seq_next_id.insert(0, "0")
        self.seq_next_id.pack(side=tk.LEFT, padx=5)

        self.seq_assign_btn = ttk.Button(seq_frame, text="Assign & Increment", command=self.seq_assign_increment)
        self.seq_assign_btn.pack(side=tk.LEFT, padx=15)

    def seq_assign_increment(self):
        try:
            default_id = int(self.seq_default_id.get())
            next_id = int(self.seq_next_id.get())

            if default_id == next_id:
                self.scan_status.config(text="Error: Default ID == Next ID")
                return

            self.app.id_service.change_id(default_id, next_id)
            self.scan_status.config(text=f"Assigning {default_id} -> {next_id}...")

            self.seq_next_id.delete(0, tk.END)
            self.seq_next_id.insert(0, str(next_id + 1))
        except ValueError:
            self.scan_status.config(text="Invalid Numbers")

    def start_scan(self):
        try:
            s = int(self.start_id_entry.get())
            e = int(self.end_id_entry.get())
            if s < 0 or e > 253 or s > e:
                self.scan_status.config(text="Invalid Range")
                return

            self.scan_status.config(text="Scanning...")
            self.result_list.delete(0, tk.END)
            self.found_ids = []
            self.app.id_service.scan(s, e)
        except ValueError:
            self.scan_status.config(text="Error")

    def process_scan_result(self, line):
        if "SCAN_END" in line:
            self.scan_status.config(text="Scan Complete")
        elif "FOUND_ID:" in line:
            try:
                found_id = line.split(":")[1].strip()
                if found_id not in self.found_ids:
                    self.result_list.insert(tk.END, f"ID: {found_id}")
                    self.found_ids.append(found_id)
            except Exception:
                pass
        elif "VERIFY_SUCCESS" in line:
            self.scan_status.config(text="SUCCESS: ID Changed & Verified!")
        elif "VERIFY_FAIL" in line:
            self.scan_status.config(text="FAILURE: Servo Rejected Change.")
        elif "NVS_UPDATED" in line:
            self.scan_status.config(text="Controller Memory Updated.")
        elif "EXECUTING" in line:
            self.scan_status.config(text="Writing to Servo Memory...")

    def on_select_id(self, event):
        del event
        selection = self.result_list.curselection()
        if selection:
            text = self.result_list.get(selection[0])
            old_id = text.split(":")[1].strip()
            self.selected_old_id.config(text=old_id)
            self.set_btn.config(state=tk.NORMAL)

    def set_new_id(self):
        old_id = self.selected_old_id.cget("text")
        new_id = self.new_id_entry.get()

        if not old_id.isdigit() or not new_id.isdigit():
            self.scan_status.config(text="Invalid IDs")
            return

        self.app.id_service.change_id(int(old_id), int(new_id))
        self.app.id_service.reset_ids()
        self.scan_status.config(text="Command Sent. CHECK LOGS.")
