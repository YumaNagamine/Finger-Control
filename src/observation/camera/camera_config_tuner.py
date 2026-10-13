from __future__ import annotations

import argparse
import json
import multiprocessing as mp
import queue
import sys
import time
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import cv2
import numpy as np

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from observation.camera.camera_param_resolver import resolve_param_path
from observation.camera.camera_utils import (
    apply_camera_settings,
    resolve_backend,
    setup_undistortion_from_config,
    undistort_frame,
)
from utils.config_loader import load_config


DEFAULT_CAMERA_CONFIG = resolve_param_path("camera_config.json")
BACKEND_OPTIONS = ("CAP_DSHOW", "CAP_ANY", "")
PREVIEW_MAX_SIZE = (960, 600)
WB_MIN_K, WB_MAX_K, WB_STEP_K = 2800, 6500, 50
SLIDER_FIELDS: tuple[dict, ...] = (
    {"key": "target_fps", "label": "Target FPS", "min": 1, "max": 240, "step": 1, "default": 90.0},
    {"key": "buffersize", "label": "Buffer Size", "min": 0, "max": 10, "step": 1, "default": 0.0},
    {"key": "auto_exposure", "label": "Auto Exposure", "min": 0, "max": 3, "step": 0.1, "default": 1.0},
    {"key": "gain", "label": "Gain", "min": 0, "max": 255, "step": 0.1, "default": 0.0},
    {"key": "exposure", "label": "Exposure", "min": -20, "max": 20, "step": 0.1, "default": -11.0},
)


def _replace_latest(target_queue, item) -> None:
    try:
        while True:
            target_queue.get_nowait()
    except queue.Empty:
        pass
    try:
        target_queue.put_nowait(item)
    except queue.Full:
        pass


def _resize_preview(frame):
    height, width = frame.shape[:2]
    scale = min(PREVIEW_MAX_SIZE[0] / width, PREVIEW_MAX_SIZE[1] / height, 1.0)
    if scale >= 1.0:
        return frame
    return cv2.resize(
        frame,
        (max(1, int(width * scale)), max(1, int(height * scale))),
        interpolation=cv2.INTER_AREA,
    )


def _readback(cap: cv2.VideoCapture) -> dict[str, float]:
    return {
        "auto_wb": float(cap.get(cv2.CAP_PROP_AUTO_WB)),
        "temperature": float(cap.get(cv2.CAP_PROP_WB_TEMPERATURE)),
        "gain": float(cap.get(cv2.CAP_PROP_GAIN)),
        "exposure": float(cap.get(cv2.CAP_PROP_EXPOSURE)),
    }


def _capture_worker(config: dict, command_queue, frame_queue, status_queue, stop_event) -> None:
    cap = None
    frame_queue.cancel_join_thread()
    status_queue.cancel_join_thread()
    try:
        backend = resolve_backend(str(config.get("backend", "CAP_DSHOW")))
        index = int(config.get("index", 0))
        cap = cv2.VideoCapture(index, backend) if backend is not None else cv2.VideoCapture(index)
        if not cap.isOpened():
            raise RuntimeError(f"Camera not available: index={index}")
        apply_camera_settings(cap, config)
        calibration = setup_undistortion_from_config(
            config, log_fn=lambda _message: None, log_prefix="[camera-tuner]"
        )
        _replace_latest(status_queue, {"text": f"Opened camera index={index}", "readback": _readback(cap)})

        while not stop_event.is_set():
            latest = None
            try:
                while True:
                    latest = command_queue.get_nowait()
            except queue.Empty:
                pass
            if latest is not None:
                config = dict(latest)
                apply_camera_settings(cap, config)
                _replace_latest(status_queue, {"text": "Applied settings", "readback": _readback(cap)})

            ok, frame = cap.read()
            if not ok or frame is None:
                time.sleep(0.01)
                continue
            frame = _resize_preview(undistort_frame(frame, calibration))
            ok, encoded = cv2.imencode(".jpg", frame, [cv2.IMWRITE_JPEG_QUALITY, 85])
            if ok:
                _replace_latest(frame_queue, encoded.tobytes())
    except Exception as exc:
        _replace_latest(status_queue, {"text": f"Camera worker failed: {type(exc).__name__}: {exc}"})
    finally:
        if cap is not None:
            cap.release()


def _photo_image(frame):
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    height, width = rgb.shape[:2]
    return tk.PhotoImage(
        data=f"P6\n{width} {height}\n255\n".encode("ascii") + rgb.tobytes(),
        format="PPM",
    )


class CameraConfigTunerApp:
    def __init__(self, root: tk.Tk, config_path: Path):
        self.root = root
        self.config_path = config_path
        self.config = load_config(config_path)
        self.context = mp.get_context("spawn")
        self.process = self.command_queue = self.frame_queue = self.status_queue = self.stop_event = None
        self.after_id = None
        self.closing = False
        self.photo = None

        self.index_var = tk.IntVar(value=int(self.config.get("index", 0)))
        self.width_var = tk.StringVar(value=str(self.config.get("width", 1600)))
        self.height_var = tk.StringVar(value=str(self.config.get("height", 1200)))
        self.backend_var = tk.StringVar(value=str(self.config.get("backend", "CAP_DSHOW")))
        self.capture_fourcc_var = tk.StringVar(value=str(self.config.get("capture_fourcc", "MJPG")))
        self.writer_fourcc_var = tk.StringVar(value=str(self.config.get("writer_fourcc", "mp4v")))
        self.auto_wb_var = tk.BooleanVar(value=bool(self.config.get("auto_white_balance", True)))
        self.temperature_var = tk.DoubleVar(value=float(self.config.get("white_balance_temperature", 4500)))
        self.status_var = tk.StringVar(value=f"Loaded: {config_path}")
        self.readback_var = tk.StringVar(value="Camera readback: waiting...")
        self.slider_vars: dict[str, tk.DoubleVar] = {}
        self.slider_labels: dict[str, tk.Label] = {}

        root.title("Camera Config Tuner")
        root.geometry("1000x900")
        root.protocol("WM_DELETE_WINDOW", self.close)
        self._build_ui()
        self._update_wb_state()
        self._start_worker()
        self._poll()

    def _build_ui(self) -> None:
        self.preview = tk.Label(self.root, text="Starting camera...")
        self.preview.pack(padx=8, pady=8)
        body = tk.Frame(self.root)
        body.pack(fill="x", padx=8)

        row = tk.Frame(body); row.pack(fill="x")
        tk.Label(row, text="Camera Index", width=20, anchor="w").pack(side="left")
        tk.Scale(row, from_=0, to=10, orient="horizontal", variable=self.index_var, length=180).pack(side="left")
        tk.Label(row, text="Backend", width=10).pack(side="left")
        tk.OptionMenu(row, self.backend_var, *BACKEND_OPTIONS).pack(side="left")

        row = tk.Frame(body); row.pack(fill="x")
        tk.Label(row, text="Resolution", width=20, anchor="w").pack(side="left")
        tk.Entry(row, textvariable=self.width_var, width=8).pack(side="left")
        tk.Label(row, text=" x ").pack(side="left")
        tk.Entry(row, textvariable=self.height_var, width=8).pack(side="left")

        row = tk.Frame(body); row.pack(fill="x")
        tk.Label(row, text="Capture FOURCC", width=20, anchor="w").pack(side="left")
        tk.Entry(row, textvariable=self.capture_fourcc_var, width=8).pack(side="left")
        tk.Label(row, text="Writer FOURCC", width=16).pack(side="left")
        tk.Entry(row, textvariable=self.writer_fourcc_var, width=8).pack(side="left")

        for field in SLIDER_FIELDS:
            row = tk.Frame(body); row.pack(fill="x", pady=1)
            tk.Label(row, text=field["label"], width=20, anchor="w").pack(side="left")
            var = tk.DoubleVar(value=float(self.config.get(field["key"], field["default"])))
            self.slider_vars[field["key"]] = var
            tk.Scale(
                row, from_=field["min"], to=field["max"], resolution=field["step"],
                orient="horizontal", variable=var, showvalue=False,
                command=lambda _value, key=field["key"]: self._update_slider_label(key),
            ).pack(side="left", fill="x", expand=True)
            label = tk.Label(row, width=10, anchor="e"); label.pack(side="left")
            self.slider_labels[field["key"]] = label
            self._update_slider_label(field["key"])

        row = tk.Frame(body); row.pack(fill="x", pady=(5, 1))
        tk.Label(row, text="Auto White Balance", width=20, anchor="w").pack(side="left")
        tk.Checkbutton(row, text="On", variable=self.auto_wb_var, command=self._wb_changed).pack(side="left")

        row = tk.Frame(body); row.pack(fill="x")
        tk.Label(row, text="WB Temperature [K]", width=20, anchor="w").pack(side="left")
        self.temperature_scale = tk.Scale(
            row, from_=WB_MIN_K, to=WB_MAX_K, resolution=WB_STEP_K, orient="horizontal",
            variable=self.temperature_var, showvalue=False, command=self._temperature_changed,
        )
        self.temperature_scale.pack(side="left", fill="x", expand=True)
        self.temperature_label = tk.Label(row, width=10, anchor="e"); self.temperature_label.pack(side="left")
        self._update_temperature_label()

        buttons = tk.Frame(self.root); buttons.pack(fill="x", padx=8, pady=6)
        tk.Button(buttons, text="Reopen Camera", command=self.reopen, width=16).pack(side="left")
        tk.Button(buttons, text="Apply", command=self.apply, width=12).pack(side="left", padx=8)
        tk.Button(buttons, text="Save", command=self.save, width=12).pack(side="left")
        tk.Label(self.root, textvariable=self.readback_var, anchor="w").pack(fill="x", padx=8)
        tk.Label(self.root, textvariable=self.status_var, anchor="w").pack(fill="x", padx=8)

    def _update_slider_label(self, key: str) -> None:
        self.slider_labels[key].configure(text=f"{self.slider_vars[key].get():.2f}")

    def _update_temperature_label(self) -> None:
        self.temperature_label.configure(text=f"{self.temperature_var.get():.0f} K")

    def _update_wb_state(self) -> None:
        self.temperature_scale.configure(state="disabled" if self.auto_wb_var.get() else "normal")

    def _wb_changed(self) -> None:
        self._update_wb_state()
        self.apply()

    def _temperature_changed(self, _value: str) -> None:
        self._update_temperature_label()
        if not self.auto_wb_var.get():
            self.apply()

    def _sync_config(self) -> bool:
        try:
            width, height = int(self.width_var.get()), int(self.height_var.get())
        except ValueError:
            self.status_var.set("Resolution must be integer values.")
            return False
        if width <= 0 or height <= 0:
            self.status_var.set("Resolution must be greater than zero.")
            return False
        self.config.update({
            "index": int(self.index_var.get()), "backend": self.backend_var.get().strip() or "CAP_ANY",
            "width": width, "height": height, "capture_fourcc": self.capture_fourcc_var.get().strip(),
            "writer_fourcc": self.writer_fourcc_var.get().strip(),
            "auto_white_balance": bool(self.auto_wb_var.get()),
            "white_balance_temperature": float(round(self.temperature_var.get())),
        })
        for field in SLIDER_FIELDS:
            value = self.slider_vars[field["key"]].get()
            self.config[field["key"]] = int(round(value)) if field["key"] == "buffersize" else round(float(value), 3)
        return True

    def _start_worker(self) -> None:
        if not self._sync_config(): return
        self.command_queue = self.context.Queue(maxsize=1)
        self.frame_queue = self.context.Queue(maxsize=1)
        self.status_queue = self.context.Queue(maxsize=1)
        self.stop_event = self.context.Event()
        self.process = self.context.Process(
            target=_capture_worker,
            args=(dict(self.config), self.command_queue, self.frame_queue, self.status_queue, self.stop_event),
            name="camera-config-tuner-capture",
        )
        self.process.start()

    def _stop_worker(self) -> None:
        if self.stop_event is not None: self.stop_event.set()
        if self.process is not None:
            self.process.join(timeout=3)
            if self.process.is_alive(): self.process.terminate(); self.process.join()
        for item in (self.command_queue, self.frame_queue, self.status_queue):
            if item is not None:
                item.cancel_join_thread(); item.close()
        self.process = None

    def _poll(self) -> None:
        if self.closing: return
        latest = None
        if self.frame_queue is not None:
            try:
                while True: latest = self.frame_queue.get_nowait()
            except queue.Empty: pass
        if latest is not None:
            frame = cv2.imdecode(np.frombuffer(latest, dtype=np.uint8), cv2.IMREAD_COLOR)
            if frame is not None:
                self.photo = _photo_image(frame); self.preview.configure(image=self.photo, text="")

        status = None
        if self.status_queue is not None:
            try:
                while True: status = self.status_queue.get_nowait()
            except queue.Empty: pass
        if status is not None:
            self.status_var.set(status["text"])
            values = status.get("readback")
            if values:
                self.readback_var.set(
                    f"Camera readback: Auto WB={values['auto_wb']:.1f}, "
                    f"temperature={values['temperature']:.0f} K, gain={values['gain']:.1f}, "
                    f"exposure={values['exposure']:.1f}"
                )
        self.after_id = self.root.after(20, self._poll)

    def reopen(self) -> None:
        self._stop_worker(); self._start_worker()

    def apply(self) -> None:
        if self._sync_config() and self.command_queue is not None:
            _replace_latest(self.command_queue, dict(self.config))

    def save(self) -> None:
        if not self._sync_config(): return
        with self.config_path.open("w", encoding="utf-8") as file:
            json.dump(self.config, file, indent=2); file.write("\n")
        self.status_var.set(f"Saved: {self.config_path}")

    def close(self) -> None:
        self.closing = True
        if self.after_id is not None:
            try: self.root.after_cancel(self.after_id)
            except tk.TclError: pass
        self._stop_worker(); self.root.destroy()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune a camera config with a multiprocess live preview.")
    parser.add_argument("--config", default=str(DEFAULT_CAMERA_CONFIG))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    root = tk.Tk()
    CameraConfigTunerApp(root, Path(args.config).resolve())
    root.mainloop()


if __name__ == "__main__":
    mp.freeze_support()
    try:
        main()
    except Exception as exc:
        messagebox.showerror("Camera Config Tuner Error", str(exc))
        raise
