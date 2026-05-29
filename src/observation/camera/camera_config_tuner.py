from __future__ import annotations

import argparse
import json
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import cv2

# Ensure repository src root is importable when running as a script.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from observation.camera.camera_utils import (
    apply_camera_settings,
    resolve_backend,
    setup_undistortion_from_config,
    undistort_frame,
)
from observation.camera.camera_param_resolver import resolve_param_path
from utils.config_loader import load_config


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CAMERA_CONFIG = resolve_param_path("camera_config.json")
WINDOW_TITLE = "Camera Config Tuner"
WINDOW_GEOMETRY = "1000x760"
BACKEND_OPTIONS = ("CAP_DSHOW", "CAP_ANY", "")
PREVIEW_MAX_WIDTH = 960
PREVIEW_MAX_HEIGHT = 800

SLIDER_FIELDS: tuple[dict, ...] = (
    {"key": "target_fps", "label": "Target FPS", "min": 1, "max": 240, "resolution": 1, "default": 90.0, "cast": float},
    {"key": "buffersize", "label": "Buffer Size", "min": 0, "max": 10, "resolution": 1, "default": 0, "cast": int},
    {
        "key": "auto_exposure",
        "label": "Auto Exposure",
        "min": 0.0,
        "max": 3.0,
        "resolution": 0.1,
        "default": 1.0,
        "cast": float,
    },
    {"key": "gain", "label": "Gain", "min": 0.0, "max": 255.0, "resolution": 0.1, "default": 0.0, "cast": float},
    {"key": "exposure", "label": "Exposure", "min": -20.0, "max": 20.0, "resolution": 0.1, "default": -11.0, "cast": float},
)


def _to_photo_image_bgr(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    height, width, _ = frame_rgb.shape
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    data = header + frame_rgb.tobytes()
    return tk.PhotoImage(data=data, format="PPM")


def _resize_preview(frame_bgr, max_width: int, max_height: int):
    height, width = frame_bgr.shape[:2]
    scale = min(max_width / width, max_height / height, 1.0)
    if scale >= 1.0:
        return frame_bgr
    new_w = max(1, int(width * scale))
    new_h = max(1, int(height * scale))
    return cv2.resize(frame_bgr, (new_w, new_h), interpolation=cv2.INTER_AREA)


def _clamp(value: float, low: float, high: float) -> float:
    return max(low, min(high, value))


def _cast_slider_value(field: dict, value: float):
    caster = field["cast"]
    if caster is int:
        return int(round(value))
    return round(float(value), 3)


class CameraConfigTunerApp:
    def __init__(self, root: tk.Tk, config_path: Path):
        self.root = root
        self.config_path = config_path
        self.camera_cfg = load_config(config_path)
        self.cap: cv2.VideoCapture | None = None
        self.calibration = None
        self._closing = False
        self._after_id: str | None = None
        self.photo = None
        self.status_var = tk.StringVar(value=f"Loaded: {self.config_path}")

        self.slider_vars: dict[str, tk.DoubleVar] = {}
        self.slider_value_labels: dict[str, tk.Label] = {}

        self.index_var = tk.IntVar(value=int(self.camera_cfg.get("index", 0)))
        self.width_var = tk.StringVar(value=str(int(self.camera_cfg.get("width", 1600))))
        self.height_var = tk.StringVar(value=str(int(self.camera_cfg.get("height", 1200))))
        self.backend_var = tk.StringVar(value=str(self.camera_cfg.get("backend", "CAP_DSHOW")))
        self.capture_fourcc_var = tk.StringVar(value=str(self.camera_cfg.get("capture_fourcc", "MJPG")))
        self.writer_fourcc_var = tk.StringVar(value=str(self.camera_cfg.get("writer_fourcc", "mp4v")))

        self.root.title(WINDOW_TITLE)
        self.root.geometry(WINDOW_GEOMETRY)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)
        self._build_ui()

        self._open_camera()
        self._apply_live_settings()
        self._update_frame()

    def _build_ui(self) -> None:
        self.preview_label = tk.Label(self.root, text="Starting camera...")
        self.preview_label.pack(padx=8, pady=8)

        body = tk.Frame(self.root)
        body.pack(fill="x", padx=8, pady=(0, 6))

        self._build_camera_selector(body)
        self._build_resolution_fields(body)
        self._build_fourcc_fields(body)
        self._build_sliders(body)

        buttons = tk.Frame(self.root)
        buttons.pack(fill="x", padx=8, pady=(0, 6))

        tk.Button(buttons, text="Reopen Camera", width=16, command=self.reopen_camera).pack(side="left")
        tk.Button(buttons, text="Apply", width=12, command=self.apply_settings).pack(side="left", padx=(8, 0))
        tk.Button(buttons, text="Save", width=12, command=self.save_config).pack(side="left", padx=(8, 0))

        tk.Label(self.root, textvariable=self.status_var, anchor="w", justify="left").pack(fill="x", padx=8, pady=(0, 8))

    def _build_camera_selector(self, parent: tk.Frame) -> None:
        row = tk.Frame(parent)
        row.pack(fill="x", pady=(0, 4))

        tk.Label(row, text="Camera Index", width=14, anchor="w").pack(side="left")
        index_scale = tk.Scale(
            row,
            from_=0,
            to=10,
            orient="horizontal",
            resolution=1,
            variable=self.index_var,
            showvalue=True,
            length=200,
        )
        index_scale.pack(side="left", padx=(0, 10))

        tk.Label(row, text="Backend", width=8, anchor="w").pack(side="left")
        backend_menu = tk.OptionMenu(row, self.backend_var, *BACKEND_OPTIONS)
        backend_menu.pack(side="left")

    def _build_resolution_fields(self, parent: tk.Frame) -> None:
        row = tk.Frame(parent)
        row.pack(fill="x", pady=(0, 4))

        tk.Label(row, text="Resolution", width=14, anchor="w").pack(side="left")
        tk.Entry(row, textvariable=self.width_var, width=8).pack(side="left")
        tk.Label(row, text="x").pack(side="left", padx=4)
        tk.Entry(row, textvariable=self.height_var, width=8).pack(side="left")

    def _build_fourcc_fields(self, parent: tk.Frame) -> None:
        row = tk.Frame(parent)
        row.pack(fill="x", pady=(0, 4))

        tk.Label(row, text="Capture FOURCC", width=14, anchor="w").pack(side="left")
        tk.Entry(row, textvariable=self.capture_fourcc_var, width=8).pack(side="left", padx=(0, 12))

        tk.Label(row, text="Writer FOURCC", width=12, anchor="w").pack(side="left")
        tk.Entry(row, textvariable=self.writer_fourcc_var, width=8).pack(side="left")

    def _build_sliders(self, parent: tk.Frame) -> None:
        for field in SLIDER_FIELDS:
            key = field["key"]
            default_val = float(field["default"])
            current_val = float(self.camera_cfg.get(key, default_val))
            current_val = _clamp(current_val, float(field["min"]), float(field["max"]))

            row = tk.Frame(parent)
            row.pack(fill="x", pady=2)

            tk.Label(row, text=field["label"], width=14, anchor="w").pack(side="left")

            var = tk.DoubleVar(value=current_val)
            self.slider_vars[key] = var

            scale = tk.Scale(
                row,
                from_=float(field["min"]),
                to=float(field["max"]),
                orient="horizontal",
                resolution=float(field["resolution"]),
                variable=var,
                showvalue=False,
                command=lambda _v, field_key=key: self._on_slider_change(field_key),
            )
            scale.pack(side="left", fill="x", expand=True)

            value_label = tk.Label(row, width=10, anchor="e")
            value_label.pack(side="left", padx=(8, 0))
            self.slider_value_labels[key] = value_label

            self._sync_slider_value_label(key)

    def _on_slider_change(self, key: str) -> None:
        field = next(item for item in SLIDER_FIELDS if item["key"] == key)
        raw = float(self.slider_vars[key].get())
        value = _cast_slider_value(field, raw)
        self.camera_cfg[key] = value
        self._sync_slider_value_label(key)

    def _sync_slider_value_label(self, key: str) -> None:
        field = next(item for item in SLIDER_FIELDS if item["key"] == key)
        value = _cast_slider_value(field, float(self.slider_vars[key].get()))
        if field["cast"] is int:
            text = f"{int(value)}"
        else:
            text = f"{float(value):.2f}"
        self.slider_value_labels[key].configure(text=text)

    def _open_camera(self) -> None:
        cam_index = int(self.index_var.get())
        backend_name = self.backend_var.get().strip() or None
        backend = resolve_backend(backend_name)

        cap = cv2.VideoCapture(cam_index, backend) if backend is not None else cv2.VideoCapture(cam_index)
        if not cap.isOpened():
            raise RuntimeError(f"Camera not available: index={cam_index}, backend={backend_name or 'default'}")

        if self.cap is not None:
            self.cap.release()
        self.cap = cap

        self.camera_cfg["index"] = cam_index
        self.camera_cfg["backend"] = backend_name or "CAP_ANY"
        self.calibration = setup_undistortion_from_config(self.camera_cfg, log_prefix="[camera-tuner]")
        self.status_var.set(f"Opened camera index={cam_index}, backend={backend_name or 'default'}")

    def _apply_live_settings(self) -> None:
        if self.cap is None:
            return
        try:
            apply_camera_settings(self.cap, self.camera_cfg)
        except Exception as exc:
            self.status_var.set(f"Failed to apply settings: {exc}")

    def _update_frame(self) -> None:
        if self._closing or self.cap is None:
            return

        ok, frame = self.cap.read()
        if ok:
            try:
                frame = undistort_frame(frame, self.calibration)
                display_frame = _resize_preview(frame, PREVIEW_MAX_WIDTH, PREVIEW_MAX_HEIGHT)
                self.photo = _to_photo_image_bgr(display_frame)
                self.preview_label.configure(image=self.photo, text="")
            except tk.TclError:
                self.preview_label.configure(text="Failed to render camera preview.")
        else:
            self.preview_label.configure(text="No frame from camera. Retrying...")

        if not self._closing:
            self._after_id = self.root.after(20, self._update_frame)

    def reopen_camera(self) -> None:
        try:
            if not self._sync_config_from_controls():
                return
            self._open_camera()
            self._apply_live_settings()
        except Exception as exc:
            self.status_var.set(f"Failed to reopen camera: {exc}")

    def _sync_config_from_controls(self) -> bool:
        try:
            width = int(self.width_var.get())
            height = int(self.height_var.get())
        except ValueError:
            self.status_var.set("Resolution must be integer values.")
            return False

        if width <= 0 or height <= 0:
            self.status_var.set("Resolution must be greater than zero.")
            return False

        self.camera_cfg["width"] = width
        self.camera_cfg["height"] = height
        self.camera_cfg["index"] = int(self.index_var.get())
        self.camera_cfg["backend"] = self.backend_var.get().strip() or "CAP_ANY"
        self.camera_cfg["capture_fourcc"] = self.capture_fourcc_var.get().strip()
        self.camera_cfg["writer_fourcc"] = self.writer_fourcc_var.get().strip()

        for field in SLIDER_FIELDS:
            key = field["key"]
            self.camera_cfg[key] = _cast_slider_value(field, float(self.slider_vars[key].get()))

        return True

    def apply_settings(self) -> None:
        if not self._sync_config_from_controls():
            return
        self._apply_live_settings()
        self.status_var.set("Applied camera settings.")

    def save_config(self) -> None:
        if not self._sync_config_from_controls():
            return

        with self.config_path.open("w", encoding="utf-8") as f:
            json.dump(self.camera_cfg, f, indent=2)
            f.write("\n")

        self.status_var.set(f"Saved: {self.config_path}")

    def on_close(self) -> None:
        if self._closing:
            return
        self._closing = True

        if self._after_id is not None:
            try:
                self.root.after_cancel(self._after_id)
            except tk.TclError:
                pass
            self._after_id = None

        if self.cap is not None:
            self.cap.release()
            self.cap = None

        try:
            self.root.quit()
        except tk.TclError:
            pass
        try:
            self.root.destroy()
        except tk.TclError:
            pass


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Tune camera_config.json via GUI sliders and live preview.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CAMERA_CONFIG))
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    root = tk.Tk()
    app = CameraConfigTunerApp(root, config_path=config_path)
    _ = app
    root.mainloop()

if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        messagebox.showerror("Camera Config Tuner Error", str(e))
        raise
