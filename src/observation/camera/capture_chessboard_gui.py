import datetime
import sys
import tkinter as tk
from pathlib import Path
from tkinter import messagebox

import cv2

# Ensure repository src root is importable when running as a script.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from observation.camera.camera_utils import apply_camera_settings, resolve_backend
from utils.config_loader import load_config


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CAMERA_CONFIG = SCRIPT_DIR / "camera_config.json"

# Change this path if you want to save captured images elsewhere.
SAVE_DIR = SCRIPT_DIR / "chessboard_imgs"
WINDOW_TITLE = "Chessboard Capture"


def _to_photo_image_bgr(frame_bgr):
    frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    height, width, _ = frame_rgb.shape
    header = f"P6\n{width} {height}\n255\n".encode("ascii")
    data = header + frame_rgb.tobytes()
    return tk.PhotoImage(data=data, format="PPM")


class CaptureApp:
    def __init__(self, root: tk.Tk, camera_cfg: dict, save_dir: Path):
        self.root = root
        self.camera_cfg = camera_cfg
        self.save_dir = save_dir
        self.save_dir.mkdir(parents=True, exist_ok=True)

        self.cap = self._open_camera()
        self.last_frame = None
        self.photo = None
        self.capture_count = 0

        self.root.title(WINDOW_TITLE)
        self.root.protocol("WM_DELETE_WINDOW", self.on_close)

        self.preview_label = tk.Label(self.root, text="Starting camera...")
        self.preview_label.pack(padx=8, pady=8)

        self.capture_button = tk.Button(self.root, text="Capture", width=16, command=self.capture)
        self.capture_button.pack(pady=(0, 6))

        self.status_var = tk.StringVar(
            value=f"Save dir: {self.save_dir} | Click 'Capture' to save an image."
        )
        self.status_label = tk.Label(self.root, textvariable=self.status_var, anchor="w", justify="left")
        self.status_label.pack(fill="x", padx=8, pady=(0, 8))

        self._update_frame()

    def _open_camera(self):
        cam_num = int(self.camera_cfg.get("index", 0))
        backend = resolve_backend(self.camera_cfg.get("backend"))
        cap = cv2.VideoCapture(cam_num, backend) if backend is not None else cv2.VideoCapture(cam_num)
        if not cap.isOpened():
            raise RuntimeError("Camera not available.")
        apply_camera_settings(cap, self.camera_cfg)
        return cap

    def _update_frame(self):
        if self.cap is None:
            return

        ok, frame = self.cap.read()
        if ok:
            self.last_frame = frame
            try:
                self.photo = _to_photo_image_bgr(frame)
                self.preview_label.configure(image=self.photo, text="")
            except tk.TclError:
                self.preview_label.configure(text="Failed to render camera preview.")
        else:
            self.preview_label.configure(text="No frame from camera. Retrying...")

        self.root.after(20, self._update_frame)

    def capture(self):
        if self.last_frame is None:
            self.status_var.set("No frame available yet.")
            return

        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        filename = self.save_dir / f"chessboard_{timestamp}.png"
        ok = cv2.imwrite(str(filename), self.last_frame)
        if not ok:
            self.status_var.set(f"Failed to save image: {filename}")
            return

        self.capture_count += 1
        self.status_var.set(f"Saved: {filename.name} | Total captured: {self.capture_count}")

    def on_close(self):
        if self.cap is not None:
            self.cap.release()
            self.cap = None
        self.root.destroy()


def main() -> None:
    camera_cfg = load_config(DEFAULT_CAMERA_CONFIG)
    root = tk.Tk()
    app = CaptureApp(root, camera_cfg=camera_cfg, save_dir=SAVE_DIR)
    _ = app
    root.mainloop()


if __name__ == "__main__":
    try:
        main()
    except Exception as e:
        messagebox.showerror("Capture Error", str(e))
        raise
