from __future__ import annotations

import argparse
import datetime
import multiprocessing as mp
import sys
import time
from pathlib import Path

import cv2
from PIL import Image, ImageTk
import tkinter as tk
import ttkbootstrap as ttk
from ttkbootstrap.widgets.scrolled import ScrolledText

from controller.interface import build_interface
from observation.camera.camera_utils import setup_undistortion_from_config, undistort_frame
from utils.config_loader import load_config


RAW_VIDEO_DIR = Path("log/side_angle/raw_video")
DEFAULT_CONFIG = Path(__file__).with_name("interface_config.json")
CAMERA_CONFIG = Path(__file__).parents[1] / "observation" / "camera" / "camera_config.json"


class ScrollerLogger:
    def __init__(self, scroller: ScrolledText):
        self.terminal = sys.stdout
        self.scroller = scroller

    def write(self, message):
        self.terminal.write(message)
        self.scroller.insert(tk.END, message)
        self.scroller.update()
        self.scroller.text.yview_moveto(1)

    def flush(self):
        pass


class ControlGUI:
    def __init__(self, root, shared, config_path: Path, mock: bool):
        self.root = root
        self.shared = shared
        self.config_path = config_path
        self.mock = mock
        self.interface = None
        self.photo_acquired_t = 0.0

        self.root.protocol("WM_DELETE_WINDOW", self.on_exit)
        self._build_layout()
        self._connect()
        self._refresh_image()

    def _build_layout(self):
        screen_width = self.root.winfo_screenwidth()
        screen_height = self.root.winfo_screenheight()
        self.root.geometry(f"{int(screen_width * 0.98)}x{int(screen_height * 0.92)}")

        nav_width = 0.04
        frame_nav = ttk.Frame(self.root, bootstyle="primary")
        frame_nav.place(relx=0, rely=0, relheight=1, relwidth=nav_width)

        frame_page = ttk.Frame(self.root)
        frame_page.place(relx=nav_width, rely=0.01, relheight=0.98, relwidth=1 - nav_width)

        self._build_video_log(frame_page)
        self._build_scales(frame_page)

    def _build_video_log(self, parent):
        left_width = 0.5
        frame_left = ttk.Frame(parent)
        frame_left.place(relx=0, rely=0, relheight=1, relwidth=left_width)

        log_video_frame = ttk.Labelframe(frame_left, text="Video", bootstyle="info")
        log_video_frame.place(relx=0.01, rely=0, relheight=0.56, relwidth=0.98)

        self.video_label = tk.Label(log_video_frame)
        self.video_label.place(relx=0, rely=0, relheight=0.95, relwidth=1)

        self.variable_fps = ttk.StringVar(value="FPS: -")
        fps_label = ttk.Label(log_video_frame, textvariable=self.variable_fps, bootstyle="info")
        fps_label.place(relx=0.2, rely=0.95, relheight=0.05, relwidth=0.2)

        log_frame = ttk.Labelframe(frame_left, text="Log", bootstyle="info")
        log_frame.place(relx=0.01, rely=0.56, relheight=0.44, relwidth=0.98)

        self.scroller_log = ScrolledText(log_frame, font=("Calibri Light", 8), bootstyle="dark", vbar=True, autohide=True)
        self.scroller_log.place(relx=0, rely=0, relheight=1, relwidth=1)
        sys.stdout = ScrollerLogger(self.scroller_log)
        sys.stderr = sys.stdout

    def _build_scales(self, parent):
        right_frame = ttk.Frame(parent)
        right_frame.place(relx=0.51, rely=0, relheight=1, relwidth=0.49)

        scale_frame = ttk.Frame(right_frame)
        scale_frame.place(relx=0, rely=0, relheight=0.8, relwidth=1)

        self.output_vars = []
        num_scales = 8
        for idx in range(num_scales):
            self.output_vars.append(ttk.DoubleVar(value=0.0))

        for idx in range(num_scales):
            frame = ttk.Frame(scale_frame)
            frame.place(relx=idx / num_scales, rely=0, relheight=1, relwidth=1 / num_scales - 0.006)

            label_frame = ttk.Labelframe(frame, text=f"Ch {idx * 2}", bootstyle="info")
            label_frame.place(relx=0, rely=0, relheight=0.11, relwidth=1)
            entry = ttk.Entry(label_frame, textvariable=self.output_vars[idx])
            entry.place(relx=0, rely=0, relheight=1, relwidth=1)

            btn_max = ttk.Button(frame, text="MAX", bootstyle="success-outline", command=lambda i=idx: self.output_vars[i].set(1.0))
            btn_max.place(relx=0, rely=0.12, relheight=0.1, relwidth=1)

            btn_min = ttk.Button(frame, text="MIN", bootstyle="success-outline", command=lambda i=idx: self.output_vars[i].set(0.0))
            btn_min.place(relx=0, rely=0.88, relheight=0.1, relwidth=1)

            scale = ttk.Scale(frame, value=0, orient=tk.VERTICAL, takefocus=1, bootstyle="SUCCESS", from_=1, to=0, variable=self.output_vars[idx])
            scale.place(relx=0, rely=0.23, relheight=0.62, relwidth=1)

        button_frame = ttk.Frame(right_frame)
        button_frame.place(relx=0, rely=0.81, relheight=0.18, relwidth=1)

        buttons = [
            ("Connect\n\nPCA-9685", "info-outline", self._connect),
            ("STOP\n\nAll channel", "danger-outline", self._stop),
            ("APPLY\n\nDuty Ratio", "warning-outline", self._apply),
            ("EXIT", "secondary-outline", self.on_exit),
        ]

        for idx, (label, style, func) in enumerate(buttons):
            btn = ttk.Button(button_frame, text=label, style=style, command=func)
            btn.place(relx=idx / len(buttons), rely=0, relheight=1, relwidth=1 / len(buttons) - 0.006)

    def _connect(self):
        print("Connecting interface...")
        try:
            self.interface = build_interface(self.config_path, mock=self.mock)
            print("Connected.")
        except Exception as err:
            print(f"Connection failed: {err}")

    def _stop(self):
        print("Stopping all channel output.")
        for var in self.output_vars:
            var.set(0.0)
        self._apply()

    def _apply(self):
        if not self.interface:
            print("No connection.")
            return
        duties = [max(0.0, min(1.0, float(v.get()))) for v in self.output_vars]
        applied = self.interface.set(duties)
        print("Applied duty:", " ".join(f"{d:.2f}" for d in applied))

    def _refresh_image(self):
        try:
            acquired_t = self.shared.get("photo_acquired_t", 0.0)
            if acquired_t and acquired_t != self.photo_acquired_t:
                self.photo_acquired_t = acquired_t
                image = self.shared.get("photo")
                if image is not None:
                    photo = ImageTk.PhotoImage(image)
                    self.video_label.configure(image=photo)
                    self.video_label.image = photo
                    dt = max(acquired_t - self.shared.get("prev_photo_acquired_t", acquired_t), 1e-6)
                    fps = int(1 / dt) if dt > 0 else 0
                    self.variable_fps.set(f"FPS: {fps}")
                    self.shared["prev_photo_acquired_t"] = acquired_t
        except Exception as err:
            print(f"Video frame load failed: {err}")

        self.root.after(10, self._refresh_image)

    def on_exit(self, *args):
        print("Exiting...")
        try:
            self._stop()
        except Exception as err:
            print(f"Stop failed: {err}")
        self.shared["exit"] = True
        self.root.destroy()
        sys.exit()


def _resolve_backend(backend_name: str | None) -> int | None:
    if not backend_name:
        return None
    return {
        "CAP_DSHOW": cv2.CAP_DSHOW,
        "CAP_ANY": cv2.CAP_ANY,
    }.get(backend_name)


def _fourcc_from_str(code: str | None) -> int | None:
    if not code or len(code) != 4:
        return None
    return cv2.VideoWriter_fourcc(*code)


def camera_process(shared, camera_cfg: dict):
    cam_num = int(camera_cfg.get("index", 0))
    backend = _resolve_backend(camera_cfg.get("backend"))
    cap = cv2.VideoCapture(cam_num, backend) if backend is not None else cv2.VideoCapture(cam_num)
    if not cap.isOpened():
        print("Camera not available.")
        return

    width = int(camera_cfg.get("width", 1600))
    height = int(camera_cfg.get("height", 1200))
    target_fps = float(camera_cfg.get("target_fps", 90))
    buffersize = int(camera_cfg.get("buffersize", 0))
    auto_exposure = float(camera_cfg.get("auto_exposure", 1))
    gain = float(camera_cfg.get("gain", 0))
    exposure = float(camera_cfg.get("exposure", -11))
    capture_fourcc = _fourcc_from_str(camera_cfg.get("capture_fourcc", "MJPG"))
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, width)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, height)
    cap.set(cv2.CAP_PROP_FPS, target_fps)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, buffersize)
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, auto_exposure)
    cap.set(cv2.CAP_PROP_GAIN, gain)
    cap.set(cv2.CAP_PROP_EXPOSURE, exposure)
    if capture_fourcc is not None:
        cap.set(cv2.CAP_PROP_FOURCC, capture_fourcc)
    calibration = setup_undistortion_from_config(camera_cfg, log_prefix="[controller-gui]")

    RAW_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = RAW_VIDEO_DIR / f"raw_{timestamp}.mp4"
    writer_fourcc = _fourcc_from_str(camera_cfg.get("writer_fourcc", "mp4v"))
    writer = cv2.VideoWriter(str(filename), writer_fourcc, target_fps, (width, height))

    try:
        while not shared.get("exit", False):
            ret, frame = cap.read()
            if not ret:
                continue
            frame = undistort_frame(frame, calibration)
            writer.write(frame)
            image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            shared["photo"] = Image.fromarray(image)
            shared["photo_acquired_t"] = time.time()
    finally:
        cap.release()
        writer.release()
        print("Camera and video writer released.")


def gui_process(shared, config_path: Path, mock: bool):
    root = ttk.Window(hdpi=True, scaling=3, themename="darkly")
    root.title("Control SMA")
    root.geometry("+0+0")
    ControlGUI(root, shared, config_path, mock)
    root.mainloop()


def parse_args():
    parser = argparse.ArgumentParser(description="SMA control GUI.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--camera-config", type=str, default=str(CAMERA_CONFIG))
    parser.add_argument("--mock", action="store_true", help="Use mock duty interface.")
    return parser.parse_args()


def main():
    args = parse_args()
    config_path = Path(args.config)
    camera_config = load_config(args.camera_config)

    with mp.Manager() as manager:
        shared = manager.dict()
        shared["photo"] = None
        shared["photo_acquired_t"] = 0.0
        shared["exit"] = False

        proc_cam = mp.Process(target=camera_process, name="CAM", args=(shared, camera_config))
        proc_gui = mp.Process(target=gui_process, name="GUI", args=(shared, config_path, args.mock))

        proc_cam.start()
        time.sleep(2)
        proc_gui.start()

        proc_gui.join()
        proc_cam.join()


if __name__ == "__main__":
    main()
