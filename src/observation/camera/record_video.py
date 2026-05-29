from __future__ import annotations

import argparse
import datetime
import sys
import time
from pathlib import Path

import cv2

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

try:
    from observation.camera.camera_utils import (
        apply_camera_settings,
        fourcc_from_str,
        resolve_backend,
        setup_undistortion_from_config,
        undistort_frame,
    )
except ModuleNotFoundError:
    # Supports running this script directly from the camera directory.
    from camera_utils import (
        apply_camera_settings,
        fourcc_from_str,
        resolve_backend,
        setup_undistortion_from_config,
        undistort_frame,
    )

from utils.config_loader import load_config
from observation.camera.camera_param_resolver import resolve_param_path


TEST_VIDEO_DIR = Path(__file__).with_name("test_video")
DEFAULT_CONFIG = resolve_param_path("camera_config_dlc.json")

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a test video from camera_config.json.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--duration-s", type=float, default=0.0, help="Recording duration in seconds.")
    parser.add_argument("--frame-limit", type=int, default=0, help="Stop after N frames (0 = ignore).")
    parser.add_argument("--no-window", action="store_true", help="Disable realtime preview window.")
    parser.add_argument("--window-name", type=str, default="Record Preview", help="Preview window title.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    camera_cfg = load_config(args.config)
    show_window = not args.no_window

    cam_num = int(camera_cfg.get("index", 0))
    backend = resolve_backend(camera_cfg.get("backend"))
    cap = cv2.VideoCapture(cam_num, backend) if backend is not None else cv2.VideoCapture(cam_num)
    if not cap.isOpened():
        raise RuntimeError("Camera not available.")

    width = int(camera_cfg.get("width", 800))
    height = int(camera_cfg.get("height", 600))
    target_fps = float(camera_cfg.get("target_fps", 60))
    apply_camera_settings(cap, camera_cfg)
    calibration = setup_undistortion_from_config(camera_cfg)

    TEST_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = TEST_VIDEO_DIR / f"test_{timestamp}.mp4"
    writer_fourcc = fourcc_from_str(camera_cfg.get("writer_fourcc", "mp4v"))
    writer = cv2.VideoWriter(str(filename), writer_fourcc, target_fps, (width, height))

    start = time.time()
    frame_count = 0
    try:
        if show_window:
            cv2.namedWindow(args.window_name, cv2.WINDOW_NORMAL)

        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = undistort_frame(frame, calibration)
            writer.write(frame)

            if show_window:
                cv2.imshow(args.window_name, frame)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            frame_count += 1
            if args.frame_limit and frame_count >= args.frame_limit:
                break
            if args.duration_s > 0 and (time.time() - start) >= args.duration_s:
                break
    finally:
        cap.release()
        writer.release()
        if show_window:
            cv2.destroyAllWindows()

    print(f"Saved test video to {filename}")


if __name__ == "__main__":
    main()
