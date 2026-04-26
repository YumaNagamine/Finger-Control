from __future__ import annotations

import argparse
import datetime
import time
from pathlib import Path

import cv2

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


TEST_VIDEO_DIR = Path(__file__).with_name("test_video")
DEFAULT_CONFIG = Path(__file__).with_name("camera_config.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Record a test video from camera_config.json.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--duration-s", type=float, default=5.0, help="Recording duration in seconds.")
    parser.add_argument("--frame-limit", type=int, default=0, help="Stop after N frames (0 = ignore).")
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    camera_cfg = load_config(args.config)

    cam_num = int(camera_cfg.get("index", 0))
    backend = resolve_backend(camera_cfg.get("backend"))
    cap = cv2.VideoCapture(cam_num, backend) if backend is not None else cv2.VideoCapture(cam_num)
    if not cap.isOpened():
        raise RuntimeError("Camera not available.")

    width = int(camera_cfg.get("width", 1600))
    height = int(camera_cfg.get("height", 1200))
    target_fps = float(camera_cfg.get("target_fps", 90))
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
        while True:
            ret, frame = cap.read()
            if not ret:
                continue
            frame = undistort_frame(frame, calibration)
            writer.write(frame)
            frame_count += 1
            if args.frame_limit and frame_count >= args.frame_limit:
                break
            if args.duration_s > 0 and (time.time() - start) >= args.duration_s:
                break
    finally:
        cap.release()
        writer.release()

    print(f"Saved test video to {filename}")


if __name__ == "__main__":
    main()
