from __future__ import annotations

import argparse
import datetime
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from observation.camera.camera_recorder import CameraRecorder
from observation.camera.camera_param_resolver import resolve_param_path
from utils.config_loader import load_config


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

    TEST_VIDEO_DIR.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename = TEST_VIDEO_DIR / f"test_{timestamp}.mp4"
    recorder = CameraRecorder(
        camera_cfg,
        filename,
        show_preview=show_window,
        preview_window_name=args.window_name,
        preview_escape_is_error=False,
        frame_limit=args.frame_limit or None,
    )

    try:
        recorder.prepare()
        started_at = time.monotonic()
        recorder.start(started_at)
        while True:
            recorder.raise_if_failed()
            if recorder.stop_requested:
                break
            if args.duration_s > 0 and (time.monotonic() - started_at) >= args.duration_s:
                break
            time.sleep(0.01)
    finally:
        recorder.close()

    print(f"Saved test video to {filename}")


if __name__ == "__main__":
    main()
