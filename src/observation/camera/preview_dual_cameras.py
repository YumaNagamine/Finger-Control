from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2
import numpy as np

# Ensure repository root is importable when running as a script.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from observation.camera.camera_utils import (
    apply_camera_settings,
    resolve_backend,
    setup_undistortion_from_config,
    undistort_frame,
)
from utils.config_loader import load_config


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR.parent / "vision" / "config_dual_camera_record.json"
DEFAULT_CAMERA_CONFIG = SCRIPT_DIR / "camera_config.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Preview two cameras side-by-side without recording.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--camera-config", type=str, default=str(DEFAULT_CAMERA_CONFIG))
    parser.add_argument("--duration-s", type=float, default=None, help="Override capture.duration_s in config.")
    parser.add_argument("--frame-limit", type=int, default=None, help="Override capture.frame_limit in config.")
    parser.add_argument("--target-fps", type=float, default=None, help="Override capture.target_fps in config.")
    parser.add_argument("--no-window", action="store_true", help="Disable OpenCV preview window.")
    return parser.parse_args()


def _open_camera(camera_cfg: dict, fallback_name: str) -> tuple[cv2.VideoCapture, str]:
    cam_index = int(camera_cfg.get("index", 0))
    backend = resolve_backend(camera_cfg.get("backend"))
    cap = cv2.VideoCapture(cam_index, backend) if backend is not None else cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Camera not available: index={cam_index}, name={fallback_name}")
    apply_camera_settings(cap, camera_cfg)
    return cap, str(camera_cfg.get("name", fallback_name))


def _optional_list_value(values: object, idx: int) -> object | None:
    if isinstance(values, list) and len(values) == 2:
        return values[idx]
    return None


def _join_preview(frame0: np.ndarray, frame1: np.ndarray) -> np.ndarray:
    h0, _ = frame0.shape[:2]
    h1, w1 = frame1.shape[:2]
    if h0 != h1:
        frame1 = cv2.resize(frame1, (w1, h0), interpolation=cv2.INTER_AREA)
    return cv2.hconcat([frame0, frame1])


def main() -> None:
    args = parse_args()
    config = load_config(Path(args.config).resolve(), required_keys=("camera",))
    base_camera_cfg = load_config(Path(args.camera_config).resolve())

    dual_camera_cfg = dict(config["camera"])
    camera_indices = dual_camera_cfg.get("indices", [0, 1])
    if not isinstance(camera_indices, list) or len(camera_indices) != 2:
        raise ValueError("camera.indices must be a list with exactly 2 integers.")
    camera_names = dual_camera_cfg.get("names", ["cam0", "cam1"])
    if not isinstance(camera_names, list) or len(camera_names) != 2:
        raise ValueError("camera.names must be a list with exactly 2 strings.")
    calibration_paths = dual_camera_cfg.get("calibration_paths", None)
    calibration_enabled = dual_camera_cfg.get("use_chessboard_calibration", None)

    camera_cfg_0 = dict(base_camera_cfg)
    camera_cfg_0["index"] = int(camera_indices[0])
    camera_cfg_0["name"] = str(camera_names[0])
    cam0_calib_path = _optional_list_value(calibration_paths, 0)
    cam0_calib_enable = _optional_list_value(calibration_enabled, 0)
    if cam0_calib_path is not None:
        camera_cfg_0["calibration_path"] = str(cam0_calib_path)
    if cam0_calib_enable is not None:
        camera_cfg_0["use_chessboard_calibration"] = bool(cam0_calib_enable)

    camera_cfg_1 = dict(base_camera_cfg)
    camera_cfg_1["index"] = int(camera_indices[1])
    camera_cfg_1["name"] = str(camera_names[1])
    cam1_calib_path = _optional_list_value(calibration_paths, 1)
    cam1_calib_enable = _optional_list_value(calibration_enabled, 1)
    if cam1_calib_path is not None:
        camera_cfg_1["calibration_path"] = str(cam1_calib_path)
    if cam1_calib_enable is not None:
        camera_cfg_1["use_chessboard_calibration"] = bool(cam1_calib_enable)

    capture_cfg = dict(config.get("capture", {}))
    output_cfg = dict(config.get("output", {}))

    duration_s = float(args.duration_s if args.duration_s is not None else capture_cfg.get("duration_s", 0.0))
    frame_limit = int(args.frame_limit if args.frame_limit is not None else capture_cfg.get("frame_limit", 0))
    target_fps = float(args.target_fps if args.target_fps is not None else capture_cfg.get("target_fps", 0.0))
    warmup_frames = max(0, int(capture_cfg.get("warmup_frames", 5)))

    show_window = (not args.no_window) and bool(output_cfg.get("show_window", True))
    preview_name = "Dual Camera Preview"

    cap0 = None
    cap1 = None
    calibration0 = None
    calibration1 = None

    try:
        cap0, _ = _open_camera(dict(camera_cfg_0), fallback_name="cam0")
        cap1, _ = _open_camera(dict(camera_cfg_1), fallback_name="cam1")
        calibration0 = setup_undistortion_from_config(camera_cfg_0, log_prefix="[preview-dual-cam0]")
        calibration1 = setup_undistortion_from_config(camera_cfg_1, log_prefix="[preview-dual-cam1]")

        for _ in range(warmup_frames):
            cap0.read()
            cap1.read()

        ok0, frame0 = cap0.read()
        ok1, frame1 = cap1.read()
        if not ok0 or frame0 is None:
            raise RuntimeError("Failed to read initial frame from cam0.")
        if not ok1 or frame1 is None:
            raise RuntimeError("Failed to read initial frame from cam1.")
        frame0 = undistort_frame(frame0, calibration0)
        frame1 = undistort_frame(frame1, calibration1)
        h0, w0 = frame0.shape[:2]
        h1, w1 = frame1.shape[:2]

        if show_window:
            cv2.namedWindow(preview_name, cv2.WINDOW_NORMAL)
            cv2.resizeWindow(preview_name, max(1, (w0 + w1) // 2), max(1, max(h0, h1) // 2))
            print("Press ESC in the preview window to stop.")

        interval_s = 0.0 if target_fps <= 0.0 else 1.0 / target_fps
        start_perf = time.perf_counter()
        next_tick = start_perf
        pair_idx = 0

        while True:
            if interval_s > 0.0:
                now = time.perf_counter()
                sleep_s = next_tick - now
                if sleep_s > 0.0:
                    time.sleep(sleep_s)
                next_tick += interval_s

            now_perf = time.perf_counter()
            elapsed_s = now_perf - start_perf

            ok0, frame0 = cap0.read()
            ok1, frame1 = cap1.read()
            if ok0 and frame0 is not None:
                frame0 = undistort_frame(frame0, calibration0)
            if ok1 and frame1 is not None:
                frame1 = undistort_frame(frame1, calibration1)

            if show_window and ok0 and frame0 is not None and ok1 and frame1 is not None:
                preview = _join_preview(frame0, frame1)
                cv2.imshow(preview_name, preview)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            if not ok0 or frame0 is None or not ok1 or frame1 is None:
                print("Stopping preview because one or both cameras returned no frame.")
                break

            pair_idx += 1
            if frame_limit > 0 and pair_idx >= frame_limit:
                break
            if duration_s > 0.0 and elapsed_s >= duration_s:
                break
    finally:
        if cap0 is not None:
            cap0.release()
        if cap1 is not None:
            cap1.release()
        if show_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
