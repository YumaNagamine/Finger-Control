from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

# Ensure repository root is importable when running as a script.
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
from observation.vision.dlc_angle_processor import DLCAngleProcessor
from utils.config_loader import load_config
from utils.path_utils import resolve_path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_DLC_CONFIG = SCRIPT_DIR / "config_deeplabcut_angle.json"
DEFAULT_CAMERA_CONFIG = resolve_param_path("camera_config.json")
WINDOW_NAME = "Realtime Joint Measurement"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Open the configured camera and estimate DLC-based joint angles in realtime."
    )
    parser.add_argument("--config", type=str, default=str(DEFAULT_DLC_CONFIG))
    parser.add_argument("--camera-config", type=str, default=str(DEFAULT_CAMERA_CONFIG))
    parser.add_argument("--frame-limit", type=int, default=0, help="Stop after N frames (0 = unlimited).")
    parser.add_argument("--duration-s", type=float, default=0.0, help="Stop after elapsed seconds (0 = unlimited).")
    parser.add_argument("--no-window", action="store_true", help="Disable the OpenCV preview window.")
    return parser.parse_args()


def _open_camera(camera_cfg: dict) -> cv2.VideoCapture:
    cam_index = int(camera_cfg.get("index", 0))
    backend = resolve_backend(camera_cfg.get("backend"))
    cap = cv2.VideoCapture(cam_index, backend) if backend is not None else cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Camera not available: index={cam_index}")
    apply_camera_settings(cap, camera_cfg)
    return cap


def _resolve_dlc_runtime_config(config_path: Path) -> dict:
    config = load_config(
        config_path,
        required_keys=("input", "dlc", "keypoints", "output"),
    )

    dlc_cfg = dict(config["dlc"])
    if dlc_cfg.get("third_party_path"):
        third_party_path = resolve_path(str(dlc_cfg["third_party_path"]), config_path.parent)
        if third_party_path is not None:
            dlc_cfg["third_party_path"] = str(third_party_path)
    if dlc_cfg.get("model_path"):
        model_path = resolve_path(str(dlc_cfg["model_path"]), config_path.parent)
        if model_path is not None:
            dlc_cfg["model_path"] = str(model_path)

    runtime_config = dict(config)
    runtime_config["dlc"] = dlc_cfg
    return runtime_config


def main() -> None:
    args = parse_args()

    config_path = Path(args.config).resolve()
    camera_config_path = Path(args.camera_config).resolve()
    runtime_config = _resolve_dlc_runtime_config(config_path)
    camera_cfg = load_config(camera_config_path)

    processor = DLCAngleProcessor(runtime_config, config_path.parent, enable_live=True)
    calibration = setup_undistortion_from_config(camera_cfg, log_prefix="[realtime-joint]")
    cap = _open_camera(camera_cfg)

    show_window = not args.no_window
    frame_idx = 0
    start_time = time.perf_counter()

    if show_window:
        cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame is None:
                raise RuntimeError("Failed to read a frame from the camera.")

            frame = undistort_frame(frame, calibration)
            result, overlay = processor.process_frame(frame_bgr=frame, frame_idx=frame_idx)
            angles = result["angles"]

            dip = angles.get("DIP", float("nan"))
            pip = angles.get("PIP", float("nan"))
            mcp = angles.get("MCP", float("nan"))
            print(
                f"frame={frame_idx} "
                f"DIP={dip:.2f} "
                f"PIP={pip:.2f} "
                f"MCP={mcp:.2f}",
                flush=True,
            )

            if show_window:
                cv2.imshow(WINDOW_NAME, overlay)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            frame_idx += 1
            elapsed_s = time.perf_counter() - start_time
            if args.frame_limit > 0 and frame_idx >= args.frame_limit:
                break
            if args.duration_s > 0.0 and elapsed_s >= args.duration_s:
                break
    finally:
        cap.release()
        if show_window:
            cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
