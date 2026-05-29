from __future__ import annotations

import argparse
import sys
from pathlib import Path

import cv2
import numpy as np

# Ensure repository src root is importable when running as a script.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from observation.camera.camera_utils import (
    apply_camera_settings,
    load_chessboard_calibration,
    resolve_backend,
    undistort_frame,
)
from observation.camera.camera_param_resolver import resolve_param_path
from utils.config_loader import load_config


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CAMERA_CONFIG = resolve_param_path("camera_config.json")
DEFAULT_CALIBRATION_PATH = resolve_param_path("camera_calibration.json")
DEFAULT_WINDOW_TITLE = "Calibration Compare (Left: RAW | Right: UNDISTORTED)"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Preview raw vs undistorted frames in one realtime window."
    )
    parser.add_argument("--camera-config", type=str, default=str(DEFAULT_CAMERA_CONFIG))
    parser.add_argument("--calibration", type=str, default=str(DEFAULT_CALIBRATION_PATH))
    parser.add_argument(
        "--camera-index",
        type=int,
        default=None,
        help="Override camera index from camera config.",
    )
    parser.add_argument("--window-title", type=str, default=DEFAULT_WINDOW_TITLE)
    return parser.parse_args()


def _draw_label(frame: np.ndarray, text: str) -> None:
    cv2.rectangle(frame, (8, 8), (260, 40), (0, 0, 0), -1)
    cv2.putText(frame, text, (16, 32), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)


def _open_camera(camera_cfg: dict, override_index: int | None) -> cv2.VideoCapture:
    cam_index = int(override_index if override_index is not None else camera_cfg.get("index", 0))
    backend = resolve_backend(camera_cfg.get("backend"))
    cap = cv2.VideoCapture(cam_index, backend) if backend is not None else cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError(f"Camera not available. index={cam_index}")
    apply_camera_settings(cap, camera_cfg)
    return cap


def main() -> None:
    args = parse_args()
    camera_cfg = load_config(args.camera_config)
    calibration = load_chessboard_calibration(args.calibration)

    cap = _open_camera(camera_cfg, args.camera_index)
    cv2.namedWindow(args.window_title, cv2.WINDOW_NORMAL)

    try:
        while True:
            ok, frame = cap.read()
            if not ok:
                continue

            undistorted = undistort_frame(frame, calibration)

            raw_view = frame.copy()
            undist_view = undistorted.copy()
            _draw_label(raw_view, "RAW")
            _draw_label(undist_view, "UNDISTORTED")

            combined = np.hstack((raw_view, undist_view))
            cv2.imshow(args.window_title, combined)

            key = cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                break
    finally:
        cap.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
