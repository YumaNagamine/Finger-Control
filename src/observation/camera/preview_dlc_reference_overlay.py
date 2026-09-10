from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import cv2

# Ensure the repository's src directory is importable when run as a script.
SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from observation.camera.camera_param_resolver import resolve_param_path
from observation.camera.camera_utils import (
    apply_camera_settings,
    resolve_backend,
    setup_undistortion_from_config,
    undistort_frame,
)
from utils.config_loader import load_config


DEFAULT_REFERENCE_VIDEO = Path(
    r"C:\Users\ynaga\Desktop\Finger-Control\logs\dual_camera\recordings"
    r"\dual_record_20260909_171426\cam0.mp4"
)
DEFAULT_CAMERA_CONFIG = resolve_param_path("camera_config_dlc.json")
WINDOW_NAME = "DLC-calibrated reference overlay"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay the DLC-calibrated first frame of a recording on the "
            "DLC-calibrated live camera image."
        )
    )
    parser.add_argument("--video", type=Path, default=DEFAULT_REFERENCE_VIDEO)
    parser.add_argument("--camera-config", type=Path, default=DEFAULT_CAMERA_CONFIG)
    parser.add_argument(
        "--camera-index",
        type=int,
        default=0,
        help="Live camera index (default: 0).",
    )
    parser.add_argument(
        "--reference-alpha",
        type=float,
        default=0.5,
        help="Reference-frame opacity from 0.0 to 1.0 (default: 0.5).",
    )
    return parser.parse_args()


def read_first_frame(video_path: Path):
    capture = cv2.VideoCapture(str(video_path))
    try:
        if not capture.isOpened():
            raise RuntimeError(f"Could not open reference video: {video_path}")
        ok, frame = capture.read()
        if not ok or frame is None:
            raise RuntimeError(f"Could not read the first frame: {video_path}")
        return frame
    finally:
        capture.release()


def open_camera(camera_cfg: dict) -> tuple[cv2.VideoCapture, str, object]:
    camera_index = int(camera_cfg.get("index", 0))
    configured_backend = resolve_backend(camera_cfg.get("backend"))
    backend_candidates = [
        (str(camera_cfg.get("backend", "CAP_ANY")), configured_backend),
        ("CAP_MSMF", cv2.CAP_MSMF),
        ("CAP_ANY", cv2.CAP_ANY),
    ]
    tried: set[int | None] = set()

    for backend_name, backend in backend_candidates:
        if backend in tried:
            continue
        tried.add(backend)
        capture = (
            cv2.VideoCapture(camera_index, backend)
            if backend is not None
            else cv2.VideoCapture(camera_index)
        )
        if not capture.isOpened():
            capture.release()
            print(f"[reference-overlay] index={camera_index} failed with {backend_name}.")
            continue

        apply_camera_settings(capture, camera_cfg)
        ok, first_frame = capture.read()
        if ok and first_frame is not None:
            print(f"[reference-overlay] index={camera_index}, backend={backend_name}")
            return capture, backend_name, first_frame

        capture.release()
        print(
            f"[reference-overlay] index={camera_index} opened with {backend_name}, "
            "but returned no frame."
        )

    raise RuntimeError(
        f"Camera index={camera_index} returned no frames with any available backend."
    )


def main() -> None:
    args = parse_args()
    if not 0.0 <= args.reference_alpha <= 1.0:
        raise ValueError("--reference-alpha must be between 0.0 and 1.0.")

    video_path = args.video.expanduser().resolve()
    camera_config_path = args.camera_config.expanduser().resolve()
    camera_cfg = load_config(camera_config_path)
    camera_cfg["index"] = args.camera_index
    calibration = setup_undistortion_from_config(
        camera_cfg,
        log_prefix="[reference-overlay]",
    )
    if calibration is None:
        raise RuntimeError(
            "DLC camera calibration is required. Check use_chessboard_calibration "
            "and calibration_path in the camera config."
        )

    reference_frame = undistort_frame(read_first_frame(video_path), calibration)
    capture, backend_name, live_frame = open_camera(camera_cfg)
    cv2.namedWindow(WINDOW_NAME, cv2.WINDOW_NORMAL)

    print("Keys: O=overlay, L=live only, R=reference only, Esc/Q=quit.")
    display_mode = "overlay"
    frame_idx = 0
    previous_live_frame = None
    try:
        while True:
            live_frame = undistort_frame(live_frame, calibration)
            if live_frame.shape != reference_frame.shape:
                raise RuntimeError(
                    "Reference and live frame resolutions differ after calibration: "
                    f"reference={reference_frame.shape[1]}x{reference_frame.shape[0]}, "
                    f"live={live_frame.shape[1]}x{live_frame.shape[0]}. "
                    "Use a camera config matching the recorded cam0 resolution."
                )

            if previous_live_frame is None:
                frame_difference = 0.0
            else:
                difference = cv2.absdiff(live_frame, previous_live_frame)
                frame_difference = sum(cv2.mean(difference)[:3]) / 3.0
            previous_live_frame = live_frame.copy()

            if display_mode == "live":
                display = live_frame.copy()
            elif display_mode == "reference":
                display = reference_frame.copy()
            else:
                display = cv2.addWeighted(
                    reference_frame,
                    args.reference_alpha,
                    live_frame,
                    1.0 - args.reference_alpha,
                    0.0,
                )

            status = (
                f"LIVE #{frame_idx} {time.strftime('%H:%M:%S')} "
                f"diff={frame_difference:.2f} index={args.camera_index} "
                f"backend={backend_name} mode={display_mode}"
            )
            cv2.putText(
                display,
                status,
                (12, 28),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.55,
                (0, 255, 0),
                2,
                cv2.LINE_AA,
            )
            cv2.imshow(WINDOW_NAME, display)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord("q"), ord("Q")):
                break
            if key in (ord("o"), ord("O")):
                display_mode = "overlay"
            elif key in (ord("l"), ord("L")):
                display_mode = "live"
            elif key in (ord("r"), ord("R")):
                display_mode = "reference"

            frame_idx += 1
            ok, live_frame = capture.read()
            if not ok or live_frame is None:
                raise RuntimeError("Failed to read a new frame from the camera.")
    finally:
        capture.release()
        cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
