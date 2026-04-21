from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
import time
from pathlib import Path

import cv2

# Ensure repository root is importable when running as a script.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from utils.config_loader import load_config


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CAMERA_CONFIG = SCRIPT_DIR / "camera_config.json"
DEFAULT_ARUCO_CONFIG = SCRIPT_DIR / "aruco_config.json"
DEFAULT_OUTPUT_CONFIG = SCRIPT_DIR / "scale_params.json"


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


def _apply_camera_settings(cap: cv2.VideoCapture, camera_cfg: dict) -> None:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(camera_cfg.get("width", 1600)))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(camera_cfg.get("height", 1200)))
    cap.set(cv2.CAP_PROP_FPS, float(camera_cfg.get("target_fps", 90)))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, int(camera_cfg.get("buffersize", 0)))
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, float(camera_cfg.get("auto_exposure", 1)))
    cap.set(cv2.CAP_PROP_GAIN, float(camera_cfg.get("gain", 0)))
    cap.set(cv2.CAP_PROP_EXPOSURE, float(camera_cfg.get("exposure", -11)))
    capture_fourcc = _fourcc_from_str(camera_cfg.get("capture_fourcc", "MJPG"))
    if capture_fourcc is not None:
        cap.set(cv2.CAP_PROP_FOURCC, capture_fourcc)


def _capture_single_frame(camera_cfg: dict, read_attempts: int = 30) -> tuple:
    cam_num = int(camera_cfg.get("index", 0))
    backend = _resolve_backend(camera_cfg.get("backend"))
    cap = cv2.VideoCapture(cam_num, backend) if backend is not None else cv2.VideoCapture(cam_num)
    if not cap.isOpened():
        raise RuntimeError("Camera not available.")

    _apply_camera_settings(cap, camera_cfg)

    frame = None
    try:
        for _ in range(max(1, read_attempts)):
            ret, maybe_frame = cap.read()
            if ret:
                frame = maybe_frame
                break
            time.sleep(0.03)
    finally:
        cap.release()

    if frame is None:
        raise RuntimeError("Failed to capture a frame from camera.")
    return frame, cam_num


def _get_aruco_module():
    aruco = getattr(cv2, "aruco", None)
    if aruco is None:
        raise RuntimeError(
            "cv2.aruco is not available. Install a build that includes ArUco (e.g. opencv-contrib-python)."
        )
    return aruco


def _detect_markers(gray_frame, dictionary_name: str):
    aruco = _get_aruco_module()

    if not hasattr(aruco, dictionary_name):
        raise ValueError(f"Unsupported ArUco dictionary: {dictionary_name}")

    dictionary = aruco.getPredefinedDictionary(getattr(aruco, dictionary_name))

    if hasattr(aruco, "ArucoDetector") and hasattr(aruco, "DetectorParameters"):
        params = aruco.DetectorParameters()
        detector = aruco.ArucoDetector(dictionary, params)
        corners, ids, _ = detector.detectMarkers(gray_frame)
    else:
        params = aruco.DetectorParameters_create()
        corners, ids, _ = aruco.detectMarkers(gray_frame, dictionary, parameters=params)

    return corners, ids


def _distance(p0, p1) -> float:
    dx = float(p0[0] - p1[0])
    dy = float(p0[1] - p1[1])
    return (dx * dx + dy * dy) ** 0.5


def _marker_edge_lengths(marker_corners) -> list[float]:
    pts = marker_corners.reshape(4, 2)
    return [_distance(pts[i], pts[(i + 1) % 4]) for i in range(4)]


def _build_failure_message(issues: list[str]) -> str:
    return "Calibration failed quality checks:\n- " + "\n- ".join(issues)


def _path_arg_or_default(cli_value: str | None, env_name: str, default_path: Path) -> Path:
    raw = cli_value if cli_value else os.environ.get(env_name)
    return Path(raw) if raw else default_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Calibrate mm/pixel from a single ArUco marker.")
    parser.add_argument("--camera-config", type=str, default=None)
    parser.add_argument("--aruco-config", type=str, default=None)
    parser.add_argument("--output", type=str, default=None, help="Output JSON path for calibrated scale.")
    return parser.parse_args()


def main() -> None:
    args = parse_args()

    camera_config_path = _path_arg_or_default(args.camera_config, "ARUCO_SCALE_CAMERA_CONFIG", DEFAULT_CAMERA_CONFIG)
    aruco_config_path = _path_arg_or_default(args.aruco_config, "ARUCO_SCALE_CONFIG", DEFAULT_ARUCO_CONFIG)
    output_path = _path_arg_or_default(args.output, "ARUCO_SCALE_OUTPUT_PATH", DEFAULT_OUTPUT_CONFIG)

    aruco_cfg = load_config(aruco_config_path, required_keys=("marker_length_mm", "marker_id", "dictionary"))

    input_mode = os.environ.get("ARUCO_SCALE_INPUT_MODE", "camera").strip().lower()
    source_info: dict[str, str | int]

    if input_mode == "camera":
        camera_cfg = load_config(camera_config_path)
        read_attempts = int(aruco_cfg.get("camera_read_attempts", 30))
        frame, camera_index = _capture_single_frame(camera_cfg, read_attempts=read_attempts)
        source_info = {"mode": "camera", "camera_index": camera_index, "camera_config_path": str(camera_config_path)}
        snapshot_path = os.environ.get("ARUCO_SCALE_SNAPSHOT_PATH")
        if snapshot_path:
            snapshot = Path(snapshot_path)
            snapshot.parent.mkdir(parents=True, exist_ok=True)
            cv2.imwrite(str(snapshot), frame)
    elif input_mode == "image":
        image_path_raw = os.environ.get("ARUCO_SCALE_IMAGE_PATH")
        if not image_path_raw:
            raise ValueError("ARUCO_SCALE_IMAGE_PATH is required when ARUCO_SCALE_INPUT_MODE=image.")
        image_path = Path(image_path_raw)
        frame = cv2.imread(str(image_path))
        if frame is None:
            raise FileNotFoundError(f"Could not read image: {image_path}")
        source_info = {"mode": "image", "image_path": str(image_path)}
    else:
        raise ValueError("ARUCO_SCALE_INPUT_MODE must be either 'camera' or 'image'.")

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    dictionary_name = str(aruco_cfg["dictionary"])
    marker_id = int(aruco_cfg["marker_id"])
    marker_length_mm = float(aruco_cfg["marker_length_mm"])

    corners, ids = _detect_markers(gray, dictionary_name)
    if ids is None or len(ids) == 0:
        raise RuntimeError("No ArUco markers were detected.")

    ids_flat = [int(x) for x in ids.flatten().tolist()]
    matched_indices = [i for i, detected_id in enumerate(ids_flat) if detected_id == marker_id]

    if not matched_indices:
        raise RuntimeError(f"Target marker id={marker_id} was not detected. Detected ids={ids_flat}")
    if len(matched_indices) > 1:
        raise RuntimeError(f"Marker id={marker_id} was detected more than once in one image.")

    edge_lengths = _marker_edge_lengths(corners[matched_indices[0]])
    edge_mean_px = sum(edge_lengths) / 4.0
    edge_min_px = min(edge_lengths)
    edge_max_px = max(edge_lengths)
    edge_ratio = (edge_max_px - edge_min_px) / edge_mean_px if edge_mean_px > 0 else float("inf")

    quality_cfg = aruco_cfg.get("quality", {})
    min_edge_px = float(quality_cfg.get("min_edge_px", 80.0))
    max_edge_ratio = float(quality_cfg.get("max_edge_ratio", 0.03))

    issues: list[str] = []
    if edge_mean_px <= 0:
        issues.append("Marker edge mean is zero or negative.")
    if edge_mean_px < min_edge_px:
        issues.append(f"Marker is too small in image: edge_mean_px={edge_mean_px:.2f} < min_edge_px={min_edge_px:.2f}")
    if edge_ratio > max_edge_ratio:
        issues.append(
            f"Edge length spread too large: edge_ratio={edge_ratio:.4f} > max_edge_ratio={max_edge_ratio:.4f}"
        )

    if issues:
        raise RuntimeError(_build_failure_message(issues))

    mm_per_pixel = marker_length_mm / edge_mean_px
    pixel_per_mm = edge_mean_px / marker_length_mm

    output = {
        "mm_per_pixel": mm_per_pixel,
        "pixel_per_mm": pixel_per_mm,
        "marker": {
            "id": marker_id,
            "length_mm": marker_length_mm,
            "dictionary": dictionary_name,
        },
        "quality": {
            "edge_lengths_px": edge_lengths,
            "edge_mean_px": edge_mean_px,
            "edge_min_px": edge_min_px,
            "edge_max_px": edge_max_px,
            "edge_ratio": edge_ratio,
            "min_edge_px_threshold": min_edge_px,
            "max_edge_ratio_threshold": max_edge_ratio,
        },
        "image": {
            "width": int(frame.shape[1]),
            "height": int(frame.shape[0]),
        },
        "source": source_info,
        "calibrated_at": datetime.datetime.now().isoformat(timespec="seconds"),
    }

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"Calibration succeeded. Saved scale parameters to {output_path}")


if __name__ == "__main__":
    main()
