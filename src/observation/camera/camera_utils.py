from __future__ import annotations

import json
from pathlib import Path

import cv2
import numpy as np


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CALIBRATION_PATH = SCRIPT_DIR / "camera_calibration.json"
CalibrationParams = tuple[np.ndarray, np.ndarray]


def resolve_backend(backend_name: str | None) -> int | None:
    if not backend_name:
        return None
    return {
        "CAP_DSHOW": cv2.CAP_DSHOW,
        "CAP_ANY": cv2.CAP_ANY,
    }.get(backend_name)


def fourcc_from_str(code: str | None) -> int | None:
    if not code or len(code) != 4:
        return None
    return cv2.VideoWriter_fourcc(*code)


def apply_camera_settings(cap: cv2.VideoCapture, camera_cfg: dict) -> None:
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, int(camera_cfg.get("width", 1600)))
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, int(camera_cfg.get("height", 1200)))
    cap.set(cv2.CAP_PROP_FPS, float(camera_cfg.get("target_fps", 90)))
    cap.set(cv2.CAP_PROP_BUFFERSIZE, int(camera_cfg.get("buffersize", 0)))
    cap.set(cv2.CAP_PROP_AUTO_EXPOSURE, float(camera_cfg.get("auto_exposure", 1)))
    cap.set(cv2.CAP_PROP_GAIN, float(camera_cfg.get("gain", 0)))
    cap.set(cv2.CAP_PROP_EXPOSURE, float(camera_cfg.get("exposure", -11)))
    capture_fourcc = fourcc_from_str(camera_cfg.get("capture_fourcc", "MJPG"))
    if capture_fourcc is not None:
        cap.set(cv2.CAP_PROP_FOURCC, capture_fourcc)


def use_chessboard_calibration(camera_cfg: dict) -> bool:
    return bool(camera_cfg.get("use_chessboard_calibration", False))


def calibration_path_from_config(camera_cfg: dict) -> Path:
    raw_path = camera_cfg.get("calibration_path")
    if not raw_path:
        return DEFAULT_CALIBRATION_PATH

    path = Path(str(raw_path))
    if not path.is_absolute():
        path = SCRIPT_DIR / path
    return path


def load_chessboard_calibration(calibration_path: str | Path) -> CalibrationParams:
    path = Path(calibration_path)
    if not path.exists():
        raise FileNotFoundError(f"Calibration file not found: {path}")

    with path.open("r", encoding="utf-8") as f:
        data = json.load(f)

    if "camera_matrix" not in data or "distortion_coefficients" not in data:
        raise ValueError(
            f"Calibration file is missing required keys in {path}: "
            "'camera_matrix' and 'distortion_coefficients'"
        )

    camera_matrix = np.asarray(data["camera_matrix"], dtype=np.float64)
    distortion_coeffs = np.asarray(data["distortion_coefficients"], dtype=np.float64).reshape(-1, 1)

    if camera_matrix.shape != (3, 3):
        raise ValueError(f"camera_matrix must be 3x3 in {path}, got shape={camera_matrix.shape}")
    if distortion_coeffs.size < 4:
        raise ValueError(
            f"distortion_coefficients must contain at least 4 values in {path}, "
            f"got {distortion_coeffs.size}"
        )

    return camera_matrix, distortion_coeffs


def load_chessboard_calibration_from_config(camera_cfg: dict) -> CalibrationParams | None:
    if not use_chessboard_calibration(camera_cfg):
        return None

    calibration_path = calibration_path_from_config(camera_cfg)
    return load_chessboard_calibration(calibration_path)


def setup_undistortion_from_config(
    camera_cfg: dict,
    log_fn=print,
    log_prefix: str = "[camera]",
) -> CalibrationParams | None:
    if not use_chessboard_calibration(camera_cfg):
        return None

    calibration_path = calibration_path_from_config(camera_cfg)
    log_fn(f"{log_prefix} Chessboard calibration is enabled: {calibration_path}")
    try:
        calibration = load_chessboard_calibration(calibration_path)
    except Exception as exc:
        log_fn(f"{log_prefix} Failed to load chessboard calibration: {exc}")
        log_fn(f"{log_prefix} Distortion correction is disabled. Continuing without undistortion.")
        return None

    log_fn(f"{log_prefix} Distortion correction is enabled.")
    return calibration


def undistort_frame(frame_bgr: np.ndarray, calibration: CalibrationParams | None) -> np.ndarray:
    if calibration is None:
        return frame_bgr
    camera_matrix, distortion_coeffs = calibration
    return cv2.undistort(frame_bgr, camera_matrix, distortion_coeffs)
