from __future__ import annotations

import cv2


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
