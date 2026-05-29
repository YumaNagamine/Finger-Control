from __future__ import annotations

import argparse
import csv
import datetime
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
from observation.camera.camera_param_resolver import resolve_param_path
from observation.vision.dlc_angle_processor import DLCAngleProcessor
from utils.config_loader import load_config
from utils.path_utils import resolve_path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "config_deeplabcut_angle.json"
DEFAULT_CAMERA_CONFIG = resolve_param_path("camera_config.json")


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Realtime keypoint/angle estimation with DeepLabCut.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--camera-config", type=str, default=str(DEFAULT_CAMERA_CONFIG))
    parser.add_argument("--input-mode", choices=["camera", "video"], default=None)
    parser.add_argument("--video", type=str, default=None, help="Override input.video_path in config.")
    parser.add_argument("--frame-limit", type=int, default=0, help="Stop after N frames (0 = unlimited).")
    parser.add_argument("--duration-s", type=float, default=0.0, help="Stop after elapsed seconds (0 = unlimited).")
    parser.add_argument("--no-window", action="store_true", help="Disable OpenCV preview window.")
    return parser.parse_args()


def _open_capture(mode: str, input_cfg: dict, camera_cfg: dict, video_override: str | None) -> tuple[cv2.VideoCapture, str]:
    if mode == "video":
        video_path = video_override or input_cfg.get("video_path")
        if not video_path:
            raise ValueError("video input requires --video or input.video_path in config.")
        cap = cv2.VideoCapture(str(video_path))
        if not cap.isOpened():
            raise RuntimeError(f"Could not open video: {video_path}")
        return cap, str(video_path)

    cam_index = int(camera_cfg.get("index", 0))
    backend = resolve_backend(camera_cfg.get("backend"))
    cap = cv2.VideoCapture(cam_index, backend) if backend is not None else cv2.VideoCapture(cam_index)
    if not cap.isOpened():
        raise RuntimeError("Camera not available.")
    apply_camera_settings(cap, camera_cfg)
    return cap, f"camera:{cam_index}"


def _prepare_csv_header(keypoint_names: list[str], angle_names: list[str]) -> list[str]:
    header = ["timestamp_iso", "elapsed_s", "frame_idx", "source", "inference_ms"]
    header.extend(angle_names)
    for name in keypoint_names:
        safe_name = name.replace(" ", "_")
        header.extend(
            [
                f"{safe_name}_x",
                f"{safe_name}_y",
                f"{safe_name}_likelihood",
                f"{safe_name}_status",
            ]
        )
    return header


def main() -> None:
    args = parse_args()

    config_path = Path(args.config).resolve()
    config = load_config(
        config_path,
        required_keys=("input", "dlc", "keypoints", "output"),
    )
    camera_cfg = load_config(args.camera_config)

    input_cfg = dict(config["input"])
    dlc_cfg = dict(config["dlc"])
    output_cfg = config["output"]

    if args.video is None and input_cfg.get("video_path"):
        resolved_video = resolve_path(str(input_cfg["video_path"]), config_path.parent)
        if resolved_video is not None:
            input_cfg["video_path"] = str(resolved_video)

    if dlc_cfg.get("third_party_path"):
        third_party_path = resolve_path(str(dlc_cfg["third_party_path"]), config_path.parent)
        if third_party_path is not None:
            dlc_cfg["third_party_path"] = str(third_party_path)
    if dlc_cfg.get("model_path"):
        model_path = resolve_path(str(dlc_cfg["model_path"]), config_path.parent)
        if model_path is not None:
            dlc_cfg["model_path"] = str(model_path)

    config_for_processor = dict(config)
    config_for_processor["dlc"] = dlc_cfg

    mode = (args.input_mode or input_cfg.get("mode", "camera")).strip().lower()
    if mode not in {"camera", "video"}:
        raise ValueError("Input mode must be either 'camera' or 'video'.")

    cap, source_name = _open_capture(mode, input_cfg, camera_cfg, args.video)
    calibration = setup_undistortion_from_config(camera_cfg, log_prefix="[dlc-angle]") if mode == "camera" else None

    csv_dir = resolve_path(output_cfg.get("csv_dir", "logs/deeplabcut_test/csv"), config_path.parent)
    video_dir = resolve_path(output_cfg.get("overlay_video_dir", "logs/deeplabcut_test/video"), config_path.parent)
    if csv_dir is None or video_dir is None:
        raise ValueError("output.csv_dir and output.overlay_video_dir must be provided.")
    csv_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_prefix = str(output_cfg.get("filename_prefix", "dlc_angle"))
    csv_path = csv_dir / f"{filename_prefix}_{timestamp}.csv"
    overlay_video_path = video_dir / f"{filename_prefix}_{timestamp}.mp4"

    save_overlay_video = bool(output_cfg.get("save_overlay_video", True))
    show_window = (not args.no_window) and bool(output_cfg.get("show_window", True))
    fallback_fps = float(output_cfg.get("fallback_fps", 30.0))
    fps_from_cap = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    output_fps = fps_from_cap if fps_from_cap > 0.0 else fallback_fps

    processor = DLCAngleProcessor(config=config_for_processor, config_base_dir=config_path.parent)
    header = _prepare_csv_header(processor.keypoint_names, processor.angle_names)

    writer = None
    frame_idx = 0
    start_time = time.perf_counter()
    if show_window:
        cv2.namedWindow("DLC Realtime Angle", cv2.WINDOW_NORMAL)

    with csv_path.open("w", newline="", encoding="utf-8") as f_csv:
        writer_csv = csv.writer(f_csv)
        writer_csv.writerow(header)

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = undistort_frame(frame, calibration)

                elapsed_s = time.perf_counter() - start_time
                result, overlay = processor.process_frame(frame_bgr=frame, frame_idx=frame_idx)

                row: list[str | int | float] = [
                    datetime.datetime.now().isoformat(timespec="milliseconds"),
                    f"{elapsed_s:.6f}",
                    frame_idx,
                    source_name,
                    f"{result['inference_ms']:.4f}",
                ]

                angles: dict[str, float] = result["angles"]
                for angle_name in processor.angle_names:
                    angle_val = angles.get(angle_name, float("nan"))
                    if np.isnan(angle_val):
                        row.append("")
                    else:
                        row.append(f"{angle_val:.6f}")

                keypoints: dict[str, dict[str, float | str | None]] = result["keypoints"]
                for name in processor.keypoint_names:
                    kp = keypoints[name]
                    x_val = kp["x"]
                    y_val = kp["y"]
                    likelihood = kp["likelihood"]
                    row.append("" if x_val is None else f"{float(x_val):.3f}")
                    row.append("" if y_val is None else f"{float(y_val):.3f}")
                    row.append("" if likelihood is None else f"{float(likelihood):.6f}")
                    row.append(str(kp["status"]))

                writer_csv.writerow(row)

                if save_overlay_video:
                    if writer is None:
                        h, w = overlay.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(str(overlay_video_path), fourcc, output_fps, (w, h))
                    writer.write(overlay)

                if show_window:
                    cv2.imshow("DLC Realtime Angle", overlay)
                    if cv2.waitKey(1) & 0xFF == 27:
                        break

                frame_idx += 1
                if args.frame_limit > 0 and frame_idx >= args.frame_limit:
                    break
                if args.duration_s > 0.0 and elapsed_s >= args.duration_s:
                    break
        finally:
            cap.release()
            if writer is not None:
                writer.release()
            if show_window:
                cv2.destroyAllWindows()

    print(f"Saved DLC angle CSV to {csv_path}")
    if save_overlay_video:
        print(f"Saved DLC overlay video to {overlay_video_path}")


if __name__ == "__main__":
    main()
