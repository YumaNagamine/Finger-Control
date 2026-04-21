from __future__ import annotations

import argparse
import csv
import datetime
import sys
import time
from pathlib import Path

import cv2

# File overview: short demo for weight displacement measurement with single camera, and records per-weight time-series results in CSV

ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from observation.camera.camera_utils import apply_camera_settings, resolve_backend
from observation.vision.weight_displacement_processor import WeightDisplacementProcessor
from utils.config_loader import load_config


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "config_weight_displacement.json"
DEFAULT_CAMERA_CONFIG = SCRIPT_DIR.parent / "camera" / "camera_config.json"
DEFAULT_SCALE_CONFIG = SCRIPT_DIR.parent / "camera" / "scale_params.json"


def _resolve_path(raw_path: str | None, base_dir: Path) -> Path | None:
    if not raw_path:
        return None
    path = Path(raw_path)
    if path.is_absolute():
        return path
    return (base_dir / path).resolve()


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure per-weight vertical displacement from colored markers.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--camera-config", type=str, default=str(DEFAULT_CAMERA_CONFIG))
    parser.add_argument("--scale-config", type=str, default=str(DEFAULT_SCALE_CONFIG))
    parser.add_argument("--input-mode", choices=["camera", "video"], default=None)
    parser.add_argument("--video", type=str, default=None, help="Override video path when input mode is video.")
    parser.add_argument("--frame-limit", type=int, default=0, help="Stop after N frames (0 = unlimited).")
    parser.add_argument("--duration-s", type=float, default=0.0, help="Stop after elapsed seconds (0 = unlimited).")
    parser.add_argument("--no-window", action="store_true", help="Disable OpenCV preview window.")
    return parser.parse_args()


def _open_capture(mode: str, input_cfg: dict, camera_cfg: dict, video_override: str | None) -> tuple[cv2.VideoCapture, str]:
    if mode == "video":
        video_path = video_override or input_cfg.get("video_path")
        if not video_path:
            raise ValueError("video input requires either --video or input.video_path in config.")
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


def main() -> None:
    args = parse_args()

    config_path = Path(args.config).resolve()
    config = load_config(config_path, required_keys=("input", "processing", "measurement", "output"))
    input_cfg = config["input"]
    processing_cfg = config["processing"]
    measurement_cfg = config["measurement"]
    output_cfg = config["output"]

    if not args.video and input_cfg.get("video_path"):
        resolved_video = _resolve_path(str(input_cfg.get("video_path")), config_path.parent)
        if resolved_video is not None:
            input_cfg = dict(input_cfg)
            input_cfg["video_path"] = str(resolved_video)

    camera_cfg = load_config(args.camera_config)
    scale_cfg = load_config(args.scale_config, required_keys=("mm_per_pixel",))
    mm_per_pixel = float(scale_cfg["mm_per_pixel"])

    mode = (args.input_mode or input_cfg.get("mode", "camera")).strip().lower()
    if mode not in {"camera", "video"}:
        raise ValueError("Input mode must be either 'camera' or 'video'.")

    cap, source_name = _open_capture(mode, input_cfg, camera_cfg, args.video)

    pipeline_name = str(measurement_cfg.get("pipeline", "weight_marker"))
    camera_id = str(measurement_cfg.get("camera_id", "cam0"))

    csv_dir = _resolve_path(output_cfg.get("csv_dir", "output/weight_csv"), config_path.parent)
    video_dir = _resolve_path(output_cfg.get("overlay_video_dir", "output/weight_video"), config_path.parent)
    if csv_dir is None or video_dir is None:
        raise ValueError("output.csv_dir and output.overlay_video_dir must be provided.")
    csv_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)

    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_prefix = str(output_cfg.get("filename_prefix", "weight_disp"))
    csv_path = csv_dir / f"{filename_prefix}_{timestamp}.csv"
    overlay_video_path = video_dir / f"{filename_prefix}_{timestamp}.mp4"

    show_window = (not args.no_window) and bool(output_cfg.get("show_window", True))
    save_overlay_video = bool(output_cfg.get("save_overlay_video", True))

    processor = WeightDisplacementProcessor(
        processing_cfg=processing_cfg,
        measurement_cfg=measurement_cfg,
        mm_per_pixel=mm_per_pixel,
    )

    fps_from_cap = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    fallback_fps = float(output_cfg.get("fallback_fps", 30.0))
    output_fps = fps_from_cap if fps_from_cap > 0.0 else fallback_fps

    writer = None
    frame_idx = 0
    start_time = time.perf_counter()

    if show_window:
        cv2.namedWindow("Weight Displacement", cv2.WINDOW_NORMAL)

    with csv_path.open("w", newline="", encoding="utf-8") as f_csv:
        writer_csv = csv.writer(f_csv)
        writer_csv.writerow(
            [
                "timestamp_iso",
                "elapsed_s",
                "frame_idx",
                "source",
                "camera_id",
                "pipeline",
                "weight_id",
                "detected",
                "quality",
                "x_px",
                "y_px",
                "baseline_y_px",
                "disp_px",
                "disp_mm",
            ]
        )

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break

                elapsed_s = time.perf_counter() - start_time
                results, overlay = processor.process_frame(frame, frame_idx=frame_idx)
                now_iso = datetime.datetime.now().isoformat(timespec="milliseconds")

                for row in results:
                    writer_csv.writerow(
                        [
                            now_iso,
                            f"{elapsed_s:.6f}",
                            frame_idx,
                            source_name,
                            camera_id,
                            pipeline_name,
                            row["weight_id"],
                            int(bool(row["detected"])),
                            row["quality"],
                            "" if row["x_px"] is None else f"{row['x_px']:.2f}",
                            "" if row["y_px"] is None else f"{row['y_px']:.2f}",
                            "" if row["baseline_y_px"] is None else f"{row['baseline_y_px']:.2f}",
                            "" if row["disp_px"] is None else f"{row['disp_px']:.4f}",
                            "" if row["disp_mm"] is None else f"{row['disp_mm']:.4f}",
                        ]
                    )

                if save_overlay_video:
                    if writer is None:
                        h, w = overlay.shape[:2]
                        fourcc = cv2.VideoWriter_fourcc(*"mp4v")
                        writer = cv2.VideoWriter(str(overlay_video_path), fourcc, output_fps, (w, h))
                    writer.write(overlay)

                if show_window:
                    cv2.imshow("Weight Displacement", overlay)
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

    print(f"Saved displacement CSV to {csv_path}")
    if save_overlay_video:
        print(f"Saved overlay video to {overlay_video_path}")


if __name__ == "__main__":
    main()
