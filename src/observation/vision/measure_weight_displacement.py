from __future__ import annotations

import argparse
import csv
import datetime
import sys
import time
from collections import defaultdict
from pathlib import Path

import cv2

# File overview: short demo for ArUco-based displacement measurement with single camera and per-frame CSV output.

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
from observation.vision.weight_displacement_processor import WeightDisplacementProcessor
from utils.config_loader import load_config
from utils.path_utils import resolve_path


SCRIPT_DIR = Path(__file__).resolve().parent
DEFAULT_CONFIG = SCRIPT_DIR / "config_weight_displacement.json"
DEFAULT_CAMERA_CONFIG = resolve_param_path("camera_config.json")
DEFAULT_VIDEO_PATH = SCRIPT_DIR.parent / "camera" / "test_video" / "test_20260507_183626.mp4"

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Measure per-marker vertical displacement from ArUco markers.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--camera-config", type=str, default=str(DEFAULT_CAMERA_CONFIG))
    parser.add_argument("--input-mode", choices=["camera", "video"], default="video")
    parser.add_argument("--video", type=str, default=DEFAULT_VIDEO_PATH, help="Override video path when input mode is video.")
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


def _draw_detection_table(overlay: "cv2.typing.MatLike", results: list[dict]) -> "cv2.typing.MatLike":
    out = overlay.copy()
    rows = sorted(results, key=lambda r: int(r.get("marker_id", -1)))
    line_h = 18
    panel_w = 360
    panel_h = 24 + line_h * (len(rows) + 1)
    cv2.rectangle(out, (8, 8), (8 + panel_w, 8 + panel_h), (0, 0, 0), -1)
    cv2.rectangle(out, (8, 8), (8 + panel_w, 8 + panel_h), (120, 120, 120), 1)
    cv2.putText(
        out,
        "ID   x[px]    y[px]    disp[mm]",
        (16, 28),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1,
        cv2.LINE_AA,
    )
    for i, row in enumerate(rows):
        marker_id = int(row.get("marker_id", -1))
        x_px = float(row.get("x_px", 0.0))
        y_px = float(row.get("y_px", 0.0))
        disp_mm = float(row.get("disp_mm", 0.0))
        text = f"{marker_id:>2d}  {x_px:>7.1f}  {y_px:>7.1f}  {disp_mm:>+8.3f}"
        y = 28 + line_h * (i + 1)
        cv2.putText(out, text, (16, y), cv2.FONT_HERSHEY_SIMPLEX, 0.48, (220, 255, 220), 1, cv2.LINE_AA)
    return out


def _save_xy_plot(
    history: dict[int, dict[str, list[float]]],
    plot_path: Path,
) -> None:
    import matplotlib.pyplot as plt

    marker_ids = sorted(history.keys())
    if not marker_ids:
        return

    fig, axes = plt.subplots(2, 1, figsize=(12, 8), sharex=True)
    ax_x, ax_y = axes
    for marker_id in marker_ids:
        data = history[marker_id]
        frames = data["frame"]
        xs = data["x"]
        ys = data["y"]
        ax_x.plot(frames, xs, label=f"aruco{marker_id}")
        ax_y.plot(frames, ys, label=f"aruco{marker_id}")

    ax_x.set_title("Marker X Over Time")
    ax_x.set_ylabel("x [px]")
    ax_x.grid(True, alpha=0.3)
    ax_x.legend(ncol=3, fontsize=9)

    ax_y.set_title("Marker Y Over Time")
    ax_y.set_xlabel("frame")
    ax_y.set_ylabel("y [px]")
    ax_y.grid(True, alpha=0.3)
    ax_y.legend(ncol=3, fontsize=9)

    fig.tight_layout()
    plot_path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(plot_path, dpi=160)
    plt.close(fig)


def main() -> None:
    args = parse_args()

    config_path = Path(args.config).resolve()
    config = load_config(config_path, required_keys=("input", "processing", "measurement", "output"))
    input_cfg = config["input"]
    processing_cfg = config["processing"]
    measurement_cfg = config["measurement"]
    output_cfg = config["output"]

    if not args.video and input_cfg.get("video_path"):
        resolved_video = resolve_path(str(input_cfg.get("video_path")), config_path.parent)
        if resolved_video is not None:
            input_cfg = dict(input_cfg)
            input_cfg["video_path"] = str(resolved_video)

    camera_cfg = load_config(args.camera_config)
    mm_per_pixel = float(measurement_cfg.get("mm_per_pixel_fallback", 1.0))

    mode = (args.input_mode or input_cfg.get("mode", "camera")).strip().lower()
    if mode not in {"camera", "video"}:
        raise ValueError("Input mode must be either 'camera' or 'video'.")

    cap, source_name = _open_capture(mode, input_cfg, camera_cfg, args.video)
    calibration = setup_undistortion_from_config(camera_cfg, log_prefix="[weight-disp]") if mode == "camera" else None

    pipeline_name = str(measurement_cfg.get("pipeline", "weight_marker"))
    camera_id = str(measurement_cfg.get("camera_id", "cam0"))

    csv_dir = resolve_path(output_cfg.get("csv_dir", "output/weight_csv"), config_path.parent)
    video_dir = resolve_path(output_cfg.get("overlay_video_dir", "output/weight_video"), config_path.parent)
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
    ended_by_esc = False
    xy_history: dict[int, dict[str, list[float]]] = defaultdict(lambda: {"frame": [], "x": [], "y": []})

    if show_window:
        cv2.namedWindow("Weight Displacement", cv2.WINDOW_NORMAL)

    with csv_path.open("w", newline="", encoding="utf-8") as f_csv:
        writer_csv = csv.writer(f_csv)
        aruco_columns = [f"aruco{i}" for i in range(int(measurement_cfg.get("num_weights", 6)))]
        writer_csv.writerow(
            [
                "timestamp_iso",
                "elapsed_s",
                "frame_idx",
                "source",
                "camera_id",
                "pipeline",
                *aruco_columns,
            ]
        )

        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                frame = undistort_frame(frame, calibration)

                elapsed_s = time.perf_counter() - start_time
                results, overlay = processor.process_frame(frame, frame_idx=frame_idx)
                overlay = _draw_detection_table(overlay, results)
                now_iso = datetime.datetime.now().isoformat(timespec="milliseconds")

                disp_map = {int(row["marker_id"]): row.get("disp_mm") for row in results}
                for row in results:
                    marker_id = int(row["marker_id"])
                    xy_history[marker_id]["frame"].append(float(frame_idx))
                    xy_history[marker_id]["x"].append(float(row["x_px"]))
                    xy_history[marker_id]["y"].append(float(row["y_px"]))
                writer_csv.writerow(
                    [
                        now_iso,
                        f"{elapsed_s:.6f}",
                        frame_idx,
                        source_name,
                        camera_id,
                        pipeline_name,
                        *[
                            ""
                            if disp_map.get(i) is None
                            else f"{float(disp_map[i]):.4f}"
                            for i in range(len(aruco_columns))
                        ],
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
                        ended_by_esc = True
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
    if ended_by_esc or mode == "video":
        xy_plot_path = csv_dir / f"{filename_prefix}_{timestamp}_xy_plot.png"
        try:
            _save_xy_plot(xy_history, xy_plot_path)
            print(f"Saved marker XY plot to {xy_plot_path}")
        except ModuleNotFoundError:
            print("Could not save XY plot: matplotlib is not installed.")


if __name__ == "__main__":
    main()
