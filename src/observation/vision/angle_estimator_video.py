from __future__ import annotations

import argparse
import csv
import datetime
import sys
from pathlib import Path

import cv2
import numpy as np

# Ensure repository root is importable when running as a script.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from observation.vision.angle_processor import AngleProcessor
from utils.config_loader import load_config


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Estimate finger joint angles from a video.")
    parser.add_argument("--config", type=str, default=str(Path(__file__).with_name("config.json")))
    parser.add_argument("--video", type=str, help="Override input video path in config.json.")
    parser.add_argument("--frame-limit", type=int, default=0, help="Process at most N frames (0 = all).")
    parser.add_argument("--no-window", action="store_true", help="Disable OpenCV preview window.")
    return parser.parse_args()


def ensure_dirs(output_cfg: dict) -> tuple[Path, Path]:
    csv_dir = Path(output_cfg["csv_dir"])
    video_dir = Path(output_cfg["video_dir"])
    csv_dir.mkdir(parents=True, exist_ok=True)
    video_dir.mkdir(parents=True, exist_ok=True)
    return csv_dir, video_dir


def store_video(frames: list, fps: float, output_path: Path) -> None:
    if not frames:
        return
    height, width, _ = frames[0].shape
    fourcc = cv2.VideoWriter_fourcc(*"mp4v")
    writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
    for frame in frames:
        writer.write(frame)
    writer.release()


def store_csv(measure: list, fps: float, output_path: Path) -> None:
    with output_path.open("w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(["frame", "angle_0", "angle_1", "angle_2", "time"])
        for row in measure:
            frame, angle_0, angle_1, angle_2 = row
            time_s = float(frame) / float(fps)
            writer.writerow([frame, angle_0, angle_1, angle_2, time_s])


def main() -> None:
    args = parse_args()
    config = load_config(args.config, required_keys=("video", "markers", "output"))

    video_cfg = config["video"]
    output_cfg = config["output"]

    video_path = Path(args.video) if args.video else Path(video_cfg["input_path"])
    if not video_path.exists():
        raise FileNotFoundError(f"Video not found: {video_path}")

    csv_dir, video_dir = ensure_dirs(output_cfg)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = csv_dir / f"{timestamp}.csv"
    video_out_path = video_dir / f"{timestamp}.mp4"

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open video: {video_path}")

    frame_shift = int(video_cfg.get("frame_shift", 0))
    frame_jump = int(video_cfg.get("frame_jump", 0))
    output_fps = float(video_cfg.get("output_fps", cap.get(cv2.CAP_PROP_FPS) or 30))

    cap.set(cv2.CAP_PROP_POS_FRAMES, frame_shift)
    cap.set(cv2.CAP_PROP_FPS, output_fps)

    processor = AngleProcessor(config)

    frames_to_store: list[np.ndarray] = []
    measurements: list[list[float]] = []
    frame_idx = frame_shift

    window_name = "Angle Estimation"
    show_window = not args.no_window and video_cfg.get("show_window", True)
    if show_window:
        cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                break

            if frame_jump and (frame_idx - frame_shift) % frame_jump != 0:
                frame_idx += 1
                continue

            lab_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2Lab)
            raw_markers = processor.detect_markers(lab_frame)
            modified = processor.modify_markers(raw_markers)
            joints = processor.estimate_joints(modified)
            angle_0, angle_1, angle_2 = processor.calculate_angles(modified)
            annotated = processor.draw_overlays(frame, raw_markers, modified, joints)

            overlay = annotated.copy()
            overlay = cv2.putText(overlay, f"frame: {frame_idx}", (40, 60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 0), 2)
            overlay = cv2.putText(overlay, f"angle0: {angle_0:.1f}", (40, 90), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            overlay = cv2.putText(overlay, f"angle1: {angle_1:.1f}", (40, 120), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)
            overlay = cv2.putText(overlay, f"angle2: {angle_2:.1f}", (40, 150), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 0), 2)

            frames_to_store.append(overlay)
            measurements.append([frame_idx, angle_0, angle_1, angle_2])

            if show_window:
                cv2.imshow(window_name, overlay)
                if cv2.waitKey(1) & 0xFF == 27:
                    break

            frame_idx += 1
            if args.frame_limit and len(measurements) >= args.frame_limit:
                break
    finally:
        cap.release()
        if show_window:
            cv2.destroyAllWindows()

    store_video(frames_to_store, output_fps, video_out_path)
    store_csv(measurements, output_fps, csv_path)
    print(f"Saved annotated video to {video_out_path}")
    print(f"Saved angles to {csv_path}")


if __name__ == "__main__":
    main()
