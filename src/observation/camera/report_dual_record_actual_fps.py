from __future__ import annotations

import csv
import statistics
from pathlib import Path

import cv2

# Set target session directory here.
SESSION_DIR = Path("logs/dual_camera/recordings/dual_record_20260507_132328")


def _load_elapsed_seconds(csv_path: Path) -> list[float]:
    if not csv_path.exists():
        raise FileNotFoundError(f"Timestamp CSV not found: {csv_path}")

    elapsed: list[float] = []
    with csv_path.open("r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        if "elapsed_s" not in (reader.fieldnames or []):
            raise ValueError(f"'elapsed_s' column is missing in {csv_path}")
        for row in reader:
            raw = str(row.get("elapsed_s", "")).strip()
            if raw == "":
                continue
            elapsed.append(float(raw))
    return elapsed


def _compute_actual_fps(elapsed: list[float]) -> dict[str, float | int | None]:
    if len(elapsed) < 2:
        return {
            "frame_count": len(elapsed),
            "duration_s": None,
            "fps_mean": None,
            "fps_median_inst": None,
            "fps_min_inst": None,
            "fps_max_inst": None,
        }

    deltas = [elapsed[i] - elapsed[i - 1] for i in range(1, len(elapsed))]
    positive_deltas = [d for d in deltas if d > 0.0]
    if not positive_deltas:
        return {
            "frame_count": len(elapsed),
            "duration_s": 0.0,
            "fps_mean": None,
            "fps_median_inst": None,
            "fps_min_inst": None,
            "fps_max_inst": None,
        }

    duration_s = elapsed[-1] - elapsed[0]
    fps_mean = (len(elapsed) - 1) / duration_s if duration_s > 0.0 else None
    inst_fps = [1.0 / d for d in positive_deltas]

    return {
        "frame_count": len(elapsed),
        "duration_s": duration_s,
        "fps_mean": fps_mean,
        "fps_median_inst": statistics.median(inst_fps),
        "fps_min_inst": min(inst_fps),
        "fps_max_inst": max(inst_fps),
    }


def _video_metadata(video_path: Path) -> dict[str, float | int | None]:
    if not video_path.exists():
        return {"header_fps": None, "frame_count": None, "duration_s": None}

    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        return {"header_fps": None, "frame_count": None, "duration_s": None}
    try:
        header_fps = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT) or 0)
        duration_s = (frame_count / header_fps) if header_fps > 0.0 else None
        return {
            "header_fps": header_fps if header_fps > 0.0 else None,
            "frame_count": frame_count if frame_count > 0 else None,
            "duration_s": duration_s,
        }
    finally:
        cap.release()


def _fmt(v: float | int | None, digits: int = 6) -> str:
    if v is None:
        return "n/a"
    if isinstance(v, int):
        return str(v)
    return f"{v:.{digits}f}"


def _report_camera(session_dir: Path, cam_name: str) -> None:
    ts_csv = session_dir / f"{cam_name}_timestamps.csv"
    video_path = session_dir / f"{cam_name}.mp4"

    elapsed = _load_elapsed_seconds(ts_csv)
    actual = _compute_actual_fps(elapsed)
    meta = _video_metadata(video_path)

    print(f"[{cam_name}]")
    print(f"  timestamp_csv: {ts_csv}")
    print(f"  video:         {video_path}")
    print(f"  frames(ts):    {_fmt(actual['frame_count'])}")
    print(f"  duration(ts):  {_fmt(actual['duration_s'])} s")
    print(f"  actual_fps:    {_fmt(actual['fps_mean'])}")
    print(f"  inst_fps_med:  {_fmt(actual['fps_median_inst'])}")
    print(f"  inst_fps_min:  {_fmt(actual['fps_min_inst'])}")
    print(f"  inst_fps_max:  {_fmt(actual['fps_max_inst'])}")
    print(f"  header_fps:    {_fmt(meta['header_fps'])}")
    print(f"  frames(video): {_fmt(meta['frame_count'])}")
    print(f"  duration(v):   {_fmt(meta['duration_s'])} s")
    if actual["fps_mean"] is not None and meta["header_fps"] is not None and actual["fps_mean"] > 0.0:
        speed_ratio = float(meta["header_fps"]) / float(actual["fps_mean"])
        print(f"  playback_ratio(header/actual): {_fmt(speed_ratio)}x")
    print()


def main() -> None:
    session_dir = SESSION_DIR.resolve()
    if "YYYYMMDD_HHMMSS" in str(SESSION_DIR):
        raise ValueError("Please update SESSION_DIR in this script before running.")
    if not session_dir.exists():
        raise FileNotFoundError(f"Session directory not found: {session_dir}")

    print(f"Session: {session_dir}")
    print()
    _report_camera(session_dir, "cam0")
    _report_camera(session_dir, "cam1")


if __name__ == "__main__":
    main()
