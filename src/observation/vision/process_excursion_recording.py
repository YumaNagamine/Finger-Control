"""Run offline DLC angle analysis for one play-excursion recording session.

Edit ``TARGET_SESSION_DIR`` below before running this file. The target must be a
session directory created by ``controller/csv_player/play_excursion_recording.py``.
"""

from __future__ import annotations

import csv
import datetime
import json
import math
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np

# Ensure the repository src directory is importable when running this file directly.
SRC_ROOT = Path(__file__).resolve().parents[2]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from observation.vision.dlc_angle_processor import DLCAngleProcessor
from observation.vision.dlc_offline_inference import (
    DLCPredictions,
    load_dlc_predictions,
    prepare_dlc_prediction_file,
)
from utils.config_loader import load_config


PROJECT_ROOT = SRC_ROOT.parent
SCRIPT_DIR = Path(__file__).resolve().parent

# ---------------------------------------------------------------------------
# User settings
# ---------------------------------------------------------------------------
TARGET_SESSION_DIR = (
    PROJECT_ROOT
    / "logs"
    / "play_excursion_recording"
    / "SET_SESSION_DIRECTORY_NAME"
)

DLC_ANGLE_CONFIG_PATH = SCRIPT_DIR / "config_deeplabcut_angle.json"
DLC_INFERENCE_SETTINGS_PATH = (
    PROJECT_ROOT
    / "third_party"
    / "robustMeasurementHighReso-2026-05-25"
    / "script"
    / "inference_settings.json"
)

ANALYSIS_DIR_NAME = "dlc_analysis"
WRITER_FOURCC = "mp4v"

VIDEO_FILENAME = "recording.mp4"
FRAME_TIMESTAMPS_FILENAME = "frame_timestamps.csv"
SESSION_MANIFEST_FILENAME = "session_manifest.json"


@dataclass(frozen=True)
class FrameTimestamp:
    frame_idx: int
    timestamp_iso: str
    elapsed_s: float


@dataclass(frozen=True)
class SessionInputs:
    session_dir: Path
    video_path: Path
    frame_timestamps_path: Path
    manifest_path: Path
    manifest: dict[str, Any]
    frame_timestamps: list[FrameTimestamp]


def _load_manifest(path: Path) -> dict[str, Any]:
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"Invalid session manifest JSON: {path}") from exc
    if not isinstance(payload, dict):
        raise ValueError(f"Session manifest must be a JSON object: {path}")
    return payload


def _load_frame_timestamps(path: Path) -> list[FrameTimestamp]:
    rows: list[FrameTimestamp] = []
    required_columns = {"frame_idx", "timestamp_iso", "elapsed_s"}

    with path.open("r", encoding="utf-8-sig", newline="") as file:
        reader = csv.DictReader(file)
        fieldnames = set(reader.fieldnames or [])
        missing_columns = sorted(required_columns - fieldnames)
        if missing_columns:
            raise ValueError(
                f"Frame timestamp CSV is missing required columns {missing_columns}: {path}"
            )

        previous_elapsed_s: float | None = None
        for row_number, row in enumerate(reader, start=2):
            try:
                frame_idx = int(row["frame_idx"])
                elapsed_s = float(row["elapsed_s"])
            except (TypeError, ValueError) as exc:
                raise ValueError(
                    f"Invalid frame_idx or elapsed_s at {path}:{row_number}"
                ) from exc

            expected_frame_idx = len(rows)
            if frame_idx != expected_frame_idx:
                raise ValueError(
                    f"Frame indices must be consecutive from zero: expected "
                    f"{expected_frame_idx}, got {frame_idx} at {path}:{row_number}"
                )
            if not math.isfinite(elapsed_s) or elapsed_s < 0.0:
                raise ValueError(
                    f"elapsed_s must be finite and non-negative at {path}:{row_number}"
                )
            if previous_elapsed_s is not None and elapsed_s < previous_elapsed_s:
                raise ValueError(
                    f"elapsed_s must be non-decreasing at {path}:{row_number}"
                )

            timestamp_iso = str(row["timestamp_iso"]).strip()
            if not timestamp_iso:
                raise ValueError(f"timestamp_iso is empty at {path}:{row_number}")

            rows.append(
                FrameTimestamp(
                    frame_idx=frame_idx,
                    timestamp_iso=timestamp_iso,
                    elapsed_s=elapsed_s,
                )
            )
            previous_elapsed_s = elapsed_s

    if not rows:
        raise ValueError(f"Frame timestamp CSV contains no frame rows: {path}")
    return rows


def _load_session_inputs(session_dir: Path) -> SessionInputs:
    resolved_session_dir = session_dir.expanduser().resolve()
    if resolved_session_dir.name == "SET_SESSION_DIRECTORY_NAME":
        raise ValueError(
            "Edit TARGET_SESSION_DIR in process_excursion_recording.py before running it."
        )
    if not resolved_session_dir.is_dir():
        raise NotADirectoryError(f"Session directory not found: {resolved_session_dir}")

    video_path = resolved_session_dir / VIDEO_FILENAME
    frame_timestamps_path = resolved_session_dir / FRAME_TIMESTAMPS_FILENAME
    manifest_path = resolved_session_dir / SESSION_MANIFEST_FILENAME
    for required_path in (video_path, frame_timestamps_path, manifest_path):
        if not required_path.is_file():
            raise FileNotFoundError(f"Required session file not found: {required_path}")

    manifest = _load_manifest(manifest_path)
    frame_timestamps = _load_frame_timestamps(frame_timestamps_path)

    saved_frames = manifest.get("saved_frames")
    if isinstance(saved_frames, bool) or not isinstance(saved_frames, int):
        raise ValueError(f"session_manifest.saved_frames must be an integer: {manifest_path}")
    if saved_frames != len(frame_timestamps):
        raise ValueError(
            "Session frame-count mismatch: "
            f"manifest saved_frames={saved_frames}, "
            f"frame_timestamps rows={len(frame_timestamps)}"
        )

    return SessionInputs(
        session_dir=resolved_session_dir,
        video_path=video_path,
        frame_timestamps_path=frame_timestamps_path,
        manifest_path=manifest_path,
        manifest=manifest,
        frame_timestamps=frame_timestamps,
    )


def _prepare_csv_header(keypoint_names: list[str], angle_names: list[str]) -> list[str]:
    header = ["frame_idx", "timestamp_iso", "elapsed_s"]
    header.extend(angle_names)
    header.extend(f"{name}_flexion_angle_deg" for name in angle_names)
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


def _numeric_cell(value: float | None, decimal_places: int = 6) -> str:
    if value is None or not np.isfinite(value):
        return ""
    return f"{float(value):.{decimal_places}f}"


def _build_csv_row(
    timestamp: FrameTimestamp,
    result: dict[str, Any],
    keypoint_names: list[str],
    angle_names: list[str],
) -> list[str | int]:
    row: list[str | int] = [
        timestamp.frame_idx,
        timestamp.timestamp_iso,
        f"{timestamp.elapsed_s:.6f}",
    ]

    angles: dict[str, float] = result["angles"]
    for name in angle_names:
        row.append(_numeric_cell(angles.get(name)))
    for name in angle_names:
        angle = angles.get(name)
        flexion_angle = None if angle is None else 180.0 - float(angle)
        row.append(_numeric_cell(flexion_angle))

    keypoints: dict[str, dict[str, float | str | None]] = result["keypoints"]
    for name in keypoint_names:
        keypoint = keypoints[name]
        row.append(_numeric_cell(keypoint["x"], decimal_places=3))
        row.append(_numeric_cell(keypoint["y"], decimal_places=3))
        row.append(_numeric_cell(keypoint["likelihood"]))
        row.append(str(keypoint["status"]))
    return row


def _resolve_output_fps(cap: cv2.VideoCapture, manifest: dict[str, Any]) -> float:
    fps_from_video = float(cap.get(cv2.CAP_PROP_FPS) or 0.0)
    if math.isfinite(fps_from_video) and fps_from_video > 0.0:
        return fps_from_video

    camera_config = manifest.get("camera_config")
    if isinstance(camera_config, dict):
        raw_target_fps = camera_config.get("target_fps")
        if isinstance(raw_target_fps, (int, float)) and not isinstance(raw_target_fps, bool):
            target_fps = float(raw_target_fps)
            if math.isfinite(target_fps) and target_fps > 0.0:
                return target_fps

    raise ValueError(
        "Video FPS is unavailable and session_manifest.camera_config.target_fps "
        "does not provide a valid value."
    )


def _validate_predictions(
    predictions: DLCPredictions,
    frame_timestamps: list[FrameTimestamp],
    keypoint_names: list[str],
) -> None:
    if predictions.frame_count != len(frame_timestamps):
        raise ValueError(
            "DLC prediction frame-count mismatch: "
            f"predictions={predictions.frame_count}, "
            f"frame_timestamps={len(frame_timestamps)}"
        )
    missing_keypoints = [name for name in keypoint_names if name not in predictions.tracks]
    if missing_keypoints:
        raise ValueError(f"DLC prediction is missing required keypoints: {missing_keypoints}")


def _write_analysis_manifest(
    path: Path,
    *,
    inputs: SessionInputs,
    prediction_h5_path: Path,
    predictions: DLCPredictions,
    output_csv_path: Path,
    output_overlay_path: Path,
    output_fps: float,
) -> None:
    payload = {
        "schema_version": 1,
        "created_at": datetime.datetime.now().isoformat(timespec="seconds"),
        "session_dir": str(inputs.session_dir),
        "frame_count": len(inputs.frame_timestamps),
        "output_fps": output_fps,
        "dlc_scorer": predictions.scorer,
        "dlc_bodyparts": list(predictions.bodyparts),
        "inputs": {
            "video": str(inputs.video_path),
            "frame_timestamps": str(inputs.frame_timestamps_path),
            "session_manifest": str(inputs.manifest_path),
            "dlc_angle_config": str(DLC_ANGLE_CONFIG_PATH.resolve()),
            "dlc_inference_settings": str(DLC_INFERENCE_SETTINGS_PATH.resolve()),
            "prediction_h5": str(prediction_h5_path.resolve()),
        },
        "outputs": {
            "angles_csv": str(output_csv_path.resolve()),
            "overlay_video": str(output_overlay_path.resolve()),
        },
    }
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def process_excursion_recording(session_dir: Path) -> dict[str, Path | int]:
    inputs = _load_session_inputs(session_dir)
    if not DLC_ANGLE_CONFIG_PATH.is_file():
        raise FileNotFoundError(f"DLC angle config not found: {DLC_ANGLE_CONFIG_PATH}")
    if not DLC_INFERENCE_SETTINGS_PATH.is_file():
        raise FileNotFoundError(
            f"DLC inference settings not found: {DLC_INFERENCE_SETTINGS_PATH}"
        )
    if len(WRITER_FOURCC) != 4:
        raise ValueError("WRITER_FOURCC must contain exactly four characters")

    dlc_config = load_config(
        DLC_ANGLE_CONFIG_PATH,
        required_keys=("dlc", "keypoints"),
    )
    dlc_processor = DLCAngleProcessor(
        dlc_config,
        DLC_ANGLE_CONFIG_PATH.parent,
        enable_live=False,
    )

    analysis_timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    analysis_dir = inputs.session_dir / ANALYSIS_DIR_NAME / analysis_timestamp
    inference_dir = analysis_dir / "dlc_inference"
    analysis_dir.mkdir(parents=True, exist_ok=False)

    prediction_h5_path = prepare_dlc_prediction_file(
        inputs.video_path,
        DLC_INFERENCE_SETTINGS_PATH,
        output_dir=inference_dir,
    )
    predictions = load_dlc_predictions(prediction_h5_path)
    _validate_predictions(
        predictions,
        inputs.frame_timestamps,
        dlc_processor.keypoint_names,
    )

    output_csv_path = analysis_dir / "dlc_angles.csv"
    output_overlay_path = analysis_dir / "dlc_overlay.mp4"
    analysis_manifest_path = analysis_dir / "analysis_manifest.json"

    cap = cv2.VideoCapture(str(inputs.video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Could not open recorded video: {inputs.video_path}")
    output_fps = _resolve_output_fps(cap, inputs.manifest)

    overlay_writer: cv2.VideoWriter | None = None
    processed_frames = 0
    try:
        with output_csv_path.open("w", encoding="utf-8", newline="") as csv_file:
            csv_writer = csv.writer(csv_file)
            csv_writer.writerow(
                _prepare_csv_header(
                    dlc_processor.keypoint_names,
                    dlc_processor.angle_names,
                )
            )

            for timestamp in inputs.frame_timestamps:
                ok, frame = cap.read()
                if not ok or frame is None:
                    raise RuntimeError(
                        f"Recorded video ended before frame {timestamp.frame_idx}: "
                        f"{inputs.video_path}"
                    )

                raw_keypoints = predictions.keypoints_for_frame(
                    timestamp.frame_idx,
                    dlc_processor.keypoint_names,
                    require_all=True,
                )
                result, overlay = dlc_processor.process_keypoints(
                    frame,
                    timestamp.frame_idx,
                    raw_keypoints,
                    inference_ms=None,
                )

                csv_writer.writerow(
                    _build_csv_row(
                        timestamp,
                        result,
                        dlc_processor.keypoint_names,
                        dlc_processor.angle_names,
                    )
                )

                if overlay_writer is None:
                    height, width = overlay.shape[:2]
                    overlay_writer = cv2.VideoWriter(
                        str(output_overlay_path),
                        cv2.VideoWriter_fourcc(*WRITER_FOURCC),
                        output_fps,
                        (width, height),
                    )
                    if not overlay_writer.isOpened():
                        raise RuntimeError(
                            f"Failed to open overlay video writer: {output_overlay_path}"
                        )
                overlay_writer.write(overlay)
                processed_frames += 1

            has_extra_frame, _ = cap.read()
            if has_extra_frame:
                raise ValueError(
                    "Recorded video contains more frames than frame_timestamps.csv"
                )
    finally:
        cap.release()
        if overlay_writer is not None:
            overlay_writer.release()

    if processed_frames != len(inputs.frame_timestamps):
        raise RuntimeError(
            f"Processed frame-count mismatch: processed={processed_frames}, "
            f"expected={len(inputs.frame_timestamps)}"
        )

    _write_analysis_manifest(
        analysis_manifest_path,
        inputs=inputs,
        prediction_h5_path=prediction_h5_path,
        predictions=predictions,
        output_csv_path=output_csv_path,
        output_overlay_path=output_overlay_path,
        output_fps=output_fps,
    )

    print(f"Saved DLC angle CSV to {output_csv_path}")
    print(f"Saved DLC overlay video to {output_overlay_path}")
    print(f"Saved analysis manifest to {analysis_manifest_path}")
    return {
        "frame_count": processed_frames,
        "analysis_dir": analysis_dir,
        "csv_path": output_csv_path,
        "overlay_path": output_overlay_path,
        "manifest_path": analysis_manifest_path,
        "prediction_h5_path": prediction_h5_path,
    }


def main() -> None:
    process_excursion_recording(TARGET_SESSION_DIR)


if __name__ == "__main__":
    main()
