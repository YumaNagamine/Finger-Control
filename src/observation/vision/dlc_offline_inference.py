from __future__ import annotations

import inspect
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import cv2
import numpy as np
import pandas as pd

from utils.config_loader import load_config
from utils.path_utils import resolve_path


@dataclass(frozen=True)
class DLCPredictions:
    """Frame-indexed keypoint tracks loaded from a DeepLabCut HDF5 file."""

    scorer: str
    frame_count: int
    tracks: dict[str, dict[str, np.ndarray]]

    @property
    def bodyparts(self) -> tuple[str, ...]:
        return tuple(self.tracks)

    def keypoints_for_frame(
        self,
        frame_idx: int,
        keypoint_names: list[str],
        *,
        require_all: bool = False,
    ) -> dict[str, tuple[float, float, float]]:
        if frame_idx < 0 or frame_idx >= self.frame_count:
            raise IndexError(
                f"DLC prediction frame index out of range: {frame_idx} "
                f"(frame_count={self.frame_count})"
            )

        missing = [name for name in keypoint_names if name not in self.tracks]
        if require_all and missing:
            raise ValueError(f"DLC prediction is missing required keypoints: {missing}")

        keypoints: dict[str, tuple[float, float, float]] = {}
        for name in keypoint_names:
            track = self.tracks.get(name)
            if track is None:
                keypoints[name] = (float("nan"), float("nan"), 0.0)
                continue
            keypoints[name] = (
                float(track["x"][frame_idx]),
                float(track["y"][frame_idx]),
                float(track["likelihood"][frame_idx]),
            )
        return keypoints


def _import_deeplabcut() -> Any:
    try:
        import deeplabcut
    except Exception as exc:  # pragma: no cover - depends on the local DLC runtime
        raise RuntimeError(
            "Failed to import deeplabcut. Install the optional DLC dependencies "
            "before running offline video inference."
        ) from exc
    return deeplabcut


def _build_analyze_kwargs(analyze_videos: Any, inference_settings: dict, output_dir: Path) -> dict:
    signature = inspect.signature(analyze_videos)
    accepted = set(signature.parameters.keys())

    requested = {
        "shuffle": inference_settings.get("shuffle"),
        "trainingsetindex": inference_settings.get("trainingsetindex"),
        "videotype": inference_settings.get("videotype"),
        "save_as_csv": inference_settings.get("save_as_csv"),
        "gputouse": inference_settings.get("gputouse"),
        "batchsize": inference_settings.get("batchsize"),
        "cropping": inference_settings.get("cropping"),
        "dynamic": inference_settings.get("dynamic"),
        "auto_track": inference_settings.get("auto_track"),
        "n_tracks": inference_settings.get("n_tracks"),
        "robust_nframes": inference_settings.get("robust_nframes"),
        "snapshotindex": inference_settings.get("snapshotindex"),
    }
    if "destfolder" in accepted:
        requested["destfolder"] = str(output_dir)

    return {
        key: value
        for key, value in requested.items()
        if value is not None and key in accepted
    }


def _find_prediction_file(video_path: Path, candidate_dirs: list[Path]) -> Path:
    matches: list[Path] = []
    for directory in candidate_dirs:
        if not directory.exists():
            continue
        for candidate in directory.glob(f"{video_path.stem}*.h5"):
            if "_filtered" in candidate.stem:
                continue
            matches.append(candidate)

    if not matches:
        searched = ", ".join(str(directory) for directory in candidate_dirs)
        raise FileNotFoundError(
            f"Raw prediction .h5 file not found after analyze step. Looked in: {searched}"
        )

    return max(matches, key=lambda path: path.stat().st_mtime)


def _extract_bodyparts(df: pd.DataFrame) -> tuple[str, list[str]]:
    if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels < 3:
        raise ValueError("Unsupported prediction format: expected MultiIndex with x/y data")

    scorer = str(df.columns.get_level_values(0)[0])
    all_bodyparts = df.columns.get_level_values(1)
    bodyparts = list(dict.fromkeys(str(bodypart) for bodypart in all_bodyparts))
    return scorer, bodyparts


def load_dlc_predictions(prediction_h5_path: Path) -> DLCPredictions:
    df = pd.read_hdf(prediction_h5_path)
    scorer, bodyparts = _extract_bodyparts(df)
    frame_count = len(df.index)
    tracks: dict[str, dict[str, np.ndarray]] = {}

    for bodypart in bodyparts:
        x_col = (scorer, bodypart, "x")
        y_col = (scorer, bodypart, "y")
        likelihood_col = (scorer, bodypart, "likelihood")
        if x_col not in df.columns or y_col not in df.columns:
            continue

        x_values = df[x_col].to_numpy(copy=False)
        y_values = df[y_col].to_numpy(copy=False)
        if likelihood_col in df.columns:
            likelihood_values = df[likelihood_col].to_numpy(copy=False)
        else:
            likelihood_values = np.ones_like(x_values, dtype=np.float32)

        if not (
            len(x_values) == frame_count
            and len(y_values) == frame_count
            and len(likelihood_values) == frame_count
        ):
            raise ValueError(f"Inconsistent prediction track length for bodypart: {bodypart}")

        tracks[bodypart] = {
            "x": x_values,
            "y": y_values,
            "likelihood": likelihood_values,
        }

    if not tracks:
        raise ValueError(f"No x/y keypoint tracks found in prediction file: {prediction_h5_path}")

    return DLCPredictions(
        scorer=scorer,
        frame_count=frame_count,
        tracks=tracks,
    )


def _draw_raw_keypoints_video(
    video_path: Path,
    predictions: DLCPredictions,
    output_path: Path,
    pcutoff: float,
    dotsize: int,
    color_bgr: tuple[int, int, int],
    show_labels: bool,
) -> None:
    cap = cv2.VideoCapture(str(video_path))
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video: {video_path}")

    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = float(cap.get(cv2.CAP_PROP_FPS))
    if fps <= 0.0:
        fps = 30.0

    writer = cv2.VideoWriter(
        str(output_path),
        cv2.VideoWriter_fourcc(*"mp4v"),
        fps,
        (width, height),
    )
    if not writer.isOpened():
        cap.release()
        raise RuntimeError(f"Failed to open output video writer: {output_path}")

    frame_idx = 0
    try:
        while frame_idx < predictions.frame_count:
            ok, frame = cap.read()
            if not ok or frame is None:
                break

            raw_keypoints = predictions.keypoints_for_frame(
                frame_idx,
                list(predictions.bodyparts),
            )
            for bodypart, (x, y, likelihood) in raw_keypoints.items():
                if not np.isfinite(x) or not np.isfinite(y) or likelihood < pcutoff:
                    continue

                center = (int(round(x)), int(round(y)))
                cv2.circle(frame, center, dotsize, color_bgr, thickness=-1, lineType=cv2.LINE_AA)
                if show_labels:
                    cv2.putText(
                        frame,
                        bodypart,
                        (center[0] + 6, center[1] - 6),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.45,
                        color_bgr,
                        1,
                        cv2.LINE_AA,
                    )
            writer.write(frame)
            frame_idx += 1
    finally:
        cap.release()
        writer.release()


def prepare_dlc_prediction_file(
    video_path: Path,
    inference_settings_path: Path,
    *,
    output_dir: Path | None = None,
) -> Path:
    """Run batch DLC inference and return the generated raw HDF5 prediction path."""

    settings = load_config(
        inference_settings_path,
        required_keys=("deeplabcut_config_path", "inference"),
    )
    settings_dir = inference_settings_path.parent
    config_path_raw = settings.get("deeplabcut_config_path")
    if not isinstance(config_path_raw, str) or not config_path_raw.strip():
        raise ValueError("`deeplabcut_config_path` is required in DLC inference settings")
    dlc_project_config_path = resolve_path(config_path_raw, settings_dir)
    if dlc_project_config_path is None or not dlc_project_config_path.is_file():
        raise FileNotFoundError(f"deeplabcut config not found: {dlc_project_config_path}")

    inference = settings["inference"]
    if not isinstance(inference, dict):
        raise ValueError("`inference` must be a JSON object")

    resolved_output_dir = output_dir or (video_path.parent / "dlc_inference")
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    deeplabcut = _import_deeplabcut()
    analyze_kwargs = _build_analyze_kwargs(
        deeplabcut.analyze_videos,
        inference,
        resolved_output_dir,
    )
    print(f"[analyze] config: {dlc_project_config_path}")
    print(f"[analyze] kwargs: {analyze_kwargs}")
    deeplabcut.analyze_videos(
        str(dlc_project_config_path),
        [str(video_path)],
        **analyze_kwargs,
    )

    candidate_dirs = [resolved_output_dir]
    if "destfolder" not in analyze_kwargs:
        candidate_dirs.append(video_path.parent)
    prediction_h5 = _find_prediction_file(video_path, candidate_dirs)
    print(f"[analyze] prediction: {prediction_h5}")

    if bool(inference.get("save_raw_labeled_video", True)):
        color_raw = inference.get("color_bgr", [0, 255, 0])
        if (
            not isinstance(color_raw, list)
            or len(color_raw) != 3
            or not all(isinstance(value, (int, float)) for value in color_raw)
        ):
            raise ValueError("`inference.color_bgr` must be [b, g, r]")
        color_bgr = tuple(int(max(0, min(255, value))) for value in color_raw)
        raw_output_path = resolved_output_dir / f"{video_path.stem}_labeled.mp4"
        predictions = load_dlc_predictions(prediction_h5)
        _draw_raw_keypoints_video(
            video_path=video_path,
            predictions=predictions,
            output_path=raw_output_path,
            pcutoff=float(inference.get("pcutoff", 0.6)),
            dotsize=int(inference.get("dotsize", 6)),
            color_bgr=color_bgr,
            show_labels=bool(inference.get("show_labels", False)),
        )
        print(f"[analyze] labeled video: {raw_output_path}")

    return prediction_h5
