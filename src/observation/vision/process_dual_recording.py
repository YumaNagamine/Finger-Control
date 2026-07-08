from __future__ import annotations

import argparse
import csv
import datetime
import inspect
import sys
from pathlib import Path

import cv2
import deeplabcut
import numpy as np
import pandas as pd

# Ensure repository root is importable when running as a script.
ROOT = Path(__file__).resolve().parents[2]
if str(ROOT) not in sys.path:
    sys.path.append(str(ROOT))

from observation.vision.dlc_angle_processor import DLCAngleProcessor
from observation.vision.weight_displacement_processor import WeightDisplacementProcessor
from utils.config_loader import load_config
from utils.path_utils import resolve_path


SCRIPT_DIR = Path(__file__).resolve().parent
PROJECT_ROOT = ROOT.parent
DEFAULT_CONFIG = SCRIPT_DIR / "config_dual_camera_process.json"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Offline post-processing for dual-camera recordings.")
    parser.add_argument("--config", type=str, default=str(DEFAULT_CONFIG))
    parser.add_argument("--frame-limit", type=int, default=0, help="Stop after N pair rows (0 = unlimited).")
    return parser.parse_args()


def _resolve_input_path(raw_path: str, config_base_dir: Path) -> Path:
    path = resolve_path(raw_path, config_base_dir)
    if path is None:
        raise ValueError(f"Invalid input path: {raw_path!r}")
    if path.exists():
        return path
    alt = resolve_path(raw_path, PROJECT_ROOT)
    if alt is not None and alt.exists():
        return alt
    raise FileNotFoundError(f"Input path not found: {raw_path}")


def _resolve_output_dir(raw_path: str, config_base_dir: Path) -> Path:
    path = resolve_path(raw_path, PROJECT_ROOT)
    if path is None:
        path = resolve_path(raw_path, config_base_dir)
    if path is None:
        raise ValueError(f"Invalid output directory: {raw_path!r}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _prepare_header(angle_names: list[str], expected_num_weights: int) -> list[str]:
    header = [
        "pair_idx",
        "timestamp_iso",
        "elapsed_s",
        "cam0_frame_idx",
        "cam1_frame_idx",
        "cam0_ok",
        "cam1_ok",
        "dlc_inference_ms",
    ]
    header.extend(angle_names)
    for i in range(expected_num_weights):
        header.append(f"weight_{i}_disp_mm")
    for i in range(expected_num_weights):
        header.append(f"weight_{i}_disp_px")
    for i in range(expected_num_weights):
        header.append(f"weight_{i}_quality")
    return header


def _row_bool(value: str | int | None) -> bool:
    if value is None:
        return False
    return str(value).strip().lower() in {"1", "true", "yes"}


def _build_analyze_kwargs(inference_settings: dict, output_dir: Path) -> dict:
    signature = inspect.signature(deeplabcut.analyze_videos)
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

    kwargs: dict[str, object] = {}
    for key, value in requested.items():
        if value is None:
            continue
        if key in accepted:
            kwargs[key] = value
    return kwargs


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
        raise FileNotFoundError(
            f"Raw prediction .h5 file not found after analyze step. Looked in: {', '.join(str(d) for d in candidate_dirs)}"
        )

    return max(matches, key=lambda p: p.stat().st_mtime)


def _extract_bodyparts(df: pd.DataFrame) -> tuple[str, list[str]]:
    if not isinstance(df.columns, pd.MultiIndex) or df.columns.nlevels < 3:
        raise ValueError("Unsupported prediction format: expected MultiIndex with x/y data")

    scorer = str(df.columns.get_level_values(0)[0])
    all_bodyparts = df.columns.get_level_values(1)
    bodyparts = list(dict.fromkeys(str(bp) for bp in all_bodyparts))
    return scorer, bodyparts


def _prepare_keypoint_tracks(
    df: pd.DataFrame,
    scorer: str,
    bodyparts: list[str],
) -> list[tuple[str, np.ndarray, np.ndarray, np.ndarray]]:
    tracks: list[tuple[str, np.ndarray, np.ndarray, np.ndarray]] = []
    for bodypart in bodyparts:
        x_col = (scorer, bodypart, "x")
        y_col = (scorer, bodypart, "y")
        l_col = (scorer, bodypart, "likelihood")
        if x_col not in df.columns or y_col not in df.columns:
            continue

        x_values = df[x_col].to_numpy(copy=False)
        y_values = df[y_col].to_numpy(copy=False)
        if l_col in df.columns:
            likelihood_values = df[l_col].to_numpy(copy=False)
        else:
            likelihood_values = np.ones_like(x_values, dtype=np.float32)
        tracks.append((bodypart, x_values, y_values, likelihood_values))
    return tracks


def _draw_keypoints_video(
    video_path: Path,
    prediction_h5: Path,
    output_path: Path,
    pcutoff: float,
    dotsize: int,
    color_bgr: tuple[int, int, int],
    show_labels: bool,
) -> None:
    df = pd.read_hdf(prediction_h5)
    scorer, bodyparts = _extract_bodyparts(df)
    keypoint_tracks = _prepare_keypoint_tracks(df, scorer, bodyparts)

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

    max_frames = len(df.index)
    frame_index = 0
    try:
        while True:
            ok, frame = cap.read()
            if not ok or frame_index >= max_frames:
                break

            for bodypart, x_values, y_values, likelihood_values in keypoint_tracks:
                x = float(x_values[frame_index])
                y = float(y_values[frame_index])
                likelihood = float(likelihood_values[frame_index])

                if np.isnan(x) or np.isnan(y) or likelihood < pcutoff:
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
            frame_index += 1
    finally:
        cap.release()
        writer.release()


def _prepare_dlc_prediction_file(
    cam0_video_path: Path,
    inference_settings_path: Path,
) -> Path:
    settings = load_config(inference_settings_path, required_keys=("deeplabcut_config_path", "inference"))
    settings_dir = inference_settings_path.parent
    config_path_raw = settings.get("deeplabcut_config_path")
    if not isinstance(config_path_raw, str) or not config_path_raw.strip():
        raise ValueError("`deeplabcut_config_path` is required in DLC inference settings")
    dlc_project_config_path = resolve_path(config_path_raw, settings_dir)
    if dlc_project_config_path is None or not dlc_project_config_path.exists():
        raise FileNotFoundError(f"deeplabcut config not found: {dlc_project_config_path}")

    inference = settings["inference"]
    if not isinstance(inference, dict):
        raise ValueError("`inference` must be a JSON object")

    output_dir = cam0_video_path.parent / "dlc_inference"
    output_dir.mkdir(parents=True, exist_ok=True)

    analyze_kwargs = _build_analyze_kwargs(inference, output_dir)
    print(f"[analyze] config: {dlc_project_config_path}")
    print(f"[analyze] kwargs: {analyze_kwargs}")
    deeplabcut.analyze_videos(str(dlc_project_config_path), [str(cam0_video_path)], **analyze_kwargs)

    prediction_h5 = _find_prediction_file(cam0_video_path, [output_dir, cam0_video_path.parent])
    print(f"[analyze] prediction: {prediction_h5}")
    if bool(inference.get("save_raw_labeled_video", True)):
        raw_output_path = output_dir / f"{cam0_video_path.stem}_labeled.mp4"
        color_raw = inference.get("color_bgr", [0, 255, 0])
        if (
            not isinstance(color_raw, list)
            or len(color_raw) != 3
            or not all(isinstance(v, (int, float)) for v in color_raw)
        ):
            raise ValueError("`inference.color_bgr` must be [b, g, r]")
        color_bgr = tuple(int(max(0, min(255, v))) for v in color_raw)
        _draw_keypoints_video(
            video_path=cam0_video_path,
            prediction_h5=prediction_h5,
            output_path=raw_output_path,
            pcutoff=float(inference.get("pcutoff", 0.6)),
            dotsize=int(inference.get("dotsize", 6)),
            color_bgr=color_bgr,
            show_labels=bool(inference.get("show_labels", False)),
        )
        print(f"[analyze] labeled video: {raw_output_path}")
    return prediction_h5


def _load_prediction_tracks(prediction_h5_path: Path) -> tuple[str, dict[str, dict[str, np.ndarray]]]:
    df = pd.read_hdf(prediction_h5_path)
    scorer, bodyparts = _extract_bodyparts(df)
    tracks: dict[str, dict[str, np.ndarray]] = {}
    for bodypart in bodyparts:
        x_col = (scorer, bodypart, "x")
        y_col = (scorer, bodypart, "y")
        l_col = (scorer, bodypart, "likelihood")
        if x_col not in df.columns or y_col not in df.columns:
            continue
        x_values = df[x_col].to_numpy(copy=False)
        y_values = df[y_col].to_numpy(copy=False)
        if l_col in df.columns:
            likelihood_values = df[l_col].to_numpy(copy=False)
        else:
            likelihood_values = np.ones_like(x_values, dtype=np.float32)
        tracks[bodypart] = {
            "x": x_values,
            "y": y_values,
            "likelihood": likelihood_values,
        }
    return scorer, tracks


def process_dual_recording_config(
    cfg: dict,
    *,
    config_base_dir: Path,
    frame_limit: int = 0,
) -> dict[str, object]:
    input_cfg = dict(cfg["input"])
    processors_cfg = dict(cfg["processors"])
    output_cfg = dict(cfg["output"])

    cam0_video_path = _resolve_input_path(str(input_cfg["cam0_video_path"]), config_base_dir)
    cam1_video_path = _resolve_input_path(str(input_cfg["cam1_video_path"]), config_base_dir)
    pair_csv_path = _resolve_input_path(str(input_cfg["pair_timestamps_csv"]), config_base_dir)

    dlc_config_path = _resolve_input_path(str(processors_cfg["dlc_config_path"]), config_base_dir)
    dlc_inference_settings_path = _resolve_input_path(
        str(processors_cfg["dlc_inference_settings_path"]),
        config_base_dir,
    )
    weight_config_path = _resolve_input_path(str(processors_cfg["weight_config_path"]), config_base_dir)
    mm_per_pixel = float(processors_cfg.get("mm_per_pixel", 1.0))
    expected_num_weights = int(processors_cfg.get("expected_num_weights", 6))
    if expected_num_weights <= 0:
        raise ValueError("processors.expected_num_weights must be >= 1")

    dlc_config = load_config(
        dlc_config_path,
        required_keys=("dlc", "keypoints"),
    )
    weight_config = load_config(
        weight_config_path,
        required_keys=("processing", "measurement"),
    )
    measurement_cfg = dict(weight_config["measurement"])
    processing_cfg = dict(weight_config["processing"])
    measurement_cfg["num_weights"] = expected_num_weights

    dlc_processor = DLCAngleProcessor(dlc_config, dlc_config_path.parent, enable_live=False)
    weight_processor = WeightDisplacementProcessor(
        processing_cfg=processing_cfg,
        measurement_cfg=measurement_cfg,
        mm_per_pixel=mm_per_pixel,
    )

    prediction_h5_path = _prepare_dlc_prediction_file(cam0_video_path, dlc_inference_settings_path)
    _, prediction_tracks = _load_prediction_tracks(prediction_h5_path)

    cap0 = cv2.VideoCapture(str(cam0_video_path))
    cap1 = cv2.VideoCapture(str(cam1_video_path))
    if not cap0.isOpened():
        raise RuntimeError(f"Could not open cam0 video: {cam0_video_path}")
    if not cap1.isOpened():
        raise RuntimeError(f"Could not open cam1 video: {cam1_video_path}")

    csv_dir = _resolve_output_dir(str(output_cfg.get("csv_dir", "logs/dual_camera/processed_csv")), config_base_dir)
    overlay_video_dir = _resolve_output_dir(
        str(output_cfg.get("overlay_video_dir", "logs/dual_camera/processed_video")),
        config_base_dir,
    )
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    base_prefix = str(output_cfg.get("filename_prefix", "dual_processed")).strip()
    recording_name = cam0_video_path.parent.name.strip()
    if recording_name:
        filename_prefix = f"{base_prefix}_{recording_name}" if base_prefix else recording_name
    else:
        filename_prefix = base_prefix or "dual_processed"
    out_csv_path = csv_dir / f"{filename_prefix}_{ts}.csv"
    out_dlc_overlay_path = overlay_video_dir / f"{filename_prefix}_{ts}_cam0_dlc_overlay.mp4"
    out_weight_overlay_path = overlay_video_dir / f"{filename_prefix}_{ts}_cam1_weight_overlay.mp4"
    save_overlay_video = bool(output_cfg.get("save_overlay_video", True))
    writer_fourcc_str = str(output_cfg.get("writer_fourcc", "mp4v"))
    if len(writer_fourcc_str) != 4:
        raise ValueError("output.writer_fourcc must be a 4-character code, e.g. 'mp4v'.")
    writer_fourcc = cv2.VideoWriter_fourcc(*writer_fourcc_str)
    fallback_fps = float(output_cfg.get("fallback_fps", 30.0))
    cap0_fps = float(cap0.get(cv2.CAP_PROP_FPS))
    cap1_fps = float(cap1.get(cv2.CAP_PROP_FPS))
    out_fps0 = cap0_fps if cap0_fps > 0.0 else fallback_fps
    out_fps1 = cap1_fps if cap1_fps > 0.0 else fallback_fps
    writer_dlc_overlay: cv2.VideoWriter | None = None
    writer_weight_overlay: cv2.VideoWriter | None = None

    row_count = 0
    try:
        with pair_csv_path.open("r", newline="", encoding="utf-8") as f_pair, out_csv_path.open(
            "w", newline="", encoding="utf-8"
        ) as f_out:
            reader = csv.DictReader(f_pair)
            writer = csv.writer(f_out)
            writer.writerow(_prepare_header(dlc_processor.angle_names, expected_num_weights))

            for pair_row in reader:
                pair_idx = pair_row.get("pair_idx", "")
                timestamp_iso = pair_row.get("timestamp_iso", "")
                elapsed_s = pair_row.get("elapsed_s", "")
                cam0_ok = _row_bool(pair_row.get("cam0_ok"))
                cam1_ok = _row_bool(pair_row.get("cam1_ok"))
                cam0_frame_idx_txt = pair_row.get("cam0_frame_idx", "")
                cam1_frame_idx_txt = pair_row.get("cam1_frame_idx", "")

                dlc_result = None
                weight_results = None
                frame0 = None
                frame1 = None

                if cam0_ok:
                    ok0, frame0 = cap0.read()
                    if not ok0 or frame0 is None:
                        print("Stopping: cam0 video ended before pair CSV.")
                        break
                    frame0_idx = int(cam0_frame_idx_txt) if cam0_frame_idx_txt != "" else row_count
                    raw_keypoints: dict[str, tuple[float, float, float]] = {}
                    for name in dlc_processor.keypoint_names:
                        track = prediction_tracks.get(name)
                        if track is None or frame0_idx >= len(track["x"]):
                            raw_keypoints[name] = (float("nan"), float("nan"), 0.0)
                            continue
                        raw_keypoints[name] = (
                            float(track["x"][frame0_idx]),
                            float(track["y"][frame0_idx]),
                            float(track["likelihood"][frame0_idx]),
                        )
                    dlc_result, dlc_overlay = dlc_processor.process_keypoints(
                        frame0,
                        frame0_idx,
                        raw_keypoints,
                        inference_ms=None,
                    )
                else:
                    dlc_overlay = None

                if cam1_ok:
                    ok1, frame1 = cap1.read()
                    if not ok1 or frame1 is None:
                        print("Stopping: cam1 video ended before pair CSV.")
                        break
                    frame1_idx = int(cam1_frame_idx_txt) if cam1_frame_idx_txt != "" else row_count
                    weight_results, weight_overlay = weight_processor.process_frame(frame1, frame1_idx)
                else:
                    weight_overlay = None

                row: list[str | int] = [
                    pair_idx,
                    timestamp_iso,
                    elapsed_s,
                    cam0_frame_idx_txt,
                    cam1_frame_idx_txt,
                    int(cam0_ok),
                    int(cam1_ok),
                    "",
                ]

                angle_values = {name: float("nan") for name in dlc_processor.angle_names}
                if dlc_result is not None:
                    if dlc_result["inference_ms"] is not None:
                        row[-1] = f"{float(dlc_result['inference_ms']):.4f}"
                    angle_values.update(dlc_result["angles"])
                for angle_name in dlc_processor.angle_names:
                    val = angle_values.get(angle_name, float("nan"))
                    row.append("" if np.isnan(val) else f"{float(val):.6f}")

                weight_map: dict[int, dict] = {}
                if weight_results is not None:
                    for weight_row in weight_results:
                        weight_map[int(weight_row["weight_id"])] = weight_row

                for i in range(expected_num_weights):
                    wrow = weight_map.get(i)
                    if wrow is None or wrow["disp_mm"] is None:
                        row.append("")
                    else:
                        row.append(f"{float(wrow['disp_mm']):.6f}")

                for i in range(expected_num_weights):
                    wrow = weight_map.get(i)
                    if wrow is None or wrow["disp_px"] is None:
                        row.append("")
                    else:
                        row.append(f"{float(wrow['disp_px']):.6f}")

                for i in range(expected_num_weights):
                    wrow = weight_map.get(i)
                    row.append("" if wrow is None else str(wrow["quality"]))

                writer.writerow(row)

                if save_overlay_video:
                    vis0 = dlc_overlay if dlc_overlay is not None else frame0
                    vis1 = weight_overlay if weight_overlay is not None else frame1

                    if vis0 is not None:
                        if writer_dlc_overlay is None:
                            out_h0, out_w0 = vis0.shape[:2]
                            writer_dlc_overlay = cv2.VideoWriter(
                                str(out_dlc_overlay_path),
                                writer_fourcc,
                                out_fps0,
                                (out_w0, out_h0),
                            )
                            if not writer_dlc_overlay.isOpened():
                                raise RuntimeError(f"Failed to open DLC overlay video writer: {out_dlc_overlay_path}")
                        writer_dlc_overlay.write(vis0)

                    if vis1 is not None:
                        if writer_weight_overlay is None:
                            out_h1, out_w1 = vis1.shape[:2]
                            writer_weight_overlay = cv2.VideoWriter(
                                str(out_weight_overlay_path),
                                writer_fourcc,
                                out_fps1,
                                (out_w1, out_h1),
                            )
                            if not writer_weight_overlay.isOpened():
                                raise RuntimeError(
                                    f"Failed to open weight overlay video writer: {out_weight_overlay_path}"
                                )
                        writer_weight_overlay.write(vis1)

                row_count += 1

                if frame_limit > 0 and row_count >= frame_limit:
                    break
    finally:
        cap0.release()
        cap1.release()
        if writer_dlc_overlay is not None:
            writer_dlc_overlay.release()
        if writer_weight_overlay is not None:
            writer_weight_overlay.release()

    print(f"Saved processed CSV to {out_csv_path}")
    if save_overlay_video and writer_dlc_overlay is not None:
        print(f"Saved DLC overlay video to {out_dlc_overlay_path}")
    if save_overlay_video and writer_weight_overlay is not None:
        print(f"Saved weight overlay video to {out_weight_overlay_path}")
    return {
        "row_count": row_count,
        "csv_path": out_csv_path,
        "dlc_overlay_path": out_dlc_overlay_path if save_overlay_video and writer_dlc_overlay is not None else None,
        "weight_overlay_path": (
            out_weight_overlay_path if save_overlay_video and writer_weight_overlay is not None else None
        ),
        "recording_dir": cam0_video_path.parent,
    }


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_config(
        config_path,
        required_keys=("input", "processors", "output"),
    )
    process_dual_recording_config(
        cfg,
        config_base_dir=config_path.parent,
        frame_limit=args.frame_limit,
    )


if __name__ == "__main__":
    main()

