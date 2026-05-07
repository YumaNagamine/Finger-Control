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


def main() -> None:
    args = parse_args()
    config_path = Path(args.config).resolve()
    cfg = load_config(
        config_path,
        required_keys=("input", "processors", "output"),
    )
    input_cfg = dict(cfg["input"])
    processors_cfg = dict(cfg["processors"])
    output_cfg = dict(cfg["output"])

    cam0_video_path = _resolve_input_path(str(input_cfg["cam0_video_path"]), config_path.parent)
    cam1_video_path = _resolve_input_path(str(input_cfg["cam1_video_path"]), config_path.parent)
    pair_csv_path = _resolve_input_path(str(input_cfg["pair_timestamps_csv"]), config_path.parent)

    dlc_config_path = _resolve_input_path(str(processors_cfg["dlc_config_path"]), config_path.parent)
    weight_config_path = _resolve_input_path(str(processors_cfg["weight_config_path"]), config_path.parent)
    mm_per_pixel = float(processors_cfg.get("mm_per_pixel", 1.0))
    expected_num_weights = int(processors_cfg.get("expected_num_weights", 6))
    if expected_num_weights <= 0:
        raise ValueError("processors.expected_num_weights must be >= 1")

    dlc_config = load_config(
        dlc_config_path,
        required_keys=("dlc", "keypoints"),
    )
    dlc_cfg = dict(dlc_config["dlc"])
    if dlc_cfg.get("third_party_path"):
        third_party_path = resolve_path(str(dlc_cfg["third_party_path"]), dlc_config_path.parent)
        if third_party_path is not None:
            dlc_cfg["third_party_path"] = str(third_party_path)
    if dlc_cfg.get("model_path"):
        model_path = resolve_path(str(dlc_cfg["model_path"]), dlc_config_path.parent)
        if model_path is not None:
            dlc_cfg["model_path"] = str(model_path)
    dlc_config_for_processor = dict(dlc_config)
    dlc_config_for_processor["dlc"] = dlc_cfg

    weight_config = load_config(
        weight_config_path,
        required_keys=("processing", "measurement"),
    )
    measurement_cfg = dict(weight_config["measurement"])
    processing_cfg = dict(weight_config["processing"])
    measurement_cfg["num_weights"] = expected_num_weights

    dlc_processor = DLCAngleProcessor(dlc_config_for_processor, dlc_config_path.parent)
    weight_processor = WeightDisplacementProcessor(
        processing_cfg=processing_cfg,
        measurement_cfg=measurement_cfg,
        mm_per_pixel=mm_per_pixel,
    )

    cap0 = cv2.VideoCapture(str(cam0_video_path))
    cap1 = cv2.VideoCapture(str(cam1_video_path))
    if not cap0.isOpened():
        raise RuntimeError(f"Could not open cam0 video: {cam0_video_path}")
    if not cap1.isOpened():
        raise RuntimeError(f"Could not open cam1 video: {cam1_video_path}")

    csv_dir = _resolve_output_dir(str(output_cfg.get("csv_dir", "logs/dual_camera/processed_csv")), config_path.parent)
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    filename_prefix = str(output_cfg.get("filename_prefix", "dual_processed"))
    out_csv_path = csv_dir / f"{filename_prefix}_{ts}.csv"

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
                    dlc_result, _ = dlc_processor.process_frame(frame0, frame0_idx)

                if cam1_ok:
                    ok1, frame1 = cap1.read()
                    if not ok1 or frame1 is None:
                        print("Stopping: cam1 video ended before pair CSV.")
                        break
                    frame1_idx = int(cam1_frame_idx_txt) if cam1_frame_idx_txt != "" else row_count
                    weight_results, _ = weight_processor.process_frame(frame1, frame1_idx)

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
                row_count += 1

                if args.frame_limit > 0 and row_count >= args.frame_limit:
                    break
    finally:
        cap0.release()
        cap1.release()

    print(f"Saved processed CSV to {out_csv_path}")


if __name__ == "__main__":
    main()
