# This script predicts tendon excursion from a processed csv file created by process_dual_recording.py.
from __future__ import annotations

import argparse
import datetime
import json
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Ensure src directory is importable when running as a script.
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from utils.path_utils import resolve_path


PROJECT_ROOT = SRC_ROOT.parent
MOMENT_ARM_AVERAGE_DIR = PROJECT_ROOT / "logs" / "dual_camera" / "moment_arm_average"
OUTPUT_DIR = PROJECT_ROOT / "logs" / "dual_camera" / "excursion_predictions"
TARGET_INPUT_CSV = (
    PROJECT_ROOT
    / "logs"
    / "dual_camera"
    / "processed_csv"
    / "dual_processed_controlTest_20260708_132620.csv"
)

TENDON_COLUMNS = ["FDP", "FDS", "EI", "DI", "PI", "LUM"]
JOINT_COLUMNS = ["DIP", "PIP", "MCP"]
FLEXION_RAD_COLUMNS = {
    "DIP": "DIP_flexion_angle_rad",
    "PIP": "PIP_flexion_angle_rad",
    "MCP": "MCP_flexion_angle_rad",
}
FLEXION_SMOOTHED_RAD_COLUMNS = {
    "DIP": "DIP_flexion_smoothed_angle_rad",
    "PIP": "PIP_flexion_smoothed_angle_rad",
    "MCP": "MCP_flexion_smoothed_angle_rad",
}
FLEXION_DELTA_RAD_COLUMNS = {
    "DIP": "DIP_flexion_delta_angle_rad",
    "PIP": "PIP_flexion_delta_angle_rad",
    "MCP": "MCP_flexion_delta_angle_rad",
}
DISP_MM_RENAME_MAP = {
    "weight_0_disp_mm": "FDP",
    "weight_1_disp_mm": "FDS",
    "weight_2_disp_mm": "EI",
    "weight_3_disp_mm": "PI",
    "weight_4_disp_mm": "DI",
    "weight_5_disp_mm": "LUM",
}
MOTION_NAMES = ("flexion", "extension")

ANGLE_SMOOTHING_WINDOW = 5
EXCURSION_SMOOTHING_WINDOW = 5
DIRECTION_DEADBAND_RAD = math.radians(0.2)
DEFAULT_DIRECTION = "flexion"



def _resolve_existing_file(raw_path: str, base_dir: Path) -> Path:
    path = resolve_path(raw_path, PROJECT_ROOT)
    if path is None:
        path = resolve_path(raw_path, base_dir)
    if path is None:
        raise ValueError(f"Invalid file path: {raw_path!r}")
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Expected file path: {path}")
    return path


def _validate_input_columns(df: pd.DataFrame) -> None:
    missing: list[str] = []
    if "elapsed_s" not in df.columns:
        missing.append("elapsed_s")

    for joint in JOINT_COLUMNS:
        if joint not in df.columns and FLEXION_RAD_COLUMNS[joint] not in df.columns:
            missing.append(f"{joint} or {FLEXION_RAD_COLUMNS[joint]}")

    for raw_column, tendon in DISP_MM_RENAME_MAP.items():
        if raw_column not in df.columns and tendon not in df.columns:
            missing.append(f"{raw_column} or {tendon}")

    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")


def _prepare_input_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    prepared = pd.DataFrame(index=df.index)
    prepared["elapsed_s"] = pd.to_numeric(df["elapsed_s"], errors="coerce")

    for raw_column, tendon in DISP_MM_RENAME_MAP.items():
        source_column = tendon if tendon in df.columns else raw_column
        prepared[tendon] = pd.to_numeric(df[source_column], errors="coerce")

    for joint in JOINT_COLUMNS:
        rad_column = FLEXION_RAD_COLUMNS[joint]
        if rad_column in df.columns:
            prepared[rad_column] = pd.to_numeric(df[rad_column], errors="coerce")
            continue

        joint_angle_deg = pd.to_numeric(df[joint], errors="coerce")
        flexion_angle_deg = 180.0 - joint_angle_deg
        prepared[rad_column] = flexion_angle_deg * math.pi / 180.0

    return prepared


def _load_moment_arm_coefficients() -> dict[tuple[str, str], dict[str, np.ndarray]]:
    coefficients: dict[tuple[str, str], dict[str, np.ndarray]] = {}

    for joint in JOINT_COLUMNS:
        for motion in MOTION_NAMES:
            json_path = MOMENT_ARM_AVERAGE_DIR / f"moment_arm_{joint}_{motion}_average.json"
            if not json_path.exists():
                raise FileNotFoundError(f"Moment arm JSON not found: {json_path}")

            payload = json.loads(json_path.read_text(encoding="utf-8"))
            tendon_coefficients = payload.get("moment_arm_coefficients")
            if not isinstance(tendon_coefficients, dict):
                raise ValueError(f"Invalid moment arm JSON format: {json_path}")

            motion_coefficients: dict[str, np.ndarray] = {}
            for tendon in TENDON_COLUMNS:
                raw_coefficients = tendon_coefficients.get(tendon)
                if not isinstance(raw_coefficients, list) or len(raw_coefficients) != 4:
                    raise ValueError(
                        f"Expected four moment arm coefficients for tendon={tendon!r} in {json_path}"
                    )
                motion_coefficients[tendon] = np.asarray(raw_coefficients, dtype=np.float64)

            coefficients[(joint, motion)] = motion_coefficients

    return coefficients


def _smooth_series(values: np.ndarray, window: int) -> np.ndarray:
    numeric = pd.Series(np.asarray(values, dtype=np.float64))
    smoothed = numeric.rolling(
        window=window,
        center=True,
        min_periods=1,
    ).mean()
    return smoothed.to_numpy(dtype=np.float64)


def _normalize_to_zero(values: np.ndarray) -> np.ndarray:
    normalized = np.asarray(values, dtype=np.float64).copy()
    finite_indices = np.flatnonzero(np.isfinite(normalized))
    if finite_indices.size == 0:
        return normalized
    normalized -= normalized[finite_indices[0]]
    return normalized


def _infer_motion_direction(
    smoothed_angles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, list[str]]:
    frame_count = len(smoothed_angles)
    delta_angles = np.full(frame_count, np.nan, dtype=np.float64)
    midpoint_angles = np.full(frame_count, np.nan, dtype=np.float64)
    directions = [DEFAULT_DIRECTION] * frame_count

    if frame_count == 0:
        return delta_angles, midpoint_angles, directions

    delta_angles[0] = 0.0
    midpoint_angles[0] = smoothed_angles[0]
    current_direction = DEFAULT_DIRECTION

    for frame_idx in range(1, frame_count):
        previous_angle = smoothed_angles[frame_idx - 1]
        current_angle = smoothed_angles[frame_idx]
        if not np.isfinite(previous_angle) or not np.isfinite(current_angle):
            directions[frame_idx] = current_direction
            continue

        delta_angle = current_angle - previous_angle
        delta_angles[frame_idx] = delta_angle
        midpoint_angles[frame_idx] = 0.5 * (previous_angle + current_angle)

        if delta_angle > DIRECTION_DEADBAND_RAD:
            current_direction = "flexion"
        elif delta_angle < -DIRECTION_DEADBAND_RAD:
            current_direction = "extension"

        directions[frame_idx] = current_direction

    return delta_angles, midpoint_angles, directions


def _build_joint_update_mask(prepared_df: pd.DataFrame) -> np.ndarray:
    frame_count = len(prepared_df)
    update_mask = np.zeros(frame_count, dtype=bool)

    if frame_count == 0:
        return update_mask

    current_valid = np.ones(frame_count, dtype=bool)
    previous_valid = np.ones(frame_count, dtype=bool)

    for joint in JOINT_COLUMNS:
        rad_values = pd.to_numeric(prepared_df[FLEXION_RAD_COLUMNS[joint]], errors="coerce").to_numpy(dtype=np.float64)
        joint_valid = np.isfinite(rad_values)
        current_valid &= joint_valid

        prev_joint_valid = np.zeros(frame_count, dtype=bool)
        if frame_count > 1:
            prev_joint_valid[1:] = joint_valid[:-1]
        previous_valid &= prev_joint_valid

    update_mask[0] = current_valid[0]
    if frame_count > 1:
        update_mask[1:] = current_valid[1:] & previous_valid[1:]
    return update_mask


def _build_enriched_dataframe(
    original_df: pd.DataFrame,
    prepared_df: pd.DataFrame,
    moment_arm_coefficients: dict[tuple[str, str], dict[str, np.ndarray]],
) -> pd.DataFrame:
    enriched_df = original_df.copy()
    joint_update_mask = _build_joint_update_mask(prepared_df)

    joint_delta_angles: dict[str, np.ndarray] = {}
    joint_midpoint_angles: dict[str, np.ndarray] = {}
    joint_directions: dict[str, list[str]] = {}

    for joint in JOINT_COLUMNS:
        raw_angles = pd.to_numeric(prepared_df[FLEXION_RAD_COLUMNS[joint]], errors="coerce").to_numpy(dtype=np.float64)
        smoothed_angles = _smooth_series(raw_angles, ANGLE_SMOOTHING_WINDOW)
        delta_angles, midpoint_angles, directions = _infer_motion_direction(smoothed_angles)

        delta_angles = delta_angles.copy()
        midpoint_angles = midpoint_angles.copy()
        delta_angles[~joint_update_mask] = np.nan
        midpoint_angles[~joint_update_mask] = np.nan

        if FLEXION_RAD_COLUMNS[joint] not in enriched_df.columns:
            enriched_df[FLEXION_RAD_COLUMNS[joint]] = raw_angles
        enriched_df[FLEXION_SMOOTHED_RAD_COLUMNS[joint]] = smoothed_angles
        enriched_df[FLEXION_DELTA_RAD_COLUMNS[joint]] = delta_angles

        joint_delta_angles[joint] = delta_angles
        joint_midpoint_angles[joint] = midpoint_angles
        joint_directions[joint] = directions

    for tendon in TENDON_COLUMNS:
        actual_excursion = pd.to_numeric(prepared_df[tendon], errors="coerce").to_numpy(dtype=np.float64)
        actual_excursion = _normalize_to_zero(actual_excursion)
        actual_excursion_smoothed = _smooth_series(actual_excursion, EXCURSION_SMOOTHING_WINDOW)
        actual_excursion_smoothed[~np.isfinite(actual_excursion)] = np.nan

        predicted_excursion = np.zeros(len(prepared_df), dtype=np.float64)
        if len(predicted_excursion) > 0:
            predicted_excursion[0] = 0.0

        for frame_idx in range(1, len(prepared_df)):
            previous_value = predicted_excursion[frame_idx - 1]
            if not joint_update_mask[frame_idx]:
                predicted_excursion[frame_idx] = previous_value
                continue

            total_delta = 0.0
            valid_frame = True

            for joint in JOINT_COLUMNS:
                delta_angle = joint_delta_angles[joint][frame_idx]
                midpoint_angle = joint_midpoint_angles[joint][frame_idx]
                if not np.isfinite(delta_angle) or not np.isfinite(midpoint_angle):
                    valid_frame = False
                    break

                motion = joint_directions[joint][frame_idx]
                coeffs = moment_arm_coefficients[(joint, motion)][tendon]
                total_delta += float(np.polyval(coeffs, midpoint_angle)) * delta_angle

            if valid_frame:
                predicted_excursion[frame_idx] = previous_value + total_delta
            else:
                predicted_excursion[frame_idx] = previous_value

        predicted_excursion = _normalize_to_zero(predicted_excursion)

        enriched_df[f"{tendon}_actual_excursion_mm"] = actual_excursion
        enriched_df[f"{tendon}_actual_excursion_smoothed_mm"] = actual_excursion_smoothed
        enriched_df[f"{tendon}_predicted_excursion_mm"] = predicted_excursion

    return enriched_df


def _plot_excursion_comparison(enriched_df: pd.DataFrame, output_path: Path) -> None:
    elapsed_s = pd.to_numeric(enriched_df["elapsed_s"], errors="coerce").to_numpy(dtype=np.float64)
    use_elapsed_time = np.isfinite(elapsed_s).all()
    x_values = elapsed_s if use_elapsed_time else np.arange(len(enriched_df), dtype=np.int32)
    x_label = "Elapsed time [s]" if use_elapsed_time else "Frame index"

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    axes_flat = axes.flatten()

    for axis, tendon in zip(axes_flat, TENDON_COLUMNS):
        actual = pd.to_numeric(
            enriched_df[f"{tendon}_actual_excursion_smoothed_mm"],
            errors="coerce",
        ).to_numpy(dtype=np.float64)
        predicted = pd.to_numeric(
            enriched_df[f"{tendon}_predicted_excursion_mm"],
            errors="coerce",
        ).to_numpy(dtype=np.float64)

        actual_valid = np.isfinite(actual)
        predicted_valid = np.isfinite(predicted)

        axis.plot(
            x_values[actual_valid],
            actual[actual_valid],
            linestyle="-",
            linewidth=2.0,
            label="Actual",
        )
        axis.plot(
            x_values[predicted_valid],
            predicted[predicted_valid],
            linestyle="--",
            linewidth=2.0,
            label="Predicted",
        )
        axis.set_title(tendon)
        axis.set_ylabel("Excursion [mm]")
        axis.grid(True, alpha=0.3)
        axis.legend()

    for axis in axes_flat[-2:]:
        axis.set_xlabel(x_label)

    fig.suptitle("Actual vs predicted tendon excursion", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _build_output_paths(input_csv_path: Path) -> tuple[Path, Path]:
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    suffix = datetime.datetime.now().strftime("prediction_%Y%m%d")
    output_stem = f"{input_csv_path.stem}_{suffix}"
    csv_path = OUTPUT_DIR / f"{output_stem}.csv"
    plot_path = OUTPUT_DIR / f"{output_stem}_predicted_vs_actual_excursion.png"
    return csv_path, plot_path


def process_csv(input_csv_path: Path) -> dict[str, Path]:
    input_df = pd.read_csv(input_csv_path)
    _validate_input_columns(input_df)
    prepared_df = _prepare_input_dataframe(input_df)
    moment_arm_coefficients = _load_moment_arm_coefficients()
    enriched_df = _build_enriched_dataframe(input_df, prepared_df, moment_arm_coefficients)

    output_csv_path, output_plot_path = _build_output_paths(input_csv_path)
    enriched_df.to_csv(output_csv_path, index=False)
    _plot_excursion_comparison(enriched_df, output_plot_path)

    return {
        "input_csv_path": input_csv_path,
        "output_csv_path": output_csv_path,
        "output_plot_path": output_plot_path,
    }


def main() -> None:
    input_csv_path = _resolve_existing_file(str(TARGET_INPUT_CSV), Path.cwd())
    result = process_csv(input_csv_path)

    print(f"Loaded input CSV: {result['input_csv_path']}")
    print(f"Saved prediction CSV: {result['output_csv_path']}")
    print(f"Saved comparison plot: {result['output_plot_path']}")


if __name__ == "__main__":
    main()


