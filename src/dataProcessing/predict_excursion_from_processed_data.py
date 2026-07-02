from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import matplotlib.patheffects as pe
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Edit these paths in code when you want to change the input/output target.
TARGET_INPUT_CSV = (
    PROJECT_ROOT
    / "logs"
    / "dual_camera"
    / "processed_csv"
    / "dual_processed_momentarmValidation (6)_20260702_175638.csv" ### INPUT TARGET CSV FILENAME LOCATED IN processed_csv directory
)
MOMENT_ARM_AVERAGE_DIR = PROJECT_ROOT / "logs" / "dual_camera" / "moment_arm_average"
OUTPUT_DIR = PROJECT_ROOT / "logs" / "dual_camera" / "momentarm_validation"
OUTPUT_FILE_PREFIX = TARGET_INPUT_CSV.stem
OUTPUT_PLOT_PATH = OUTPUT_DIR / f"{OUTPUT_FILE_PREFIX}_predicted_vs_actual_excursion.png"
OUTPUT_CSV_PATH = OUTPUT_DIR / f"{OUTPUT_FILE_PREFIX}_predicted_vs_actual_excursion.csv"
OUTPUT_TIME_ANGLE_PLOT_PATH = OUTPUT_DIR / f"{OUTPUT_FILE_PREFIX}_time_vs_joint_angle.png"

TENDON_COLUMNS = ["FDP", "FDS", "EI", "DI", "PI", "LUM"]
JOINT_COLUMNS = ["DIP", "PIP", "MCP"]
FLEXION_RAD_COLUMNS = {
    "DIP": "DIP_flexion_angle_rad",
    "PIP": "PIP_flexion_angle_rad",
    "MCP": "MCP_flexion_angle_rad",
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


def _validate_processed_dataframe(df: pd.DataFrame) -> None:
    required_columns = ["elapsed_s", *TENDON_COLUMNS, *FLEXION_RAD_COLUMNS.values()]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Processed CSV is missing required columns: {missing}")


def _prepare_input_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared = prepared.rename(columns=DISP_MM_RENAME_MAP)

    for joint in JOINT_COLUMNS:
        rad_column = FLEXION_RAD_COLUMNS[joint]
        if rad_column in prepared.columns:
            continue
        if joint not in prepared.columns:
            raise ValueError(f"Input CSV is missing joint column required to derive {rad_column}: {joint}")
        joint_angle_deg = pd.to_numeric(prepared[joint], errors="coerce")
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


def _smooth_angle_series(series: pd.Series) -> np.ndarray:
    numeric = pd.to_numeric(series, errors="coerce")
    smoothed = numeric.rolling(
        window=ANGLE_SMOOTHING_WINDOW,
        center=True,
        min_periods=1,
    ).mean()
    return smoothed.to_numpy(dtype=np.float64)


def _smooth_excursion_values(values: np.ndarray) -> np.ndarray:
    numeric = pd.Series(np.asarray(values, dtype=np.float64))
    smoothed = numeric.rolling(
        window=EXCURSION_SMOOTHING_WINDOW,
        center=True,
        min_periods=1,
    ).mean()
    return smoothed.to_numpy(dtype=np.float64)


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


def _normalize_to_zero(values: np.ndarray) -> np.ndarray:
    normalized = np.asarray(values, dtype=np.float64).copy()
    finite_indices = np.flatnonzero(np.isfinite(normalized))
    if finite_indices.size == 0:
        return normalized
    normalized -= normalized[finite_indices[0]]
    return normalized


def _build_prediction_dataframe(
    df: pd.DataFrame,
    moment_arm_coefficients: dict[tuple[str, str], dict[str, np.ndarray]],
) -> pd.DataFrame:
    comparison_df = pd.DataFrame(index=df.index)
    comparison_df["elapsed_s"] = pd.to_numeric(df["elapsed_s"], errors="coerce")

    joint_delta_angles: dict[str, np.ndarray] = {}
    joint_midpoint_angles: dict[str, np.ndarray] = {}
    joint_directions: dict[str, list[str]] = {}

    for joint in JOINT_COLUMNS:
        smoothed_angles = _smooth_angle_series(df[FLEXION_RAD_COLUMNS[joint]])
        delta_angles, midpoint_angles, directions = _infer_motion_direction(smoothed_angles)

        comparison_df[f"{joint}_actual_angle_rad"] = pd.to_numeric(df[FLEXION_RAD_COLUMNS[joint]], errors="coerce")
        comparison_df[f"{joint}_actual_angle_deg"] = np.rad2deg(comparison_df[f"{joint}_actual_angle_rad"])
        comparison_df[f"{joint}_smoothed_angle_rad"] = smoothed_angles
        comparison_df[f"{joint}_smoothed_angle_deg"] = np.rad2deg(smoothed_angles)
        comparison_df[f"{joint}_delta_angle_rad"] = delta_angles
        comparison_df[f"{joint}_direction"] = directions

        joint_delta_angles[joint] = delta_angles
        joint_midpoint_angles[joint] = midpoint_angles
        joint_directions[joint] = directions

    for tendon in TENDON_COLUMNS:
        actual_excursion = pd.to_numeric(df[tendon], errors="coerce").to_numpy(dtype=np.float64)
        actual_excursion = _normalize_to_zero(actual_excursion)
        actual_excursion_smoothed = _smooth_excursion_values(actual_excursion)

        predicted_delta = np.full(len(df), np.nan, dtype=np.float64)
        predicted_cumulative = np.zeros(len(df), dtype=np.float64)

        for frame_idx in range(len(df)):
            total_delta = 0.0
            any_valid_joint = False

            for joint in JOINT_COLUMNS:
                delta_angle = joint_delta_angles[joint][frame_idx]
                midpoint_angle = joint_midpoint_angles[joint][frame_idx]
                if not np.isfinite(delta_angle) or not np.isfinite(midpoint_angle):
                    continue

                motion = joint_directions[joint][frame_idx]
                coeffs = moment_arm_coefficients[(joint, motion)][tendon]
                moment_arm = float(np.polyval(coeffs, midpoint_angle))
                total_delta += moment_arm * delta_angle
                any_valid_joint = True

            if any_valid_joint:
                predicted_delta[frame_idx] = total_delta

            if frame_idx == 0:
                continue

            previous_value = predicted_cumulative[frame_idx - 1]
            if np.isfinite(predicted_delta[frame_idx]):
                predicted_cumulative[frame_idx] = previous_value + predicted_delta[frame_idx]
            else:
                predicted_cumulative[frame_idx] = previous_value

        predicted_cumulative = _normalize_to_zero(predicted_cumulative)

        comparison_df[f"{tendon}_actual_excursion_mm"] = actual_excursion
        comparison_df[f"{tendon}_actual_excursion_smoothed_mm"] = actual_excursion_smoothed
        comparison_df[f"{tendon}_predicted_delta_mm"] = predicted_delta
        comparison_df[f"{tendon}_predicted_excursion_mm"] = predicted_cumulative
        comparison_df[f"{tendon}_error_mm"] = predicted_cumulative - actual_excursion_smoothed

    return comparison_df


def _plot_excursion_comparison(comparison_df: pd.DataFrame, output_path: Path) -> None:
    elapsed_s = comparison_df["elapsed_s"].to_numpy(dtype=np.float64)
    use_elapsed_time = np.isfinite(elapsed_s).all()
    x_values = elapsed_s if use_elapsed_time else np.arange(len(comparison_df), dtype=np.int32)
    x_label = "Elapsed time [s]" if use_elapsed_time else "Frame index"

    fig, axes = plt.subplots(3, 2, figsize=(14, 10), sharex=True)
    axes_flat = axes.flatten()

    for axis, tendon in zip(axes_flat, TENDON_COLUMNS):
        actual = comparison_df[f"{tendon}_actual_excursion_smoothed_mm"].to_numpy(dtype=np.float64)
        predicted = comparison_df[f"{tendon}_predicted_excursion_mm"].to_numpy(dtype=np.float64)

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


def _plot_time_vs_joint_angle(comparison_df: pd.DataFrame, output_path: Path) -> None:
    elapsed_s = comparison_df["elapsed_s"].to_numpy(dtype=np.float64)
    use_elapsed_time = np.isfinite(elapsed_s).all()
    x_values = elapsed_s if use_elapsed_time else np.arange(len(comparison_df), dtype=np.int32)
    x_label = "Elapsed time [s]" if use_elapsed_time else "Frame index"

    fig, ax = plt.subplots(figsize=(12, 6))
    line_colors = {
        ("DIP", "actual"): "#1f77b4",
        ("DIP", "smoothed"): "#17becf",
        ("PIP", "actual"): "#ff7f0e",
        ("PIP", "smoothed"): "#d62728",
        ("MCP", "actual"): "#2ca02c",
        ("MCP", "smoothed"): "#9467bd",
    }

    for joint in JOINT_COLUMNS:
        actual_deg = comparison_df[f"{joint}_actual_angle_deg"].to_numpy(dtype=np.float64)
        smoothed_deg = comparison_df[f"{joint}_smoothed_angle_deg"].to_numpy(dtype=np.float64)
        actual_color = line_colors[(joint, "actual")]
        smoothed_color = line_colors[(joint, "smoothed")]

        actual_valid = np.isfinite(actual_deg)
        smoothed_valid = np.isfinite(smoothed_deg)

        ax.plot(
            x_values[actual_valid],
            actual_deg[actual_valid],
            color=actual_color,
            linestyle="-",
            linewidth=2.2,
            alpha=0.95,
            label=f"{joint} actual",
        )
        smoothed_line, = ax.plot(
            x_values[smoothed_valid],
            smoothed_deg[smoothed_valid],
            color=smoothed_color,
            linestyle="--",
            linewidth=2.4,
            alpha=1.0,
            label=f"{joint} smoothed",
        )
        smoothed_line.set_path_effects([pe.Stroke(linewidth=3.6, foreground="white"), pe.Normal()])

    ax.set_xlabel(x_label)
    ax.set_ylabel("Flexion angle [deg]")
    ax.set_title("Time vs joint flexion angle")
    ax.grid(True, alpha=0.3)
    ax.legend(ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    if not TARGET_INPUT_CSV.exists():
        raise FileNotFoundError(f"Target input CSV not found: {TARGET_INPUT_CSV}")

    processed_df = pd.read_csv(TARGET_INPUT_CSV)
    processed_df = _prepare_input_dataframe(processed_df)
    _validate_processed_dataframe(processed_df)
    moment_arm_coefficients = _load_moment_arm_coefficients()
    comparison_df = _build_prediction_dataframe(processed_df, moment_arm_coefficients)

    OUTPUT_CSV_PATH.parent.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(OUTPUT_CSV_PATH, index=False)
    _plot_excursion_comparison(comparison_df, OUTPUT_PLOT_PATH)
    _plot_time_vs_joint_angle(comparison_df, OUTPUT_TIME_ANGLE_PLOT_PATH)

    print(f"Loaded input CSV: {TARGET_INPUT_CSV}")
    print(f"Saved comparison CSV: {OUTPUT_CSV_PATH}")
    print(f"Saved comparison plot: {OUTPUT_PLOT_PATH}")
    print(f"Saved time-angle plot: {OUTPUT_TIME_ANGLE_PLOT_PATH}")


if __name__ == "__main__":
    main()

