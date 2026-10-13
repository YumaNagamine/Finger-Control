from __future__ import annotations

import argparse
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
MOMENT_ARM_AVERAGE_DIR = PROJECT_ROOT / "logs" / "dual_camera_motionBenchmark" / "moment_arm_average"
DEFAULT_OUTPUT_DIR = PROJECT_ROOT / "logs" / "dual_camera_motionBenchmark" / "excursion_predictions"

TENDON_COLUMNS = ["FDP", "FDS", "ED", "DI", "PI", "LUM"]
JOINT_COLUMNS = ["DIP", "PIP", "MCP"]
FLEXION_RAD_COLUMNS = {
    "DIP": "DIP_flexion_angle_rad",
    "PIP": "PIP_flexion_angle_rad",
    "MCP": "MCP_flexion_angle_rad",
}
DISP_MM_RENAME_MAP = {
    "weight_0_disp_mm": "FDP",
    "weight_1_disp_mm": "FDS",
    "weight_2_disp_mm": "ED",
    "weight_3_disp_mm": "PI",
    "weight_4_disp_mm": "DI",
    "weight_5_disp_mm": "LUM",
}
ANGLE_SMOOTHING_WINDOW = 5
EXCURSION_SMOOTHING_WINDOW = 5


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Predict tendon excursion from one processed dual-camera CSV."
    )
    parser.add_argument("--input-csv", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, default=DEFAULT_OUTPUT_DIR)
    return parser.parse_args()


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


def _load_moment_arm_coefficients() -> dict[str, dict[str, np.ndarray]]:
    coefficients: dict[str, dict[str, np.ndarray]] = {}

    for joint in JOINT_COLUMNS:
        json_path = MOMENT_ARM_AVERAGE_DIR / f"moment_arm_{joint}_average.json"
        if not json_path.exists():
            raise FileNotFoundError(f"Moment arm JSON not found: {json_path}")

        payload = json.loads(json_path.read_text(encoding="utf-8"))
        tendon_coefficients = payload.get("moment_arm_coefficients")
        if not isinstance(tendon_coefficients, dict):
            raise ValueError(f"Invalid moment arm JSON format: {json_path}")

        joint_coefficients: dict[str, np.ndarray] = {}
        for tendon in TENDON_COLUMNS:
            raw_coefficients = tendon_coefficients.get(tendon)
            if not isinstance(raw_coefficients, list) or len(raw_coefficients) != 4:
                raise ValueError(
                    f"Expected four moment arm coefficients for tendon={tendon!r} in {json_path}"
                )
            joint_coefficients[tendon] = np.asarray(raw_coefficients, dtype=np.float64)

        coefficients[joint] = joint_coefficients

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


def _calculate_angle_changes(
    smoothed_angles: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    frame_count = len(smoothed_angles)
    delta_angles = np.full(frame_count, np.nan, dtype=np.float64)
    midpoint_angles = np.full(frame_count, np.nan, dtype=np.float64)

    if frame_count == 0:
        return delta_angles, midpoint_angles

    delta_angles[0] = 0.0
    midpoint_angles[0] = smoothed_angles[0]
    for frame_idx in range(1, frame_count):
        previous_angle = smoothed_angles[frame_idx - 1]
        current_angle = smoothed_angles[frame_idx]
        if not np.isfinite(previous_angle) or not np.isfinite(current_angle):
            continue

        delta_angle = current_angle - previous_angle
        delta_angles[frame_idx] = delta_angle
        midpoint_angles[frame_idx] = 0.5 * (previous_angle + current_angle)

    return delta_angles, midpoint_angles


def _normalize_to_zero(values: np.ndarray) -> np.ndarray:
    normalized = np.asarray(values, dtype=np.float64).copy()
    finite_indices = np.flatnonzero(np.isfinite(normalized))
    if finite_indices.size == 0:
        return normalized
    normalized -= normalized[finite_indices[0]]
    return normalized


def _build_prediction_dataframe(
    df: pd.DataFrame,
    moment_arm_coefficients: dict[str, dict[str, np.ndarray]],
) -> pd.DataFrame:
    comparison_df = pd.DataFrame(index=df.index)
    comparison_df["elapsed_s"] = pd.to_numeric(df["elapsed_s"], errors="coerce")

    joint_delta_angles: dict[str, np.ndarray] = {}
    joint_midpoint_angles: dict[str, np.ndarray] = {}

    for joint in JOINT_COLUMNS:
        smoothed_angles = _smooth_angle_series(df[FLEXION_RAD_COLUMNS[joint]])
        delta_angles, midpoint_angles = _calculate_angle_changes(smoothed_angles)

        comparison_df[f"{joint}_actual_angle_rad"] = pd.to_numeric(df[FLEXION_RAD_COLUMNS[joint]], errors="coerce")
        comparison_df[f"{joint}_actual_angle_deg"] = np.rad2deg(comparison_df[f"{joint}_actual_angle_rad"])
        comparison_df[f"{joint}_smoothed_angle_rad"] = smoothed_angles
        comparison_df[f"{joint}_smoothed_angle_deg"] = np.rad2deg(smoothed_angles)
        comparison_df[f"{joint}_delta_angle_rad"] = delta_angles
        joint_delta_angles[joint] = delta_angles
        joint_midpoint_angles[joint] = midpoint_angles

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

                coeffs = moment_arm_coefficients[joint][tendon]
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


def run_prediction(target_input_csv: Path, output_dir: Path) -> dict[str, Path]:
    target_input_csv = target_input_csv.expanduser().resolve()
    output_dir = output_dir.expanduser().resolve()
    if not target_input_csv.exists():
        raise FileNotFoundError(f"Target input CSV not found: {target_input_csv}")

    processed_df = pd.read_csv(target_input_csv)
    processed_df = _prepare_input_dataframe(processed_df)
    _validate_processed_dataframe(processed_df)
    moment_arm_coefficients = _load_moment_arm_coefficients()
    comparison_df = _build_prediction_dataframe(processed_df, moment_arm_coefficients)

    output_dir.mkdir(parents=True, exist_ok=True)
    output_file_prefix = target_input_csv.stem
    output_csv_path = output_dir / f"{output_file_prefix}_prediction.csv"
    output_plot_path = output_dir / f"{output_file_prefix}_predicted_vs_actual_excursion.png"
    output_time_angle_plot_path = output_dir / f"{output_file_prefix}_time_vs_joint_angle.png"
    comparison_df.to_csv(output_csv_path, index=False)
    _plot_excursion_comparison(comparison_df, output_plot_path)
    _plot_time_vs_joint_angle(comparison_df, output_time_angle_plot_path)

    print(f"Loaded input CSV: {target_input_csv}")
    print(f"Saved comparison CSV: {output_csv_path}")
    print(f"Saved comparison plot: {output_plot_path}")
    print(f"Saved time-angle plot: {output_time_angle_plot_path}")
    return {
        "csv": output_csv_path,
        "plot": output_plot_path,
        "time_angle_plot": output_time_angle_plot_path,
    }


def main() -> None:
    args = _parse_args()
    run_prediction(args.input_csv, args.output_dir)


if __name__ == "__main__":
    main()

