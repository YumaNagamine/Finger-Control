from __future__ import annotations

import itertools
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Edit these paths in code when you want to change the input/output target.
TARGET_INPUT_CSV = (
    PROJECT_ROOT
    / "logs"
    / "dual_camera"
    / "processed_csv"
    / "momentarm_validation_20260626_193529.csv"
)
MOMENT_ARM_AVERAGE_DIR = PROJECT_ROOT / "logs" / "dual_camera" / "moment_arm_average"
OUTPUT_DIR = PROJECT_ROOT / "logs" / "dual_camera" / "momentarm_validation"
OUTPUT_FILE_PREFIX = TARGET_INPUT_CSV.stem
OUTPUT_PLOT_PATH = OUTPUT_DIR / f"{OUTPUT_FILE_PREFIX}_predicted_vs_actual_joint_angles.png"
OUTPUT_CSV_PATH = OUTPUT_DIR / f"{OUTPUT_FILE_PREFIX}_predicted_vs_actual_joint_angles.csv"

TENDON_COLUMNS = ["FDP", "FDS", "EI", "DI", "PI", "LUM"]
JOINT_COLUMNS = ["DIP", "PIP", "MCP"]
FLEXION_DEG_COLUMNS = {
    "DIP": "DIP_flexion_angle_deg",
    "PIP": "PIP_flexion_angle_deg",
    "MCP": "MCP_flexion_angle_deg",
}
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
DIRECTION_COMBINATIONS = list(itertools.product(MOTION_NAMES, repeat=len(JOINT_COLUMNS)))

EXCURSION_SMOOTHING_WINDOW = 5
ANGLE_SMOOTHING_WINDOW = 5
EXCURSION_DEADBAND_MM = 0.02
THETA_UPDATE_DEADBAND_RAD = math.radians(0.05)


def _prepare_input_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    prepared = df.copy()
    prepared = prepared.rename(columns=DISP_MM_RENAME_MAP)

    for joint in JOINT_COLUMNS:
        deg_column = FLEXION_DEG_COLUMNS[joint]
        rad_column = FLEXION_RAD_COLUMNS[joint]
        joint_angle_deg = pd.to_numeric(prepared[joint], errors="coerce")
        flexion_angle_deg = 180.0 - joint_angle_deg
        prepared[deg_column] = flexion_angle_deg
        prepared[rad_column] = flexion_angle_deg * math.pi / 180.0

    return prepared


def _validate_input_dataframe(df: pd.DataFrame) -> None:
    required_columns = ["elapsed_s", *TENDON_COLUMNS, *FLEXION_RAD_COLUMNS.values()]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")


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


def _build_moment_arm_matrix(
    theta_by_joint: dict[str, float],
    motion_by_joint: dict[str, str],
    moment_arm_coefficients: dict[tuple[str, str], dict[str, np.ndarray]],
) -> np.ndarray:
    matrix = np.zeros((len(TENDON_COLUMNS), len(JOINT_COLUMNS)), dtype=np.float64)

    for tendon_idx, tendon in enumerate(TENDON_COLUMNS):
        for joint_idx, joint in enumerate(JOINT_COLUMNS):
            theta = theta_by_joint[joint]
            motion = motion_by_joint[joint]
            coeffs = moment_arm_coefficients[(joint, motion)][tendon]
            matrix[tendon_idx, joint_idx] = float(np.polyval(coeffs, theta))

    return matrix


def _estimate_delta_theta(
    delta_excursion: np.ndarray,
    theta_by_joint: dict[str, float],
    moment_arm_coefficients: dict[tuple[str, str], dict[str, np.ndarray]],
) -> tuple[np.ndarray, tuple[str, str, str], float]:
    finite_mask = np.isfinite(delta_excursion)
    if int(np.count_nonzero(finite_mask)) < len(JOINT_COLUMNS):
        return np.zeros(len(JOINT_COLUMNS), dtype=np.float64), DIRECTION_COMBINATIONS[0], float("nan")

    if np.nanmax(np.abs(delta_excursion[finite_mask])) <= EXCURSION_DEADBAND_MM:
        return np.zeros(len(JOINT_COLUMNS), dtype=np.float64), DIRECTION_COMBINATIONS[0], 0.0

    delta_excursion_valid = delta_excursion[finite_mask]
    best_delta_theta: np.ndarray | None = None
    best_combination: tuple[str, str, str] | None = None
    best_residual = float("inf")

    for motion_combination in DIRECTION_COMBINATIONS:
        motion_by_joint = {
            joint: motion_combination[joint_idx] for joint_idx, joint in enumerate(JOINT_COLUMNS)
        }
        moment_arm_matrix = _build_moment_arm_matrix(
            theta_by_joint=theta_by_joint,
            motion_by_joint=motion_by_joint,
            moment_arm_coefficients=moment_arm_coefficients,
        )
        moment_arm_matrix_valid = moment_arm_matrix[finite_mask, :]
        delta_theta, _, _, _ = np.linalg.lstsq(moment_arm_matrix_valid, delta_excursion_valid, rcond=None)
        delta_theta = np.asarray(delta_theta, dtype=np.float64)
        delta_theta[np.abs(delta_theta) < THETA_UPDATE_DEADBAND_RAD] = 0.0

        reconstructed = moment_arm_matrix_valid @ delta_theta
        residual = float(np.linalg.norm(reconstructed - delta_excursion_valid))
        if residual < best_residual:
            best_delta_theta = delta_theta
            best_combination = motion_combination
            best_residual = residual

    if best_delta_theta is None or best_combination is None:
        raise RuntimeError("Failed to estimate delta theta from excursion")

    return best_delta_theta, best_combination, best_residual


def build_joint_prediction_dataframe(
    df: pd.DataFrame,
    moment_arm_coefficients: dict[tuple[str, str], dict[str, np.ndarray]],
) -> pd.DataFrame:
    comparison_df = pd.DataFrame(index=df.index)
    comparison_df["elapsed_s"] = pd.to_numeric(df["elapsed_s"], errors="coerce")

    smoothed_excursions: dict[str, np.ndarray] = {}
    delta_excursions_by_tendon: dict[str, np.ndarray] = {}
    for tendon in TENDON_COLUMNS:
        actual_excursion = pd.to_numeric(df[tendon], errors="coerce").to_numpy(dtype=np.float64)
        actual_excursion = _normalize_to_zero(actual_excursion)
        smoothed_excursion = _smooth_series(actual_excursion, EXCURSION_SMOOTHING_WINDOW)
        delta_excursion = np.full(len(df), np.nan, dtype=np.float64)
        if len(df) > 0:
            delta_excursion[0] = 0.0
        delta_excursion[1:] = np.diff(smoothed_excursion)

        comparison_df[f"{tendon}_actual_excursion_mm"] = actual_excursion
        comparison_df[f"{tendon}_actual_excursion_smoothed_mm"] = smoothed_excursion
        comparison_df[f"{tendon}_delta_excursion_mm"] = delta_excursion

        smoothed_excursions[tendon] = smoothed_excursion
        delta_excursions_by_tendon[tendon] = delta_excursion

    actual_angle_rad: dict[str, np.ndarray] = {}
    actual_angle_deg_smoothed: dict[str, np.ndarray] = {}
    predicted_angle_rad: dict[str, np.ndarray] = {}
    predicted_delta_rad: dict[str, np.ndarray] = {}
    for joint in JOINT_COLUMNS:
        actual_rad = pd.to_numeric(df[FLEXION_RAD_COLUMNS[joint]], errors="coerce").to_numpy(dtype=np.float64)
        actual_deg = pd.to_numeric(df[FLEXION_DEG_COLUMNS[joint]], errors="coerce").to_numpy(dtype=np.float64)
        actual_angle_rad[joint] = actual_rad
        actual_angle_deg_smoothed[joint] = _smooth_series(actual_deg, ANGLE_SMOOTHING_WINDOW)
        predicted_angle_rad[joint] = np.full(len(df), np.nan, dtype=np.float64)
        predicted_delta_rad[joint] = np.full(len(df), np.nan, dtype=np.float64)

    for joint in JOINT_COLUMNS:
        initial_valid_idx = np.flatnonzero(np.isfinite(actual_angle_rad[joint]))
        if initial_valid_idx.size == 0:
            raise ValueError(f"No finite initial angle found for joint={joint}")
        predicted_angle_rad[joint][0] = actual_angle_rad[joint][initial_valid_idx[0]]
        predicted_delta_rad[joint][0] = 0.0

    chosen_motion_by_joint: dict[str, list[str]] = {joint: [MOTION_NAMES[0]] * len(df) for joint in JOINT_COLUMNS}
    residual_norm = np.full(len(df), np.nan, dtype=np.float64)

    for frame_idx in range(1, len(df)):
        theta_state = {
            joint: float(predicted_angle_rad[joint][frame_idx - 1]) for joint in JOINT_COLUMNS
        }
        if not all(np.isfinite(theta_state[joint]) for joint in JOINT_COLUMNS):
            raise ValueError(f"Predicted state became non-finite before frame {frame_idx}")

        delta_excursion_vector = np.asarray(
            [delta_excursions_by_tendon[tendon][frame_idx] for tendon in TENDON_COLUMNS],
            dtype=np.float64,
        )
        delta_theta, motion_combination, residual = _estimate_delta_theta(
            delta_excursion=delta_excursion_vector,
            theta_by_joint=theta_state,
            moment_arm_coefficients=moment_arm_coefficients,
        )

        residual_norm[frame_idx] = residual
        for joint_idx, joint in enumerate(JOINT_COLUMNS):
            predicted_delta_rad[joint][frame_idx] = delta_theta[joint_idx]
            predicted_angle_rad[joint][frame_idx] = predicted_angle_rad[joint][frame_idx - 1] + delta_theta[joint_idx]
            chosen_motion_by_joint[joint][frame_idx] = motion_combination[joint_idx]

    comparison_df["residual_norm"] = residual_norm

    for joint in JOINT_COLUMNS:
        actual_rad = actual_angle_rad[joint]
        predicted_rad = predicted_angle_rad[joint]
        comparison_df[f"{joint}_actual_angle_rad"] = actual_rad
        comparison_df[f"{joint}_actual_angle_deg"] = np.rad2deg(actual_rad)
        comparison_df[f"{joint}_actual_angle_deg_smoothed"] = actual_angle_deg_smoothed[joint]
        comparison_df[f"{joint}_predicted_delta_rad"] = predicted_delta_rad[joint]
        comparison_df[f"{joint}_predicted_angle_rad"] = predicted_rad
        comparison_df[f"{joint}_predicted_angle_deg"] = np.rad2deg(predicted_rad)
        comparison_df[f"{joint}_error_deg"] = (
            np.rad2deg(predicted_rad) - actual_angle_deg_smoothed[joint]
        )
        comparison_df[f"{joint}_chosen_motion"] = chosen_motion_by_joint[joint]

    return comparison_df


def plot_joint_comparison(comparison_df: pd.DataFrame, output_path: Path) -> None:
    elapsed_s = comparison_df["elapsed_s"].to_numpy(dtype=np.float64)
    use_elapsed_time = np.isfinite(elapsed_s).all()
    x_values = elapsed_s if use_elapsed_time else np.arange(len(comparison_df), dtype=np.int32)
    x_label = "Elapsed time [s]" if use_elapsed_time else "Frame index"

    fig, axes = plt.subplots(3, 1, figsize=(12, 10), sharex=True)

    for axis, joint in zip(axes, JOINT_COLUMNS):
        actual_deg = comparison_df[f"{joint}_actual_angle_deg_smoothed"].to_numpy(dtype=np.float64)
        predicted_deg = comparison_df[f"{joint}_predicted_angle_deg"].to_numpy(dtype=np.float64)

        actual_valid = np.isfinite(actual_deg)
        predicted_valid = np.isfinite(predicted_deg)

        axis.plot(
            x_values[actual_valid],
            actual_deg[actual_valid],
            linestyle="-",
            linewidth=2.0,
            label="Actual",
        )
        axis.plot(
            x_values[predicted_valid],
            predicted_deg[predicted_valid],
            linestyle="--",
            linewidth=2.0,
            label="Predicted",
        )
        axis.set_title(joint)
        axis.set_ylabel("Flexion angle [deg]")
        axis.grid(True, alpha=0.3)
        axis.legend()

    axes[-1].set_xlabel(x_label)
    fig.suptitle("Actual vs predicted joint movement", fontsize=14)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def main() -> None:
    if not TARGET_INPUT_CSV.exists():
        raise FileNotFoundError(f"Target input CSV not found: {TARGET_INPUT_CSV}")

    input_df = pd.read_csv(TARGET_INPUT_CSV)
    prepared_df = _prepare_input_dataframe(input_df)
    _validate_input_dataframe(prepared_df)
    moment_arm_coefficients = _load_moment_arm_coefficients()
    comparison_df = build_joint_prediction_dataframe(prepared_df, moment_arm_coefficients)

    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    comparison_df.to_csv(OUTPUT_CSV_PATH, index=False)
    plot_joint_comparison(comparison_df, OUTPUT_PLOT_PATH)

    print(f"Loaded input CSV: {TARGET_INPUT_CSV}")
    print(f"Saved comparison CSV: {OUTPUT_CSV_PATH}")
    print(f"Saved comparison plot: {OUTPUT_PLOT_PATH}")


if __name__ == "__main__":
    main()

