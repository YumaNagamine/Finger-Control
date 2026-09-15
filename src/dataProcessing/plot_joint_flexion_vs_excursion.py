from __future__ import annotations

import argparse
from datetime import datetime
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
INPUT_DIR = PROJECT_ROOT / "logs" / "dual_camera" / "processed_csv"
CLEANED_DIR = INPUT_DIR / "cleaned"
OUTPUT_DIR = PROJECT_ROOT / "logs" / "dual_camera" / "processed_plot"

CSV_FILE_NAME = "dual_processed_dual_record_20260909_170528_20260909_181911.csv"
TENDON_NAME = "FDS"
TARGET_ROW_COUNT = 0
MOVING_WINDOW_SIZE = 61
BASELINE_FRAME_COUNT = 5
MAX_INTERPOLATION_GAP = 5
OUTLIER_WINDOW_SIZE = 11
OUTLIER_SIGMA = 3.5
JUMP_RETURN_MAX_FRAMES = 15
JUMP_SIGMA = 8.0
ACTIVE_RANGE_LOW = 0.02
ACTIVE_RANGE_HIGH = 0.98
ROM_ENDPOINT_FRACTION = 0.05
OVERLAY_RAW_DATA = True
RAW_DATA_ALPHA = 0.2

TENDON_EXCURSION_COLUMNS = {
    "FDP": "weight_0_disp_mm",
    "FDS": "weight_1_disp_mm",
    "EI": "weight_2_disp_mm",
    "PI": "weight_3_disp_mm",
    "DI": "weight_4_disp_mm",
    "LUM": "weight_5_disp_mm",
}
TENDON_QUALITY_COLUMNS = {
    tendon_name: excursion_column.replace("_disp_mm", "_quality")
    for tendon_name, excursion_column in TENDON_EXCURSION_COLUMNS.items()
}
JOINT_COLORS = {
    "DIP": "#0072B2",
    "PIP": "#D55E00",
    "MCP": "#009E73",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Clean a dual-processed CSV and plot joint displacement vs excursion."
    )
    parser.add_argument("--csv", default=CSV_FILE_NAME, help="Input CSV path or file name.")
    parser.add_argument("--tendon", default=TENDON_NAME, choices=TENDON_EXCURSION_COLUMNS)
    parser.add_argument("--rows", type=int, default=TARGET_ROW_COUNT)
    parser.add_argument(
        "--cleaned-root",
        default=str(CLEANED_DIR),
        help="Directory for cleaned CSV and ROM summary outputs.",
    )
    parser.add_argument(
        "--output-root",
        default=str(OUTPUT_DIR),
        help="Directory for plot outputs.",
    )
    parser.add_argument("--window", type=int, default=MOVING_WINDOW_SIZE)
    parser.add_argument("--baseline-frames", type=int, default=BASELINE_FRAME_COUNT)
    parser.add_argument("--outlier-window", type=int, default=OUTLIER_WINDOW_SIZE)
    parser.add_argument("--outlier-sigma", type=float, default=OUTLIER_SIGMA)
    parser.add_argument(
        "--jump-return-frames", type=int, default=JUMP_RETURN_MAX_FRAMES
    )
    parser.add_argument("--jump-sigma", type=float, default=JUMP_SIGMA)
    parser.add_argument("--no-outlier-filter", action="store_true")
    parser.add_argument("--no-raw-overlay", action="store_true")
    return parser.parse_args()


def _resolve_csv(raw_path: str) -> Path:
    path = Path(raw_path)
    if not path.is_absolute():
        working_path = Path.cwd() / path
        path = working_path if working_path.is_file() else INPUT_DIR / path
    path = path.resolve()
    if not path.is_file():
        raise FileNotFoundError(f"CSV file not found: {path}")
    return path


def _validate_args(args: argparse.Namespace) -> None:
    if args.rows < 0:
        raise ValueError("--rows must be 0 or greater.")
    if args.window <= 0 or args.window % 2 == 0:
        raise ValueError("--window must be a positive odd integer.")
    if args.baseline_frames <= 0:
        raise ValueError("--baseline-frames must be 1 or greater.")
    if args.outlier_window < 3 or args.outlier_window % 2 == 0:
        raise ValueError("--outlier-window must be an odd integer of 3 or greater.")
    if args.outlier_sigma <= 0:
        raise ValueError("--outlier-sigma must be greater than 0.")
    if args.jump_return_frames < 1:
        raise ValueError("--jump-return-frames must be 1 or greater.")
    if args.jump_sigma <= 0:
        raise ValueError("--jump-sigma must be greater than 0.")


def _robust_step_threshold(series: pd.Series, sigma: float) -> float:
    steps = series.diff().dropna()
    if steps.empty:
        return np.inf

    center = float(steps.median())
    mad = float((steps - center).abs().median())
    median_abs_step = float(steps.abs().median())
    return max(sigma * 1.4826 * mad, 4.0 * median_abs_step, 1e-9)


def _jump_return_outliers(
    series: pd.Series,
    max_frames: int,
    sigma: float,
) -> pd.Series:
    """Find short plateaus created by a large jump followed by a return jump."""
    numeric = pd.to_numeric(series, errors="coerce")
    steps = numeric.diff()
    threshold = _robust_step_threshold(numeric, sigma)
    large_positions = np.flatnonzero((steps.abs() > threshold).to_numpy())
    outliers = pd.Series(False, index=numeric.index)

    position_index = 0
    while position_index < len(large_positions):
        start = int(large_positions[position_index])
        if start == 0 or not np.isfinite(numeric.iloc[start - 1]):
            position_index += 1
            continue

        first_jump = float(steps.iloc[start])
        for candidate_index in range(position_index + 1, len(large_positions)):
            end = int(large_positions[candidate_index])
            if end - start > max_frames:
                break
            second_jump = float(steps.iloc[end])
            if first_jump * second_jump >= 0:
                continue
            return_error = abs(float(numeric.iloc[end] - numeric.iloc[start - 1]))
            return_tolerance = max(threshold, 0.25 * abs(first_jump))
            if return_error <= return_tolerance:
                outliers.iloc[start:end] = True
                position_index = candidate_index
                break
        position_index += 1

    return outliers


def _hampel_outliers(
    series: pd.Series,
    window: int,
    sigma: float,
) -> pd.Series:
    numeric = pd.to_numeric(series, errors="coerce")
    minimum_samples = window // 2 + 1
    local_median = numeric.rolling(
        window=window, center=True, min_periods=minimum_samples
    ).median()

    def local_mad(values: np.ndarray) -> float:
        finite = values[np.isfinite(values)]
        if finite.size == 0:
            return np.nan
        median = np.median(finite)
        return float(np.median(np.abs(finite - median)))

    local_mad_values = numeric.rolling(
        window=window, center=True, min_periods=minimum_samples
    ).apply(local_mad, raw=True)
    residual = (numeric - local_median).abs()
    noise_floor = 4.0 * float(numeric.diff().abs().median(skipna=True))
    if not np.isfinite(noise_floor):
        noise_floor = 1e-9
    threshold = (sigma * 1.4826 * local_mad_values).clip(
        lower=max(noise_floor, 1e-9)
    )
    return numeric.notna() & local_median.notna() & (residual > threshold)


def _filter_outliers(
    series: pd.Series,
    window: int,
    sigma: float,
    jump_return_frames: int,
    jump_sigma: float,
    enabled: bool,
) -> tuple[pd.Series, pd.Series]:
    numeric = pd.to_numeric(series, errors="coerce")
    if not enabled:
        return numeric, pd.Series(False, index=numeric.index)

    jump_outliers = _jump_return_outliers(
        numeric, max_frames=jump_return_frames, sigma=jump_sigma
    )
    without_jumps = numeric.mask(jump_outliers)
    hampel_outliers = _hampel_outliers(without_jumps, window=window, sigma=sigma)
    combined = jump_outliers | hampel_outliers
    return numeric.mask(combined), combined


def _active_motion_mask(excursion: pd.Series, window: int) -> pd.Series:
    detection = excursion.interpolate(limit=MAX_INTERPOLATION_GAP, limit_area="inside")
    detection = detection.rolling(window=window, center=True, min_periods=1).mean()
    finite = detection.dropna()
    if finite.empty:
        raise ValueError("No valid excursion values were found.")

    minimum = float(finite.min())
    maximum = float(finite.max())
    span = maximum - minimum
    if not np.isfinite(span) or span <= 0:
        raise ValueError("Excursion has no usable range.")

    low = minimum + ACTIVE_RANGE_LOW * span
    high = minimum + ACTIVE_RANGE_HIGH * span
    candidates = detection.between(low, high)
    positions = np.flatnonzero(candidates.to_numpy())
    if positions.size == 0:
        raise ValueError("Could not identify the active-motion interval.")

    active = pd.Series(False, index=excursion.index)
    active.iloc[positions[0] : positions[-1] + 1] = True
    return active


def build_cleaned_dataframe(
    dataframe: pd.DataFrame,
    source_path: Path,
    tendon_name: str,
    excursion_column: str,
    window: int,
    baseline_frames: int,
    outlier_window: int,
    outlier_sigma: float,
    jump_return_frames: int,
    jump_sigma: float,
    filter_outliers: bool,
) -> pd.DataFrame:
    required = [excursion_column, *JOINT_COLORS]
    missing = [column for column in required if column not in dataframe.columns]
    if missing:
        raise ValueError(f"CSV is missing required columns: {missing}")

    cleaned = dataframe.copy()
    raw_excursion = pd.to_numeric(dataframe[excursion_column], errors="coerce")
    quality_column = TENDON_QUALITY_COLUMNS[tendon_name]
    if quality_column in dataframe.columns:
        tendon_quality_ok = dataframe[quality_column].astype(str).eq("ok")
    else:
        tendon_quality_ok = pd.Series(True, index=dataframe.index)

    filtered_excursion, excursion_outlier = _filter_outliers(
        raw_excursion,
        window=outlier_window,
        sigma=outlier_sigma,
        jump_return_frames=jump_return_frames,
        jump_sigma=jump_sigma,
        enabled=filter_outliers,
    )
    # A quality flag is part of the measurement protocol.  Keep non-ok frames
    # in the diagnostic CSV, but do not let them define motion or ROM.
    filtered_excursion = filtered_excursion.where(tendon_quality_ok)
    active = _active_motion_mask(filtered_excursion, window)
    excursion_filled = filtered_excursion.interpolate(
        limit=MAX_INTERPOLATION_GAP, limit_area="inside"
    )
    excursion_smoothed = excursion_filled.rolling(
        window=window, center=True, min_periods=window
    ).mean()

    cleaned["analysis_source_file"] = source_path.name
    cleaned["analysis_tendon"] = tendon_name
    cleaned["analysis_window_frames"] = window
    cleaned["analysis_baseline_frames"] = baseline_frames
    cleaned["analysis_outlier_window_frames"] = outlier_window
    cleaned["analysis_outlier_sigma"] = outlier_sigma
    cleaned["analysis_jump_return_max_frames"] = jump_return_frames
    cleaned["analysis_jump_sigma"] = jump_sigma
    cleaned["analysis_outlier_filter_enabled"] = filter_outliers
    cleaned["analysis_quality_column"] = quality_column
    cleaned["tendon_quality_ok"] = tendon_quality_ok
    cleaned["active_motion"] = active
    cleaned["excursion_raw_mm"] = raw_excursion
    cleaned["excursion_outlier"] = excursion_outlier
    cleaned["excursion_filtered_mm"] = filtered_excursion
    cleaned["excursion_interpolated"] = (
        filtered_excursion.isna() & excursion_filled.notna()
    )
    cleaned["excursion_smoothed_mm"] = excursion_smoothed.where(active)

    ready = active & excursion_smoothed.notna()
    for joint in JOINT_COLORS:
        raw_flexion = 180.0 - pd.to_numeric(dataframe[joint], errors="coerce")
        filtered_flexion, joint_outlier = _filter_outliers(
            raw_flexion,
            window=outlier_window,
            sigma=outlier_sigma,
            jump_return_frames=jump_return_frames,
            jump_sigma=jump_sigma,
            enabled=filter_outliers,
        )
        baseline = filtered_flexion.where(tendon_quality_ok).iloc[:baseline_frames].median(
            skipna=True
        )
        if not np.isfinite(baseline):
            raise ValueError(
                f"No valid {joint} values in the first {baseline_frames} frames."
            )

        filled = filtered_flexion.interpolate(
            limit=MAX_INTERPOLATION_GAP, limit_area="inside"
        )
        smoothed = filled.rolling(window=window, center=True, min_periods=window).mean()
        displacement = smoothed - baseline

        cleaned[f"{joint}_flexion_raw_deg"] = raw_flexion
        cleaned[f"{joint}_outlier"] = joint_outlier
        cleaned[f"{joint}_filtered_deg"] = filtered_flexion
        cleaned[f"{joint}_interpolated"] = filtered_flexion.isna() & filled.notna()
        cleaned[f"{joint}_smoothed_deg"] = smoothed.where(active)
        cleaned[f"{joint}_displacement_deg"] = displacement.where(active)
        joint_ready = active & tendon_quality_ok & smoothed.notna()
        cleaned[f"analysis_ready_{joint}"] = joint_ready
        ready &= joint_ready

    cleaned["analysis_ready_all_joints"] = ready
    return cleaned


def _plot_cleaned(
    cleaned: pd.DataFrame,
    tendon_name: str,
    output_path: Path,
    overlay_raw: bool,
) -> None:
    fig, ax = plt.subplots(figsize=(10, 6.5))
    active = cleaned["active_motion"].astype(bool)
    quality_ok = cleaned.get(
        "tendon_quality_ok", pd.Series(True, index=cleaned.index)
    ).fillna(False).astype(bool)
    processed_x = pd.to_numeric(cleaned["excursion_smoothed_mm"], errors="coerce")
    raw_x = pd.to_numeric(cleaned["excursion_filtered_mm"], errors="coerce")

    for joint, color in JOINT_COLORS.items():
        baseline_frame_count = int(cleaned["analysis_baseline_frames"].iloc[0])
        baseline = pd.to_numeric(
            cleaned[f"{joint}_filtered_deg"], errors="coerce"
        ).where(quality_ok).iloc[:baseline_frame_count].median(skipna=True)

        if overlay_raw:
            raw_y = (
                pd.to_numeric(cleaned[f"{joint}_filtered_deg"], errors="coerce")
                - baseline
            )
            raw_valid = active & quality_ok & raw_x.notna() & raw_y.notna()
            ax.plot(
                raw_x[raw_valid],
                raw_y[raw_valid],
                color=color,
                linewidth=0.9,
                alpha=RAW_DATA_ALPHA,
                label="_nolegend_",
                zorder=1,
            )

        processed_y = pd.to_numeric(
            cleaned[f"{joint}_displacement_deg"], errors="coerce"
        )
        valid = quality_ok & processed_x.notna() & processed_y.notna()
        if valid.any():
            ax.plot(
                processed_x[valid],
                processed_y[valid],
                color=color,
                linewidth=2.2,
                label=joint,
                zorder=2,
            )

    ax.set_xlabel(f"{tendon_name} tendon excursion [mm]")
    ax.set_ylabel("Joint displacement [deg]")
    ax.set_title(tendon_name)
    ax.axhline(0.0, color="#666666", linewidth=1.0, alpha=0.6)
    ax.grid(True, alpha=0.3)
    ax.legend(title="Joint")
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def calculate_rom_summary(cleaned: pd.DataFrame) -> pd.DataFrame:
    """Calculate protocol-specific observed ROM from each joint's valid active interval."""
    active = cleaned["active_motion"].fillna(False).astype(bool)
    quality_ok = cleaned.get(
        "tendon_quality_ok", pd.Series(True, index=cleaned.index)
    ).fillna(False).astype(bool)
    rows: list[dict[str, object]] = []

    for joint in JOINT_COLORS:
        angle = pd.to_numeric(cleaned[f"{joint}_smoothed_deg"], errors="coerce")
        valid = active & quality_ok & angle.notna()
        positions = np.flatnonzero(valid.to_numpy())
        if positions.size < 2:
            rows.append(
                {
                    "joint": joint,
                    "rom_deg": np.nan,
                    "rom_signed_deg": np.nan,
                    "start_angle_deg": np.nan,
                    "end_angle_deg": np.nan,
                    "active_valid_points": int(positions.size),
                    "endpoint_points": 0,
                    "status": "insufficient_valid_points",
                }
            )
            continue

        endpoint_points = max(1, int(np.ceil(positions.size * ROM_ENDPOINT_FRACTION)))
        start_values = angle.iloc[positions[:endpoint_points]].to_numpy(dtype=float)
        end_values = angle.iloc[positions[-endpoint_points:]].to_numpy(dtype=float)
        start_angle = float(np.median(start_values))
        end_angle = float(np.median(end_values))
        signed_rom = end_angle - start_angle
        rows.append(
            {
                "joint": joint,
                "rom_deg": abs(signed_rom),
                "rom_signed_deg": signed_rom,
                "start_angle_deg": start_angle,
                "end_angle_deg": end_angle,
                "active_valid_points": int(positions.size),
                "endpoint_points": endpoint_points,
                "status": "ok",
            }
        )

    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    _validate_args(args)
    csv_path = _resolve_csv(args.csv)
    tendon_name = args.tendon.upper()
    excursion_column = TENDON_EXCURSION_COLUMNS[tendon_name]
    dataframe = pd.read_csv(csv_path, nrows=args.rows or None)

    cleaned = build_cleaned_dataframe(
        dataframe=dataframe,
        source_path=csv_path,
        tendon_name=tendon_name,
        excursion_column=excursion_column,
        window=args.window,
        baseline_frames=args.baseline_frames,
        outlier_window=args.outlier_window,
        outlier_sigma=args.outlier_sigma,
        jump_return_frames=args.jump_return_frames,
        jump_sigma=args.jump_sigma,
        filter_outliers=not args.no_outlier_filter,
    )

    run_id = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
    filter_label = "outlier_filtered" if not args.no_outlier_filter else "no_outlier_filter"
    output_stem = (
        f"{tendon_name}_cleaned_w{args.window}_raw_overlay_{filter_label}_{run_id}"
    )
    cleaned_root = Path(args.cleaned_root).expanduser()
    output_root = Path(args.output_root).expanduser()
    if not cleaned_root.is_absolute():
        cleaned_root = PROJECT_ROOT / cleaned_root
    if not output_root.is_absolute():
        output_root = PROJECT_ROOT / output_root
    cleaned_root.mkdir(parents=True, exist_ok=True)
    output_root.mkdir(parents=True, exist_ok=True)
    cleaned_path = cleaned_root / f"{output_stem}.csv"
    plot_path = output_root / f"{output_stem}.png"
    cleaned.to_csv(cleaned_path, index=False)
    _plot_cleaned(cleaned, tendon_name, plot_path, not args.no_raw_overlay)
    rom_summary = calculate_rom_summary(cleaned)
    rom_summary_path = cleaned_root / f"{output_stem}_rom_summary.csv"
    rom_summary.to_csv(rom_summary_path, index=False)

    active_count = int(cleaned["active_motion"].sum())
    ready_count = int(cleaned["analysis_ready_all_joints"].sum())
    excursion_outlier_count = int(cleaned["excursion_outlier"].sum())
    print(f"Rows copied: {len(cleaned)}")
    print(f"Active-motion rows: {active_count}")
    print(f"Rows ready for all joints: {ready_count}")
    print(f"Excursion outliers removed: {excursion_outlier_count}")
    for joint in JOINT_COLORS:
        print(f"{joint} outliers removed: {int(cleaned[f'{joint}_outlier'].sum())}")
    print("ROM summary:")
    for row in rom_summary.itertuples(index=False):
        print(f"{row.joint}: ROM={row.rom_deg:.3f} deg ({row.status})")
    print(f"Saved cleaned CSV: {cleaned_path}")
    print(f"Saved plot: {plot_path}")
    print(f"Saved ROM summary: {rom_summary_path}")


if __name__ == "__main__":
    main()
