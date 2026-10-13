from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.signal import savgol_filter


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = (
    PROJECT_ROOT / "logs" / "dual_camera-260907C" / "analysis" / "dual_processed_csv"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "logs" / "analysis_comparison" / "260907C" / "directional_moment_arm"
)

TENDONS = ["FDP", "FDS", "ED", "DI", "PI", "LUM"]
DISP_COLUMNS = {
    "FDP": "weight_0_disp_mm",
    "FDS": "weight_1_disp_mm",
    "ED": "weight_2_disp_mm",
    "PI": "weight_3_disp_mm",
    "DI": "weight_4_disp_mm",
    "LUM": "weight_5_disp_mm",
}
QUALITY_COLUMNS = {
    tendon: column.replace("_disp_mm", "_quality")
    for tendon, column in DISP_COLUMNS.items()
}
EVALUATION_ANGLE_RANGES_DEG = {
    "DIP": (-10.0, 40.0),
    "PIP": (0.0, 90.0),
    "MCP": (-10.0, 90.0),
}

# FDP/FDS are treated as flexor tendons. ED is treated as an extensor for the
# general hypothesis. DI/PI/LUM use the extension prediction explicitly stated
# for this analysis and can be changed here without changing the fitting code.
EXPECTED_STABLE_DIRECTION = {
    "FDP": "extension",
    "FDS": "extension",
    "ED": "flexion",
    "DI": "extension",
    "PI": "extension",
    "LUM": "extension",
}
DIRECTIONS = ("flexion", "extension")


@dataclass
class FitResult:
    center_rad: float
    scale_rad: float
    coefficients: np.ndarray
    derivative_coefficients: np.ndarray
    r_squared: float
    rmse_mm: float
    residual_mad_mm: float
    n_points: int
    angle_min_deg: float
    angle_max_deg: float

    def predict_excursion(self, angle_rad: np.ndarray) -> np.ndarray:
        normalized = (angle_rad - self.center_rad) / self.scale_rad
        return np.polyval(self.coefficients, normalized)

    def predict_moment_arm(self, angle_rad: np.ndarray) -> np.ndarray:
        normalized = (angle_rad - self.center_rad) / self.scale_rad
        return np.polyval(self.derivative_coefficients, normalized) / self.scale_rad


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare trial-to-trial moment-arm stability separately for flexion "
            "and extension samples from the same video."
        )
    )
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--smooth-window", type=int, default=31)
    parser.add_argument(
        "--direction-threshold-deg-s",
        type=float,
        default=1.0,
        help="Absolute smoothed joint speed below this value is excluded.",
    )
    parser.add_argument(
        "--min-direction-run-frames",
        type=int,
        default=5,
        help="Discard direction runs shorter than this number of frames.",
    )
    parser.add_argument("--fit-degree", type=int, default=4)
    parser.add_argument(
        "--fit-margin-deg",
        type=float,
        default=5.0,
        help=(
            "Margin used to extend the human range for fitting and to keep "
            "the fit inside the robust robot range."
        ),
    )
    parser.add_argument(
        "--comparison-margin-deg",
        type=float,
        default=2.0,
        help="Inset from the fitting range used for human comparison.",
    )
    parser.add_argument("--grid-points", type=int, default=51)
    parser.add_argument(
        "--include-non-ok",
        action="store_true",
        help="Include tendon samples whose quality flag is not ok.",
    )
    parser.add_argument("--no-plots", action="store_true")
    return parser.parse_args()


def _validate_args(args: argparse.Namespace) -> None:
    if args.smooth_window < 5 or args.smooth_window % 2 == 0:
        raise ValueError("--smooth-window must be an odd integer of 5 or greater.")
    if args.direction_threshold_deg_s < 0.0:
        raise ValueError("--direction-threshold-deg-s must be non-negative.")
    if args.min_direction_run_frames < 1:
        raise ValueError("--min-direction-run-frames must be 1 or greater.")
    if args.fit_degree < 1:
        raise ValueError("--fit-degree must be 1 or greater.")
    if args.fit_margin_deg < 0.0:
        raise ValueError("--fit-margin-deg must be non-negative.")
    if args.comparison_margin_deg < 0.0:
        raise ValueError("--comparison-margin-deg must be non-negative.")
    if args.grid_points < 3:
        raise ValueError("--grid-points must be 3 or greater.")


def _resolve_input_root(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path = path.resolve()
    if not path.is_dir():
        raise FileNotFoundError(f"Input directory not found: {path}")
    return path


def _resolve_output_root(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _discover_recordings(input_root: Path) -> list[tuple[Path, str, int]]:
    recordings: list[tuple[Path, str, int]] = []
    pattern = re.compile(r"_(DIP|PIP|MCP)_(\d+)_\1(?:\s|_|$)")
    base_trial_pattern = re.compile(r"_(DIP|PIP|MCP)_\1_")
    # 2607C stores the processed CSVs directly under the input root, while
    # 2608A stores them under one directory per joint/trial.
    for path in sorted(input_root.rglob("*.csv")):
        match = pattern.search(path.stem)
        if match is not None:
            recordings.append((path, match.group(1), int(match.group(2))))
            continue
        # Some datasets use the unnumbered filename for trial 1, e.g.
        # dual_processed_2607B_DIP_DIP_*.csv.
        base_trial_match = base_trial_pattern.search(path.stem)
        if base_trial_match is not None:
            recordings.append((path, base_trial_match.group(1), 1))
    if not recordings:
        raise FileNotFoundError(f"No target-joint recordings found under {input_root}")
    return recordings


def _determine_range_specs(
    recordings: list[tuple[Path, str, int]],
    fit_margin_deg: float,
    comparison_margin_deg: float,
) -> dict[str, dict[str, float]]:
    """Determine human, robot, fitting, and comparison ranges per joint.

    The robot range is the intersection of per-trial 1st--99th percentile
    ranges.  This avoids letting a single noisy endpoint define the analysis
    range while still retaining the range common to all trials.
    """
    per_joint_trial_ranges: dict[str, list[tuple[float, float]]] = {
        joint: [] for joint in EVALUATION_ANGLE_RANGES_DEG
    }
    for path, joint, _trial in recordings:
        dataframe = pd.read_csv(path, usecols=[joint])
        angle = 180.0 - pd.to_numeric(dataframe[joint], errors="coerce")
        angle = angle[np.isfinite(angle)].to_numpy(dtype=float)
        if angle.size < 10:
            raise ValueError(f"Too few valid angle samples in {path.name}: {angle.size}")
        per_joint_trial_ranges[joint].append(
            (float(np.percentile(angle, 1.0)), float(np.percentile(angle, 99.0)))
        )

    specs: dict[str, dict[str, float]] = {}
    for joint, human_range in EVALUATION_ANGLE_RANGES_DEG.items():
        trial_ranges = per_joint_trial_ranges[joint]
        if not trial_ranges:
            raise ValueError(f"No recordings found for joint {joint}.")
        robot_common_min = max(lower for lower, _upper in trial_ranges)
        robot_common_max = min(upper for _lower, upper in trial_ranges)
        human_min, human_max = human_range

        fitting_min = max(
            human_min - fit_margin_deg,
            robot_common_min + fit_margin_deg,
        )
        fitting_max = min(
            human_max + fit_margin_deg,
            robot_common_max - fit_margin_deg,
        )
        comparison_min = max(human_min, fitting_min + comparison_margin_deg)
        comparison_max = min(human_max, fitting_max - comparison_margin_deg)
        if fitting_max - fitting_min < 5.0:
            raise ValueError(
                f"Fitting range is too narrow for {joint}: "
                f"{fitting_min:.2f} to {fitting_max:.2f} deg"
            )
        if comparison_max - comparison_min < 5.0:
            raise ValueError(
                f"Comparison range is too narrow for {joint}: "
                f"{comparison_min:.2f} to {comparison_max:.2f} deg"
            )
        specs[joint] = {
            "human_angle_min_deg": human_min,
            "human_angle_max_deg": human_max,
            "robot_common_p01_min_deg": robot_common_min,
            "robot_common_p99_max_deg": robot_common_max,
            "fitting_angle_min_deg": fitting_min,
            "fitting_angle_max_deg": fitting_max,
            "comparison_target_min_deg": comparison_min,
            "comparison_target_max_deg": comparison_max,
        }
    return specs


def _odd_window(requested: int, n_points: int) -> int:
    window = min(requested, n_points if n_points % 2 else n_points - 1)
    if window < 5:
        raise ValueError(f"At least 5 valid angle samples are required, got {n_points}.")
    return window


def _retain_long_runs(mask: np.ndarray, minimum_length: int) -> np.ndarray:
    retained = np.zeros(mask.size, dtype=bool)
    positions = np.flatnonzero(mask)
    if positions.size == 0:
        return retained
    breaks = np.flatnonzero(np.diff(positions) > 1)
    starts = np.r_[0, breaks + 1]
    ends = np.r_[breaks, positions.size - 1]
    for start, end in zip(starts, ends):
        if end - start + 1 >= minimum_length:
            retained[positions[start] : positions[end] + 1] = True
    return retained


def _direction_masks(
    angle_deg: np.ndarray,
    time_s: np.ndarray,
    smooth_window: int,
    speed_threshold_deg_s: float,
    minimum_run_frames: int,
) -> dict[str, np.ndarray]:
    window = _odd_window(smooth_window, angle_deg.size)
    smoothed_angle = savgol_filter(angle_deg, window_length=window, polyorder=2)
    speed_deg_s = np.gradient(smoothed_angle, time_s)

    flexion = _retain_long_runs(
        speed_deg_s > speed_threshold_deg_s, minimum_run_frames
    )
    extension = _retain_long_runs(
        speed_deg_s < -speed_threshold_deg_s, minimum_run_frames
    )
    return {"flexion": flexion, "extension": extension}


def _fit_polynomial(
    angle_rad: np.ndarray,
    excursion_mm: np.ndarray,
    fit_degree: int,
    nominal_range_deg: tuple[float, float],
) -> FitResult | None:
    valid = np.isfinite(angle_rad) & np.isfinite(excursion_mm)
    if int(np.count_nonzero(valid)) < fit_degree + 1:
        return None

    angle_rad = angle_rad[valid]
    excursion_mm = excursion_mm[valid]
    nominal_min_rad, nominal_max_rad = np.deg2rad(nominal_range_deg)
    center_rad = (nominal_min_rad + nominal_max_rad) / 2.0
    scale_rad = (nominal_max_rad - nominal_min_rad) / 2.0
    normalized = (angle_rad - center_rad) / scale_rad
    coefficients = np.polyfit(normalized, excursion_mm, deg=fit_degree)
    derivative_coefficients = np.polyder(coefficients)
    fitted = np.polyval(coefficients, normalized)
    residuals = excursion_mm - fitted
    residual_sum_squares = float(np.sum(residuals**2))
    total_sum_squares = float(np.sum((excursion_mm - np.mean(excursion_mm)) ** 2))
    r_squared = (
        1.0 - residual_sum_squares / total_sum_squares
        if total_sum_squares > 0.0
        else float("nan")
    )
    return FitResult(
        center_rad=center_rad,
        scale_rad=scale_rad,
        coefficients=coefficients,
        derivative_coefficients=derivative_coefficients,
        r_squared=r_squared,
        rmse_mm=float(np.sqrt(np.mean(residuals**2))),
        residual_mad_mm=float(
            np.median(np.abs(residuals - np.median(residuals)))
        ),
        n_points=int(angle_rad.size),
        angle_min_deg=float(np.rad2deg(np.min(angle_rad))),
        angle_max_deg=float(np.rad2deg(np.max(angle_rad))),
    )


def _load_recording(
    path: Path,
    joint: str,
    trial: int,
    args: argparse.Namespace,
    fitting_range_deg: tuple[float, float],
) -> tuple[dict[str, dict[str, object]], dict[str, object]]:
    dataframe = pd.read_csv(path)
    required = {"elapsed_s", joint, *DISP_COLUMNS.values()}
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise ValueError(f"{path.name} is missing columns: {missing}")

    angle = 180.0 - pd.to_numeric(dataframe[joint], errors="coerce")
    time = pd.to_numeric(dataframe["elapsed_s"], errors="coerce")
    valid = angle.notna() & time.notna()
    angle = angle[valid].to_numpy(dtype=float)
    time = time[valid].to_numpy(dtype=float)
    if angle.size < args.smooth_window:
        raise ValueError(f"Too few valid angle samples in {path.name}: {angle.size}")

    direction_masks = _direction_masks(
        angle,
        time,
        smooth_window=args.smooth_window,
        speed_threshold_deg_s=args.direction_threshold_deg_s,
        minimum_run_frames=args.min_direction_run_frames,
    )
    range_min_deg, range_max_deg = fitting_range_deg
    in_range = (angle >= range_min_deg) & (angle <= range_max_deg)
    angle_rad = np.deg2rad(angle)

    results: dict[str, dict[str, object]] = {}
    for tendon in TENDONS:
        excursion = pd.to_numeric(dataframe.loc[valid, DISP_COLUMNS[tendon]], errors="coerce").to_numpy(
            dtype=float
        )
        quality_column = QUALITY_COLUMNS[tendon]
        if args.include_non_ok or quality_column not in dataframe.columns:
            quality_ok = np.ones(angle.size, dtype=bool)
        else:
            quality_ok = dataframe.loc[valid, quality_column].astype(str).eq("ok").to_numpy()

        for direction in DIRECTIONS:
            sample_mask = in_range & direction_masks[direction] & quality_ok & np.isfinite(excursion)
            fit = _fit_polynomial(
                angle_rad[sample_mask],
                excursion[sample_mask],
                fit_degree=args.fit_degree,
                nominal_range_deg=fitting_range_deg,
            )
            results[f"{tendon}:{direction}"] = {
                "joint": joint,
                "trial": trial,
                "source": path.stem,
                "tendon": tendon,
                "direction": direction,
                "fit": fit,
                "num_in_range_direction_points": int(np.count_nonzero(sample_mask)),
                "num_quality_excluded_in_range": int(
                    np.count_nonzero(in_range & direction_masks[direction] & ~quality_ok)
                ),
            }

    recording_summary = {
        "source": path.stem,
        "joint": joint,
        "trial": trial,
        "num_valid_angle_points": int(angle.size),
        "angle_min_deg": float(np.min(angle)),
        "angle_max_deg": float(np.max(angle)),
        "fitting_angle_min_deg": range_min_deg,
        "fitting_angle_max_deg": range_max_deg,
        "flexion_points_in_range": int(np.count_nonzero(in_range & direction_masks["flexion"])),
        "extension_points_in_range": int(np.count_nonzero(in_range & direction_masks["extension"])),
    }
    return results, recording_summary


def _fit_row(result: dict[str, object]) -> dict[str, object]:
    fit = result["fit"]
    row = {
        "source": result["source"],
        "trial": result["trial"],
        "joint": result["joint"],
        "tendon": result["tendon"],
        "direction": result["direction"],
        "num_points": result["num_in_range_direction_points"],
        "num_quality_excluded_in_range": result["num_quality_excluded_in_range"],
        "fit_r_squared": np.nan,
        "fit_rmse_mm": np.nan,
        "fit_residual_mad_mm": np.nan,
        "angle_min_deg": np.nan,
        "angle_max_deg": np.nan,
        "angle_span_deg": np.nan,
        "status": "insufficient_points",
    }
    if fit is not None:
        row.update(
            {
                "fit_r_squared": fit.r_squared,
                "fit_rmse_mm": fit.rmse_mm,
                "fit_residual_mad_mm": fit.residual_mad_mm,
                "angle_min_deg": fit.angle_min_deg,
                "angle_max_deg": fit.angle_max_deg,
                "angle_span_deg": fit.angle_max_deg - fit.angle_min_deg,
                "status": "ok",
            }
        )
    return row


def _common_angle_grid(
    result_list: list[dict[str, object]],
    joint: str,
    grid_points: int,
    comparison_range_deg: tuple[float, float],
) -> np.ndarray | None:
    fits = [result["fit"] for result in result_list if result["fit"] is not None]
    if len(fits) != len(result_list):
        return None
    lower = max(float(fit.angle_min_deg) for fit in fits)
    upper = min(float(fit.angle_max_deg) for fit in fits)
    comparison_lower, comparison_upper = comparison_range_deg
    lower = max(lower, comparison_lower)
    upper = min(upper, comparison_upper)
    if not np.isfinite(lower) or not np.isfinite(upper) or upper - lower < 5.0:
        return None
    return np.linspace(lower, upper, grid_points)


def _curve_rows_and_stability(
    all_results: list[dict[str, object]],
    args: argparse.Namespace,
    range_specs: dict[str, dict[str, float]],
) -> tuple[list[dict[str, object]], list[dict[str, object]]]:
    curve_rows: list[dict[str, object]] = []
    stability_rows: list[dict[str, object]] = []
    grouped: dict[tuple[str, str], dict[str, list[dict[str, object]]]] = {}
    for result in all_results:
        key = (str(result["joint"]), str(result["tendon"]))
        grouped.setdefault(key, {direction: [] for direction in DIRECTIONS})[
            str(result["direction"])
        ].append(result)

    for (joint, tendon), by_direction in sorted(grouped.items()):
        if any(len(by_direction[direction]) < 2 for direction in DIRECTIONS):
            continue
        all_direction_results = by_direction["flexion"] + by_direction["extension"]
        range_spec = range_specs[joint]
        comparison_range_deg = (
            range_spec["comparison_target_min_deg"],
            range_spec["comparison_target_max_deg"],
        )
        grid_deg = _common_angle_grid(
            all_direction_results,
            joint,
            args.grid_points,
            comparison_range_deg,
        )
        if grid_deg is None:
            continue
        grid_rad = np.deg2rad(grid_deg)
        direction_stats: dict[str, dict[str, float]] = {}

        for direction in DIRECTIONS:
            trial_excursion: list[np.ndarray] = []
            trial_moment_arm: list[np.ndarray] = []
            for result in sorted(by_direction[direction], key=lambda item: int(item["trial"])):
                fit = result["fit"]
                assert fit is not None
                excursion = fit.predict_excursion(grid_rad)
                moment_arm = fit.predict_moment_arm(grid_rad)
                trial_excursion.append(excursion)
                trial_moment_arm.append(moment_arm)
                for angle_deg, excursion_value, moment_arm_value in zip(
                    grid_deg, excursion, moment_arm
                ):
                    curve_rows.append(
                        {
                            "source": result["source"],
                            "trial": result["trial"],
                            "joint": joint,
                            "tendon": tendon,
                            "direction": direction,
                            "angle_deg": angle_deg,
                            "excursion_fit_mm": excursion_value,
                            "moment_arm_mm_per_rad": moment_arm_value,
                        }
                    )

            excursion_array = np.vstack(trial_excursion)
            moment_arm_array = np.vstack(trial_moment_arm)
            excursion_sd = np.std(excursion_array, axis=0, ddof=1)
            moment_arm_sd = np.std(moment_arm_array, axis=0, ddof=1)
            direction_results = by_direction[direction]
            direction_stats[direction] = {
                "mean_excursion_sd_mm": float(np.mean(excursion_sd)),
                "max_excursion_sd_mm": float(np.max(excursion_sd)),
                "mean_moment_arm_sd_mm_per_rad": float(np.mean(moment_arm_sd)),
                "max_moment_arm_sd_mm_per_rad": float(np.max(moment_arm_sd)),
                "mean_fit_rmse_mm": float(
                    np.mean([result["fit"].rmse_mm for result in direction_results])
                ),
                "mean_fit_r_squared": float(
                    np.nanmean([result["fit"].r_squared for result in direction_results])
                ),
                "mean_num_points": float(
                    np.mean([result["num_in_range_direction_points"] for result in direction_results])
                ),
            }

        flexion_stats = direction_stats["flexion"]
        extension_stats = direction_stats["extension"]
        observed = (
            "flexion"
            if flexion_stats["mean_moment_arm_sd_mm_per_rad"]
            < extension_stats["mean_moment_arm_sd_mm_per_rad"]
            else "extension"
        )
        expected = EXPECTED_STABLE_DIRECTION[tendon]
        expected_sd = direction_stats[expected]["mean_moment_arm_sd_mm_per_rad"]
        other_direction = "extension" if expected == "flexion" else "flexion"
        other_sd = direction_stats[other_direction]["mean_moment_arm_sd_mm_per_rad"]
        stability_rows.append(
            {
                "joint": joint,
                "tendon": tendon,
                "common_angle_min_deg": float(grid_deg[0]),
                "common_angle_max_deg": float(grid_deg[-1]),
                **range_spec,
                "expected_stable_direction": expected,
                "observed_more_stable_direction": observed,
                "hypothesis_supported_descriptively": bool(observed == expected),
                "flexion_mean_moment_arm_sd_mm_per_rad": flexion_stats[
                    "mean_moment_arm_sd_mm_per_rad"
                ],
                "extension_mean_moment_arm_sd_mm_per_rad": extension_stats[
                    "mean_moment_arm_sd_mm_per_rad"
                ],
                "expected_minus_other_moment_arm_sd_mm_per_rad": expected_sd - other_sd,
                "other_minus_expected_moment_arm_sd_mm_per_rad": other_sd - expected_sd,
                "expected_to_other_sd_ratio": (
                    expected_sd / other_sd if other_sd > 0.0 else np.nan
                ),
                "flexion_max_moment_arm_sd_mm_per_rad": flexion_stats[
                    "max_moment_arm_sd_mm_per_rad"
                ],
                "extension_max_moment_arm_sd_mm_per_rad": extension_stats[
                    "max_moment_arm_sd_mm_per_rad"
                ],
                "flexion_mean_excursion_sd_mm": flexion_stats["mean_excursion_sd_mm"],
                "extension_mean_excursion_sd_mm": extension_stats["mean_excursion_sd_mm"],
                "flexion_mean_fit_rmse_mm": flexion_stats["mean_fit_rmse_mm"],
                "extension_mean_fit_rmse_mm": extension_stats["mean_fit_rmse_mm"],
                "flexion_mean_fit_r_squared": flexion_stats["mean_fit_r_squared"],
                "extension_mean_fit_r_squared": extension_stats["mean_fit_r_squared"],
                "flexion_mean_num_points": flexion_stats["mean_num_points"],
                "extension_mean_num_points": extension_stats["mean_num_points"],
                "status": "ok",
            }
        )
    return curve_rows, stability_rows


def _plot_curves(curve_df: pd.DataFrame, output_root: Path) -> None:
    plot_root = output_root / "plots"
    plot_root.mkdir(parents=True, exist_ok=True)
    for (joint, tendon), group in curve_df.groupby(["joint", "tendon"], sort=True):
        fig, ax = plt.subplots(figsize=(8.5, 5.5))
        for (direction, trial), trial_group in group.groupby(["direction", "trial"], sort=True):
            linestyle = "-" if direction == "flexion" else "--"
            ax.plot(
                trial_group["angle_deg"],
                trial_group["moment_arm_mm_per_rad"],
                linestyle=linestyle,
                linewidth=1.2,
                alpha=0.75,
                label=f"{direction}, trial {trial}",
            )
        for direction, direction_group in group.groupby("direction", sort=True):
            mean_curve = direction_group.groupby("angle_deg", as_index=False)[
                "moment_arm_mm_per_rad"
            ].mean()
            linestyle = "-" if direction == "flexion" else "--"
            ax.plot(
                mean_curve["angle_deg"],
                mean_curve["moment_arm_mm_per_rad"],
                linestyle=linestyle,
                linewidth=2.8,
                label=f"{direction} mean",
            )
        ax.set_title(f"{joint} - {tendon}: directional moment arm")
        ax.set_xlabel(f"{joint} flexion angle [deg]")
        ax.set_ylabel("Moment arm [mm/rad]")
        ax.grid(True, alpha=0.3)
        ax.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(plot_root / f"{joint.lower()}_{tendon.lower()}_directional_moment_arm.png", dpi=220)
        plt.close(fig)


def main() -> None:
    args = parse_args()
    _validate_args(args)
    input_root = _resolve_input_root(args.input_root)
    output_root = _resolve_output_root(args.output_root)
    recordings = _discover_recordings(input_root)
    range_specs = _determine_range_specs(
        recordings,
        fit_margin_deg=args.fit_margin_deg,
        comparison_margin_deg=args.comparison_margin_deg,
    )

    print("\nRange specification by joint:")
    for joint in ("DIP", "PIP", "MCP"):
        spec = range_specs[joint]
        print(
            f"{joint}: human={spec['human_angle_min_deg']:.2f}..{spec['human_angle_max_deg']:.2f}, "
            f"robot common p01-p99={spec['robot_common_p01_min_deg']:.2f}.."
            f"{spec['robot_common_p99_max_deg']:.2f}, "
            f"fit={spec['fitting_angle_min_deg']:.2f}..{spec['fitting_angle_max_deg']:.2f}, "
            f"compare target={spec['comparison_target_min_deg']:.2f}.."
            f"{spec['comparison_target_max_deg']:.2f}"
        )

    all_results: list[dict[str, object]] = []
    recording_rows: list[dict[str, object]] = []
    fit_rows: list[dict[str, object]] = []
    for path, joint, trial in recordings:
        fitting_range_deg = (
            range_specs[joint]["fitting_angle_min_deg"],
            range_specs[joint]["fitting_angle_max_deg"],
        )
        results, recording_summary = _load_recording(
            path,
            joint,
            trial,
            args,
            fitting_range_deg=fitting_range_deg,
        )
        recording_rows.append(recording_summary)
        all_results.extend(results.values())
        fit_rows.extend(_fit_row(result) for result in results.values())
        print(
            f"{path.name}: {joint} trial {trial}, "
            f"flexion={recording_summary['flexion_points_in_range']}, "
            f"extension={recording_summary['extension_points_in_range']}"
        )

    curve_rows, stability_rows = _curve_rows_and_stability(
        all_results,
        args,
        range_specs,
    )
    output_root.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(recording_rows).to_csv(output_root / "direction_recording_summary.csv", index=False)
    pd.DataFrame(fit_rows).sort_values(
        ["joint", "tendon", "direction", "trial"]
    ).to_csv(output_root / "directional_moment_arm_trial_summary.csv", index=False)
    curve_df = pd.DataFrame(curve_rows)
    curve_df.to_csv(output_root / "directional_moment_arm_curves.csv", index=False)
    stability_df = pd.DataFrame(stability_rows).sort_values(["joint", "tendon"])
    stability_df.to_csv(output_root / "directional_moment_arm_stability.csv", index=False)
    if not args.no_plots and not curve_df.empty:
        _plot_curves(curve_df, output_root)

    print("\nDirectional stability summary:")
    if stability_df.empty:
        print("No joint/tendon combination had a usable common angle range.")
    else:
        columns = [
            "joint",
            "tendon",
            "expected_stable_direction",
            "observed_more_stable_direction",
            "hypothesis_supported_descriptively",
            "flexion_mean_moment_arm_sd_mm_per_rad",
            "extension_mean_moment_arm_sd_mm_per_rad",
            "expected_to_other_sd_ratio",
            "common_angle_min_deg",
            "common_angle_max_deg",
        ]
        print(stability_df[columns].to_string(index=False, float_format=lambda value: f"{value:.4f}"))
    print(f"\nSaved outputs under {output_root}")


if __name__ == "__main__":
    main()
