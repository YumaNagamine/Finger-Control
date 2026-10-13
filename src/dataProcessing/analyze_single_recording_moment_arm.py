from __future__ import annotations

import argparse
import json
import math
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import LSQUnivariateSpline, PchipInterpolator, make_smoothing_spline
from scipy.signal import savgol_filter
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.metrics import mean_squared_error, r2_score


PROJECT_ROOT = Path(__file__).resolve().parents[2]

# Change only this value to analyze another joint.
TARGET_JOINT = "MCP"  # "MCP", "PIP", or "DIP"

# Leave as None to use the newest processed CSV whose name contains TARGET_JOINT.
INPUT_CSV: Path | None = None
INPUT_ROOT = PROJECT_ROOT / "logs" / "dual_camera" / "processed_csv"
OUTPUT_ROOT = PROJECT_ROOT / "logs" / "dual_camera" / "single_recording_moment_arm"

ANGLE_RANGES_DEG = {
    "MCP": (-10.0, 90.0),
    "PIP": (0.0, 90.0),
    "DIP": (-10.0, 40.0),
}
TENDONS = ("FDP", "FDS", "ED", "DI", "PI", "LUM")
EXCURSION_COLUMNS = {
    "FDP": "weight_0_disp_mm",
    "FDS": "weight_1_disp_mm",
    "ED": "weight_2_disp_mm",
    "PI": "weight_3_disp_mm",
    "DI": "weight_4_disp_mm",
    "LUM": "weight_5_disp_mm",
}

SMOOTHING_DURATION_S = 1.0
SAVGOL_POLYORDER = 3
ANGLE_BIN_WIDTH_DEG = 2.0
MIN_SAMPLES_PER_BIN = 5
MIN_ANGULAR_SPEED_DEG_S = 2.0
OUTLIER_MAD_MULTIPLIER = 4.0
SPLINE_KNOT_SPACINGS_DEG = (5.0, 10.0, 15.0, 20.0, 30.0)

METHODS = (
    ("poly2", "2nd-order polynomial"),
    ("poly3", "3rd-order polynomial"),
    ("poly4", "4th-order polynomial"),
    ("smoothing_spline", "Smoothing spline"),
    ("piecewise_cubic", "Piecewise cubic spline"),
    ("gpr", "Gaussian process regression"),
    ("pchip", "PCHIP"),
)

Predictor = Callable[[np.ndarray], np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze one repeated-motion recording as a single pooled dataset."
    )
    parser.add_argument("--csv", type=Path, default=None)
    parser.add_argument("--output-root", type=Path, default=OUTPUT_ROOT)
    parser.add_argument("--joint", choices=tuple(ANGLE_RANGES_DEG), default=TARGET_JOINT)
    return parser.parse_args()


def _resolve_input_csv(cli_path: Path | None) -> Path:
    if cli_path is not None:
        path = cli_path.expanduser().resolve()
    elif INPUT_CSV is not None:
        path = INPUT_CSV.expanduser().resolve()
    else:
        candidates = sorted(
            INPUT_ROOT.glob(f"*{TARGET_JOINT}*.csv"),
            key=lambda candidate: candidate.stat().st_mtime,
            reverse=True,
        )
        if not candidates:
            raise FileNotFoundError(
                f"No processed CSV containing {TARGET_JOINT!r} was found under {INPUT_ROOT}"
            )
        path = candidates[0].resolve()
    if not path.is_file():
        raise FileNotFoundError(f"Processed CSV not found: {path}")
    return path


def _odd_window(sample_interval_s: float, sample_count: int) -> int:
    window = max(
        SAVGOL_POLYORDER + 2,
        int(round(SMOOTHING_DURATION_S / sample_interval_s)),
    )
    if window % 2 == 0:
        window += 1
    maximum = sample_count if sample_count % 2 == 1 else sample_count - 1
    window = min(window, maximum)
    if window <= SAVGOL_POLYORDER:
        raise ValueError("Too few samples for Savitzky-Golay smoothing")
    return window


def _numeric(frame: pd.DataFrame, column: str) -> np.ndarray:
    return pd.to_numeric(frame[column], errors="coerce").to_numpy(dtype=np.float64)


def _validate_columns(frame: pd.DataFrame) -> None:
    required = ["elapsed_s", TARGET_JOINT, *EXCURSION_COLUMNS.values()]
    missing = [column for column in required if column not in frame.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")


def _resample_uniformly(frame: pd.DataFrame) -> tuple[pd.DataFrame, float]:
    columns = ["elapsed_s", TARGET_JOINT, *EXCURSION_COLUMNS.values()]
    numeric = frame.loc[:, columns].apply(pd.to_numeric, errors="coerce")
    numeric = numeric.dropna(subset=["elapsed_s"]).sort_values("elapsed_s")
    numeric = numeric.groupby("elapsed_s", as_index=False).mean(numeric_only=True)
    elapsed = numeric["elapsed_s"].to_numpy(dtype=np.float64)
    positive_dt = np.diff(elapsed)
    positive_dt = positive_dt[np.isfinite(positive_dt) & (positive_dt > 0.0)]
    if positive_dt.size == 0:
        raise ValueError("No positive time interval was found")
    sample_interval_s = float(np.median(positive_dt))
    uniform_time = np.arange(elapsed[0], elapsed[-1] + sample_interval_s * 0.5, sample_interval_s)
    result = pd.DataFrame({"elapsed_s": uniform_time})
    for column in columns[1:]:
        values = numeric[column].to_numpy(dtype=np.float64)
        valid = np.isfinite(elapsed) & np.isfinite(values)
        if int(np.count_nonzero(valid)) < 2:
            result[column] = np.nan
        else:
            result[column] = np.interp(uniform_time, elapsed[valid], values[valid])
    return result, sample_interval_s


def _plot_raw_joint_excursion(frame: pd.DataFrame, output_path: Path) -> None:
    angle_deg = 180.0 - _numeric(frame, TARGET_JOINT)
    angle_min, angle_max = ANGLE_RANGES_DEG[TARGET_JOINT]
    fig, axes = plt.subplots(3, 2, figsize=(14, 13), sharex=True)
    for axis, tendon in zip(axes.flat, TENDONS):
        excursion = _numeric(frame, EXCURSION_COLUMNS[tendon])
        valid = (
            np.isfinite(angle_deg)
            & np.isfinite(excursion)
            & (angle_deg >= angle_min)
            & (angle_deg <= angle_max)
        )
        axis.scatter(angle_deg[valid], excursion[valid], s=9, alpha=0.32, color="#4C78A8")
        axis.set_title(tendon)
        axis.set_ylabel("Raw excursion [mm]")
        axis.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel(f"{TARGET_JOINT} flexion angle [deg]")
    axes[-1, 1].set_xlabel(f"{TARGET_JOINT} flexion angle [deg]")
    fig.suptitle(
        f"{TARGET_JOINT} flexion angle vs tendon excursion: all raw samples",
        fontsize=15,
    )
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _instantaneous_table(
    uniform: pd.DataFrame,
    sample_interval_s: float,
) -> tuple[pd.DataFrame, int]:
    window = _odd_window(sample_interval_s, len(uniform))
    elapsed = _numeric(uniform, "elapsed_s")
    angle_deg = 180.0 - _numeric(uniform, TARGET_JOINT)
    angle_smooth_deg = savgol_filter(
        angle_deg, window, SAVGOL_POLYORDER, mode="interp"
    )
    angular_speed_deg_s = savgol_filter(
        angle_deg,
        window,
        SAVGOL_POLYORDER,
        deriv=1,
        delta=sample_interval_s,
        mode="interp",
    )
    angular_speed_rad_s = np.deg2rad(angular_speed_deg_s)
    result = pd.DataFrame(
        {
            "elapsed_s": elapsed,
            "flexion_angle_deg": angle_smooth_deg,
            "flexion_angle_rad": np.deg2rad(angle_smooth_deg),
            "angular_speed_deg_s": angular_speed_deg_s,
            "direction": np.where(angular_speed_deg_s >= 0.0, "flexion", "extension"),
        }
    )
    angle_min, angle_max = ANGLE_RANGES_DEG[TARGET_JOINT]
    base_valid = (
        np.isfinite(angle_smooth_deg)
        & np.isfinite(angular_speed_deg_s)
        & (angle_smooth_deg >= angle_min)
        & (angle_smooth_deg <= angle_max)
        & (np.abs(angular_speed_deg_s) >= MIN_ANGULAR_SPEED_DEG_S)
    )
    for tendon in TENDONS:
        excursion = _numeric(uniform, EXCURSION_COLUMNS[tendon])
        excursion_smooth = savgol_filter(
            excursion, window, SAVGOL_POLYORDER, mode="interp"
        )
        excursion_speed = savgol_filter(
            excursion,
            window,
            SAVGOL_POLYORDER,
            deriv=1,
            delta=sample_interval_s,
            mode="interp",
        )
        moment_arm = np.full_like(excursion_speed, np.nan)
        valid = base_valid & np.isfinite(excursion_speed) & np.isfinite(angular_speed_rad_s)
        moment_arm[valid] = excursion_speed[valid] / angular_speed_rad_s[valid]
        result[f"{tendon}_excursion_smoothed_mm"] = excursion_smooth
        result[f"{tendon}_excursion_speed_mm_s"] = excursion_speed
        result[f"{tendon}_moment_arm_mm_per_rad"] = moment_arm
    result["base_valid"] = base_valid
    return result, window


def _bin_moment_arms(instantaneous: pd.DataFrame) -> pd.DataFrame:
    angle_min, _ = ANGLE_RANGES_DEG[TARGET_JOINT]
    frames: list[pd.DataFrame] = []
    angles = instantaneous["flexion_angle_deg"].to_numpy(dtype=np.float64)
    directions = instantaneous["direction"].to_numpy(dtype=str)
    bin_ids = np.floor((angles - angle_min) / ANGLE_BIN_WIDTH_DEG).astype(np.int64)
    for tendon in TENDONS:
        values = instantaneous[f"{tendon}_moment_arm_mm_per_rad"].to_numpy(dtype=np.float64)
        valid = np.isfinite(angles) & np.isfinite(values)
        points = pd.DataFrame(
            {
                "bin": bin_ids[valid],
                "angle_deg": angles[valid],
                "moment_arm_mm_per_rad": values[valid],
                "direction": directions[valid],
            }
        )
        rows: list[dict[str, float | int | str]] = []
        for bin_id, group in points.groupby("bin", sort=True):
            values_in_bin = group["moment_arm_mm_per_rad"].to_numpy(dtype=np.float64)
            center = float(np.median(values_in_bin))
            mad = float(np.median(np.abs(values_in_bin - center)))
            if mad > 0.0:
                robust_sigma = 1.4826 * mad
                group = group.loc[
                    np.abs(group["moment_arm_mm_per_rad"] - center)
                    <= OUTLIER_MAD_MULTIPLIER * robust_sigma
                ]
            if len(group) < MIN_SAMPLES_PER_BIN:
                continue
            kept = group["moment_arm_mm_per_rad"].to_numpy(dtype=np.float64)
            flexion_values = group.loc[
                group["direction"] == "flexion", "moment_arm_mm_per_rad"
            ].to_numpy(dtype=np.float64)
            extension_values = group.loc[
                group["direction"] == "extension", "moment_arm_mm_per_rad"
            ].to_numpy(dtype=np.float64)
            q25, q75 = np.quantile(kept, [0.25, 0.75])
            rows.append(
                {
                    "joint": TARGET_JOINT,
                    "tendon": tendon,
                    "bin": int(bin_id),
                    "angle_deg": float(group["angle_deg"].median()),
                    "moment_arm_median": float(np.median(kept)),
                    "moment_arm_mean": float(np.mean(kept)),
                    "moment_arm_std": float(np.std(kept, ddof=1)) if len(kept) > 1 else 0.0,
                    "moment_arm_q25": float(q25),
                    "moment_arm_q75": float(q75),
                    "sample_count": int(len(kept)),
                    "flexion_count": int(len(flexion_values)),
                    "extension_count": int(len(extension_values)),
                    "flexion_median": (
                        float(np.median(flexion_values)) if len(flexion_values) else np.nan
                    ),
                    "extension_median": (
                        float(np.median(extension_values)) if len(extension_values) else np.nan
                    ),
                }
            )
        frames.append(pd.DataFrame(rows))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _all_moment_arm_points(instantaneous: pd.DataFrame) -> pd.DataFrame:
    """Keep every valid point after applying the existing per-bin MAD filter."""
    angle_min, _ = ANGLE_RANGES_DEG[TARGET_JOINT]
    frames: list[pd.DataFrame] = []
    angles = instantaneous["flexion_angle_deg"].to_numpy(dtype=np.float64)
    bin_ids = np.floor((angles - angle_min) / ANGLE_BIN_WIDTH_DEG).astype(np.int64)
    for tendon in TENDONS:
        values = instantaneous[f"{tendon}_moment_arm_mm_per_rad"].to_numpy(dtype=np.float64)
        valid = np.isfinite(angles) & np.isfinite(values)
        points = pd.DataFrame(
            {
                "joint": TARGET_JOINT,
                "tendon": tendon,
                "bin": bin_ids[valid],
                "angle_deg": angles[valid],
                "moment_arm_mm_per_rad": values[valid],
            }
        )
        kept_groups: list[pd.DataFrame] = []
        for _, group in points.groupby("bin", sort=True):
            values_in_bin = group["moment_arm_mm_per_rad"].to_numpy(dtype=np.float64)
            center = float(np.median(values_in_bin))
            mad = float(np.median(np.abs(values_in_bin - center)))
            if mad > 0.0:
                robust_sigma = 1.4826 * mad
                group = group.loc[
                    np.abs(group["moment_arm_mm_per_rad"] - center)
                    <= OUTLIER_MAD_MULTIPLIER * robust_sigma
                ]
            if len(group) >= MIN_SAMPLES_PER_BIN:
                kept_groups.append(group)
        if kept_groups:
            frames.append(pd.concat(kept_groups, ignore_index=True))
    if not frames:
        return pd.DataFrame()
    return pd.concat(frames, ignore_index=True)


def _fit_piecewise_cubic(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    knot_spacing_deg: float,
) -> Predictor:
    knots = np.arange(
        math.ceil(x.min() / knot_spacing_deg) * knot_spacing_deg,
        x.max(),
        knot_spacing_deg,
    )
    knots = knots[(knots > x.min()) & (knots < x.max())]
    spline = LSQUnivariateSpline(x, y, knots, w=weights, k=3)
    return lambda values: np.asarray(spline(values), dtype=np.float64)


def _fit_method(
    method: str,
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> Predictor:
    if method.startswith("poly"):
        degree = int(method[-1])
        coefficients = np.polyfit(x, y, degree, w=weights)
        return lambda values: np.polyval(coefficients, values)
    if method == "smoothing_spline":
        spline = make_smoothing_spline(x, y, w=weights, lam=None)
        return lambda values: np.asarray(spline(values), dtype=np.float64)
    if method == "piecewise_cubic":
        return _fit_piecewise_cubic(x, y, weights, knot_spacing_deg=15.0)
    if method == "gpr":
        kernel = (
            ConstantKernel(1.0, (1e-2, 1e2))
            * RBF(20.0, (2.0, 200.0))
            + WhiteKernel(0.05, (1e-5, 10.0))
        )
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=0.05**2,
            normalize_y=True,
            n_restarts_optimizer=0,
            random_state=0,
        )
        model.fit(x[:, None], y)
        return lambda values: model.predict(np.asarray(values)[:, None])
    if method == "pchip":
        interpolator = PchipInterpolator(x, y, extrapolate=False)
        return lambda values: np.asarray(interpolator(values), dtype=np.float64)
    raise ValueError(f"Unsupported fitting method: {method}")


def _cross_validated_metrics(
    method: str,
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
) -> tuple[float, float, int]:
    predictions = np.full_like(y, np.nan)
    fold_ids = np.arange(len(x)) % 5
    for fold in range(5):
        test = fold_ids == fold
        train = ~test
        if int(np.count_nonzero(train)) < 8:
            continue
        try:
            predictor = _fit_method(method, x[train], y[train], weights[train])
            predictions[test] = predictor(x[test])
        except (ValueError, np.linalg.LinAlgError):
            continue
    valid = np.isfinite(predictions)
    if int(np.count_nonzero(valid)) < 2:
        return float("nan"), float("nan"), int(np.count_nonzero(valid))
    return (
        float(mean_squared_error(y[valid], predictions[valid]) ** 0.5),
        float(r2_score(y[valid], predictions[valid])),
        int(np.count_nonzero(valid)),
    )


def _cross_validated_piecewise_metrics(
    x: np.ndarray,
    y: np.ndarray,
    weights: np.ndarray,
    knot_spacing_deg: float,
) -> tuple[float, float, int]:
    predictions = np.full_like(y, np.nan)
    fold_ids = np.arange(len(x)) % 5
    for fold in range(5):
        test = fold_ids == fold
        train = ~test
        if int(np.count_nonzero(train)) < 8:
            continue
        try:
            predictor = _fit_piecewise_cubic(
                x[train], y[train], weights[train], knot_spacing_deg
            )
            predictions[test] = predictor(x[test])
        except (ValueError, np.linalg.LinAlgError):
            continue
    valid = np.isfinite(predictions)
    if int(np.count_nonzero(valid)) < 2:
        return float("nan"), float("nan"), int(np.count_nonzero(valid))
    return (
        float(mean_squared_error(y[valid], predictions[valid]) ** 0.5),
        float(r2_score(y[valid], predictions[valid])),
        int(np.count_nonzero(valid)),
    )


def _save_piecewise_interval_comparisons(
    all_points: pd.DataFrame,
    binned: pd.DataFrame,
    piecewise_root: Path,
) -> None:
    interval_metrics: list[dict[str, float | int | str]] = []
    for knot_spacing_deg in SPLINE_KNOT_SPACINGS_DEG:
        spacing_label = f"{int(knot_spacing_deg):02d}deg"
        interval_dir = piecewise_root / f"interval_{spacing_label}"
        interval_dir.mkdir(parents=True, exist_ok=True)
        curve_frames: list[pd.DataFrame] = []
        fig, axes = plt.subplots(3, 2, figsize=(14, 13), sharex=True)
        for axis, tendon in zip(axes.flat, TENDONS):
            points = all_points.loc[all_points["tendon"] == tendon].sort_values("angle_deg")
            summaries = binned.loc[binned["tendon"] == tendon].sort_values("angle_deg")
            x = points["angle_deg"].to_numpy(dtype=np.float64)
            y = points["moment_arm_mm_per_rad"].to_numpy(dtype=np.float64)
            weights = np.ones_like(x)
            grid = np.linspace(float(x.min()), float(x.max()), 500)
            try:
                predictor = _fit_piecewise_cubic(
                    x, y, weights, knot_spacing_deg
                )
                values = predictor(grid)
                rmse, r_squared, evaluated_bins = _cross_validated_piecewise_metrics(
                    x, y, weights, knot_spacing_deg
                )
            except (ValueError, np.linalg.LinAlgError):
                values = np.full_like(grid, np.nan)
                rmse, r_squared, evaluated_bins = float("nan"), float("nan"), 0
            interval_metrics.append(
                {
                    "joint": TARGET_JOINT,
                    "tendon": tendon,
                    "knot_spacing_deg": knot_spacing_deg,
                    "cv_rmse_mm_per_rad": rmse,
                    "cv_r2": r_squared,
                    "evaluated_bins": evaluated_bins,
                    "fitting_points": len(points),
                    "populated_bins": len(summaries),
                }
            )
            axis.scatter(
                x, y, s=5, color="#777777", alpha=0.16, label="All fitting points"
            )
            axis.plot(
                grid,
                values,
                color="#0072B2",
                linewidth=2.3,
                label=f"Cubic spline, {knot_spacing_deg:g} deg knots",
            )
            axis.axhline(0.0, color="#C00000", linestyle="--", linewidth=1.0)
            axis.set_title(tendon)
            axis.set_ylabel("Moment arm [mm/rad]")
            axis.grid(True, alpha=0.3)
            axis.legend(fontsize=8)
            curve_frames.append(
                pd.DataFrame(
                    {
                        "joint": TARGET_JOINT,
                        "tendon": tendon,
                        "knot_spacing_deg": knot_spacing_deg,
                        "angle_deg": grid,
                        "moment_arm_mm_per_rad": values,
                    }
                )
            )
        axes[-1, 0].set_xlabel(f"{TARGET_JOINT} flexion angle [deg]")
        axes[-1, 1].set_xlabel(f"{TARGET_JOINT} flexion angle [deg]")
        fig.suptitle(
            f"{TARGET_JOINT}: piecewise cubic spline, {knot_spacing_deg:g} deg knot spacing",
            fontsize=15,
        )
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
        fig.savefig(
            interval_dir
            / f"{TARGET_JOINT}_piecewise_cubic_{spacing_label}_all_tendons.png",
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(fig)
        pd.concat(curve_frames, ignore_index=True).to_csv(
            interval_dir / f"{TARGET_JOINT}_piecewise_cubic_{spacing_label}_curves.csv",
            index=False,
        )
    pd.DataFrame(interval_metrics).to_csv(
        piecewise_root / f"{TARGET_JOINT}_knot_spacing_metrics.csv",
        index=False,
    )


def _fit_and_plot(
    all_points: pd.DataFrame,
    binned: pd.DataFrame,
    output_dir: Path,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    metrics_rows: list[dict[str, float | int | str]] = []
    curve_frames: list[pd.DataFrame] = []
    curves_by_tendon: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
    for tendon in TENDONS:
        points = all_points.loc[all_points["tendon"] == tendon].sort_values("angle_deg")
        summaries = binned.loc[binned["tendon"] == tendon].sort_values("angle_deg")
        if len(points) < 8:
            raise ValueError(f"Too few fitting points for {TARGET_JOINT}-{tendon}: {len(points)}")
        x = points["angle_deg"].to_numpy(dtype=np.float64)
        y = points["moment_arm_mm_per_rad"].to_numpy(dtype=np.float64)
        weights = np.ones_like(x)
        grid = np.linspace(float(x.min()), float(x.max()), 500)
        curves: dict[str, np.ndarray] = {}
        for method, title in METHODS:
            try:
                predictor = _fit_method(method, x, y, weights)
                values = predictor(grid)
                rmse, r_squared, evaluated_bins = _cross_validated_metrics(
                    method, x, y, weights
                )
            except (ValueError, np.linalg.LinAlgError):
                values = np.full_like(grid, np.nan)
                rmse, r_squared, evaluated_bins = float("nan"), float("nan"), 0
            curves[method] = values
            metrics_rows.append(
                {
                    "joint": TARGET_JOINT,
                    "tendon": tendon,
                    "method": method,
                    "cv_rmse_mm_per_rad": rmse,
                    "cv_r2": r_squared,
                    "evaluated_bins": evaluated_bins,
                    "fitting_points": len(points),
                    "populated_bins": len(summaries),
                }
            )
        curve_frame = pd.DataFrame(
            {"joint": TARGET_JOINT, "tendon": tendon, "angle_deg": grid}
        )
        for method, _ in METHODS:
            curve_frame[method] = curves[method]
        curve_frames.append(curve_frame)
        curves_by_tendon[tendon] = {
            method: (grid, curves[method]) for method, _ in METHODS
        }

        fig, axis = plt.subplots(figsize=(11, 7))
        axis.scatter(x, y, s=7, color="#777777", alpha=0.18, label="All fitting points")
        for method, title in METHODS:
            axis.plot(grid, curves[method], linewidth=1.8, label=title)
        axis.axhline(0.0, color="#C00000", linestyle="--", linewidth=1.0)
        axis.set_xlabel(f"{TARGET_JOINT} flexion angle [deg]")
        axis.set_ylabel("Moment arm [mm/rad]")
        axis.set_title(f"{TARGET_JOINT}-{tendon}: pooled moment-arm fitting comparison")
        axis.grid(True, alpha=0.3)
        axis.legend(fontsize=8, ncol=2)
        fig.tight_layout()
        fig.savefig(output_dir / f"{TARGET_JOINT}_{tendon}_fitting_comparison.png", dpi=220)
        plt.close(fig)

    method_plot_root = output_dir / "method_plots"
    method_plot_root.mkdir(parents=True, exist_ok=True)
    all_curves = pd.concat(curve_frames, ignore_index=True)
    for method, method_title in METHODS:
        method_dir = method_plot_root / method
        method_dir.mkdir(parents=True, exist_ok=True)
        method_curve_rows: list[pd.DataFrame] = []
        fig, axes = plt.subplots(3, 2, figsize=(14, 13), sharex=True)
        for axis, tendon in zip(axes.flat, TENDONS):
            points = all_points.loc[all_points["tendon"] == tendon]
            axis.scatter(
                points["angle_deg"], points["moment_arm_mm_per_rad"],
                s=5, color="#777777", alpha=0.16, label="All fitting points",
            )
            grid, values = curves_by_tendon[tendon][method]
            axis.plot(
                grid,
                values,
                linewidth=2.3,
                color="#0072B2",
                label=method_title,
            )
            axis.axhline(0.0, color="#C00000", linestyle="--", linewidth=1.0)
            axis.set_title(tendon)
            axis.set_ylabel("Moment arm [mm/rad]")
            axis.grid(True, alpha=0.3)
            axis.legend(fontsize=8)
            method_curve_rows.append(
                pd.DataFrame(
                    {
                        "joint": TARGET_JOINT,
                        "tendon": tendon,
                        "angle_deg": grid,
                        "moment_arm_mm_per_rad": values,
                    }
                )
            )
        axes[-1, 0].set_xlabel(f"{TARGET_JOINT} flexion angle [deg]")
        axes[-1, 1].set_xlabel(f"{TARGET_JOINT} flexion angle [deg]")
        fig.suptitle(f"{TARGET_JOINT}: {method_title}", fontsize=15)
        fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
        fig.savefig(
            method_dir / f"{TARGET_JOINT}_{method}_all_tendons.png",
            dpi=220,
            bbox_inches="tight",
        )
        plt.close(fig)
        pd.concat(method_curve_rows, ignore_index=True).to_csv(
            method_dir / f"{TARGET_JOINT}_{method}_curves.csv",
            index=False,
        )

    _save_piecewise_interval_comparisons(
        all_points,
        binned,
        method_plot_root / "piecewise_cubic",
    )

    return pd.DataFrame(metrics_rows), all_curves


def analyze(csv_path: Path, output_root: Path) -> dict[str, Path]:
    if TARGET_JOINT not in ANGLE_RANGES_DEG:
        raise ValueError(f"Unsupported TARGET_JOINT: {TARGET_JOINT}")
    frame = pd.read_csv(csv_path)
    _validate_columns(frame)
    output_dir = output_root.expanduser().resolve() / TARGET_JOINT / csv_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    raw_plot_path = output_dir / f"{TARGET_JOINT}_joint_excursion_raw.png"
    if not raw_plot_path.exists():
        _plot_raw_joint_excursion(frame, raw_plot_path)
    uniform, sample_interval_s = _resample_uniformly(frame)
    instantaneous, smoothing_window_samples = _instantaneous_table(
        uniform, sample_interval_s
    )
    instantaneous_path = output_dir / "instantaneous_moment_arm.csv"
    instantaneous.to_csv(instantaneous_path, index=False)
    binned = _bin_moment_arms(instantaneous)
    if binned.empty:
        raise ValueError("No angle bins contained enough valid instantaneous samples")
    binned_path = output_dir / "angle_binned_moment_arm.csv"
    binned.to_csv(binned_path, index=False)
    all_points = _all_moment_arm_points(instantaneous)
    if all_points.empty:
        raise ValueError("No valid instantaneous points remained for fitting")
    all_points_path = output_dir / "all_fitting_points.csv"
    all_points.to_csv(all_points_path, index=False)
    metrics, curves = _fit_and_plot(all_points, binned, output_dir)
    metrics_path = output_dir / "fitting_metrics.csv"
    curves_path = output_dir / "fitted_moment_arm_curves.csv"
    metrics.to_csv(metrics_path, index=False)
    curves.to_csv(curves_path, index=False)

    observed_angle = 180.0 - _numeric(frame, TARGET_JOINT)
    finite_angle = observed_angle[np.isfinite(observed_angle)]
    metadata = {
        "input_csv": str(csv_path),
        "target_joint": TARGET_JOINT,
        "configured_angle_range_deg": list(ANGLE_RANGES_DEG[TARGET_JOINT]),
        "observed_angle_range_deg": [float(finite_angle.min()), float(finite_angle.max())],
        "sample_interval_s": sample_interval_s,
        "smoothing_duration_s": SMOOTHING_DURATION_S,
        "smoothing_window_samples": smoothing_window_samples,
        "savgol_polyorder": SAVGOL_POLYORDER,
        "minimum_angular_speed_deg_s": MIN_ANGULAR_SPEED_DEG_S,
        "angle_bin_width_deg": ANGLE_BIN_WIDTH_DEG,
        "minimum_samples_per_bin": MIN_SAMPLES_PER_BIN,
        "fitting_methods": [method for method, _ in METHODS],
        "piecewise_cubic_knot_spacings_deg": list(SPLINE_KNOT_SPACINGS_DEG),
        "cycle_segmentation": False,
        "fitting_input": "all instantaneous points after per-bin MAD filtering",
    }
    metadata_path = output_dir / "analysis_metadata.json"
    metadata_path.write_text(json.dumps(metadata, indent=2), encoding="utf-8")
    return {
        "output_dir": output_dir,
        "raw_plot": raw_plot_path,
        "instantaneous_csv": instantaneous_path,
        "binned_csv": binned_path,
        "all_fitting_points_csv": all_points_path,
        "metrics_csv": metrics_path,
        "curves_csv": curves_path,
        "metadata": metadata_path,
    }


def main() -> None:
    global TARGET_JOINT
    args = parse_args()
    TARGET_JOINT = args.joint
    csv_path = _resolve_input_csv(args.csv)
    result = analyze(csv_path, args.output_root)
    print(json.dumps({key: str(value) for key, value in result.items()}, indent=2))


if __name__ == "__main__":
    main()
