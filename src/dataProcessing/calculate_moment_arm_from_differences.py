from __future__ import annotations

import json
import math
import warnings
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
INPUT_ROOT = PROJECT_ROOT / "logs" / "dual_camera-finger1" / "processed_csv"
OUTPUT_ROOT = PROJECT_ROOT / "logs" / "dual_camera-finger1" / "moment_arm_from_differences"

JOINTS = ("DIP", "PIP", "MCP")
TENDONS = ("FDP", "FDS", "ED", "DI", "PI", "LUM")
EXCURSION_COLUMNS = {
    "FDP": "weight_0_disp_mm",
    "FDS": "weight_1_disp_mm",
    "ED": "weight_2_disp_mm",
    "PI": "weight_3_disp_mm",
    "DI": "weight_4_disp_mm",
    "LUM": "weight_5_disp_mm",
}
ANGLE_RANGES_DEG = {
    "DIP": (0.0, 90.0),
    "PIP": (0.0, 90.0),
    "MCP": (-30.0, 100.0),
}

SMOOTHING_DURATION_S = 0.6
SAVGOL_POLYORDER = 3
ANGLE_BIN_WIDTH_DEG = 2.0
MIN_ANGLE_THRESHOLD_DEG = 0.25
ANGLE_THRESHOLD_SIGMA_MULTIPLIER = 3.0
LOW_MOTION_QUANTILE = 0.2
OUTLIER_MAD_MULTIPLIER = 4.0
MIN_TRIALS_PER_BIN = 3

METHODS = (
    ("poly2", "2nd-order polynomial"),
    ("poly3", "3rd-order polynomial"),
    ("poly4", "4th-order polynomial"),
    ("smoothing_spline", "Smoothing spline"),
    ("piecewise_cubic", "Piecewise cubic LSQ spline"),
    ("gpr", "Gaussian process regression"),
    ("pchip", "PCHIP"),
)

Predictor = Callable[[np.ndarray], np.ndarray]


def _infer_joint(path: Path) -> str:
    matches = [joint for joint in JOINTS if joint in path.stem]
    if len(matches) != 1:
        raise ValueError(f"Could not infer one target joint from {path.name}")
    return matches[0]


def _interpolate(values: np.ndarray) -> np.ndarray:
    return (
        pd.Series(values, dtype=np.float64)
        .interpolate(limit=5, limit_direction="both")
        .to_numpy(dtype=np.float64)
    )


def _savgol_window(elapsed_s: np.ndarray, sample_count: int) -> int:
    positive_dt = np.diff(elapsed_s)
    positive_dt = positive_dt[np.isfinite(positive_dt) & (positive_dt > 0.0)]
    if positive_dt.size == 0:
        raise ValueError("No positive elapsed-time interval was found")
    window = max(SAVGOL_POLYORDER + 2, int(round(SMOOTHING_DURATION_S / np.median(positive_dt))))
    if window % 2 == 0:
        window += 1
    maximum = sample_count if sample_count % 2 == 1 else sample_count - 1
    window = min(window, maximum)
    if window <= SAVGOL_POLYORDER:
        raise ValueError("Too few samples for Savitzky-Golay smoothing")
    return window


def _smooth(values: np.ndarray, window: int) -> np.ndarray:
    return savgol_filter(
        _interpolate(values),
        window_length=window,
        polyorder=SAVGOL_POLYORDER,
        mode="interp",
    )


def _load_trials() -> dict[str, list[dict[str, object]]]:
    trials: dict[str, list[dict[str, object]]] = {joint: [] for joint in JOINTS}
    for csv_path in sorted(INPUT_ROOT.glob("*.csv"), key=lambda path: path.name.lower()):
        joint = _infer_joint(csv_path)
        dataframe = pd.read_csv(csv_path)
        required = ["elapsed_s", joint, *EXCURSION_COLUMNS.values()]
        missing = [column for column in required if column not in dataframe.columns]
        if missing:
            raise ValueError(f"Missing columns in {csv_path}: {missing}")

        elapsed_s = pd.to_numeric(dataframe["elapsed_s"], errors="coerce").to_numpy(dtype=np.float64)
        angle_deg = 180.0 - pd.to_numeric(dataframe[joint], errors="coerce").to_numpy(dtype=np.float64)
        window = _savgol_window(elapsed_s, len(dataframe))
        smoothed_angle_deg = _smooth(angle_deg, window)
        smoothed_excursions = {
            tendon: _smooth(
                pd.to_numeric(dataframe[column], errors="coerce").to_numpy(dtype=np.float64),
                window,
            )
            for tendon, column in EXCURSION_COLUMNS.items()
        }
        trials[joint].append(
            {
                "name": csv_path.stem,
                "window": window,
                "angle_deg": smoothed_angle_deg,
                "excursions_mm": smoothed_excursions,
            }
        )
    return trials


def _angle_thresholds(trials: dict[str, list[dict[str, object]]]) -> dict[str, float]:
    thresholds: dict[str, float] = {}
    for joint, joint_trials in trials.items():
        pooled_deltas = np.concatenate(
            [np.diff(np.asarray(trial["angle_deg"], dtype=np.float64)) for trial in joint_trials]
        )
        pooled_deltas = pooled_deltas[np.isfinite(pooled_deltas)]
        cutoff = float(np.quantile(np.abs(pooled_deltas), LOW_MOTION_QUANTILE))
        low_motion = pooled_deltas[np.abs(pooled_deltas) <= cutoff]
        center = float(np.median(low_motion))
        sigma = 1.4826 * float(np.median(np.abs(low_motion - center)))
        thresholds[joint] = max(
            MIN_ANGLE_THRESHOLD_DEG,
            ANGLE_THRESHOLD_SIGMA_MULTIPLIER * sigma,
        )
    return thresholds


def _trial_binned_moment_arms(
    trial: dict[str, object],
    joint: str,
    tendon: str,
    angle_threshold_deg: float,
) -> pd.DataFrame:
    angle_deg = np.asarray(trial["angle_deg"], dtype=np.float64)
    excursion_mm = np.asarray(trial["excursions_mm"][tendon], dtype=np.float64)
    delta_angle_deg = np.diff(angle_deg)
    delta_angle_rad = np.deg2rad(delta_angle_deg)
    delta_excursion_mm = np.diff(excursion_mm)
    midpoint_deg = 0.5 * (angle_deg[:-1] + angle_deg[1:])
    angle_min, angle_max = ANGLE_RANGES_DEG[joint]
    valid = (
        np.isfinite(midpoint_deg)
        & np.isfinite(delta_angle_rad)
        & np.isfinite(delta_excursion_mm)
        & (np.abs(delta_angle_deg) >= angle_threshold_deg)
        & (midpoint_deg >= angle_min)
        & (midpoint_deg <= angle_max)
    )
    moment_arm = delta_excursion_mm[valid] / delta_angle_rad[valid]
    angles = midpoint_deg[valid]
    bin_id = np.floor((angles - angle_min) / ANGLE_BIN_WIDTH_DEG).astype(np.int64)
    binned = (
        pd.DataFrame({"bin": bin_id, "angle_deg": angles, "moment_arm": moment_arm})
        .groupby("bin", as_index=False)
        .agg(angle_deg=("angle_deg", "median"), moment_arm=("moment_arm", "median"), samples=("moment_arm", "size"))
    )
    binned["trial"] = str(trial["name"])
    return binned


def _aggregate_trials(points: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, float | int]] = []
    for bin_id, group in points.groupby("bin", sort=True):
        values = group["moment_arm"].to_numpy(dtype=np.float64)
        center = float(np.median(values))
        mad_sigma = 1.4826 * float(np.median(np.abs(values - center)))
        if mad_sigma > 0.0:
            keep = np.abs(values - center) <= OUTLIER_MAD_MULTIPLIER * mad_sigma
            group = group.iloc[np.flatnonzero(keep)]
        if len(group) < MIN_TRIALS_PER_BIN:
            continue
        kept_values = group["moment_arm"].to_numpy(dtype=np.float64)
        rows.append(
            {
                "bin": int(bin_id),
                "angle_deg": float(group["angle_deg"].mean()),
                "moment_arm_mean": float(np.mean(kept_values)),
                "moment_arm_std": float(np.std(kept_values, ddof=1)) if len(kept_values) > 1 else 0.0,
                "trial_count": int(len(group)),
            }
        )
    return pd.DataFrame(rows)


def _fit_method(method: str, x: np.ndarray, y: np.ndarray, std: np.ndarray, count: np.ndarray) -> Predictor:
    if method.startswith("poly"):
        degree = int(method[-1])
        coefficients = np.polyfit(x, y, degree, w=np.sqrt(count))
        return lambda values: np.polyval(coefficients, values)
    if method == "smoothing_spline":
        weights = np.sqrt(count) / np.maximum(std, np.median(std[std > 0.0]) if np.any(std > 0.0) else 1.0)
        spline = make_smoothing_spline(x, y, w=weights, lam=None)
        return lambda values: np.asarray(spline(values), dtype=np.float64)
    if method == "piecewise_cubic":
        knots = np.arange(math.ceil(x.min() / 15.0) * 15.0, x.max(), 15.0)
        knots = knots[(knots > x.min()) & (knots < x.max())]
        spline = LSQUnivariateSpline(x, y, knots, w=np.sqrt(count), k=3)
        return lambda values: np.asarray(spline(values), dtype=np.float64)
    if method == "gpr":
        positive_std = std[std > 0.0]
        noise_floor = float(np.median(positive_std)) * 0.1 if positive_std.size else 0.05
        standard_error = std / np.sqrt(np.maximum(count, 1.0))
        alpha = np.maximum(standard_error, noise_floor) ** 2
        kernel = ConstantKernel(1.0, (1e-2, 1e2)) * RBF(20.0, (2.0, 200.0)) + WhiteKernel(0.05, (1e-5, 10.0))
        model = GaussianProcessRegressor(
            kernel=kernel,
            alpha=alpha,
            normalize_y=True,
            n_restarts_optimizer=0,
            random_state=0,
        )
        model.fit(x[:, None], y)
        return lambda values: model.predict(np.asarray(values, dtype=np.float64)[:, None])
    if method == "pchip":
        interpolator = PchipInterpolator(x, y, extrapolate=False)
        return lambda values: np.asarray(interpolator(values), dtype=np.float64)
    raise ValueError(f"Unsupported fitting method: {method}")


def _cross_validated_metrics(method: str, aggregate: pd.DataFrame) -> tuple[float, float]:
    x = aggregate["angle_deg"].to_numpy(dtype=np.float64)
    y = aggregate["moment_arm_mean"].to_numpy(dtype=np.float64)
    std = aggregate["moment_arm_std"].to_numpy(dtype=np.float64)
    count = aggregate["trial_count"].to_numpy(dtype=np.float64)
    predictions = np.full_like(y, np.nan)
    for fold in range(5):
        test = np.arange(len(x)) % 5 == fold
        train = ~test
        if int(np.count_nonzero(train)) < 8:
            continue
        predictor = _fit_method(method, x[train], y[train], std[train], count[train])
        predictions[test] = predictor(x[test])
    valid = np.isfinite(predictions)
    return (
        float(mean_squared_error(y[valid], predictions[valid]) ** 0.5),
        float(r2_score(y[valid], predictions[valid])),
    )


def _plot_pair(
    joint: str,
    tendon: str,
    points: pd.DataFrame,
    aggregate: pd.DataFrame,
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    output_path: Path,
) -> None:
    fig, axis = plt.subplots(figsize=(12, 7))
    for _, trial_points in points.groupby("trial"):
        axis.scatter(
            trial_points["angle_deg"],
            trial_points["moment_arm"],
            color="#999999",
            s=12,
            alpha=0.25,
            label="_nolegend_",
        )
    axis.errorbar(
        aggregate["angle_deg"],
        aggregate["moment_arm_mean"],
        yerr=aggregate["moment_arm_std"],
        fmt="o",
        color="#111111",
        markersize=3.5,
        linewidth=0.8,
        alpha=0.75,
        label="Trial mean ± SD",
    )
    for method, title in METHODS:
        grid, values = curves[method]
        axis.plot(grid, values, linewidth=2.0, label=title)
    axis.axhline(0.0, color="#C00000", linestyle="--", linewidth=1.1)
    axis.set_xlim(ANGLE_RANGES_DEG[joint])
    axis.set_xlabel(f"{joint} flexion angle [deg]")
    axis.set_ylabel(f"{tendon} moment arm [mm/rad]")
    axis.set_title(f"Difference-based moment arm fitting: {joint}–{tendon}")
    axis.grid(True, alpha=0.3)
    axis.legend(fontsize=8, ncol=2)
    fig.tight_layout()
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_joint_summary(
    joint: str,
    curves_by_tendon: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(15, 15), sharex=True)
    for axis, tendon in zip(axes.flat, TENDONS):
        for method, title in METHODS:
            grid, values = curves_by_tendon[tendon][method]
            axis.plot(grid, values, linewidth=1.8, label=title)
        axis.axhline(0.0, color="#C00000", linestyle="--", linewidth=1.0)
        axis.set_xlim(ANGLE_RANGES_DEG[joint])
        axis.set_title(tendon)
        axis.set_ylabel("Moment arm [mm/rad]")
        axis.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel(f"{joint} flexion angle [deg]")
    axes[-1, 1].set_xlabel(f"{joint} flexion angle [deg]")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=4, fontsize=9)
    fig.suptitle(f"{joint} difference-based moment arm fitting comparison", fontsize=16)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def _plot_joint_method(
    joint: str,
    method: str,
    method_title: str,
    points_by_tendon: dict[str, pd.DataFrame],
    aggregate_by_tendon: dict[str, pd.DataFrame],
    curves_by_tendon: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]],
    output_path: Path,
) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(15, 15), sharex=True)
    for axis, tendon in zip(axes.flat, TENDONS):
        points = points_by_tendon[tendon]
        aggregate = aggregate_by_tendon[tendon]
        for _, trial_points in points.groupby("trial"):
            axis.scatter(
                trial_points["angle_deg"],
                trial_points["moment_arm"],
                color="#999999",
                s=9,
                alpha=0.18,
                label="_nolegend_",
            )
        axis.errorbar(
            aggregate["angle_deg"],
            aggregate["moment_arm_mean"],
            yerr=aggregate["moment_arm_std"],
            fmt="o",
            color="#111111",
            markersize=3.0,
            linewidth=0.7,
            alpha=0.7,
            label="Trial mean ± SD",
        )
        grid, values = curves_by_tendon[tendon][method]
        axis.plot(
            grid,
            values,
            color="#0072B2",
            linewidth=2.5,
            label=method_title,
        )
        axis.axhline(0.0, color="#C00000", linestyle="--", linewidth=1.0)
        axis.set_xlim(ANGLE_RANGES_DEG[joint])
        axis.set_title(tendon)
        axis.set_ylabel("Moment arm [mm/rad]")
        axis.grid(True, alpha=0.3)
    axes[-1, 0].set_xlabel(f"{joint} flexion angle [deg]")
    axes[-1, 1].set_xlabel(f"{joint} flexion angle [deg]")
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2, fontsize=10)
    fig.suptitle(f"{joint} difference-based moment arm: {method_title}", fontsize=16)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.95))
    fig.savefig(output_path, dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    pair_plot_root = OUTPUT_ROOT / "pair_plots"
    pair_plot_root.mkdir(parents=True, exist_ok=True)
    method_plot_root = OUTPUT_ROOT / "method_plots"
    method_plot_root.mkdir(parents=True, exist_ok=True)
    trials = _load_trials()
    thresholds = _angle_thresholds(trials)
    metrics_rows: list[dict[str, object]] = []
    point_tables: list[pd.DataFrame] = []
    curve_rows: list[pd.DataFrame] = []

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for joint in JOINTS:
            curves_by_tendon: dict[str, dict[str, tuple[np.ndarray, np.ndarray]]] = {}
            points_by_tendon: dict[str, pd.DataFrame] = {}
            aggregate_by_tendon: dict[str, pd.DataFrame] = {}
            for tendon in TENDONS:
                trial_points = []
                for trial in trials[joint]:
                    trial_points.append(
                        _trial_binned_moment_arms(
                            trial,
                            joint,
                            tendon,
                            thresholds[joint],
                        )
                    )
                points = pd.concat(trial_points, ignore_index=True)
                points["joint"] = joint
                points["tendon"] = tendon
                point_tables.append(points)
                aggregate = _aggregate_trials(points)
                if len(aggregate) < 8:
                    raise ValueError(f"Too few aggregate bins for {joint}–{tendon}")

                x = aggregate["angle_deg"].to_numpy(dtype=np.float64)
                y = aggregate["moment_arm_mean"].to_numpy(dtype=np.float64)
                std = aggregate["moment_arm_std"].to_numpy(dtype=np.float64)
                count = aggregate["trial_count"].to_numpy(dtype=np.float64)
                grid = np.linspace(float(x.min()), float(x.max()), 500)
                curves: dict[str, tuple[np.ndarray, np.ndarray]] = {}
                curve_frame = pd.DataFrame({"joint": joint, "tendon": tendon, "angle_deg": grid})
                for method, _ in METHODS:
                    predictor = _fit_method(method, x, y, std, count)
                    values = predictor(grid)
                    curves[method] = (grid, values)
                    curve_frame[method] = values
                    cv_rmse, cv_r2 = _cross_validated_metrics(method, aggregate)
                    metrics_rows.append(
                        {
                            "joint": joint,
                            "tendon": tendon,
                            "method": method,
                            "angle_threshold_deg": thresholds[joint],
                            "aggregate_bins": len(aggregate),
                            "cv_rmse_mm_per_rad": cv_rmse,
                            "cv_r2": cv_r2,
                            "min_fitted_moment_arm": float(np.nanmin(values)),
                            "max_fitted_moment_arm": float(np.nanmax(values)),
                            "negative_fraction": float(np.mean(values < 0.0)),
                        }
                    )
                curve_rows.append(curve_frame)
                curves_by_tendon[tendon] = curves
                points_by_tendon[tendon] = points
                aggregate_by_tendon[tendon] = aggregate
                _plot_pair(
                    joint,
                    tendon,
                    points,
                    aggregate,
                    curves,
                    pair_plot_root / f"{joint}_{tendon}_moment_arm_fits.png",
                )
            _plot_joint_summary(
                joint,
                curves_by_tendon,
                OUTPUT_ROOT / f"{joint}_moment_arm_fitting_comparison.png",
            )
            joint_method_root = method_plot_root / joint
            joint_method_root.mkdir(parents=True, exist_ok=True)
            for method, method_title in METHODS:
                _plot_joint_method(
                    joint,
                    method,
                    method_title,
                    points_by_tendon,
                    aggregate_by_tendon,
                    curves_by_tendon,
                    joint_method_root / f"{joint}_{method}_moment_arm.png",
                )

    pd.concat(point_tables, ignore_index=True).to_csv(
        OUTPUT_ROOT / "trial_binned_moment_arm_points.csv",
        index=False,
    )
    pd.concat(curve_rows, ignore_index=True).to_csv(
        OUTPUT_ROOT / "fitted_moment_arm_curves.csv",
        index=False,
    )
    pd.DataFrame(metrics_rows).to_csv(OUTPUT_ROOT / "fitting_metrics.csv", index=False)
    metadata = {
        "input_root": str(INPUT_ROOT),
        "smoothing": {
            "method": "Savitzky-Golay",
            "duration_s": SMOOTHING_DURATION_S,
            "polyorder": SAVGOL_POLYORDER,
        },
        "angle_threshold_deg": thresholds,
        "angle_bin_width_deg": ANGLE_BIN_WIDTH_DEG,
        "minimum_trials_per_bin": MIN_TRIALS_PER_BIN,
        "fitting_methods": [method for method, _ in METHODS],
    }
    (OUTPUT_ROOT / "analysis_metadata.json").write_text(
        json.dumps(metadata, indent=2),
        encoding="utf-8",
    )
    print(json.dumps(metadata, indent=2))
    print(f"Saved results under {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
