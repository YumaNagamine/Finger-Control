from __future__ import annotations

import warnings
from pathlib import Path
from typing import Callable

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.interpolate import PchipInterpolator, make_smoothing_spline
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import ConstantKernel, RBF, WhiteKernel
from sklearn.isotonic import IsotonicRegression


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_ROOT = PROJECT_ROOT / "logs" / "dual_camera-finger1" / "processed_data"
OUTPUT_ROOT = (
    PROJECT_ROOT
    / "logs"
    / "dual_camera-finger1"
    / "dip_fitting_comparison"
    / "mean_moment_arm"
)
JOINT = "DIP"
TENDON = "FDP"
ANGLE_COLUMN = "DIP_flexion_angle_degree"
ANGLE_MIN_DEG = 0.0
ANGLE_MAX_DEG = 90.0
GRID_DEG = np.linspace(ANGLE_MIN_DEG, ANGLE_MAX_DEG, 361)
RAD_PER_DEG = np.pi / 180.0
PCHIP_BIN_WIDTH_DEG = 2.0

METHODS = (
    ("poly2", "2nd-order polynomial"),
    ("poly3", "3rd-order polynomial"),
    ("poly4", "4th-order polynomial"),
    ("smoothing_spline", "Smoothing spline"),
    ("gpr", "Gaussian process regression"),
    ("monotone_pchip", "Monotone regression + PCHIP"),
)

Predictor = Callable[[np.ndarray], np.ndarray]


def _unique_xy(x_deg: np.ndarray, y_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    grouped = (
        pd.DataFrame({"x": x_deg, "y": y_mm})
        .groupby("x", as_index=False, sort=True)["y"]
        .mean()
    )
    return grouped["x"].to_numpy(dtype=np.float64), grouped["y"].to_numpy(dtype=np.float64)


def _load_trials() -> list[tuple[str, np.ndarray, np.ndarray]]:
    trials: list[tuple[str, np.ndarray, np.ndarray]] = []
    for csv_path in sorted(
        PROCESSED_DATA_ROOT.glob("*DIP*/processed_csv.csv"),
        key=lambda path: path.parent.name.lower(),
    ):
        dataframe = pd.read_csv(csv_path)
        x_deg = pd.to_numeric(dataframe[ANGLE_COLUMN], errors="coerce").to_numpy(dtype=np.float64)
        y_mm = pd.to_numeric(dataframe[TENDON], errors="coerce").to_numpy(dtype=np.float64)
        valid = np.isfinite(x_deg) & np.isfinite(y_mm)
        x_deg, y_mm = _unique_xy(x_deg[valid], y_mm[valid])
        if len(x_deg) < 5:
            raise ValueError(f"Insufficient valid points: {csv_path}")
        trials.append((csv_path.parent.name, x_deg, y_mm))
    if not trials:
        raise FileNotFoundError(f"No DIP processed data found under {PROCESSED_DATA_ROOT}")
    return trials


def _polynomial_derivative(x_deg: np.ndarray, y_mm: np.ndarray, degree: int) -> Predictor:
    coefficients = np.polyfit(x_deg, y_mm, degree)
    derivative_coefficients = np.polyder(coefficients)
    return lambda values: np.polyval(derivative_coefficients, values) / RAD_PER_DEG


def _smoothing_spline_derivative(x_deg: np.ndarray, y_mm: np.ndarray) -> Predictor:
    spline = make_smoothing_spline(x_deg, y_mm, lam=None)
    derivative = spline.derivative()
    return lambda values: np.asarray(derivative(values), dtype=np.float64) / RAD_PER_DEG


def _gpr_derivative(x_deg: np.ndarray, y_mm: np.ndarray) -> Predictor:
    kernel = (
        ConstantKernel(1.0, (1e-2, 1e2)) * RBF(20.0, (2.0, 200.0))
        + WhiteKernel(0.05, (1e-5, 10.0))
    )
    model = GaussianProcessRegressor(
        kernel=kernel,
        normalize_y=True,
        n_restarts_optimizer=0,
        random_state=0,
    )
    model.fit(x_deg[:, None], y_mm)

    def derivative(values: np.ndarray) -> np.ndarray:
        values = np.asarray(values, dtype=np.float64)
        step_deg = 0.05
        upper = model.predict((values + step_deg)[:, None])
        lower = model.predict((values - step_deg)[:, None])
        return (upper - lower) / (2.0 * step_deg) / RAD_PER_DEG

    return derivative


def _monotone_pchip_derivative(x_deg: np.ndarray, y_mm: np.ndarray) -> Predictor:
    bin_ids = np.floor(x_deg / PCHIP_BIN_WIDTH_DEG).astype(np.int64)
    binned = (
        pd.DataFrame({"bin": bin_ids, "x": x_deg, "y": y_mm})
        .groupby("bin", as_index=False)
        .agg(x=("x", "median"), y=("y", "median"))
        .sort_values("x")
    )
    binned_x = binned["x"].to_numpy(dtype=np.float64)
    binned_y = binned["y"].to_numpy(dtype=np.float64)
    isotonic = IsotonicRegression(increasing=True, out_of_bounds="clip")
    monotone_y = isotonic.fit_transform(binned_x, binned_y)
    derivative = PchipInterpolator(binned_x, monotone_y, extrapolate=True).derivative()
    return lambda values: np.asarray(derivative(values), dtype=np.float64) / RAD_PER_DEG


def _build_derivative(method: str, x_deg: np.ndarray, y_mm: np.ndarray) -> Predictor:
    if method.startswith("poly"):
        return _polynomial_derivative(x_deg, y_mm, int(method[-1]))
    if method == "smoothing_spline":
        return _smoothing_spline_derivative(x_deg, y_mm)
    if method == "gpr":
        return _gpr_derivative(x_deg, y_mm)
    if method == "monotone_pchip":
        return _monotone_pchip_derivative(x_deg, y_mm)
    raise ValueError(f"Unsupported method: {method}")


def _evaluate_methods(
    trials: list[tuple[str, np.ndarray, np.ndarray]],
) -> tuple[dict[str, np.ndarray], pd.DataFrame]:
    curves_by_method: dict[str, np.ndarray] = {}
    output = pd.DataFrame({"angle_deg": GRID_DEG})

    with warnings.catch_warnings():
        warnings.simplefilter("ignore")
        for method, _ in METHODS:
            trial_curves: list[np.ndarray] = []
            for trial_name, x_deg, y_mm in trials:
                derivative = _build_derivative(method, x_deg, y_mm)
                curve = derivative(GRID_DEG)
                trial_curves.append(curve)
                output[f"{method}__{trial_name}"] = curve
            curves_by_method[method] = np.asarray(trial_curves, dtype=np.float64)

    return curves_by_method, output


def _summary_curves(curves: np.ndarray) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=RuntimeWarning)
        mean = np.nanmean(curves, axis=0)
        lower = np.nanpercentile(curves, 10.0, axis=0)
        upper = np.nanpercentile(curves, 90.0, axis=0)
    return mean, lower, upper


def _draw_method(
    axis: plt.Axes,
    curves: np.ndarray,
    title: str,
    *,
    show_trial_labels: bool,
) -> None:
    for index, curve in enumerate(curves, start=1):
        axis.plot(
            GRID_DEG,
            curve,
            color="#888888",
            linewidth=0.9,
            alpha=0.45,
            label=f"Trial {index}" if show_trial_labels else "_nolegend_",
        )
    mean, lower, upper = _summary_curves(curves)
    axis.fill_between(GRID_DEG, lower, upper, color="#0072B2", alpha=0.18, label="10–90 percentile")
    axis.plot(GRID_DEG, mean, color="#0072B2", linewidth=2.8, label="Mean")
    axis.axhline(0.0, color="#C00000", linewidth=1.2, linestyle="--")
    axis.set_xlim(ANGLE_MIN_DEG, ANGLE_MAX_DEG)
    axis.set_title(title)
    axis.set_xlabel("DIP flexion angle [deg]")
    axis.set_ylabel("FDP moment arm [mm/rad]")
    axis.grid(True, alpha=0.3)


def _save_individual_plots(curves_by_method: dict[str, np.ndarray]) -> None:
    for method, title in METHODS:
        fig, axis = plt.subplots(figsize=(10, 6))
        _draw_method(axis, curves_by_method[method], title, show_trial_labels=False)
        axis.legend()
        fig.tight_layout()
        fig.savefig(OUTPUT_ROOT / f"dip_fdp_mean_moment_arm_{method}.png", dpi=220, bbox_inches="tight")
        plt.close(fig)


def _save_panel_plot(curves_by_method: dict[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(3, 2, figsize=(15, 15), sharex=True)
    for axis, (method, title) in zip(axes.flat, METHODS):
        _draw_method(axis, curves_by_method[method], title, show_trial_labels=False)
    handles, labels = axes.flat[0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=2)
    fig.suptitle("Mean DIP–FDP moment arm by fitting method", fontsize=16)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.96))
    fig.savefig(OUTPUT_ROOT / "dip_fdp_mean_moment_arm_all_methods.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def _save_mean_comparison(curves_by_method: dict[str, np.ndarray]) -> None:
    fig, axis = plt.subplots(figsize=(11, 7))
    for method, title in METHODS:
        mean, _, _ = _summary_curves(curves_by_method[method])
        axis.plot(GRID_DEG, mean, linewidth=2.2, label=title)
    axis.axhline(0.0, color="#C00000", linewidth=1.2, linestyle="--")
    axis.set_xlim(ANGLE_MIN_DEG, ANGLE_MAX_DEG)
    axis.set_xlabel("DIP flexion angle [deg]")
    axis.set_ylabel("Mean FDP moment arm [mm/rad]")
    axis.set_title("Mean DIP–FDP moment arm comparison")
    axis.grid(True, alpha=0.3)
    axis.legend()
    fig.tight_layout()
    fig.savefig(OUTPUT_ROOT / "dip_fdp_mean_moment_arm_comparison.png", dpi=220, bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    trials = _load_trials()
    curves_by_method, curve_table = _evaluate_methods(trials)

    for method, _ in METHODS:
        mean, lower, upper = _summary_curves(curves_by_method[method])
        curve_table[f"{method}__mean"] = mean
        curve_table[f"{method}__p10"] = lower
        curve_table[f"{method}__p90"] = upper

    curve_table.to_csv(OUTPUT_ROOT / "dip_fdp_mean_moment_arm_curves.csv", index=False)
    _save_individual_plots(curves_by_method)
    _save_panel_plot(curves_by_method)
    _save_mean_comparison(curves_by_method)
    print(f"Loaded {len(trials)} DIP trials")
    print(f"Saved comparison outputs under {OUTPUT_ROOT}")


if __name__ == "__main__":
    main()
