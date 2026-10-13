from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.lines as mlines
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.dataProcessing import plot_all_flexion_angle_260907a_comparison as source


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = source.FINAL_ROOT
DEFAULT_260907A_ROOT = source.DEFAULT_260907A_ROOT
DEFAULT_OUTPUT_ROOT = (
    DEFAULT_INPUT_ROOT / "figures" / "joint_overlay_and_fdp_coupling"
)
JOINT_COLORS = {
    "DIP": "#0072B2",
    "PIP": "#D55E00",
    "MCP": "#009E73",
}
JOINT_STYLES = {"DIP": "-", "PIP": "--", "MCP": ":"}
REPRESENTATIVE_FRACTIONS = (0.25, 0.50, 0.75, 1.00)
RATIO_DENOMINATOR_THRESHOLD_DEG = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot all-joint flexion-angle overlays and FDP DIP/PIP "
            "coupling ratios for 2607B, 2608A, and 260907A."
        )
    )
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--260907a-root", default=str(DEFAULT_260907A_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser.parse_args()


def _resolve_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _sorted_unique_curve(
    excursion: np.ndarray,
    angle: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    order = np.argsort(excursion, kind="stable")
    x = excursion[order]
    y = angle[order]
    unique_x, inverse = np.unique(x, return_inverse=True)
    sums = np.bincount(inverse, weights=y)
    counts = np.bincount(inverse)
    return unique_x, sums / counts


def _trial_curve(
    trial: dict[str, object],
    joint: str,
) -> tuple[np.ndarray, np.ndarray]:
    x = trial["relative_excursion_mm"]
    y = trial["angles_deg"][joint]
    mask = trial["valid_excursion"] & np.isfinite(x) & np.isfinite(y)
    return _sorted_unique_curve(x[mask], y[mask])


def _trial_displacement_curve(
    trial: dict[str, object],
    joint: str,
) -> tuple[np.ndarray, np.ndarray]:
    x = trial["relative_excursion_mm"]
    y = trial["angles_deg"][joint]
    mask = trial["valid_excursion"] & np.isfinite(x) & np.isfinite(y)
    # Baseline is the first valid point in time, before sorting by excursion.
    baseline = float(y[mask][0])
    return _sorted_unique_curve(x[mask], y[mask] - baseline)


def _common_x_limits(
    trials: list[dict[str, object]],
) -> dict[str, tuple[float, float]]:
    limits: dict[str, tuple[float, float]] = {}
    for tendon in source.TENDONS:
        maxima = []
        for trial in trials:
            if trial["tendon"] != tendon:
                continue
            x = trial["relative_excursion_mm"]
            mask = trial["valid_excursion"]
            maxima.append(float(np.max(x[mask])))
        common_max = min(maxima)
        if common_max <= 0.0:
            raise ValueError(f"Non-positive common {tendon} excursion range")
        limits[tendon] = (0.0, common_max)
    return limits


def _joint_y_limits(
    trials: list[dict[str, object]],
) -> dict[str, tuple[float, float]]:
    limits: dict[str, tuple[float, float]] = {}
    for tendon in source.TENDONS:
        values = []
        for trial in trials:
            if trial["tendon"] != tendon:
                continue
            for joint in source.JOINTS:
                x, y = _trial_curve(trial, joint)
                values.append(y[np.isfinite(x) & (x >= 0.0)])
        combined = np.concatenate(values)
        lower = float(np.min(combined))
        upper = float(np.max(combined))
        padding = max(1.0, 0.05 * (upper - lower))
        limits[tendon] = (lower - padding, upper + padding)
    return limits


def _legends(figure: plt.Figure) -> None:
    finger_handles = [
        mlines.Line2D(
            [0],
            [0],
            color=source.FINGER_COLORS[finger],
            linewidth=2.0,
            label=finger,
        )
        for finger in source.FINGER_IDS
    ]
    joint_handles = [
        mlines.Line2D(
            [0],
            [0],
            color="#444444",
            linestyle=JOINT_STYLES[joint],
            linewidth=1.8,
            label=joint,
        )
        for joint in source.JOINTS
    ]
    figure.legend(
        handles=finger_handles,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.30, 0.965),
        title="finger",
    )
    figure.legend(
        handles=joint_handles,
        loc="upper center",
        ncol=3,
        bbox_to_anchor=(0.72, 0.965),
        title="joint",
    )


def _plot_joint_overlay(
    trials: list[dict[str, object]],
    output_path: Path,
    mean_by_finger: bool,
) -> None:
    figure, axes = plt.subplots(
        nrows=1,
        ncols=len(source.TENDONS),
        figsize=(15.5, 5.2),
        squeeze=False,
    )
    x_limits = _common_x_limits(trials)
    y_limits = _joint_y_limits(trials)

    for column_index, tendon in enumerate(source.TENDONS):
        axis = axes[0, column_index]
        tendon_trials = [trial for trial in trials if trial["tendon"] == tendon]
        if mean_by_finger:
            x_grid = np.linspace(0.0, x_limits[tendon][1], 201)
            for finger in source.FINGER_IDS:
                finger_trials = [
                    trial
                    for trial in tendon_trials
                    if trial["finger"] == finger
                ]
                for joint in source.JOINTS:
                    curves = [
                        np.interp(x_grid, *_trial_curve(trial, joint))
                        for trial in finger_trials
                    ]
                    mean_curve = np.mean(np.vstack(curves), axis=0)
                    axis.plot(
                        x_grid,
                        mean_curve,
                        color=source.FINGER_COLORS[finger],
                        linestyle=JOINT_STYLES[joint],
                        linewidth=2.0,
                    )
        else:
            for trial in tendon_trials:
                for joint in source.JOINTS:
                    x, y = _trial_curve(trial, joint)
                    axis.plot(
                        x,
                        y,
                        color=source.FINGER_COLORS[trial["finger"]],
                        linestyle=JOINT_STYLES[joint],
                        linewidth=0.85,
                        alpha=0.35,
                    )

        axis.set_ylim(*y_limits[tendon])
        axis.axvline(0.0, color="#777777", linewidth=0.8, alpha=0.7)
        axis.grid(True, alpha=0.25)
        axis.set_xlim(*x_limits[tendon])
        axis.set_title(tendon)
        axis.set_xlabel(f"relative {tendon} excursion [mm]")
        if column_index == 0:
            axis.set_ylabel("flexion angle [deg]")

    _legends(figure)
    subtitle = (
        "all smooth trials; 260907C excluded and 260907A included"
        if not mean_by_finger
        else "mean of three smooth trials for each finger; 260907C excluded and 260907A included"
    )
    figure.suptitle(
        "All joints overlaid for each tendon\n" + subtitle,
        fontsize=14,
        y=1.02,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.84))
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _coupling_data(
    trials: list[dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    fdp_trials = [trial for trial in trials if trial["tendon"] == "FDP"]
    common_max = min(
        float(np.max(trial["relative_excursion_mm"][trial["valid_excursion"]]))
        for trial in fdp_trials
    )
    grid = np.linspace(0.0, common_max, 201)
    fractions = np.asarray(REPRESENTATIVE_FRACTIONS, dtype=float)
    representative_grid = fractions * common_max
    curve_rows: list[dict[str, object]] = []
    representative_rows: list[dict[str, object]] = []

    finger_mean_ratios: dict[str, np.ndarray] = {}
    for finger in source.FINGER_IDS:
        finger_trials = [
            trial for trial in fdp_trials if trial["finger"] == finger
        ]
        trial_dip = []
        trial_pip = []
        for trial in finger_trials:
            dip_x, dip_y = _trial_displacement_curve(trial, "DIP")
            pip_x, pip_y = _trial_displacement_curve(trial, "PIP")
            trial_dip.append(np.interp(grid, dip_x, dip_y))
            trial_pip.append(np.interp(grid, pip_x, pip_y))
        dip_matrix = np.vstack(trial_dip)
        pip_matrix = np.vstack(trial_pip)
        mean_dip = np.mean(dip_matrix, axis=0)
        mean_pip = np.mean(pip_matrix, axis=0)
        ratio = np.full(grid.shape, np.nan)
        stable = np.abs(mean_pip) >= RATIO_DENOMINATOR_THRESHOLD_DEG
        ratio[stable] = mean_dip[stable] / mean_pip[stable]
        finger_mean_ratios[finger] = ratio

        for trial_index, trial in enumerate(finger_trials):
            trial_ratio = np.full(grid.shape, np.nan)
            stable_trial = np.abs(pip_matrix[trial_index]) >= RATIO_DENOMINATOR_THRESHOLD_DEG
            trial_ratio[stable_trial] = (
                dip_matrix[trial_index, stable_trial]
                / pip_matrix[trial_index, stable_trial]
            )
            for x_value, dip, pip, ratio_value in zip(
                grid,
                dip_matrix[trial_index],
                pip_matrix[trial_index],
                trial_ratio,
            ):
                curve_rows.append(
                    {
                        "level": "trial",
                        "finger": finger,
                        "trial": int(trial["trial"]),
                        "relative_excursion_mm": float(x_value),
                        "DIP_displacement_deg": float(dip),
                        "PIP_displacement_deg": float(pip),
                        "DIP_over_PIP_ratio": float(ratio_value)
                        if np.isfinite(ratio_value)
                        else np.nan,
                    }
                )
        for x_value, dip, pip, ratio_value in zip(grid, mean_dip, mean_pip, ratio):
            curve_rows.append(
                {
                    "level": "finger_mean",
                    "finger": finger,
                    "trial": "mean",
                    "relative_excursion_mm": float(x_value),
                    "DIP_displacement_deg": float(dip),
                    "PIP_displacement_deg": float(pip),
                    "DIP_over_PIP_ratio": float(ratio_value)
                    if np.isfinite(ratio_value)
                    else np.nan,
                }
            )
        dip_rep = np.interp(representative_grid, grid, mean_dip)
        pip_rep = np.interp(representative_grid, grid, mean_pip)
        ratio_rep = np.interp(representative_grid, grid, ratio)
        for fraction, x_value, dip, pip, ratio_value in zip(
            fractions,
            representative_grid,
            dip_rep,
            pip_rep,
            ratio_rep,
        ):
            representative_rows.append(
                {
                    "level": "finger_mean",
                    "finger": finger,
                    "fraction": float(fraction),
                    "relative_excursion_mm": float(x_value),
                    "DIP_displacement_deg": float(dip),
                    "PIP_displacement_deg": float(pip),
                    "DIP_over_PIP_ratio": float(ratio_value)
                    if np.isfinite(ratio_value)
                    else np.nan,
                }
            )

    ratio_stack = np.vstack(
        [finger_mean_ratios[finger] for finger in source.FINGER_IDS]
    )
    all_finger_ratio = np.full(grid.shape, np.nan)
    all_fingers_stable = np.all(np.isfinite(ratio_stack), axis=0)
    all_finger_ratio[all_fingers_stable] = np.mean(
        ratio_stack[:, all_fingers_stable],
        axis=0,
    )
    for x_value, ratio_value in zip(grid, all_finger_ratio):
        curve_rows.append(
            {
                "level": "all_finger_mean",
                "finger": "all",
                "trial": "mean",
                "relative_excursion_mm": float(x_value),
                "DIP_displacement_deg": np.nan,
                "PIP_displacement_deg": np.nan,
                "DIP_over_PIP_ratio": float(ratio_value)
                if np.isfinite(ratio_value)
                else np.nan,
            }
        )
    all_rep = np.interp(representative_grid, grid, all_finger_ratio)
    for fraction, x_value, ratio_value in zip(fractions, representative_grid, all_rep):
        representative_rows.append(
            {
                "level": "all_finger_mean",
                "finger": "all",
                "fraction": float(fraction),
                "relative_excursion_mm": float(x_value),
                "DIP_displacement_deg": np.nan,
                "PIP_displacement_deg": np.nan,
                "DIP_over_PIP_ratio": float(ratio_value)
                if np.isfinite(ratio_value)
                else np.nan,
            }
        )
    return pd.DataFrame(curve_rows), pd.DataFrame(representative_rows)


def _plot_coupling(
    curve_df: pd.DataFrame,
    representative_df: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    for finger in source.FINGER_IDS:
        values = curve_df.query("level == 'finger_mean' and finger == @finger")
        axis.plot(
            values["relative_excursion_mm"],
            values["DIP_over_PIP_ratio"],
            color=source.FINGER_COLORS[finger],
            linewidth=2.0,
            label=finger,
        )
        points = representative_df.query("level == 'finger_mean' and finger == @finger")
        axis.plot(
            points["relative_excursion_mm"],
            points["DIP_over_PIP_ratio"],
            color=source.FINGER_COLORS[finger],
            marker="o",
            linestyle="none",
            markersize=4,
        )
    all_values = curve_df.query("level == 'all_finger_mean'")
    axis.plot(
        all_values["relative_excursion_mm"],
        all_values["DIP_over_PIP_ratio"],
        color="#222222",
        linewidth=2.8,
        label="all-finger mean",
    )
    all_points = representative_df.query("level == 'all_finger_mean'")
    axis.plot(
        all_points["relative_excursion_mm"],
        all_points["DIP_over_PIP_ratio"],
        color="#222222",
        marker="o",
        linestyle="none",
        markersize=5,
    )
    axis.axhline(0.0, color="#777777", linewidth=0.8, alpha=0.7)
    axis.grid(True, alpha=0.25)
    axis.set_xlabel("relative FDP excursion [mm]")
    axis.set_ylabel("ΔDIP / ΔPIP")
    axis.set_title(
        "FDP DIP–PIP coupling ratio\n"
        "per-finger trial mean and all-finger mean"
    )
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    input_root = _resolve_path(args.input_root)
    original_260907a_root = _resolve_path(args.__dict__["260907a_root"])
    output_root = _resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    trials: list[dict[str, object]] = []
    for finger in source.FINGER_IDS:
        for tendon in source.TENDONS:
            paths = source._find_paths(
                finger,
                tendon,
                input_root,
                original_260907a_root,
            )
            trials.extend(
                source._load_trial(path, finger=finger, tendon=tendon)
                for path in paths
            )

    _plot_joint_overlay(
        trials,
        output_root / "all_trials_joint_overlay.png",
        mean_by_finger=False,
    )
    _plot_joint_overlay(
        trials,
        output_root / "per_finger_trial_mean_joint_overlay.png",
        mean_by_finger=True,
    )
    curve_df, representative_df = _coupling_data(trials)
    _plot_coupling(
        curve_df,
        representative_df,
        output_root / "fdp_dip_pip_coupling_ratio.png",
    )
    curve_df.to_csv(output_root / "fdp_dip_pip_coupling_ratio_curve.csv", index=False)
    representative_df.to_csv(
        output_root / "fdp_dip_pip_coupling_ratio_representative_points.csv",
        index=False,
    )
    print(f"Loaded {len(trials)} trials")
    print(f"Saved outputs under {output_root}")


if __name__ == "__main__":
    main()
