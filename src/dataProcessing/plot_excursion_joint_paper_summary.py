from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.ticker import MultipleLocator

from src.dataProcessing import plot_flexion_angle_relative_excursion as source


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = source.DEFAULT_INPUT_ROOT
DEFAULT_OUTPUT_ROOT = (
    DEFAULT_INPUT_ROOT / "figures" / "excursion_joint_paper_summary"
)
LINE_COLOR = "#202020"
JOINT_COLORS = {
    "DIP": "#0072B2",
    "PIP": "#D55E00",
    "MCP": "#009E73",
}
JOINT_STYLES = {
    "DIP": {"linestyle": "-", "marker": "o"},
    "PIP": {"linestyle": "--", "marker": "s"},
    "MCP": {"linestyle": ":", "marker": "^"},
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot paper-ready excursion-joint curves as one all-finger mean "
            "per tendon with all three joints overlaid."
        )
    )
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--grid-points", type=int, default=201)
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
    valid = np.isfinite(excursion) & np.isfinite(angle)
    excursion = excursion[valid]
    angle = angle[valid]
    order = np.argsort(excursion, kind="stable")
    excursion = excursion[order]
    angle = angle[order]
    unique_excursion, inverse = np.unique(excursion, return_inverse=True)
    sums = np.bincount(inverse, weights=angle)
    counts = np.bincount(inverse)
    return unique_excursion, sums / counts


def _curve_on_grid(
    trial: dict[str, object],
    joint: str,
    grid: np.ndarray,
) -> np.ndarray:
    x = trial["relative_excursion_mm"]
    angles = trial["angles_deg"]
    mask = trial["active_excursion_mask"]
    assert isinstance(x, np.ndarray)
    assert isinstance(angles, dict)
    assert isinstance(mask, np.ndarray)
    x, y = _sorted_unique_curve(x[mask], angles[joint][mask])
    if x.size < 2:
        return np.full(grid.shape, np.nan)
    values = np.full(grid.shape, np.nan)
    inside = (grid >= x[0]) & (grid <= x[-1])
    values[inside] = np.interp(grid[inside], x, y)
    return values


def _aggregate_joint(
    tendon_trials: list[dict[str, object]],
    joint: str,
    grid: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    finger_means: dict[str, np.ndarray] = {}
    for finger in source.FINGER_IDS:
        finger_trials = [
            trial for trial in tendon_trials if trial["finger"] == finger
        ]
        trial_curves = np.vstack(
            [_curve_on_grid(trial, joint, grid) for trial in finger_trials]
        )
        finger_means[finger] = np.nanmean(trial_curves, axis=0)

    stacked = np.vstack([finger_means[finger] for finger in source.FINGER_IDS])
    grand_mean = np.nanmean(stacked, axis=0)
    between_finger_sd = (
        np.nanstd(stacked, axis=0, ddof=1)
        if stacked.shape[0] > 1
        else np.full(grid.shape, np.nan)
    )
    return grand_mean, between_finger_sd, finger_means


def _rounded_y_limits(curves: list[np.ndarray]) -> tuple[float, float]:
    finite = np.concatenate([curve[np.isfinite(curve)] for curve in curves])
    lower = float(np.floor(np.min(finite) / 20.0) * 20.0)
    upper = float(np.ceil(np.max(finite) / 20.0) * 20.0)
    if upper <= lower:
        upper = lower + 20.0
    return lower, upper


def _plot_tendon(
    tendon: str,
    tendon_trials: list[dict[str, object]],
    output_path: Path,
    curve_rows: list[dict[str, object]],
    grid_points: int,
) -> None:
    common_max = min(
        float(
            np.max(
                trial["relative_excursion_mm"][trial["active_excursion_mask"]]
            )
        )
        for trial in tendon_trials
    )
    grid = np.linspace(0.0, common_max, grid_points)
    means: dict[str, np.ndarray] = {}
    sds: dict[str, np.ndarray] = {}
    finger_means_by_joint: dict[str, dict[str, np.ndarray]] = {}

    for joint in source.JOINTS:
        mean_curve, sd_curve, finger_means = _aggregate_joint(
            tendon_trials,
            joint,
            grid,
        )
        means[joint] = mean_curve
        sds[joint] = sd_curve
        finger_means_by_joint[joint] = finger_means
        for index, excursion in enumerate(grid):
            row: dict[str, object] = {
                "tendon": tendon,
                "joint": joint,
                "relative_excursion_mm": float(excursion),
                "all_finger_mean_deg": float(mean_curve[index]),
                "between_finger_sd_deg": float(sd_curve[index]),
                "finger_count": len(source.FINGER_IDS),
                "trial_count_per_finger": 3,
                "common_excursion_max_mm": common_max,
            }
            for finger in source.FINGER_IDS:
                row[f"{finger}_trial_mean_deg"] = float(
                    finger_means[finger][index]
                )
            curve_rows.append(row)

    y_min, y_max = _rounded_y_limits(list(means.values()))
    figure, axis = plt.subplots(figsize=(8.2, 5.8))

    for joint in source.JOINTS:
        style = JOINT_STYLES[joint]
        mean_curve = means[joint]
        sd_curve = sds[joint]
        for finger in source.FINGER_IDS:
            finger_curve = finger_means_by_joint[joint][finger]
            finite_finger = np.isfinite(finger_curve)
            axis.plot(
                grid[finite_finger],
                finger_curve[finite_finger],
                color=JOINT_COLORS[joint],
                linewidth=1.0,
                linestyle=style["linestyle"],
                alpha=0.85,
                label="_nolegend_",
                zorder=1,
            )
        finite = np.isfinite(mean_curve)
        axis.plot(
            grid[finite],
            mean_curve[finite],
            color=JOINT_COLORS[joint],
            linewidth=1.35,
            linestyle=style["linestyle"],
            marker=style["marker"],
            markersize=4.0,
            markerfacecolor="white",
            markeredgecolor=LINE_COLOR,
            markeredgewidth=0.8,
            markevery=max(1, grid_points // 10),
            label=joint,
            zorder=2,
        )

    axis.set_xlim(0.0, common_max)
    axis.set_ylim(y_min, y_max)
    axis.yaxis.set_major_locator(MultipleLocator(20.0))
    axis.grid(True, which="major", axis="both", alpha=0.25)
    axis.axvline(0.0, color="#666666", linewidth=0.7, alpha=0.7)
    axis.set_xlabel(f"relative {tendon} excursion [mm]")
    axis.set_ylabel("flexion angle [deg]")
    axis.set_title(
        f"{tendon}: excursion–joint relationship\n"
        "individual finger means and all-finger mean"
    )
    mean_handles = [
        plt.Line2D(
            [0],
            [0],
            color=JOINT_COLORS[joint],
            linewidth=1.35,
            linestyle=JOINT_STYLES[joint]["linestyle"],
            marker=JOINT_STYLES[joint]["marker"],
            markersize=4.0,
            markerfacecolor="white",
            markeredgecolor=LINE_COLOR,
            markeredgewidth=0.8,
            label=f"{joint} mean",
        )
        for joint in source.JOINTS
    ]
    axis.legend(
        handles=mean_handles,
        frameon=False,
        ncol=3,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    if args.grid_points < 3:
        raise ValueError("--grid-points must be at least 3.")

    input_root = _resolve_path(args.input_root)
    output_root = _resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    trials = source._discover_trials(input_root)
    curve_rows: list[dict[str, object]] = []

    for tendon in source.TENDONS:
        tendon_trials = [trial for trial in trials if trial["tendon"] == tendon]
        if len(tendon_trials) != len(source.FINGER_IDS) * 3:
            raise ValueError(
                f"Expected 3 trials for each finger in {tendon}; "
                f"found {len(tendon_trials)} total trials."
            )
        _plot_tendon(
            tendon,
            tendon_trials,
            output_root / f"excursion_joint_{tendon}_all_finger_mean.png",
            curve_rows,
            args.grid_points,
        )

    pd.DataFrame(curve_rows).to_csv(
        output_root / "excursion_joint_all_finger_mean_curves.csv",
        index=False,
    )
    print(f"Saved outputs under {output_root}")


if __name__ == "__main__":
    main()
