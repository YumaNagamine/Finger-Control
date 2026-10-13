from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.dataProcessing import plot_flexion_angle_relative_excursion as source


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = source.DEFAULT_INPUT_ROOT
DEFAULT_OUTPUT_ROOT = (
    DEFAULT_INPUT_ROOT / "figures" / "flexion_angle_relative_excursion_trial_mean"
)
FINGER_COLORS = {
    "2607B": "#0072B2",
    "2608A": "#D55E00",
    "260907C": "#009E73",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay per-finger trial-mean flexion-angle curves for all "
            "tendons and joints."
        )
    )
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
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
    angles = trial["angles_deg"]
    mask = trial["active_excursion_mask"]
    assert isinstance(x, np.ndarray)
    assert isinstance(angles, dict)
    assert isinstance(mask, np.ndarray)
    y = angles[joint]
    valid = mask & np.isfinite(x) & np.isfinite(y)
    return _sorted_unique_curve(x[valid], y[valid])


def _common_x_limits(
    trials: list[dict[str, object]],
) -> dict[str, tuple[float, float]]:
    limits: dict[str, tuple[float, float]] = {}
    for tendon in source.TENDONS:
        tendon_trials = [trial for trial in trials if trial["tendon"] == tendon]
        maxima = []
        for trial in tendon_trials:
            x = trial["relative_excursion_mm"]
            mask = trial["active_excursion_mask"]
            assert isinstance(x, np.ndarray)
            assert isinstance(mask, np.ndarray)
            maxima.append(float(np.max(x[mask])))
        common_max = min(maxima)
        if common_max <= 0.0:
            raise ValueError(f"Non-positive common excursion range for {tendon}")
        limits[tendon] = (0.0, common_max)
    return limits


def _plot_comparison(
    trials: list[dict[str, object]],
    output_path: Path,
    curve_output_path: Path,
    x_limits: dict[str, tuple[float, float]],
) -> None:
    figure, axes = plt.subplots(
        nrows=len(source.TENDONS),
        ncols=len(source.JOINTS),
        figsize=(15.5, 11.0),
        squeeze=False,
    )
    curve_rows: list[dict[str, object]] = []

    y_limits: dict[tuple[str, str], tuple[float, float]] = {}
    for tendon in source.TENDONS:
        for joint in source.JOINTS:
            values = []
            for trial in trials:
                if trial["tendon"] != tendon:
                    continue
                x, y = _trial_curve(trial, joint)
                values.append(y)
            combined = np.concatenate(values)
            lower = float(np.min(combined))
            upper = float(np.max(combined))
            padding = max(1.0, 0.05 * (upper - lower))
            y_limits[(tendon, joint)] = (lower - padding, upper + padding)

    for row_index, tendon in enumerate(source.TENDONS):
        tendon_trials = [trial for trial in trials if trial["tendon"] == tendon]
        grid = np.linspace(0.0, x_limits[tendon][1], 201)
        for column_index, joint in enumerate(source.JOINTS):
            axis = axes[row_index, column_index]
            for finger in source.FINGER_IDS:
                finger_trials = [
                    trial
                    for trial in tendon_trials
                    if trial["finger"] == finger
                ]
                interpolated = np.vstack(
                    [
                        np.interp(grid, *_trial_curve(trial, joint))
                        for trial in finger_trials
                    ]
                )
                mean_curve = np.mean(interpolated, axis=0)
                axis.plot(
                    grid,
                    mean_curve,
                    color=FINGER_COLORS[finger],
                    linewidth=2.0,
                    label=finger,
                )
                for excursion, angle in zip(grid, mean_curve):
                    curve_rows.append(
                        {
                            "finger": finger,
                            "tendon": tendon,
                            "joint": joint,
                            "relative_excursion_mm": float(excursion),
                            "mean_flexion_angle_deg": float(angle),
                            "trial_count": int(len(finger_trials)),
                        }
                    )

            axis.axvline(0.0, color="#777777", linewidth=0.8, alpha=0.7)
            axis.grid(True, alpha=0.25)
            axis.set_xlim(*x_limits[tendon])
            axis.set_ylim(*y_limits[(tendon, joint)])
            if row_index == 0:
                axis.set_title(joint)
            axis.set_xlabel(f"relative {tendon} excursion [mm]")
            if column_index == 0:
                axis.set_ylabel(f"{tendon}\n{joint} flexion angle [deg]")

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=FINGER_COLORS[finger],
            linewidth=2.2,
            label=finger,
        )
        for finger in source.FINGER_IDS
    ]
    figure.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.965))
    figure.suptitle(
        "Per-finger mean flexion angle vs relative excursion\n"
        "mean of three smooth trials; no grand mean across fingers",
        fontsize=14,
        y=0.995,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    pd.DataFrame(curve_rows).to_csv(curve_output_path, index=False)


def main() -> None:
    args = parse_args()
    input_root = _resolve_path(args.input_root)
    output_root = _resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    trials = source._discover_trials(input_root)
    x_limits = _common_x_limits(trials)
    _plot_comparison(
        trials,
        output_root / "per_finger_trial_mean_flexion_angle.png",
        output_root / "per_finger_trial_mean_flexion_angle.csv",
        x_limits=x_limits,
    )
    print(f"Saved outputs under {output_root}")


if __name__ == "__main__":
    main()
