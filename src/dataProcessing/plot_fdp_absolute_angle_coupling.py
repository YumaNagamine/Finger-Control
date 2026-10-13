from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from src.dataProcessing import plot_all_flexion_angle_260907a_comparison as source


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = source.FINAL_ROOT
DEFAULT_260907A_ROOT = source.DEFAULT_260907A_ROOT
DEFAULT_OUTPUT_ROOT = (
    DEFAULT_INPUT_ROOT / "figures" / "fdp_absolute_angle_coupling"
)
REPRESENTATIVE_FRACTIONS = (0.0, 0.25, 0.50, 0.75, 1.00)
RATIO_DENOMINATOR_THRESHOLD_DEG = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot FDP DIP/PIP coupling as an absolute flexion-angle ratio "
            "for 2607B, 2608A, and 260907A."
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


def _trial_angle_curve(
    trial: dict[str, object],
    joint: str,
) -> tuple[np.ndarray, np.ndarray]:
    x = trial["relative_excursion_mm"]
    y = trial["angles_deg"][joint]
    mask = trial["valid_excursion"] & np.isfinite(x) & np.isfinite(y)
    return _sorted_unique_curve(x[mask], y[mask])


def _load_trials(input_root: Path, original_260907a_root: Path) -> list[dict[str, object]]:
    trials: list[dict[str, object]] = []
    for finger in source.FINGER_IDS:
        paths = source._find_paths(
            finger,
            "FDP",
            input_root,
            original_260907a_root,
        )
        trials.extend(
            source._load_trial(path, finger=finger, tendon="FDP")
            for path in paths
        )
    return trials


def _calculate_ratio(
    trials: list[dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    common_max = min(
        float(np.max(trial["relative_excursion_mm"][trial["valid_excursion"]]))
        for trial in trials
    )
    grid = np.linspace(0.0, common_max, 201)
    representative_grid = np.asarray(REPRESENTATIVE_FRACTIONS) * common_max
    curve_rows: list[dict[str, object]] = []
    representative_rows: list[dict[str, object]] = []
    finger_ratios: dict[str, np.ndarray] = {}

    for finger in source.FINGER_IDS:
        finger_trials = [trial for trial in trials if trial["finger"] == finger]
        dip_curves = np.vstack(
            [np.interp(grid, *_trial_angle_curve(trial, "DIP")) for trial in finger_trials]
        )
        pip_curves = np.vstack(
            [np.interp(grid, *_trial_angle_curve(trial, "PIP")) for trial in finger_trials]
        )

        for trial_index, trial in enumerate(finger_trials):
            trial_ratio = np.full(grid.shape, np.nan)
            denominator_ok = np.abs(pip_curves[trial_index]) >= RATIO_DENOMINATOR_THRESHOLD_DEG
            trial_ratio[denominator_ok] = (
                np.abs(dip_curves[trial_index, denominator_ok])
                / np.abs(pip_curves[trial_index, denominator_ok])
            )
            for x_value, dip, pip, ratio in zip(
                grid,
                dip_curves[trial_index],
                pip_curves[trial_index],
                trial_ratio,
            ):
                curve_rows.append(
                    {
                        "level": "trial",
                        "finger": finger,
                        "trial": int(trial["trial"]),
                        "relative_excursion_mm": float(x_value),
                        "DIP_flexion_angle_deg": float(dip),
                        "PIP_flexion_angle_deg": float(pip),
                        "absolute_DIP_over_PIP_ratio": float(ratio)
                        if np.isfinite(ratio)
                        else np.nan,
                    }
                )

        mean_dip = np.mean(dip_curves, axis=0)
        mean_pip = np.mean(pip_curves, axis=0)
        ratio = np.full(grid.shape, np.nan)
        denominator_ok = np.abs(mean_pip) >= RATIO_DENOMINATOR_THRESHOLD_DEG
        ratio[denominator_ok] = (
            np.abs(mean_dip[denominator_ok]) / np.abs(mean_pip[denominator_ok])
        )
        finger_ratios[finger] = ratio
        for x_value, dip, pip, ratio_value in zip(grid, mean_dip, mean_pip, ratio):
            curve_rows.append(
                {
                    "level": "finger_mean",
                    "finger": finger,
                    "trial": "mean",
                    "relative_excursion_mm": float(x_value),
                    "DIP_flexion_angle_deg": float(dip),
                    "PIP_flexion_angle_deg": float(pip),
                    "absolute_DIP_over_PIP_ratio": float(ratio_value)
                    if np.isfinite(ratio_value)
                    else np.nan,
                }
            )

        dip_rep = np.interp(representative_grid, grid, mean_dip)
        pip_rep = np.interp(representative_grid, grid, mean_pip)
        ratio_rep = np.interp(representative_grid, grid, ratio)
        for fraction, x_value, dip, pip, ratio_value in zip(
            REPRESENTATIVE_FRACTIONS,
            representative_grid,
            dip_rep,
            pip_rep,
            ratio_rep,
        ):
            representative_rows.append(
                {
                    "level": "finger_mean",
                    "finger": finger,
                    "fraction": fraction,
                    "relative_excursion_mm": float(x_value),
                    "DIP_flexion_angle_deg": float(dip),
                    "PIP_flexion_angle_deg": float(pip),
                    "absolute_DIP_over_PIP_ratio": float(ratio_value)
                    if np.isfinite(ratio_value)
                    else np.nan,
                }
            )

    ratio_stack = np.vstack([finger_ratios[finger] for finger in source.FINGER_IDS])
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
                "DIP_flexion_angle_deg": np.nan,
                "PIP_flexion_angle_deg": np.nan,
                "absolute_DIP_over_PIP_ratio": float(ratio_value)
                if np.isfinite(ratio_value)
                else np.nan,
            }
        )
    all_rep = np.interp(representative_grid, grid, all_finger_ratio)
    for fraction, x_value, ratio_value in zip(
        REPRESENTATIVE_FRACTIONS,
        representative_grid,
        all_rep,
    ):
        representative_rows.append(
            {
                "level": "all_finger_mean",
                "finger": "all",
                "fraction": fraction,
                "relative_excursion_mm": float(x_value),
                "DIP_flexion_angle_deg": np.nan,
                "PIP_flexion_angle_deg": np.nan,
                "absolute_DIP_over_PIP_ratio": float(ratio_value)
                if np.isfinite(ratio_value)
                else np.nan,
            }
        )
    return pd.DataFrame(curve_rows), pd.DataFrame(representative_rows)


def _plot(
    curve_df: pd.DataFrame,
    representative_df: pd.DataFrame,
    output_path: Path,
) -> None:
    figure, axis = plt.subplots(figsize=(8.5, 5.5))
    for finger in source.FINGER_IDS:
        values = curve_df.query("level == 'finger_mean' and finger == @finger")
        axis.plot(
            values["relative_excursion_mm"],
            values["absolute_DIP_over_PIP_ratio"],
            color=source.FINGER_COLORS[finger],
            linewidth=2.0,
            label=finger,
        )
        points = representative_df.query("level == 'finger_mean' and finger == @finger")
        axis.plot(
            points["relative_excursion_mm"],
            points["absolute_DIP_over_PIP_ratio"],
            color=source.FINGER_COLORS[finger],
            marker="o",
            linestyle="none",
            markersize=4,
        )

    all_values = curve_df.query("level == 'all_finger_mean'")
    axis.plot(
        all_values["relative_excursion_mm"],
        all_values["absolute_DIP_over_PIP_ratio"],
        color="#222222",
        linewidth=2.8,
        label="all-finger mean",
    )
    all_points = representative_df.query("level == 'all_finger_mean'")
    axis.plot(
        all_points["relative_excursion_mm"],
        all_points["absolute_DIP_over_PIP_ratio"],
        color="#222222",
        marker="o",
        linestyle="none",
        markersize=5,
    )
    axis.grid(True, alpha=0.25)
    axis.set_xlabel("relative FDP excursion [mm]")
    axis.set_ylabel("|DIP flexion angle| / |PIP flexion angle|")
    axis.set_title(
        "FDP DIP–PIP coupling ratio\n"
        "absolute flexion-angle ratio; per-finger and all-finger means"
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

    trials = _load_trials(input_root, original_260907a_root)
    curve_df, representative_df = _calculate_ratio(trials)
    _plot(
        curve_df,
        representative_df,
        output_root / "fdp_absolute_angle_coupling_ratio.png",
    )
    curve_df.to_csv(output_root / "fdp_absolute_angle_coupling_ratio_curve.csv", index=False)
    representative_df.to_csv(
        output_root / "fdp_absolute_angle_coupling_ratio_representative_points.csv",
        index=False,
    )
    print(f"Loaded {len(trials)} FDP trials")
    print(f"Saved outputs under {output_root}")


if __name__ == "__main__":
    main()
