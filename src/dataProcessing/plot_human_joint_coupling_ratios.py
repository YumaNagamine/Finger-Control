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
    DEFAULT_INPUT_ROOT / "figures" / "human_joint_coupling_ratios"
)
RATIO_PAIRS = {
    "PIP_over_MCP": ("PIP", "MCP"),
    "DIP_over_MCP": ("DIP", "MCP"),
}
REPRESENTATIVE_FRACTIONS = (0.0, 0.25, 0.50, 0.75, 1.00)
RATIO_DENOMINATOR_THRESHOLD_DEG = 1.0


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Evaluate human-style PIP/MCP and DIP/MCP joint-coupling "
            "ratios for 2607B, 2608A, and 260907A."
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


def _interp_inside(
    grid: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    result = np.full(grid.shape, np.nan)
    if x.size < 2:
        return result
    inside = (grid >= x[0]) & (grid <= x[-1])
    result[inside] = np.interp(grid[inside], x, y)
    return result


def _load_trials(input_root: Path, original_260907a_root: Path) -> list[dict[str, object]]:
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
    return trials


def _target_ratio(pair: str, tendon: str) -> float:
    if pair == "PIP_over_MCP":
        return 1.4
    if tendon == "FDS":
        return 0.0
    return 0.8


def _calculate_ratios(
    trials: list[dict[str, object]],
) -> tuple[pd.DataFrame, pd.DataFrame, dict[str, tuple[float, float]]]:
    x_limits: dict[str, tuple[float, float]] = {}
    for tendon in source.TENDONS:
        maxima = []
        for trial in trials:
            if trial["tendon"] != tendon:
                continue
            x = trial["relative_excursion_mm"]
            mask = trial["valid_excursion"]
            maxima.append(float(np.max(x[mask])))
        x_limits[tendon] = (0.0, min(maxima))

    curve_rows: list[dict[str, object]] = []
    representative_rows: list[dict[str, object]] = []
    finger_ratios: dict[tuple[str, str, str], np.ndarray] = {}

    for pair, (numerator_joint, denominator_joint) in RATIO_PAIRS.items():
        for tendon in source.TENDONS:
            x_grid = np.linspace(0.0, x_limits[tendon][1], 201)
            target = _target_ratio(pair, tendon)
            tendon_trials = [trial for trial in trials if trial["tendon"] == tendon]

            for finger in source.FINGER_IDS:
                finger_trials = [
                    trial
                    for trial in tendon_trials
                    if trial["finger"] == finger
                ]
                numerator_curves = []
                denominator_curves = []
                ratio_curves = []
                for trial in finger_trials:
                    numerator_x, numerator_y = _trial_angle_curve(
                        trial,
                        numerator_joint,
                    )
                    denominator_x, denominator_y = _trial_angle_curve(
                        trial,
                        denominator_joint,
                    )
                    numerator = _interp_inside(x_grid, numerator_x, numerator_y)
                    denominator = _interp_inside(x_grid, denominator_x, denominator_y)
                    ratio = np.full(x_grid.shape, np.nan)
                    denominator_ok = np.abs(denominator) >= RATIO_DENOMINATOR_THRESHOLD_DEG
                    ratio[denominator_ok] = (
                        np.abs(numerator[denominator_ok])
                        / np.abs(denominator[denominator_ok])
                    )
                    numerator_curves.append(numerator)
                    denominator_curves.append(denominator)
                    ratio_curves.append(ratio)
                    for x_value, numerator_value, denominator_value, ratio_value in zip(
                        x_grid,
                        numerator,
                        denominator,
                        ratio,
                    ):
                        curve_rows.append(
                            {
                                "level": "trial",
                                "finger": finger,
                                "trial": int(trial["trial"]),
                                "tendon": tendon,
                                "ratio_pair": pair,
                                "relative_excursion_mm": float(x_value),
                                "numerator_joint": numerator_joint,
                                "denominator_joint": denominator_joint,
                                "numerator_angle_deg": float(numerator_value),
                                "denominator_angle_deg": float(denominator_value),
                                "ratio": float(ratio_value)
                                if np.isfinite(ratio_value)
                                else np.nan,
                                "human_target_ratio": target,
                            }
                        )

                ratio_matrix = np.vstack(ratio_curves)
                all_trials_stable = np.all(np.isfinite(ratio_matrix), axis=0)
                finger_ratio = np.full(x_grid.shape, np.nan)
                finger_ratio[all_trials_stable] = np.mean(
                    ratio_matrix[:, all_trials_stable],
                    axis=0,
                )
                finger_ratios[(pair, tendon, finger)] = finger_ratio
                mean_numerator = np.mean(np.vstack(numerator_curves), axis=0)
                mean_denominator = np.mean(np.vstack(denominator_curves), axis=0)
                for x_value, numerator_value, denominator_value, ratio_value in zip(
                    x_grid,
                    mean_numerator,
                    mean_denominator,
                    finger_ratio,
                ):
                    curve_rows.append(
                        {
                            "level": "finger_mean",
                            "finger": finger,
                            "trial": "mean",
                            "tendon": tendon,
                            "ratio_pair": pair,
                            "relative_excursion_mm": float(x_value),
                            "numerator_joint": numerator_joint,
                            "denominator_joint": denominator_joint,
                            "numerator_angle_deg": float(numerator_value),
                            "denominator_angle_deg": float(denominator_value),
                            "ratio": float(ratio_value)
                            if np.isfinite(ratio_value)
                            else np.nan,
                            "human_target_ratio": target,
                        }
                    )

                representative_grid = (
                    np.asarray(REPRESENTATIVE_FRACTIONS) * x_limits[tendon][1]
                )
                representative_ratio = _interp_inside(
                    representative_grid,
                    x_grid,
                    finger_ratio,
                )
                representative_numerator = _interp_inside(
                    representative_grid,
                    x_grid,
                    mean_numerator,
                )
                representative_denominator = _interp_inside(
                    representative_grid,
                    x_grid,
                    mean_denominator,
                )
                for fraction, x_value, numerator_value, denominator_value, ratio_value in zip(
                    REPRESENTATIVE_FRACTIONS,
                    representative_grid,
                    representative_numerator,
                    representative_denominator,
                    representative_ratio,
                ):
                    representative_rows.append(
                        {
                            "level": "finger_mean",
                            "finger": finger,
                            "tendon": tendon,
                            "ratio_pair": pair,
                            "fraction": float(fraction),
                            "relative_excursion_mm": float(x_value),
                            "numerator_angle_deg": float(numerator_value),
                            "denominator_angle_deg": float(denominator_value),
                            "ratio": float(ratio_value)
                            if np.isfinite(ratio_value)
                            else np.nan,
                            "human_target_ratio": target,
                            "ratio_error": float(ratio_value - target)
                            if np.isfinite(ratio_value)
                            else np.nan,
                        }
                    )

            ratio_stack = np.vstack(
                [
                    finger_ratios[(pair, tendon, finger)]
                    for finger in source.FINGER_IDS
                ]
            )
            all_fingers_stable = np.all(np.isfinite(ratio_stack), axis=0)
            all_finger_ratio = np.full(x_grid.shape, np.nan)
            all_finger_ratio[all_fingers_stable] = np.mean(
                ratio_stack[:, all_fingers_stable],
                axis=0,
            )
            for x_value, ratio_value in zip(x_grid, all_finger_ratio):
                curve_rows.append(
                    {
                        "level": "all_finger_mean",
                        "finger": "all",
                        "trial": "mean",
                        "tendon": tendon,
                        "ratio_pair": pair,
                        "relative_excursion_mm": float(x_value),
                        "numerator_joint": numerator_joint,
                        "denominator_joint": denominator_joint,
                        "numerator_angle_deg": np.nan,
                        "denominator_angle_deg": np.nan,
                        "ratio": float(ratio_value)
                        if np.isfinite(ratio_value)
                        else np.nan,
                        "human_target_ratio": target,
                    }
                )
            representative_grid = (
                np.asarray(REPRESENTATIVE_FRACTIONS) * x_limits[tendon][1]
            )
            representative_ratio = _interp_inside(
                representative_grid,
                x_grid,
                all_finger_ratio,
            )
            for fraction, x_value, ratio_value in zip(
                REPRESENTATIVE_FRACTIONS,
                representative_grid,
                representative_ratio,
            ):
                representative_rows.append(
                    {
                        "level": "all_finger_mean",
                        "finger": "all",
                        "tendon": tendon,
                        "ratio_pair": pair,
                        "fraction": float(fraction),
                        "relative_excursion_mm": float(x_value),
                        "numerator_angle_deg": np.nan,
                        "denominator_angle_deg": np.nan,
                        "ratio": float(ratio_value)
                        if np.isfinite(ratio_value)
                        else np.nan,
                        "human_target_ratio": target,
                        "ratio_error": float(ratio_value - target)
                        if np.isfinite(ratio_value)
                        else np.nan,
                    }
                )

    return pd.DataFrame(curve_rows), pd.DataFrame(representative_rows), x_limits


def _plot_pair(
    curve_df: pd.DataFrame,
    representative_df: pd.DataFrame,
    pair: str,
    output_path: Path,
    x_limits: dict[str, tuple[float, float]],
) -> None:
    numerator_joint, denominator_joint = RATIO_PAIRS[pair]
    figure, axes = plt.subplots(
        nrows=1,
        ncols=len(source.TENDONS),
        figsize=(15.5, 5.4),
        squeeze=False,
    )
    for column_index, tendon in enumerate(source.TENDONS):
        axis = axes[0, column_index]
        target = _target_ratio(pair, tendon)
        for finger in source.FINGER_IDS:
            values = curve_df.query(
                "level == 'finger_mean' and finger == @finger "
                "and tendon == @tendon and ratio_pair == @pair"
            )
            axis.plot(
                values["relative_excursion_mm"],
                values["ratio"],
                color=source.FINGER_COLORS[finger],
                linewidth=2.0,
                label=finger,
            )
            points = representative_df.query(
                "level == 'finger_mean' and finger == @finger "
                "and tendon == @tendon and ratio_pair == @pair"
            )
            axis.plot(
                points["relative_excursion_mm"],
                points["ratio"],
                color=source.FINGER_COLORS[finger],
                marker="o",
                linestyle="none",
                markersize=4,
            )
        all_values = curve_df.query(
            "level == 'all_finger_mean' and tendon == @tendon "
            "and ratio_pair == @pair"
        )
        axis.plot(
            all_values["relative_excursion_mm"],
            all_values["ratio"],
            color="#222222",
            linewidth=2.8,
            label="all-finger mean",
        )
        all_points = representative_df.query(
            "level == 'all_finger_mean' and tendon == @tendon "
            "and ratio_pair == @pair"
        )
        axis.plot(
            all_points["relative_excursion_mm"],
            all_points["ratio"],
            color="#222222",
            marker="o",
            linestyle="none",
            markersize=5,
        )
        axis.axhline(
            target,
            color="#777777",
            linestyle="--",
            linewidth=1.2,
            label=f"human target = {target:g}",
        )
        axis.grid(True, alpha=0.25)
        axis.set_xlim(*x_limits[tendon])
        axis.set_title(tendon)
        axis.set_xlabel(f"relative {tendon} excursion [mm]")
        if column_index == 0:
            axis.set_ylabel(f"|{numerator_joint}| / |{denominator_joint}|")

    handles, labels = axes[0, 0].get_legend_handles_labels()
    figure.legend(handles, labels, loc="upper center", ncol=4, bbox_to_anchor=(0.5, 0.965))
    figure.suptitle(
        f"Human-style joint coupling: |{numerator_joint}| / |{denominator_joint}|\n"
        "per-finger mean of three smooth trials and all-finger mean; "
        "260907C excluded",
        fontsize=14,
        y=1.02,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.88))
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    input_root = _resolve_path(args.input_root)
    original_260907a_root = _resolve_path(args.__dict__["260907a_root"])
    output_root = _resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    trials = _load_trials(input_root, original_260907a_root)
    curve_df, representative_df, x_limits = _calculate_ratios(trials)
    _plot_pair(
        curve_df,
        representative_df,
        "PIP_over_MCP",
        output_root / "pip_mcp_ratio_vs_excursion.png",
        x_limits,
    )
    _plot_pair(
        curve_df,
        representative_df,
        "DIP_over_MCP",
        output_root / "dip_mcp_ratio_vs_excursion.png",
        x_limits,
    )
    curve_df.to_csv(output_root / "human_joint_coupling_ratio_curves.csv", index=False)
    representative_df.to_csv(
        output_root / "human_joint_coupling_ratio_representative_points.csv",
        index=False,
    )
    print(f"Loaded {len(trials)} trials")
    print(f"Saved outputs under {output_root}")


if __name__ == "__main__":
    main()
