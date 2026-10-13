from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = (
    PROJECT_ROOT / "logs" / "analysis_comparison" / "260907C" / "directional_moment_arm_safe"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "logs" / "analysis_comparison" / "260907C" / "direction_average_report"
)
JOINTS = ["DIP", "PIP", "MCP"]
TENDONS = ["FDP", "FDS", "ED", "DI", "PI", "LUM"]
DIRECTIONS = ["flexion", "extension"]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create direction-averaged moment-arm curves and stability figures."
    )
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--label", default="finger")
    return parser.parse_args()


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


def _load_direction_average(input_root: Path) -> pd.DataFrame:
    path = input_root / "directional_moment_arm_curves.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Missing directional curve table: {path}")
    curves = pd.read_csv(path)
    required = {
        "joint",
        "tendon",
        "direction",
        "trial",
        "angle_deg",
        "excursion_fit_mm",
        "moment_arm_mm_per_rad",
    }
    missing = sorted(required.difference(curves.columns))
    if missing:
        raise ValueError(f"Directional curve table is missing columns: {missing}")

    index_columns = ["joint", "tendon", "trial", "angle_deg"]
    flexion = curves[curves["direction"] == "flexion"].set_index(index_columns)
    extension = curves[curves["direction"] == "extension"].set_index(index_columns)
    common_index = flexion.index.intersection(extension.index)
    if common_index.empty:
        raise ValueError("No common flexion/extension angle grid was found.")
    flexion = flexion.loc[common_index]
    extension = extension.loc[common_index]
    averaged = pd.DataFrame(index=common_index)
    averaged["excursion_average_mm"] = 0.5 * (
        flexion["excursion_fit_mm"] + extension["excursion_fit_mm"]
    )
    averaged["moment_arm_average_mm_per_rad"] = 0.5 * (
        flexion["moment_arm_mm_per_rad"]
        + extension["moment_arm_mm_per_rad"]
    )
    return averaged.reset_index()


def _write_stability_table(curves: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    rows = []
    for (joint, tendon), group in curves.groupby(["joint", "tendon"], sort=True):
        pivot = group.pivot(
            index="angle_deg",
            columns="trial",
            values="moment_arm_average_mm_per_rad",
        ).sort_index()
        trial_sd = pivot.std(axis=1, ddof=1)
        excursion_pivot = group.pivot(
            index="angle_deg",
            columns="trial",
            values="excursion_average_mm",
        ).sort_index()
        excursion_sd = excursion_pivot.std(axis=1, ddof=1)
        rows.append(
            {
                "joint": joint,
                "tendon": tendon,
                "common_angle_min_deg": float(pivot.index.min()),
                "common_angle_max_deg": float(pivot.index.max()),
                "mean_moment_arm_sd_mm_per_rad": float(trial_sd.mean()),
                "max_moment_arm_sd_mm_per_rad": float(trial_sd.max()),
                "mean_excursion_sd_mm": float(excursion_sd.mean()),
                "max_excursion_sd_mm": float(excursion_sd.max()),
                "num_trials": int(pivot.shape[1]),
                "status": "ok" if pivot.shape[1] >= 2 else "insufficient_trials",
            }
        )
    table = pd.DataFrame(rows).sort_values(["joint", "tendon"])
    table.to_csv(output_root / "direction_average_moment_arm_stability.csv", index=False)
    return table


def _plot_curves(curves: pd.DataFrame, output_path: Path, label: str) -> None:
    figure, axes = plt.subplots(
        nrows=len(JOINTS),
        ncols=len(TENDONS),
        figsize=(20, 9.5),
        squeeze=False,
    )
    for row_index, joint in enumerate(JOINTS):
        for column_index, tendon in enumerate(TENDONS):
            axis = axes[row_index, column_index]
            panel = curves[(curves["joint"] == joint) & (curves["tendon"] == tendon)]
            trial_curves = panel.pivot(
                index="angle_deg",
                columns="trial",
                values="moment_arm_average_mm_per_rad",
            ).sort_index()
            x = trial_curves.index.to_numpy(dtype=float)
            mean = trial_curves.mean(axis=1).to_numpy(dtype=float)
            sd = trial_curves.std(axis=1, ddof=1).to_numpy(dtype=float)
            axis.fill_between(x, mean - sd, mean + sd, color="#009E73", alpha=0.18)
            axis.plot(x, mean, color="#009E73", linewidth=2.0)
            axis.axhline(0.0, color="#777777", linewidth=0.6, alpha=0.7)
            axis.grid(True, alpha=0.22)
            if row_index == 0:
                axis.set_title(tendon)
            if column_index == 0:
                axis.set_ylabel(f"{joint}\nmm/rad")
            if row_index == len(JOINTS) - 1:
                axis.set_xlabel("angle [deg]")
    figure.suptitle(
        f"{label}: direction-averaged moment-arm curves (mean ± trial SD)",
        fontsize=14,
        y=1.01,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_stability_bars(
    stability: pd.DataFrame,
    output_path: Path,
    label: str,
) -> None:
    figure, axes = plt.subplots(
        nrows=len(JOINTS),
        ncols=len(TENDONS),
        figsize=(18, 8.5),
        sharex=True,
        squeeze=False,
    )
    for row_index, joint in enumerate(JOINTS):
        joint_rows = stability[stability["joint"] == joint]
        row_max = max(float(joint_rows["mean_moment_arm_sd_mm_per_rad"].max()), 0.05)
        for column_index, tendon in enumerate(TENDONS):
            axis = axes[row_index, column_index]
            values = joint_rows.set_index("tendon").loc[tendon]
            height = float(values["mean_moment_arm_sd_mm_per_rad"])
            axis.bar([0], [height], color="#009E73", width=0.72)
            axis.set_ylim(0.0, max(row_max * 1.28, 0.05))
            axis.set_xticks([0], ["avg"])
            axis.grid(axis="y", alpha=0.25)
            axis.text(
                0,
                height + row_max * 0.035,
                f"{height:.2f}",
                ha="center",
                va="bottom",
                fontsize=7,
            )
            if row_index == 0:
                axis.set_title(tendon)
            if column_index == 0:
                axis.set_ylabel(f"{joint}\nSD [mm/rad]")
    figure.suptitle(
        f"{label}: trial-to-trial variability of direction-averaged moment arms",
        fontsize=14,
    )
    figure.text(
        0.5,
        0.02,
        "avg = (flexion moment arm + extension moment arm) / 2",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.95))
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    input_root = _resolve_input_root(args.input_root)
    output_root = _resolve_output_root(args.output_root)
    curves = _load_direction_average(input_root)
    curves.to_csv(output_root / "direction_average_moment_arm_curves.csv", index=False)
    stability = _write_stability_table(curves, output_root)
    _plot_curves(
        curves,
        output_root / "direction_average_moment_arm_curves.png",
        args.label,
    )
    _plot_stability_bars(
        stability,
        output_root / "direction_average_moment_arm_stability_bars.png",
        args.label,
    )
    print(f"Saved direction-average report under {output_root}")
    print(
        stability[
            ["joint", "tendon", "mean_moment_arm_sd_mm_per_rad"]
        ].to_string(index=False, float_format=lambda value: f"{value:.4f}")
    )


if __name__ == "__main__":
    main()
