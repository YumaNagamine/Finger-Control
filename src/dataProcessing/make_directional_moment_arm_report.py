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
    PROJECT_ROOT / "logs" / "analysis_comparison" / "260907C" / "directional_moment_arm"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "logs" / "analysis_comparison" / "260907C" / "directional_report"

JOINTS = ["DIP", "PIP", "MCP"]
TENDONS = ["FDP", "FDS", "ED", "DI", "PI", "LUM"]
DIRECTIONS = ["flexion", "extension"]
DIRECTION_COLORS = {"flexion": "#0072B2", "extension": "#D55E00"}
DIRECTION_LINESTYLES = {"flexion": "-", "extension": "--"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create tables and figures from directional moment-arm analysis outputs."
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


def _load_tables(input_root: Path) -> tuple[pd.DataFrame, pd.DataFrame]:
    stability_path = input_root / "directional_moment_arm_stability.csv"
    curves_path = input_root / "directional_moment_arm_curves.csv"
    if not stability_path.is_file() or not curves_path.is_file():
        raise FileNotFoundError(
            "Expected directional_moment_arm_stability.csv and "
            f"directional_moment_arm_curves.csv under {input_root}"
        )
    stability = pd.read_csv(stability_path)
    curves = pd.read_csv(curves_path)
    return stability, curves


def _write_summary_table(stability: pd.DataFrame, output_root: Path) -> pd.DataFrame:
    columns = [
        "joint",
        "tendon",
        "common_angle_min_deg",
        "common_angle_max_deg",
        "expected_stable_direction",
        "observed_more_stable_direction",
        "hypothesis_supported_descriptively",
        "flexion_mean_moment_arm_sd_mm_per_rad",
        "extension_mean_moment_arm_sd_mm_per_rad",
        "expected_to_other_sd_ratio",
        "flexion_mean_excursion_sd_mm",
        "extension_mean_excursion_sd_mm",
        "flexion_mean_fit_r_squared",
        "extension_mean_fit_r_squared",
        "status",
    ]
    missing = [column for column in columns if column not in stability.columns]
    if missing:
        raise ValueError(f"Stability table is missing columns: {missing}")
    summary = stability[columns].copy()
    summary["moment_arm_sd_difference_extension_minus_flexion"] = (
        summary["extension_mean_moment_arm_sd_mm_per_rad"]
        - summary["flexion_mean_moment_arm_sd_mm_per_rad"]
    )
    summary.to_csv(output_root / "directional_stability_table.csv", index=False)
    return summary


def _plot_stability_bars(summary: pd.DataFrame, output_path: Path, label: str) -> None:
    figure, axes = plt.subplots(
        nrows=len(JOINTS),
        ncols=len(TENDONS),
        figsize=(18, 8.5),
        sharex=True,
        squeeze=False,
    )
    for row_index, joint in enumerate(JOINTS):
        joint_rows = summary[summary["joint"] == joint]
        row_max = float(
            joint_rows[
                [
                    "flexion_mean_moment_arm_sd_mm_per_rad",
                    "extension_mean_moment_arm_sd_mm_per_rad",
                ]
            ].to_numpy().max()
        )
        for column_index, tendon in enumerate(TENDONS):
            axis = axes[row_index, column_index]
            values = joint_rows.set_index("tendon").loc[tendon]
            heights = [
                values["flexion_mean_moment_arm_sd_mm_per_rad"],
                values["extension_mean_moment_arm_sd_mm_per_rad"],
            ]
            axis.bar(
                [0, 1],
                heights,
                color=[DIRECTION_COLORS[direction] for direction in DIRECTIONS],
                width=0.72,
            )
            axis.set_ylim(0.0, max(row_max * 1.28, 0.05))
            axis.set_xticks([0, 1], ["F", "E"])
            axis.grid(axis="y", alpha=0.25)
            for x_position, height in enumerate(heights):
                axis.text(
                    x_position,
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
        f"{label}: trial-to-trial moment-arm variability by direction",
        fontsize=14,
    )
    figure.text(
        0.5,
        0.02,
        "F = flexion, E = extension; bars are mean SD across the common angle grid",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.05, 1, 0.95))
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _plot_directional_curves(curves: pd.DataFrame, output_path: Path, label: str) -> None:
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
            for direction in DIRECTIONS:
                direction_panel = panel[panel["direction"] == direction]
                if direction_panel.empty:
                    continue
                trial_curves = direction_panel.pivot(
                    index="angle_deg", columns="trial", values="moment_arm_mm_per_rad"
                ).sort_index()
                x = trial_curves.index.to_numpy(dtype=float)
                mean = trial_curves.mean(axis=1).to_numpy(dtype=float)
                sd = trial_curves.std(axis=1, ddof=1).to_numpy(dtype=float)
                color = DIRECTION_COLORS[direction]
                linestyle = DIRECTION_LINESTYLES[direction]
                axis.fill_between(x, mean - sd, mean + sd, color=color, alpha=0.12)
                axis.plot(
                    x,
                    mean,
                    color=color,
                    linestyle=linestyle,
                    linewidth=1.7,
                    label=direction,
                )
            axis.axhline(0.0, color="#777777", linewidth=0.6, alpha=0.7)
            axis.grid(True, alpha=0.22)
            if row_index == 0:
                axis.set_title(tendon)
            if column_index == 0:
                axis.set_ylabel(f"{joint}\nmm/rad")
            if row_index == len(JOINTS) - 1:
                axis.set_xlabel("angle [deg]")
    handles = [
        plt.Line2D(
            [0],
            [0],
            color=DIRECTION_COLORS[direction],
            linestyle=DIRECTION_LINESTYLES[direction],
            linewidth=2,
            label=direction,
        )
        for direction in DIRECTIONS
    ]
    figure.legend(handles=handles, loc="upper center", ncol=2, bbox_to_anchor=(0.5, 0.98))
    figure.suptitle(
        f"{label}: directional moment-arm curves (mean ± trial SD)",
        fontsize=14,
        y=1.01,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.95))
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    input_root = _resolve_input_root(args.input_root)
    output_root = _resolve_output_root(args.output_root)
    stability, curves = _load_tables(input_root)
    summary = _write_summary_table(stability, output_root)
    _plot_stability_bars(summary, output_root / "directional_stability_bars.png", args.label)
    _plot_directional_curves(
        curves,
        output_root / "directional_moment_arm_curves.png",
        args.label,
    )
    print(f"Saved report under {output_root}")
    print(summary[[
        "joint",
        "tendon",
        "observed_more_stable_direction",
        "flexion_mean_moment_arm_sd_mm_per_rad",
        "extension_mean_moment_arm_sd_mm_per_rad",
    ]].to_string(index=False, float_format=lambda value: f"{value:.4f}"))


if __name__ == "__main__":
    main()
