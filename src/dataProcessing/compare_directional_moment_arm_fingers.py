from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
FINGER_REPORT_ROOTS = {
    "2608A": PROJECT_ROOT / "logs" / "analysis_comparison" / "2608A" / "directional_report_safe",
    "260907C": PROJECT_ROOT / "logs" / "analysis_comparison" / "260907C" / "directional_report_safe",
    "2607B": PROJECT_ROOT / "logs" / "analysis_comparison" / "2607B" / "directional_report_safe",
}
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "logs" / "analysis_comparison" / "three_finger_comparison"
JOINTS = ["DIP", "PIP", "MCP"]
TENDONS = ["FDP", "FDS", "ED", "DI", "PI", "LUM"]
DIRECTIONS = ["flexion", "extension"]
DIRECTION_COLORS = {"flexion": "#0072B2", "extension": "#D55E00"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Compare directional moment-arm stability across finger IDs."
    )
    parser.add_argument(
        "--output-root",
        default=str(DEFAULT_OUTPUT_ROOT),
        help="Directory for the combined table and figure.",
    )
    return parser.parse_args()


def _resolve_output_root(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _load_stability_tables() -> pd.DataFrame:
    tables = []
    for finger, report_root in FINGER_REPORT_ROOTS.items():
        path = report_root / "directional_stability_table.csv"
        if not path.is_file():
            raise FileNotFoundError(f"Missing stability table for {finger}: {path}")
        table = pd.read_csv(path)
        table.insert(0, "finger_id", finger)
        tables.append(table)
    return pd.concat(tables, ignore_index=True)


def _plot_stability_bars(summary: pd.DataFrame, output_path: Path) -> None:
    finger_ids = list(FINGER_REPORT_ROOTS)
    figure, axes = plt.subplots(
        nrows=len(finger_ids) * len(JOINTS),
        ncols=len(TENDONS),
        figsize=(19, 19),
        sharex=True,
        squeeze=False,
    )

    row_limits: dict[str, float] = {}
    for joint in JOINTS:
        joint_rows = summary[summary["joint"] == joint]
        maximum = float(
            joint_rows[
                [
                    "flexion_mean_moment_arm_sd_mm_per_rad",
                    "extension_mean_moment_arm_sd_mm_per_rad",
                ]
            ].to_numpy().max()
        )
        row_limits[joint] = max(maximum * 1.28, 0.05)

    for finger_index, finger_id in enumerate(finger_ids):
        for joint_index, joint in enumerate(JOINTS):
            row_index = finger_index * len(JOINTS) + joint_index
            joint_rows = summary[
                (summary["finger_id"] == finger_id) & (summary["joint"] == joint)
            ].set_index("tendon")
            for column_index, tendon in enumerate(TENDONS):
                axis = axes[row_index, column_index]
                values = joint_rows.loc[tendon]
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
                axis.set_ylim(0.0, row_limits[joint])
                axis.set_xticks([0, 1], ["F", "E"])
                axis.grid(axis="y", alpha=0.25)
                for x_position, height in enumerate(heights):
                    axis.text(
                        x_position,
                        height + row_limits[joint] * 0.025,
                        f"{height:.2f}",
                        ha="center",
                        va="bottom",
                        fontsize=6,
                    )
                if row_index == 0:
                    axis.set_title(tendon)
                if column_index == 0:
                    axis.set_ylabel(f"{finger_id} {joint}\nSD [mm/rad]")

    figure.suptitle(
        "Directional moment-arm variability across finger IDs",
        fontsize=15,
    )
    figure.text(
        0.5,
        0.015,
        "F = flexion, E = extension; bars are mean SD across each finger's common angle grid",
        ha="center",
        fontsize=10,
    )
    figure.tight_layout(rect=(0, 0.035, 1, 0.97), h_pad=1.1)
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    output_root = _resolve_output_root(args.output_root)
    summary = _load_stability_tables()
    summary.to_csv(output_root / "three_finger_directional_stability_table.csv", index=False)
    _plot_stability_bars(
        summary,
        output_root / "three_finger_directional_stability_bars.png",
    )
    print(f"Saved combined comparison under {output_root}")


if __name__ == "__main__":
    main()
