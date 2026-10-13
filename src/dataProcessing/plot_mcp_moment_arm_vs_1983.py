from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_REFERENCE = (
    PROJECT_ROOT
    / "src"
    / "dataProcessing"
    / "HumanMomentArm"
    / "moment arm data MCP from 1983.csv"
)
DEFAULT_COEFFICIENTS = (
    PROJECT_ROOT
    / "logs"
    / "dual_camera"
    / "excursion_fit_poly4_restricted"
    / "dual_processed_MCP_dual_record_legacy_20260912_193519_20260912_194442"
    / "polynomial_coefficients.csv"
)
DEFAULT_OUTPUT = PROJECT_ROOT / "logs" / "dual_camera" / "mcp_moment_arm_vs_1983"

TENDONS = ("FDP", "FDS", "ED", "DI", "PI", "LUM")
REFERENCE_COLUMN_PAIRS = {
    "FDP": (0, 1),
    "FDS": (2, 3),
    "ED": (4, 5),
    "LUM": (6, 7),
    "DI": (8, 9),
    "PI": (10, 11),
}
ANGLE_RANGE_DEG = (-10.0, 90.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument("--reference", type=Path, default=DEFAULT_REFERENCE)
    parser.add_argument("--coefficients", type=Path, default=DEFAULT_COEFFICIENTS)
    parser.add_argument("--output", type=Path, default=DEFAULT_OUTPUT)
    return parser.parse_args()


def load_reference(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    # The file has a descriptive first row and repeated X/Y columns in row two.
    frame = pd.read_csv(path, skiprows=1)
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for tendon, (x_index, y_index) in REFERENCE_COLUMN_PAIRS.items():
        angle = pd.to_numeric(frame.iloc[:, x_index], errors="coerce").to_numpy(float)
        moment_arm = pd.to_numeric(frame.iloc[:, y_index], errors="coerce").to_numpy(float)
        valid = np.isfinite(angle) & np.isfinite(moment_arm)
        in_range = valid & (angle >= ANGLE_RANGE_DEG[0]) & (angle <= ANGLE_RANGE_DEG[1])
        order = np.argsort(angle[in_range])
        # Sign convention in the 1983 data is opposite to the current experiment.
        result[tendon] = (angle[in_range][order], -moment_arm[in_range][order])
    return result


def current_curves(path: Path) -> dict[str, tuple[np.ndarray, np.ndarray]]:
    coefficients = pd.read_csv(path)
    angle_deg = np.linspace(*ANGLE_RANGE_DEG, 501)
    angle_rad = np.deg2rad(angle_deg)
    result: dict[str, tuple[np.ndarray, np.ndarray]] = {}
    for tendon in TENDONS:
        row = coefficients.loc[coefficients["tendon"] == tendon]
        if len(row) != 1:
            raise ValueError(f"Expected one coefficient row for {tendon}, found {len(row)}")
        c3, c2, c1, c0 = (
            float(row.iloc[0][name])
            for name in ("moment_arm_c3", "moment_arm_c2", "moment_arm_c1", "moment_arm_c0")
        )
        result[tendon] = (angle_deg, np.polyval([c3, c2, c1, c0], angle_rad))
    return result


def style_axis(axis: plt.Axes, tendon: str) -> None:
    axis.axhline(0.0, color="#777777", linewidth=0.8)
    axis.set_xlim(ANGLE_RANGE_DEG)
    axis.set_title(tendon)
    axis.set_xlabel("MCP flexion angle [deg]")
    axis.set_ylabel("Moment arm [mm/rad]")
    axis.grid(True, alpha=0.3)
    axis.legend()


def main() -> None:
    args = parse_args()
    reference_path = args.reference.expanduser().resolve()
    coefficient_path = args.coefficients.expanduser().resolve()
    output_dir = args.output.expanduser().resolve() / coefficient_path.parent.name
    output_dir.mkdir(parents=True, exist_ok=True)

    reference = load_reference(reference_path)
    current = current_curves(coefficient_path)
    comparison_rows: list[pd.DataFrame] = []

    fig, axes = plt.subplots(3, 2, figsize=(14, 13), sharex=True)
    for axis, tendon in zip(axes.flat, TENDONS):
        current_x, current_y = current[tendon]
        reference_x, reference_y = reference[tendon]
        axis.plot(current_x, current_y, linewidth=2.4, label="Current experiment")
        axis.plot(
            reference_x,
            reference_y,
            linestyle="--",
            linewidth=2.2,
            label="1983 (sign reversed)",
        )
        style_axis(axis, tendon)

        individual, individual_axis = plt.subplots(figsize=(9, 6))
        individual_axis.plot(current_x, current_y, linewidth=2.5, label="Current experiment")
        individual_axis.plot(
            reference_x,
            reference_y,
            linestyle="--",
            linewidth=2.3,
            label="1983 (sign reversed)",
        )
        style_axis(individual_axis, tendon)
        individual.tight_layout()
        individual.savefig(output_dir / f"MCP_{tendon}_vs_1983.png", dpi=220)
        plt.close(individual)

        comparison_rows.append(
            pd.DataFrame(
                {
                    "tendon": tendon,
                    "dataset": "current_experiment",
                    "angle_deg": current_x,
                    "moment_arm_mm_per_rad": current_y,
                }
            )
        )
        comparison_rows.append(
            pd.DataFrame(
                {
                    "tendon": tendon,
                    "dataset": "1983_sign_reversed",
                    "angle_deg": reference_x,
                    "moment_arm_mm_per_rad": reference_y,
                }
            )
        )

    fig.suptitle("MCP moment arm: current experiment vs 1983", fontsize=16)
    fig.tight_layout(rect=(0.0, 0.0, 1.0, 0.97))
    figure_path = output_dir / "MCP_all_tendons_vs_1983.png"
    fig.savefig(figure_path, dpi=220, bbox_inches="tight")
    plt.close(fig)

    comparison_path = output_dir / "MCP_moment_arm_comparison.csv"
    pd.concat(comparison_rows, ignore_index=True).to_csv(comparison_path, index=False)
    print(f"Saved figure: {figure_path}")
    print(f"Saved comparison data: {comparison_path}")


if __name__ == "__main__":
    main()
