from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from plot_fdp_dip_flexion_angle import (
    FINGER_IDS,
    _discover_trials,
    _resolve_path,
    _sorted_unique_curve,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = (
    PROJECT_ROOT / "logs" / "analysis_comparison" / "excursion_joint_final"
)
DEFAULT_OUTPUT_ROOT = DEFAULT_INPUT_ROOT / "figures" / "fdp_dip_flexion_angle"
FINGER_COLORS = {
    "2607B": "#0072B2",
    "2608A": "#D55E00",
    "260907C": "#009E73",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot FDP-DIP three-trial mean curves registered at a common "
            "DIP flexion angle."
        )
    )
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--reference-finger", default="2607B", choices=FINGER_IDS)
    parser.add_argument("--reference-angle-deg", type=float, default=8.0)
    return parser.parse_args()


def _mean_curve(
    trials: list[dict[str, object]],
    grid_points: int = 201,
) -> tuple[np.ndarray, np.ndarray]:
    sorted_curves = []
    max_excursions = []
    for trial in trials:
        x = trial["relative_excursion_mm"]
        y = trial["flexion_angle_deg"]
        assert isinstance(x, np.ndarray)
        assert isinstance(y, np.ndarray)
        sorted_x, sorted_y = _sorted_unique_curve(x, y)
        sorted_curves.append((sorted_x, sorted_y))
        max_excursions.append(float(np.max(sorted_x)))

    common_max = min(max_excursions)
    if common_max <= 0.0:
        raise ValueError("Non-positive common excursion range")
    grid = np.linspace(0.0, common_max, grid_points)
    interpolated = np.vstack(
        [np.interp(grid, x, y) for x, y in sorted_curves]
    )
    return grid, np.mean(interpolated, axis=0)


def _first_crossing_x(
    x: np.ndarray,
    y: np.ndarray,
    target: float,
) -> float:
    difference = y - target
    exact = np.flatnonzero(np.isclose(difference, 0.0, atol=1e-10))
    if exact.size:
        return float(x[exact[0]])

    crossing = np.flatnonzero(difference[:-1] * difference[1:] < 0.0)
    if crossing.size:
        index = int(crossing[0])
        return float(
            x[index]
            + (target - y[index])
            * (x[index + 1] - x[index])
            / (y[index + 1] - y[index])
        )
    raise ValueError(f"Reference angle {target} deg is outside the mean curve")


def _plot_registered_means(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    reference_finger: str,
    reference_angle: float,
    output_path: Path,
) -> pd.DataFrame:
    reference_x, reference_y = curves[reference_finger]
    reference_anchor_x = _first_crossing_x(
        reference_x,
        reference_y,
        reference_angle,
    )

    figure, axis = plt.subplots(figsize=(8.8, 6.0))
    summary_rows: list[dict[str, object]] = []
    for finger in FINGER_IDS:
        x, y = curves[finger]
        crossing_x = _first_crossing_x(x, y, reference_angle)
        shift_x = reference_anchor_x - crossing_x
        registered_x = x + shift_x
        axis.plot(
            registered_x,
            y,
            color=FINGER_COLORS[finger],
            linewidth=2.4,
            label=finger,
        )
        axis.scatter(
            [reference_anchor_x],
            [reference_angle],
            color=FINGER_COLORS[finger],
            s=32,
            zorder=3,
        )
        summary_rows.append(
            {
                "finger": finger,
                "reference_finger": reference_finger,
                "reference_angle_deg": reference_angle,
                "reference_x_after_registration_mm": reference_anchor_x,
                "reference_x_before_registration_mm": crossing_x,
                "x_shift_mm": shift_x,
                "curve_max_relative_excursion_mm": float(np.max(x)),
            }
        )

    axis.axhline(reference_angle, color="#777777", linewidth=0.9, alpha=0.7)
    axis.axvline(reference_anchor_x, color="#777777", linewidth=0.9, alpha=0.7)
    axis.set_xlabel("registered FDP excursion [mm]")
    axis.set_ylabel("DIP flexion angle [deg]")
    axis.set_title(
        "FDP–DIP three-trial mean curves\n"
        f"registered at DIP flexion angle = {reference_angle:g}° "
        f"({reference_finger} anchor)",
    )
    axis.grid(True, alpha=0.25)
    axis.legend(title="finger", frameon=True)
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return pd.DataFrame(summary_rows)


def main() -> None:
    args = parse_args()
    input_root = _resolve_path(args.input_root)
    output_root = _resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    trials_by_finger = _discover_trials(input_root)
    curves = {
        finger: _mean_curve(trials_by_finger[finger])
        for finger in FINGER_IDS
    }
    summary = _plot_registered_means(
        curves,
        reference_finger=args.reference_finger,
        reference_angle=args.reference_angle_deg,
        output_path=(
            output_root
            / "FDP_DIP_three_trial_mean_registered_at_8deg.png"
        ),
    )
    summary.to_csv(
        output_root / "FDP_DIP_three_trial_mean_registration_summary.csv",
        index=False,
    )
    print(summary.to_string(index=False))
    print(f"Saved outputs under {output_root}")


if __name__ == "__main__":
    main()
