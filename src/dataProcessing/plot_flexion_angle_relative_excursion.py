from __future__ import annotations

import argparse
import re
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_INPUT_ROOT = (
    PROJECT_ROOT / "logs" / "analysis_comparison" / "excursion_joint_final"
)
DEFAULT_OUTPUT_ROOT = (
    DEFAULT_INPUT_ROOT / "figures" / "flexion_angle_relative_excursion"
)
FINGER_IDS = ("2607B", "2608A", "260907C")
TENDONS = ("FDP", "FDS", "ED")
JOINTS = ("DIP", "PIP", "MCP")
TRIAL_COLORS = {1: "#0072B2", 2: "#D55E00", 3: "#009E73"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot smoothed flexion angles against excursion relative to "
            "the first valid smoothed point."
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


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _trial_number(path: Path, tendon: str) -> int:
    match = re.search(rf"{tendon}[_ -]?(\d+)", str(path), flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Could not infer {tendon} trial number from {path}")
    return int(match.group(1))


def _load_trial(path: Path, finger: str, tendon: str) -> dict[str, object]:
    dataframe = pd.read_csv(path)
    required = {
        "active_motion",
        "excursion_smoothed_mm",
        *[f"{joint}_smoothed_deg" for joint in JOINTS],
    }
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    active = _as_bool(dataframe["active_motion"])
    if "tendon_quality_ok" in dataframe.columns:
        active &= _as_bool(dataframe["tendon_quality_ok"])
    excursion = pd.to_numeric(dataframe["excursion_smoothed_mm"], errors="coerce")

    # Use one excursion origin per tendon trial so all three joint curves
    # in a panel share the same relative excursion coordinate.
    valid_excursion = active & excursion.notna()
    if int(valid_excursion.sum()) < 2:
        raise ValueError(f"Not enough valid smoothed excursion points in {path}")
    first_index = int(np.flatnonzero(valid_excursion.to_numpy())[0])
    first_excursion = float(excursion.iloc[first_index])
    relative_excursion = (excursion - first_excursion).to_numpy(dtype=float)

    angles: dict[str, np.ndarray] = {}
    for joint in JOINTS:
        angle = pd.to_numeric(dataframe[f"{joint}_smoothed_deg"], errors="coerce")
        valid = (valid_excursion & angle.notna()).to_numpy(dtype=bool)
        if int(valid.sum()) < 2:
            raise ValueError(f"Not enough valid {joint} points in {path}")
        angles[joint] = angle.to_numpy(dtype=float)

    return {
        "finger": finger,
        "tendon": tendon,
        "trial": _trial_number(path, tendon),
        "path": path,
        "first_frame_index": first_index,
        "first_excursion_mm": first_excursion,
        "relative_excursion_mm": relative_excursion,
        "angles_deg": angles,
        "active_excursion_mask": valid_excursion.to_numpy(dtype=bool),
    }


def _discover_trials(input_root: Path) -> list[dict[str, object]]:
    trials: list[dict[str, object]] = []
    for finger in FINGER_IDS:
        for tendon in TENDONS:
            paths = sorted((input_root / finger).glob(f"{tendon}_*/cleaned.csv"))
            if not paths:
                raise FileNotFoundError(
                    f"No {tendon} cleaned.csv files found under {input_root / finger}"
                )
            tendon_trials = [
                _load_trial(path, finger=finger, tendon=tendon) for path in paths
            ]
            found = {int(trial["trial"]) for trial in tendon_trials}
            if found != {1, 2, 3}:
                raise ValueError(
                    f"Expected {tendon} trials 1, 2, 3 for {finger}; "
                    f"found {sorted(found)}"
                )
            trials.extend(sorted(tendon_trials, key=lambda trial: int(trial["trial"])))
    return trials


def _plot_finger(
    finger: str,
    trials: list[dict[str, object]],
    output_path: Path,
    x_limits: dict[str, tuple[float, float]],
    y_limits: dict[tuple[str, str], tuple[float, float]],
) -> pd.DataFrame:
    figure, axes = plt.subplots(
        nrows=len(TENDONS),
        ncols=len(JOINTS),
        figsize=(15.5, 11.0),
        squeeze=False,
    )
    summary_rows: list[dict[str, object]] = []

    for row_index, tendon in enumerate(TENDONS):
        tendon_trials = [trial for trial in trials if trial["tendon"] == tendon]
        for column_index, joint in enumerate(JOINTS):
            axis = axes[row_index, column_index]
            for trial in tendon_trials:
                x = trial["relative_excursion_mm"]
                angles = trial["angles_deg"]
                mask = trial["active_excursion_mask"]
                assert isinstance(x, np.ndarray)
                assert isinstance(angles, dict)
                assert isinstance(mask, np.ndarray)
                y = angles[joint]
                valid = mask & np.isfinite(x) & np.isfinite(y)
                axis.plot(
                    x[valid],
                    y[valid],
                    color=TRIAL_COLORS[int(trial["trial"])],
                    linewidth=1.25,
                    alpha=0.82,
                    label=f"trial {trial['trial']}",
                )
                summary_rows.append(
                    {
                        "finger": finger,
                        "tendon": tendon,
                        "joint": joint,
                        "trial": int(trial["trial"]),
                        "first_frame_index": int(trial["first_frame_index"]),
                        "first_excursion_mm": float(trial["first_excursion_mm"]),
                        "relative_excursion_min_mm": float(np.min(x[valid])),
                        "relative_excursion_max_mm": float(np.max(x[valid])),
                        "first_flexion_angle_deg": float(y[valid][0]),
                        "last_flexion_angle_deg": float(y[valid][-1]),
                        "flexion_angle_range_deg": float(np.ptp(y[valid])),
                        "valid_points": int(valid.sum()),
                        "source": str(trial["path"]),
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
            color=TRIAL_COLORS[trial],
            linewidth=1.8,
            label=f"trial {trial}",
        )
        for trial in (1, 2, 3)
    ]
    figure.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.965))
    figure.suptitle(
        f"{finger}: flexion angle vs relative excursion\n"
        "smooth curves only; excursion zeroed at the first valid smoothed point",
        fontsize=14,
        y=0.995,
    )
    figure.tight_layout(rect=(0.0, 0.0, 1.0, 0.92))
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return pd.DataFrame(summary_rows)


def main() -> None:
    args = parse_args()
    input_root = _resolve_path(args.input_root)
    output_root = _resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    trials = _discover_trials(input_root)
    x_limits: dict[str, tuple[float, float]] = {}
    for tendon in TENDONS:
        tendon_trials = [trial for trial in trials if trial["tendon"] == tendon]
        maximum = max(
            float(
                np.max(
                    trial["relative_excursion_mm"][trial["active_excursion_mask"]]
                )
            )
            for trial in tendon_trials
        )
        minimum = min(
            float(
                np.min(
                    trial["relative_excursion_mm"][trial["active_excursion_mask"]]
                )
            )
            for trial in tendon_trials
        )
        padding = max(0.25, 0.02 * (maximum - minimum))
        x_limits[tendon] = (minimum - padding, maximum + padding)

    y_limits: dict[tuple[str, str], tuple[float, float]] = {}
    for tendon in TENDONS:
        tendon_trials = [trial for trial in trials if trial["tendon"] == tendon]
        for joint in JOINTS:
            values = []
            for trial in tendon_trials:
                x = trial["relative_excursion_mm"]
                y = trial["angles_deg"][joint]
                mask = trial["active_excursion_mask"] & np.isfinite(x) & np.isfinite(y)
                values.append(y[mask])
            combined = np.concatenate(values)
            lower = float(np.min(combined))
            upper = float(np.max(combined))
            padding = max(1.0, 0.05 * (upper - lower))
            y_limits[(tendon, joint)] = (lower - padding, upper + padding)

    all_summary: list[pd.DataFrame] = []
    for finger in FINGER_IDS:
        finger_trials = [trial for trial in trials if trial["finger"] == finger]
        summary = _plot_finger(
            finger,
            finger_trials,
            output_root / f"{finger}_flexion_angle_relative_excursion.png",
            x_limits=x_limits,
            y_limits=y_limits,
        )
        all_summary.append(summary)
        print(f"{finger}: plotted {len(finger_trials)} trials")

    pd.concat(all_summary, ignore_index=True).to_csv(
        output_root / "flexion_angle_relative_excursion_summary.csv",
        index=False,
    )
    print(f"Saved outputs under {output_root}")


if __name__ == "__main__":
    main()
