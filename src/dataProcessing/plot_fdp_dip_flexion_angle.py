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
DEFAULT_OUTPUT_ROOT = DEFAULT_INPUT_ROOT / "figures" / "fdp_dip_flexion_angle"
FINGER_IDS = ("2607B", "2608A", "260907C")
TRIAL_COLORS = {1: "#0072B2", 2: "#D55E00", 3: "#009E73"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot FDP-DIP flexion angle against excursion relative to the "
            "first valid smoothed point."
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


def _trial_number(path: Path) -> int:
    match = re.search(r"FDP[_ -]?(\d+)", str(path), flags=re.IGNORECASE)
    if match is None:
        raise ValueError(f"Could not infer FDP trial number from {path}")
    return int(match.group(1))


def _load_trial(path: Path, finger: str) -> dict[str, object]:
    dataframe = pd.read_csv(path)
    required = {
        "active_motion",
        "excursion_smoothed_mm",
        "DIP_smoothed_deg",
    }
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    active = _as_bool(dataframe["active_motion"])
    if "tendon_quality_ok" in dataframe.columns:
        active &= _as_bool(dataframe["tendon_quality_ok"])
    excursion = pd.to_numeric(dataframe["excursion_smoothed_mm"], errors="coerce")
    flexion_angle = pd.to_numeric(dataframe["DIP_smoothed_deg"], errors="coerce")
    valid = active & excursion.notna() & flexion_angle.notna()
    if int(valid.sum()) < 2:
        raise ValueError(f"Not enough valid FDP-DIP points in {path}")

    first_index = int(np.flatnonzero(valid.to_numpy())[0])
    first_excursion = float(excursion.iloc[first_index])
    relative_excursion = (excursion - first_excursion).to_numpy(dtype=float)
    angle = flexion_angle.to_numpy(dtype=float)
    valid_values = valid.to_numpy(dtype=bool)
    relative_excursion = relative_excursion[valid_values]
    angle = angle[valid_values]

    return {
        "finger": finger,
        "trial": _trial_number(path),
        "path": path,
        "first_frame_index": first_index,
        "first_excursion_mm": first_excursion,
        "relative_excursion_mm": relative_excursion,
        "flexion_angle_deg": angle,
    }


def _discover_trials(input_root: Path) -> dict[str, list[dict[str, object]]]:
    trials_by_finger: dict[str, list[dict[str, object]]] = {}
    for finger in FINGER_IDS:
        finger_root = input_root / finger
        paths = sorted(finger_root.glob("FDP_*/cleaned.csv"))
        if not paths:
            raise FileNotFoundError(f"No FDP cleaned.csv files found under {finger_root}")
        trials = [_load_trial(path, finger) for path in paths]
        found = {int(trial["trial"]) for trial in trials}
        expected = {1, 2, 3}
        if found != expected:
            raise ValueError(f"Expected FDP trials {sorted(expected)} for {finger}, found {sorted(found)}")
        trials_by_finger[finger] = sorted(trials, key=lambda trial: int(trial["trial"]))
    return trials_by_finger


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


def _plot(
    trials_by_finger: dict[str, list[dict[str, object]]],
    output_path: Path,
) -> pd.DataFrame:
    figure, axes = plt.subplots(
        nrows=1,
        ncols=len(FINGER_IDS),
        figsize=(15.5, 4.9),
        squeeze=False,
        sharey=True,
    )
    summary_rows: list[dict[str, object]] = []

    for column_index, finger in enumerate(FINGER_IDS):
        axis = axes[0, column_index]
        trials = trials_by_finger[finger]
        for trial in trials:
            x = trial["relative_excursion_mm"]
            y = trial["flexion_angle_deg"]
            assert isinstance(x, np.ndarray)
            assert isinstance(y, np.ndarray)
            axis.plot(
                x,
                y,
                color=TRIAL_COLORS[int(trial["trial"])],
                linewidth=1.5,
                alpha=0.8,
                label=f"trial {trial['trial']}",
            )
            summary_rows.append(
                {
                    "finger": finger,
                    "trial": int(trial["trial"]),
                    "first_frame_index": int(trial["first_frame_index"]),
                    "first_excursion_mm": float(trial["first_excursion_mm"]),
                    "relative_excursion_min_mm": float(np.min(x)),
                    "relative_excursion_max_mm": float(np.max(x)),
                    "first_flexion_angle_deg": float(y[0]),
                    "last_flexion_angle_deg": float(y[-1]),
                    "flexion_angle_range_deg": float(np.ptp(y)),
                    "valid_points": int(x.size),
                    "source": str(trial["path"]),
                }
            )

        axis.axvline(0.0, color="#777777", linewidth=0.8, alpha=0.7)
        axis.grid(True, alpha=0.25)
        axis.set_title(finger)
        axis.set_xlabel("relative FDP excursion [mm]")
        axis.legend(frameon=True, fontsize=9)

    axes[0, 0].set_ylabel("DIP flexion angle [deg]")
    figure.suptitle(
        "FDP–DIP relationship: flexion angle vs excursion\n"
        "smooth curves only; excursion zeroed at the first valid smoothed point",
        fontsize=14,
        y=1.02,
    )
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
    summary = _plot(
        trials_by_finger,
        output_root / "FDP_DIP_flexion_angle_relative_excursion_smooth_only.png",
    )
    summary.to_csv(
        output_root / "FDP_DIP_flexion_angle_smooth_only_summary.csv",
        index=False,
    )
    print(f"Saved outputs under {output_root}")


if __name__ == "__main__":
    main()
