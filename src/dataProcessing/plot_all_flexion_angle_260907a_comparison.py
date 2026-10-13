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
FINAL_ROOT = (
    PROJECT_ROOT / "logs" / "analysis_comparison" / "excursion_joint_final"
)
DEFAULT_260907A_ROOT = (
    PROJECT_ROOT / "logs" / "dual_camera" / "analysis" / "260907A"
)
DEFAULT_OUTPUT_ROOT = (
    FINAL_ROOT / "figures" / "260907a_replaced_all_tendons"
)
FINGER_IDS = ("2607B", "2608A", "260907A")
TENDONS = ("FDP", "FDS", "ED")
JOINTS = ("DIP", "PIP", "MCP")
FINGER_COLORS = {
    "2607B": "#0072B2",
    "2608A": "#D55E00",
    "260907A": "#CC79A7",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare per-finger trial-mean flexion-angle curves for "
            "2607B, 2608A, and 260907A across all tendons and joints."
        )
    )
    parser.add_argument("--input-root", default=str(FINAL_ROOT))
    parser.add_argument("--260907a-root", default=str(DEFAULT_260907A_ROOT))
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
    match = re.search(
        rf"(?:{tendon}[_ -]?|trial[_ -]?)(0*[1-9]\d*)",
        str(path),
        flags=re.IGNORECASE,
    )
    if match is None:
        raise ValueError(f"Could not infer {tendon} trial number from {path}")
    return int(match.group(1))


def _find_paths(
    finger: str,
    tendon: str,
    input_root: Path,
    original_260907a_root: Path,
) -> list[Path]:
    if finger == "260907A":
        paths = [
            path
            for path in (original_260907a_root / tendon).glob(
                "trial_*/cleaned/*.csv"
            )
            if "rom_summary" not in path.name.lower()
        ]
    else:
        paths = list((input_root / finger).glob(f"{tendon}_*/cleaned.csv"))
    paths = sorted(paths, key=lambda path: _trial_number(path, tendon))
    found = {_trial_number(path, tendon) for path in paths}
    if found != {1, 2, 3}:
        raise ValueError(
            f"Expected {tendon} trials 1, 2, 3 for {finger}; found {paths}"
        )
    return paths


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
    valid_excursion = active & excursion.notna()
    if int(valid_excursion.sum()) < 2:
        raise ValueError(f"Not enough valid excursion points in {path}")

    first_index = int(np.flatnonzero(valid_excursion.to_numpy())[0])
    first_excursion = float(excursion.iloc[first_index])
    relative_excursion = (excursion - first_excursion).to_numpy(dtype=float)
    angles = {
        joint: pd.to_numeric(dataframe[f"{joint}_smoothed_deg"], errors="coerce")
        .to_numpy(dtype=float)
        for joint in JOINTS
    }
    return {
        "finger": finger,
        "tendon": tendon,
        "trial": _trial_number(path, tendon),
        "path": path,
        "relative_excursion_mm": relative_excursion,
        "angles_deg": angles,
        "valid_excursion": valid_excursion.to_numpy(dtype=bool),
        "first_frame_index": first_index,
        "first_excursion_mm": first_excursion,
    }


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
    y = trial["angles_deg"][joint]
    mask = trial["valid_excursion"] & np.isfinite(x) & np.isfinite(y)
    return _sorted_unique_curve(x[mask], y[mask])


def _plot(
    trials: list[dict[str, object]],
    output_path: Path,
    curve_output_path: Path,
) -> None:
    figure, axes = plt.subplots(
        nrows=len(TENDONS),
        ncols=len(JOINTS),
        figsize=(15.5, 11.0),
        squeeze=False,
    )
    curve_rows: list[dict[str, object]] = []

    x_limits: dict[str, tuple[float, float]] = {}
    for tendon in TENDONS:
        maxima = []
        for trial in trials:
            if trial["tendon"] != tendon:
                continue
            x = trial["relative_excursion_mm"]
            mask = trial["valid_excursion"]
            maxima.append(float(np.max(x[mask])))
        common_max = min(maxima)
        if common_max <= 0.0:
            raise ValueError(f"Non-positive common {tendon} excursion range")
        x_limits[tendon] = (0.0, common_max)

    y_limits: dict[tuple[str, str], tuple[float, float]] = {}
    for tendon in TENDONS:
        for joint in JOINTS:
            values = [
                _trial_curve(trial, joint)[1]
                for trial in trials
                if trial["tendon"] == tendon
            ]
            combined = np.concatenate(values)
            lower = float(np.min(combined))
            upper = float(np.max(combined))
            padding = max(1.0, 0.05 * (upper - lower))
            y_limits[(tendon, joint)] = (lower - padding, upper + padding)

    for row_index, tendon in enumerate(TENDONS):
        tendon_trials = [trial for trial in trials if trial["tendon"] == tendon]
        x_grid = np.linspace(0.0, x_limits[tendon][1], 201)
        for column_index, joint in enumerate(JOINTS):
            axis = axes[row_index, column_index]
            for finger in FINGER_IDS:
                finger_trials = [
                    trial
                    for trial in tendon_trials
                    if trial["finger"] == finger
                ]
                interpolated = np.vstack(
                    [
                        np.interp(x_grid, *_trial_curve(trial, joint))
                        for trial in finger_trials
                    ]
                )
                mean_curve = np.mean(interpolated, axis=0)
                axis.plot(
                    x_grid,
                    mean_curve,
                    color=FINGER_COLORS[finger],
                    linewidth=2.0,
                    label=finger,
                )
                for excursion, angle in zip(x_grid, mean_curve):
                    curve_rows.append(
                        {
                            "finger": finger,
                            "tendon": tendon,
                            "joint": joint,
                            "relative_excursion_mm": float(excursion),
                            "mean_flexion_angle_deg": float(angle),
                            "trial_count": len(finger_trials),
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
        for finger in FINGER_IDS
    ]
    figure.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.965))
    figure.suptitle(
        "Per-finger mean flexion angle vs relative excursion\n"
        "mean of three smooth trials; 260907C excluded and 260907A included",
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
    original_260907a_root = _resolve_path(args.__dict__["260907a_root"])
    output_root = _resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    trials: list[dict[str, object]] = []
    for finger in FINGER_IDS:
        for tendon in TENDONS:
            paths = _find_paths(
                finger,
                tendon,
                input_root,
                original_260907a_root,
            )
            trials.extend(
                _load_trial(path, finger=finger, tendon=tendon) for path in paths
            )
    _plot(
        trials,
        output_root / "all_tendons_flexion_angle_260907A_replaced.png",
        output_root / "all_tendons_flexion_angle_260907A_replaced.csv",
    )
    print(f"Loaded {len(trials)} trials")
    print(f"Saved outputs under {output_root}")


if __name__ == "__main__":
    main()
