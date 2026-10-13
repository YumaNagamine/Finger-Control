from __future__ import annotations

import argparse
import re
from dataclasses import dataclass
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "logs" / "analysis_comparison" / "excursion_joint_comparison"
)
FINGER_ROOTS = {
    "2607B": PROJECT_ROOT / "logs" / "dual_camera-2607B" / "analysis" / "2607B_excursion_joint",
    "2608A": PROJECT_ROOT / "logs" / "dual_camera-2608A" / "analysis",
    "260907C": PROJECT_ROOT / "logs" / "dual_camera-260907C" / "analysis" / "excursion-joint",
}

TENDONS = ("FDP", "FDS", "ED")
JOINTS = ("DIP", "PIP", "MCP")
TENDON_COLORS = {
    "2607B": "#0072B2",
    "2608A": "#D55E00",
    "260907C": "#009E73",
}
EXCURSION_COLUMN = "excursion_smoothed_mm"
DISPLACEMENT_COLUMNS = {
    joint: f"{joint}_displacement_deg" for joint in JOINTS
}


@dataclass
class TrialData:
    finger: str
    tendon: str
    trial: int
    source: str
    path: Path
    excursion_mm: np.ndarray
    displacement_deg: dict[str, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Plot excursion-joint relationships for single-tendon FDP, FDS, "
            "and ED experiments across three robot finger individuals."
        )
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--min-valid-points",
        type=int,
        default=20,
        help="Minimum active valid points required for a trial to be included.",
    )
    parser.add_argument(
        "--percentile",
        type=float,
        default=1.0,
        help="Per-trial percentile used to define the robust common excursion range.",
    )
    return parser.parse_args()


def _resolve_output_root(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    path.mkdir(parents=True, exist_ok=True)
    return path.resolve()


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _infer_tendon(path: Path, dataframe: pd.DataFrame) -> str | None:
    if "analysis_tendon" in dataframe.columns:
        values = dataframe["analysis_tendon"].dropna().astype(str).str.upper()
        if not values.empty and values.iloc[0] in TENDONS:
            return values.iloc[0]
    match = re.search(r"(?:^|[\\/_-])(FDP|FDS|ED)(?:[_-]|$)", str(path).upper())
    return match.group(1) if match else None


def _infer_trial(path: Path, tendon: str) -> int | None:
    match = re.search(rf"{tendon}[_ -]?(\d+)", str(path), flags=re.IGNORECASE)
    return int(match.group(1)) if match else None


def _discover_cleaned_files(finger: str, root: Path) -> list[tuple[Path, str, int]]:
    if not root.is_dir():
        raise FileNotFoundError(f"Input directory not found for {finger}: {root}")
    discovered: list[tuple[Path, str, int]] = []
    for path in sorted(root.rglob("*cleaned.csv")):
        try:
            header = pd.read_csv(path, nrows=0)
        except Exception:
            continue
        tendon = _infer_tendon(path, header)
        if tendon is None:
            continue
        trial = _infer_trial(path, tendon)
        if trial is None:
            continue
        discovered.append((path, tendon, trial))
    return discovered


def _load_trial(
    finger: str,
    path: Path,
    tendon: str,
    trial: int,
    min_valid_points: int,
) -> TrialData | None:
    required = {
        EXCURSION_COLUMN,
        "active_motion",
        *DISPLACEMENT_COLUMNS.values(),
    }
    dataframe = pd.read_csv(path)
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    active = _as_bool(dataframe["active_motion"])
    if "tendon_quality_ok" in dataframe.columns:
        active &= _as_bool(dataframe["tendon_quality_ok"])
    excursion = pd.to_numeric(dataframe[EXCURSION_COLUMN], errors="coerce")
    displacement = {
        joint: pd.to_numeric(dataframe[column], errors="coerce")
        for joint, column in DISPLACEMENT_COLUMNS.items()
    }
    valid_excursion = active & excursion.notna()
    if int(valid_excursion.sum()) < min_valid_points:
        return None
    valid_joint_counts = {
        joint: int((valid_excursion & values.notna()).sum())
        for joint, values in displacement.items()
    }
    if max(valid_joint_counts.values()) < min_valid_points:
        return None

    return TrialData(
        finger=finger,
        tendon=tendon,
        trial=trial,
        source=path.stem,
        path=path,
        excursion_mm=excursion[valid_excursion].to_numpy(dtype=float),
        displacement_deg={
            joint: values[valid_excursion].to_numpy(dtype=float)
            for joint, values in displacement.items()
        },
    )


def _sorted_curve(x: np.ndarray, y: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    order = np.argsort(x)
    sorted_x = x[order]
    sorted_y = y[order]
    unique_x, inverse = np.unique(sorted_x, return_inverse=True)
    sums = np.bincount(inverse, weights=sorted_y)
    counts = np.bincount(inverse)
    return unique_x, sums / counts


def _common_grid(
    trials: list[TrialData],
    percentile: float,
    grid_points: int = 101,
) -> tuple[np.ndarray, float, float]:
    ranges = [
        (
            float(np.percentile(trial.excursion_mm, percentile)),
            float(np.percentile(trial.excursion_mm, 100.0 - percentile)),
        )
        for trial in trials
    ]
    lower = max(item[0] for item in ranges)
    upper = min(item[1] for item in ranges)
    if not np.isfinite(lower) or not np.isfinite(upper) or upper - lower < 0.5:
        raise ValueError(
            f"Common excursion range is too narrow: {lower:.3f} to {upper:.3f} mm"
        )
    return np.linspace(lower, upper, grid_points), lower, upper


def _interpolate_trial(trial: TrialData, joint: str, grid: np.ndarray) -> np.ndarray:
    x, y = _sorted_curve(trial.excursion_mm, trial.displacement_deg[joint])
    if x.size < 2:
        return np.full(grid.shape, np.nan)
    values = np.full(grid.shape, np.nan)
    inside = (grid >= x[0]) & (grid <= x[-1])
    values[inside] = np.interp(grid[inside], x, y)
    return values


def _plot_comparison(
    trials_by_tendon: dict[str, list[TrialData]],
    output_path: Path,
    percentile: float,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    figure, axes = plt.subplots(
        nrows=len(JOINTS),
        ncols=len(TENDONS),
        figsize=(15.5, 10.0),
        squeeze=False,
        sharex=False,
    )
    curve_rows: list[dict[str, object]] = []
    range_rows: list[dict[str, object]] = []

    for column_index, tendon in enumerate(TENDONS):
        tendon_trials = trials_by_tendon[tendon]
        grid, common_min, common_max = _common_grid(
            tendon_trials,
            percentile=percentile,
        )
        range_rows.append(
            {
                "tendon": tendon,
                "common_excursion_min_mm": common_min,
                "common_excursion_max_mm": common_max,
                "n_trials": len(tendon_trials),
            }
        )

        for row_index, joint in enumerate(JOINTS):
            axis = axes[row_index, column_index]
            for finger in FINGER_ROOTS:
                finger_trials = [
                    trial for trial in tendon_trials if trial.finger == finger
                ]
                if not finger_trials:
                    continue
                color = TENDON_COLORS[finger]
                trial_curves = np.vstack(
                    [_interpolate_trial(trial, joint, grid) for trial in finger_trials]
                )
                mean_curve = np.nanmean(trial_curves, axis=0)
                sd_curve = (
                    np.nanstd(trial_curves, axis=0, ddof=1)
                    if len(finger_trials) > 1
                    else np.full(grid.shape, np.nan)
                )
                for trial, values in zip(finger_trials, trial_curves):
                    axis.plot(
                        grid,
                        values,
                        color=color,
                        linewidth=0.8,
                        alpha=0.35,
                    )
                    for excursion, displacement in zip(grid, values):
                        curve_rows.append(
                            {
                                "finger": finger,
                                "tendon": tendon,
                                "joint": joint,
                                "trial": trial.trial,
                                "excursion_mm": excursion,
                                "joint_displacement_deg": displacement,
                                "curve_type": "trial_interpolated",
                            }
                        )
                axis.plot(
                    grid,
                    mean_curve,
                    color=color,
                    linewidth=2.2,
                    label=finger,
                )
                axis.fill_between(
                    grid,
                    mean_curve - sd_curve,
                    mean_curve + sd_curve,
                    color=color,
                    alpha=0.12,
                )
                for excursion, displacement, sd in zip(grid, mean_curve, sd_curve):
                    curve_rows.append(
                        {
                            "finger": finger,
                            "tendon": tendon,
                            "joint": joint,
                            "trial": "mean",
                            "excursion_mm": excursion,
                            "joint_displacement_deg": displacement,
                            "trial_sd_deg": sd,
                            "curve_type": "finger_mean",
                        }
                    )

            axis.axhline(0.0, color="#777777", linewidth=0.7, alpha=0.7)
            axis.grid(True, alpha=0.25)
            if row_index == 0:
                axis.set_title(tendon)
            if column_index == 0:
                axis.set_ylabel(f"{joint}\nΔ joint [deg]")
            if row_index == len(JOINTS) - 1:
                axis.set_xlabel(f"{tendon} excursion [mm]")

    handles = [
        plt.Line2D(
            [0],
            [0],
            color=TENDON_COLORS[finger],
            linewidth=2.2,
            label=finger,
        )
        for finger in FINGER_ROOTS
    ]
    figure.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.945))
    figure.suptitle(
        "Excursion–joint relationships: single-tendon pull\n"
        "mean ± trial SD; shaded bands are within-finger trial variability",
        fontsize=14,
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.88))
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return pd.DataFrame(curve_rows), pd.DataFrame(range_rows)


def _plot_trial_paths(
    trials_by_tendon: dict[str, list[TrialData]],
    output_path: Path,
) -> None:
    figure, axes = plt.subplots(
        nrows=len(JOINTS),
        ncols=len(TENDONS),
        figsize=(15.5, 10.0),
        squeeze=False,
    )
    for column_index, tendon in enumerate(TENDONS):
        for row_index, joint in enumerate(JOINTS):
            axis = axes[row_index, column_index]
            for finger in FINGER_ROOTS:
                finger_trials = [
                    trial for trial in trials_by_tendon[tendon] if trial.finger == finger
                ]
                for trial in finger_trials:
                    axis.plot(
                        trial.excursion_mm[np.isfinite(trial.displacement_deg[joint])],
                        trial.displacement_deg[joint][np.isfinite(trial.displacement_deg[joint])],
                        color=TENDON_COLORS[finger],
                        linewidth=0.9,
                        alpha=0.6,
                        label=finger if trial.trial == finger_trials[0].trial else "_nolegend_",
                    )
            axis.axhline(0.0, color="#777777", linewidth=0.7, alpha=0.7)
            axis.grid(True, alpha=0.25)
            if row_index == 0:
                axis.set_title(tendon)
            if column_index == 0:
                axis.set_ylabel(f"{joint}\nΔ joint [deg]")
            if row_index == len(JOINTS) - 1:
                axis.set_xlabel(f"{tendon} excursion [mm]")
    handles = [
        plt.Line2D(
            [0],
            [0],
            color=TENDON_COLORS[finger],
            linewidth=1.5,
            label=finger,
        )
        for finger in FINGER_ROOTS
    ]
    figure.legend(handles=handles, loc="upper center", ncol=3, bbox_to_anchor=(0.5, 0.945))
    figure.suptitle(
        "Excursion–joint relationships: individual trial paths",
        fontsize=14,
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.90))
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _summarize_at_standard_points(
    trials_by_tendon: dict[str, list[TrialData]],
    percentile: float,
) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    standard_positions = (0.0, 0.25, 0.5, 0.75, 1.0)
    for tendon in TENDONS:
        tendon_trials = trials_by_tendon[tendon]
        grid, common_min, common_max = _common_grid(
            tendon_trials,
            percentile=percentile,
        )
        standard_grid = np.interp(
            standard_positions,
            np.linspace(0.0, 1.0, grid.size),
            grid,
        )
        for joint in JOINTS:
            for finger in FINGER_ROOTS:
                finger_trials = [
                    trial for trial in tendon_trials if trial.finger == finger
                ]
                if not finger_trials:
                    continue
                values = np.vstack(
                    [
                        _interpolate_trial(trial, joint, standard_grid)
                        for trial in finger_trials
                    ]
                )
                row: dict[str, object] = {
                    "finger": finger,
                    "tendon": tendon,
                    "joint": joint,
                    "common_excursion_min_mm": common_min,
                    "common_excursion_max_mm": common_max,
                    "trial_count": len(finger_trials),
                    "trial_rom_mean_deg": np.nan,
                    "trial_rom_sd_deg": np.nan,
                }
                trial_rom_values = []
                for trial in finger_trials:
                    finite = trial.displacement_deg[joint][
                        np.isfinite(trial.displacement_deg[joint])
                    ]
                    if finite.size >= 2:
                        trial_rom_values.append(np.ptp(finite))
                if trial_rom_values:
                    row["trial_rom_mean_deg"] = float(np.mean(trial_rom_values))
                if len(trial_rom_values) > 1:
                    row["trial_rom_sd_deg"] = float(np.std(trial_rom_values, ddof=1))
                mean_values = np.nanmean(values, axis=0)
                sd_values = (
                    np.nanstd(values, axis=0, ddof=1)
                    if len(finger_trials) > 1
                    else np.full(len(standard_positions), np.nan)
                )
                for position, mean_value, sd_value in zip(
                    standard_positions, mean_values, sd_values
                ):
                    suffix = f"{int(position * 100):02d}pct"
                    row[f"excursion_{suffix}_mm"] = float(
                        common_min + position * (common_max - common_min)
                    )
                    row[f"joint_mean_{suffix}_deg"] = float(mean_value)
                    row[f"joint_trial_sd_{suffix}_deg"] = float(sd_value)
                rows.append(row)
    return pd.DataFrame(rows)


def main() -> None:
    args = parse_args()
    if args.min_valid_points < 2:
        raise ValueError("--min-valid-points must be at least 2.")
    if not 0.0 <= args.percentile < 50.0:
        raise ValueError("--percentile must be in [0, 50).")

    output_root = _resolve_output_root(args.output_root)
    all_trials: list[TrialData] = []
    discovery_rows: list[dict[str, object]] = []
    for finger, root in FINGER_ROOTS.items():
        files = _discover_cleaned_files(finger, root)
        for path, tendon, trial in files:
            loaded = _load_trial(
                finger,
                path,
                tendon,
                trial,
                min_valid_points=args.min_valid_points,
            )
            discovery_rows.append(
                {
                    "finger": finger,
                    "tendon": tendon,
                    "trial": trial,
                    "path": str(path),
                    "included": loaded is not None,
                    "valid_points": int(loaded.excursion_mm.size) if loaded else 0,
                }
            )
            if loaded is not None:
                all_trials.append(loaded)

    trials_by_tendon = {
        tendon: [trial for trial in all_trials if trial.tendon == tendon]
        for tendon in TENDONS
    }
    missing = {
        tendon: [
            finger
            for finger in FINGER_ROOTS
            if not any(
                trial.finger == finger for trial in trials_by_tendon[tendon]
            )
        ]
        for tendon in TENDONS
    }
    missing = {tendon: fingers for tendon, fingers in missing.items() if fingers}
    if missing:
        raise ValueError(f"Missing included trials by tendon: {missing}")

    curve_df, range_df = _plot_comparison(
        trials_by_tendon,
        output_root / "excursion_joint_finger_mean.png",
        percentile=args.percentile,
    )
    _plot_trial_paths(
        trials_by_tendon,
        output_root / "excursion_joint_individual_trials.png",
    )
    curve_df.to_csv(output_root / "excursion_joint_curves.csv", index=False)
    range_df.to_csv(output_root / "excursion_joint_common_ranges.csv", index=False)
    _summarize_at_standard_points(
        trials_by_tendon,
        percentile=args.percentile,
    ).to_csv(output_root / "excursion_joint_standard_summary.csv", index=False)
    pd.DataFrame(discovery_rows).sort_values(
        ["finger", "tendon", "trial"]
    ).to_csv(output_root / "excursion_joint_trial_manifest.csv", index=False)

    print("Included trials:")
    print(
        pd.DataFrame(discovery_rows)
        .query("included")
        .groupby(["finger", "tendon"])
        .size()
        .to_string()
    )
    print("\nCommon excursion ranges:")
    print(range_df.to_string(index=False, float_format=lambda value: f"{value:.3f}"))
    print(f"\nSaved outputs under {output_root}")


if __name__ == "__main__":
    main()
