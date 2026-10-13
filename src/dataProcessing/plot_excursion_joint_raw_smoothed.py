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
    PROJECT_ROOT / "logs" / "analysis_comparison" / "excursion_joint_raw_smoothed"
)
FINGER_ROOTS = {
    "2607B": PROJECT_ROOT / "logs" / "dual_camera-2607B" / "analysis" / "2607B_excursion_joint",
    "2608A": PROJECT_ROOT / "logs" / "dual_camera-2608A" / "analysis",
    "260907C": PROJECT_ROOT / "logs" / "dual_camera-260907C" / "analysis" / "excursion-joint",
}

TENDONS = ("FDP", "FDS", "ED")
JOINTS = ("DIP", "PIP", "MCP")
TRIAL_COLORS = {1: "#0072B2", 2: "#D55E00", 3: "#009E73"}
BASELINE_FRAMES = 5


@dataclass
class TrialData:
    finger: str
    tendon: str
    trial: int
    source: str
    path: Path
    active: np.ndarray
    raw_excursion_mm: np.ndarray
    smoothed_excursion_mm: np.ndarray
    raw_displacement_deg: dict[str, np.ndarray]
    smoothed_displacement_deg: dict[str, np.ndarray]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay raw and smoothed excursion-joint data separately for "
            "2607B, 2608A, and 260907C."
        )
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--smooth-only",
        action="store_true",
        help="Plot only the processed smoothed curves and omit raw overlays.",
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
        if trial is not None:
            discovered.append((path, tendon, trial))
    return discovered


def _baseline(dataframe: pd.DataFrame, joint: str) -> float:
    filtered_column = f"{joint}_filtered_deg"
    values = pd.to_numeric(dataframe[filtered_column], errors="coerce")
    if "tendon_quality_ok" in dataframe.columns:
        values = values.where(_as_bool(dataframe["tendon_quality_ok"]))
    baseline_values = values.iloc[:BASELINE_FRAMES].dropna()
    if baseline_values.empty:
        raw_values = pd.to_numeric(
            dataframe[f"{joint}_flexion_raw_deg"], errors="coerce"
        ).iloc[:BASELINE_FRAMES].dropna()
        if raw_values.empty:
            return np.nan
        return float(raw_values.median())
    return float(baseline_values.median())


def _load_trial(
    finger: str,
    path: Path,
    tendon: str,
    trial: int,
) -> TrialData:
    dataframe = pd.read_csv(path)
    required = {
        "active_motion",
        "excursion_raw_mm",
        "excursion_smoothed_mm",
        *[f"{joint}_flexion_raw_deg" for joint in JOINTS],
        *[f"{joint}_displacement_deg" for joint in JOINTS],
        *[f"{joint}_filtered_deg" for joint in JOINTS],
    }
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise ValueError(f"{path} is missing columns: {missing}")

    active = _as_bool(dataframe["active_motion"]).to_numpy(dtype=bool)
    raw_excursion = pd.to_numeric(
        dataframe["excursion_raw_mm"], errors="coerce"
    ).to_numpy(dtype=float)
    smoothed_excursion = pd.to_numeric(
        dataframe["excursion_smoothed_mm"], errors="coerce"
    ).to_numpy(dtype=float)
    raw_displacement: dict[str, np.ndarray] = {}
    smoothed_displacement: dict[str, np.ndarray] = {}
    for joint in JOINTS:
        baseline = _baseline(dataframe, joint)
        raw_angle = pd.to_numeric(
            dataframe[f"{joint}_flexion_raw_deg"], errors="coerce"
        ).to_numpy(dtype=float)
        raw_displacement[joint] = raw_angle - baseline
        smoothed_displacement[joint] = pd.to_numeric(
            dataframe[f"{joint}_displacement_deg"], errors="coerce"
        ).to_numpy(dtype=float)

    return TrialData(
        finger=finger,
        tendon=tendon,
        trial=trial,
        source=path.stem,
        path=path,
        active=active,
        raw_excursion_mm=raw_excursion,
        smoothed_excursion_mm=smoothed_excursion,
        raw_displacement_deg=raw_displacement,
        smoothed_displacement_deg=smoothed_displacement,
    )


def _plot_finger(
    finger: str,
    trials: list[TrialData],
    output_path: Path,
    plot_raw: bool = True,
) -> pd.DataFrame:
    figure, axes = plt.subplots(
        nrows=len(TENDONS),
        ncols=len(JOINTS),
        figsize=(15.5, 11.0),
        squeeze=False,
    )
    summary_rows: list[dict[str, object]] = []
    for row_index, tendon in enumerate(TENDONS):
        tendon_trials = sorted(
            [trial for trial in trials if trial.tendon == tendon],
            key=lambda item: item.trial,
        )
        for column_index, joint in enumerate(JOINTS):
            axis = axes[row_index, column_index]
            for trial in tendon_trials:
                color = TRIAL_COLORS.get(trial.trial, "#444444")
                raw_valid = (
                    trial.active
                    & np.isfinite(trial.raw_excursion_mm)
                    & np.isfinite(trial.raw_displacement_deg[joint])
                )
                smooth_valid = (
                    trial.active
                    & np.isfinite(trial.smoothed_excursion_mm)
                    & np.isfinite(trial.smoothed_displacement_deg[joint])
                )
                if plot_raw:
                    axis.plot(
                        trial.raw_excursion_mm[raw_valid],
                        trial.raw_displacement_deg[joint][raw_valid],
                        color=color,
                        linewidth=1.0,
                        alpha=0.65,
                        linestyle="--",
                        zorder=1,
                    )
                axis.plot(
                    trial.smoothed_excursion_mm[smooth_valid],
                    trial.smoothed_displacement_deg[joint][smooth_valid],
                    color=color,
                    linewidth=2.4,
                    alpha=1.0,
                    zorder=2,
                    label=f"trial {trial.trial}",
                )
                summary_rows.append(
                    {
                        "finger": finger,
                        "tendon": tendon,
                        "joint": joint,
                        "trial": trial.trial,
                        "source": trial.source,
                        "raw_active_valid_points": int(raw_valid.sum()),
                        "smoothed_active_valid_points": int(smooth_valid.sum()),
                        "raw_excursion_min_mm": float(
                            np.min(trial.raw_excursion_mm[raw_valid])
                        )
                        if raw_valid.any()
                        else np.nan,
                        "raw_excursion_max_mm": float(
                            np.max(trial.raw_excursion_mm[raw_valid])
                        )
                        if raw_valid.any()
                        else np.nan,
                        "smoothed_excursion_min_mm": float(
                            np.min(trial.smoothed_excursion_mm[smooth_valid])
                        )
                        if smooth_valid.any()
                        else np.nan,
                        "smoothed_excursion_max_mm": float(
                            np.max(trial.smoothed_excursion_mm[smooth_valid])
                        )
                        if smooth_valid.any()
                        else np.nan,
                    }
                )
            axis.axhline(0.0, color="#777777", linewidth=0.7, alpha=0.7)
            axis.grid(True, alpha=0.25)
            if row_index == 0:
                axis.set_title(joint)
            if column_index == 0:
                axis.set_ylabel(f"{tendon}\nΔ joint [deg]")
            if row_index == len(TENDONS) - 1:
                axis.set_xlabel(f"{tendon} excursion [mm]")

    trial_handles = [
        plt.Line2D(
            [0],
            [0],
            color=TRIAL_COLORS[trial],
            linewidth=2.0,
            label=f"trial {trial}",
        )
        for trial in sorted(TRIAL_COLORS)
    ]
    style_handles = [
        plt.Line2D([0], [0], color="#333333", linewidth=2.4, label="smoothed")
    ]
    if plot_raw:
        style_handles.insert(
            0,
            plt.Line2D(
                [0],
                [0],
                color="#333333",
                linewidth=1.0,
                linestyle="--",
                label="raw",
            ),
        )
    figure.legend(
        handles=trial_handles + style_handles,
        loc="upper center",
        ncol=5,
        bbox_to_anchor=(0.5, 0.945),
    )
    title = "smoothed only" if not plot_raw else "raw vs smoothed overlay"
    figure.suptitle(
        f"{finger}: excursion–joint relationships\n{title}",
        fontsize=15,
        y=0.995,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.89))
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)
    return pd.DataFrame(summary_rows)


def main() -> None:
    args = parse_args()
    output_root = _resolve_output_root(args.output_root)
    all_summary: list[pd.DataFrame] = []
    output_suffix = "smoothed_only" if args.smooth_only else "raw_smoothed_overlay"
    summary_filename = (
        "smooth_only_summary.csv"
        if args.smooth_only
        else "raw_smoothed_overlay_summary.csv"
    )
    for finger, root in FINGER_ROOTS.items():
        trials: list[TrialData] = []
        for path, tendon, trial in _discover_cleaned_files(finger, root):
            trials.append(_load_trial(finger, path, tendon, trial))
        expected = {(tendon, trial) for tendon in TENDONS for trial in (1, 2, 3)}
        found = {(item.tendon, item.trial) for item in trials}
        missing = sorted(expected - found)
        if missing:
            raise ValueError(f"Missing files for {finger}: {missing}")
        summary = _plot_finger(
            finger,
            trials,
            output_root / f"{finger}_{output_suffix}.png",
            plot_raw=not args.smooth_only,
        )
        all_summary.append(summary)
        print(f"{finger}: plotted {len(trials)} trials")

    pd.concat(all_summary, ignore_index=True).to_csv(
        output_root / summary_filename,
        index=False,
    )
    print(f"Saved outputs under {output_root}")


if __name__ == "__main__":
    main()
