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
    PROJECT_ROOT / "logs" / "dual_camera" / "analysis" / "all_tensioned_FDP"
)
DEFAULT_OUTPUT_ROOT = (
    DEFAULT_INPUT_ROOT / "analysis_manual_cut_coupling"
)
MANUAL_CUT_END_PAIR_IDX = 604
RATIO_DENOMINATOR_THRESHOLD_DEG = 1.0
REPRESENTATIVE_FRACTIONS = (0.0, 0.25, 0.50, 0.75, 1.00)
RATIO_PAIRS = {
    "PIP_over_MCP": ("PIP", "MCP", 1.4),
    "DIP_over_MCP": ("DIP", "MCP", 0.8),
    "DIP_over_PIP": ("DIP", "PIP", None),
}
RATIO_LABELS = {
    "PIP_over_MCP": "|PIP| / |MCP|",
    "DIP_over_MCP": "|DIP| / |MCP|",
    "DIP_over_PIP": "|DIP| / |PIP|",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Manually cut the reversal in all-tensioned FDP data and "
            "calculate absolute joint-angle coupling ratios."
        )
    )
    parser.add_argument("--input-root", default=str(DEFAULT_INPUT_ROOT))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument(
        "--cut-end-pair-idx",
        type=int,
        default=MANUAL_CUT_END_PAIR_IDX,
        help="Inclusive last pair_idx retained before the excursion reversal.",
    )
    return parser.parse_args()


def _resolve_path(raw_path: str) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _find_input_csv(input_root: Path) -> Path:
    paths = sorted(
        path
        for path in (input_root / "cleaned").glob("*.csv")
        if "rom_summary" not in path.name.lower()
    )
    if len(paths) != 1:
        raise ValueError(f"Expected one cleaned data CSV, found {paths}")
    return paths[0]


def _as_bool(series: pd.Series) -> pd.Series:
    if pd.api.types.is_bool_dtype(series):
        return series.fillna(False)
    return series.astype(str).str.strip().str.lower().isin({"true", "1", "yes"})


def _load_data(input_csv: Path, cut_end_pair_idx: int) -> tuple[pd.DataFrame, int]:
    dataframe = pd.read_csv(input_csv)
    required = {
        "pair_idx",
        "active_motion",
        "excursion_smoothed_mm",
        "DIP_smoothed_deg",
        "PIP_smoothed_deg",
        "MCP_smoothed_deg",
    }
    missing = sorted(required.difference(dataframe.columns))
    if missing:
        raise ValueError(f"{input_csv} is missing columns: {missing}")

    active = _as_bool(dataframe["active_motion"])
    if "tendon_quality_ok" in dataframe.columns:
        active &= _as_bool(dataframe["tendon_quality_ok"])
    excursion = pd.to_numeric(dataframe["excursion_smoothed_mm"], errors="coerce")
    valid_excursion = active & excursion.notna()
    first_index = int(np.flatnonzero(valid_excursion.to_numpy())[0])
    pair_idx = pd.to_numeric(dataframe["pair_idx"], errors="coerce")
    if not bool((pair_idx == cut_end_pair_idx).any()):
        raise ValueError(f"Manual cut pair_idx={cut_end_pair_idx} is not in {input_csv}")
    kept = pair_idx <= cut_end_pair_idx
    dataframe = dataframe.loc[kept].copy()
    return dataframe, first_index


def _valid_arrays(dataframe: pd.DataFrame) -> tuple[np.ndarray, dict[str, np.ndarray], np.ndarray]:
    active = _as_bool(dataframe["active_motion"])
    if "tendon_quality_ok" in dataframe.columns:
        active &= _as_bool(dataframe["tendon_quality_ok"])
    excursion = pd.to_numeric(dataframe["excursion_smoothed_mm"], errors="coerce")
    valid_excursion = active & excursion.notna()
    if int(valid_excursion.sum()) < 2:
        raise ValueError("Not enough valid points after manual cut")
    first_excursion = float(excursion[valid_excursion].iloc[0])
    relative_excursion = (excursion - first_excursion).to_numpy(dtype=float)
    angles = {
        joint: pd.to_numeric(dataframe[f"{joint}_smoothed_deg"], errors="coerce")
        .to_numpy(dtype=float)
        for joint in ("DIP", "PIP", "MCP")
    }
    return relative_excursion, angles, valid_excursion.to_numpy(dtype=bool)


def _sorted_unique_curve(
    x: np.ndarray,
    y: np.ndarray,
) -> tuple[np.ndarray, np.ndarray]:
    valid = np.isfinite(x) & np.isfinite(y)
    x = x[valid]
    y = y[valid]
    order = np.argsort(x, kind="stable")
    x = x[order]
    y = y[order]
    unique_x, inverse = np.unique(x, return_inverse=True)
    sums = np.bincount(inverse, weights=y)
    counts = np.bincount(inverse)
    return unique_x, sums / counts


def _interp_inside(
    grid: np.ndarray,
    x: np.ndarray,
    y: np.ndarray,
) -> np.ndarray:
    finite = np.isfinite(x) & np.isfinite(y)
    x = x[finite]
    y = y[finite]
    result = np.full(grid.shape, np.nan)
    if x.size < 2:
        return result
    inside = (grid >= x[0]) & (grid <= x[-1])
    result[inside] = np.interp(grid[inside], x, y)
    return result


def _plot_cut_qc(
    original: pd.DataFrame,
    cut_end_pair_idx: int,
    output_path: Path,
) -> None:
    pair_idx = pd.to_numeric(original["pair_idx"], errors="coerce")
    excursion = pd.to_numeric(original["excursion_smoothed_mm"], errors="coerce")
    kept = pair_idx <= cut_end_pair_idx
    figure, axis = plt.subplots(figsize=(8.5, 4.8))
    axis.plot(pair_idx, excursion, color="#999999", linewidth=1.2, label="full smoothed excursion")
    axis.plot(
        pair_idx[kept],
        excursion[kept],
        color="#0072B2",
        linewidth=2.0,
        label="retained forward segment",
    )
    axis.axvline(
        cut_end_pair_idx,
        color="#D55E00",
        linestyle="--",
        linewidth=1.5,
        label=f"manual cut: pair_idx={cut_end_pair_idx}",
    )
    axis.axvspan(cut_end_pair_idx, float(pair_idx.max()), color="#D55E00", alpha=0.08)
    axis.grid(True, alpha=0.25)
    axis.set_xlabel("pair index")
    axis.set_ylabel("smoothed FDP excursion [mm]")
    axis.set_title("All-tensioned FDP: manual reversal cut")
    axis.legend()
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def _calculate_ratios(
    dataframe: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame, float]:
    relative_excursion, angles, valid = _valid_arrays(dataframe)
    x = relative_excursion[valid]
    maximum = float(np.max(x))
    if maximum <= 0.0:
        raise ValueError("Non-positive retained FDP excursion range")
    grid = np.linspace(0.0, maximum, 201)
    representative_grid = np.asarray(REPRESENTATIVE_FRACTIONS) * maximum
    curve_rows: list[dict[str, object]] = []
    representative_rows: list[dict[str, object]] = []

    for pair, (numerator_joint, denominator_joint, target) in RATIO_PAIRS.items():
        numerator_x, numerator_y = _sorted_unique_curve(
            x,
            angles[numerator_joint][valid],
        )
        denominator_x, denominator_y = _sorted_unique_curve(
            x,
            angles[denominator_joint][valid],
        )
        numerator = _interp_inside(grid, numerator_x, numerator_y)
        denominator = _interp_inside(grid, denominator_x, denominator_y)
        ratio = np.full(grid.shape, np.nan)
        denominator_ok = np.abs(denominator) >= RATIO_DENOMINATOR_THRESHOLD_DEG
        ratio[denominator_ok] = (
            np.abs(numerator[denominator_ok])
            / np.abs(denominator[denominator_ok])
        )
        for x_value, numerator_value, denominator_value, ratio_value in zip(
            grid,
            numerator,
            denominator,
            ratio,
        ):
            curve_rows.append(
                {
                    "tendon": "FDP",
                    "ratio_pair": pair,
                    "relative_excursion_mm": float(x_value),
                    "numerator_joint": numerator_joint,
                    "denominator_joint": denominator_joint,
                    "numerator_angle_deg": float(numerator_value),
                    "denominator_angle_deg": float(denominator_value),
                    "ratio": float(ratio_value)
                    if np.isfinite(ratio_value)
                    else np.nan,
                    "human_target_ratio": target,
                }
            )
        rep_ratio = _interp_inside(representative_grid, grid, ratio)
        rep_num = _interp_inside(representative_grid, grid, numerator)
        rep_den = _interp_inside(representative_grid, grid, denominator)
        for fraction, x_value, num_value, den_value, ratio_value in zip(
            REPRESENTATIVE_FRACTIONS,
            representative_grid,
            rep_num,
            rep_den,
            rep_ratio,
        ):
            representative_rows.append(
                {
                    "tendon": "FDP",
                    "ratio_pair": pair,
                    "fraction": float(fraction),
                    "relative_excursion_mm": float(x_value),
                    "numerator_joint": numerator_joint,
                    "denominator_joint": denominator_joint,
                    "numerator_angle_deg": float(num_value),
                    "denominator_angle_deg": float(den_value),
                    "ratio": float(ratio_value)
                    if np.isfinite(ratio_value)
                    else np.nan,
                    "human_target_ratio": target,
                    "ratio_error": (
                        float(ratio_value - target)
                        if target is not None and np.isfinite(ratio_value)
                        else np.nan
                    ),
                }
            )
    return pd.DataFrame(curve_rows), pd.DataFrame(representative_rows), maximum


def _plot_ratios(
    curve_df: pd.DataFrame,
    output_path: Path,
    maximum: float,
) -> None:
    figure, axes = plt.subplots(
        nrows=1,
        ncols=len(RATIO_PAIRS),
        figsize=(15.0, 4.8),
        squeeze=False,
    )
    for column_index, pair in enumerate(RATIO_PAIRS):
        axis = axes[0, column_index]
        values = curve_df.query("ratio_pair == @pair")
        target_values = values["human_target_ratio"].dropna()
        target = float(target_values.iloc[0]) if not target_values.empty else None
        axis.plot(
            values["relative_excursion_mm"],
            values["ratio"],
            color="#0072B2",
            linewidth=2.0,
            label="all-tensioned FDP",
        )
        if target is not None:
            axis.axhline(
                target,
                color="#D55E00",
                linestyle="--",
                linewidth=1.3,
                label=f"human target = {target:g}",
            )
        axis.grid(True, alpha=0.25)
        axis.set_xlim(0.0, maximum)
        axis.set_title(RATIO_LABELS.get(pair, pair))
        axis.set_xlabel("relative FDP excursion [mm]")
        if column_index == 0:
            axis.set_ylabel("absolute flexion-angle ratio")
        axis.legend()
    figure.suptitle(
        "All-tensioned FDP joint coupling after manual reversal cut",
        fontsize=14,
        y=1.02,
    )
    figure.tight_layout()
    figure.savefig(output_path, dpi=300, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    input_root = _resolve_path(args.input_root)
    output_root = _resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)
    input_csv = _find_input_csv(input_root)
    original = pd.read_csv(input_csv)
    cut_data, _ = _load_data(input_csv, args.cut_end_pair_idx)
    cut_data.to_csv(output_root / "all_tensioned_FDP_forward_cut.csv", index=False)
    _plot_cut_qc(
        original,
        args.cut_end_pair_idx,
        output_root / "all_tensioned_FDP_manual_cut_qc.png",
    )
    curve_df, representative_df, maximum = _calculate_ratios(cut_data)
    _plot_ratios(
        curve_df,
        output_root / "all_tensioned_FDP_coupling_ratios.png",
        maximum,
    )
    curve_df.to_csv(output_root / "all_tensioned_FDP_coupling_ratio_curve.csv", index=False)
    representative_df.to_csv(
        output_root / "all_tensioned_FDP_coupling_ratio_representative_points.csv",
        index=False,
    )
    print(f"Input: {input_csv}")
    print(f"Manual cut: pair_idx <= {args.cut_end_pair_idx}")
    print(f"Retained relative excursion max: {maximum:.3f} mm")
    print(f"Saved outputs under {output_root}")


if __name__ == "__main__":
    main()
