from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROBOT_CURVES = (
    PROJECT_ROOT
    / "logs"
    / "analysis_comparison"
    / "2608A"
    / "directional_moment_arm"
    / "directional_moment_arm_curves.csv"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT
    / "logs"
    / "analysis_comparison"
    / "2608A"
    / "excursion_table2_comparison"
)

DIRECTIONS = ("flexion", "extension")
TENDONS = ("FDP", "FDS", "ED", "DI", "PI", "LUM")
DEG_100_IN_RAD = np.deg2rad(100.0)
JOINT_ORDER = {"DIP": 0, "PIP": 1, "MCP": 2}
PAPER_TENDON_ORDER = {"FP": 0, "FS": 1, "EI": 2, "EC": 3, "LU": 4, "RI": 5, "UI": 6}

# Table 2 reports flexion-extension values only here.  The MCP
# abduction-adduction row is intentionally excluded because the robot dataset
# used for this comparison contains no corresponding MCP abduction experiment.
TABLE2_ROWS = (
    {"joint": "DIP", "motion": "flexion_extension", "rom_mean_deg": 50.36, "rom_sd_deg": 6.60, "paper_tendon": "FP", "human_mean_mm_per_100deg": 7.30, "human_sd_mm_per_100deg": 1.17},
    {"joint": "DIP", "motion": "flexion_extension", "rom_mean_deg": 50.36, "rom_sd_deg": 6.60, "paper_tendon": "EI", "human_mean_mm_per_100deg": 2.94, "human_sd_mm_per_100deg": 0.88},
    {"joint": "DIP", "motion": "flexion_extension", "rom_mean_deg": 50.36, "rom_sd_deg": 6.60, "paper_tendon": "EC", "human_mean_mm_per_100deg": 3.04, "human_sd_mm_per_100deg": 0.78},
    {"joint": "DIP", "motion": "flexion_extension", "rom_mean_deg": 50.36, "rom_sd_deg": 6.60, "paper_tendon": "LU", "human_mean_mm_per_100deg": 1.12, "human_sd_mm_per_100deg": 0.87},
    {"joint": "DIP", "motion": "flexion_extension", "rom_mean_deg": 50.36, "rom_sd_deg": 6.60, "paper_tendon": "UI", "human_mean_mm_per_100deg": 2.39, "human_sd_mm_per_100deg": 0.72},
    {"joint": "PIP", "motion": "flexion_extension", "rom_mean_deg": 89.50, "rom_sd_deg": 11.70, "paper_tendon": "FP", "human_mean_mm_per_100deg": 14.06, "human_sd_mm_per_100deg": 2.03},
    {"joint": "PIP", "motion": "flexion_extension", "rom_mean_deg": 89.50, "rom_sd_deg": 11.70, "paper_tendon": "FS", "human_mean_mm_per_100deg": 10.81, "human_sd_mm_per_100deg": 1.92},
    {"joint": "PIP", "motion": "flexion_extension", "rom_mean_deg": 89.50, "rom_sd_deg": 11.70, "paper_tendon": "EI", "human_mean_mm_per_100deg": 5.37, "human_sd_mm_per_100deg": 0.51},
    {"joint": "PIP", "motion": "flexion_extension", "rom_mean_deg": 89.50, "rom_sd_deg": 11.70, "paper_tendon": "EC", "human_mean_mm_per_100deg": 5.58, "human_sd_mm_per_100deg": 1.09},
    {"joint": "PIP", "motion": "flexion_extension", "rom_mean_deg": 89.50, "rom_sd_deg": 11.70, "paper_tendon": "LU", "human_mean_mm_per_100deg": 2.99, "human_sd_mm_per_100deg": 1.11},
    {"joint": "PIP", "motion": "flexion_extension", "rom_mean_deg": 89.50, "rom_sd_deg": 11.70, "paper_tendon": "UI", "human_mean_mm_per_100deg": 4.72, "human_sd_mm_per_100deg": 0.88},
    {"joint": "MCP", "motion": "flexion_extension", "rom_mean_deg": 85.30, "rom_sd_deg": 18.40, "paper_tendon": "FP", "human_mean_mm_per_100deg": 18.98, "human_sd_mm_per_100deg": 2.58},
    {"joint": "MCP", "motion": "flexion_extension", "rom_mean_deg": 85.30, "rom_sd_deg": 18.40, "paper_tendon": "FS", "human_mean_mm_per_100deg": 21.04, "human_sd_mm_per_100deg": 2.50},
    {"joint": "MCP", "motion": "flexion_extension", "rom_mean_deg": 85.30, "rom_sd_deg": 18.40, "paper_tendon": "EI", "human_mean_mm_per_100deg": 15.07, "human_sd_mm_per_100deg": 2.42},
    {"joint": "MCP", "motion": "flexion_extension", "rom_mean_deg": 85.30, "rom_sd_deg": 18.40, "paper_tendon": "EC", "human_mean_mm_per_100deg": 13.96, "human_sd_mm_per_100deg": 2.24},
    {"joint": "MCP", "motion": "flexion_extension", "rom_mean_deg": 85.30, "rom_sd_deg": 18.40, "paper_tendon": "LU", "human_mean_mm_per_100deg": 15.26, "human_sd_mm_per_100deg": 3.97},
    {"joint": "MCP", "motion": "flexion_extension", "rom_mean_deg": 85.30, "rom_sd_deg": 18.40, "paper_tendon": "RI", "human_mean_mm_per_100deg": 5.88, "human_sd_mm_per_100deg": 2.29},
    {"joint": "MCP", "motion": "flexion_extension", "rom_mean_deg": 85.30, "rom_sd_deg": 18.40, "paper_tendon": "UI", "human_mean_mm_per_100deg": 7.87, "human_sd_mm_per_100deg": 2.37},
)

PAPER_TO_ROBOT = {
    "FP": "FDP",
    "FS": "FDS",
    "EI": "ED",
    "LU": "LUM",
    "RI": "DI",
    "UI": "PI",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Compare 2608A tendon excursion per 100 degrees with An et al. "
            "(1983) Table 2."
        )
    )
    parser.add_argument("--robot-curves", default=str(DEFAULT_ROBOT_CURVES))
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    return parser.parse_args()


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _load_robot_curves(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Robot curve file not found: {path}")
    curves = pd.read_csv(path)
    required = {
        "trial",
        "joint",
        "tendon",
        "direction",
        "angle_deg",
        "excursion_fit_mm",
    }
    missing = sorted(required.difference(curves.columns))
    if missing:
        raise ValueError(f"Robot curve table is missing columns: {missing}")

    curves = curves[
        curves["joint"].isin(["DIP", "PIP", "MCP"])
        & curves["tendon"].isin(TENDONS)
        & curves["direction"].isin(DIRECTIONS)
    ].copy()
    curves["trial_number"] = pd.to_numeric(curves["trial"], errors="coerce")
    curves["angle_deg"] = pd.to_numeric(curves["angle_deg"], errors="coerce")
    curves["excursion_fit_mm"] = pd.to_numeric(
        curves["excursion_fit_mm"], errors="coerce"
    )
    curves = curves.dropna(
        subset=["trial_number", "angle_deg", "excursion_fit_mm"]
    )
    curves["trial_number"] = curves["trial_number"].astype(int)
    if curves.empty:
        raise ValueError("No valid robot excursion curves were found.")
    return curves


def _endpoint_value(group: pd.DataFrame, angle_deg: float) -> float:
    ordered = (
        group[["angle_deg", "excursion_fit_mm"]]
        .groupby("angle_deg", as_index=False)["excursion_fit_mm"]
        .mean()
        .sort_values("angle_deg")
    )
    x = ordered["angle_deg"].to_numpy(dtype=float)
    y = ordered["excursion_fit_mm"].to_numpy(dtype=float)
    if angle_deg < x[0] - 1e-8 or angle_deg > x[-1] + 1e-8:
        raise ValueError(
            f"Requested angle {angle_deg:.6f} is outside curve range "
            f"{x[0]:.6f} to {x[-1]:.6f}."
        )
    return float(np.interp(angle_deg, x, y))


def _calculate_trial_rows(curves: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    pair_groups = curves.groupby(["joint", "tendon"], sort=True)
    for (joint, tendon), pair in pair_groups:
        group_map: dict[tuple[str, int], pd.DataFrame] = {}
        for (direction, trial), group in pair.groupby(
            ["direction", "trial_number"], sort=True
        ):
            group_map[(str(direction), int(trial))] = group.sort_values("angle_deg")

        if not group_map:
            continue
        lower = max(float(group["angle_deg"].min()) for group in group_map.values())
        upper = min(float(group["angle_deg"].max()) for group in group_map.values())
        if upper <= lower:
            continue
        interval_deg = upper - lower
        interval_rad = np.deg2rad(interval_deg)

        for direction in DIRECTIONS:
            direction_trials = sorted(
                trial for (candidate_direction, trial) in group_map
                if candidate_direction == direction
            )
            for trial in direction_trials:
                group = group_map[(direction, trial)]
                excursion_min = _endpoint_value(group, lower)
                excursion_max = _endpoint_value(group, upper)
                delta_excursion = excursion_max - excursion_min
                signed_per_100 = delta_excursion / interval_rad * DEG_100_IN_RAD
                rows.append(
                    {
                        "joint": joint,
                        "robot_tendon": tendon,
                        "direction": direction,
                        "trial": trial,
                        "angle_min_deg": lower,
                        "angle_max_deg": upper,
                        "angle_interval_deg": interval_deg,
                        "angle_interval_rad": interval_rad,
                        "excursion_at_min_angle_mm": excursion_min,
                        "excursion_at_max_angle_mm": excursion_max,
                        "signed_excursion_difference_mm": delta_excursion,
                        "mean_moment_arm_over_interval_mm_per_rad": (
                            delta_excursion / interval_rad
                        ),
                        "excursion_mm_per_100deg_signed": signed_per_100,
                        "excursion_mm_per_100deg_magnitude": abs(signed_per_100),
                        "status": "ok",
                    }
                )
    result = pd.DataFrame(rows)
    if result.empty:
        raise ValueError("No trial-level excursion values could be calculated.")
    result["_joint_order"] = result["joint"].map(JOINT_ORDER)
    return result.sort_values(
        ["_joint_order", "robot_tendon", "direction", "trial"]
    ).drop(columns="_joint_order")


def _mean_sd(values: pd.Series) -> tuple[float, float, int]:
    numeric = pd.to_numeric(values, errors="coerce").dropna().to_numpy(dtype=float)
    if numeric.size == 0:
        return np.nan, np.nan, 0
    sd = float(np.std(numeric, ddof=1)) if numeric.size > 1 else 0.0
    return float(np.mean(numeric)), sd, int(numeric.size)


def _summarize_robot(trial_rows: pd.DataFrame) -> pd.DataFrame:
    rows: list[dict[str, object]] = []
    for (joint, tendon), pair in trial_rows.groupby(
        ["joint", "robot_tendon"], sort=True
    ):
        direction_summary: dict[str, tuple[float, float, int]] = {}
        direction_trial_values: dict[str, pd.Series] = {}
        for direction in DIRECTIONS:
            direction_data = pair[pair["direction"] == direction]
            mean, sd, count = _mean_sd(
                direction_data["excursion_mm_per_100deg_magnitude"]
            )
            direction_summary[direction] = (mean, sd, count)
            direction_trial_values[direction] = direction_data.set_index("trial")[
                "excursion_mm_per_100deg_magnitude"
            ]

        balanced = pd.concat(direction_trial_values, axis=1, join="inner")
        if not balanced.empty:
            balanced_values = balanced[list(DIRECTIONS)].mean(axis=1)
            balanced_mean, balanced_sd, balanced_count = _mean_sd(balanced_values)
        else:
            balanced_mean, balanced_sd, balanced_count = np.nan, np.nan, 0

        first = pair.iloc[0]
        flexion_mean, flexion_sd, flexion_count = direction_summary["flexion"]
        extension_mean, extension_sd, extension_count = direction_summary["extension"]
        rows.append(
            {
                "joint": joint,
                "robot_tendon": tendon,
                "angle_min_deg": float(first["angle_min_deg"]),
                "angle_max_deg": float(first["angle_max_deg"]),
                "angle_interval_deg": float(first["angle_interval_deg"]),
                "angle_interval_rad": float(first["angle_interval_rad"]),
                "robot_flexion_mean_mm_per_100deg": flexion_mean,
                "robot_flexion_sd_mm_per_100deg": flexion_sd,
                "robot_flexion_n": flexion_count,
                "robot_extension_mean_mm_per_100deg": extension_mean,
                "robot_extension_sd_mm_per_100deg": extension_sd,
                "robot_extension_n": extension_count,
                "robot_direction_balanced_mean_mm_per_100deg": balanced_mean,
                "robot_direction_balanced_sd_mm_per_100deg": balanced_sd,
                "robot_direction_balanced_n": balanced_count,
                "status": "ok",
            }
        )
    result = pd.DataFrame(rows)
    result["_joint_order"] = result["joint"].map(JOINT_ORDER)
    return result.sort_values(["_joint_order", "robot_tendon"]).drop(
        columns="_joint_order"
    )


def _human_reference_table() -> pd.DataFrame:
    table = pd.DataFrame(TABLE2_ROWS)
    table["robot_tendon"] = table["paper_tendon"].map(PAPER_TO_ROBOT)
    table["source"] = "An et al. (1983), Table 2"
    return table


def _comparison_table(
    robot_summary: pd.DataFrame,
    human_table: pd.DataFrame,
) -> pd.DataFrame:
    comparison = human_table.merge(
        robot_summary,
        on=["joint", "robot_tendon"],
        how="left",
        suffixes=("_human", "_robot"),
    )
    comparison["difference_robot_minus_human_mm_per_100deg"] = (
        comparison["robot_direction_balanced_mean_mm_per_100deg"]
        - comparison["human_mean_mm_per_100deg"]
    )
    comparison["robot_to_human_ratio"] = (
        comparison["robot_direction_balanced_mean_mm_per_100deg"]
        / comparison["human_mean_mm_per_100deg"]
    )
    comparison["comparison_status"] = np.where(
        comparison["robot_direction_balanced_mean_mm_per_100deg"].notna(),
        "matched",
        "human_reference_only",
    )
    columns = [
        "joint",
        "motion",
        "paper_tendon",
        "robot_tendon",
        "rom_mean_deg",
        "rom_sd_deg",
        "human_mean_mm_per_100deg",
        "human_sd_mm_per_100deg",
        "angle_min_deg",
        "angle_max_deg",
        "angle_interval_deg",
        "robot_flexion_mean_mm_per_100deg",
        "robot_flexion_sd_mm_per_100deg",
        "robot_extension_mean_mm_per_100deg",
        "robot_extension_sd_mm_per_100deg",
        "robot_direction_balanced_mean_mm_per_100deg",
        "robot_direction_balanced_sd_mm_per_100deg",
        "difference_robot_minus_human_mm_per_100deg",
        "robot_to_human_ratio",
        "comparison_status",
        "source",
    ]
    comparison["_joint_order"] = comparison["joint"].map(JOINT_ORDER)
    comparison["_paper_tendon_order"] = comparison["paper_tendon"].map(
        PAPER_TENDON_ORDER
    )
    return comparison[columns + ["_joint_order", "_paper_tendon_order"]].sort_values(
        ["_joint_order", "_paper_tendon_order"]
    ).drop(columns=["_joint_order", "_paper_tendon_order"])


def _format_number(value: object, digits: int = 2) -> str:
    if pd.isna(value):
        return "NA"
    return f"{float(value):.{digits}f}"


def _format_mean_sd(mean: object, sd: object) -> str:
    if pd.isna(mean):
        return "NA"
    return f"{float(mean):.2f} ({float(sd):.2f})"


def _write_markdown(
    path: Path,
    comparison: pd.DataFrame,
    robot_summary: pd.DataFrame,
) -> None:
    lines = [
        "# 2608A excursion comparison with An et al. Table 2",
        "",
        "The robot values are calculated from the existing trial-specific moment-arm fitting curves.",
        "For each trial and direction, the fitted tendon excursion difference over the common moment-arm evaluation interval was divided by the interval in radians and multiplied by 100 degrees converted to radians.",
        "The primary robot value is the direction-balanced mean of the flexion and extension magnitudes.",
        "Human values are the flexion-extension means reported in Table 2. The MCP abduction-adduction row is excluded.",
        "",
        "## Matched comparisons",
        "",
        "Units: mm/100 deg. Values are mean (SD). Robot SD is the trial-to-trial SD, and human SD is the specimen SD reported by An et al.",
        "",
        "| Joint | Human tendon | Robot tendon | Robot flexion | Robot extension | Robot balanced mean | Human Table 2 | Robot - human | Robot/human |",
        "|---|---|---|---:|---:|---:|---:|---:|---:|",
    ]
    matched = comparison[comparison["comparison_status"] == "matched"]
    for row in matched.itertuples(index=False):
        lines.append(
            "| {joint} | {paper} | {robot} | {flex} | {ext} | {balanced} | {human} | {diff} | {ratio} |".format(
                joint=row.joint,
                paper=row.paper_tendon,
                robot=row.robot_tendon,
                flex=_format_mean_sd(
                    row.robot_flexion_mean_mm_per_100deg,
                    row.robot_flexion_sd_mm_per_100deg,
                ),
                ext=_format_mean_sd(
                    row.robot_extension_mean_mm_per_100deg,
                    row.robot_extension_sd_mm_per_100deg,
                ),
                balanced=_format_mean_sd(
                    row.robot_direction_balanced_mean_mm_per_100deg,
                    row.robot_direction_balanced_sd_mm_per_100deg,
                ),
                human=_format_mean_sd(
                    row.human_mean_mm_per_100deg,
                    row.human_sd_mm_per_100deg,
                ),
                diff=_format_number(row.difference_robot_minus_human_mm_per_100deg),
                ratio=_format_number(row.robot_to_human_ratio),
            )
        )

    human_only = comparison[comparison["comparison_status"] != "matched"]
    lines.extend(
        [
            "",
            "## Human reference rows without a corresponding robot tendon",
            "",
            "| Joint | Human tendon | Human Table 2 [mm/100 deg] | Reason |",
            "|---|---|---:|---|",
        ]
    )
    for row in human_only.itertuples(index=False):
        lines.append(
            f"| {row.joint} | {row.paper_tendon} | "
            f"{_format_mean_sd(row.human_mean_mm_per_100deg, row.human_sd_mm_per_100deg)} | No robot {row.paper_tendon} tendon channel |"
        )

    lines.extend(
        [
            "",
            "## Robot-only pairs without a Table 2 value",
            "",
            "| Joint | Robot tendon | Robot balanced mean [mm/100 deg] |",
            "|---|---|---:|",
        ]
    )
    mapped_robot = set(
        comparison.loc[comparison["comparison_status"] == "matched", ["joint", "robot_tendon"]]
        .itertuples(index=False, name=None)
    )
    for row in robot_summary.itertuples(index=False):
        if (row.joint, row.robot_tendon) in mapped_robot:
            continue
        lines.append(
            f"| {row.joint} | {row.robot_tendon} | "
            f"{_format_mean_sd(row.robot_direction_balanced_mean_mm_per_100deg, row.robot_direction_balanced_sd_mm_per_100deg)} |"
        )

    lines.extend(
        [
            "",
            "## Source and tendon mapping",
            "",
            "FP→FDP, FS→FDS, EI→ED, LU→LUM, RI→DI, and UI→PI.",
            "EC is retained as a human reference row but is not directly compared because the robot dataset has no EC channel.",
            "",
            "Source: K. N. An et al., Journal of Biomechanics, 1983, Table 2. DOI: https://doi.org/10.1016/0021-9290(83)90074-X",
        ]
    )
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def main() -> None:
    args = parse_args()
    robot_curves_path = _resolve_path(args.robot_curves)
    output_root = _resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    robot_curves = _load_robot_curves(robot_curves_path)
    trial_rows = _calculate_trial_rows(robot_curves)
    robot_summary = _summarize_robot(trial_rows)
    human_table = _human_reference_table()
    comparison = _comparison_table(robot_summary, human_table)

    trial_rows.to_csv(
        output_root / "robot_excursion_per_100deg_trial.csv", index=False
    )
    robot_summary.to_csv(
        output_root / "robot_excursion_per_100deg_summary.csv", index=False
    )
    human_table.to_csv(
        output_root / "an1983_table2_human_reference.csv", index=False
    )
    comparison.to_csv(
        output_root / "table2_excursion_comparison.csv", index=False
    )
    _write_markdown(
        output_root / "table2_excursion_comparison.md",
        comparison,
        robot_summary,
    )

    print(f"Saved Table 2 excursion comparison under {output_root}")
    print("\nMatched comparison summary [mm/100 deg]:")
    display_columns = [
        "joint",
        "paper_tendon",
        "robot_tendon",
        "robot_direction_balanced_mean_mm_per_100deg",
        "human_mean_mm_per_100deg",
        "comparison_status",
    ]
    print(
        comparison[comparison["comparison_status"] == "matched"][display_columns]
        .to_string(index=False)
    )


if __name__ == "__main__":
    main()
