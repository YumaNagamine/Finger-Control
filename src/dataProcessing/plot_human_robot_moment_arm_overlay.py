from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_HUMAN_ROOT = Path(r"C:\Users\ynaga\Downloads")
DEFAULT_HUMAN_FILES = {
    "DIP": "moment arm data DIP 2021_sorted_by_X.csv",
    "PIP": "moment arm data PIP 2021_sorted_by_X.csv",
    "MCP": "moment arm data MCP 2021_sorted_by_X.csv",
}
DEFAULT_HUMAN_1983_FILE = "moment arm data MCP from 1983_sorted_by_X.csv"
DEFAULT_ROBOT_ROOT = {
    robot_id: PROJECT_ROOT
    / "logs"
    / "analysis_comparison"
    / robot_id
    / "directional_moment_arm_safe"
    for robot_id in ("2607B", "2608A")
}
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "logs" / "analysis_comparison"

JOINTS = ["DIP", "PIP", "MCP"]
PLOT_GROUPS = [
    ("FDP", ["FDP"]),
    ("FDS", ["FDS"]),
    ("ED + EC", ["ED", "EC"]),
    ("DI", ["DI"]),
    ("PI", ["PI"]),
    ("LUM", ["LUM"]),
]
ROBOT_TENDONS = ["FDP", "FDS", "ED", "DI", "PI", "LUM"]
PAPER_TO_ROBOT_TENDON = {"EI": "ED", "LU": "LUM"}
DIRECTIONS = ["flexion", "extension"]
COLORS = {"flexion": "#0072B2", "extension": "#D55E00"}
HUMAN_LINE_STYLES = {
    "ED": {"color": "#222222", "linestyle": ":"},
    "EC": {"color": "#555555", "linestyle": (0, (1, 1))},
}
HUMAN_1983_STYLE = {"color": "#3F3F3F", "linestyle": "-."}
# The vertical scale is shared by every panel and by both robot figures so
# that moment-arm magnitudes remain directly comparable. The horizontal axis
# remains joint-specific because DIP, PIP, and MCP cover different ranges.
COMMON_Y_LIMITS = (-15.0, 17.0)
COMMON_Y_TICKS = np.arange(-15.0, 18.0, 5.0)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description=(
            "Overlay reconstructed human moment-arm curves on direction-separated "
            "robot curves for one robot finger."
        )
    )
    parser.add_argument("--robot-id", required=True, choices=["2607B", "2608A"])
    parser.add_argument("--human-root", default=str(DEFAULT_HUMAN_ROOT))
    parser.add_argument(
        "--human-1983",
        default=str(DEFAULT_HUMAN_ROOT / DEFAULT_HUMAN_1983_FILE),
        help="MCP human reference CSV reconstructed from the 1983 paper.",
    )
    parser.add_argument("--robot-root", default=None)
    parser.add_argument("--output-root", default=None)
    parser.add_argument("--label", default=None)
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def _resolve_path(raw_path: str | Path) -> Path:
    path = Path(raw_path).expanduser()
    if not path.is_absolute():
        path = PROJECT_ROOT / path
    return path.resolve()


def _load_human_joint(
    path: Path,
    joint: str,
    source: str = "human_paper_2021",
) -> tuple[pd.DataFrame, list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"Human reference file not found: {path}")
    raw = pd.read_csv(path, header=None)
    if raw.shape[0] < 3:
        raise ValueError(f"Human reference file has too few rows: {path}")

    rows: list[dict[str, object]] = []
    paper_tendons: list[str] = []
    for column in range(0, raw.shape[1] - 1, 2):
        paper_name = str(raw.iat[0, column]).strip()
        if not paper_name or paper_name.lower() == "nan":
            continue
        robot_name = PAPER_TO_ROBOT_TENDON.get(paper_name, paper_name)
        x = pd.to_numeric(raw.iloc[2:, column], errors="coerce")
        y = pd.to_numeric(raw.iloc[2:, column + 1], errors="coerce")
        valid = x.notna() & y.notna()
        if not bool(valid.any()):
            continue
        paper_tendons.append(paper_name)
        for angle_deg, moment_arm in zip(x[valid], y[valid]):
            rows.append(
                {
                    "source": source,
                    "joint": joint,
                    "tendon": robot_name,
                    "paper_tendon": paper_name,
                    "direction": "paper",
                    "trial": "paper",
                    "angle_deg": float(angle_deg),
                    "moment_arm_mm_per_rad": float(moment_arm),
                    "sign_transform": "none",
                }
            )

    if not rows:
        raise ValueError(f"No numeric tendon data found in {path}")
    return pd.DataFrame(rows), paper_tendons


def _load_human_1983(path: Path) -> tuple[pd.DataFrame, list[str]]:
    if not path.is_file():
        raise FileNotFoundError(f"1983 human reference file not found: {path}")
    raw = pd.read_csv(path, header=None)
    if raw.shape[0] < 3:
        raise ValueError(f"1983 human reference file has too few rows: {path}")

    name_to_robot_tendon = {
        "FDP dataset": "FDP",
        "FDP": "FDP",
        "FDS": "FDS",
        # The submission comparison uses the 1983 Extensor series as EC.
        "Extensor": "EC",
        "EI": "ED",
        "LU": "LUM",
        "LUM": "LUM",
        "DI": "DI",
        "PI": "PI",
    }
    rows: list[dict[str, object]] = []
    tendons: list[str] = []
    for column in range(0, raw.shape[1] - 1, 2):
        paper_name = str(raw.iat[0, column]).strip()
        robot_name = name_to_robot_tendon.get(paper_name)
        if robot_name is None:
            continue
        x = pd.to_numeric(raw.iloc[2:, column], errors="coerce")
        y = pd.to_numeric(raw.iloc[2:, column + 1], errors="coerce")
        valid = x.notna() & y.notna()
        if not bool(valid.any()):
            continue
        tendons.append(robot_name)
        for angle_deg, moment_arm in zip(x[valid], y[valid]):
            rows.append(
                {
                    "source": "human_paper_1983",
                    "joint": "MCP",
                    "tendon": robot_name,
                    "paper_tendon": paper_name,
                    "direction": "paper",
                    "trial": "paper",
                    "angle_deg": float(angle_deg),
                    "moment_arm_mm_per_rad": -float(moment_arm),
                    "sign_transform": "reversed",
                }
            )
    if not rows:
        raise ValueError(f"No numeric tendon data found in 1983 reference file: {path}")
    return pd.DataFrame(rows), tendons


def _load_human_data(
    human_root: Path,
    human_1983_path: Path,
) -> tuple[pd.DataFrame, dict[str, list[str]]]:
    frames: list[pd.DataFrame] = []
    availability: dict[str, list[str]] = {}
    for joint in JOINTS:
        frame, paper_tendons = _load_human_joint(
            human_root / DEFAULT_HUMAN_FILES[joint],
            joint,
            source="human_paper_2021",
        )
        # Use EC rather than the EDC series for the submission comparison.
        frame = frame[frame["tendon"] != "EDC"].copy()
        paper_tendons = [name for name in paper_tendons if name != "EDC"]
        frames.append(frame)
        availability[joint] = list(
            dict.fromkeys(PAPER_TO_ROBOT_TENDON.get(name, name) for name in paper_tendons)
        )
    reference_1983, tendons_1983 = _load_human_1983(human_1983_path)
    frames.append(reference_1983)
    availability["MCP"] = list(dict.fromkeys(availability["MCP"] + tendons_1983))
    return pd.concat(frames, ignore_index=True), availability


def _load_robot_data(robot_root: Path) -> pd.DataFrame:
    path = robot_root / "directional_moment_arm_curves.csv"
    if not path.is_file():
        raise FileNotFoundError(f"Robot directional curve file not found: {path}")
    curves = pd.read_csv(path)
    required = {
        "joint",
        "tendon",
        "direction",
        "trial",
        "angle_deg",
        "moment_arm_mm_per_rad",
    }
    missing = sorted(required.difference(curves.columns))
    if missing:
        raise ValueError(f"Robot curve table is missing columns: {missing}")
    curves = curves[curves["direction"].isin(DIRECTIONS)].copy()
    curves["angle_deg"] = pd.to_numeric(curves["angle_deg"], errors="coerce")
    curves["moment_arm_mm_per_rad"] = pd.to_numeric(
        curves["moment_arm_mm_per_rad"], errors="coerce"
    )
    curves = curves.dropna(subset=["angle_deg", "moment_arm_mm_per_rad"])
    return curves


def _joint_robot_range(robot: pd.DataFrame, joint: str) -> tuple[float, float]:
    group = robot[robot["joint"] == joint]
    if group.empty:
        raise ValueError(f"No robot data found for joint {joint}")
    return float(group["angle_deg"].min()), float(group["angle_deg"].max())


def _common_range(
    human: pd.DataFrame,
    robot: pd.DataFrame,
    joint: str,
    tendon: str,
) -> tuple[float, float, str]:
    human_group = human[(human["joint"] == joint) & (human["tendon"] == tendon)]
    human_min = float(human_group["angle_deg"].min())
    human_max = float(human_group["angle_deg"].max())
    if tendon == "EC":
        robot_min, robot_max = _joint_robot_range(robot, joint)
        basis = "joint robot range (paper-only EC)"
    else:
        robot_group = robot[(robot["joint"] == joint) & (robot["tendon"] == tendon)]
        if robot_group.empty:
            raise ValueError(f"No robot data for {joint}-{tendon}")
        robot_min = float(robot_group["angle_deg"].min())
        robot_max = float(robot_group["angle_deg"].max())
        basis = "human/robot tendon intersection"
    lower = max(human_min, robot_min)
    upper = min(human_max, robot_max)
    if upper <= lower:
        raise ValueError(
            f"No common angle range for {joint}-{tendon}: human "
            f"{human_min:.2f}..{human_max:.2f}, robot {robot_min:.2f}..{robot_max:.2f}"
        )
    return lower, upper, basis


def _summarize_overlay(
    human: pd.DataFrame,
    robot: pd.DataFrame,
    availability: dict[str, list[str]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve_rows: list[dict[str, object]] = []
    summary_rows: list[dict[str, object]] = []
    for joint in JOINTS:
        paper_names = availability[joint]
        selected_tendons = [
            PAPER_TO_ROBOT_TENDON.get(name, name) for name in paper_names
        ]
        for tendon in selected_tendons:
            human_group = human[(human["joint"] == joint) & (human["tendon"] == tendon)]
            if human_group.empty:
                continue
            common_min, common_max, basis = _common_range(
                human, robot, joint, tendon
            )
            human_group = human_group[
                human_group["angle_deg"].between(common_min, common_max)
            ].sort_values("angle_deg")
            for row in human_group.itertuples(index=False):
                curve_rows.append(
                    {
                        "source": row.source,
                        "joint": joint,
                        "tendon": tendon,
                        "paper_tendon": row.paper_tendon,
                        "direction": "paper",
                        "trial": "paper",
                        "angle_deg": row.angle_deg,
                        "moment_arm_mm_per_rad": row.moment_arm_mm_per_rad,
                        "sign_transform": row.sign_transform,
                        "trial_sd_mm_per_rad": np.nan,
                        "num_robot_trials": np.nan,
                        "common_angle_min_deg": common_min,
                        "common_angle_max_deg": common_max,
                    }
                )

            robot_group = robot[
                (robot["joint"] == joint)
                & (robot["tendon"] == tendon)
                & robot["angle_deg"].between(common_min, common_max)
            ]
            for direction in DIRECTIONS:
                direction_group = robot_group[robot_group["direction"] == direction]
                if direction_group.empty:
                    continue
                stats = (
                    direction_group.groupby("angle_deg")["moment_arm_mm_per_rad"]
                    .agg(["mean", "std", "count"])
                    .reset_index()
                    .sort_values("angle_deg")
                )
                for row in stats.itertuples(index=False):
                    curve_rows.append(
                        {
                            "source": "robot",
                            "joint": joint,
                            "tendon": tendon,
                            "paper_tendon": np.nan,
                            "direction": direction,
                            "trial": "mean",
                            "angle_deg": row.angle_deg,
                            "moment_arm_mm_per_rad": row.mean,
                            "sign_transform": np.nan,
                            "trial_sd_mm_per_rad": row.std,
                            "num_robot_trials": row.count,
                            "common_angle_min_deg": common_min,
                            "common_angle_max_deg": common_max,
                        }
                    )
            paper_names_for_tendon = sorted(
                set(
                    human_group["paper_tendon"]
                    .dropna()
                    .astype(str)
                    .tolist()
                )
            )
            robot_data_present = tendon in ROBOT_TENDONS and not robot_group.empty
            summary_rows.append(
                {
                    "joint": joint,
                    "tendon": tendon,
                    "paper_tendon": ",".join(paper_names_for_tendon),
                    "human_sources": ",".join(sorted(human_group["source"].unique())),
                    "paper_data_present": True,
                    "robot_data_present": robot_data_present,
                    "comparison_basis": basis,
                    "human_angle_min_deg": float(human_group["angle_deg"].min()),
                    "human_angle_max_deg": float(human_group["angle_deg"].max()),
                    "common_angle_min_deg": common_min,
                    "common_angle_max_deg": common_max,
                    "robot_flexion_trials": int(
                        robot_group[robot_group["direction"] == "flexion"]["trial"].nunique()
                    )
                    if robot_data_present
                    else 0,
                    "robot_extension_trials": int(
                        robot_group[robot_group["direction"] == "extension"]["trial"].nunique()
                    )
                    if robot_data_present
                    else 0,
                    "status": "ok" if robot_data_present else "paper_only_no_robot_EC",
                }
            )

    return pd.DataFrame(curve_rows), pd.DataFrame(summary_rows)


def _plot_overlay(
    curves: pd.DataFrame,
    availability: dict[str, list[str]],
    robot_id: str,
    output_path: Path,
    dpi: int,
) -> None:
    figure, axes = plt.subplots(
        nrows=len(JOINTS),
        ncols=len(PLOT_GROUPS),
        figsize=(20, 10.5),
        sharey=True,
        squeeze=False,
    )
    for row_index, joint in enumerate(JOINTS):
        paper_tendons = set(availability[joint])
        for column_index, (group_title, group_tendons) in enumerate(PLOT_GROUPS):
            axis = axes[row_index, column_index]
            displayed_tendons = paper_tendons.intersection(group_tendons)
            if not displayed_tendons:
                axis.text(
                    0.5,
                    0.5,
                    "not reported\nin paper",
                    ha="center",
                    va="center",
                    transform=axis.transAxes,
                    color="#777777",
                    fontsize=8,
                )
                axis.set_axis_off()
                continue

            panel = curves[
                (curves["joint"] == joint)
                & curves["tendon"].isin(displayed_tendons)
            ]
            for direction in DIRECTIONS:
                direction_panel = panel[panel["direction"] == direction]
                if direction_panel.empty:
                    continue
                axis.fill_between(
                    direction_panel["angle_deg"].to_numpy(dtype=float),
                    (
                        direction_panel["moment_arm_mm_per_rad"]
                        - direction_panel["trial_sd_mm_per_rad"].fillna(0.0)
                    ).to_numpy(dtype=float),
                    (
                        direction_panel["moment_arm_mm_per_rad"]
                        + direction_panel["trial_sd_mm_per_rad"].fillna(0.0)
                    ).to_numpy(dtype=float),
                    color=COLORS[direction],
                    alpha=0.12,
                    linewidth=0.0,
                )
                axis.plot(
                    direction_panel["angle_deg"],
                    direction_panel["moment_arm_mm_per_rad"],
                    color=COLORS[direction],
                    linestyle="-" if direction == "flexion" else "--",
                    linewidth=1.8,
                    label=f"robot {direction}",
                )

            for tendon in group_tendons:
                human_panel = panel[
                    (panel["tendon"] == tendon) & (panel["direction"] == "paper")
                ]
                for source in human_panel["source"].dropna().unique():
                    source_panel = human_panel[human_panel["source"] == source]
                    if source_panel.empty:
                        continue
                    if source == "human_paper_1983":
                        style = HUMAN_1983_STYLE
                        label = "human paper 1983 EC (sign reversed)"
                    else:
                        style = HUMAN_LINE_STYLES.get(
                            tendon, {"color": "#222222", "linestyle": ":"}
                        )
                        label = f"human paper 2021 {tendon}"
                    axis.plot(
                        source_panel["angle_deg"],
                        source_panel["moment_arm_mm_per_rad"],
                        color=style["color"],
                        linestyle=style["linestyle"],
                        linewidth=2.2,
                        label=label,
                    )
            axis.axhline(0.0, color="#777777", linewidth=0.6, alpha=0.7)
            axis.grid(True, alpha=0.22)
            axis.set_title(group_title)
            axis.set_ylim(*COMMON_Y_LIMITS)
            axis.set_yticks(COMMON_Y_TICKS)
            if column_index == 0:
                axis.set_ylabel(f"{joint}\nmm/rad")
            if row_index == len(JOINTS) - 1:
                axis.set_xlabel("flexion angle [deg]")

    legend_handles = [
        Line2D([0], [0], color=COLORS["flexion"], linestyle="-", linewidth=2.0, label="robot flexion mean"),
        Line2D([0], [0], color=COLORS["extension"], linestyle="--", linewidth=2.0, label="robot extension mean"),
        Line2D([0], [0], color=HUMAN_LINE_STYLES["ED"]["color"], linestyle=HUMAN_LINE_STYLES["ED"]["linestyle"], linewidth=2.2, label="human paper 2021 EI"),
        Line2D([0], [0], color=HUMAN_1983_STYLE["color"], linestyle=HUMAN_1983_STYLE["linestyle"], linewidth=2.2, label="human paper 1983 EC (sign reversed)"),
        Line2D([0], [0], color="#555555", linewidth=7.0, alpha=0.18, label="robot trial SD"),
    ]
    figure.legend(
        handles=legend_handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.99),
        ncol=3,
        frameon=False,
    )
    figure.suptitle(
        f"{robot_id}: robot directional moment arms overlaid with human reference data",
        fontsize=14,
        y=1.035,
    )
    figure.tight_layout(rect=(0, 0, 1, 0.94))
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    human_root = _resolve_path(args.human_root)
    robot_root = _resolve_path(args.robot_root or DEFAULT_ROBOT_ROOT[args.robot_id])
    output_root = _resolve_path(
        args.output_root
        or (DEFAULT_OUTPUT_ROOT / args.robot_id / "human_robot_overlay")
    )
    output_root.mkdir(parents=True, exist_ok=True)

    human, availability = _load_human_data(
        human_root,
        _resolve_path(args.human_1983),
    )
    robot = _load_robot_data(robot_root)
    curves, summary = _summarize_overlay(human, robot, availability)
    curves.to_csv(output_root / "human_robot_overlay_curves.csv", index=False)
    summary.to_csv(output_root / "human_robot_overlay_summary.csv", index=False)
    _plot_overlay(
        curves,
        availability,
        args.robot_id,
        output_root / "human_robot_moment_arm_overlay.png",
        args.dpi,
    )

    print(f"Saved human/robot overlay under {output_root}")
    print(summary.to_string(index=False, float_format=lambda value: f"{value:.3f}"))


if __name__ == "__main__":
    main()
