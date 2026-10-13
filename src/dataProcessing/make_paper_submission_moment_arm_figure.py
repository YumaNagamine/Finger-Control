from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from matplotlib.lines import Line2D

from plot_human_robot_moment_arm_overlay import (
    DEFAULT_HUMAN_ROOT,
    DEFAULT_HUMAN_1983_FILE,
    _load_human_data,
    _resolve_path,
)


PROJECT_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_ROBOT_CURVES = (
    PROJECT_ROOT
    / "logs"
    / "analysis_comparison"
    / "2608A"
    / "directional_moment_arm_safe"
    / "directional_moment_arm_curves.csv"
)
DEFAULT_OUTPUT_ROOT = (
    PROJECT_ROOT / "logs" / "analysis_comparison" / "paper_submission_2608A"
)

JOINTS = ["DIP", "PIP", "MCP"]
MAIN_PANEL_ROWS = [
    ("FDP", ["FDP"]),
    ("FDS", ["FDS"]),
    ("ED", ["ED"]),
    ("LUM", ["LUM"]),
]
ADDITIONAL_PANEL_ROWS = [
    ("PI", ["PI"]),
    ("DI", ["DI"]),
]
ALL_PANEL_ROWS = MAIN_PANEL_ROWS + ADDITIONAL_PANEL_ROWS
ROBOT_TENDONS = ["FDP", "FDS", "ED", "DI", "PI", "LUM"]
SELECTED_DIRECTION = {
    "FDP": "flexion",
    "FDS": "flexion",
    "ED": "extension",
    "DI": "extension",
    "PI": "extension",
    "LUM": "extension",
}
JOINT_X_LIMITS = {
    "DIP": (-10.0, 40.0),
    "PIP": (0.0, 90.0),
    "MCP": (-10.0, 90.0),
}
JOINT_X_TICKS = {
    "DIP": np.arange(-10.0, 41.0, 10.0),
    "PIP": np.arange(0.0, 91.0, 20.0),
    "MCP": np.arange(-10.0, 91.0, 20.0),
}
# Keep the same vertical scale as the preceding comparison figures.
COMMON_Y_LIMITS = (-15.0, 17.0)
COMMON_Y_TICKS = np.arange(-15.0, 18.0, 5.0)
FIGURE_CAPTION = """Figure. Moment-arm comparison between the robotic finger (This work) and human reference data. The robotic curves are the mean of three 2608A trials; FDP and FDS were selected from flexion trials, whereas ED, LUM, PI, and DI were selected from extension trials. No dispersion is shown. The 1983 human reference values were sign-reversed before plotting. The vertical axis is shared across panels, while the horizontal limits are joint-specific.

References
[1] Francis-Pester FW, Thomas R, Sforzin D, Ackland DC. The moment arms and leverage of the human finger muscles. Journal of Biomechanics. 2021;116:110180. https://doi.org/10.1016/j.jbiomech.2020.110180
[2] An KN, Ueba Y, Chao EY, Cooney WP, Linscheid RL. Tendon excursion and moment arm of index finger muscles. Journal of Biomechanics. 1983;16:419-425. https://doi.org/10.1016/0021-9290(83)90074-X
"""


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Create the paper-submission moment-arm figure for 2608A."
    )
    parser.add_argument("--robot-curves", default=str(DEFAULT_ROBOT_CURVES))
    parser.add_argument("--human-root", default=str(DEFAULT_HUMAN_ROOT))
    parser.add_argument(
        "--human-1983",
        default=str(DEFAULT_HUMAN_ROOT / DEFAULT_HUMAN_1983_FILE),
    )
    parser.add_argument("--output-root", default=str(DEFAULT_OUTPUT_ROOT))
    parser.add_argument("--dpi", type=int, default=300)
    return parser.parse_args()


def _load_robot_curves(path: Path) -> pd.DataFrame:
    if not path.is_file():
        raise FileNotFoundError(f"Robot curve file not found: {path}")
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
    curves = curves[
        curves["joint"].isin(JOINTS)
        & curves["tendon"].isin(ROBOT_TENDONS)
        & curves["direction"].isin(set(SELECTED_DIRECTION.values()))
    ].copy()
    curves["angle_deg"] = pd.to_numeric(curves["angle_deg"], errors="coerce")
    curves["moment_arm_mm_per_rad"] = pd.to_numeric(
        curves["moment_arm_mm_per_rad"], errors="coerce"
    )
    curves = curves.dropna(subset=["angle_deg", "moment_arm_mm_per_rad"])
    return curves


def _robot_mean_curve(
    robot: pd.DataFrame,
    joint: str,
    tendon: str,
    direction: str,
) -> pd.DataFrame:
    group = robot[
        (robot["joint"] == joint)
        & (robot["tendon"] == tendon)
        & (robot["direction"] == direction)
    ]
    if group.empty:
        raise ValueError(f"No robot data for {joint}-{tendon}-{direction}")
    return (
        group.groupby("angle_deg")["moment_arm_mm_per_rad"]
        .agg(["mean", "count"])
        .reset_index()
        .sort_values("angle_deg")
    )


def _human_plot_range(
    human: pd.DataFrame,
    robot: pd.DataFrame,
    joint: str,
    tendon: str,
) -> tuple[float, float]:
    group = human[(human["joint"] == joint) & (human["tendon"] == tendon)]
    if group.empty:
        return JOINT_X_LIMITS[joint]
    lower = max(float(group["angle_deg"].min()), JOINT_X_LIMITS[joint][0])
    upper = min(float(group["angle_deg"].max()), JOINT_X_LIMITS[joint][1])
    if tendon in ROBOT_TENDONS:
        robot_group = robot[(robot["joint"] == joint) & (robot["tendon"] == tendon)]
        if not robot_group.empty:
            lower = max(lower, float(robot_group["angle_deg"].min()))
            upper = min(upper, float(robot_group["angle_deg"].max()))
    return lower, upper


def _build_plot_data(
    robot: pd.DataFrame,
    human: pd.DataFrame,
    panel_specs: list[tuple[str, list[str]]],
) -> tuple[pd.DataFrame, pd.DataFrame]:
    curve_rows: list[dict[str, object]] = []
    panel_rows: list[dict[str, object]] = []
    for row_label, tendons in panel_specs:
        for joint in JOINTS:
            panel_has_human = False
            panel_has_robot = False
            for tendon in tendons:
                if tendon in ROBOT_TENDONS:
                    direction = SELECTED_DIRECTION[tendon]
                    mean_curve = _robot_mean_curve(robot, joint, tendon, direction)
                    panel_has_robot = True
                    for row in mean_curve.itertuples(index=False):
                        curve_rows.append(
                            {
                                "source": "robot_2608A",
                                "joint": joint,
                                "panel": row_label,
                                "tendon": tendon,
                                "direction": direction,
                                "trial": "mean_of_3",
                                "angle_deg": row.angle_deg,
                                "moment_arm_mm_per_rad": row.mean,
                                "num_trials": row.count,
                                "sign_transform": "none",
                            }
                        )

                human_group = human[
                    (human["joint"] == joint) & (human["tendon"] == tendon)
                ]
                if human_group.empty:
                    continue
                panel_has_human = True
                lower, upper = _human_plot_range(human, robot, joint, tendon)
                human_group = human_group[
                    human_group["angle_deg"].between(lower, upper)
                ].sort_values(["source", "angle_deg"])
                for row in human_group.itertuples(index=False):
                    curve_rows.append(
                        {
                            "source": row.source,
                            "joint": joint,
                            "panel": row_label,
                            "tendon": tendon,
                            "direction": "paper_reference",
                            "trial": "paper",
                            "angle_deg": row.angle_deg,
                            "moment_arm_mm_per_rad": row.moment_arm_mm_per_rad,
                            "num_trials": np.nan,
                            "sign_transform": row.sign_transform,
                        }
                    )
            panel_rows.append(
                {
                    "panel": row_label,
                    "joint": joint,
                    "robot_data_present": panel_has_robot,
                    "human_data_present": panel_has_human,
                    "status": (
                        "ok"
                        if panel_has_human
                        else "robot_only_no_human_reference"
                    ),
                }
            )
    return pd.DataFrame(curve_rows), pd.DataFrame(panel_rows)


def _styles(grayscale: bool) -> dict[str, dict[str, object]]:
    if grayscale:
        return {
            "this_work": {
                "color": "#111111",
                "linestyle": "-",
                "marker": "o",
                "label": "This work",
            },
            "human_paper_2021": {
                "color": "#222222",
                "linestyle": ":",
                "marker": "s",
                "label": "Francis-Pester et al.",
            },
            "human_paper_1983": {
                "color": "#555555",
                "linestyle": "-.",
                "marker": "^",
                "label": "An et al.",
            },
        }
    return {
        "this_work": {
            "color": "#0072B2",
            "linestyle": "-",
            "marker": "o",
            "label": "This work",
        },
        "human_paper_2021": {
            "color": "#222222",
            "linestyle": ":",
            "marker": "s",
            "label": "Francis-Pester et al.",
        },
        "human_paper_1983": {
            "color": "#555555",
            "linestyle": "-.",
            "marker": "^",
            "label": "An et al.",
        },
    }


def _plot_figure(
    curves: pd.DataFrame,
    panel_specs: list[tuple[str, list[str]]],
    joints: list[str],
    output_path: Path,
    grayscale: bool,
    dpi: int,
    title: str,
    figsize: tuple[float, float],
    horizontal: bool = False,
) -> None:
    styles = _styles(grayscale)
    if horizontal:
        layout = [
            (0, index, row_label, tendons, joint)
            for index, (row_label, tendons) in enumerate(panel_specs)
            for joint in joints
        ]
        nrows = 1
        ncols = len(layout)
    else:
        layout = [
            (row_index, column_index, row_label, tendons, joint)
            for row_index, (row_label, tendons) in enumerate(panel_specs)
            for column_index, joint in enumerate(joints)
        ]
        nrows = len(panel_specs)
        ncols = len(joints)
    figure, axes = plt.subplots(
        nrows=nrows,
        ncols=ncols,
        figsize=figsize,
        sharey=True,
        squeeze=False,
    )
    for row_index, column_index, row_label, tendons, joint in layout:
        axis = axes[row_index, column_index]
        panel = curves[
            (curves["panel"] == row_label) & (curves["joint"] == joint)
        ]

        for tendon in tendons:
            robot_panel = panel[panel["tendon"] == tendon]
            direction = SELECTED_DIRECTION.get(tendon)
            if direction is not None:
                robot_panel = robot_panel[robot_panel["source"] == "robot_2608A"]
                if not robot_panel.empty:
                    style = styles["this_work"]
                    markevery = max(len(robot_panel) // 10, 1)
                    axis.plot(
                        robot_panel["angle_deg"],
                        robot_panel["moment_arm_mm_per_rad"],
                        color=style["color"],
                        linestyle=style["linestyle"],
                        marker=style["marker"],
                        markevery=markevery,
                        markersize=4.0,
                        linewidth=2.4,
                        label=style["label"],
                    )

            human_panel = panel[
                (panel["tendon"] == tendon)
                & (panel["direction"] == "paper_reference")
            ]
            for source in human_panel["source"].dropna().unique():
                source_panel = human_panel[human_panel["source"] == source]
                style = styles[source]
                markevery = max(len(source_panel) // 7, 1)
                axis.plot(
                    source_panel["angle_deg"],
                    source_panel["moment_arm_mm_per_rad"],
                    color=style["color"],
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    markevery=markevery,
                    markersize=4.0,
                    linewidth=2.2,
                    label=style["label"],
                )

        axis.axhline(0.0, color="#777777", linewidth=0.6, alpha=0.75)
        axis.set_xlim(*JOINT_X_LIMITS[joint])
        axis.set_xticks(JOINT_X_TICKS[joint])
        axis.set_ylim(*COMMON_Y_LIMITS)
        axis.set_yticks(COMMON_Y_TICKS)
        axis.grid(True, alpha=0.22)
        if horizontal:
            axis.set_title(f"{joint} {row_label}", fontsize=15)
            axis.set_xlabel("flexion angle [deg]", fontsize=13, labelpad=8)
            if column_index == 0:
                axis.set_ylabel("mm/rad", fontsize=13, labelpad=8)
        else:
            if row_index == 0:
                axis.set_title(joint, fontsize=15, pad=10)
            if column_index == 0:
                axis.set_ylabel(f"{row_label} [mm/rad]", fontsize=13, labelpad=10)
            if row_index == len(panel_specs) - 1:
                axis.set_xlabel("flexion angle [deg]", fontsize=13, labelpad=8)

    handles = [
        Line2D(
            [0],
            [0],
            color=styles["this_work"]["color"],
            linestyle=styles["this_work"]["linestyle"],
            marker=styles["this_work"]["marker"],
            linewidth=2.4,
            markersize=4,
            label=styles["this_work"]["label"],
        ),
        Line2D(
            [0],
            [0],
            color=styles["human_paper_2021"]["color"],
            linestyle=styles["human_paper_2021"]["linestyle"],
            marker=styles["human_paper_2021"]["marker"],
            linewidth=2.2,
            markersize=4.5,
            label=styles["human_paper_2021"]["label"],
        ),
        Line2D(
            [0],
            [0],
            color=styles["human_paper_1983"]["color"],
            linestyle=styles["human_paper_1983"]["linestyle"],
            marker=styles["human_paper_1983"]["marker"],
            linewidth=2.2,
            markersize=4.5,
            label=styles["human_paper_1983"]["label"],
        ),
    ]
    figure.legend(
        handles=handles,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.94),
        ncol=3,
        frameon=False,
        fontsize=14,
        handlelength=3.0,
        columnspacing=2.0,
    )
    figure.tight_layout(rect=(0.03, 0.03, 0.98, 0.88), h_pad=1.0, w_pad=0.8)
    figure.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(figure)


def main() -> None:
    args = parse_args()
    robot_path = _resolve_path(args.robot_curves)
    human_root = _resolve_path(args.human_root)
    human_1983_path = _resolve_path(args.human_1983)
    output_root = _resolve_path(args.output_root)
    output_root.mkdir(parents=True, exist_ok=True)

    robot = _load_robot_curves(robot_path)
    human, _availability = _load_human_data(human_root, human_1983_path)
    curves, panels = _build_plot_data(robot, human, ALL_PANEL_ROWS)
    curves.to_csv(output_root / "paper_moment_arm_2608A_curves.csv", index=False)
    panels.to_csv(output_root / "paper_moment_arm_2608A_panel_summary.csv", index=False)
    _plot_figure(
        curves,
        MAIN_PANEL_ROWS,
        JOINTS,
        output_root / "paper_moment_arm_2608A_main.png",
        grayscale=False,
        dpi=args.dpi,
        title="",
        figsize=(10.5, 14.0),
    )
    _plot_figure(
        curves,
        MAIN_PANEL_ROWS,
        JOINTS,
        output_root / "paper_moment_arm_2608A_main_grayscale.png",
        grayscale=True,
        dpi=args.dpi,
        title="",
        figsize=(10.5, 14.0),
    )
    _plot_figure(
        curves,
        ADDITIONAL_PANEL_ROWS,
        ["MCP"],
        output_root / "paper_moment_arm_2608A_mcp_pi_di.png",
        grayscale=False,
        dpi=args.dpi,
        title="",
        figsize=(8.5, 6.8),
        horizontal=True,
    )
    _plot_figure(
        curves,
        ADDITIONAL_PANEL_ROWS,
        ["MCP"],
        output_root / "paper_moment_arm_2608A_mcp_pi_di_grayscale.png",
        grayscale=True,
        dpi=args.dpi,
        title="",
        figsize=(8.5, 6.8),
        horizontal=True,
    )
    (output_root / "paper_moment_arm_2608A_caption.txt").write_text(
        FIGURE_CAPTION,
        encoding="utf-8",
    )
    print(f"Saved paper-submission figures under {output_root}")
    print(panels.to_string(index=False))


if __name__ == "__main__":
    main()
