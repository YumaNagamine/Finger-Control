from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_ROOT = PROJECT_ROOT / "logs" / "dual_camera-finger1" / "processed_data"
OUTPUT_ROOT = PROJECT_ROOT / "logs" / "dual_camera-finger1" / "moment_arm_average"
TENDON_COLUMNS = ["FDP", "FDS", "ED", "DI", "PI", "LUM"]
MOMENT_ARM_COEFFICIENT_COLUMNS = ["moment_arm_c3", "moment_arm_c2", "moment_arm_c1", "moment_arm_c0"]
PLOT_DEGREE_LIMITS = {
    "DIP": (0.0, 90.0),
    "PIP": (0.0, 90.0),
    "MCP": (-30.0, 100.0),
}
MOMENT_ARM_Y_LIMITS = (-20.0, 25.0)
MOMENT_ARM_BOX_ASPECT = 0.45

MOTIONS = ("flexion", "extension")
AVERAGE_SPECS = [
    {"output_name": f"moment_arm_{joint}_average", "joint": joint}
    for joint in ("DIP", "PIP", "MCP")
]


def _format_polynomial(coeffs: list[float], variable_name: str = "x") -> str:
    degree = len(coeffs) - 1
    parts: list[str] = []
    for idx, coeff in enumerate(coeffs):
        power = degree - idx
        if power == 0:
            term = f"{coeff:.5g}"
        elif power == 1:
            term = f"{coeff:.5g}*{variable_name}"
        else:
            term = f"{coeff:.5g}*{variable_name}^{power}"
        parts.append(term)
    return " + ".join(parts).replace("+ -", "- ")


def _load_polynomial_coefficients(csv_path: Path, joint: str) -> pd.DataFrame:
    if not csv_path.exists():
        raise FileNotFoundError(f"polynomial_coefficients.csv not found: {csv_path}")

    df = pd.read_csv(csv_path)
    required_columns = ["joint", "tendon", *MOMENT_ARM_COEFFICIENT_COLUMNS]
    missing = [column for column in required_columns if column not in df.columns]
    if missing:
        raise ValueError(f"Missing columns in {csv_path}: {missing}")

    joint_df = df[df["joint"] == joint].copy()
    if joint_df.empty:
        raise ValueError(f"No rows found for joint={joint!r} in {csv_path}")
    return joint_df


def _curve_coefficients_for_joint(
    source_curves_path: Path,
    joint: str,
    source_id: str,
) -> tuple[dict[str, object], dict[str, list[float]]]:
    """Convert the paper-selected curve into coefficients used for both directions."""
    curves = pd.read_csv(source_curves_path)
    required_columns = {
        "source",
        "trial",
        "joint",
        "tendon",
        "direction",
        "angle_deg",
        "moment_arm_mm_per_rad",
    }
    missing = sorted(required_columns.difference(curves.columns))
    if missing:
        raise ValueError(f"Missing columns in {source_curves_path}: {missing}")

    robot = curves[
        (curves["source"] == source_id)
        & (curves["trial"] == "mean_of_3")
        & (curves["joint"] == joint)
    ].copy()
    if robot.empty:
        raise ValueError(
            f"No source rows found for source={source_id!r}, joint={joint!r} "
            f"in {source_curves_path}"
        )

    robot["angle_deg"] = pd.to_numeric(robot["angle_deg"], errors="coerce")
    robot["moment_arm_mm_per_rad"] = pd.to_numeric(
        robot["moment_arm_mm_per_rad"], errors="coerce"
    )
    robot = robot.dropna(subset=["angle_deg", "moment_arm_mm_per_rad"])

    coefficients: dict[str, list[float]] = {}
    source_directions: dict[str, str] = {}
    source_tendon_labels: dict[str, str] = {}
    fit_metrics: dict[str, dict[str, object]] = {}

    for tendon in TENDON_COLUMNS:
        source_tendon = tendon
        if tendon == "ED" and not (robot["tendon"] == "ED").any():
            source_tendon = "EI"
        tendon_df = robot[robot["tendon"] == source_tendon].copy()
        if tendon_df.empty:
            raise ValueError(f"No curve rows found for tendon={tendon!r}")

        tendon_df = tendon_df.groupby("angle_deg", as_index=False).agg(
            moment_arm_mm_per_rad=("moment_arm_mm_per_rad", "mean"),
            direction=("direction", "first"),
        ).sort_values("angle_deg")
        if len(tendon_df) < 4:
            raise ValueError(
                f"At least four curve points are required for {joint}-{tendon}; "
                f"found {len(tendon_df)}"
            )

        angle_rad = np.deg2rad(tendon_df["angle_deg"].to_numpy(dtype=float))
        moment_arm = tendon_df["moment_arm_mm_per_rad"].to_numpy(dtype=float)
        fitted_coefficients = np.polyfit(angle_rad, moment_arm, deg=3)
        fitted_values = np.polyval(fitted_coefficients, angle_rad)
        residuals = moment_arm - fitted_values

        coefficients[tendon] = [float(value) for value in fitted_coefficients]
        source_directions[tendon] = str(tendon_df["direction"].iloc[0])
        source_tendon_labels[tendon] = source_tendon
        fit_metrics[tendon] = {
            "num_curve_points": int(len(tendon_df)),
            "angle_min_deg": float(tendon_df["angle_deg"].min()),
            "angle_max_deg": float(tendon_df["angle_deg"].max()),
            "fit_rmse_mm_per_rad": float(np.sqrt(np.mean(residuals**2))),
            "fit_max_abs_error_mm_per_rad": float(np.max(np.abs(residuals))),
        }

    payload = {
        "output_name": f"moment_arm_{joint}_average",
        "joint": joint,
        "source_curves": str(source_curves_path),
        "source_id": source_id,
        "source_trial": "mean_of_3",
        "source_direction_by_tendon": source_directions,
        "source_tendon_labels": source_tendon_labels,
        "direction_policy": (
            "Use the selected source direction as the ground truth for both "
            "flexion and extension prediction."
        ),
        "prediction_direction_dependent": False,
        "angle_unit": "rad",
        "moment_arm_unit": "mm/rad",
        "fit_degree": 3,
        "coefficient_order": MOMENT_ARM_COEFFICIENT_COLUMNS,
        "moment_arm_coefficients": coefficients,
        "moment_arm_polynomials": {
            tendon: _format_polynomial(coeffs, "x")
            for tendon, coeffs in coefficients.items()
        },
        "fit_metrics": fit_metrics,
    }
    return payload, coefficients


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Generate averaged moment-arm coefficient JSON files."
    )
    parser.add_argument("--source-curves", type=Path)
    parser.add_argument("--source-id", default="robot_2608A")
    parser.add_argument("--output-root", type=Path)
    return parser.parse_args()


def _find_source_dirs(joint: str, motion: str) -> list[Path]:
    valid_joints = {"DIP", "PIP", "MCP"}
    valid_motions = {"flexion", "extension"}
    if joint not in valid_joints:
        raise ValueError(f"Unsupported joint value: {joint!r}")
    if motion not in valid_motions:
        raise ValueError(f"Unsupported motion value: {motion!r}")

    matched_dirs = [
        directory
        for directory in PROCESSED_DATA_ROOT.iterdir()
        if directory.is_dir() and joint in directory.name and motion in directory.name
    ]
    matched_dirs = sorted(matched_dirs, key=lambda path: path.name.lower())
    if not matched_dirs:
        raise ValueError(f"No processed_data directories matched joint={joint!r}, motion={motion!r}")
    return matched_dirs


def _average_coefficients_for_spec(spec: dict[str, object]) -> tuple[dict[str, object], dict[str, list[float]]]:
    output_name = str(spec["output_name"])
    joint = str(spec["joint"])
    motion_to_tendon_rows: dict[str, dict[str, list[list[float]]]] = {
        motion: {tendon: [] for tendon in TENDON_COLUMNS}
        for motion in MOTIONS
    }
    used_sources: list[str] = []

    for motion in MOTIONS:
        for source_dir in _find_source_dirs(joint, motion):
            coefficient_csv_path = source_dir / "polynomial_coefficients.csv"
            joint_df = _load_polynomial_coefficients(coefficient_csv_path, joint)
            used_sources.append(str(source_dir))

            for tendon in TENDON_COLUMNS:
                tendon_df = joint_df[joint_df["tendon"] == tendon]
                if tendon_df.empty:
                    raise ValueError(f"No coefficient row for tendon={tendon!r} in {coefficient_csv_path}")
                row = tendon_df.iloc[0]
                coeffs = [float(row[column]) for column in MOMENT_ARM_COEFFICIENT_COLUMNS]
                if not np.all(np.isfinite(coeffs)):
                    raise ValueError(f"Non-finite coefficients for tendon={tendon!r} in {coefficient_csv_path}")
                motion_to_tendon_rows[motion][tendon].append(coeffs)

    averaged_coefficients: dict[str, list[float]] = {}
    for tendon in TENDON_COLUMNS:
        motion_means = [
            np.mean(np.asarray(motion_to_tendon_rows[motion][tendon], dtype=np.float64), axis=0)
            for motion in MOTIONS
        ]
        averaged = np.mean(np.asarray(motion_means, dtype=np.float64), axis=0)
        averaged_coefficients[tendon] = [float(value) for value in averaged]

    json_payload = {
        "output_name": output_name,
        "joint": joint,
        "source_motions": list(MOTIONS),
        "aggregation": "equal_weight_per_motion",
        "source_dirs": used_sources,
        "coefficient_order": MOMENT_ARM_COEFFICIENT_COLUMNS,
        "moment_arm_coefficients": averaged_coefficients,
        "moment_arm_polynomials": {
            tendon: _format_polynomial(coeffs, "x") for tendon, coeffs in averaged_coefficients.items()
        },
    }
    return json_payload, averaged_coefficients


def _plot_average_moment_arm(output_name: str, joint: str, averaged_coefficients: dict[str, list[float]]) -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)

    deg_min, deg_max = PLOT_DEGREE_LIMITS[joint]
    degree_values = np.linspace(deg_min, deg_max, 300)
    rad_values = degree_values * math.pi / 180.0

    fig, ax = plt.subplots(figsize=(12, 6))
    equation_lines: list[str] = []

    for tendon in TENDON_COLUMNS:
        coeffs = averaged_coefficients[tendon]
        moment_arm_values = np.polyval(np.asarray(coeffs, dtype=np.float64), rad_values)
        ax.plot(degree_values, moment_arm_values, linewidth=2.0, label=tendon)
        equation_lines.append(f"{tendon}: ma(rad) = {_format_polynomial(coeffs, 'x')}")

    ax.set_xlim((deg_min, deg_max))
    ax.set_ylim(MOMENT_ARM_Y_LIMITS)
    ax.set_box_aspect(MOMENT_ARM_BOX_ASPECT)
    ax.set_xlabel(f"{joint} flexion angle [deg]")
    ax.set_ylabel("Moment arm [mm/rad]")
    ax.set_title(f"Average moment arm: {output_name}")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.subplots_adjust(right=0.72)
    fig.text(0.74, 0.98, "\n".join(equation_lines), va="top", ha="left", fontsize=8, family="monospace")

    output_path = OUTPUT_ROOT / f"{output_name}.png"
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)
    return output_path


def _save_average_json(output_name: str, payload: dict[str, object]) -> Path:
    OUTPUT_ROOT.mkdir(parents=True, exist_ok=True)
    output_path = OUTPUT_ROOT / f"{output_name}.json"
    output_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
    return output_path


def process_average_spec(spec: dict[str, object]) -> dict[str, Path]:
    output_name = str(spec["output_name"])
    joint = str(spec["joint"])
    if joint not in PLOT_DEGREE_LIMITS:
        raise ValueError(f"Unsupported joint value: {joint!r}")
    print(f"Searching processed_data directories for joint={joint}, motions={MOTIONS}")

    payload, averaged_coefficients = _average_coefficients_for_spec(spec)
    for source_dir in payload["source_dirs"]:
        print(f"Matched source: {source_dir}")
    json_path = _save_average_json(output_name, payload)
    plot_path = _plot_average_moment_arm(output_name, joint, averaged_coefficients)
    return {"json_path": json_path, "plot_path": plot_path}


def main() -> None:
    args = _parse_args()
    global OUTPUT_ROOT
    if args.output_root is not None:
        OUTPUT_ROOT = args.output_root.expanduser().resolve()

    if args.source_curves is not None:
        source_curves_path = args.source_curves.expanduser().resolve()
        if not source_curves_path.is_file():
            raise FileNotFoundError(f"Source curve CSV not found: {source_curves_path}")

        print(f"Source curve CSV: {source_curves_path}")
        print(f"Output root: {OUTPUT_ROOT}")
        for joint in ("DIP", "PIP", "MCP"):
            payload, coefficients = _curve_coefficients_for_joint(
                source_curves_path=source_curves_path,
                joint=joint,
                source_id=args.source_id,
            )
            output_name = f"moment_arm_{joint}_average"
            json_path = _save_average_json(output_name, payload)
            plot_path = _plot_average_moment_arm(output_name, joint, coefficients)
            print(f"Saved JSON: {json_path}")
            print(f"Saved plot: {plot_path}")
        return

    print(f"Processed data root: {PROCESSED_DATA_ROOT}")
    print(f"Output root: {OUTPUT_ROOT}")
    print(f"Number of average specs: {len(AVERAGE_SPECS)}")

    for spec in AVERAGE_SPECS:
        output_name = str(spec["output_name"])
        print(f"Processing average spec: {output_name}")
        result = process_average_spec(spec)
        print(f"Saved JSON: {result['json_path']}")
        print(f"Saved plot: {result['plot_path']}")


if __name__ == "__main__":
    main()
