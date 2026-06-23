from __future__ import annotations

import json
import math
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


PROJECT_ROOT = Path(__file__).resolve().parents[2]
PROCESSED_DATA_ROOT = PROJECT_ROOT / "logs" / "dual_camera" / "processed_data"
OUTPUT_ROOT = PROJECT_ROOT / "logs" / "dual_camera" / "moment_arm_average"
TENDON_COLUMNS = ["FDP", "FDS", "EI", "DI", "PI", "LUM"]
MOMENT_ARM_COEFFICIENT_COLUMNS = ["moment_arm_c3", "moment_arm_c2", "moment_arm_c1", "moment_arm_c0"]
PLOT_DEGREE_LIMITS = {
    "DIP": (0.0, 90.0),
    "PIP": (0.0, 90.0),
    "MCP": (-20.0, 100.0),
}
MOMENT_ARM_Y_LIMITS = (-20.0, 25.0)
MOMENT_ARM_BOX_ASPECT = 0.45

# Edit these specifications in code when you want to generate a new average plot/json pair.
AVERAGE_SPECS = [
    {
        "output_name": "moment_arm_PIP_flexion_average",
        "joint": "PIP",
        "motion": "flexion",
    },
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
    motion = str(spec["motion"])
    source_dirs = _find_source_dirs(joint, motion)

    joint_to_tendon_rows: dict[str, list[list[float]]] = {tendon: [] for tendon in TENDON_COLUMNS}
    used_sources: list[str] = []

    for source_dir in source_dirs:
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
            joint_to_tendon_rows[tendon].append(coeffs)

    averaged_coefficients: dict[str, list[float]] = {}
    for tendon, coeff_rows in joint_to_tendon_rows.items():
        if not coeff_rows:
            raise ValueError(f"No coefficient rows collected for tendon={tendon!r}")
        averaged = np.mean(np.asarray(coeff_rows, dtype=np.float64), axis=0)
        averaged_coefficients[tendon] = [float(value) for value in averaged]

    json_payload = {
        "output_name": output_name,
        "joint": joint,
        "motion": motion,
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
    motion = str(spec["motion"])
    if joint not in PLOT_DEGREE_LIMITS:
        raise ValueError(f"Unsupported joint value: {joint!r}")
    print(f"Searching processed_data directories for joint={joint}, motion={motion}")

    payload, averaged_coefficients = _average_coefficients_for_spec(spec)
    for source_dir in payload["source_dirs"]:
        print(f"Matched source: {source_dir}")
    json_path = _save_average_json(output_name, payload)
    plot_path = _plot_average_moment_arm(output_name, joint, averaged_coefficients)
    return {"json_path": json_path, "plot_path": plot_path}


def main() -> None:
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
