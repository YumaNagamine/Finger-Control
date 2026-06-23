from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


# Ensure src directory is importable when running as a script.
SRC_ROOT = Path(__file__).resolve().parents[1]
if str(SRC_ROOT) not in sys.path:
    sys.path.append(str(SRC_ROOT))

from utils.path_utils import resolve_path


PROJECT_ROOT = SRC_ROOT.parent
DEFAULT_INPUT_CSV = (
    PROJECT_ROOT / "logs" / "dual_camera" / "processed_csv" / "ValidateExtension.csv"
)
DEFAULT_OUTPUT_ROOT = PROJECT_ROOT / "logs" / "dual_camera" / "processed_data"
TENDON_COLUMNS = ["FDP", "FDS", "EI", "DI", "PI", "LUM"]
JOINT_COLUMNS = ["DIP", "PIP", "MCP"]
MOMENT_ARM_X_LIMITS = {
    "DIP": (0.0, 90.0),
    "PIP": (0.0, 90.0),
    "MCP": (-10.0, 100.0),
}
MOMENT_ARM_Y_LIMITS = (-20.0, 25.0)
MOMENT_ARM_BOX_ASPECT = 0.45
FLEXION_DEG_COLUMNS = {
    "DIP": "DIP_flexion_angle_degree",
    "PIP": "PIP_flexion_angle_degree",
    "MCP": "MCP_flexion_angle_degree",
}
FLEXION_RAD_COLUMNS = {
    "DIP": "DIP_flexion_angle_rad",
    "PIP": "PIP_flexion_angle_rad",
    "MCP": "MCP_flexion_angle_rad",
}
DISP_MM_RENAME_MAP = {
    "weight_0_disp_mm": "FDP",
    "weight_1_disp_mm": "FDS",
    "weight_2_disp_mm": "EI",
    "weight_3_disp_mm": "DI",
    "weight_4_disp_mm": "PI",
    "weight_5_disp_mm": "LUM",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Post-process one dual-camera processed CSV.")
    parser.add_argument("--csv", type=str, default=str(DEFAULT_INPUT_CSV))
    parser.add_argument("--output-root", type=str, default=str(DEFAULT_OUTPUT_ROOT))
    return parser.parse_args()


def _resolve_existing_file(raw_path: str, base_dir: Path) -> Path:
    path = resolve_path(raw_path, PROJECT_ROOT)
    if path is None:
        path = resolve_path(raw_path, base_dir)
    if path is None:
        raise ValueError(f"Invalid file path: {raw_path!r}")
    if not path.exists():
        raise FileNotFoundError(f"File not found: {path}")
    if not path.is_file():
        raise FileNotFoundError(f"Expected file path: {path}")
    return path


def _resolve_output_dir(raw_path: str, base_dir: Path) -> Path:
    path = resolve_path(raw_path, PROJECT_ROOT)
    if path is None:
        path = resolve_path(raw_path, base_dir)
    if path is None:
        raise ValueError(f"Invalid output directory path: {raw_path!r}")
    path.mkdir(parents=True, exist_ok=True)
    return path


def _required_columns() -> list[str]:
    required = ["elapsed_s"]
    required.extend(JOINT_COLUMNS)
    required.extend(DISP_MM_RENAME_MAP.keys())
    return required


def _validate_input_columns(df: pd.DataFrame) -> None:
    missing = [column for column in _required_columns() if column not in df.columns]
    if missing:
        raise ValueError(f"Input CSV is missing required columns: {missing}")


def _infer_target_joint(csv_path: Path) -> str:
    matches = [joint for joint in JOINT_COLUMNS if joint in csv_path.stem]
    if len(matches) != 1:
        print("Could not process! joint name is required in csv file name!")
        raise ValueError(f"Could not infer a single target joint from CSV file name: {csv_path.name}")
    return matches[0]


def build_processed_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    _validate_input_columns(df)
    processed = df.copy()
    processed = processed.rename(columns=DISP_MM_RENAME_MAP)

    for joint in JOINT_COLUMNS:
        deg_column = FLEXION_DEG_COLUMNS[joint]
        rad_column = FLEXION_RAD_COLUMNS[joint]
        processed[deg_column] = 180.0 - pd.to_numeric(processed[joint], errors="coerce")
        processed[rad_column] = processed[deg_column] * math.pi / 180.0

    return processed


def _format_polynomial(coeffs: np.ndarray, variable_name: str = "x") -> str:
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


def _fit_polynomial(x_rad: np.ndarray, y_mm: np.ndarray) -> tuple[np.ndarray, np.ndarray] | None:
    valid = np.isfinite(x_rad) & np.isfinite(y_mm)
    if int(np.count_nonzero(valid)) < 5:
        return None
    coeffs = np.polyfit(x_rad[valid], y_mm[valid], deg=4)
    derivative_coeffs = np.polyder(coeffs)
    return coeffs, derivative_coeffs


def _plot_joint_angles_vs_time(df: pd.DataFrame, output_path: Path) -> None:
    fig, ax = plt.subplots(figsize=(10, 5))
    x = pd.to_numeric(df["elapsed_s"], errors="coerce").to_numpy()
    for joint in JOINT_COLUMNS:
        y = pd.to_numeric(df[joint], errors="coerce").to_numpy()
        valid = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[valid], y[valid], s=10, alpha=0.75, label=joint)
    ax.set_xlabel("Time [s]")
    ax.set_ylabel("Joint angle [deg]")
    ax.set_title("Joint angle vs time")
    ax.grid(True, alpha=0.3)
    ax.legend()
    fig.tight_layout()
    fig.savefig(output_path, dpi=200)
    plt.close(fig)


def _plot_excursion_vs_flexion(
    df: pd.DataFrame,
    joint: str,
    output_path: Path,
    fit_results: dict[str, tuple[np.ndarray, np.ndarray] | None],
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.subplots_adjust(right=0.72)

    x = pd.to_numeric(df[FLEXION_DEG_COLUMNS[joint]], errors="coerce").to_numpy()
    equation_lines: list[str] = []

    for tendon in TENDON_COLUMNS:
        y = pd.to_numeric(df[tendon], errors="coerce").to_numpy()
        valid = np.isfinite(x) & np.isfinite(y)
        ax.scatter(x[valid], y[valid], s=10, alpha=0.7, label=tendon)

        fit_result = fit_results[tendon]
        if fit_result is None:
            equation_lines.append(f"{tendon}: insufficient valid points")
            continue
        coeffs, _ = fit_result
        equation_lines.append(f"{tendon}: y(rad) = {_format_polynomial(coeffs, 'x')}")

    ax.set_xlabel(f"{joint} flexion angle [deg]")
    ax.set_ylabel("Tendon excursion [mm]")
    ax.set_title(f"{joint} flexion angle vs tendon excursion")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.text(0.74, 0.98, "\n".join(equation_lines), va="top", ha="left", fontsize=8, family="monospace")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _plot_moment_arm(
    joint: str,
    df: pd.DataFrame,
    output_path: Path,
    fit_results: dict[str, tuple[np.ndarray, np.ndarray] | None],
) -> None:
    fig, ax = plt.subplots(figsize=(12, 6))
    fig.subplots_adjust(right=0.72)

    x_deg = pd.to_numeric(df[FLEXION_DEG_COLUMNS[joint]], errors="coerce").to_numpy()
    x_rad = pd.to_numeric(df[FLEXION_RAD_COLUMNS[joint]], errors="coerce").to_numpy()
    equation_lines: list[str] = []

    for tendon in TENDON_COLUMNS:
        fit_result = fit_results[tendon]
        if fit_result is None:
            equation_lines.append(f"{tendon}: insufficient valid points")
            continue

        _, derivative_coeffs = fit_result
        y = pd.to_numeric(df[tendon], errors="coerce").to_numpy()
        valid = np.isfinite(x_rad) & np.isfinite(x_deg) & np.isfinite(y)
        if int(np.count_nonzero(valid)) < 5:
            equation_lines.append(f"{tendon}: insufficient valid points")
            continue

        x_rad_valid = x_rad[valid]
        x_deg_valid = x_deg[valid]
        x_plot_rad = np.linspace(float(np.min(x_rad_valid)), float(np.max(x_rad_valid)), 200)
        x_plot_deg = np.linspace(float(np.min(x_deg_valid)), float(np.max(x_deg_valid)), 200)
        moment_arm = np.polyval(derivative_coeffs, x_plot_rad)
        ax.plot(x_plot_deg, moment_arm, label=tendon, linewidth=2.0)
        equation_lines.append(f"{tendon}: ma(rad) = {_format_polynomial(derivative_coeffs, 'x')}")

    ax.set_xlim(MOMENT_ARM_X_LIMITS[joint])
    ax.set_ylim(MOMENT_ARM_Y_LIMITS)
    ax.set_box_aspect(MOMENT_ARM_BOX_ASPECT)
    ax.set_xlabel(f"{joint} flexion angle [deg]")
    ax.set_ylabel("Moment arm [mm/rad]")
    ax.set_title(f"{joint} flexion angle vs moment arm")
    ax.grid(True, alpha=0.3)
    ax.legend(loc="upper left")
    fig.text(0.74, 0.98, "\n".join(equation_lines), va="top", ha="left", fontsize=8, family="monospace")
    fig.savefig(output_path, dpi=200, bbox_inches="tight")
    plt.close(fig)


def _collect_polynomial_rows(
    joint: str,
    fit_results: dict[str, tuple[np.ndarray, np.ndarray] | None],
    df: pd.DataFrame,
) -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    x_rad = pd.to_numeric(df[FLEXION_RAD_COLUMNS[joint]], errors="coerce").to_numpy()

    for tendon in TENDON_COLUMNS:
        y = pd.to_numeric(df[tendon], errors="coerce").to_numpy()
        valid = np.isfinite(x_rad) & np.isfinite(y)
        fit_result = fit_results[tendon]
        row: dict[str, object] = {
            "joint": joint,
            "tendon": tendon,
            "num_valid_points": int(np.count_nonzero(valid)),
            "excursion_polynomial": "",
            "moment_arm_polynomial": "",
        }

        for coeff_idx in range(5):
            row[f"excursion_c{4 - coeff_idx}"] = np.nan
        for coeff_idx in range(4):
            row[f"moment_arm_c{3 - coeff_idx}"] = np.nan

        if fit_result is not None:
            coeffs, derivative_coeffs = fit_result
            row["excursion_polynomial"] = _format_polynomial(coeffs, "x")
            row["moment_arm_polynomial"] = _format_polynomial(derivative_coeffs, "x")
            for coeff_idx, coeff in enumerate(coeffs):
                row[f"excursion_c{4 - coeff_idx}"] = float(coeff)
            for coeff_idx, coeff in enumerate(derivative_coeffs):
                row[f"moment_arm_c{3 - coeff_idx}"] = float(coeff)

        rows.append(row)
    return rows


def process_processed_csv(csv_path: Path, output_root: Path) -> dict[str, Path]:
    print(f"Processing {csv_path.name}...")
    target_joint = _infer_target_joint(csv_path)
    df = pd.read_csv(csv_path)
    processed_df = build_processed_dataframe(df)

    output_dir = output_root / csv_path.stem
    output_dir.mkdir(parents=True, exist_ok=True)

    processed_csv_path = output_dir / "processed_csv.csv"
    processed_df.to_csv(processed_csv_path, index=False)

    _plot_joint_angles_vs_time(processed_df, output_dir / "joint_angles_vs_time.png")

    coefficient_rows: list[dict[str, object]] = []
    for joint in [target_joint]:
        fit_results: dict[str, tuple[np.ndarray, np.ndarray] | None] = {}
        x_rad = pd.to_numeric(processed_df[FLEXION_RAD_COLUMNS[joint]], errors="coerce").to_numpy()

        for tendon in TENDON_COLUMNS:
            y = pd.to_numeric(processed_df[tendon], errors="coerce").to_numpy()
            fit_results[tendon] = _fit_polynomial(x_rad, y)

        _plot_excursion_vs_flexion(
            processed_df,
            joint,
            output_dir / f"{joint.lower()}_flexion_vs_excursion.png",
            fit_results,
        )
        _plot_moment_arm(
            joint,
            processed_df,
            output_dir / f"{joint.lower()}_moment_arm.png",
            fit_results,
        )
        coefficient_rows.extend(_collect_polynomial_rows(joint, fit_results, processed_df))

    coefficients_df = pd.DataFrame(coefficient_rows)
    coefficient_csv_path = output_dir / "polynomial_coefficients.csv"
    coefficients_df.to_csv(coefficient_csv_path, index=False)

    return {
        "processed_csv_path": processed_csv_path,
        "output_dir": output_dir,
        "coefficient_csv_path": coefficient_csv_path,
    }


def main() -> None:
    args = parse_args()
    csv_path = _resolve_existing_file(args.csv, Path.cwd())
    output_root = _resolve_output_dir(args.output_root, Path.cwd())
    print(f"Using input CSV: {csv_path}")
    print(f"Using output root: {output_root}")
    result = process_processed_csv(csv_path, output_root)
    print(f"Saved processed CSV to {result['processed_csv_path']}")
    print(f"Saved outputs under {result['output_dir']}")


if __name__ == "__main__":
    main()
