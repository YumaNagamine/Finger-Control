"""Fit per-tendon encoder excursion scales from measured calibration samples."""

from __future__ import annotations

import argparse
import csv
import json
import math
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable, Mapping

from controller.servo_mapping import TENDONS


REQUIRED_COLUMNS = ("tendon", "excursion_mm", "encoder_deg")


def _fit_line(samples: Iterable[tuple[float, float]]) -> dict[str, Any]:
    points = list(samples)
    if len(points) < 2:
        raise ValueError("At least two samples are required")
    x_values = [point[0] for point in points]
    y_values = [point[1] for point in points]
    x_mean = sum(x_values) / len(x_values)
    y_mean = sum(y_values) / len(y_values)
    denominator = sum((value - x_mean) ** 2 for value in x_values)
    if denominator <= 0.0:
        raise ValueError("excursion_mm must contain at least two distinct values")
    slope = sum(
        (x_value - x_mean) * (y_value - y_mean)
        for x_value, y_value in points
    ) / denominator
    intercept = y_mean - slope * x_mean
    residuals = [
        y_value - (slope * x_value + intercept)
        for x_value, y_value in points
    ]
    rmse = math.sqrt(sum(value * value for value in residuals) / len(residuals))
    total = sum((value - y_mean) ** 2 for value in y_values)
    r_squared = None if total <= 0.0 else 1.0 - sum(
        value * value for value in residuals
    ) / total
    return {
        "sample_count": len(points),
        "slope_deg_per_mm": slope,
        "intercept_deg": intercept,
        "rmse_deg": rmse,
        "r_squared": r_squared,
        "excursion_min_mm": min(x_values),
        "excursion_max_mm": max(x_values),
    }


def _finite(row: Mapping[str, str], name: str, row_number: int) -> float:
    try:
        value = float(row[name])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"row {row_number}: {name} must be numeric") from exc
    if not math.isfinite(value):
        raise ValueError(f"row {row_number}: {name} must be finite")
    return value


def fit_encoder_calibration(input_csv: str | Path) -> dict[str, Any]:
    path = Path(input_csv)
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        reader = csv.DictReader(handle)
        missing = [name for name in REQUIRED_COLUMNS if name not in (reader.fieldnames or [])]
        if missing:
            raise ValueError("Missing CSV columns: " + ", ".join(missing))
        for row_number, raw in enumerate(reader, start=2):
            tendon = str(raw.get("tendon", "")).strip()
            if tendon not in TENDONS:
                raise ValueError(f"row {row_number}: unknown tendon {tendon!r}")
            rows.append(
                {
                    "tendon": tendon,
                    "excursion_mm": _finite(raw, "excursion_mm", row_number),
                    "encoder_deg": _finite(raw, "encoder_deg", row_number),
                    "direction": str(raw.get("direction", "unspecified")).strip()
                    or "unspecified",
                    "load_n": (
                        None
                        if not str(raw.get("load_n", "")).strip()
                        else _finite(raw, "load_n", row_number)
                    ),
                    "trial": str(raw.get("trial", "")).strip() or None,
                }
            )

    by_tendon: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        by_tendon[row["tendon"]].append(row)
    missing_tendons = [tendon for tendon in TENDONS if tendon not in by_tendon]
    if missing_tendons:
        raise ValueError("No samples for: " + ", ".join(missing_tendons))

    tendon_reports: dict[str, Any] = {}
    recommended_scales: dict[str, float] = {}
    recommended_directions: dict[str, int] = {}
    for tendon in TENDONS:
        tendon_rows = by_tendon[tendon]
        overall = _fit_line(
            (row["excursion_mm"], row["encoder_deg"])
            for row in tendon_rows
        )
        direction_groups: dict[str, list[dict[str, Any]]] = defaultdict(list)
        for row in tendon_rows:
            direction_groups[row["direction"]].append(row)
        direction_fits = {}
        for direction, group in sorted(direction_groups.items()):
            if len(group) < 2 or len({row["excursion_mm"] for row in group}) < 2:
                continue
            direction_fits[direction] = _fit_line(
                (row["excursion_mm"], row["encoder_deg"])
                for row in group
            )
        slope = float(overall["slope_deg_per_mm"])
        if slope == 0.0:
            raise ValueError(f"{tendon}: fitted slope is zero")
        positive_slopes = [
            abs(float(item["slope_deg_per_mm"]))
            for item in direction_fits.values()
        ]
        hysteresis = (
            None
            if len(positive_slopes) < 2
            else max(positive_slopes) - min(positive_slopes)
        )
        loads = [float(row["load_n"]) for row in tendon_rows if row["load_n"] is not None]
        tendon_reports[tendon] = {
            "overall_fit": overall,
            "direction_fits": direction_fits,
            "direction_slope_spread_deg_per_mm": hysteresis,
            "load_n_min": None if not loads else min(loads),
            "load_n_max": None if not loads else max(loads),
            "trial_count": len(
                {row["trial"] for row in tendon_rows if row["trial"] is not None}
            ),
        }
        recommended_scales[tendon] = abs(slope)
        recommended_directions[tendon] = 1 if slope > 0.0 else -1

    return {
        "schema_version": 1,
        "source_csv": str(path.resolve()),
        "sample_count": len(rows),
        "tendon_order": list(TENDONS),
        "recommended_encoder_degrees_per_mm": recommended_scales,
        "recommended_encoder_direction": recommended_directions,
        "tendons": tendon_reports,
        "status": "candidate_requires_review",
        "notes": (
            "This artifact does not update excursion_encoder_calibration.json. "
            "Review repeatability, direction spread, load range, and physical sign first."
        ),
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Fit per-tendon encoder degrees/mm from measured samples."
    )
    parser.add_argument("input_csv", type=Path)
    parser.add_argument("output_json", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    report = fit_encoder_calibration(args.input_csv)
    args.output_json.parent.mkdir(parents=True, exist_ok=True)
    args.output_json.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False),
        encoding="utf-8",
    )
    print(args.output_json)


if __name__ == "__main__":
    main()
