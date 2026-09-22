"""Reanalyse a saved servo/DLC step-response trial without moving hardware."""

from __future__ import annotations

import argparse
import csv
import json
from datetime import datetime
from pathlib import Path
from types import SimpleNamespace
from typing import Sequence

from controller.measure_servo_dlc_step_response import analyse_step_response


ANALYSIS_OPTIONS = (
    "servo_onset_threshold_count",
    "joint_onset_threshold_deg",
    "consecutive_detection_count",
    "noise_scale",
    "servo_settling_band_count",
    "servo_speed_threshold",
    "joint_settling_band_deg",
    "joint_settling_slope_deg",
    "settling_consecutive_count",
    "settling_terminal_window",
)


def parse_args(argv: Sequence[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Reanalyse raw telemetry and vision from one saved trial."
    )
    parser.add_argument("trial_directory")
    parser.add_argument("--servo-onset-threshold-count", type=float)
    parser.add_argument("--joint-onset-threshold-deg", type=float)
    parser.add_argument("--consecutive-detection-count", type=int)
    parser.add_argument("--noise-scale", type=float)
    parser.add_argument("--servo-settling-band-count", type=float)
    parser.add_argument("--servo-speed-threshold", type=float)
    parser.add_argument("--joint-settling-band-deg", type=float)
    parser.add_argument("--joint-settling-slope-deg", type=float)
    parser.add_argument("--settling-consecutive-count", type=int)
    parser.add_argument("--settling-terminal-window", type=int)
    parser.add_argument("--output", help="Output JSON path; defaults inside the trial.")
    return parser.parse_args(argv)


def _optional_float(value: str) -> float | None:
    return None if value == "" else float(value)


def _load_telemetry(path: Path) -> list[SimpleNamespace]:
    records: list[SimpleNamespace] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            positions = tuple(float(value) for value in json.loads(row["positions_json"]))
            raw_speeds = row.get("speeds_json", "")
            speeds = (
                None
                if not raw_speeds
                else tuple(float(value) for value in json.loads(raw_speeds))
            )
            records.append(
                SimpleNamespace(
                    received_at=float(row["received_at"]),
                    positions=positions,
                    speeds=speeds,
                )
            )
    return records


def _load_vision(path: Path) -> list[SimpleNamespace]:
    records: list[SimpleNamespace] = []
    with path.open(newline="", encoding="utf-8") as handle:
        for row in csv.DictReader(handle):
            records.append(
                SimpleNamespace(
                    captured_at=float(row["captured_at"]),
                    inference_finished_at=float(row["inference_finished_at"]),
                    flexion_angles_rad=(
                        float(row["DIP_angle_rad"]),
                        float(row["PIP_angle_rad"]),
                        float(row["MCP_angle_rad"]),
                    ),
                    valid=bool(int(row["valid"])),
                    held_last=bool(int(row["held_last"])),
                )
            )
    return records


def reanalyse_trial(
    trial_directory: Path,
    overrides: argparse.Namespace,
) -> dict[str, object]:
    metadata = json.loads((trial_directory / "trial.json").read_text(encoding="utf-8"))
    if metadata.get("commanded_at") is None:
        raise ValueError("trial has no executed command to analyse")
    saved_args = metadata.get("measurement_args")
    if not isinstance(saved_args, dict):
        raise ValueError("trial.json has no measurement_args object")
    resolved = dict(saved_args)
    if resolved.get("movement_direction") is None:
        resolved["movement_direction"] = metadata.get("movement_direction")
    if resolved.get("movement_direction") not in ("increase", "decrease"):
        raise ValueError("trial has no valid movement_direction")
    for name in ANALYSIS_OPTIONS:
        value = getattr(overrides, name, None)
        if value is not None:
            resolved[name] = value
        if resolved.get(name) is None:
            raise ValueError(f"analysis option {name} is unavailable")

    analysis_args = argparse.Namespace(**resolved)
    analysis = analyse_step_response(
        args=analysis_args,
        measurements=_load_vision(trial_directory / "vision.csv"),
        telemetry_records=_load_telemetry(trial_directory / "telemetry.csv"),
        commanded_at=float(metadata["commanded_at"]),
        command_requested_at=(
            None
            if metadata.get("command_requested_at") is None
            else float(metadata["command_requested_at"])
        ),
        return_requested_at=(
            None
            if metadata.get("return_requested_at") is None
            else float(metadata["return_requested_at"])
        ),
        servo_id=int(metadata["servo_id"]),
        start_position=(
            None
            if metadata.get("start_position") is None
            else int(metadata["start_position"])
        ),
        target_position=(
            None
            if metadata.get("target_position") is None
            else int(metadata["target_position"])
        ),
    )
    return {
        "schema_version": 1,
        "source_trial": str(trial_directory),
        "generated_at": datetime.now().isoformat(timespec="seconds"),
        "quantity_note": (
            "telemetry load is a servo-reported load/torque indicator, "
            "not fingertip or grip force"
        ),
        "analysis_args": {name: resolved[name] for name in ANALYSIS_OPTIONS},
        "analysis": analysis,
    }


def main(argv: Sequence[str] | None = None) -> int:
    args = parse_args(argv)
    trial_directory = Path(args.trial_directory).expanduser().resolve()
    result = reanalyse_trial(trial_directory, args)
    output = (
        Path(args.output).expanduser().resolve()
        if args.output
        else trial_directory
        / f"analysis_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
    )
    output.write_text(
        json.dumps(result, ensure_ascii=False, indent=2, allow_nan=False),
        encoding="utf-8",
    )
    print(f"Analysis: {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
