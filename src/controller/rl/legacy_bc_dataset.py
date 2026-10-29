"""Reconstruct causal behavior-cloning pairs from legacy CSV-player logs.

The common six-dimensional observation intentionally describes a feed-forward
plan, not feedback state: ``[q0_plan, q_target]``.  Its action is the absolute
six-tendon excursion target relative to the first row of the source plan.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import re
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from controller.rl.physics_teacher import JOINTS, MomentArmTeacherModel
from controller.servo_mapping import TENDONS


LEGACY_BC_SCHEMA_VERSION = 1
COMMON6_FEATURES = tuple(
    [f"q0_plan_{joint}_rad" for joint in JOINTS]
    + [f"q_target_{joint}_rad" for joint in JOINTS]
)
SENSOR15_FEATURES = tuple(
    [f"q_target_{joint}_rad" for joint in JOINTS]
    + [f"q_observed_{joint}_rad" for joint in JOINTS]
    + [f"{tendon}_force_n" for tendon in TENDONS]
    + ["vision_capture_age_s", "vision_inference_age_s", "servo_age_s"]
)
ACTION_FEATURES = tuple(f"{tendon}_target_excursion_mm" for tendon in TENDONS)


@dataclass(frozen=True)
class LegacyBCConfig:
    decision_hz: float = 5.0
    max_vision_capture_age_s: float = 0.5
    max_vision_inference_age_s: float = 0.5
    max_servo_age_s: float = 0.25
    validation_families: tuple[str, ...] = ("Edinburgh", "FullExtension")


@dataclass
class PairTable:
    observations: np.ndarray
    actions_mm: np.ndarray
    metadata: list[dict[str, Any]]
    feature_names: tuple[str, ...]

    def subset(self, mask: Sequence[bool]) -> "PairTable":
        keep = np.asarray(mask, dtype=bool)
        return PairTable(
            self.observations[keep],
            self.actions_mm[keep],
            [row for row, selected in zip(self.metadata, keep) if selected],
            self.feature_names,
        )


@dataclass
class LegacyBCDataset:
    common6_completed: PairTable
    sensor15_all: PairTable
    sensor15_completed: PairTable
    exclusions: list[dict[str, Any]]
    report: dict[str, Any]

    def save(self, output_dir: str | Path) -> dict[str, Path]:
        root = Path(output_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        paths: dict[str, Path] = {}
        for name, table in (
            ("common6_completed", self.common6_completed),
            ("sensor15_all", self.sensor15_all),
            ("sensor15_completed", self.sensor15_completed),
        ):
            data_path = root / f"bc_{name}.npz"
            np.savez_compressed(
                data_path,
                observations=table.observations,
                actions_mm=table.actions_mm,
                feature_names=np.asarray(table.feature_names),
                action_names=np.asarray(ACTION_FEATURES),
                session_ids=np.asarray([r["session_id"] for r in table.metadata]),
                trajectory_families=np.asarray(
                    [r["trajectory_family"] for r in table.metadata]
                ),
                splits=np.asarray([r["split"] for r in table.metadata]),
                in_model_domain=np.asarray(
                    [r["in_moment_arm_domain"] for r in table.metadata], dtype=bool
                ),
            )
            metadata_path = root / f"bc_{name}_metadata.jsonl"
            _write_jsonl(metadata_path, table.metadata)
            paths[name] = data_path
            paths[f"{name}_metadata"] = metadata_path
        exclusions_path = root / "bc_exclusions.jsonl"
        _write_jsonl(exclusions_path, self.exclusions)
        report_path = root / "bc_dataset_report.json"
        report_path.write_text(
            json.dumps(self.report, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        paths["exclusions"] = exclusions_path
        paths["report"] = report_path
        return paths


class LegacyBCDatasetProcessor:
    def __init__(
        self,
        log_root: str | Path,
        moment_model: MomentArmTeacherModel,
        config: LegacyBCConfig | None = None,
    ) -> None:
        self.log_root = Path(log_root).resolve()
        self.model = moment_model
        self.config = config or LegacyBCConfig()
        if self.config.decision_hz <= 0.0:
            raise ValueError("decision_hz must be positive")

    def process(self) -> LegacyBCDataset:
        common_obs: list[np.ndarray] = []
        common_actions: list[np.ndarray] = []
        common_meta: list[dict[str, Any]] = []
        sensor_obs: list[np.ndarray] = []
        sensor_actions: list[np.ndarray] = []
        sensor_meta: list[dict[str, Any]] = []
        exclusions: list[dict[str, Any]] = []

        manifests = sorted(self.log_root.rglob("session_manifest.json"))
        for manifest_path in manifests:
            try:
                payload = json.loads(manifest_path.read_text(encoding="utf-8"))
                self._process_session(
                    manifest_path,
                    payload,
                    common_obs,
                    common_actions,
                    common_meta,
                    sensor_obs,
                    sensor_actions,
                    sensor_meta,
                    exclusions,
                )
            except Exception as exc:  # keep a reproducible audit of corrupt sessions
                exclusions.append(
                    {
                        "stage": "session",
                        "session_path": str(manifest_path.parent),
                        "reason": "session_parse_error",
                        "detail": f"{type(exc).__name__}: {exc}",
                    }
                )

        common = _make_table(common_obs, common_actions, common_meta, COMMON6_FEATURES)
        sensor_all = _make_table(sensor_obs, sensor_actions, sensor_meta, SENSOR15_FEATURES)
        completed_mask = [row["termination_reason"] == "completed" for row in sensor_meta]
        sensor_completed = sensor_all.subset(completed_mask)
        report = self._build_report(common, sensor_all, sensor_completed, exclusions)
        return LegacyBCDataset(common, sensor_all, sensor_completed, exclusions, report)

    def _process_session(
        self,
        manifest_path: Path,
        manifest: Mapping[str, Any],
        common_obs: list[np.ndarray],
        common_actions: list[np.ndarray],
        common_meta: list[dict[str, Any]],
        sensor_obs: list[np.ndarray],
        sensor_actions: list[np.ndarray],
        sensor_meta: list[dict[str, Any]],
        exclusions: list[dict[str, Any]],
    ) -> None:
        session_dir = manifest_path.parent
        session_id = str(manifest.get("run_id") or session_dir.name)
        family = _trajectory_family(session_dir)
        control_path = session_dir / "control_trace.csv"
        angle_path = session_dir / "angle_trace.csv"
        input_path = _resolve_input_csv(manifest, self.log_root)
        missing = [
            name
            for name, path in (("control_trace", control_path), ("input_csv", input_path))
            if path is None or not path.is_file()
        ]
        if missing:
            exclusions.append(
                {
                    "stage": "session",
                    "session_id": session_id,
                    "trajectory_family": family,
                    "reason": "missing_required_file",
                    "detail": missing,
                }
            )
            return
        control = _read_csv(control_path)
        planned = _read_csv(input_path)
        if not control or not planned:
            exclusions.append(
                {
                    "stage": "session",
                    "session_id": session_id,
                    "trajectory_family": family,
                    "reason": "empty_required_file",
                }
            )
            return

        source_times = np.asarray([_number(r, "elapsed_s") for r in planned])
        control_times = np.asarray([_number(r, "elapsed_s") for r in control])
        if not _strictly_nondecreasing(source_times) or not _strictly_nondecreasing(control_times):
            raise ValueError("elapsed_s is not monotonic")
        q_plan = np.column_stack(
            [np.asarray([_number(r, f"{j}_smoothed_angle_rad") for r in planned]) for j in JOINTS]
        )
        action_plan = np.column_stack(
            [np.asarray([_number(r, f"{t}_predicted_excursion_mm") for r in planned]) for t in TENDONS]
        )
        q0 = q_plan[0].copy()
        action_origin = action_plan[0].copy()
        termination = str(manifest.get("termination_reason") or "unknown")
        completed = bool(manifest.get("completed")) and termination == "completed"
        angle_rows = _unique_angle_rows(_read_csv(angle_path)) if angle_path.is_file() else []
        interval = 1.0 / self.config.decision_hz
        # Never extend a legacy FF plan by np.interp's endpoint hold.  Some
        # failed sessions kept logging during return-to-origin after the source
        # trajectory ended; those rows do not contain recorded teacher actions.
        decision_end = min(control_times[-1], source_times[-1])
        count = int(math.floor((decision_end - source_times[0] + 1e-12) / interval)) + 1

        for decision_index in range(max(0, count)):
            source_t = float(source_times[0] + decision_index * interval)
            control_index = int(np.searchsorted(control_times, source_t, side="left"))
            if control_index >= len(control):
                continue
            row = control[control_index]
            q_target = np.asarray(
                [np.interp(source_t, source_times, q_plan[:, i]) for i in range(3)]
            )
            action = np.asarray(
                [np.interp(source_t, source_times, action_plan[:, i]) for i in range(6)]
            ) - action_origin
            decision_time = _number(row, "control_tick_monotonic_s")
            in_domain = _postures_in_domain(self.model, q0, q_target)
            base_meta = {
                "pair_id": f"{session_id}:{decision_index:05d}",
                "session_id": session_id,
                "trajectory_family": family,
                "decision_index": decision_index,
                "decision_hz": self.config.decision_hz,
                "source_elapsed_s": source_t,
                "control_elapsed_s": _number(row, "elapsed_s"),
                "decision_monotonic_s": decision_time,
                "action_source": "linear interpolation of source FF predicted_excursion_mm minus its first row",
                "action_reference": "session_plan_origin",
                "action_is_incremental": False,
                "q0_semantics": "q0_plan; not q0_measured",
                "termination_reason": termination,
                "in_moment_arm_domain": in_domain,
                "split": (
                    "validation"
                    if family in self.config.validation_families
                    else "training"
                ),
                "input_csv_path": str(input_path),
                "control_trace_path": str(control_path),
            }
            if completed:
                common_obs.append(np.concatenate([q0, q_target]))
                common_actions.append(action)
                common_meta.append(dict(base_meta, observation_kind="common6_plan"))

            sensor_result = self._causal_sensor_observation(
                row, angle_rows, decision_time, q_target
            )
            if isinstance(sensor_result, str):
                exclusions.append(
                    {
                        **base_meta,
                        "stage": "sensor15_pair",
                        "reason": sensor_result,
                    }
                )
            else:
                observation, sensor_extra = sensor_result
                sensor_obs.append(observation)
                sensor_actions.append(action)
                sensor_meta.append(
                    dict(base_meta, observation_kind="sensor15_causal", **sensor_extra)
                )

    def _causal_sensor_observation(
        self,
        control_row: Mapping[str, str],
        angle_rows: Sequence[Mapping[str, str]],
        decision_time: float,
        q_target: np.ndarray,
    ) -> tuple[np.ndarray, dict[str, Any]] | str:
        try:
            servo_received = _number(control_row, "servo_feedback_received_monotonic_s")
            servo_age = _number(control_row, "servo_telemetry_age_s")
            encoder_received = [
                _number(control_row, f"{t}_encoder_received_monotonic_s") for t in TENDONS
            ]
            forces = np.asarray([_number(control_row, f"{t}_force_n") for t in TENDONS])
        except (KeyError, ValueError):
            return "missing_or_invalid_control_telemetry"
        if servo_received > decision_time + 1e-9 or any(t > decision_time + 1e-9 for t in encoder_received):
            return "future_control_telemetry"
        if servo_age < 0.0 or servo_age > self.config.max_servo_age_s:
            return "stale_servo_telemetry"
        available = []
        for row in angle_rows:
            try:
                capture = _number(row, "captured_at_monotonic")
                inference = _number(row, "inference_finished_at_monotonic")
                valid = int(float(row.get("valid", "0"))) == 1
                held = int(float(row.get("held_last", "0"))) == 1
            except (KeyError, TypeError, ValueError):
                continue
            if valid and not held and capture <= decision_time and inference <= decision_time:
                available.append((inference, capture, row))
        if not available:
            return "no_causal_valid_vision"
        inference, capture, vision = max(available, key=lambda item: (item[0], item[1]))
        capture_age = decision_time - capture
        inference_age = decision_time - inference
        if capture_age > self.config.max_vision_capture_age_s:
            return "stale_vision_capture"
        if inference_age > self.config.max_vision_inference_age_s:
            return "stale_vision_inference"
        observed = np.deg2rad([_number(vision, f"{joint}_deg") for joint in JOINTS])
        values = np.concatenate(
            [q_target, observed, forces, [capture_age, inference_age, servo_age]]
        )
        if not np.isfinite(values).all():
            return "nonfinite_sensor_observation"
        return values, {
            "vision_frame_idx": int(float(vision["frame_idx"])),
            "vision_capture_monotonic_s": capture,
            "vision_inference_finished_monotonic_s": inference,
            "vision_capture_age_s": capture_age,
            "vision_inference_age_s": inference_age,
            "servo_feedback_received_monotonic_s": servo_received,
            "servo_age_s": servo_age,
        }

    def _build_report(
        self,
        common: PairTable,
        sensor_all: PairTable,
        sensor_completed: PairTable,
        exclusions: Sequence[Mapping[str, Any]],
    ) -> dict[str, Any]:
        train = common.subset([r["split"] == "training" for r in common.metadata])
        validation = common.subset([r["split"] == "validation" for r in common.metadata])
        domain_mask = np.asarray(
            [row["in_moment_arm_domain"] for row in common.metadata], dtype=bool
        )
        physics_prediction = np.asarray(
            [
                self.model.action_from_session_origin(obs[:3], obs[3:])
                for obs, keep in zip(common.observations, domain_mask)
                if keep
            ]
        )
        physics_error = physics_prediction - common.actions_mm[domain_mask]
        physics_rmse = np.sqrt(np.mean(physics_error**2, axis=0))
        return {
            "schema_version": LEGACY_BC_SCHEMA_VERSION,
            "purpose": "offline FF-command behavior cloning; not a measured-state feedback policy",
            "common6_observation": list(COMMON6_FEATURES),
            "sensor15_observation": list(SENSOR15_FEATURES),
            "action": list(ACTION_FEATURES),
            "q0_plan_is_q0_measured": False,
            "action_reference": "session plan origin",
            "counts": {
                "common6_completed": len(common.metadata),
                "sensor15_all": len(sensor_all.metadata),
                "sensor15_completed": len(sensor_completed.metadata),
                "training_common6": len(train.metadata),
                "validation_common6": len(validation.metadata),
                "excluded_sensor15_attempts": sum(
                    row.get("stage") == "sensor15_pair" for row in exclusions
                ),
            },
            "split": {
                "unit": "session and trajectory family",
                "validation_families": list(self.config.validation_families),
                "training": _table_statistics(train),
                "validation": _table_statistics(validation),
            },
            "sensor15_all": _table_statistics(sensor_all),
            "sensor15_completed": _table_statistics(sensor_completed),
            "recorded_ff_vs_moment_arm_recalculation": {
                "comparison_pairs": int(domain_mask.sum()),
                "excluded_out_of_domain_pairs": int((~domain_mask).sum()),
                "per_tendon_rmse_mm": {
                    tendon: float(physics_rmse[index])
                    for index, tendon in enumerate(TENDONS)
                },
                "rmse_range_mm": [
                    float(np.min(physics_rmse)), float(np.max(physics_rmse))
                ],
                "interpretation": (
                    "Recorded C3 FF labels already match the same moment-arm "
                    "model within numerical/interpolation error; C3 is not "
                    "physics-free."
                ),
            },
            "exclusion_reasons": _counts(r.get("reason", "unknown") for r in exclusions),
            "moment_arm_model": self.model.provenance(),
        }


def load_pair_table(path: str | Path, metadata_path: str | Path) -> PairTable:
    with np.load(path, allow_pickle=False) as data:
        observations = data["observations"]
        actions = data["actions_mm"]
        names = tuple(str(v) for v in data["feature_names"])
    metadata = [json.loads(line) for line in Path(metadata_path).read_text(encoding="utf-8").splitlines() if line]
    if len(metadata) != len(observations):
        raise ValueError("Pair data and metadata row counts differ")
    return PairTable(observations, actions, metadata, names)


def _resolve_input_csv(manifest: Mapping[str, Any], log_root: Path) -> Path | None:
    raw = manifest.get("input_csv_path") or manifest.get("input_csv")
    if not raw:
        return None
    path = Path(str(raw))
    if path.is_file():
        return path.resolve()
    # Logs may have moved with the repository; preserve the suffix after logs/.
    parts = list(path.parts)
    lowered = [part.lower() for part in parts]
    if "logs" in lowered:
        suffix = parts[lowered.index("logs") + 1 :]
        candidate = log_root.parent.joinpath(*suffix)
        if candidate.is_file():
            return candidate.resolve()
    matches = list(log_root.parent.rglob(path.name))
    return matches[0].resolve() if len(matches) == 1 else path


def _trajectory_family(session_dir: Path) -> str:
    name = re.sub(r"\s*\(\d+\)$", "", session_dir.name)
    aliases = {
        "MCPflesion": "MCPFlexion",
        "MCPflexion": "MCPFlexion",
        "middleFlexion": "MIddleFlexion",
        "shallowFlexion": "ShallowFlexion",
    }
    if re.fullmatch(r"\d{8}_\d{6}", name) and session_dir.parent.name == "debugging":
        return "MCPFlexion"
    return aliases.get(name, name)


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _unique_angle_rows(rows: Iterable[Mapping[str, str]]) -> list[Mapping[str, str]]:
    unique: dict[tuple[str, str, str], Mapping[str, str]] = {}
    for row in rows:
        key = (
            str(row.get("frame_idx", "")),
            str(row.get("captured_at_monotonic", "")),
            str(row.get("inference_finished_at_monotonic", "")),
        )
        unique.setdefault(key, row)
    return list(unique.values())


def _number(row: Mapping[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"{key} is non-finite")
    return value


def _strictly_nondecreasing(values: np.ndarray) -> bool:
    return bool(np.isfinite(values).all() and np.all(np.diff(values) >= 0.0))


def _postures_in_domain(
    model: MomentArmTeacherModel, q0: np.ndarray, target: np.ndarray
) -> bool:
    for index, joint in enumerate(JOINTS):
        low, high = model.valid_ranges_rad[joint]
        if not (low - 1e-12 <= q0[index] <= high + 1e-12):
            return False
        if not (low - 1e-12 <= target[index] <= high + 1e-12):
            return False
    return True


def _make_table(
    observations: Sequence[np.ndarray],
    actions: Sequence[np.ndarray],
    metadata: list[dict[str, Any]],
    names: tuple[str, ...],
) -> PairTable:
    return PairTable(
        np.asarray(observations, dtype=np.float64).reshape(-1, len(names)),
        np.asarray(actions, dtype=np.float64).reshape(-1, len(TENDONS)),
        metadata,
        names,
    )


def _counts(values: Iterable[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return dict(sorted(result.items()))


def _range_stats(values: np.ndarray, names: Sequence[str]) -> dict[str, Any]:
    if len(values) == 0:
        return {}
    return {
        name: {
            "min": float(np.min(values[:, index])),
            "max": float(np.max(values[:, index])),
            "mean": float(np.mean(values[:, index])),
            "std": float(np.std(values[:, index])),
        }
        for index, name in enumerate(names)
    }


def _table_statistics(table: PairTable) -> dict[str, Any]:
    domain = np.asarray([r["in_moment_arm_domain"] for r in table.metadata], dtype=bool)
    sessions = _counts(r["session_id"] for r in table.metadata)
    return {
        "pairs": len(table.metadata),
        "sessions": len(sessions),
        "pairs_by_family": _counts(r["trajectory_family"] for r in table.metadata),
        "pairs_by_session": sessions,
        "moment_arm_domain": {
            "inside": int(domain.sum()),
            "outside": int((~domain).sum()),
        },
        "observation_distribution": _range_stats(table.observations, table.feature_names),
        "action_distribution_mm": _range_stats(table.actions_mm, ACTION_FEATURES),
    }


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    with path.open("w", encoding="utf-8", newline="\n") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), ensure_ascii=False, allow_nan=False) + "\n")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Build causal BC pairs from legacy logs")
    parser.add_argument("log_root", type=Path)
    parser.add_argument("moment_arm_directory", type=Path)
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("--decision-hz", type=float, default=5.0)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    model = MomentArmTeacherModel.from_directory(args.moment_arm_directory)
    dataset = LegacyBCDatasetProcessor(
        args.log_root, model, LegacyBCConfig(decision_hz=args.decision_hz)
    ).process()
    dataset.save(args.output_dir)
    print(json.dumps(dataset.report["counts"], indent=2))


if __name__ == "__main__":
    main()
