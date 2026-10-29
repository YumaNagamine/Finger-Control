"""Causally reconstruct provisional AWAC transitions from legacy FF logs.

Legacy playback changed the feed-forward target at every control tick.  This
module therefore never converts those logs into a fictitious 5 Hz ZOH action.
Each retained transition covers exactly one adjacent pair of recorded control
rows.  The physical servo application timestamp was not logged and remains
unknown.
"""

from __future__ import annotations

import argparse
import csv
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from controller.rl.legacy_bc_dataset import (
    _resolve_input_csv,
    _trajectory_family,
    _unique_angle_rows,
)
from controller.rl.physics_teacher import JOINTS
from controller.servo_mapping import TENDONS


SCHEMA_VERSION = 1
OBSERVATION_FEATURES = tuple(
    [f"q0_plan_{joint}_rad" for joint in JOINTS]
    + [f"q_target_{joint}_rad" for joint in JOINTS]
    + [f"q_observed_{joint}_rad" for joint in JOINTS]
    + [f"{tendon}_measured_excursion_mm" for tendon in TENDONS]
    + [f"{tendon}_force_n" for tendon in TENDONS]
    + [f"{tendon}_previous_target_excursion_mm" for tendon in TENDONS]
    + ["vision_capture_age_s", "vision_inference_age_s", "servo_age_s"]
)
ACTION_FEATURES = tuple(f"{tendon}_target_excursion_mm" for tendon in TENDONS)


@dataclass(frozen=True)
class LegacyAWACConfig:
    require_completed_sessions: bool = True
    max_vision_capture_age_s: float = 0.5
    max_vision_inference_age_s: float = 0.5
    max_servo_age_s: float = 0.25
    max_control_interval_s: float = 0.15
    action_fidelity_tolerance_mm: float = 0.01
    encoder_degrees_per_mm: float = 2.8648


@dataclass
class LegacyAWACDataset:
    observations: np.ndarray
    actions_mm: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    metadata: list[dict[str, Any]]
    exclusions: list[dict[str, Any]]
    report: dict[str, Any]

    def save(self, output_dir: str | Path) -> dict[str, Path]:
        root = Path(output_dir).resolve()
        root.mkdir(parents=True, exist_ok=True)
        data_path = root / "legacy_awac_transitions.npz"
        np.savez_compressed(
            data_path,
            observations=self.observations,
            actions_mm=self.actions_mm,
            rewards=self.rewards,
            next_observations=self.next_observations,
            terminals=np.zeros(len(self.rewards), dtype=np.float32),
            timeouts=np.ones(len(self.rewards), dtype=np.float32),
            observation_features=np.asarray(OBSERVATION_FEATURES),
            action_features=np.asarray(ACTION_FEATURES),
        )
        metadata_path = root / "legacy_awac_transition_metadata.jsonl"
        _write_jsonl(metadata_path, self.metadata)
        exclusions_path = root / "legacy_awac_exclusions.jsonl"
        _write_jsonl(exclusions_path, self.exclusions)
        report_path = root / "legacy_awac_dataset_report.json"
        report_path.write_text(
            json.dumps(self.report, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        return {
            "data": data_path,
            "metadata": metadata_path,
            "exclusions": exclusions_path,
            "report": report_path,
        }


class LegacyAWACDatasetProcessor:
    def __init__(self, log_root: str | Path, config: LegacyAWACConfig | None = None):
        self.log_root = Path(log_root).resolve()
        self.config = config or LegacyAWACConfig()

    def process(self) -> LegacyAWACDataset:
        observations: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        rewards: list[float] = []
        next_observations: list[np.ndarray] = []
        metadata: list[dict[str, Any]] = []
        exclusions: list[dict[str, Any]] = []
        session_audits: list[dict[str, Any]] = []
        for manifest_path in sorted(self.log_root.rglob("session_manifest.json")):
            try:
                manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
                audit = self._process_session(
                    manifest_path,
                    manifest,
                    observations,
                    actions,
                    rewards,
                    next_observations,
                    metadata,
                    exclusions,
                )
                session_audits.append(audit)
            except Exception as exc:
                exclusions.append(
                    {
                        "stage": "session",
                        "session_path": str(manifest_path.parent),
                        "reason": "session_parse_error",
                        "detail": f"{type(exc).__name__}: {exc}",
                    }
                )
        obs = np.asarray(observations, dtype=np.float32).reshape(-1, len(OBSERVATION_FEATURES))
        act = np.asarray(actions, dtype=np.float32).reshape(-1, len(TENDONS))
        rew = np.asarray(rewards, dtype=np.float32).reshape(-1, 1)
        next_obs = np.asarray(next_observations, dtype=np.float32).reshape(
            -1, len(OBSERVATION_FEATURES)
        )
        report = self._build_report(obs, act, rew, next_obs, metadata, exclusions, session_audits)
        return LegacyAWACDataset(obs, act, rew, next_obs, metadata, exclusions, report)

    def _process_session(
        self,
        manifest_path: Path,
        manifest: Mapping[str, Any],
        observations: list[np.ndarray],
        actions: list[np.ndarray],
        rewards: list[float],
        next_observations: list[np.ndarray],
        metadata: list[dict[str, Any]],
        exclusions: list[dict[str, Any]],
    ) -> dict[str, Any]:
        session_dir = manifest_path.parent
        session_id = str(manifest.get("run_id") or session_dir.name)
        family = _trajectory_family(session_dir)
        completed = bool(manifest.get("completed")) and manifest.get("termination_reason") == "completed"
        paths = {
            "control": session_dir / "control_trace.csv",
            "angle": session_dir / "angle_trace.csv",
            "source": _resolve_input_csv(manifest, self.log_root),
        }
        missing = [name for name, path in paths.items() if path is None or not path.is_file()]
        base = {"session_id": session_id, "trajectory_family": family}
        if missing:
            exclusions.append({**base, "stage": "session", "reason": "missing_required_file", "detail": missing})
            return {**base, "eligible": False, "reason": "missing_required_file"}
        if self.config.require_completed_sessions and not completed:
            exclusions.append({**base, "stage": "session", "reason": "noncompleted_session"})
            return {**base, "eligible": False, "reason": "noncompleted_session"}

        control = _read_csv(paths["control"])
        source = _read_csv(paths["source"])
        angle = _unique_angle_rows(_read_csv(paths["angle"]))
        if len(control) < 2 or len(source) < 2:
            exclusions.append({**base, "stage": "session", "reason": "insufficient_rows"})
            return {**base, "eligible": False, "reason": "insufficient_rows"}

        source_t = _column(source, "elapsed_s")
        control_t = _column(control, "elapsed_s")
        tick_t = _column(control, "control_tick_monotonic_s")
        submit_t = _column(control, "command_submitted_monotonic_s")
        if np.any(np.diff(source_t) < 0) or np.any(np.diff(control_t) < 0) or np.any(np.diff(tick_t) <= 0):
            raise ValueError("non-monotonic source/control timestamps")
        q_plan = np.column_stack([_column(source, f"{joint}_smoothed_angle_rad") for joint in JOINTS])
        source_action = np.column_stack(
            [_column(source, f"{tendon}_predicted_excursion_mm") for tendon in TENDONS]
        )
        source_action -= source_action[0]
        q0_plan = q_plan[0]
        recorded_action = np.column_stack(
            [_column(control, f"{tendon}_target_excursion_mm") for tendon in TENDONS]
        )
        source_at_control = _interpolate_matrix(control_t, source_t, source_action)
        fidelity_error = recorded_action - source_at_control
        playback = np.asarray([row.get("phase", "") == "playback" for row in control])
        in_source = (control_t >= source_t[0] - 1e-9) & (control_t <= source_t[-1] + 1e-9)
        audit_mask = playback & in_source

        actual_encoder = np.column_stack(
            [_column(control, f"{tendon}_actual_encoder_deg") for tendon in TENDONS]
        )
        measured_excursion = (actual_encoder - actual_encoder[0]) / self.config.encoder_degrees_per_mm
        force = np.column_stack([_column(control, f"{tendon}_force_n") for tendon in TENDONS])
        q_target_control = _interpolate_matrix(control_t, source_t, q_plan)
        vision = _valid_vision(angle)
        accepted_before = len(metadata)

        for index in range(1, len(control) - 1):
            interval_base = {**base, "control_row_index": index}
            if not playback[index] or not playback[index + 1] or not in_source[index] or not in_source[index + 1]:
                exclusions.append({**interval_base, "stage": "transition", "reason": "outside_playback_source_support"})
                continue
            duration = submit_t[index + 1] - submit_t[index]
            if not (0.0 < duration <= self.config.max_control_interval_s):
                exclusions.append({**interval_base, "stage": "transition", "reason": "invalid_control_interval"})
                continue
            if np.max(np.abs(fidelity_error[index])) > self.config.action_fidelity_tolerance_mm:
                exclusions.append({**interval_base, "stage": "transition", "reason": "recorded_source_action_mismatch"})
                continue
            current = self._observation(
                index, control, tick_t, q0_plan, q_target_control, measured_excursion,
                force, recorded_action, vision,
            )
            following = self._observation(
                index + 1, control, tick_t, q0_plan, q_target_control, measured_excursion,
                force, recorded_action, vision,
            )
            if isinstance(current, str) or isinstance(following, str):
                reason = current if isinstance(current, str) else f"next_{following}"
                exclusions.append({**interval_base, "stage": "transition", "reason": reason})
                continue
            obs, obs_meta = current
            next_obs, next_meta = following
            frames = [
                row for row in vision
                if submit_t[index] <= row["capture"] < submit_t[index + 1]
            ]
            if not frames:
                exclusions.append({**interval_base, "stage": "transition", "reason": "no_reward_capture_in_action_period"})
                continue
            reward_frame = max(frames, key=lambda row: row["capture"])
            capture_source_t = float(np.interp(reward_frame["capture"], tick_t, control_t))
            if not (source_t[0] <= capture_source_t <= source_t[-1]):
                exclusions.append({**interval_base, "stage": "transition", "reason": "reward_capture_outside_source_support"})
                continue
            reward_target = np.asarray(
                [np.interp(capture_source_t, source_t, q_plan[:, j]) for j in range(3)]
            )
            reward_observed = np.deg2rad(reward_frame["angles_deg"])
            error = reward_observed - reward_target
            reward = -float(np.mean(error * error))
            if not all(np.isfinite(value).all() if isinstance(value, np.ndarray) else math.isfinite(value)
                       for value in (obs, next_obs, recorded_action[index], reward)):
                exclusions.append({**interval_base, "stage": "transition", "reason": "nonfinite_transition"})
                continue
            observations.append(obs)
            actions.append(recorded_action[index].copy())
            rewards.append(reward)
            next_observations.append(next_obs)
            metadata.append(
                {
                    **interval_base,
                    "transition_id": f"{session_id}:{index:05d}",
                    "confidence": "legacy_approximate",
                    "action_semantics": "recorded absolute target excursion at one native legacy control tick",
                    "action_reference": "session encoder origin",
                    "action_is_5hz_zoh": False,
                    "action_reflected_in_pid_monotonic_s": float(tick_t[index]),
                    "command_transport_submitted_monotonic_s": float(submit_t[index]),
                    "next_command_transport_submitted_monotonic_s": float(submit_t[index + 1]),
                    "physical_servo_application_monotonic_s": None,
                    "action_period_s": float(duration),
                    "source_elapsed_s": float(control_t[index]),
                    "source_interpolated_action_mm": source_at_control[index].tolist(),
                    "recorded_minus_source_action_mm": fidelity_error[index].tolist(),
                    "observation": obs_meta,
                    "next_observation": next_meta,
                    "reward": {
                        "definition": "negative mean squared joint-angle tracking error in rad^2",
                        "capture_monotonic_s": reward_frame["capture"],
                        "inference_finished_monotonic_s": reward_frame["inference"],
                        "inference_available_by_next_decision": bool(reward_frame["inference"] <= tick_t[index + 1]),
                        "capture_mapped_source_elapsed_s": capture_source_t,
                        "target_angle_rad": reward_target.tolist(),
                        "observed_angle_rad": reward_observed.tolist(),
                        "measurement_delay_s": float(reward_frame["inference"] - reward_frame["capture"]),
                    },
                    "episode_boundary": {
                        "terminated": False,
                        "truncated": True,
                        "reason": "legacy_sparse_reward_one_step_fragment",
                    },
                }
            )

        errors = fidelity_error[audit_mask]
        return {
            **base,
            "eligible": True,
            "completed": completed,
            "control_rows": len(control),
            "unique_valid_vision_frames": len(vision),
            "accepted_transitions": len(metadata) - accepted_before,
            "control_interval_s": _stats(np.diff(submit_t[audit_mask])) if np.sum(audit_mask) > 1 else {},
            "recorded_vs_source_action": _error_stats(errors),
            "transport_submit_minus_pid_tick_s": _stats((submit_t - tick_t)[audit_mask]),
        }

    def _observation(
        self,
        index: int,
        control: Sequence[Mapping[str, str]],
        tick_t: np.ndarray,
        q0_plan: np.ndarray,
        q_target: np.ndarray,
        measured_excursion: np.ndarray,
        force: np.ndarray,
        recorded_action: np.ndarray,
        vision: Sequence[Mapping[str, Any]],
    ) -> tuple[np.ndarray, dict[str, Any]] | str:
        row = control[index]
        decision = tick_t[index]
        try:
            servo_received = _number(row, "servo_feedback_received_monotonic_s")
            servo_age = _number(row, "servo_telemetry_age_s")
            encoder_received = [_number(row, f"{t}_encoder_received_monotonic_s") for t in TENDONS]
        except (KeyError, ValueError):
            return "missing_control_telemetry"
        if servo_received > decision + 1e-9 or any(value > decision + 1e-9 for value in encoder_received):
            return "future_control_telemetry"
        if not (0.0 <= servo_age <= self.config.max_servo_age_s):
            return "stale_servo_telemetry"
        if any(row.get(f"{t}_force_status") != "OK" for t in TENDONS):
            return "unsafe_or_invalid_force_status"
        causal = [frame for frame in vision if frame["capture"] <= decision and frame["inference"] <= decision]
        if not causal:
            return "no_causal_valid_vision"
        frame = max(causal, key=lambda item: (item["inference"], item["capture"]))
        capture_age = decision - frame["capture"]
        inference_age = decision - frame["inference"]
        if capture_age > self.config.max_vision_capture_age_s:
            return "stale_vision_capture"
        if inference_age > self.config.max_vision_inference_age_s:
            return "stale_vision_inference"
        observed = np.deg2rad(frame["angles_deg"])
        values = np.concatenate(
            [
                q0_plan,
                q_target[index],
                observed,
                measured_excursion[index],
                force[index],
                recorded_action[index - 1],
                [capture_age, inference_age, servo_age],
            ]
        ).astype(np.float32)
        return values, {
            "decision_monotonic_s": float(decision),
            "vision_capture_monotonic_s": frame["capture"],
            "vision_inference_finished_monotonic_s": frame["inference"],
            "vision_capture_age_s": float(capture_age),
            "vision_inference_age_s": float(inference_age),
            "servo_feedback_received_monotonic_s": float(servo_received),
            "servo_age_s": float(servo_age),
        }

    def _build_report(self, obs, actions, rewards, next_obs, metadata, exclusions, audits):
        eligible = [row for row in audits if row.get("eligible")]
        action_errors = np.asarray(
            [row["recorded_minus_source_action_mm"] for row in metadata], dtype=np.float64
        ).reshape(-1, 6)
        # This is a measured one-tick response gap, not an observed physical
        # command.  The actual servo application instant was not recorded.
        one_tick_response_error = next_obs[:, 9:15] - actions
        action_periods = np.asarray([row["action_period_s"] for row in metadata])
        inference_delays = np.asarray(
            [row["reward"]["measurement_delay_s"] for row in metadata]
        )
        reward_available = np.asarray(
            [row["reward"]["inference_available_by_next_decision"] for row in metadata],
            dtype=bool,
        )
        return {
            "schema_version": SCHEMA_VERSION,
            "purpose": "provisional offline AWAC pipeline test; not a performance comparison",
            "observation_features": list(OBSERVATION_FEATURES),
            "action_features": list(ACTION_FEATURES),
            "counts": {
                "manifests": len(audits),
                "eligible_completed_sessions": len(eligible),
                "accepted_transitions": len(metadata),
                "excluded_records": len(exclusions),
            },
            "reliability": {
                "classification": "legacy_approximate" if len(metadata) else "unusable",
                "logged_command_target": "recorded directly",
                "pid_reflection_time": "recorded control_tick_monotonic_s",
                "transport_submit_time": "recorded command_submitted_monotonic_s",
                "physical_servo_application_time": "unknown and not imputed",
                "reward_capture_target_alignment": "capture time mapped through recorded control tick to source elapsed time",
                "reward_inference_completion_used_in_observation": "only when <= decision timestamp",
                "five_hz_zoh_assumed": False,
                "limitation": "one-step fragments only; intervals without a captured angle are not fabricated",
            },
            "accepted_action_fidelity": _error_stats(action_errors),
            "recorded_command_vs_next_tick_measured_excursion": {
                **_error_stats(one_tick_response_error),
                "interpretation": (
                    "Measured encoder excursion at the next control tick minus the "
                    "recorded target. This quantifies one-tick tracking response, not "
                    "the unknown physical application command or instant."
                ),
            },
            "reward": _stats(rewards.reshape(-1)) if len(rewards) else {},
            "timing": {
                "native_action_period_s": _stats(action_periods),
                "reward_capture_to_inference_delay_s": _stats(inference_delays),
                "reward_inference_available_by_next_decision": {
                    "count": int(np.sum(reward_available)),
                    "fraction": float(np.mean(reward_available)) if len(reward_available) else 0.0,
                },
            },
            "trajectory_family_counts": _counts(
                row["trajectory_family"] for row in metadata
            ),
            "observation_shape": list(obs.shape),
            "action_shape": list(actions.shape),
            "exclusion_reasons": _counts(row.get("reason", "unknown") for row in exclusions),
            "session_audits": audits,
        }


def load_legacy_awac_dataset(path: str | Path) -> dict[str, np.ndarray]:
    with np.load(path, allow_pickle=False) as data:
        return {name: data[name] for name in data.files}


def _read_csv(path: Path) -> list[dict[str, str]]:
    with path.open("r", encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def _number(row: Mapping[str, str], key: str) -> float:
    value = float(row[key])
    if not math.isfinite(value):
        raise ValueError(f"{key} is non-finite")
    return value


def _column(rows: Sequence[Mapping[str, str]], key: str) -> np.ndarray:
    return np.asarray([_number(row, key) for row in rows], dtype=np.float64)


def _interpolate_matrix(query: np.ndarray, times: np.ndarray, values: np.ndarray) -> np.ndarray:
    return np.column_stack([np.interp(query, times, values[:, index]) for index in range(values.shape[1])])


def _valid_vision(rows: Sequence[Mapping[str, str]]) -> list[dict[str, Any]]:
    result = []
    for row in rows:
        try:
            if int(float(row.get("valid", "0"))) != 1 or int(float(row.get("held_last", "0"))) == 1:
                continue
            values = {
                "frame_idx": int(float(row["frame_idx"])),
                "capture": _number(row, "captured_at_monotonic"),
                "inference": _number(row, "inference_finished_at_monotonic"),
                "angles_deg": np.asarray([_number(row, f"{joint}_deg") for joint in JOINTS]),
            }
            if values["inference"] < values["capture"]:
                continue
            result.append(values)
        except (KeyError, TypeError, ValueError):
            continue
    return sorted(result, key=lambda item: (item["capture"], item["inference"]))


def _stats(values: np.ndarray) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    if array.size == 0:
        return {}
    return {
        "count": int(array.size),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "p50": float(np.quantile(array, 0.5)),
        "p95": float(np.quantile(array, 0.95)),
    }


def _error_stats(error: np.ndarray) -> dict[str, Any]:
    array = np.asarray(error, dtype=np.float64).reshape(-1, len(TENDONS))
    if len(array) == 0:
        return {}
    return {
        "pair_count": len(array),
        "overall_mae_mm": float(np.mean(np.abs(array))),
        "overall_rmse_mm": float(np.sqrt(np.mean(array * array))),
        "max_absolute_mm": float(np.max(np.abs(array))),
        "per_tendon_rmse_mm": {
            tendon: float(np.sqrt(np.mean(array[:, index] ** 2)))
            for index, tendon in enumerate(TENDONS)
        },
    }


def _counts(values: Iterable[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[value] = result.get(value, 0) + 1
    return dict(sorted(result.items()))


def _write_jsonl(path: Path, rows: Sequence[Mapping[str, Any]]) -> None:
    path.write_text(
        "".join(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n" for row in rows),
        encoding="utf-8",
    )


def main() -> None:
    parser = argparse.ArgumentParser(description="Build provisional legacy AWAC transitions")
    parser.add_argument("log_root", type=Path)
    parser.add_argument("output_dir", type=Path)
    args = parser.parse_args()
    dataset = LegacyAWACDatasetProcessor(args.log_root).process()
    paths = dataset.save(args.output_dir)
    print(json.dumps({"counts": dataset.report["counts"], "paths": {k: str(v) for k, v in paths.items()}}, indent=2))


if __name__ == "__main__":
    main()
