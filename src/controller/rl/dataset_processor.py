"""Build causally aligned AWAC transitions from raw CSV-player sessions."""

from __future__ import annotations

import argparse
import bisect
import csv
import json
import math
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence

import numpy as np

from controller.csv_player.trajectory import ExcursionTrajectory
from controller.servo_mapping import TENDONS


JOINTS = ("DIP", "PIP", "MCP")
DATASET_SCHEMA_VERSION = 3
OBSERVATION_FEATURES = (
    *(f"current_{joint}_rad" for joint in JOINTS),
    *(f"goal_error_{joint}_rad" for joint in JOINTS),
    *(f"{tendon}_encoder_excursion_mm" for tendon in TENDONS),
    *(f"{tendon}_force_n" for tendon in TENDONS),
    *(f"previous_pid_reflected_delta_{tendon}_mm" for tendon in TENDONS),
)
_UNCONFIRMED_FINAL_ACTION_TERMINATIONS = frozenset(
    {
        "safety_filter_rejection",
        "dlc_feedback_unavailable",
        "communication_timeout",
        "communication_error",
        "force_telemetry_unavailable",
    }
)


@dataclass(frozen=True)
class DatasetProcessorConfig:
    max_observation_capture_age_s: float = 0.5
    max_observation_inference_age_s: float = 0.5
    max_servo_feedback_age_s: float = 0.25
    max_reward_capture_to_period_end_s: float = 0.25
    require_force: bool = True
    reject_held_vision: bool = True
    safety_constraint_is_terminal: bool | None = None

    def __post_init__(self) -> None:
        positive = (
            self.max_observation_capture_age_s,
            self.max_observation_inference_age_s,
            self.max_servo_feedback_age_s,
            self.max_reward_capture_to_period_end_s,
        )
        if any(not math.isfinite(value) or value <= 0.0 for value in positive):
            raise ValueError("Dataset timing thresholds must be finite and positive")


@dataclass(frozen=True)
class EpisodeBoundary:
    reason: str
    terminated: bool
    truncated: bool
    known: bool


@dataclass
class ProcessedDataset:
    observations: np.ndarray
    actions: np.ndarray
    rewards: np.ndarray
    next_observations: np.ndarray
    terminated: np.ndarray
    truncated: np.ndarray
    episode_ids: np.ndarray
    decision_ids: np.ndarray
    metadata: list[dict[str, Any]]
    report: dict[str, Any]

    def save(self, output_dir: str | Path) -> dict[str, Path]:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        arrays_path = root / "awac_transitions.npz"
        metadata_path = root / "transition_metadata.jsonl"
        report_path = root / "dataset_report.json"
        np.savez_compressed(
            arrays_path,
            schema_version=np.asarray([DATASET_SCHEMA_VERSION], dtype=np.int32),
            observation_features=np.asarray(OBSERVATION_FEATURES, dtype=np.str_),
            action_features=np.asarray(TENDONS, dtype=np.str_),
            observations=self.observations,
            actions=self.actions,
            rewards=self.rewards,
            next_observations=self.next_observations,
            terminated=self.terminated,
            truncated=self.truncated,
            episode_ids=self.episode_ids,
            decision_ids=self.decision_ids,
        )
        metadata_path.write_text(
            "".join(
                json.dumps(item, ensure_ascii=False, allow_nan=False) + "\n"
                for item in self.metadata
            ),
            encoding="utf-8",
        )
        report_path.write_text(
            json.dumps(self.report, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        return {
            "arrays": arrays_path,
            "metadata": metadata_path,
            "report": report_path,
        }


def classify_episode_boundary(
    manifest: Mapping[str, Any],
    config: DatasetProcessorConfig,
) -> EpisodeBoundary:
    """Map factual stop causes to RL boundary semantics without guessing."""

    explicit = manifest.get("rl_boundary")
    if isinstance(explicit, Mapping):
        reason = str(explicit.get("reason", "explicit"))
        terminated = bool(explicit.get("terminated", False))
        truncated = bool(explicit.get("truncated", False))
        if terminated and truncated:
            return EpisodeBoundary(reason, False, False, False)
        return EpisodeBoundary(reason, terminated, truncated, True)

    if bool(manifest.get("task_goal_achieved", False)):
        return EpisodeBoundary("task_goal_achieved", True, False, True)

    reason = str(manifest.get("termination_reason", "unknown"))
    if reason == "completed":
        return EpisodeBoundary("finite_horizon_completed", False, True, True)
    if reason in {"operator_stop", "force_quit"}:
        return EpisodeBoundary("operator_interrupt", False, True, True)
    if reason == "safety_filter_rejection":
        return EpisodeBoundary("safety_action_rejected", False, True, True)
    if reason == "dlc_feedback_unavailable":
        return EpisodeBoundary("dlc_feedback_unavailable", False, True, True)
    if reason in {
        "force_telemetry_unavailable",
        "communication_error",
        "communication_timeout",
    }:
        return EpisodeBoundary("sensor_or_communication_failure", False, True, True)
    if reason == "force_hard_limit":
        if config.safety_constraint_is_terminal is True:
            return EpisodeBoundary("safety_constraint_violation", True, False, True)
        if config.safety_constraint_is_terminal is False:
            return EpisodeBoundary("safety_intervention", False, True, True)
        return EpisodeBoundary(
            "safety_constraint_semantics_unresolved",
            False,
            False,
            False,
        )
    if reason == "error":
        return EpisodeBoundary("runtime_failure", False, True, True)
    return EpisodeBoundary("unknown_boundary", False, False, False)


class AWACDatasetProcessor:
    def __init__(self, config: DatasetProcessorConfig | None = None) -> None:
        self.config = config or DatasetProcessorConfig()

    def process(self, session_dirs: Iterable[str | Path]) -> ProcessedDataset:
        observations: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        rewards: list[float] = []
        next_observations: list[np.ndarray] = []
        terminated: list[bool] = []
        truncated: list[bool] = []
        episode_ids: list[str] = []
        decision_ids: list[int] = []
        metadata: list[dict[str, Any]] = []
        session_reports: list[dict[str, Any]] = []
        total_rejections: Counter[str] = Counter()

        for raw_dir in session_dirs:
            session_dir = Path(raw_dir)
            try:
                result = self._process_session(session_dir)
            except (OSError, ValueError) as exc:
                rejection = "session_processing_error"
                result = {
                    "transitions": [],
                    "rejections": Counter({rejection: 1}),
                    "report": {
                        "run_id": session_dir.name,
                        "session_dir": str(session_dir),
                        "transition_count": 0,
                        "rejection_counts": {rejection: 1},
                        "processing_error": f"{type(exc).__name__}: {exc}",
                    },
                }
            session_reports.append(result["report"])
            total_rejections.update(result["rejections"])
            for transition in result["transitions"]:
                observations.append(transition.pop("observation"))
                actions.append(transition.pop("action"))
                rewards.append(float(transition.pop("reward")))
                next_observations.append(transition.pop("next_observation"))
                terminated.append(bool(transition["terminated"]))
                truncated.append(bool(transition["truncated"]))
                episode_ids.append(str(transition["episode_id"]))
                decision_ids.append(int(transition["decision_id"]))
                metadata.append(transition)

        width = len(OBSERVATION_FEATURES)
        action_width = len(TENDONS)
        count = len(observations)
        report = {
            "schema_version": DATASET_SCHEMA_VERSION,
            "transition_count": count,
            "session_count": len(session_reports),
            "observation_features": list(OBSERVATION_FEATURES),
            "action_features": list(TENDONS),
            "action_semantics": (
                "pid-reflected incremental tendon excursion in millimetres; "
                "derived as the current filtered absolute PID target minus "
                "the previous decision target for legacy absolute sessions"
            ),
            "reward_definition": (
                "negative mean squared joint tracking error in rad^2; the "
                "measured posture is assigned by camera capture timestamp to "
                "one action interval and compared with the source-trajectory "
                "target interpolated at that capture's recorded source time"
            ),
            "causality_rules": {
                "observation": (
                    "capture <= decision and inference completion <= decision"
                ),
                "reward": (
                    "valid non-held capture inside [action_start, action_end); "
                    "the capture-to-source-time mapping is required; inference "
                    "may complete later and its delay is reported"
                ),
                "missing_values": "reject transition; never impute",
            },
            "boundary_rules": {
                "goal": "terminated only when explicitly recorded",
                "finite_horizon": "truncated",
                "operator_or_communication_interrupt": "truncated",
                "safety_constraint": (
                    "explicit configuration required; unresolved by default"
                ),
                "unconfirmed_final_action": (
                    "drop the final action interval after a safety, sensor, or "
                    "communication interruption because plant application is not confirmed"
                ),
            },
            "rejection_counts": dict(sorted(total_rejections.items())),
            "sessions": session_reports,
        }
        return ProcessedDataset(
            observations=np.asarray(observations, dtype=np.float32).reshape(count, width),
            actions=np.asarray(actions, dtype=np.float32).reshape(count, action_width),
            rewards=np.asarray(rewards, dtype=np.float32),
            next_observations=np.asarray(
                next_observations,
                dtype=np.float32,
            ).reshape(count, width),
            terminated=np.asarray(terminated, dtype=np.bool_),
            truncated=np.asarray(truncated, dtype=np.bool_),
            episode_ids=np.asarray(episode_ids, dtype=np.str_),
            decision_ids=np.asarray(decision_ids, dtype=np.int64),
            metadata=metadata,
            report=report,
        )

    def _process_session(self, session_dir: Path) -> dict[str, Any]:
        manifest_path = session_dir / "session_manifest.json"
        events_path = session_dir / "raw_events.jsonl"
        manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
        events = _load_events(events_path)
        run_id = str(manifest.get("run_id", session_dir.name))
        rejections: Counter[str] = Counter()
        transitions: list[dict[str, Any]] = []
        boundary = classify_episode_boundary(manifest, self.config)

        logger_stats = manifest.get("raw_logger_stats")
        if not isinstance(logger_stats, Mapping) or not bool(
            logger_stats.get("complete", False)
        ):
            rejections["raw_log_incomplete"] += 1
            return {
                "transitions": transitions,
                "rejections": rejections,
                "report": _session_report(
                    run_id,
                    0,
                    rejections,
                    boundary,
                    events,
                    logger_stats,
                ),
            }

        starts = _events_of_type(events, "control_session_start")
        decisions = _events_of_type(events, "decision")
        ticks = _events_of_type(events, "pid_tick")
        visions = _events_of_type(events, "vision_observation")
        playback_ends = _events_of_type(events, "playback_end")
        session_ends = _events_of_type(events, "session_end")
        if (
            len(starts) != 1
            or not decisions
            or not ticks
            or not playback_ends
            or not session_ends
        ):
            rejections["missing_required_raw_events"] += 1
            return {
                "transitions": transitions,
                "rejections": rejections,
                "report": _session_report(
                    run_id,
                    len(decisions),
                    rejections,
                    boundary,
                    events,
                    logger_stats,
                ),
            }

        if str(manifest.get("playback_mode")) != "rl_ff_zoh":
            rejections["not_rl_ff_zoh_session"] += 1
            return {
                "transitions": transitions,
                "rejections": rejections,
                "report": _session_report(
                    run_id,
                    len(decisions),
                    rejections,
                    boundary,
                    events,
                    logger_stats,
                ),
            }

        trajectory_path = manifest.get("input_csv_path")
        if not trajectory_path:
            rejections["missing_input_trajectory"] += 1
            return {
                "transitions": transitions,
                "rejections": rejections,
                "report": _session_report(
                    run_id,
                    len(decisions),
                    rejections,
                    boundary,
                    events,
                    logger_stats,
                ),
            }
        trajectory = ExcursionTrajectory.from_csv(Path(str(trajectory_path)))
        if trajectory.target_angles_deg is None:
            rejections["trajectory_missing_target_angles"] += len(decisions)
            return {
                "transitions": transitions,
                "rejections": rejections,
                "report": _session_report(
                    run_id,
                    len(decisions),
                    rejections,
                    boundary,
                    events,
                    logger_stats,
                ),
            }

        origin_payload = starts[0]["payload"]
        origins = _ordered_finite_mapping(
            origin_payload.get("session_encoder_origin_deg"),
            TENDONS,
        )
        calibration = origin_payload.get("calibration")
        if not isinstance(calibration, Mapping):
            rejections["missing_session_calibration"] += len(decisions)
            origins = None
            scales = None
        else:
            scales = _ordered_finite_mapping(
                calibration.get("encoder_degrees_per_mm"),
                TENDONS,
            )
        if origins is None or scales is None or np.any(scales == 0.0):
            rejections["invalid_session_origin_or_calibration"] += len(decisions)
            return {
                "transitions": transitions,
                "rejections": rejections,
                "report": _session_report(
                    run_id,
                    len(decisions),
                    rejections,
                    boundary,
                    events,
                    logger_stats,
                ),
            }

        decisions.sort(key=_event_time)
        ticks = [
            event
            for event in ticks
            if event.get("payload", {}).get("phase") == "playback"
        ]
        ticks.sort(key=_event_time)
        visions.sort(
            key=lambda event: float(
                event["payload"]["inference_finished_monotonic_s"]
            )
        )
        playback_end_s = _event_time(playback_ends[-1])

        for index, decision in enumerate(decisions):
            start_s = float(
                decision["payload"]["pid_target_reflected_monotonic_s"]
            )
            if index + 1 < len(decisions):
                end_s = float(
                    decisions[index + 1]["payload"][
                        "pid_target_reflected_monotonic_s"
                    ]
                )
                next_decision = decisions[index + 1]
            else:
                end_s = playback_end_s
                next_decision = None
            if not math.isfinite(start_s) or not math.isfinite(end_s) or end_s <= start_s:
                rejections["invalid_action_period"] += 1
                continue

            is_last_period = index == len(decisions) - 1
            if (
                is_last_period
                and str(manifest.get("termination_reason", ""))
                in _UNCONFIRMED_FINAL_ACTION_TERMINATIONS
            ):
                # A PC-side command record cannot prove physical application.
                # After these interruptions, retain no final interval whose
                # end-state/action application may be ambiguous.
                rejections["unconfirmed_final_action_interval"] += 1
                continue

            decision_payload = decision["payload"]
            applied_action_payload = decision_payload.get(
                "pid_reflected_target_mm",
                decision_payload.get(
                    "applied_action_excursion_mm",
                    decision_payload.get("action_excursion_mm"),
                ),
            )
            applied_target = _ordered_finite_mapping(
                applied_action_payload,
                TENDONS,
            )
            if applied_target is None:
                rejections["invalid_action"] += 1
                continue
            previous_target = np.zeros(len(TENDONS), dtype=np.float64)
            if index > 0:
                previous_payload = decisions[index - 1]["payload"]
                previous_mapping = previous_payload.get(
                    "pid_reflected_target_mm",
                    previous_payload.get(
                        "applied_action_excursion_mm",
                        previous_payload.get("action_excursion_mm"),
                    ),
                )
                parsed_previous = _ordered_finite_mapping(previous_mapping, TENDONS)
                if parsed_previous is None:
                    rejections["invalid_previous_action"] += 1
                    continue
                previous_target = parsed_previous
            action = applied_target - previous_target
            previous_increment = np.zeros(len(TENDONS), dtype=np.float64)
            if index > 0:
                earlier_target = np.zeros(len(TENDONS), dtype=np.float64)
                if index > 1:
                    earlier_payload = decisions[index - 2]["payload"]
                    earlier_mapping = earlier_payload.get(
                        "pid_reflected_target_mm",
                        earlier_payload.get(
                            "applied_action_excursion_mm",
                            earlier_payload.get("action_excursion_mm"),
                        ),
                    )
                    parsed_earlier = _ordered_finite_mapping(earlier_mapping, TENDONS)
                    if parsed_earlier is None:
                        rejections["invalid_previous_action"] += 1
                        continue
                    earlier_target = parsed_earlier
                previous_increment = previous_target - earlier_target
            scheduled_s = float(decision["payload"]["scheduled_elapsed_s"])
            target_rad = _target_angles_rad(trajectory, scheduled_s)
            next_scheduled_s = (
                scheduled_s
                if next_decision is None
                else float(next_decision["payload"]["scheduled_elapsed_s"])
            )
            next_target_rad = _target_angles_rad(trajectory, next_scheduled_s)

            obs = self._observation_at(
                start_s,
                target_rad,
                ticks,
                visions,
                origins,
                scales,
                previous_increment,
            )
            if obs[0] is None:
                rejections[str(obs[2])] += 1
                continue
            next_obs = self._observation_at(
                end_s,
                next_target_rad,
                ticks,
                visions,
                origins,
                scales,
                action,
            )
            if next_obs[0] is None:
                rejections[f"next_{next_obs[2]}"] += 1
                continue

            reward_result = self._reward_for_period(
                start_s,
                end_s,
                trajectory,
                visions,
            )
            if reward_result[0] is None:
                rejections[str(reward_result[2])] += 1
                continue

            attach_boundary = is_last_period and boundary.known
            transition_metadata = {
                "episode_id": run_id,
                "decision_id": int(decision["payload"]["decision_id"]),
                "action_start_monotonic_s": start_s,
                "action_end_monotonic_s": end_s,
                "action_duration_s": end_s - start_s,
                "action_source_elapsed_s": scheduled_s,
                "action_source_field": "pid_reflected_delta_mm",
                "action_semantics": "incremental_excursion_mm",
                "collection_condition": manifest.get(
                    "collection_condition",
                    _infer_collection_condition(decision_payload),
                ),
                "controller_type": manifest.get("controller_type"),
                "goal_id": manifest.get("goal_id"),
                "initialization_condition": manifest.get("initialization_condition"),
                "nominal_action_excursion_mm": decision_payload.get(
                    "nominal_action_excursion_mm"
                ),
                "feedback_action_excursion_mm": decision_payload.get(
                    "feedback_action_excursion_mm"
                ),
                "exploration_action_excursion_mm": decision_payload.get(
                    "exploration_action_excursion_mm"
                ),
                "requested_action_excursion_mm": decision_payload.get(
                    "requested_action_excursion_mm",
                    decision_payload.get("action_excursion_mm"),
                ),
                "applied_action_excursion_mm": (
                    applied_action_payload
                ),
                "previous_pid_reflected_target_mm": dict(zip(TENDONS, previous_target)),
                "pid_reflected_target_mm": dict(zip(TENDONS, applied_target)),
                "pid_reflected_delta_mm": dict(zip(TENDONS, action)),
                "requested_delta_mm": decision_payload.get("requested_delta_mm"),
                "requested_target_mm": decision_payload.get(
                    "requested_target_mm",
                    decision_payload.get("requested_action_excursion_mm"),
                ),
                "filtered_target_mm": decision_payload.get(
                    "filtered_target_mm",
                    applied_action_payload,
                ),
                "action_clipping": bool(
                    decision_payload.get("safety_intervened", False)
                ),
                "physical_servo_applied_time_status": decision_payload.get(
                    "physical_servo_applied_time_status",
                    "unobserved",
                ),
                "confirmed_applied_command": {
                    "pid_reflected_target_mm": dict(zip(TENDONS, applied_target)),
                    "transport_status": decision_payload.get(
                        "transport_status",
                        "not_joined_at_decision_event",
                    ),
                    "physical_applied_status": decision_payload.get(
                        "physical_servo_applied_time_status",
                        "unobserved",
                    ),
                },
                "action_timestamp_monotonic_s": start_s,
                "observation_timestamp_monotonic_s": obs[1][
                    "vision_capture_monotonic_s"
                ],
                "sensor_freshness": {
                    "vision_capture_age_s": obs[1]["vision_capture_age_s"],
                    "vision_inference_age_s": obs[1]["vision_inference_age_s"],
                    "servo_feedback_age_s": obs[1]["servo_feedback_age_s"],
                },
                "safety_filter_enabled": bool(
                    decision_payload.get("safety_filter_enabled", False)
                ),
                "safety_intervened": bool(
                    decision_payload.get("safety_intervened", False)
                ),
                "safety_rejected": bool(
                    decision_payload.get("safety_rejected", False)
                ),
                "safety_reasons": list(
                    decision_payload.get("safety_reasons") or []
                ),
                "observation_timing": obs[1],
                "next_observation_timing": next_obs[1],
                "reward_timing": reward_result[1],
                "reward_target_joint_rad": reward_result[1][
                    "target_joint_rad"
                ],
                "terminated": boundary.terminated if attach_boundary else False,
                "truncated": boundary.truncated if attach_boundary else False,
                "boundary_reason": boundary.reason if is_last_period else None,
                "termination_reason": (
                    manifest.get("termination_reason") if is_last_period else None
                ),
                "boundary_known": boundary.known if is_last_period else True,
                "observation": obs[0],
                "action": action,
                "reward": float(reward_result[0]),
                "next_observation": next_obs[0],
            }
            transitions.append(transition_metadata)

        if transitions and not (
            transitions[-1]["terminated"] or transitions[-1]["truncated"]
        ):
            if transitions[-1]["decision_id"] != int(
                decisions[-1]["payload"]["decision_id"]
            ):
                rejections["episode_boundary_not_attached_to_rejected_last_period"] += 1
            elif not boundary.known:
                rejections["episode_boundary_semantics_unresolved"] += 1

        return {
            "transitions": transitions,
            "rejections": rejections,
            "report": _session_report(
                run_id,
                len(decisions),
                rejections,
                boundary,
                events,
                logger_stats,
                accepted=len(transitions),
            ),
        }

    def _observation_at(
        self,
        boundary_s: float,
        target_rad: np.ndarray,
        ticks: Sequence[Mapping[str, Any]],
        visions: Sequence[Mapping[str, Any]],
        origins: np.ndarray,
        scales: np.ndarray,
        previous_action: np.ndarray,
    ) -> tuple[np.ndarray | None, dict[str, Any] | None, str | None]:
        tick = _latest_event_at_or_before(ticks, boundary_s)
        if tick is None:
            return None, None, "missing_control_observation"
        tick_payload = tick["payload"]
        servo_age_s = float(tick_payload.get("servo_feedback_age_s", math.inf))
        if servo_age_s > self.config.max_servo_feedback_age_s:
            return None, None, "stale_servo_feedback"

        available_vision = [
            event
            for event in visions
            if float(event["payload"]["capture_monotonic_s"]) <= boundary_s
            and float(event["payload"]["inference_finished_monotonic_s"])
            <= boundary_s
            and bool(event["payload"].get("valid", False))
            and not (
                self.config.reject_held_vision
                and bool(event["payload"].get("held_last", False))
            )
        ]
        if not available_vision:
            return None, None, "no_causally_available_vision"
        vision = max(
            available_vision,
            key=lambda event: float(event["payload"]["capture_monotonic_s"]),
        )
        vision_payload = vision["payload"]
        capture_s = float(vision_payload["capture_monotonic_s"])
        inference_s = float(vision_payload["inference_finished_monotonic_s"])
        capture_age_s = boundary_s - capture_s
        inference_age_s = boundary_s - inference_s
        if capture_age_s > self.config.max_observation_capture_age_s:
            return None, None, "stale_vision_capture"
        if inference_age_s > self.config.max_observation_inference_age_s:
            return None, None, "stale_vision_inference"

        observed_deg = _ordered_finite_mapping(
            vision_payload.get("angles_deg"),
            JOINTS,
        )
        tendon_payload = tick_payload.get("tendons")
        if observed_deg is None or not isinstance(tendon_payload, Mapping):
            return None, None, "invalid_observation_values"
        encoder_deg: list[float] = []
        force_n: list[float] = []
        for tendon in TENDONS:
            item = tendon_payload.get(tendon)
            if not isinstance(item, Mapping):
                return None, None, "missing_tendon_state"
            try:
                encoder_value = float(item["actual_encoder_deg"])
                force_value = float(item["force_n"])
            except (KeyError, TypeError, ValueError):
                if self.config.require_force:
                    return None, None, "missing_force_or_encoder"
                return None, None, "unsupported_optional_force_schema"
            if not math.isfinite(encoder_value) or not math.isfinite(force_value):
                return None, None, "non_finite_force_or_encoder"
            encoder_deg.append(encoder_value)
            force_n.append(force_value)
        encoder_excursion_mm = (
            np.asarray(encoder_deg, dtype=np.float64) - origins
        ) / scales
        observed_rad = np.deg2rad(observed_deg)
        observation = np.concatenate(
            (
                observed_rad,
                target_rad - observed_rad,
                encoder_excursion_mm,
                np.asarray(force_n, dtype=np.float64),
                np.asarray(previous_action, dtype=np.float64),
            )
        )
        if observation.shape != (len(OBSERVATION_FEATURES),) or not np.isfinite(
            observation
        ).all():
            return None, None, "invalid_observation_vector"
        timing = {
            "decision_boundary_monotonic_s": boundary_s,
            "control_tick_monotonic_s": _event_time(tick),
            "vision_frame_idx": int(vision_payload["frame_idx"]),
            "vision_capture_monotonic_s": capture_s,
            "vision_inference_finished_monotonic_s": inference_s,
            "vision_capture_age_s": capture_age_s,
            "vision_inference_age_s": inference_age_s,
            "servo_feedback_age_s": servo_age_s,
        }
        return observation, timing, None

    def _reward_for_period(
        self,
        start_s: float,
        end_s: float,
        trajectory: ExcursionTrajectory,
        visions: Sequence[Mapping[str, Any]],
    ) -> tuple[float | None, dict[str, Any] | None, str | None]:
        candidates = [
            event
            for event in visions
            if start_s
            <= float(event["payload"]["capture_monotonic_s"])
            < end_s
            and bool(event["payload"].get("valid", False))
            and not (
                self.config.reject_held_vision
                and bool(event["payload"].get("held_last", False))
            )
        ]
        if not candidates:
            return None, None, "missing_reward_capture_in_action_period"
        vision = max(
            candidates,
            key=lambda event: float(event["payload"]["capture_monotonic_s"]),
        )
        payload = vision["payload"]
        capture_s = float(payload["capture_monotonic_s"])
        inference_s = float(payload["inference_finished_monotonic_s"])
        end_offset_s = end_s - capture_s
        if end_offset_s > self.config.max_reward_capture_to_period_end_s:
            return None, None, "reward_capture_too_far_from_period_end"
        observed_deg = _ordered_finite_mapping(payload.get("angles_deg"), JOINTS)
        if observed_deg is None:
            return None, None, "invalid_reward_angles"
        try:
            source_elapsed_s = float(payload["source_elapsed_s_at_capture"])
        except (KeyError, TypeError, ValueError):
            return None, None, "missing_reward_capture_source_time"
        playback_elapsed_s = (
            source_elapsed_s - trajectory.source_time_origin_s
        ) / trajectory.time_scale
        if (
            not math.isfinite(playback_elapsed_s)
            or playback_elapsed_s < -1e-9
            or playback_elapsed_s > trajectory.duration_s + 1e-9
        ):
            return None, None, "invalid_reward_capture_source_time"
        target_rad = _target_angles_rad(trajectory, playback_elapsed_s)
        observed_rad = np.deg2rad(observed_deg)
        reward = -float(np.mean(np.square(observed_rad - target_rad)))
        timing = {
            "active_action_interval_start_monotonic_s": start_s,
            "active_action_interval_end_monotonic_s": end_s,
            "vision_frame_idx": int(payload["frame_idx"]),
            "capture_monotonic_s": capture_s,
            "inference_finished_monotonic_s": inference_s,
            "capture_to_period_end_s": end_offset_s,
            "inference_delay_s": inference_s - capture_s,
            "reward_available_after_period_end_s": max(0.0, inference_s - end_s),
            "measurement_belongs_to_action_period": True,
            "source_elapsed_s_at_capture": source_elapsed_s,
            "playback_elapsed_s_at_capture": playback_elapsed_s,
            "target_joint_rad": dict(zip(JOINTS, target_rad)),
            "observed_joint_rad": dict(zip(JOINTS, observed_rad)),
        }
        return reward, timing, None


def _load_events(path: Path) -> list[dict[str, Any]]:
    events: list[dict[str, Any]] = []
    expected_sequence = 0
    with path.open("r", encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            if not line.strip():
                continue
            event = json.loads(line)
            sequence = int(event.get("event_sequence", -1))
            if sequence != expected_sequence:
                raise ValueError(
                    f"Raw event sequence gap at {path}:{line_number}; "
                    f"expected {expected_sequence}, got {sequence}"
                )
            expected_sequence += 1
            events.append(event)
    return events


def _events_of_type(
    events: Sequence[Mapping[str, Any]],
    event_type: str,
) -> list[dict[str, Any]]:
    return [dict(event) for event in events if event.get("event_type") == event_type]


def _event_time(event: Mapping[str, Any]) -> float:
    return float(event["event_monotonic_s"])


def _latest_event_at_or_before(
    events: Sequence[Mapping[str, Any]],
    boundary_s: float,
) -> Mapping[str, Any] | None:
    times = [_event_time(event) for event in events]
    index = bisect.bisect_right(times, boundary_s + 1e-9) - 1
    return None if index < 0 else events[index]


def _ordered_finite_mapping(
    raw: object,
    names: Sequence[str],
) -> np.ndarray | None:
    if not isinstance(raw, Mapping):
        return None
    try:
        values = np.asarray([float(raw[name]) for name in names], dtype=np.float64)
    except (KeyError, TypeError, ValueError):
        return None
    if values.shape != (len(names),) or not np.isfinite(values).all():
        return None
    return values


def _infer_collection_condition(payload: Mapping[str, Any]) -> str:
    exploration = _ordered_finite_mapping(
        payload.get("exploration_action_excursion_mm"),
        TENDONS,
    )
    if exploration is not None and np.any(np.abs(exploration) > 0.0):
        return "D2"
    feedback = _ordered_finite_mapping(
        payload.get("feedback_action_excursion_mm"),
        TENDONS,
    )
    if feedback is not None and np.any(np.abs(feedback) > 0.0):
        return "D1"
    return "D0"


def _target_angles_rad(
    trajectory: ExcursionTrajectory,
    elapsed_s: float,
) -> np.ndarray:
    assert trajectory.target_angles_deg is not None
    times = trajectory.elapsed_times_s
    source_time = min(
        max(0.0, float(elapsed_s)) * trajectory.time_scale,
        times[-1],
    )
    right = bisect.bisect_right(times, source_time)
    if right <= 0:
        values = trajectory.target_angles_deg[0]
    elif right >= len(times):
        values = trajectory.target_angles_deg[-1]
    else:
        left = right - 1
        span = times[right] - times[left]
        fraction = 0.0 if span <= 0.0 else (source_time - times[left]) / span
        values = tuple(
            a + (b - a) * fraction
            for a, b in zip(
                trajectory.target_angles_deg[left],
                trajectory.target_angles_deg[right],
            )
        )
    return np.deg2rad(np.asarray(values, dtype=np.float64))


def _session_report(
    run_id: str,
    decision_count: int,
    rejections: Counter[str],
    boundary: EpisodeBoundary,
    events: Sequence[Mapping[str, Any]],
    logger_stats: object,
    *,
    accepted: int = 0,
) -> dict[str, Any]:
    ticks = _events_of_type(events, "pid_tick")
    tick_times = np.asarray([_event_time(event) for event in ticks], dtype=np.float64)
    periods = np.diff(tick_times) if len(tick_times) > 1 else np.asarray([])
    decisions = _events_of_type(events, "decision")
    decision_times = np.asarray(
        [_event_time(event) for event in decisions],
        dtype=np.float64,
    )
    decision_periods = (
        np.diff(decision_times)
        if len(decision_times) > 1
        else np.asarray([])
    )
    visions = _events_of_type(events, "vision_observation")
    initial_postures = _events_of_type(events, "initial_posture")
    action_coverage: dict[str, dict[str, float] | None] = {}
    tracking: dict[str, dict[str, float] | None] = {}
    for tendon in TENDONS:
        action_values = []
        for event in decisions:
            payload = event.get("payload", {})
            applied = payload.get(
                "applied_action_excursion_mm",
                payload.get("action_excursion_mm", {}),
            )
            raw = applied.get(tendon) if isinstance(applied, Mapping) else None
            try:
                value = float(raw)
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                action_values.append(value)
        action_coverage[tendon] = (
            None
            if not action_values
            else {
                "min_mm": float(np.min(action_values)),
                "max_mm": float(np.max(action_values)),
                "std_mm": float(np.std(action_values)),
            }
        )
        errors = []
        for event in ticks:
            raw = (
                event.get("payload", {})
                .get("tendons", {})
                .get(tendon, {})
                .get("pid", {})
                .get("position_error_deg")
            )
            try:
                value = abs(float(raw))
            except (TypeError, ValueError):
                continue
            if math.isfinite(value):
                errors.append(value)
        tracking[tendon] = (
            None
            if not errors
            else {
                "mean_abs_error_deg": float(np.mean(errors)),
                "p95_abs_error_deg": float(np.percentile(errors, 95)),
                "max_abs_error_deg": float(np.max(errors)),
            }
        )
    return {
        "run_id": run_id,
        "decision_count": decision_count,
        "pid_tick_count": len(ticks),
        "accepted_transition_count": accepted,
        "rejection_counts": dict(sorted(rejections.items())),
        "boundary": {
            "reason": boundary.reason,
            "terminated": boundary.terminated,
            "truncated": boundary.truncated,
            "known": boundary.known,
        },
        "control_period_s": {
            "count": int(periods.size),
            "mean": None if periods.size == 0 else float(np.mean(periods)),
            "p95": None
            if periods.size == 0
            else float(np.percentile(periods, 95)),
            "max": None if periods.size == 0 else float(np.max(periods)),
            "logged_violation_count": sum(
                bool(event.get("payload", {}).get("period_violation", False))
                for event in ticks
            ),
        },
        "decision_period_s": {
            "count": int(decision_periods.size),
            "mean": None
            if decision_periods.size == 0
            else float(np.mean(decision_periods)),
            "p95": None
            if decision_periods.size == 0
            else float(np.percentile(decision_periods, 95)),
            "max": None
            if decision_periods.size == 0
            else float(np.max(decision_periods)),
        },
        "vision_quality": {
            "event_count": len(visions),
            "valid_count": sum(
                bool(event.get("payload", {}).get("valid", False))
                for event in visions
            ),
            "held_last_count": sum(
                bool(event.get("payload", {}).get("held_last", False))
                for event in visions
            ),
        },
        "safety": {
            "enabled_decision_count": sum(
                bool(event.get("payload", {}).get("safety_filter_enabled", False))
                for event in decisions
            ),
            "intervention_count": sum(
                bool(event.get("payload", {}).get("safety_intervened", False))
                for event in decisions
            ),
            "rejection_count": sum(
                bool(event.get("payload", {}).get("safety_rejected", False))
                for event in decisions
            ),
        },
        "initial_posture": (
            None
            if not initial_postures
            else initial_postures[-1].get("payload")
        ),
        "action_coverage": action_coverage,
        "pid_tracking": tracking,
        "raw_logger_stats": logger_stats,
    }


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Build causally aligned AWAC transitions from raw sessions."
    )
    parser.add_argument("output_dir", type=Path)
    parser.add_argument("session_dirs", nargs="+", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    dataset = AWACDatasetProcessor().process(args.session_dirs)
    paths = dataset.save(args.output_dir)
    print(f"Transitions: {dataset.report['transition_count']}")
    for name, path in paths.items():
        print(f"{name}: {path}")


if __name__ == "__main__":
    main()
