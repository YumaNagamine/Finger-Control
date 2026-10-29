"""Version-2 physics teacher sampling with explicit feasibility semantics.

Method A samples near recorded FF trajectories. Method B covers the validated
mathematical moment-arm domain without asserting physical reachability. Method
C samples a local convex hull estimated from passive-motion camera records.
None of these datasets contains tension, reward, next state, or RL transitions.
"""

from __future__ import annotations

import csv
import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np

from controller.rl.legacy_bc_dataset import COMMON6_FEATURES, PairTable
from controller.rl.physics_teacher import JOINTS, MomentArmTeacherModel
from controller.servo_mapping import TENDONS


TEACHER_SAMPLING_SCHEMA_VERSION = 2


@dataclass
class TeacherSampleSet:
    observations: np.ndarray
    actions_mm: np.ndarray
    metadata: list[dict[str, Any]]
    report: dict[str, Any]

    def save(self, output_dir: str | Path, name: str) -> dict[str, Path]:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        data_path = root / f"{name}.npz"
        np.savez_compressed(
            data_path,
            observations=self.observations,
            actions_mm=self.actions_mm,
            feature_names=np.asarray(COMMON6_FEATURES),
            action_names=np.asarray(
                [f"{tendon}_target_excursion_mm" for tendon in TENDONS]
            ),
        )
        metadata_path = root / f"{name}_metadata.jsonl"
        with metadata_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in self.metadata:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        report_path = root / f"{name}_report.json"
        report_path.write_text(
            json.dumps(self.report, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        return {"data": data_path, "metadata": metadata_path, "report": report_path}


@dataclass
class PassiveReachabilityModel:
    postures_rad: np.ndarray
    families: np.ndarray
    source_rows: np.ndarray
    source_files: list[dict[str, Any]]
    neighbor_count: int

    @classmethod
    def from_directory(
        cls,
        directory: str | Path,
        moment_model: MomentArmTeacherModel,
        allowed_families: Sequence[str],
        neighbor_count: int = 12,
    ) -> "PassiveReachabilityModel":
        root = Path(directory).resolve()
        postures: list[np.ndarray] = []
        families: list[str] = []
        rows: list[int] = []
        sources: list[dict[str, Any]] = []
        allowed = set(allowed_families)
        for path in sorted(root.glob("dual_processed_*.csv")):
            family = _family_from_passive_filename(path.name)
            if family not in allowed:
                continue
            raw_bytes = path.read_bytes()
            accepted = 0
            with path.open("r", encoding="utf-8-sig", newline="") as handle:
                for row_index, row in enumerate(csv.DictReader(handle)):
                    try:
                        # The benchmark stores anatomical inner angles in deg.
                        posture = np.deg2rad(
                            [180.0 - float(row[joint]) for joint in JOINTS]
                        )
                    except (KeyError, TypeError, ValueError):
                        continue
                    if np.isfinite(posture).all() and _posture_in_domain(
                        moment_model, posture
                    ):
                        postures.append(posture)
                        families.append(family)
                        rows.append(row_index)
                        accepted += 1
            sources.append(
                {
                    "path": str(path),
                    "sha256": hashlib.sha256(raw_bytes).hexdigest(),
                    "trajectory_family": family,
                    "accepted_rows": accepted,
                    "angle_conversion": "flexion_deg = 180 - recorded_inner_angle_deg",
                }
            )
        values = np.asarray(postures, dtype=np.float64).reshape(-1, len(JOINTS))
        if len(values) < 2:
            raise ValueError("Passive reachability model has fewer than two valid rows")
        return cls(
            postures_rad=values,
            families=np.asarray(families),
            source_rows=np.asarray(rows, dtype=np.int64),
            source_files=sources,
            neighbor_count=neighbor_count,
        )

    def sample(self, count: int, rng: np.random.Generator) -> tuple[np.ndarray, list[dict[str, Any]]]:
        """Sample local same-family line segments to preserve joint coupling."""

        samples: list[np.ndarray] = []
        metadata: list[dict[str, Any]] = []
        family_names = sorted(set(self.families.tolist()))
        family_indices = {
            family: np.flatnonzero(self.families == family) for family in family_names
        }
        scale = np.maximum(np.std(self.postures_rad, axis=0), 1e-6)
        while len(samples) < count:
            family = family_names[int(rng.integers(len(family_names)))]
            candidates = family_indices[family]
            anchor_index = int(candidates[int(rng.integers(len(candidates)))])
            anchor = self.postures_rad[anchor_index]
            distances = np.linalg.norm(
                (self.postures_rad[candidates] - anchor) / scale, axis=1
            )
            order = np.argsort(distances)
            neighbors = candidates[order[1 : 1 + min(self.neighbor_count, len(order) - 1)]]
            if len(neighbors) == 0:
                continue
            neighbor_index = int(neighbors[int(rng.integers(len(neighbors)))])
            alpha = float(rng.uniform(0.05, 0.95))
            posture = alpha * anchor + (1.0 - alpha) * self.postures_rad[neighbor_index]
            samples.append(posture)
            metadata.append(
                {
                    "passive_family": family,
                    "passive_anchor_global_index": anchor_index,
                    "passive_neighbor_global_index": neighbor_index,
                    "passive_anchor_source_row": int(self.source_rows[anchor_index]),
                    "passive_neighbor_source_row": int(self.source_rows[neighbor_index]),
                    "interpolation_alpha": alpha,
                    "reachability_basis": "same-family local passive-motion convex segment",
                }
            )
        return np.asarray(samples), metadata

    def report(self) -> dict[str, Any]:
        covariance = np.cov(self.postures_rad, rowvar=False)
        correlation = np.corrcoef(self.postures_rad, rowvar=False)
        return {
            "semantic_scope": (
                "empirical passive-motion neighborhood; not proof of safe active reachability"
            ),
            "sample_rows": len(self.postures_rad),
            "families": _counts(self.families.tolist()),
            "neighbor_count": self.neighbor_count,
            "joint_range_deg": {
                joint: [
                    float(np.rad2deg(np.min(self.postures_rad[:, index]))),
                    float(np.rad2deg(np.max(self.postures_rad[:, index]))),
                ]
                for index, joint in enumerate(JOINTS)
            },
            "joint_correlation": correlation.tolist(),
            "joint_covariance_rad2": covariance.tolist(),
            "source_files": self.source_files,
        }


class PhysicsTeacherSamplerV2:
    def __init__(
        self,
        moment_model: MomentArmTeacherModel,
        reachability_model: PassiveReachabilityModel,
    ) -> None:
        self.model = moment_model
        self.reachability = reachability_model

    def sample_method_a(
        self,
        real_training: PairTable,
        count: int,
        seed: int,
        *,
        minimum_novelty_deg: float = 0.02,
    ) -> TeacherSampleSet:
        rng = np.random.default_rng(seed)
        by_session: dict[str, list[int]] = {}
        for index, row in enumerate(real_training.metadata):
            if row["in_moment_arm_domain"]:
                by_session.setdefault(row["session_id"], []).append(index)
        by_session = {key: value for key, value in by_session.items() if len(value) >= 2}
        if not by_session:
            raise ValueError("Method A needs two in-domain points in one training session")
        sessions = sorted(by_session)
        observations: list[np.ndarray] = []
        actions: list[np.ndarray] = []
        metadata: list[dict[str, Any]] = []
        real_deg = np.rad2deg(real_training.observations)
        attempts = 0
        while len(observations) < count and attempts < count * 500:
            attempts += 1
            session = sessions[int(rng.integers(len(sessions)))]
            first, second = rng.choice(by_session[session], 2, replace=False).tolist()
            if not np.allclose(
                real_training.observations[first, :3],
                real_training.observations[second, :3], atol=1e-12, rtol=0.0,
            ):
                continue
            alpha = float(rng.uniform(0.05, 0.95))
            q0 = real_training.observations[first, :3]
            target = (
                alpha * real_training.observations[first, 3:]
                + (1.0 - alpha) * real_training.observations[second, 3:]
            )
            observation = np.concatenate([q0, target])
            nearest = float(
                np.min(np.linalg.norm(real_deg - np.rad2deg(observation), axis=1))
            )
            if nearest < minimum_novelty_deg:
                continue
            action = self.model.action_from_session_origin(q0, target)
            observations.append(observation)
            actions.append(action)
            metadata.append(
                self._metadata(
                    "A_trajectory_near", len(observations) - 1, q0, target,
                    {
                        "source_session_id": session,
                        "source_family": real_training.metadata[first]["trajectory_family"],
                        "parent_pair_ids": [
                            real_training.metadata[first]["pair_id"],
                            real_training.metadata[second]["pair_id"],
                        ],
                        "interpolation_alpha": alpha,
                        "nearest_selected_real_distance_deg": nearest,
                        "reachability_status": "near recorded FF trajectory",
                    },
                )
            )
        if len(observations) != count:
            raise RuntimeError(f"Method A generated {len(observations)} / {count}")
        return self._dataset("A_trajectory_near", observations, actions, metadata, seed)

    def sample_method_b(self, count: int, seed: int) -> TeacherSampleSet:
        rng = np.random.default_rng(seed)
        q0 = _latin_hypercube_postures(self.model, count, rng)
        target = _latin_hypercube_postures(self.model, count, rng)
        observations = np.concatenate([q0, target], axis=1)
        actions = np.asarray(
            [self.model.action_from_session_origin(a, b) for a, b in zip(q0, target)]
        )
        metadata = [
            self._metadata(
                "B_model_domain", index, q0[index], target[index],
                {
                    "sampling": "independent Latin hypercube over each validated joint range",
                    "reachability_status": "unknown; mathematical model domain only",
                    "physical_reachability_asserted": False,
                },
            )
            for index in range(count)
        ]
        return self._dataset("B_model_domain", observations, actions, metadata, seed)

    def sample_method_c(self, count: int, seed: int) -> TeacherSampleSet:
        rng = np.random.default_rng(seed)
        q0, q0_meta = self.reachability.sample(count, rng)
        target, target_meta = self.reachability.sample(count, rng)
        observations = np.concatenate([q0, target], axis=1)
        actions = np.asarray(
            [self.model.action_from_session_origin(a, b) for a, b in zip(q0, target)]
        )
        metadata = [
            self._metadata(
                "C_passive_reachable", index, q0[index], target[index],
                {
                    "q0_passive_source": q0_meta[index],
                    "target_passive_source": target_meta[index],
                    "reachability_status": "inside empirical passive-motion local hull",
                    "physical_reachability_asserted": False,
                },
            )
            for index in range(count)
        ]
        return self._dataset("C_passive_reachable", observations, actions, metadata, seed)

    def _metadata(
        self,
        method: str,
        index: int,
        q0: np.ndarray,
        target: np.ndarray,
        extra: Mapping[str, Any],
    ) -> dict[str, Any]:
        return {
            "teacher_pair_id": f"v2:{method}:{index:06d}",
            "source_kind": "moment_arm_geometry_teacher",
            "teacher_method": method,
            "q0_plan_rad": np.asarray(q0).tolist(),
            "target_posture_rad": np.asarray(target).tolist(),
            "action_reference": "session_plan_origin",
            "action_is_incremental": False,
            "angle_unit": "rad",
            "action_unit": "mm",
            "integration_path": (
                "analytic endpoint integral of separable direction-independent joint polynomials"
            ),
            "in_moment_arm_domain": True,
            "contains_measured_tension": False,
            "contains_next_state": False,
            "contains_reward": False,
            "is_rl_transition": False,
            **dict(extra),
        }

    def _dataset(
        self,
        method: str,
        observations: Sequence[np.ndarray] | np.ndarray,
        actions: Sequence[np.ndarray] | np.ndarray,
        metadata: list[dict[str, Any]],
        seed: int,
    ) -> TeacherSampleSet:
        obs = np.asarray(observations, dtype=np.float64).reshape(-1, 6)
        act = np.asarray(actions, dtype=np.float64).reshape(-1, 6)
        report = {
            "schema_version": TEACHER_SAMPLING_SCHEMA_VERSION,
            "method": method,
            "seed": seed,
            "pairs": len(obs),
            "all_in_moment_arm_domain": True,
            "contains_tension": False,
            "contains_next_state": False,
            "contains_reward": False,
            "contains_rl_transitions": False,
            "observation_range_deg": _range_report(np.rad2deg(obs), COMMON6_FEATURES),
            "action_range_mm": _range_report(
                act, [f"{tendon}_action_mm" for tendon in TENDONS]
            ),
            "moment_arm_model": self.model.provenance(),
            "passive_reachability_model": (
                self.reachability.report() if method == "C_passive_reachable" else None
            ),
        }
        return TeacherSampleSet(obs, act, metadata, report)


def combine_teacher_sets(
    sets: Sequence[TeacherSampleSet], name: str = "ABC_combined"
) -> TeacherSampleSet:
    observations = np.concatenate([item.observations for item in sets])
    actions = np.concatenate([item.actions_mm for item in sets])
    metadata = [row for item in sets for row in item.metadata]
    return TeacherSampleSet(
        observations,
        actions,
        metadata,
        {
            "schema_version": TEACHER_SAMPLING_SCHEMA_VERSION,
            "method": name,
            "pairs": len(observations),
            "component_counts": {item.report["method"]: len(item.observations) for item in sets},
            "contains_tension": False,
            "contains_next_state": False,
            "contains_reward": False,
            "contains_rl_transitions": False,
        },
    )


def _latin_hypercube_postures(
    model: MomentArmTeacherModel, count: int, rng: np.random.Generator
) -> np.ndarray:
    values = np.empty((count, len(JOINTS)), dtype=np.float64)
    for joint_index, joint in enumerate(JOINTS):
        low, high = model.valid_ranges_rad[joint]
        bins = (np.arange(count) + rng.uniform(size=count)) / count
        rng.shuffle(bins)
        values[:, joint_index] = low + bins * (high - low)
    return values


def _family_from_passive_filename(name: str) -> str:
    stem = Path(name).stem
    prefix = "dual_processed_"
    remainder = stem[len(prefix) :] if stem.startswith(prefix) else stem
    marker = remainder.rfind("_20")
    family = remainder[:marker] if marker >= 0 else remainder
    aliases = {"MCPFlexion": "MCPFlexion", "MIddleFlexion": "MIddleFlexion"}
    return aliases.get(family, family)


def _posture_in_domain(model: MomentArmTeacherModel, posture: np.ndarray) -> bool:
    for index, joint in enumerate(JOINTS):
        low, high = model.valid_ranges_rad[joint]
        if not low <= posture[index] <= high:
            return False
    return True


def _range_report(values: np.ndarray, names: Sequence[str]) -> dict[str, list[float]]:
    return {
        name: [float(np.min(values[:, index])), float(np.max(values[:, index]))]
        for index, name in enumerate(names)
    }


def _counts(values: Sequence[str]) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[str(value)] = result.get(str(value), 0) + 1
    return dict(sorted(result.items()))
