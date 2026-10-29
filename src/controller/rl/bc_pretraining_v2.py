"""Version-2 C3/C4 BC study focused on in-domain FF imitation."""

from __future__ import annotations

import argparse
import csv
import json
import math
import random
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch
from torch import nn

from controller.rl.bc_pretraining import StandardScalerState
from controller.rl.legacy_bc_dataset import COMMON6_FEATURES, PairTable, load_pair_table
from controller.rl.physics_teacher import MomentArmTeacherModel
from controller.rl.physics_teacher_sampling_v2 import (
    PassiveReachabilityModel,
    PhysicsTeacherSamplerV2,
    TeacherSampleSet,
    combine_teacher_sets,
)
from controller.servo_mapping import TENDONS


EXPERIMENT_SCHEMA_VERSION = 2


@dataclass(frozen=True)
class ActionTransform:
    mode: str
    center: tuple[float, ...]
    scale: tuple[float, ...]

    @classmethod
    def fit(
        cls, actions: np.ndarray, mode: str, wide_tanh_std_multiplier: float = 6.0
    ) -> "ActionTransform":
        if mode == "linear_standard":
            center = np.mean(actions, axis=0)
            scale = np.std(actions, axis=0)
        elif mode == "tanh_minmax_legacy":
            minimum, maximum = np.min(actions, axis=0), np.max(actions, axis=0)
            center = 0.5 * (minimum + maximum)
            scale = 0.5 * (maximum - minimum)
        elif mode == "tanh_wide_standard":
            center = np.mean(actions, axis=0)
            scale = wide_tanh_std_multiplier * np.std(actions, axis=0)
        else:
            raise ValueError(f"Unknown output mode: {mode}")
        scale = np.asarray(scale, dtype=np.float64)
        scale[scale < 1e-12] = 1.0
        return cls(mode, tuple(np.asarray(center).tolist()), tuple(scale.tolist()))

    @property
    def bounded(self) -> bool:
        return self.mode.startswith("tanh_")

    def target(self, actions: np.ndarray) -> np.ndarray:
        normalized = (actions - np.asarray(self.center)) / np.asarray(self.scale)
        if self.bounded and np.any(np.abs(normalized) >= 1.0):
            # Training labels outside a bounded transform cannot be represented.
            # Keep their boundary target finite while recording saturation stats.
            normalized = np.clip(normalized, -0.999999, 0.999999)
        return normalized

    def decode(self, network_output: np.ndarray) -> np.ndarray:
        return np.asarray(self.center) + np.asarray(self.scale) * network_output

    def representable(self, actions: np.ndarray) -> np.ndarray:
        if not self.bounded:
            return np.ones(len(actions), dtype=bool)
        normalized = (actions - np.asarray(self.center)) / np.asarray(self.scale)
        return np.all(np.abs(normalized) <= 1.0, axis=1)

    def physical_output_range_mm(self) -> dict[str, list[float] | None]:
        if not self.bounded:
            return {tendon: None for tendon in TENDONS}
        center, scale = np.asarray(self.center), np.asarray(self.scale)
        return {
            tendon: [float(center[i] - scale[i]), float(center[i] + scale[i])]
            for i, tendon in enumerate(TENDONS)
        }


class BCActorV2(nn.Module):
    def __init__(self, output_mode: str) -> None:
        super().__init__()
        self.output_mode = output_mode
        self.encoder = nn.Sequential(
            nn.Linear(6, 128), nn.ReLU(), nn.Linear(128, 128), nn.ReLU()
        )
        self.mean = nn.Linear(128, 6)

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        output = self.mean(self.encoder(observations))
        return torch.tanh(output) if self.output_mode.startswith("tanh_") else output


@dataclass
class EvaluationSet:
    name: str
    observations: np.ndarray
    actions_mm: np.ndarray
    metadata: list[dict[str, Any]]
    label_source: str

    def save(self, root: Path) -> None:
        np.savez_compressed(
            root / f"{self.name}.npz",
            observations=self.observations,
            actions_mm=self.actions_mm,
        )
        (root / f"{self.name}_metadata.jsonl").write_text(
            "".join(json.dumps(row, ensure_ascii=False) + "\n" for row in self.metadata),
            encoding="utf-8",
        )


def build_nested_balanced_order(table: PairTable) -> list[int]:
    """Return a nested order balanced by family, session, and elapsed time."""

    by_family_session: dict[str, dict[str, list[int]]] = {}
    for index, row in enumerate(table.metadata):
        by_family_session.setdefault(row["trajectory_family"], {}).setdefault(
            row["session_id"], []
        ).append(index)
    queues: dict[tuple[str, str], list[int]] = {}
    for family, sessions in by_family_session.items():
        for session, indices in sessions.items():
            indices.sort(key=lambda i: table.metadata[i]["source_elapsed_s"])
            queues[(family, session)] = _temporal_farthest_order(indices, table)
    family_sessions = {
        family: sorted(sessions) for family, sessions in by_family_session.items()
    }
    family_cursor = {family: 0 for family in family_sessions}
    order: list[int] = []
    while len(order) < len(table.metadata):
        progressed = False
        for family in sorted(family_sessions):
            sessions = family_sessions[family]
            for offset in range(len(sessions)):
                position = (family_cursor[family] + offset) % len(sessions)
                session = sessions[position]
                queue = queues[(family, session)]
                if queue:
                    order.append(queue.pop(0))
                    family_cursor[family] = (position + 1) % len(sessions)
                    progressed = True
                    break
        if not progressed:
            break
    if sorted(order) != list(range(len(table.metadata))):
        raise RuntimeError("Nested selector did not cover every training pair")
    return order


def _temporal_farthest_order(indices: list[int], table: PairTable) -> list[int]:
    if len(indices) <= 2:
        return list(indices)
    times = np.asarray([table.metadata[i]["source_elapsed_s"] for i in indices])
    selected_positions = [0, len(indices) - 1]
    order = [indices[0], indices[-1]]
    remaining = set(range(1, len(indices) - 1))
    while remaining:
        position = max(
            remaining,
            key=lambda p: min(abs(times[p] - times[s]) for s in selected_positions),
        )
        selected_positions.append(position)
        remaining.remove(position)
        order.append(indices[position])
    return order


def _subset(table: PairTable, indices: Sequence[int]) -> PairTable:
    selected = np.asarray(indices, dtype=int)
    return PairTable(
        table.observations[selected], table.actions_mm[selected],
        [table.metadata[i] for i in selected], table.feature_names,
    )


def _set_seed(seed: int) -> None:
    random.seed(seed); np.random.seed(seed); torch.manual_seed(seed)
    torch.use_deterministic_algorithms(True)


def train_run(
    *,
    condition: str,
    seed: int,
    real_train: PairTable,
    physics: TeacherSampleSet | None,
    evaluation_sets: Sequence[EvaluationSet],
    output_mode: str,
    config: Mapping[str, Any],
    model_config: Mapping[str, Any],
    output_root: Path,
) -> dict[str, Any]:
    _set_seed(seed)
    observation_scaler = StandardScalerState.fit(real_train.observations)
    action_transform = ActionTransform.fit(
        real_train.actions_mm, output_mode,
        float(model_config["wide_tanh_std_multiplier"]),
    )
    actor = BCActorV2(output_mode)
    optimizer = torch.optim.Adam(actor.parameters(), lr=float(config["learning_rate"]))
    real_x = torch.as_tensor(
        observation_scaler.transform(real_train.observations), dtype=torch.float32
    )
    real_y = torch.as_tensor(
        action_transform.target(real_train.actions_mm), dtype=torch.float32
    )
    if physics is not None:
        physics_x = torch.as_tensor(
            observation_scaler.transform(physics.observations), dtype=torch.float32
        )
        physics_y = torch.as_tensor(
            action_transform.target(physics.actions_mm), dtype=torch.float32
        )
    else:
        physics_x = physics_y = None
    generator = torch.Generator().manual_seed(seed + 83000)
    batch_size = int(config["batch_size"])
    physics_count = (
        int(round(batch_size * float(config["physics_batch_fraction"])))
        if physics is not None else 0
    )
    real_count = batch_size - physics_count
    history: list[dict[str, float]] = []
    for update in range(1, int(config["gradient_updates"]) + 1):
        real_indices = torch.randint(len(real_x), (real_count,), generator=generator)
        batch_x, batch_y = real_x[real_indices], real_y[real_indices]
        if physics_x is not None and physics_y is not None:
            physics_indices = torch.randint(
                len(physics_x), (physics_count,), generator=generator
            )
            batch_x = torch.cat([batch_x, physics_x[physics_indices]])
            batch_y = torch.cat([batch_y, physics_y[physics_indices]])
        prediction = actor(batch_x)
        loss = torch.mean((prediction - batch_y) ** 2)
        optimizer.zero_grad(set_to_none=True); loss.backward(); optimizer.step()
        if update == 1 or update % int(config["evaluation_interval"]) == 0:
            history.append({"update": update, "training_transformed_mse": float(loss.detach())})
    metrics: dict[str, Any] = {}
    predictions: dict[str, np.ndarray] = {}
    for evaluation in evaluation_sets:
        predicted = predict_mm(actor, evaluation.observations, observation_scaler, action_transform)
        predictions[evaluation.name] = predicted
        metrics[evaluation.name] = error_metrics(
            evaluation.actions_mm, predicted, evaluation.metadata
        )
        metrics[evaluation.name]["label_source"] = evaluation.label_source
        metrics[evaluation.name]["representable_target_pairs"] = int(
            action_transform.representable(evaluation.actions_mm).sum()
        )
        metrics[evaluation.name]["unrepresentable_target_pairs"] = int(
            (~action_transform.representable(evaluation.actions_mm)).sum()
        )
    run_dir = output_root / "runs" / condition / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "condition": condition,
        "seed": seed,
        "output_mode": output_mode,
        "architecture": {"input": 6, "hidden": [128, 128], "output": 6},
        "state_dict": actor.state_dict(),
        "observation_scaler": asdict(observation_scaler),
        "action_transform": asdict(action_transform),
        "model_output_range_mm": action_transform.physical_output_range_mm(),
        "safety_action_limits_embedded": False,
        "scaler_fit_source": "selected real training pairs only",
        "scaler_fit_pair_ids": [row["pair_id"] for row in real_train.metadata],
        "real_training_pairs": len(real_train.observations),
        "physics_training_pairs": 0 if physics is None else len(physics.observations),
        "physics_unrepresentable_pairs": (
            0 if physics is None else int(
                (~action_transform.representable(physics.actions_mm)).sum()
            )
        ),
        "physics_method": None if physics is None else physics.report["method"],
        "metrics": metrics,
    }
    torch.save(checkpoint, run_dir / "checkpoint.pt")
    (run_dir / "scalers.json").write_text(
        json.dumps(
            {
                "fit_source": "selected real training pairs only",
                "observation_scaler": asdict(observation_scaler),
                "action_transform": asdict(action_transform),
                "safety_action_limits_embedded": False,
            },
            indent=2,
            allow_nan=False,
        ),
        encoding="utf-8",
    )
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"
    )
    with (run_dir / "loss_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0])); writer.writeheader(); writer.writerows(history)
    for name, prediction in predictions.items():
        np.savez_compressed(run_dir / f"predictions_{name}.npz", prediction_mm=prediction)
    return {
        "condition": condition,
        "seed": seed,
        "output_mode": output_mode,
        "real_pairs": len(real_train.observations),
        "physics_pairs": 0 if physics is None else len(physics.observations),
        "physics_method": None if physics is None else physics.report["method"],
        "metrics": metrics,
        "checkpoint": str(run_dir / "checkpoint.pt"),
    }


def predict_mm(
    actor: BCActorV2,
    observations: np.ndarray,
    observation_scaler: StandardScalerState,
    transform: ActionTransform,
) -> np.ndarray:
    actor.eval()
    with torch.no_grad():
        output = actor(
            torch.as_tensor(
                observation_scaler.transform(observations), dtype=torch.float32
            )
        ).numpy()
    return transform.decode(output)


def error_metrics(
    truth: np.ndarray, prediction: np.ndarray, metadata: Sequence[Mapping[str, Any]]
) -> dict[str, Any]:
    error = prediction - truth
    result: dict[str, Any] = {
        "pairs": len(truth),
        "mae_mm": float(np.mean(np.abs(error))),
        "rmse_mm": float(np.sqrt(np.mean(error**2))),
        "per_tendon_rmse_mm": {
            tendon: float(np.sqrt(np.mean(error[:, i] ** 2)))
            for i, tendon in enumerate(TENDONS)
        },
    }
    if metadata and "in_moment_arm_domain" in metadata[0]:
        for label, inside in (("inside", True), ("outside", False)):
            mask = np.asarray([bool(row["in_moment_arm_domain"]) == inside for row in metadata])
            if mask.any():
                result[f"{label}_pairs"] = int(mask.sum())
                result[f"{label}_rmse_mm"] = float(np.sqrt(np.mean(error[mask] ** 2)))
    return result


def build_evaluation_sets(
    full_train: PairTable,
    legacy_validation: PairTable,
    sampler: PhysicsTeacherSamplerV2,
    config: Mapping[str, Any],
    output_root: Path,
) -> list[EvaluationSet]:
    eval_root = output_root / "evaluation_datasets"; eval_root.mkdir(parents=True, exist_ok=True)
    real_inside = legacy_validation.subset(
        [row["in_moment_arm_domain"] for row in legacy_validation.metadata]
    )
    method_b = sampler.sample_method_b(
        int(config["independent_model_domain_pairs"]), int(config["seed_model_domain"])
    )
    method_c = sampler.sample_method_c(
        int(config["independent_passive_reachable_pairs"]), int(config["seed_passive_reachable"])
    )
    near = sampler.sample_method_a(
        full_train, int(config["trajectory_near_pairs"]),
        int(config["seed_trajectory_near"]), minimum_novelty_deg=0.02,
    )
    for name, teacher in (
        ("eval_A_independent_model_domain", method_b),
        ("eval_A_independent_passive_reachable", method_c),
        ("eval_B_trajectory_near", near),
    ):
        teacher.save(eval_root, name)
    evaluations = [
        EvaluationSet(
            "eval_A_real_heldout_in_domain", real_inside.observations,
            real_inside.actions_mm, real_inside.metadata,
            "unused real log FF commands; labels originally produced by moment-arm FF",
        ),
        EvaluationSet(
            "eval_A_independent_model_domain", method_b.observations,
            method_b.actions_mm, method_b.metadata,
            "independent analytical moment-arm labels; physical reachability not asserted",
        ),
        EvaluationSet(
            "eval_A_independent_passive_reachable", method_c.observations,
            method_c.actions_mm, method_c.metadata,
            "independent analytical labels in passive-motion local hull",
        ),
        EvaluationSet(
            "eval_B_trajectory_near", near.observations, near.actions_mm,
            near.metadata, "independent analytical labels near training trajectories",
        ),
        EvaluationSet(
            "eval_C_legacy_family_ood", legacy_validation.observations,
            legacy_validation.actions_mm, legacy_validation.metadata,
            "unused real log FF commands from Edinburgh and FullExtension",
        ),
    ]
    for evaluation in evaluations:
        evaluation.save(eval_root)
    report = {
        evaluation.name: {
            "pairs": len(evaluation.observations),
            "label_source": evaluation.label_source,
        }
        for evaluation in evaluations
    }
    (eval_root / "evaluation_dataset_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return evaluations


def run_v2_experiment(
    dataset_dir: str | Path,
    moment_arm_dir: str | Path,
    passive_motion_dir: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    output_root = Path(output_dir); output_root.mkdir(parents=True, exist_ok=True)
    table = load_pair_table(
        Path(dataset_dir) / "bc_common6_completed.npz",
        Path(dataset_dir) / "bc_common6_completed_metadata.jsonl",
    )
    full_train = table.subset([row["split"] == "training" for row in table.metadata])
    legacy_validation = table.subset([row["split"] == "validation" for row in table.metadata])
    moment_model = MomentArmTeacherModel.from_directory(moment_arm_dir)
    reachability = PassiveReachabilityModel.from_directory(
        passive_motion_dir, moment_model,
        config["teacher"]["passive_training_families"],
        int(config["teacher"]["passive_neighbor_count"]),
    )
    sampler = PhysicsTeacherSamplerV2(moment_model, reachability)
    evaluations = build_evaluation_sets(
        full_train, legacy_validation, sampler, config["evaluation"], output_root
    )
    order = build_nested_balanced_order(full_train)
    subset_report: dict[str, Any] = {}
    subsets: dict[int, PairTable] = {}
    for count in config["real_data_curve"]["pair_counts"]:
        subset = _subset(full_train, order[: int(count)])
        subsets[int(count)] = subset
        subset_report[str(count)] = {
            "pairs": len(subset.metadata),
            "sessions": len({row["session_id"] for row in subset.metadata}),
            "families": _counts(row["trajectory_family"] for row in subset.metadata),
            "session_counts": _counts(row["session_id"] for row in subset.metadata),
            "pair_ids": [row["pair_id"] for row in subset.metadata],
        }
    (output_root / "real_subset_report.json").write_text(
        json.dumps(subset_report, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    (output_root / "passive_reachability_report.json").write_text(
        json.dumps(reachability.report(), indent=2, ensure_ascii=False), encoding="utf-8"
    )
    teacher_root = output_root / "teacher_datasets"; teacher_root.mkdir(exist_ok=True)
    b_pool = sampler.sample_method_b(
        int(config["teacher"]["pairs_per_method"]), int(config["teacher"]["training_seed"]) + 2
    )
    c_pool = sampler.sample_method_c(
        int(config["teacher"]["pairs_per_method"]), int(config["teacher"]["training_seed"]) + 3
    )
    b_pool.save(teacher_root, "teacher_B_model_domain")
    c_pool.save(teacher_root, "teacher_C_passive_reachable")

    runs: list[dict[str, Any]] = []
    training_cfg = config["training"]; model_cfg = config["model"]
    primary_mode = model_cfg["primary_output_mode"]
    teacher_by_count: dict[int, TeacherSampleSet] = {}
    for count, subset in subsets.items():
        a_pool = sampler.sample_method_a(
            subset, int(config["teacher"]["pairs_per_method"]),
            int(config["teacher"]["training_seed"]) + count,
            minimum_novelty_deg=float(config["teacher"]["minimum_method_a_novelty_deg"]),
        )
        a_pool.save(teacher_root, f"teacher_A_trajectory_near_real_{count}")
        abc = combine_teacher_sets([a_pool, b_pool, c_pool], f"ABC_real_{count}")
        abc.save(teacher_root, f"teacher_ABC_real_{count}")
        teacher_by_count[count] = abc
        for seed in training_cfg["seeds"]:
            runs.append(train_run(
                condition=f"curve_C3_real_{count}", seed=int(seed), real_train=subset,
                physics=None, evaluation_sets=evaluations, output_mode=primary_mode,
                config=training_cfg, model_config=model_cfg, output_root=output_root,
            ))
            runs.append(train_run(
                condition=f"curve_C4_ABC_real_{count}", seed=int(seed), real_train=subset,
                physics=abc, evaluation_sets=evaluations, output_mode=primary_mode,
                config=training_cfg, model_config=model_cfg, output_root=output_root,
            ))

    method_count = int(config["method_comparison_real_pairs"])
    method_subset = subsets[method_count]
    method_a = sampler.sample_method_a(
        method_subset, int(config["teacher"]["pairs_per_method"]),
        int(config["teacher"]["training_seed"]) + 101,
        minimum_novelty_deg=float(config["teacher"]["minimum_method_a_novelty_deg"]),
    )
    for method_name, teacher in (
        ("A", method_a), ("B", b_pool), ("C", c_pool),
        ("ABC", teacher_by_count[method_count]),
    ):
        for seed in training_cfg["seeds"]:
            runs.append(train_run(
                condition=f"method_C4_{method_name}_real_{method_count}", seed=int(seed),
                real_train=method_subset, physics=teacher, evaluation_sets=evaluations,
                output_mode=primary_mode, config=training_cfg, model_config=model_cfg,
                output_root=output_root,
            ))

    output_count = int(config["output_mode_comparison_real_pairs"])
    output_subset, output_physics = subsets[output_count], teacher_by_count[output_count]
    for mode in ("tanh_minmax_legacy", "tanh_wide_standard"):
        for label, teacher in (("C3", None), ("C4_ABC", output_physics)):
            for seed in training_cfg["seeds"]:
                runs.append(train_run(
                    condition=f"output_{mode}_{label}_real_{output_count}", seed=int(seed),
                    real_train=output_subset, physics=teacher,
                    evaluation_sets=evaluations, output_mode=mode,
                    config=training_cfg, model_config=model_cfg, output_root=output_root,
                ))

    summary = aggregate_results(runs, config, moment_model, evaluations)
    (output_root / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    _write_flat_csv(runs, output_root / "evaluation_runs.csv")
    make_plots(summary, runs, evaluations, output_root)
    write_report(summary, output_root / "experiment_report.md")
    (output_root / "resolved_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False), encoding="utf-8"
    )
    return summary


def aggregate_results(
    runs: Sequence[Mapping[str, Any]],
    config: Mapping[str, Any],
    model: MomentArmTeacherModel,
    evaluations: Sequence[EvaluationSet],
) -> dict[str, Any]:
    conditions = sorted({str(run["condition"]) for run in runs})
    aggregated: dict[str, Any] = {}
    for condition in conditions:
        selected = [run for run in runs if run["condition"] == condition]
        aggregated[condition] = {
            "output_mode": selected[0]["output_mode"],
            "real_pairs": selected[0]["real_pairs"],
            "physics_pairs": selected[0]["physics_pairs"],
            "physics_method": selected[0]["physics_method"],
            "evaluations": {
                evaluation.name: _aggregate_evaluation(
                    selected, evaluation.name
                )
                for evaluation in evaluations
            },
        }
    analytical = {
        evaluation.name: _analytical_reference(evaluation, model)
        for evaluation in evaluations
    }
    return {
        "schema_version": EXPERIMENT_SCHEMA_VERSION,
        "research_scope": (
            "FF function approximation and BC-stage data efficiency only; not hardware tracking "
            "and not online-RL experience efficiency"
        ),
        "conditions": aggregated,
        "analytical_moment_arm_reference": analytical,
        "run_count": len(runs),
        "seeds": config["training"]["seeds"],
        "limitations": [
            "Synthetic analytical tests measure imitation of the same moment-arm model.",
            "Method B does not assert physical reachability of independent joint combinations.",
            "Method C estimates only a passive-motion local hull, not safe active reachability.",
            "The six-dimensional BC is a feed-forward approximator, not a measured-state feedback policy.",
        ],
    }


def _aggregate_evaluation(
    selected: Sequence[Mapping[str, Any]], evaluation_name: str
) -> dict[str, Any]:
    first = selected[0]["metrics"][evaluation_name]
    result: dict[str, Any] = {
        "rmse_mm": _mean_std([run["metrics"][evaluation_name]["rmse_mm"] for run in selected]),
        "mae_mm": _mean_std([run["metrics"][evaluation_name]["mae_mm"] for run in selected]),
        "unrepresentable_target_pairs": first["unrepresentable_target_pairs"],
        "label_source": first["label_source"],
    }
    for key in ("inside_pairs", "outside_pairs"):
        if key in first:
            result[key] = first[key]
    for key in ("inside_rmse_mm", "outside_rmse_mm"):
        if key in first:
            result[key] = _mean_std(
                [run["metrics"][evaluation_name][key] for run in selected]
            )
    return result


def _analytical_reference(
    evaluation: EvaluationSet, model: MomentArmTeacherModel
) -> dict[str, Any]:
    if evaluation.label_source.startswith("independent analytical"):
        return {
            "rmse_mm": 0.0,
            "evaluated_pairs": len(evaluation.observations),
            "skipped_out_of_domain_pairs": 0,
            "label_source": evaluation.label_source,
        }
    rmse, evaluated = _analytical_rmse(evaluation, model)
    return {
        "rmse_mm": rmse,
        "evaluated_pairs": evaluated,
        "skipped_out_of_domain_pairs": len(evaluation.observations) - evaluated,
        "label_source": evaluation.label_source,
    }


def _analytical_rmse(
    evaluation: EvaluationSet, model: MomentArmTeacherModel
) -> tuple[float | None, int]:
    predictions: list[np.ndarray] = []
    truth: list[np.ndarray] = []
    for observation, action, metadata in zip(
        evaluation.observations, evaluation.actions_mm, evaluation.metadata
    ):
        if metadata.get("in_moment_arm_domain", True):
            try:
                predictions.append(model.action_from_session_origin(observation[:3], observation[3:]))
                truth.append(action)
            except ValueError:
                continue
    if not predictions:
        return None, 0
    return (
        float(np.sqrt(np.mean((np.asarray(predictions) - np.asarray(truth)) ** 2))),
        len(predictions),
    )


def _mean_std(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values); return {"mean": float(array.mean()), "std": float(array.std())}


def _counts(values: Sequence[str] | Any) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values: result[str(value)] = result.get(str(value), 0) + 1
    return dict(sorted(result.items()))


def _write_flat_csv(runs: Sequence[Mapping[str, Any]], path: Path) -> None:
    rows: list[dict[str, Any]] = []
    for run in runs:
        for name, metrics in run["metrics"].items():
            rows.append({
                "condition": run["condition"], "seed": run["seed"],
                "output_mode": run["output_mode"], "real_pairs": run["real_pairs"],
                "physics_pairs": run["physics_pairs"], "physics_method": run["physics_method"],
                "evaluation": name, "mae_mm": metrics["mae_mm"], "rmse_mm": metrics["rmse_mm"],
                "unrepresentable_target_pairs": metrics["unrepresentable_target_pairs"],
            })
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0])); writer.writeheader(); writer.writerows(rows)


def make_plots(
    summary: Mapping[str, Any], runs: Sequence[Mapping[str, Any]],
    evaluations: Sequence[EvaluationSet], output_root: Path,
) -> None:
    curve_evals = [
        "eval_A_real_heldout_in_domain", "eval_A_independent_model_domain",
        "eval_A_independent_passive_reachable", "eval_B_trajectory_near",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(11, 8), constrained_layout=True)
    for axis, evaluation in zip(axes.flat, curve_evals):
        for label, prefix in (("C3", "curve_C3_real_"), ("C4 ABC", "curve_C4_ABC_real_")):
            points = []
            for condition, data in summary["conditions"].items():
                if condition.startswith(prefix):
                    points.append((data["real_pairs"], data["evaluations"][evaluation]["rmse_mm"]))
            points.sort()
            axis.errorbar(
                [p[0] for p in points], [p[1]["mean"] for p in points],
                yerr=[p[1]["std"] for p in points], marker="o", capsize=3, label=label,
            )
        axis.set_xscale("log"); axis.set_xlabel("recorded real FF teacher pairs")
        axis.set_ylabel("Action RMSE [mm]"); axis.set_title(evaluation); axis.legend()
    fig.savefig(output_root / "real_data_learning_curves.png", dpi=170); plt.close(fig)

    evaluation = "eval_A_independent_model_domain"
    method_conditions = [c for c in summary["conditions"] if c.startswith("method_C4_")]
    fig, axis = plt.subplots(figsize=(8, 4))
    labels = [c.replace("method_C4_", "").replace("_real_100", "") for c in method_conditions]
    means = [summary["conditions"][c]["evaluations"][evaluation]["rmse_mm"]["mean"] for c in method_conditions]
    stds = [summary["conditions"][c]["evaluations"][evaluation]["rmse_mm"]["std"] for c in method_conditions]
    axis.bar(labels, means, yerr=stds, capsize=4); axis.set_ylabel("independent model-domain RMSE [mm]")
    axis.set_title("Physics teacher method comparison (100 real pairs)")
    fig.tight_layout(); fig.savefig(output_root / "teacher_method_comparison.png", dpi=170); plt.close(fig)

    output_conditions = [
        "curve_C3_real_621", "curve_C4_ABC_real_621",
        "output_tanh_minmax_legacy_C3_real_621", "output_tanh_minmax_legacy_C4_ABC_real_621",
        "output_tanh_wide_standard_C3_real_621", "output_tanh_wide_standard_C4_ABC_real_621",
    ]
    fig, axis = plt.subplots(figsize=(11, 5))
    labels = [c.replace("output_", "").replace("curve_", "") for c in output_conditions]
    means = [summary["conditions"][c]["evaluations"][evaluation]["rmse_mm"]["mean"] for c in output_conditions]
    stds = [summary["conditions"][c]["evaluations"][evaluation]["rmse_mm"]["std"] for c in output_conditions]
    axis.bar(np.arange(len(labels)), means, yerr=stds, capsize=3)
    axis.set_xticks(np.arange(len(labels)), labels, rotation=25, ha="right")
    axis.set_ylabel("independent model-domain RMSE [mm]")
    axis.set_title("Output parameterization and range-limit effect")
    fig.tight_layout(); fig.savefig(output_root / "output_range_comparison.png", dpi=170); plt.close(fig)

    # C3/C4/analytical predictions from the same seed on the passive-reachable test.
    target_eval = next(e for e in evaluations if e.name == "eval_A_independent_passive_reachable")
    selected_runs = {
        run["condition"]: run for run in runs
        if run["seed"] == 7 and run["condition"] in ("curve_C3_real_621", "curve_C4_ABC_real_621")
    }
    predictions = {}
    for condition, run in selected_runs.items():
        file = Path(run["checkpoint"]).parent / f"predictions_{target_eval.name}.npz"
        with np.load(file) as data: predictions[condition] = data["prediction_mm"]
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for index, (axis, tendon) in enumerate(zip(axes.flat, TENDONS)):
        truth = target_eval.actions_mm[:, index]
        axis.scatter(truth, truth, marker="x", s=12, label="Analytical FF")
        for condition, prediction in predictions.items():
            axis.scatter(truth, prediction[:, index], s=6, alpha=0.35, label=condition)
        low, high = float(truth.min()), float(truth.max()); axis.plot([low, high], [low, high], color="black", lw=0.8)
        axis.set_title(tendon); axis.set_xlabel("Analytical FF [mm]"); axis.set_ylabel("Policy output [mm]")
    axes.flat[0].legend(fontsize=7)
    fig.savefig(output_root / "c3_c4_analytical_prediction_comparison.png", dpi=170); plt.close(fig)


def write_report(summary: Mapping[str, Any], path: Path) -> None:
    conditions = summary["conditions"]
    lines = [
        "# C3/C4 BC Version 2 Report", "",
        "This experiment measures moment-arm FF function approximation. It does not measure hardware tracking or online-RL experience efficiency.", "",
        "## Real-data learning curve", "",
        "| Real pairs | C3 model-domain | C4 model-domain | C3 passive-reachable | C4 passive-reachable | C3 real heldout | C4 real heldout |",
        "|---:|---:|---:|---:|---:|---:|---:|",
    ]
    for count in (50, 100, 200, 400, 621):
        c3 = conditions[f"curve_C3_real_{count}"]["evaluations"]
        c4 = conditions[f"curve_C4_ABC_real_{count}"]["evaluations"]
        values = []
        for key in (
            "eval_A_independent_model_domain",
            "eval_A_independent_passive_reachable",
            "eval_A_real_heldout_in_domain",
        ):
            values.extend([c3[key]["rmse_mm"]["mean"], c4[key]["rmse_mm"]["mean"]])
        lines.append(
            f"| {count} | " + " | ".join(f"{value:.6f}" for value in values) + " |"
        )
    lines.extend([
        "", "## Teacher method comparison at 100 real pairs", "",
        "| Method | Model-domain RMSE | Passive-reachable RMSE | Trajectory-near RMSE | Real heldout RMSE |",
        "|---|---:|---:|---:|---:|",
    ])
    for method in ("A", "B", "C", "ABC"):
        values = conditions[f"method_C4_{method}_real_100"]["evaluations"]
        lines.append(
            f"| {method} | {values['eval_A_independent_model_domain']['rmse_mm']['mean']:.6f} | "
            f"{values['eval_A_independent_passive_reachable']['rmse_mm']['mean']:.6f} | "
            f"{values['eval_B_trajectory_near']['rmse_mm']['mean']:.6f} | "
            f"{values['eval_A_real_heldout_in_domain']['rmse_mm']['mean']:.6f} |"
        )
    lines.extend([
        "", "## Output-range comparison at 621 real pairs", "",
        "| Condition | Model-domain RMSE | Passive-reachable RMSE | Family-OOD RMSE | Unrepresentable model-domain pairs |",
        "|---|---:|---:|---:|---:|",
    ])
    for condition in (
        "curve_C3_real_621", "curve_C4_ABC_real_621",
        "output_tanh_minmax_legacy_C3_real_621",
        "output_tanh_minmax_legacy_C4_ABC_real_621",
        "output_tanh_wide_standard_C3_real_621",
        "output_tanh_wide_standard_C4_ABC_real_621",
    ):
        values = conditions[condition]["evaluations"]
        lines.append(
            f"| {condition} | {values['eval_A_independent_model_domain']['rmse_mm']['mean']:.6f} | "
            f"{values['eval_A_independent_passive_reachable']['rmse_mm']['mean']:.6f} | "
            f"{values['eval_C_legacy_family_ood']['rmse_mm']['mean']:.6f} | "
            f"{values['eval_A_independent_model_domain']['unrepresentable_target_pairs']} |"
        )
    lines.extend([
        "", "Synthetic tests use the same analytical model as their labels; zero Analytical FF error is tautological, not independent hardware validation.",
        "The 6D BC is an FF approximator, not a measured-state feedback policy. Online-RL experience efficiency remains untested.", "",
    ])
    path.write_text("\n".join(lines), encoding="utf-8")


def rebuild_reports(
    output_dir: str | Path,
    moment_arm_dir: str | Path,
    config_path: str | Path,
) -> dict[str, Any]:
    """Rebuild aggregate artifacts from completed runs without retraining."""
    root = Path(output_dir)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    eval_report = json.loads(
        (root / "evaluation_datasets" / "evaluation_dataset_report.json").read_text(encoding="utf-8")
    )
    evaluations: list[EvaluationSet] = []
    for name, info in eval_report.items():
        with np.load(root / "evaluation_datasets" / f"{name}.npz") as data:
            obs, actions = data["observations"], data["actions_mm"]
        metadata = [
            json.loads(line) for line in
            (root / "evaluation_datasets" / f"{name}_metadata.jsonl").read_text(encoding="utf-8").splitlines()
            if line
        ]
        evaluations.append(EvaluationSet(name, obs, actions, metadata, info["label_source"]))
    runs: list[dict[str, Any]] = []
    for checkpoint_path in sorted((root / "runs").rglob("checkpoint.pt")):
        payload = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        metrics = json.loads((checkpoint_path.parent / "metrics.json").read_text(encoding="utf-8"))
        (checkpoint_path.parent / "scalers.json").write_text(
            json.dumps(
                {
                    "fit_source": "selected real training pairs only",
                    "observation_scaler": payload["observation_scaler"],
                    "action_transform": payload["action_transform"],
                    "safety_action_limits_embedded": False,
                },
                indent=2,
                allow_nan=False,
            ),
            encoding="utf-8",
        )
        runs.append({
            "condition": payload["condition"], "seed": payload["seed"],
            "output_mode": payload["output_mode"],
            "real_pairs": payload["real_training_pairs"],
            "physics_pairs": payload["physics_training_pairs"],
            "physics_method": payload["physics_method"],
            "metrics": metrics, "checkpoint": str(checkpoint_path),
        })
    summary = aggregate_results(
        runs, config, MomentArmTeacherModel.from_directory(moment_arm_dir), evaluations
    )
    (root / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    _write_flat_csv(runs, root / "evaluation_runs.csv")
    make_plots(summary, runs, evaluations, root)
    write_report(summary, root / "experiment_report.md")
    _write_independence_report(root, evaluations)
    return summary


def _write_independence_report(
    root: Path, evaluations: Sequence[EvaluationSet]
) -> None:
    teacher_root = root / "teacher_datasets"
    evaluation_by_name = {item.name: item for item in evaluations}

    def load_observations(path: Path) -> np.ndarray:
        with np.load(path) as data:
            return data["observations"]

    def exact_overlap(left: np.ndarray, right: np.ndarray) -> int:
        right_keys = {tuple(row) for row in np.round(right, 12)}
        return sum(tuple(row) in right_keys for row in np.round(left, 12))

    b_train = load_observations(teacher_root / "teacher_B_model_domain.npz")
    c_train = load_observations(teacher_root / "teacher_C_passive_reachable.npz")
    b_test = evaluation_by_name["eval_A_independent_model_domain"].observations
    c_test = evaluation_by_name["eval_A_independent_passive_reachable"].observations
    validation_sessions = {
        row["session_id"]
        for row in evaluation_by_name["eval_C_legacy_family_ood"].metadata
    }
    a_source_sessions: set[str] = set()
    for metadata_path in teacher_root.glob("teacher_A_*_metadata.jsonl"):
        for line in metadata_path.read_text(encoding="utf-8").splitlines():
            if line:
                row = json.loads(line)
                if row.get("source_session_id"):
                    a_source_sessions.add(str(row["source_session_id"]))
    passive_report = json.loads(
        (root / "passive_reachability_report.json").read_text(encoding="utf-8")
    )
    report = {
        "B_training_vs_independent_test_exact_input_overlap": exact_overlap(b_train, b_test),
        "C_training_vs_independent_test_exact_input_overlap": exact_overlap(c_train, c_test),
        "A_teacher_source_validation_session_intersection": sorted(
            a_source_sessions & validation_sessions
        ),
        "passive_model_validation_family_intersection": sorted(
            set(passive_report["families"]) & {"Edinburgh", "FullExtension"}
        ),
        "interpretation": (
            "Zero exact overlap does not make analytical tests independent hardware evidence; "
            "their labels still come from the same moment-arm model."
        ),
    }
    (root / "dataset_independence_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False), encoding="utf-8"
    )


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run version-2 C3/C4 BC study")
    parser.add_argument("dataset_dir", type=Path); parser.add_argument("moment_arm_dir", type=Path)
    parser.add_argument("passive_motion_dir", type=Path); parser.add_argument("config", type=Path)
    parser.add_argument("output_dir", type=Path); return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_v2_experiment(
        args.dataset_dir, args.moment_arm_dir, args.passive_motion_dir,
        args.config, args.output_dir,
    )
    print(json.dumps({"run_count": summary["run_count"]}, indent=2))


if __name__ == "__main__":
    main()
