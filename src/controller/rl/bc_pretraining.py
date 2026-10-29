"""Reproducible C3/C4 offline behavior-cloning experiment.

This model approximates the recorded feed-forward command map
``[q0_plan, q_target] -> absolute target tendon excursion``.  It is not a
feedback policy over measured posture or tension and does not measure robot
joint-angle tracking performance.
"""

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

from controller.rl.legacy_bc_dataset import (
    ACTION_FEATURES,
    COMMON6_FEATURES,
    PairTable,
    load_pair_table,
)
from controller.rl.physics_teacher import JOINTS, MomentArmTeacherModel
from controller.servo_mapping import TENDONS


@dataclass(frozen=True)
class StandardScalerState:
    mean: tuple[float, ...]
    std: tuple[float, ...]

    @classmethod
    def fit(cls, values: np.ndarray) -> "StandardScalerState":
        mean = np.mean(values, axis=0)
        std = np.std(values, axis=0)
        std[std < 1e-12] = 1.0
        return cls(tuple(mean.tolist()), tuple(std.tolist()))

    def transform(self, values: np.ndarray) -> np.ndarray:
        return (values - np.asarray(self.mean)) / np.asarray(self.std)

    def inverse(self, values: np.ndarray) -> np.ndarray:
        return values * np.asarray(self.std) + np.asarray(self.mean)


@dataclass(frozen=True)
class MinMaxActionScalerState:
    minimum: tuple[float, ...]
    maximum: tuple[float, ...]

    @classmethod
    def fit(cls, values: np.ndarray) -> "MinMaxActionScalerState":
        minimum = np.min(values, axis=0)
        maximum = np.max(values, axis=0)
        if np.any(maximum - minimum < 1e-12):
            raise ValueError("Action scaler has a constant tendon dimension")
        return cls(tuple(minimum.tolist()), tuple(maximum.tolist()))

    def transform(self, values: np.ndarray) -> np.ndarray:
        low, high = np.asarray(self.minimum), np.asarray(self.maximum)
        return 2.0 * (values - low) / (high - low) - 1.0

    def inverse(self, values: np.ndarray) -> np.ndarray:
        low, high = np.asarray(self.minimum), np.asarray(self.maximum)
        return low + 0.5 * (values + 1.0) * (high - low)


class CompatibleBCActor(nn.Module):
    """MLP whose parameters map directly to d3rlpy VectorEncoder + mean head."""

    def __init__(self, observation_dim: int = 6, hidden_units: Sequence[int] = (128, 128)) -> None:
        super().__init__()
        if list(hidden_units) != [128, 128]:
            raise ValueError("This experiment requires hidden_units=[128, 128]")
        self.encoder = nn.Sequential(
            nn.Linear(observation_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 128),
            nn.ReLU(),
        )
        self.mean = nn.Linear(128, len(TENDONS))

    def forward(self, observations: torch.Tensor) -> torch.Tensor:
        return torch.tanh(self.mean(self.encoder(observations)))


@dataclass
class SyntheticTeacherPool:
    observations: np.ndarray
    actions_mm: np.ndarray
    metadata: list[dict[str, Any]]
    report: dict[str, Any]

    def save(self, output_dir: str | Path) -> dict[str, Path]:
        root = Path(output_dir)
        root.mkdir(parents=True, exist_ok=True)
        data_path = root / "moment_arm_teacher_train_pool.npz"
        np.savez_compressed(
            data_path,
            observations=self.observations,
            actions_mm=self.actions_mm,
            feature_names=np.asarray(COMMON6_FEATURES),
            action_names=np.asarray(ACTION_FEATURES),
        )
        metadata_path = root / "moment_arm_teacher_train_pool_metadata.jsonl"
        with metadata_path.open("w", encoding="utf-8", newline="\n") as handle:
            for row in self.metadata:
                handle.write(json.dumps(row, ensure_ascii=False, allow_nan=False) + "\n")
        report_path = root / "moment_arm_teacher_train_pool_report.json"
        report_path.write_text(
            json.dumps(self.report, indent=2, ensure_ascii=False, allow_nan=False),
            encoding="utf-8",
        )
        return {"data": data_path, "metadata": metadata_path, "report": report_path}


def generate_synthetic_teacher_pool(
    train: PairTable,
    model: MomentArmTeacherModel,
    count: int,
    config: Mapping[str, Any],
) -> SyntheticTeacherPool:
    """Sample only convex segments observed inside each training session."""

    rng = np.random.default_rng(int(config["random_seed"]))
    alpha_low, alpha_high = map(float, config["interpolation_alpha_range"])
    minimum_novelty_deg = float(config["minimum_novelty_deg"])
    max_attempts = int(config["maximum_attempt_multiplier"]) * max(count, 1)
    eligible: dict[str, list[int]] = {}
    for index, row in enumerate(train.metadata):
        if row["split"] == "training" and row["in_moment_arm_domain"]:
            eligible.setdefault(row["session_id"], []).append(index)
    eligible = {key: indices for key, indices in eligible.items() if len(indices) >= 2}
    if not eligible:
        raise ValueError("No in-domain training session has two teacher pairs")
    session_ids = sorted(eligible)
    existing_deg = np.rad2deg(train.observations)
    generated: list[np.ndarray] = []
    actions: list[np.ndarray] = []
    metadata: list[dict[str, Any]] = []
    nearest_distances: list[float] = []
    attempts = 0
    while len(generated) < count and attempts < max_attempts:
        attempts += 1
        session_id = session_ids[int(rng.integers(len(session_ids)))]
        indices = eligible[session_id]
        first, second = rng.choice(indices, size=2, replace=False).tolist()
        alpha = float(rng.uniform(alpha_low, alpha_high))
        q0 = train.observations[first, :3].copy()
        if not np.allclose(q0, train.observations[second, :3], atol=1e-12, rtol=0.0):
            continue
        target = (
            alpha * train.observations[first, 3:]
            + (1.0 - alpha) * train.observations[second, 3:]
        )
        observation = np.concatenate([q0, target])
        nearest = float(np.min(np.linalg.norm(existing_deg - np.rad2deg(observation), axis=1)))
        if nearest < minimum_novelty_deg:
            continue
        if generated:
            generated_deg = np.rad2deg(np.asarray(generated))
            if float(np.min(np.linalg.norm(generated_deg - np.rad2deg(observation), axis=1))) < 1e-8:
                continue
        try:
            action = model.action_from_session_origin(q0, target)
        except ValueError:
            continue
        parent = train.metadata[first]
        generated.append(observation)
        actions.append(action)
        nearest_distances.append(nearest)
        metadata.append(
            {
                "teacher_pair_id": f"physics:{len(generated)-1:06d}",
                "source_kind": "moment_arm_synthetic",
                "source_session_id": session_id,
                "trajectory_family": parent["trajectory_family"],
                "parent_pair_ids": [
                    train.metadata[first]["pair_id"],
                    train.metadata[second]["pair_id"],
                ],
                "q0_plan_rad": q0.tolist(),
                "target_posture_rad": target.tolist(),
                "interpolation_alpha": alpha,
                "sampling_method": config["sampling_method"],
                "integration_path": "analytic separable joint integral from q0_plan to interpolated target",
                "nearest_real_training_observation_distance_deg": nearest,
                "in_moment_arm_domain": True,
                "contains_measured_tension": False,
                "contains_next_state": False,
                "contains_reward": False,
                "action_reference": "session_plan_origin",
                "action_is_incremental": False,
                "model_version": model.provenance(),
            }
        )
    if len(generated) != count:
        raise RuntimeError(f"Generated {len(generated)} of {count} requested teacher pairs")
    distances = np.asarray(nearest_distances)
    report = {
        "requested_pairs": count,
        "generated_pairs": len(generated),
        "attempts": attempts,
        "eligible_training_sessions": session_ids,
        "validation_pairs_used": 0,
        "out_of_domain_pairs": 0,
        "duplicate_rate_at_minimum_novelty": 0.0,
        "minimum_novelty_deg": minimum_novelty_deg,
        "nearest_real_distance_deg": {
            "min": float(distances.min()),
            "median": float(np.median(distances)),
            "max": float(distances.max()),
        },
        "pairs_by_family": _string_counts(row["trajectory_family"] for row in metadata),
        "observation_range_rad": {
            name: [float(np.min(np.asarray(generated)[:, index])), float(np.max(np.asarray(generated)[:, index]))]
            for index, name in enumerate(COMMON6_FEATURES)
        },
        "action_range_mm": {
            name: [float(np.min(np.asarray(actions)[:, index])), float(np.max(np.asarray(actions)[:, index]))]
            for index, name in enumerate(ACTION_FEATURES)
        },
        "sampling_config": dict(config),
        "model": model.provenance(),
    }
    return SyntheticTeacherPool(np.asarray(generated), np.asarray(actions), metadata, report)


def _string_counts(values: Sequence[str] | Any) -> dict[str, int]:
    counts: dict[str, int] = {}
    for value in values:
        key = str(value)
        counts[key] = counts.get(key, 0) + 1
    return dict(sorted(counts.items()))


def _set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
    torch.use_deterministic_algorithms(True)


def _predict_mm(
    actor: CompatibleBCActor,
    observations: np.ndarray,
    observation_scaler: StandardScalerState,
    action_scaler: MinMaxActionScalerState,
) -> np.ndarray:
    actor.eval()
    with torch.no_grad():
        tensor = torch.as_tensor(
            observation_scaler.transform(observations), dtype=torch.float32
        )
        normalized = actor(tensor).cpu().numpy()
    return action_scaler.inverse(normalized)


def _error_metrics(
    truth: np.ndarray,
    prediction: np.ndarray,
    metadata: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    error = prediction - truth
    absolute = np.abs(error)
    squared = error**2
    result: dict[str, Any] = {
        "pairs": len(truth),
        "mae_mm": float(np.mean(absolute)),
        "rmse_mm": float(np.sqrt(np.mean(squared))),
        "per_tendon": {
            tendon: {
                "mae_mm": float(np.mean(absolute[:, index])),
                "rmse_mm": float(np.sqrt(np.mean(squared[:, index]))),
            }
            for index, tendon in enumerate(TENDONS)
        },
    }
    for group_name, key in (("per_family", "trajectory_family"),):
        result[group_name] = {}
        for value in sorted({str(row[key]) for row in metadata}):
            mask = np.asarray([str(row[key]) == value for row in metadata])
            result[group_name][value] = {
                "pairs": int(mask.sum()),
                "mae_mm": float(np.mean(absolute[mask])),
                "rmse_mm": float(np.sqrt(np.mean(squared[mask]))),
            }
    result["by_moment_arm_domain"] = {}
    for label, expected in (("inside", True), ("outside", False)):
        mask = np.asarray([bool(row["in_moment_arm_domain"]) == expected for row in metadata])
        if mask.any():
            result["by_moment_arm_domain"][label] = {
                "pairs": int(mask.sum()),
                "mae_mm": float(np.mean(absolute[mask])),
                "rmse_mm": float(np.sqrt(np.mean(squared[mask]))),
            }
    return result


def train_condition(
    condition: str,
    seed: int,
    real_train: PairTable,
    validation: PairTable,
    synthetic_observations: np.ndarray,
    synthetic_actions: np.ndarray,
    synthetic_weight: float,
    config: Mapping[str, Any],
    observation_scaler: StandardScalerState,
    action_scaler: MinMaxActionScalerState,
    output_dir: Path,
) -> tuple[dict[str, Any], np.ndarray]:
    _set_seed(seed)
    actor = CompatibleBCActor()
    train_x = np.concatenate([real_train.observations, synthetic_observations])
    train_y = np.concatenate([real_train.actions_mm, synthetic_actions])
    sample_weights = np.concatenate(
        [
            np.ones(len(real_train.observations)),
            np.full(len(synthetic_observations), synthetic_weight),
        ]
    )
    x = torch.as_tensor(observation_scaler.transform(train_x), dtype=torch.float32)
    y = torch.as_tensor(action_scaler.transform(train_y), dtype=torch.float32)
    weights = torch.as_tensor(sample_weights, dtype=torch.float32)
    optimizer = torch.optim.Adam(actor.parameters(), lr=float(config["learning_rate"]))
    generator = torch.Generator().manual_seed(seed + 100_000)
    history: list[dict[str, float]] = []
    batch_size = int(config["batch_size"])
    updates = int(config["gradient_updates"])
    interval = int(config["evaluation_interval"])
    for update in range(1, updates + 1):
        indices = torch.randint(len(x), (batch_size,), generator=generator)
        prediction = actor(x[indices])
        per_sample = torch.mean((prediction - y[indices]) ** 2, dim=1)
        batch_weights = weights[indices]
        loss = torch.sum(batch_weights * per_sample) / torch.sum(batch_weights)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()
        if update == 1 or update % interval == 0 or update == updates:
            validation_prediction = _predict_mm(
                actor, validation.observations, observation_scaler, action_scaler
            )
            history.append(
                {
                    "update": update,
                    "train_normalized_mse": float(loss.detach()),
                    "validation_rmse_mm": float(
                        np.sqrt(np.mean((validation_prediction - validation.actions_mm) ** 2))
                    ),
                }
            )
    validation_prediction = _predict_mm(
        actor, validation.observations, observation_scaler, action_scaler
    )
    metrics = _error_metrics(validation.actions_mm, validation_prediction, validation.metadata)
    run_dir = output_dir / condition / f"seed_{seed}"
    run_dir.mkdir(parents=True, exist_ok=True)
    checkpoint = {
        "schema_version": 1,
        "condition": condition,
        "seed": seed,
        "model_class": "CompatibleBCActor",
        "architecture": {"input": 6, "hidden": [128, 128], "activation": "relu", "output": 6, "output_activation": "tanh"},
        "state_dict": actor.state_dict(),
        "observation_scaler": asdict(observation_scaler),
        "action_scaler": asdict(action_scaler),
        "observation_features": list(COMMON6_FEATURES),
        "action_features": list(ACTION_FEATURES),
        "action_reference": "session_plan_origin",
        "q0_semantics": "q0_plan_not_q0_measured",
        "training_config": dict(config),
        "real_training_pairs": len(real_train.observations),
        "synthetic_training_pairs": len(synthetic_observations),
        "metrics": metrics,
    }
    torch.save(checkpoint, run_dir / "checkpoint.pt")
    (run_dir / "metrics.json").write_text(
        json.dumps(metrics, indent=2, allow_nan=False), encoding="utf-8"
    )
    with (run_dir / "loss_curve.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(history[0]))
        writer.writeheader()
        writer.writerows(history)
    _plot_loss(history, run_dir / "loss_curve.png", condition, seed)
    np.savez_compressed(
        run_dir / "validation_predictions.npz",
        prediction_mm=validation_prediction,
        teacher_mm=validation.actions_mm,
    )
    return metrics, validation_prediction


def _plot_loss(history: Sequence[Mapping[str, float]], path: Path, condition: str, seed: int) -> None:
    fig, left = plt.subplots(figsize=(7, 4))
    updates = [row["update"] for row in history]
    left.plot(updates, [row["train_normalized_mse"] for row in history], label="train normalized MSE")
    left.set_xlabel("gradient update")
    left.set_ylabel("train normalized MSE")
    right = left.twinx()
    right.plot(updates, [row["validation_rmse_mm"] for row in history], color="tab:orange", label="validation RMSE")
    right.set_ylabel("validation RMSE [mm]")
    left.set_title(f"{condition}, seed={seed}")
    fig.tight_layout()
    fig.savefig(path, dpi=160)
    plt.close(fig)


def _direct_physics_baseline(validation: PairTable, model: MomentArmTeacherModel) -> tuple[dict[str, Any], np.ndarray]:
    predictions = np.full_like(validation.actions_mm, np.nan)
    mask = np.asarray([row["in_moment_arm_domain"] for row in validation.metadata], dtype=bool)
    for index in np.flatnonzero(mask):
        predictions[index] = model.action_from_session_origin(
            validation.observations[index, :3], validation.observations[index, 3:]
        )
    metrics = _error_metrics(
        validation.actions_mm[mask], predictions[mask],
        [row for row, keep in zip(validation.metadata, mask) if keep],
    )
    metrics["coverage"] = {"inside_pairs": int(mask.sum()), "outside_pairs_unavailable": int((~mask).sum())}
    return metrics, predictions


def _plot_prediction_comparison(
    validation: PairTable,
    condition_predictions: Mapping[str, np.ndarray],
    physics_prediction: np.ndarray,
    path: Path,
) -> None:
    fig, axes = plt.subplots(2, 3, figsize=(13, 8), constrained_layout=True)
    for tendon_index, (axis, tendon) in enumerate(zip(axes.flat, TENDONS)):
        truth = validation.actions_mm[:, tendon_index]
        for label, prediction in condition_predictions.items():
            axis.scatter(truth, prediction[:, tendon_index], s=9, alpha=0.5, label=label)
        mask = np.isfinite(physics_prediction[:, tendon_index])
        axis.scatter(truth[mask], physics_prediction[mask, tendon_index], s=12, marker="x", label="direct physics")
        low = min(float(np.min(truth)), *(float(np.nanmin(p[:, tendon_index])) for p in condition_predictions.values()))
        high = max(float(np.max(truth)), *(float(np.nanmax(p[:, tendon_index])) for p in condition_predictions.values()))
        axis.plot([low, high], [low, high], color="black", linewidth=0.8)
        axis.set_title(tendon)
        axis.set_xlabel("recorded FF teacher [mm]")
        axis.set_ylabel("prediction [mm]")
    axes.flat[0].legend(fontsize=7)
    fig.savefig(path, dpi=170)
    plt.close(fig)


def run_experiment(
    dataset_dir: str | Path,
    moment_arm_dir: str | Path,
    config_path: str | Path,
    output_dir: str | Path,
) -> dict[str, Any]:
    dataset_root, output_root = Path(dataset_dir), Path(output_dir)
    config = json.loads(Path(config_path).read_text(encoding="utf-8"))
    table = load_pair_table(
        dataset_root / "bc_common6_completed.npz",
        dataset_root / "bc_common6_completed_metadata.jsonl",
    )
    train = table.subset([row["split"] == "training" for row in table.metadata])
    validation = table.subset([row["split"] == "validation" for row in table.metadata])
    if len(train.metadata) != 621 or len(validation.metadata) != 155:
        raise ValueError(f"Expected 621/155 split, got {len(train.metadata)}/{len(validation.metadata)}")
    model = MomentArmTeacherModel.from_directory(moment_arm_dir)
    observation_scaler = StandardScalerState.fit(train.observations)
    action_scaler = MinMaxActionScalerState.fit(train.actions_mm)
    maximum_ratio = max(float(v) for v in config["augmentation"]["ratios"])
    pool_size = int(math.ceil(maximum_ratio * len(train.observations)))
    pool = generate_synthetic_teacher_pool(train, model, pool_size, config["augmentation"])
    output_root.mkdir(parents=True, exist_ok=True)
    pool.save(output_root / "teacher_dataset")
    scaler_payload = {
        "fit_source": "real_training_pairs_only",
        "training_pairs": len(train.metadata),
        "observation": asdict(observation_scaler),
        "action": asdict(action_scaler),
    }
    (output_root / "scalers.json").write_text(
        json.dumps(scaler_payload, indent=2, allow_nan=False), encoding="utf-8"
    )
    (output_root / "resolved_config.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )

    repeat_count = int(round(float(config["augmentation"]["c3_repeat_ratio"]) * len(train.observations)))
    conditions: dict[str, tuple[np.ndarray, np.ndarray, float]] = {
        "C3_real_only": (np.empty((0, 6)), np.empty((0, 6)), 1.0),
        "C3_real_repeat_control": (
            train.observations[:repeat_count % len(train.observations)] if repeat_count < len(train.observations) else np.tile(train.observations, (int(math.ceil(repeat_count / len(train.observations))), 1))[:repeat_count],
            train.actions_mm[:repeat_count % len(train.actions_mm)] if repeat_count < len(train.actions_mm) else np.tile(train.actions_mm, (int(math.ceil(repeat_count / len(train.actions_mm))), 1))[:repeat_count],
            1.0,
        ),
    }
    for ratio in config["augmentation"]["ratios"]:
        count = int(round(float(ratio) * len(train.observations)))
        conditions[f"C4_physics_{float(ratio):g}"] = (
            pool.observations[:count], pool.actions_mm[:count],
            float(config["augmentation"]["physics_loss_weight"]),
        )

    rows: list[dict[str, Any]] = []
    detailed_metrics: dict[str, list[dict[str, Any]]] = {
        condition: [] for condition in conditions
    }
    predictions: dict[tuple[str, int], np.ndarray] = {}
    for condition, (synthetic_x, synthetic_y, weight) in conditions.items():
        for seed in config["training"]["seeds"]:
            metrics, prediction = train_condition(
                condition, int(seed), train, validation, synthetic_x, synthetic_y,
                weight, config["training"], observation_scaler, action_scaler, output_root,
            )
            predictions[(condition, int(seed))] = prediction
            detailed_metrics[condition].append(metrics)
            rows.append(
                {
                    "condition": condition,
                    "seed": int(seed),
                    "real_pairs": len(train.observations),
                    "additional_pairs": len(synthetic_x),
                    "validation_mae_mm": metrics["mae_mm"],
                    "validation_rmse_mm": metrics["rmse_mm"],
                    "validation_inside_rmse_mm": metrics["by_moment_arm_domain"]["inside"]["rmse_mm"],
                    "validation_outside_rmse_mm": metrics["by_moment_arm_domain"]["outside"]["rmse_mm"],
                }
            )
    physics_metrics, physics_prediction = _direct_physics_baseline(validation, model)
    summaries: dict[str, Any] = {}
    for condition in conditions:
        selected = [row for row in rows if row["condition"] == condition]
        summaries[condition] = {
            key: {"mean": float(np.mean([r[key] for r in selected])), "std": float(np.std([r[key] for r in selected]))}
            for key in ("validation_mae_mm", "validation_rmse_mm", "validation_inside_rmse_mm", "validation_outside_rmse_mm")
        }
        summaries[condition]["per_tendon"] = {
            tendon: {
                metric: _mean_std(
                    [run["per_tendon"][tendon][metric] for run in detailed_metrics[condition]]
                )
                for metric in ("mae_mm", "rmse_mm")
            }
            for tendon in TENDONS
        }
        families = sorted(detailed_metrics[condition][0]["per_family"])
        summaries[condition]["per_family"] = {
            family: {
                metric: _mean_std(
                    [run["per_family"][family][metric] for run in detailed_metrics[condition]]
                )
                for metric in ("mae_mm", "rmse_mm")
            }
            for family in families
        }
    c3_rmse = summaries["C3_real_only"]["validation_rmse_mm"]["mean"]
    for condition, values in summaries.items():
        values["rmse_change_vs_C3_percent"] = float(
            100.0 * (values["validation_rmse_mm"]["mean"] - c3_rmse) / c3_rmse
        )
    dataset_report_path = dataset_root / "bc_dataset_report.json"
    dataset_quality = (
        json.loads(dataset_report_path.read_text(encoding="utf-8"))
        if dataset_report_path.is_file()
        else None
    )
    summary = {
        "interpretation": (
            "C3 labels already originate from moment-arm FF. This experiment tests whether "
            "additional synthetic moment-arm pairs improve FF-command approximation; it does "
            "not test hardware joint tracking or online-RL experience efficiency."
        ),
        "split": {"training_pairs": len(train.metadata), "validation_pairs": len(validation.metadata), "validation_families": ["Edinburgh", "FullExtension"]},
        "conditions": summaries,
        "per_run": rows,
        "direct_physics_in_domain_reference": physics_metrics,
        "teacher_pool": pool.report,
        "dataset_quality": dataset_quality,
    }
    (output_root / "evaluation_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    with (output_root / "evaluation_table.csv").open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    mean_predictions = {
        condition: np.mean([predictions[(condition, int(seed))] for seed in config["training"]["seeds"]], axis=0)
        for condition in ("C3_real_only", f"C4_physics_{float(config['augmentation']['primary_ratio']):g}")
    }
    _plot_prediction_comparison(validation, mean_predictions, physics_prediction, output_root / "c3_c4_prediction_comparison.png")
    _plot_augmentation_curve(rows, output_root / "augmentation_validation_curve.png")
    _write_markdown_report(summary, output_root / "experiment_report.md")
    return summary


def _mean_std(values: Sequence[float]) -> dict[str, float]:
    array = np.asarray(values, dtype=np.float64)
    return {"mean": float(np.mean(array)), "std": float(np.std(array))}


def _write_markdown_report(summary: Mapping[str, Any], path: Path) -> None:
    split = summary["split"]
    lines = [
        "# C3/C4 Offline BC Preliminary Experiment",
        "",
        "This is an FF-command function-approximation experiment. It is not an evaluation of robot joint-angle tracking, feedback control, or online-RL experience efficiency.",
        "",
        f"The fixed family/session split contains {split['training_pairs']} training pairs and {split['validation_pairs']} validation pairs. Validation families are Edinburgh and FullExtension.",
        "",
        "| Condition | Validation MAE [mm] | Validation RMSE [mm] | In-domain RMSE [mm] | Out-of-domain RMSE [mm] | RMSE change vs C3 |",
        "|---|---:|---:|---:|---:|---:|",
    ]
    for condition, values in summary["conditions"].items():
        lines.append(
            f"| {condition} | {values['validation_mae_mm']['mean']:.6f} ± {values['validation_mae_mm']['std']:.6f} | "
            f"{values['validation_rmse_mm']['mean']:.6f} ± {values['validation_rmse_mm']['std']:.6f} | "
            f"{values['validation_inside_rmse_mm']['mean']:.6f} | {values['validation_outside_rmse_mm']['mean']:.6f} | "
            f"{values['rmse_change_vs_C3_percent']:+.3f}% |"
        )
    direct = summary["direct_physics_in_domain_reference"]
    lines.extend(
        [
            "",
            "The direct moment-arm reference is defined only for the 46 in-domain validation pairs; 109 validation pairs are outside the validated model range and are not extrapolated.",
            f"Its in-domain RMSE against recorded FF actions is {direct['rmse_mm']:.9f} mm.",
            "",
            "C3 labels were themselves generated by the moment-arm FF pipeline. Therefore C3 is not a physics-free controller. C4 tests only the marginal effect of additional synthetic pairs.",
            "",
        ]
    )
    path.write_text("\n".join(lines), encoding="utf-8")


def _plot_augmentation_curve(rows: Sequence[Mapping[str, Any]], path: Path) -> None:
    ratios, means, stds = [], [], []
    conditions = sorted({str(r["condition"]) for r in rows if str(r["condition"]).startswith("C4_physics_")}, key=lambda name: float(name.rsplit("_", 1)[1]))
    for condition in conditions:
        values = [float(r["validation_rmse_mm"]) for r in rows if r["condition"] == condition]
        ratios.append(float(condition.rsplit("_", 1)[1])); means.append(float(np.mean(values))); stds.append(float(np.std(values)))
    c3 = [float(r["validation_rmse_mm"]) for r in rows if r["condition"] == "C3_real_only"]
    fig, axis = plt.subplots(figsize=(6, 4))
    axis.errorbar(ratios, means, yerr=stds, marker="o", capsize=4, label="C4 physics augmentation")
    axis.axhline(np.mean(c3), color="tab:orange", linestyle="--", label="C3 mean")
    axis.set_xlabel("synthetic / real training pair ratio")
    axis.set_ylabel("validation RMSE [mm]")
    axis.legend(); fig.tight_layout(); fig.savefig(path, dpi=170); plt.close(fig)


def load_bc_checkpoint(path: str | Path) -> tuple[CompatibleBCActor, StandardScalerState, MinMaxActionScalerState, dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    actor = CompatibleBCActor(observation_dim=int(payload["architecture"]["input"]))
    actor.load_state_dict(payload["state_dict"])
    actor.eval()
    obs = StandardScalerState(**{k: tuple(v) for k, v in payload["observation_scaler"].items()})
    action = MinMaxActionScalerState(**{k: tuple(v) for k, v in payload["action_scaler"].items()})
    return actor, obs, action, payload


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Run C3/C4 offline BC pretraining")
    parser.add_argument("dataset_dir", type=Path)
    parser.add_argument("moment_arm_directory", type=Path)
    parser.add_argument("config", type=Path)
    parser.add_argument("output_dir", type=Path)
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    summary = run_experiment(args.dataset_dir, args.moment_arm_directory, args.config, args.output_dir)
    print(json.dumps(summary["conditions"], indent=2))


if __name__ == "__main__":
    main()
