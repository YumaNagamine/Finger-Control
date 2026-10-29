"""Run a non-hardware d3rlpy AWAC smoke trial from C3/C4 BC actors."""

from __future__ import annotations

import argparse
import json
import random
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from controller.rl.awac_transfer_v2 import load_v2_checkpoint
from controller.rl.bc_pretraining_v2 import ActionTransform
from controller.rl.legacy_awac_dataset import load_legacy_awac_dataset
from controller.servo_mapping import TENDONS


TRIAL_SCHEMA_VERSION = 1
BC_FEATURE_INDICES = tuple(range(6))


def build_replay_buffer(data: Mapping[str, np.ndarray]):
    """Represent each reliable legacy interval as an explicit one-step fragment."""
    from d3rlpy.constants import ActionSpace
    from d3rlpy.dataset import Episode, InfiniteBuffer, ReplayBuffer

    episodes = []
    for obs, action, reward, next_obs in zip(
        data["observations"], data["actions_mm"], data["rewards"], data["next_observations"]
    ):
        episodes.append(
            Episode(
                observations=np.stack([obs, next_obs]).astype(np.float32),
                actions=np.stack([action, action]).astype(np.float32),
                rewards=np.asarray([[float(reward[0])], [0.0]], dtype=np.float32),
                terminated=False,
            )
        )
    return ReplayBuffer(
        InfiniteBuffer(),
        episodes=episodes,
        action_space=ActionSpace.CONTINUOUS,
        action_size=6,
    )


def _checkpoint_scalers(checkpoint: Path) -> tuple[dict[str, Any], ActionTransform]:
    _, payload = load_v2_checkpoint(checkpoint)
    transform = ActionTransform(
        mode=payload["action_transform"]["mode"],
        center=tuple(payload["action_transform"]["center"]),
        scale=tuple(payload["action_transform"]["scale"]),
    )
    return payload["observation_scaler"], transform


def build_initialized_awac(
    checkpoint: Path,
    observations: np.ndarray,
    next_observations: np.ndarray,
    *,
    batch_size: int,
    seed: int,
):
    import d3rlpy

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    d3rlpy.seed(seed)
    actor, payload = load_v2_checkpoint(checkpoint)
    if payload["output_mode"] != "tanh_wide_standard":
        raise ValueError("Offline AWAC trial requires the validated tanh_wide_standard companion checkpoint")
    obs_state, transform = _checkpoint_scalers(checkpoint)
    all_observations = np.concatenate([observations, next_observations], axis=0).astype(np.float64)
    full_mean = np.mean(all_observations, axis=0)
    full_std = np.std(all_observations, axis=0)
    full_std[full_std < 1e-8] = 1.0
    full_mean[list(BC_FEATURE_INDICES)] = np.asarray(obs_state["mean"])
    full_std[list(BC_FEATURE_INDICES)] = np.asarray(obs_state["std"])
    center = np.asarray(transform.center, dtype=np.float32)
    scale = np.asarray(transform.scale, dtype=np.float32)
    encoder = d3rlpy.models.VectorEncoderFactory(hidden_units=[128, 128], activation="relu")
    algo = d3rlpy.algos.AWACConfig(
        batch_size=batch_size,
        actor_learning_rate=3e-4,
        critic_learning_rate=3e-4,
        actor_encoder_factory=encoder,
        critic_encoder_factory=encoder,
        observation_scaler=d3rlpy.preprocessing.StandardObservationScaler(
            mean=full_mean.astype(np.float32), std=full_std.astype(np.float32), eps=0.0
        ),
        action_scaler=d3rlpy.preprocessing.MinMaxActionScaler(
            minimum=center - scale, maximum=center + scale
        ),
    ).create(device=False)
    algo.create_impl((observations.shape[1],), 6)
    policy = algo.impl.policy
    state = policy.state_dict()
    source = actor.state_dict()
    with torch.no_grad():
        first = torch.zeros_like(state["_encoder._layers.0.weight"])
        first[:, list(BC_FEATURE_INDICES)] = source["encoder.0.weight"]
        state["_encoder._layers.0.weight"] = first
        state["_encoder._layers.0.bias"] = source["encoder.0.bias"].clone()
        state["_encoder._layers.2.weight"] = source["encoder.2.weight"].clone()
        state["_encoder._layers.2.bias"] = source["encoder.2.bias"].clone()
        state["_mu.weight"] = source["mean.weight"].clone()
        state["_mu.bias"] = source["mean.bias"].clone()
        policy.load_state_dict(state, strict=True)

    common = observations[:, :6].astype(np.float32)
    scaled = (common - np.asarray(obs_state["mean"], dtype=np.float32)) / np.asarray(
        obs_state["std"], dtype=np.float32
    )
    with torch.no_grad():
        bc_network = actor(torch.as_tensor(scaled, dtype=torch.float32)).numpy()
    bc_mm = transform.decode(bc_network)
    awac_mm = algo.predict(observations.astype(np.float32))
    difference = np.abs(bc_mm - awac_mm)
    transfer = {
        "compatible": bool(np.max(difference) <= 1e-5),
        "max_absolute_difference_mm": float(np.max(difference)),
        "mean_absolute_difference_mm": float(np.mean(difference)),
        "output_mode": payload["output_mode"],
        "bc_condition": payload["condition"],
        "copied": ["selected first-layer columns", "second encoder layer", "Gaussian mean head"],
        "not_copied": ["Gaussian logstd", "critics", "target critics", "optimizers"],
        "extra_observation_columns_zero_initialized": True,
    }
    return algo, transfer, {
        "mean": full_mean.tolist(),
        "std": full_std.tolist(),
        "fit_source": "offline dataset observations and next_observations; BC six columns retained from BC training",
    }


def _finite_summary(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64)
    return {
        "all_finite": bool(np.isfinite(array).all()),
        "count": int(array.size),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
    }


def _action_summary(actions: np.ndarray) -> dict[str, Any]:
    return {
        tendon: _finite_summary(actions[:, index])
        for index, tendon in enumerate(TENDONS)
    }


def _action_error_summary(predicted: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    error = np.asarray(predicted) - np.asarray(target)
    return {
        "overall_mae_mm": float(np.mean(np.abs(error))),
        "overall_rmse_mm": float(np.sqrt(np.mean(error * error))),
        "per_tendon_rmse_mm": {
            tendon: float(np.sqrt(np.mean(error[:, index] ** 2)))
            for index, tendon in enumerate(TENDONS)
        },
    }


def run_condition(
    name: str,
    checkpoint: Path,
    data: Mapping[str, np.ndarray],
    replay,
    output_dir: Path,
    *,
    n_steps: int,
    steps_per_epoch: int,
    batch_size: int,
    seed: int,
) -> dict[str, Any]:
    from d3rlpy.logging import FileAdapterFactory

    condition_dir = output_dir / name
    condition_dir.mkdir(parents=True, exist_ok=True)
    algo, transfer, observation_scaler = build_initialized_awac(
        checkpoint,
        data["observations"],
        data["next_observations"],
        batch_size=batch_size,
        seed=seed,
    )
    initial_actions = algo.predict(data["observations"])
    initial_q = algo.predict_value(data["observations"], data["actions_mm"])
    history = algo.fit(
        replay,
        n_steps=n_steps,
        n_steps_per_epoch=steps_per_epoch,
        experiment_name=name,
        with_timestamp=False,
        logger_adapter=FileAdapterFactory(root_dir=str(condition_dir / "logs")),
        show_progress=False,
        save_interval=max(1, n_steps // steps_per_epoch + 1),
    )
    final_actions = algo.predict(data["observations"])
    final_q = algo.predict_value(data["observations"], data["actions_mm"])
    checkpoint_path = condition_dir / "awac.d3"
    algo.save(str(checkpoint_path))
    import d3rlpy

    reloaded = d3rlpy.load_learnable(str(checkpoint_path), device=False)
    reload_actions = reloaded.predict(data["observations"][:64])
    reload_difference = np.abs(reload_actions - final_actions[:64])
    history_rows = [{"epoch": int(epoch), **{key: float(value) for key, value in metrics.items()}} for epoch, metrics in history]
    (condition_dir / "history.json").write_text(json.dumps(history_rows, indent=2), encoding="utf-8")
    np.savez_compressed(
        condition_dir / "diagnostics.npz",
        initial_actions_mm=initial_actions,
        final_actions_mm=final_actions,
        initial_q=initial_q,
        final_q=final_q,
    )
    losses_finite = all(np.isfinite(list(row.values())[1:]).all() for row in history_rows)
    return {
        "condition": name,
        "source_bc_checkpoint": str(checkpoint.resolve()),
        "awac_checkpoint": str(checkpoint_path.resolve()),
        "transfer": transfer,
        "observation_scaler": observation_scaler,
        "initial_action_distribution_mm": _action_summary(initial_actions),
        "final_action_distribution_mm": _action_summary(final_actions),
        "initial_logged_action_error": _action_error_summary(initial_actions, data["actions_mm"]),
        "final_logged_action_error": _action_error_summary(final_actions, data["actions_mm"]),
        "initial_q": _finite_summary(initial_q),
        "final_q": _finite_summary(final_q),
        "history": history_rows,
        "losses_all_finite": bool(losses_finite),
        "checkpoint_reload_max_absolute_action_difference_mm": float(np.max(reload_difference)),
    }


def _make_plots(output_dir: Path, data: Mapping[str, np.ndarray], results: Sequence[Mapping[str, Any]]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for result in results:
        history = result["history"]
        x = [row["epoch"] for row in history]
        for axis, key in zip(axes, ("critic_loss", "actor_loss")):
            if history and key in history[0]:
                axis.plot(x, [row[key] for row in history], marker="o", label=result["condition"])
                axis.set_title(key)
                axis.set_xlabel("epoch")
                axis.grid(alpha=0.3)
    for axis in axes:
        axis.legend()
    fig.tight_layout()
    fig.savefig(output_dir / "learning_curves.png", dpi=160)
    plt.close(fig)

    diagnostics = [np.load(output_dir / result["condition"] / "diagnostics.npz") for result in results]
    fig, axes = plt.subplots(2, 3, figsize=(13, 7))
    for tendon_index, axis in enumerate(axes.flat):
        axis.hist(data["actions_mm"][:, tendon_index], bins=30, histtype="step", label="dataset")
        for result, diagnostic in zip(results, diagnostics):
            axis.hist(diagnostic["final_actions_mm"][:, tendon_index], bins=30, histtype="step", label=result["condition"])
        axis.set_title(TENDONS[tendon_index])
        axis.set_xlabel("target excursion [mm]")
    axes.flat[0].legend()
    fig.tight_layout()
    fig.savefig(output_dir / "action_distributions.png", dpi=160)
    plt.close(fig)

    fig, axis = plt.subplots(figsize=(7, 4))
    for result, diagnostic in zip(results, diagnostics):
        axis.hist(diagnostic["final_q"].reshape(-1), bins=40, histtype="step", label=result["condition"])
    axis.set_xlabel("Q(s, logged action)")
    axis.set_ylabel("count")
    axis.legend()
    axis.grid(alpha=0.3)
    fig.tight_layout()
    fig.savefig(output_dir / "q_value_distributions.png", dpi=160)
    plt.close(fig)
    for diagnostic in diagnostics:
        diagnostic.close()


def run_trial(config: Mapping[str, Any]) -> dict[str, Any]:
    output_dir = Path(config["output_dir"]).resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    data = load_legacy_awac_dataset(config["dataset_path"])
    count = len(data["rewards"])
    minimum = int(config["minimum_real_transitions"])
    if count < minimum:
        raise RuntimeError(
            f"Only {count} real transitions are available (< {minimum}). "
            "Do not impute legacy data; run the separate mock pipeline instead."
        )
    c3_obs, c3_action = _checkpoint_scalers(Path(config["c3_checkpoint"]))
    c4_obs, c4_action = _checkpoint_scalers(Path(config["c4_checkpoint"]))
    if c3_obs != c4_obs or c3_action != c4_action:
        raise ValueError("C3/C4 checkpoints do not share identical scalers")
    replay = build_replay_buffer(data)
    results = []
    for name, key in (("C3", "c3_checkpoint"), ("C4", "c4_checkpoint")):
        results.append(
            run_condition(
                name,
                Path(config[key]),
                data,
                replay,
                output_dir,
                n_steps=int(config["n_steps"]),
                steps_per_epoch=int(config["steps_per_epoch"]),
                batch_size=int(config["batch_size"]),
                seed=int(config["seed"]),
            )
        )
    _make_plots(output_dir, data, results)
    report = {
        "schema_version": TRIAL_SCHEMA_VERSION,
        "d3rlpy_version": "2.8.1",
        "trial_purpose": "pipeline validation only; not evidence of robot tracking performance or AWAC efficacy",
        "dataset_path": str(Path(config["dataset_path"]).resolve()),
        "dataset_transition_count": count,
        "dataset_kind": "legacy_approximate native-control-tick one-step fragments",
        "five_hz_zoh_dataset": False,
        "same_dataset_and_update_configuration_for_c3_c4": True,
        "configuration": dict(config),
        "logged_action_distribution_mm": _action_summary(data["actions_mm"]),
        "results": results,
        "interpretation_limits": [
            "Legacy FF targets changed at native approximately 50 Hz control ticks; they were not 5 Hz ZOH actions.",
            "The physical servo application timestamp and applied physical command are unobserved.",
            "The sparse one-step legacy fragments test software execution, not long-horizon offline control quality.",
            "No hardware was opened or driven and no online RL was run.",
        ],
    }
    (output_dir / "awac_trial_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Run C3/C4 provisional offline AWAC trial")
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = run_trial(config)
    print(json.dumps({"transitions": report["dataset_transition_count"], "conditions": [r["condition"] for r in report["results"]]}, indent=2))


if __name__ == "__main__":
    main()
