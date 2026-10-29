"""Diagnose whether legacy-log AWAC differs materially from Gaussian BC.

This module is intentionally offline-only.  It compares an unweighted
Gaussian maximum-likelihood actor update with d3rlpy 2.8.1 AWAC, records the
actual mini-batch-normalized advantage weights, and runs a reward-permutation
diagnostic without inventing transitions.
"""

from __future__ import annotations

import argparse
import copy
import json
import math
import random
import types
from pathlib import Path
from typing import Any, Mapping, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import torch

from controller.rl.legacy_awac_dataset import load_legacy_awac_dataset
from controller.rl.offline_awac_trial import build_initialized_awac, build_replay_buffer
from controller.servo_mapping import TENDONS


SCHEMA_VERSION = 1


class AdvantageRecorder:
    def __init__(self, lam: float):
        self.lam = float(lam)
        self.advantages: list[np.ndarray] = []
        self.weights: list[np.ndarray] = []
        self.update_ess: list[float] = []

    def append(self, advantages: torch.Tensor, weights: torch.Tensor) -> None:
        adv = advantages.detach().cpu().numpy().reshape(-1).astype(np.float64)
        weight = weights.detach().cpu().numpy().reshape(-1).astype(np.float64)
        self.advantages.append(adv)
        self.weights.append(weight)
        self.update_ess.append(effective_sample_size(weight))

    def arrays(self) -> tuple[np.ndarray, np.ndarray]:
        return np.concatenate(self.advantages), np.concatenate(self.weights)

    def report(self) -> dict[str, Any]:
        advantages, weights = self.arrays()
        raw_exp = np.exp(np.clip(advantages / self.lam, -80.0, 80.0))
        return {
            "formula": "weight = softmax(advantage / lambda) * mini_batch_size",
            "lambda": self.lam,
            "advantage": distribution_summary(advantages),
            "raw_exp_advantage_over_lambda": {
                **distribution_summary(raw_exp),
                "numerical_note": "Only the reported raw exponential is clipped to exp([-80,80]); d3rlpy uses stable softmax without this clipping.",
            },
            "actual_normalized_actor_weight": distribution_summary(weights),
            "global_repeated_sample_ess": effective_sample_size(weights),
            "global_repeated_sample_ess_fraction": effective_sample_size(weights) / len(weights),
            "per_update_ess": distribution_summary(np.asarray(self.update_ess)),
            "per_update_ess_fraction_mean": float(np.mean(self.update_ess) / len(self.weights[0])),
            "recorded_actor_updates": len(self.advantages),
            "recorded_repeated_samples": len(advantages),
        }


def effective_sample_size(weights: np.ndarray) -> float:
    values = np.asarray(weights, dtype=np.float64).reshape(-1)
    denominator = float(np.sum(values * values))
    return 0.0 if denominator == 0.0 else float(np.sum(values) ** 2 / denominator)


def distribution_summary(values: np.ndarray) -> dict[str, Any]:
    array = np.asarray(values, dtype=np.float64).reshape(-1)
    quantiles = (0.0, 0.01, 0.05, 0.25, 0.5, 0.75, 0.95, 0.99, 1.0)
    return {
        "count": len(array),
        "mean": float(np.mean(array)),
        "std": float(np.std(array)),
        "min": float(np.min(array)),
        "max": float(np.max(array)),
        "quantiles": {f"p{int(q * 100):02d}": float(np.quantile(array, q)) for q in quantiles},
        "all_finite": bool(np.isfinite(array).all()),
    }


def _compute_awac_weights(impl, obs_t, act_t, recorder: AdvantageRecorder | None):
    from d3rlpy.algos.qlearning.torch.awac_impl import (
        build_gaussian_distribution,
        expand_and_repeat_recursively,
        flatten_left_recursively,
        get_batch_size,
    )
    import torch.nn.functional as functional

    with torch.no_grad():
        batch_size = get_batch_size(obs_t)
        q_values = impl._q_func_forwarder.compute_expected_q(obs_t, act_t, "min")
        dist = build_gaussian_distribution(impl._modules.policy(obs_t))
        policy_actions = dist.sample_n(impl._n_action_samples)
        flat_actions = policy_actions.reshape(-1, impl.action_size)
        repeated = expand_and_repeat_recursively(obs_t, impl._n_action_samples)
        flat_obs = flatten_left_recursively(repeated, dim=1)
        flat_v = impl._q_func_forwarder.compute_expected_q(flat_obs, flat_actions, "min")
        v_values = flat_v.view(batch_size, -1, 1).mean(dim=1)
        advantages = (q_values - v_values).view(-1)
        weights = functional.softmax(advantages / impl._lam, dim=0).view(-1, 1)
        weights = weights * advantages.numel()
    if recorder is not None:
        recorder.append(advantages, weights)
    return weights


def install_advantage_recorder(algo) -> AdvantageRecorder:
    recorder = AdvantageRecorder(algo.impl._lam)

    def wrapped(impl, obs_t, act_t):
        return _compute_awac_weights(impl, obs_t, act_t, recorder)

    algo.impl._compute_weights = types.MethodType(wrapped, algo.impl)
    return recorder


def train_gaussian_bc(
    checkpoint: Path,
    data: Mapping[str, np.ndarray],
    output_dir: Path,
    *,
    n_steps: int,
    batch_size: int,
    seed: int,
) -> tuple[Any, dict[str, Any]]:
    """Train the same d3rlpy NormalPolicy with unweighted Gaussian NLL."""
    from d3rlpy.algos.qlearning.torch.awac_impl import build_gaussian_distribution

    algo, transfer, scaler = build_initialized_awac(
        checkpoint,
        data["observations"],
        data["next_observations"],
        batch_size=batch_size,
        seed=seed,
    )
    rng = np.random.default_rng(seed)
    losses: list[dict[str, float]] = []
    bucket: list[float] = []
    policy = algo.impl.policy
    optimizer = algo.impl._modules.actor_optim
    for step in range(1, n_steps + 1):
        indices = rng.integers(0, len(data["actions_mm"]), size=batch_size)
        obs = torch.as_tensor(data["observations"][indices], dtype=torch.float32)
        action = torch.as_tensor(data["actions_mm"][indices], dtype=torch.float32)
        obs = algo.observation_scaler.transform(obs)
        action = algo.action_scaler.transform(action)
        dist = build_gaussian_distribution(policy(obs))
        loss = -dist.log_prob(action).sum()
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()
        bucket.append(float(loss.detach().cpu()))
        if step % 100 == 0 or step == n_steps:
            losses.append({"step": step, "gaussian_nll_sum": float(np.mean(bucket))})
            bucket = []
    output_dir.mkdir(parents=True, exist_ok=True)
    torch.save(
        {
            "kind": "unweighted_gaussian_bc_diagnostic",
            "policy_state_dict": policy.state_dict(),
            "observation_scaler": scaler,
            "source_bc_checkpoint": str(checkpoint.resolve()),
            "updates": n_steps,
            "batch_size": batch_size,
            "seed": seed,
        },
        output_dir / "gaussian_bc_policy.pt",
    )
    report = {
        "kind": "unweighted Gaussian maximum-likelihood BC",
        "loss_formula": "-sum(log Normal(squashed_mu, std).prob(scaled logged action))",
        "same_normal_policy_as_awac": True,
        "updates": n_steps,
        "batch_size": batch_size,
        "seed": seed,
        "transfer": transfer,
        "history": losses,
    }
    return algo, report


def train_awac(
    checkpoint: Path,
    data: Mapping[str, np.ndarray],
    output_dir: Path,
    *,
    n_steps: int,
    steps_per_epoch: int,
    batch_size: int,
    seed: int,
    reward_permutation: np.ndarray | None = None,
) -> tuple[Any, dict[str, Any], AdvantageRecorder]:
    from d3rlpy.logging import NoopAdapterFactory

    run_data = dict(data)
    label = "observed"
    if reward_permutation is not None:
        run_data["rewards"] = np.asarray(data["rewards"])[reward_permutation].copy()
        label = "permuted_diagnostic"
    algo, transfer, scaler = build_initialized_awac(
        checkpoint,
        data["observations"],
        data["next_observations"],
        batch_size=batch_size,
        seed=seed,
    )
    recorder = install_advantage_recorder(algo)
    replay = build_replay_buffer(run_data)
    history = algo.fit(
        replay,
        n_steps=n_steps,
        n_steps_per_epoch=steps_per_epoch,
        experiment_name=f"awac_{label}",
        with_timestamp=False,
        logger_adapter=NoopAdapterFactory(),
        show_progress=False,
        save_interval=n_steps + 1,
    )
    output_dir.mkdir(parents=True, exist_ok=True)
    algo.save(str(output_dir / "awac.d3"))
    advantages, weights = recorder.arrays()
    np.savez_compressed(
        output_dir / "training_advantage_weights.npz",
        advantages=advantages,
        normalized_weights=weights,
        per_update_ess=np.asarray(recorder.update_ess),
    )
    report = {
        "reward_assignment": label,
        "transfer": transfer,
        "scaler": scaler,
        "history": [
            {"epoch": int(epoch), **{key: float(value) for key, value in metrics.items()}}
            for epoch, metrics in history
        ],
        "training_advantage_weights": recorder.report(),
    }
    return algo, report, recorder


def stable_advantage(algo, observations: np.ndarray, actions_mm: np.ndarray, *, samples: int, seed: int):
    from d3rlpy.algos.qlearning.torch.awac_impl import build_gaussian_distribution

    torch.manual_seed(seed)
    obs = torch.as_tensor(observations, dtype=torch.float32)
    act = torch.as_tensor(actions_mm, dtype=torch.float32)
    obs = algo.observation_scaler.transform(obs)
    act = algo.action_scaler.transform(act)
    with torch.no_grad():
        q = algo.impl._q_func_forwarder.compute_expected_q(obs, act, "min")
        dist = build_gaussian_distribution(algo.impl.policy(obs))
        policy_actions = dist.sample_n(samples)
        batch, _, action_size = policy_actions.shape
        flat_actions = policy_actions.reshape(-1, action_size)
        flat_obs = obs[:, None, :].expand(batch, samples, obs.shape[1]).reshape(-1, obs.shape[1])
        flat_v = algo.impl._q_func_forwarder.compute_expected_q(flat_obs, flat_actions, "min")
        v = flat_v.view(batch, samples, 1).mean(dim=1)
        advantage = q - v
    return q.cpu().numpy().reshape(-1), v.cpu().numpy().reshape(-1), advantage.cpu().numpy().reshape(-1)


def bootstrap_diagnostic(algo, data: Mapping[str, np.ndarray], *, samples: int, seed: int) -> dict[str, Any]:
    from d3rlpy.models.torch.policies import build_squashed_gaussian_distribution

    torch.manual_seed(seed)
    next_obs = torch.as_tensor(data["next_observations"], dtype=torch.float32)
    next_obs = algo.observation_scaler.transform(next_obs)
    with torch.no_grad():
        dist = build_squashed_gaussian_distribution(algo.impl.policy(next_obs))
        actions, log_probs = dist.sample_n_with_log_prob(samples)
        batch, count, action_size = actions.shape
        flat_obs = next_obs[:, None, :].expand(batch, count, next_obs.shape[1]).reshape(-1, next_obs.shape[1])
        flat_actions = actions.reshape(-1, action_size)
        q = algo.impl._targ_q_func_forwarder.compute_target(flat_obs, flat_actions, reduction="min")
        q = q.view(batch, count, 1)
        entropy_regularized_next = (q - log_probs).mean(dim=1).cpu().numpy().reshape(-1)
    reward = np.asarray(data["rewards"]).reshape(-1)
    bootstrap = 0.99 * entropy_regularized_next
    target = reward + bootstrap
    return {
        "reward": distribution_summary(reward),
        "gamma_times_entropy_regularized_next_q": distribution_summary(bootstrap),
        "bellman_target": distribution_summary(target),
        "fraction_abs_bootstrap_greater_than_abs_reward": float(np.mean(np.abs(bootstrap) > np.abs(reward))),
        "fixed_entropy_temperature": 1.0,
        "monte_carlo_policy_samples": samples,
        "interpretation": "All fragments are nonterminal truncations, so the target bootstraps from an unobserved continuation. The term also contains d3rlpy AWAC's fixed-temperature entropy contribution.",
    }


def action_error(prediction: np.ndarray, target: np.ndarray) -> dict[str, Any]:
    error = np.asarray(prediction) - np.asarray(target)
    return {
        "mae_mm": float(np.mean(np.abs(error))),
        "rmse_mm": float(np.sqrt(np.mean(error * error))),
        "per_tendon_rmse_mm": {
            tendon: float(np.sqrt(np.mean(error[:, index] ** 2)))
            for index, tendon in enumerate(TENDONS)
        },
    }


def evaluate_policy(algo, eval_path: Path) -> tuple[dict[str, Any], np.ndarray]:
    with np.load(eval_path, allow_pickle=False) as payload:
        common = payload["observations"].astype(np.float32)
        target = payload["actions_mm"].astype(np.float32)
    # Extra feedback features are undefined for synthetic unseen postures.
    # Setting them to the offline-training mean makes their standardized value zero.
    full = np.tile(np.asarray(algo.observation_scaler.mean, dtype=np.float32), (len(common), 1))
    full[:, :6] = common
    prediction = algo.predict(full)
    return {
        **action_error(prediction, target),
        "samples": len(common),
        "extra_feedback_context": "offline dataset mean (zero after standardization)",
    }, prediction


def _rank(values: np.ndarray) -> np.ndarray:
    order = np.argsort(values, kind="mergesort")
    result = np.empty(len(values), dtype=np.float64)
    result[order] = np.arange(len(values), dtype=np.float64)
    return result


def _correlation(left: np.ndarray, right: np.ndarray, *, rank: bool = False) -> float:
    x = _rank(left) if rank else np.asarray(left, dtype=np.float64)
    y = _rank(right) if rank else np.asarray(right, dtype=np.float64)
    if np.std(x) < 1e-12 or np.std(y) < 1e-12:
        return 0.0
    return float(np.corrcoef(x, y)[0, 1])


def high_low_analysis(
    advantages: np.ndarray,
    data: Mapping[str, np.ndarray],
    metadata: Sequence[Mapping[str, Any]],
) -> dict[str, Any]:
    count = max(1, int(math.ceil(0.2 * len(advantages))))
    order = np.argsort(advantages)
    low, high = order[:count], order[-count:]
    rewards = np.asarray(data["rewards"]).reshape(-1)
    tracking_rmse_deg = np.degrees(np.sqrt(np.maximum(0.0, -rewards)))
    obs = np.asarray(data["observations"])
    force_mean = np.mean(obs[:, 15:21], axis=1)
    target_norm = np.linalg.norm(obs[:, 3:6] - obs[:, :3], axis=1)

    def group(indices: np.ndarray) -> dict[str, Any]:
        return {
            "count": len(indices),
            "advantage": distribution_summary(advantages[indices]),
            "reward": distribution_summary(rewards[indices]),
            "joint_tracking_rmse_deg": distribution_summary(tracking_rmse_deg[indices]),
            "q0_plan_mean_rad": np.mean(obs[indices, :3], axis=0).tolist(),
            "q_target_mean_rad": np.mean(obs[indices, 3:6], axis=0).tolist(),
            "target_motion_norm_rad": distribution_summary(target_norm[indices]),
            "mean_tension_n": distribution_summary(force_mean[indices]),
            "trajectory_family_counts": _counts(metadata[index]["trajectory_family"] for index in indices),
        }

    sessions = np.asarray([row["session_id"] for row in metadata])
    centered_adv = advantages.copy()
    centered_reward = rewards.copy()
    centered_error = tracking_rmse_deg.copy()
    centered_force = force_mean.copy()
    for session in np.unique(sessions):
        mask = sessions == session
        centered_adv[mask] -= np.mean(centered_adv[mask])
        centered_reward[mask] -= np.mean(centered_reward[mask])
        centered_error[mask] -= np.mean(centered_error[mask])
        centered_force[mask] -= np.mean(centered_force[mask])
    return {
        "selection": "top and bottom 20% by final MC advantage",
        "low": group(low),
        "high": group(high),
        "unadjusted_correlations": {
            "spearman_advantage_reward": _correlation(advantages, rewards, rank=True),
            "spearman_advantage_tracking_error": _correlation(advantages, tracking_rmse_deg, rank=True),
            "spearman_advantage_mean_tension": _correlation(advantages, force_mean, rank=True),
            "spearman_advantage_target_motion_norm": _correlation(advantages, target_norm, rank=True),
        },
        "within_session_centered_correlations": {
            "pearson_advantage_reward": _correlation(centered_adv, centered_reward),
            "pearson_advantage_tracking_error": _correlation(centered_adv, centered_error),
            "pearson_advantage_mean_tension": _correlation(centered_adv, centered_force),
        },
        "confounding_warning": "These are observational comparisons. q0, q_target, family, tension and action coverage differ across samples; within-session centering removes only session-level offsets and does not identify a causal action effect.",
    }


def _counts(values) -> dict[str, int]:
    result: dict[str, int] = {}
    for value in values:
        result[str(value)] = result.get(str(value), 0) + 1
    return dict(sorted(result.items()))


def _metadata(path: Path) -> list[dict[str, Any]]:
    return [json.loads(line) for line in path.read_text(encoding="utf-8").splitlines() if line]


def _plot(output: Path, condition_reports: Mapping[str, Mapping[str, Any]], arrays: Mapping[str, np.ndarray]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11, 4))
    for name in ("C3", "C4"):
        axes[0].hist(arrays[f"{name}_weights"], bins=50, histtype="step", label=name)
        axes[1].hist(arrays[f"{name}_advantages"], bins=50, histtype="step", label=name)
    axes[0].set_title("AWAC training actor weights")
    axes[0].set_xlabel("softmax(A/lambda) * batch size")
    axes[1].set_title("AWAC training advantages")
    axes[1].set_xlabel("Q(s,a_data) - V_pi(s)")
    for axis in axes:
        axis.legend(); axis.grid(alpha=0.3)
    fig.tight_layout(); fig.savefig(output / "advantage_weight_distributions.png", dpi=160); plt.close(fig)

    labels = []
    values = []
    for condition in ("C3", "C4"):
        evaluation = condition_reports[condition]["unseen_evaluation"]
        for model in ("initial_bc", "gaussian_bc", "awac", "awac_reward_shuffle"):
            labels.append(f"{condition}\n{model}")
            values.append(evaluation[model]["rmse_mm"])
    fig, axis = plt.subplots(figsize=(11, 4))
    axis.bar(labels, values)
    axis.set_ylabel("FF action RMSE [mm]")
    axis.set_title("Independent unseen posture evaluation")
    axis.grid(axis="y", alpha=0.3)
    fig.tight_layout(); fig.savefig(output / "unseen_posture_ff_rmse.png", dpi=160); plt.close(fig)


def run_diagnostic(config: Mapping[str, Any]) -> dict[str, Any]:
    output = Path(config["output_dir"]).resolve(); output.mkdir(parents=True, exist_ok=True)
    data = load_legacy_awac_dataset(config["dataset_path"])
    metadata = _metadata(Path(config["metadata_path"]))
    if len(metadata) != len(data["rewards"]):
        raise ValueError("Transition metadata count differs from dataset")
    rng = np.random.default_rng(int(config["reward_shuffle_seed"]))
    reward_permutation = rng.permutation(len(data["rewards"]))
    condition_reports: dict[str, Any] = {}
    plot_arrays: dict[str, np.ndarray] = {}
    for condition, key in (("C3", "c3_checkpoint"), ("C4", "c4_checkpoint")):
        condition_dir = output / condition; checkpoint = Path(config[key])
        initial, initial_transfer, _ = build_initialized_awac(
            checkpoint, data["observations"], data["next_observations"],
            batch_size=int(config["batch_size"]), seed=int(config["seed"]),
        )
        gaussian_bc, bc_report = train_gaussian_bc(
            checkpoint, data, condition_dir / "gaussian_bc",
            n_steps=int(config["n_steps"]), batch_size=int(config["batch_size"]), seed=int(config["seed"]),
        )
        awac, awac_report, recorder = train_awac(
            checkpoint, data, condition_dir / "awac",
            n_steps=int(config["n_steps"]), steps_per_epoch=int(config["steps_per_epoch"]),
            batch_size=int(config["batch_size"]), seed=int(config["seed"]),
        )
        shuffled, shuffled_report, _ = train_awac(
            checkpoint, data, condition_dir / "awac_reward_shuffle",
            n_steps=int(config["n_steps"]), steps_per_epoch=int(config["steps_per_epoch"]),
            batch_size=int(config["batch_size"]), seed=int(config["seed"]), reward_permutation=reward_permutation,
        )
        initial_q, initial_v, initial_advantage = stable_advantage(
            initial, data["observations"], data["actions_mm"],
            samples=int(config["diagnostic_policy_samples"]), seed=int(config["seed"]) + 1,
        )
        final_q, final_v, advantage = stable_advantage(
            awac, data["observations"], data["actions_mm"],
            samples=int(config["diagnostic_policy_samples"]), seed=int(config["seed"]) + 1,
        )
        shuffled_q, shuffled_v, shuffled_advantage = stable_advantage(
            shuffled, data["observations"], data["actions_mm"],
            samples=int(config["diagnostic_policy_samples"]), seed=int(config["seed"]) + 1,
        )
        eval_path = Path(config["unseen_evaluation_path"])
        initial_eval, _ = evaluate_policy(initial, eval_path)
        bc_eval, bc_prediction = evaluate_policy(gaussian_bc, eval_path)
        awac_eval, awac_prediction = evaluate_policy(awac, eval_path)
        shuffled_eval, shuffled_prediction = evaluate_policy(shuffled, eval_path)
        dataset_bc = gaussian_bc.predict(data["observations"])
        dataset_awac = awac.predict(data["observations"])
        dataset_shuffle = shuffled.predict(data["observations"])
        condition_reports[condition] = {
            "initial_transfer": initial_transfer,
            "gaussian_bc": bc_report,
            "awac": awac_report,
            "awac_reward_shuffle": shuffled_report,
            "logged_dataset_action_error": {
                "gaussian_bc": action_error(dataset_bc, data["actions_mm"]),
                "awac": action_error(dataset_awac, data["actions_mm"]),
                "awac_reward_shuffle": action_error(dataset_shuffle, data["actions_mm"]),
            },
            "actor_output_differences": {
                "awac_vs_gaussian_bc_logged_dataset": action_error(dataset_awac, dataset_bc),
                "awac_vs_reward_shuffle_logged_dataset": action_error(dataset_awac, dataset_shuffle),
                "awac_vs_gaussian_bc_unseen": action_error(awac_prediction, bc_prediction),
                "awac_vs_reward_shuffle_unseen": action_error(awac_prediction, shuffled_prediction),
            },
            "unseen_evaluation": {
                "dataset": str(eval_path.resolve()),
                "initial_bc": initial_eval,
                "gaussian_bc": bc_eval,
                "awac": awac_eval,
                "awac_reward_shuffle": shuffled_eval,
            },
            "final_advantage": distribution_summary(advantage),
            "reward_shuffled_final_advantage": distribution_summary(shuffled_advantage),
            "q_value_evolution": {
                "initial_logged_action_q": distribution_summary(initial_q),
                "initial_policy_v": distribution_summary(initial_v),
                "initial_advantage": distribution_summary(initial_advantage),
                "final_logged_action_q": distribution_summary(final_q),
                "final_policy_v": distribution_summary(final_v),
                "final_advantage": distribution_summary(advantage),
                "reward_shuffled_final_logged_action_q": distribution_summary(shuffled_q),
                "reward_shuffled_final_policy_v": distribution_summary(shuffled_v),
            },
            "high_low_advantage_comparison": high_low_analysis(advantage, data, metadata),
            "bootstrap": bootstrap_diagnostic(
                awac, data, samples=int(config["diagnostic_policy_samples"]), seed=int(config["seed"]) + 2
            ),
        }
        adv_array, weight_array = recorder.arrays()
        plot_arrays[f"{condition}_advantages"] = adv_array
        plot_arrays[f"{condition}_weights"] = weight_array
    _plot(output, condition_reports, plot_arrays)
    report = {
        "schema_version": SCHEMA_VERSION,
        "purpose": "diagnose whether AWAC actor updates use informative advantage weighting beyond Gaussian BC",
        "dataset_transition_count": len(data["rewards"]),
        "observation_dim": data["observations"].shape[1],
        "same_dataset_scaler_initial_actor_seed_and_actor_structure": True,
        "configuration": dict(config),
        "reward_shuffle": {
            "seed": int(config["reward_shuffle_seed"]),
            "permutation_saved": True,
            "diagnostic_only": True,
        },
        "conditions": condition_reports,
        "interpretation_limits": [
            "There is only one logged FF action per state; no within-state counterfactual action comparison is observed.",
            "Every sample is a nonterminal one-step fragment, so Q bootstraps from an unobserved continuation.",
            "High/low advantage group differences are observational and confounded by state, goal, family and tension.",
            "Synthetic unseen-posture evaluation measures Moment-arm FF imitation with extra feedback inputs fixed to the offline mean, not robot tracking.",
            "Actor update counts, seed, architecture and scalers match, but Gaussian BC uses a NumPy mini-batch sampler while AWAC uses d3rlpy ReplayBuffer sampling.",
        ],
    }
    np.save(output / "reward_permutation.npy", reward_permutation)
    (output / "diagnostic_report.json").write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Compare Gaussian BC and AWAC advantage weighting")
    parser.add_argument("config", type=Path)
    args = parser.parse_args()
    config = json.loads(args.config.read_text(encoding="utf-8"))
    report = run_diagnostic(config)
    print(json.dumps({"transitions": report["dataset_transition_count"], "conditions": list(report["conditions"])}, indent=2))


if __name__ == "__main__":
    main()
