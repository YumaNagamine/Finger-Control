"""AWAC transfer compatibility checks for version-2 BC checkpoints."""

from __future__ import annotations

import json
from pathlib import Path
import argparse
from typing import Any, Sequence

import numpy as np
import torch

from controller.rl.bc_pretraining_v2 import ActionTransform, BCActorV2


SUPPORTED_D3RLPY_VERSION = "2.8.1"


def load_v2_checkpoint(path: str | Path) -> tuple[BCActorV2, dict[str, Any]]:
    payload = torch.load(Path(path), map_location="cpu", weights_only=False)
    actor = BCActorV2(str(payload["output_mode"])); actor.load_state_dict(payload["state_dict"]); actor.eval()
    return actor, payload


def validate_v2_awac_transfer(
    checkpoint_path: str | Path,
    *,
    full_observation_dim: int = 18,
    bc_feature_indices: Sequence[int] = (0, 1, 2, 3, 4, 5),
    sample_count: int = 32,
    seed: int = 5102,
) -> dict[str, Any]:
    import d3rlpy

    actor, payload = load_v2_checkpoint(checkpoint_path)
    mode = str(payload["output_mode"])
    if mode == "linear_standard":
        return {
            "compatible": False,
            "d3rlpy_version": d3rlpy.__version__,
            "output_mode": mode,
            "reason": (
                "d3rlpy 2.8.1 AWAC NormalPolicy applies tanh to the Gaussian mean; "
                "an unbounded linear BC output cannot be globally function-preserved by "
                "directly copying a linear mean head"
            ),
            "safe_options": [
                "use the validated tanh_wide_standard companion checkpoint",
                "distill linear BC actions into a squashed AWAC actor on an explicit domain",
                "implement and validate a custom unsquashed Gaussian actor",
            ],
        }
    if mode not in ("tanh_minmax_legacy", "tanh_wide_standard"):
        raise ValueError(f"Unsupported output mode: {mode}")
    if d3rlpy.__version__ != SUPPORTED_D3RLPY_VERSION:
        raise RuntimeError(f"Expected d3rlpy {SUPPORTED_D3RLPY_VERSION}, got {d3rlpy.__version__}")
    indices = tuple(int(value) for value in bc_feature_indices)
    if len(indices) != 6 or len(set(indices)) != 6 or max(indices) >= full_observation_dim:
        raise ValueError("Invalid bc_feature_indices")
    obs_state = payload["observation_scaler"]
    transform = ActionTransform(**{
        "mode": payload["action_transform"]["mode"],
        "center": tuple(payload["action_transform"]["center"]),
        "scale": tuple(payload["action_transform"]["scale"]),
    })
    full_mean = np.zeros(full_observation_dim, dtype=np.float32)
    full_std = np.ones(full_observation_dim, dtype=np.float32)
    full_mean[list(indices)] = np.asarray(obs_state["mean"], dtype=np.float32)
    full_std[list(indices)] = np.asarray(obs_state["std"], dtype=np.float32)
    center, scale = np.asarray(transform.center), np.asarray(transform.scale)
    encoder = d3rlpy.models.VectorEncoderFactory(hidden_units=[128, 128], activation="relu")
    awac = d3rlpy.algos.AWACConfig(
        actor_encoder_factory=encoder,
        critic_encoder_factory=encoder,
        observation_scaler=d3rlpy.preprocessing.StandardObservationScaler(
            mean=full_mean, std=full_std, eps=0.0
        ),
        action_scaler=d3rlpy.preprocessing.MinMaxActionScaler(
            minimum=(center - scale).astype(np.float32),
            maximum=(center + scale).astype(np.float32),
        ),
    ).create(device=False)
    awac.create_impl((full_observation_dim,), 6)
    policy = awac.impl.policy
    state = policy.state_dict(); source = actor.state_dict()
    expected_first = (128, full_observation_dim)
    if tuple(state["_encoder._layers.0.weight"].shape) != expected_first:
        raise RuntimeError("Unexpected d3rlpy first-layer shape")
    with torch.no_grad():
        first = torch.zeros_like(state["_encoder._layers.0.weight"])
        first[:, list(indices)] = source["encoder.0.weight"]
        state["_encoder._layers.0.weight"] = first
        state["_encoder._layers.0.bias"] = source["encoder.0.bias"].clone()
        state["_encoder._layers.2.weight"] = source["encoder.2.weight"].clone()
        state["_encoder._layers.2.bias"] = source["encoder.2.bias"].clone()
        state["_mu.weight"] = source["mean.weight"].clone()
        state["_mu.bias"] = source["mean.bias"].clone()
        policy.load_state_dict(state, strict=True)
    rng = np.random.default_rng(seed)
    common = rng.normal(
        np.asarray(obs_state["mean"]), np.asarray(obs_state["std"]),
        size=(sample_count, 6),
    ).astype(np.float32)
    full = rng.normal(size=(sample_count, full_observation_dim)).astype(np.float32)
    full[:, list(indices)] = common
    scaled = (common - np.asarray(obs_state["mean"])) / np.asarray(obs_state["std"])
    with torch.no_grad(): bc_output = actor(torch.as_tensor(scaled, dtype=torch.float32)).numpy()
    bc_mm = transform.decode(bc_output)
    awac_mm = awac.predict(full)
    difference = np.abs(bc_mm - awac_mm)
    return {
        "compatible": bool(np.max(difference) <= 1e-5),
        "d3rlpy_version": d3rlpy.__version__,
        "output_mode": mode,
        "max_absolute_difference_mm": float(np.max(difference)),
        "mean_absolute_difference_mm": float(np.mean(difference)),
        "sample_count": sample_count,
        "copied": ["selected first-layer columns", "encoder second layer", "Gaussian mu head"],
        "not_copied": ["logstd", "critics", "optimizers"],
        "extra_observation_columns_zero_initialized": True,
    }


def save_transfer_report(checkpoint: str | Path, output: str | Path) -> dict[str, Any]:
    report = validate_v2_awac_transfer(checkpoint)
    Path(output).write_text(json.dumps(report, indent=2), encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description="Validate version-2 BC to AWAC transfer")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output", type=Path)
    args = parser.parse_args()
    report = save_transfer_report(args.checkpoint, args.output)
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
