"""Technical validation of function-preserving BC -> d3rlpy AWAC transfer."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np
import torch

from controller.rl.bc_pretraining import load_bc_checkpoint


SUPPORTED_D3RLPY_VERSION = "2.8.1"


def build_transferred_awac(
    checkpoint_path: str | Path,
    full_observation_dim: int,
    bc_feature_indices: Sequence[int],
    *,
    extra_observation_mean: float = 0.0,
    extra_observation_std: float = 1.0,
) -> tuple[Any, dict[str, Any]]:
    """Build an untrained AWAC and transfer the deterministic BC mean function.

    Only the Gaussian mean is transferred.  AWAC log-standard-deviation and
    critics remain newly initialized, so this does not constitute a trained
    AWAC checkpoint.
    """

    import d3rlpy

    if d3rlpy.__version__ != SUPPORTED_D3RLPY_VERSION:
        raise RuntimeError(
            f"Transfer keys were validated for d3rlpy {SUPPORTED_D3RLPY_VERSION}; "
            f"installed version is {d3rlpy.__version__}"
        )
    indices = tuple(int(value) for value in bc_feature_indices)
    if len(indices) != 6 or len(set(indices)) != 6:
        raise ValueError("bc_feature_indices must contain six unique indices")
    if min(indices) < 0 or max(indices) >= full_observation_dim:
        raise ValueError("bc_feature_indices are outside full_observation_dim")
    actor, obs_scaler, action_scaler, checkpoint = load_bc_checkpoint(checkpoint_path)
    full_mean = np.full(full_observation_dim, extra_observation_mean, dtype=np.float32)
    full_std = np.full(full_observation_dim, extra_observation_std, dtype=np.float32)
    full_mean[list(indices)] = np.asarray(obs_scaler.mean, dtype=np.float32)
    full_std[list(indices)] = np.asarray(obs_scaler.std, dtype=np.float32)
    observation_scaler = d3rlpy.preprocessing.StandardObservationScaler(
        mean=full_mean, std=full_std, eps=0.0
    )
    output_scaler = d3rlpy.preprocessing.MinMaxActionScaler(
        minimum=np.asarray(action_scaler.minimum, dtype=np.float32),
        maximum=np.asarray(action_scaler.maximum, dtype=np.float32),
    )
    encoder = d3rlpy.models.VectorEncoderFactory(
        hidden_units=[128, 128], activation="relu", use_batch_norm=False,
        use_layer_norm=False, dropout_rate=None,
    )
    config = d3rlpy.algos.AWACConfig(
        actor_encoder_factory=encoder,
        critic_encoder_factory=encoder,
        observation_scaler=observation_scaler,
        action_scaler=output_scaler,
    )
    awac = config.create(device=False)
    awac.create_impl((full_observation_dim,), 6)
    policy = awac.impl.policy
    state = policy.state_dict()
    expected = {
        "_logstd": (1, 6),
        "_encoder._layers.0.weight": (128, full_observation_dim),
        "_encoder._layers.0.bias": (128,),
        "_encoder._layers.2.weight": (128, 128),
        "_encoder._layers.2.bias": (128,),
        "_mu.weight": (6, 128),
        "_mu.bias": (6,),
    }
    actual = {name: tuple(value.shape) for name, value in state.items()}
    if actual != expected:
        raise RuntimeError(f"Unexpected d3rlpy Gaussian actor state: {actual}")
    source = actor.state_dict()
    with torch.no_grad():
        first = torch.zeros_like(state["_encoder._layers.0.weight"])
        first[:, list(indices)] = source["encoder.0.weight"]
        state["_encoder._layers.0.weight"] = first
        state["_encoder._layers.0.bias"] = source["encoder.0.bias"].clone()
        state["_encoder._layers.2.weight"] = source["encoder.2.weight"].clone()
        state["_encoder._layers.2.bias"] = source["encoder.2.bias"].clone()
        state["_mu.weight"] = source["mean.weight"].clone()
        state["_mu.bias"] = source["mean.bias"].clone()
        # Deliberately preserve _logstd: deterministic BC does not identify it.
        policy.load_state_dict(state, strict=True)
    report = {
        "d3rlpy_version": d3rlpy.__version__,
        "checkpoint": str(Path(checkpoint_path).resolve()),
        "checkpoint_condition": checkpoint["condition"],
        "full_observation_dim": full_observation_dim,
        "bc_feature_indices": list(indices),
        "copied": {
            "BC encoder.0": "AWAC _encoder._layers.0 selected columns",
            "BC encoder.2": "AWAC _encoder._layers.2",
            "BC mean": "AWAC _mu",
        },
        "zero_initialized_extra_first_layer_columns": [
            index for index in range(full_observation_dim) if index not in indices
        ],
        "not_copied": ["AWAC _logstd", "critics", "target critics", "optimizers"],
        "scaler_compatibility": {
            "bc_feature_observation_mean_std_copied": True,
            "action_min_max_copied": True,
            "extra_feature_mean": extra_observation_mean,
            "extra_feature_std": extra_observation_std,
        },
        "actor_state_shapes": {key: list(value) for key, value in actual.items()},
    }
    return awac, report


def validate_function_preservation(
    checkpoint_path: str | Path,
    *,
    full_observation_dim: int = 12,
    bc_feature_indices: Sequence[int] = (1, 3, 5, 7, 9, 11),
    sample_count: int = 32,
    seed: int = 1234,
) -> dict[str, Any]:
    actor, obs_scaler, action_scaler, _ = load_bc_checkpoint(checkpoint_path)
    awac, report = build_transferred_awac(
        checkpoint_path, full_observation_dim, bc_feature_indices
    )
    rng = np.random.default_rng(seed)
    bc_raw = rng.normal(
        np.asarray(obs_scaler.mean), np.asarray(obs_scaler.std), size=(sample_count, 6)
    ).astype(np.float32)
    full_raw = rng.normal(size=(sample_count, full_observation_dim)).astype(np.float32)
    full_raw[:, list(bc_feature_indices)] = bc_raw
    with torch.no_grad():
        scaled = (bc_raw - np.asarray(obs_scaler.mean)) / np.asarray(obs_scaler.std)
        normalized = actor(torch.as_tensor(scaled, dtype=torch.float32)).numpy()
    bc_mm = action_scaler.inverse(normalized)
    awac_mm = awac.predict(full_raw)
    difference = np.abs(bc_mm - awac_mm)
    max_difference = float(np.max(difference))
    report["mock_validation"] = {
        "sample_count": sample_count,
        "max_absolute_mean_difference_mm": max_difference,
        "mean_absolute_mean_difference_mm": float(np.mean(difference)),
        "passed": max_difference <= 1e-5,
        "comparison": "deterministic BC action vs transferred AWAC Gaussian mean after inverse action scaling",
    }
    if not report["mock_validation"]["passed"]:
        raise RuntimeError(f"AWAC mean did not preserve BC function: max difference {max_difference}")
    return report


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Validate BC to AWAC actor transfer")
    parser.add_argument("checkpoint", type=Path)
    parser.add_argument("output_report", type=Path)
    parser.add_argument("--full-observation-dim", type=int, default=12)
    parser.add_argument("--bc-feature-indices", default="1,3,5,7,9,11")
    return parser.parse_args()


def main() -> None:
    args = _parse_args()
    indices = tuple(int(value) for value in args.bc_feature_indices.split(","))
    report = validate_function_preservation(
        args.checkpoint,
        full_observation_dim=args.full_observation_dim,
        bc_feature_indices=indices,
    )
    args.output_report.parent.mkdir(parents=True, exist_ok=True)
    args.output_report.write_text(
        json.dumps(report, indent=2, ensure_ascii=False, allow_nan=False), encoding="utf-8"
    )
    print(json.dumps(report["mock_validation"], indent=2))


if __name__ == "__main__":
    main()
