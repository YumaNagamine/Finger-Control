"""Dataset-independent construction and inference helpers for RL policies.

The observation width is deliberately not part of this module. d3rlpy builds
the policy network from the observation shape of the dataset passed to fit.
This keeps future additions, such as calibrated servo torque features, out of
the Actor implementation itself.
"""

from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Mapping, Protocol, Sequence

import numpy as np

from controller.servo_mapping import TENDONS


class ActorConfigurationError(ValueError):
    """Raised when an Actor configuration cannot define a compatible policy."""


class PredictablePolicy(Protocol):
    def predict(self, observations: np.ndarray) -> np.ndarray:
        """Return one six-tendon action for each observation."""


def _mapping(value: object, name: str) -> Mapping[str, Any]:
    if not isinstance(value, Mapping):
        raise ActorConfigurationError(f"{name} must be an object")
    return value


def _positive_int(value: object, name: str) -> int:
    if isinstance(value, bool):
        raise ActorConfigurationError(f"{name} must be a positive integer")
    try:
        parsed = int(value)
    except (TypeError, ValueError) as exc:
        raise ActorConfigurationError(f"{name} must be a positive integer") from exc
    if parsed <= 0 or parsed != value:
        raise ActorConfigurationError(f"{name} must be a positive integer")
    return parsed


def _finite_positive_float(value: object, name: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ActorConfigurationError(f"{name} must be positive and finite") from exc
    if not math.isfinite(parsed) or parsed <= 0.0:
        raise ActorConfigurationError(f"{name} must be positive and finite")
    return parsed


def load_actor_settings(path: str | Path) -> dict[str, Any]:
    """Load and validate policy settings without importing d3rlpy."""

    config_path = Path(path).expanduser().resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    root = _mapping(config, "config")
    policy = _mapping(root.get("policy"), "policy")

    action_names = policy.get("action_names")
    if not isinstance(action_names, list) or tuple(action_names) != TENDONS:
        raise ActorConfigurationError(
            f"policy.action_names must be {list(TENDONS)} in canonical order"
        )

    network = _mapping(policy.get("network"), "policy.network")
    hidden_units = network.get("hidden_units")
    if not isinstance(hidden_units, list) or not hidden_units:
        raise ActorConfigurationError(
            "policy.network.hidden_units must be a non-empty list"
        )
    validated_units = [
        _positive_int(value, f"policy.network.hidden_units[{index}]")
        for index, value in enumerate(hidden_units)
    ]
    activation = network.get("activation")
    if not isinstance(activation, str) or not activation:
        raise ActorConfigurationError(
            "policy.network.activation must be a non-empty string"
        )
    for option in ("use_batch_norm", "use_layer_norm"):
        if not isinstance(network.get(option), bool):
            raise ActorConfigurationError(f"policy.network.{option} must be boolean")
    dropout_rate = network.get("dropout_rate")
    if dropout_rate is not None:
        dropout_rate = float(dropout_rate)
        if not math.isfinite(dropout_rate) or not 0.0 <= dropout_rate < 1.0:
            raise ActorConfigurationError(
                "policy.network.dropout_rate must be null or in [0, 1)"
            )

    algorithms = _mapping(root.get("algorithms"), "algorithms")
    bc = _mapping(algorithms.get("bc"), "algorithms.bc")
    awac = _mapping(algorithms.get("awac"), "algorithms.awac")
    _positive_int(bc.get("batch_size"), "algorithms.bc.batch_size")
    _finite_positive_float(bc.get("learning_rate"), "algorithms.bc.learning_rate")
    _positive_int(awac.get("batch_size"), "algorithms.awac.batch_size")
    for option in ("actor_learning_rate", "critic_learning_rate", "gamma", "tau", "lam"):
        _finite_positive_float(awac.get(option), f"algorithms.awac.{option}")
    _positive_int(awac.get("n_action_samples"), "algorithms.awac.n_action_samples")
    _positive_int(awac.get("n_critics"), "algorithms.awac.n_critics")

    validated = dict(config)
    validated["policy"] = dict(policy)
    validated["policy"]["network"] = dict(network)
    validated["policy"]["network"]["hidden_units"] = validated_units
    validated["policy"]["network"]["dropout_rate"] = dropout_rate
    return validated


def build_algorithm(
    algorithm: str,
    settings: Mapping[str, Any],
    *,
    device: str = "cpu",
) -> Any:
    """Build a d3rlpy BC or AWAC algorithm from validated settings.

    No observation dimension is supplied here. d3rlpy initializes that shape
    when the algorithm first receives a dataset.
    """

    normalized_algorithm = algorithm.lower()
    if normalized_algorithm not in ("bc", "awac"):
        raise ActorConfigurationError("algorithm must be 'bc' or 'awac'")

    try:
        import d3rlpy
    except ImportError as exc:
        raise RuntimeError(
            "d3rlpy is required to build the policy; install the optional RL "
            "dependencies before training"
        ) from exc

    policy = _mapping(settings.get("policy"), "policy")
    network = _mapping(policy.get("network"), "policy.network")
    encoder = d3rlpy.models.VectorEncoderFactory(
        hidden_units=list(network["hidden_units"]),
        activation=str(network["activation"]),
        use_batch_norm=bool(network["use_batch_norm"]),
        use_layer_norm=bool(network["use_layer_norm"]),
        dropout_rate=network.get("dropout_rate"),
    )
    scaler = d3rlpy.preprocessing.StandardObservationScaler()
    algorithms = _mapping(settings.get("algorithms"), "algorithms")

    if normalized_algorithm == "bc":
        options = _mapping(algorithms.get("bc"), "algorithms.bc")
        config = d3rlpy.algos.BCConfig(
            batch_size=int(options["batch_size"]),
            learning_rate=float(options["learning_rate"]),
            encoder_factory=encoder,
            observation_scaler=scaler,
        )
    else:
        options = _mapping(algorithms.get("awac"), "algorithms.awac")
        config = d3rlpy.algos.AWACConfig(
            batch_size=int(options["batch_size"]),
            actor_learning_rate=float(options["actor_learning_rate"]),
            critic_learning_rate=float(options["critic_learning_rate"]),
            gamma=float(options["gamma"]),
            tau=float(options["tau"]),
            lam=float(options["lam"]),
            n_action_samples=int(options["n_action_samples"]),
            n_critics=int(options["n_critics"]),
            actor_encoder_factory=encoder,
            critic_encoder_factory=encoder,
            observation_scaler=scaler,
        )
    return config.create(device=device)


def predict_actions(
    policy: PredictablePolicy,
    observations: np.ndarray | Sequence[Sequence[float]],
) -> np.ndarray:
    """Validate observations and return unmodified six-tendon predictions."""

    batch = np.asarray(observations, dtype=np.float32)
    if batch.ndim != 2 or batch.shape[0] == 0 or batch.shape[1] == 0:
        raise ValueError("observations must have shape (batch, features)")
    if not np.isfinite(batch).all():
        raise ValueError("observations contain NaN or infinity")

    actions = np.asarray(policy.predict(batch), dtype=np.float32)
    expected_shape = (batch.shape[0], len(TENDONS))
    if actions.shape != expected_shape:
        raise ValueError(
            f"policy returned shape {actions.shape}; expected {expected_shape}"
        )
    if not np.isfinite(actions).all():
        raise ValueError("policy returned NaN or infinity")
    return actions
