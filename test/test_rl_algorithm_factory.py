from __future__ import annotations

import json
import sys
import tempfile
from pathlib import Path
from unittest import TestCase

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from controller.rl.algorithm_factory import (
    AlgorithmConfigurationError,
    load_algorithm_settings,
    predict_actions,
)
from controller.servo_mapping import TENDONS


CONFIG_PATH = SRC / "controller" / "rl" / "config_offline.json"


class _ZeroPolicy:
    def predict(self, observations: np.ndarray) -> np.ndarray:
        return np.zeros((len(observations), len(TENDONS)), dtype=np.float32)


class _OutOfRangePolicy:
    def predict(self, observations: np.ndarray) -> np.ndarray:
        return np.full((len(observations), len(TENDONS)), 1.25, dtype=np.float32)


class AlgorithmFactoryTest(TestCase):
    def test_config_uses_canonical_tendon_order(self):
        settings = load_algorithm_settings(CONFIG_PATH)
        self.assertEqual(tuple(settings["policy"]["action_names"]), TENDONS)

    def test_observation_width_is_not_hard_coded(self):
        policy = _ZeroPolicy()
        self.assertEqual(predict_actions(policy, np.zeros((2, 54))).shape, (2, 6))
        self.assertEqual(predict_actions(policy, np.zeros((2, 60))).shape, (2, 6))

    def test_prediction_is_not_silently_clipped(self):
        actions = predict_actions(_OutOfRangePolicy(), np.zeros((1, 54)))
        self.assertTrue(np.all(actions == np.float32(1.25)))

    def test_wrong_tendon_order_is_rejected(self):
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        config["policy"]["action_names"] = list(reversed(TENDONS))
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(AlgorithmConfigurationError, "canonical order"):
                load_algorithm_settings(path)

    def test_boolean_numeric_setting_is_rejected(self):
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        config["algorithms"]["awac"]["gamma"] = True
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(AlgorithmConfigurationError, "positive and finite"):
                load_algorithm_settings(path)

    def test_invalid_dropout_uses_configuration_error(self):
        config = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
        config["policy"]["network"]["dropout_rate"] = "not-a-number"
        with tempfile.TemporaryDirectory() as temporary_directory:
            path = Path(temporary_directory) / "config.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(AlgorithmConfigurationError, "dropout_rate"):
                load_algorithm_settings(path)

    def test_non_finite_observation_is_rejected(self):
        with self.assertRaisesRegex(ValueError, "NaN or infinity"):
            predict_actions(_ZeroPolicy(), [[0.0, float("nan")]])
