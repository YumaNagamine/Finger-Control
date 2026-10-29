from __future__ import annotations

import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from unittest import TestCase

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from controller.rl.awac_transfer import validate_function_preservation
from controller.rl.bc_pretraining import (
    CompatibleBCActor,
    MinMaxActionScalerState,
    StandardScalerState,
)


class AWACTransferTest(TestCase):
    def test_extra_observation_columns_preserve_bc_mean(self):
        torch.manual_seed(3)
        actor = CompatibleBCActor()
        observation_scaler = StandardScalerState(
            mean=(0.1, 0.2, 0.3, 0.4, 0.5, 0.6),
            std=(0.5, 0.6, 0.7, 0.8, 0.9, 1.0),
        )
        action_scaler = MinMaxActionScalerState(
            minimum=(-10.0, -9.0, -8.0, -7.0, -6.0, -5.0),
            maximum=(10.0, 11.0, 12.0, 13.0, 14.0, 15.0),
        )
        payload = {
            "schema_version": 1,
            "condition": "test",
            "seed": 3,
            "architecture": {"input": 6, "hidden": [128, 128]},
            "state_dict": actor.state_dict(),
            "observation_scaler": asdict(observation_scaler),
            "action_scaler": asdict(action_scaler),
        }
        with tempfile.TemporaryDirectory() as temporary_directory:
            checkpoint = Path(temporary_directory) / "checkpoint.pt"
            torch.save(payload, checkpoint)
            report = validate_function_preservation(
                checkpoint,
                full_observation_dim=14,
                bc_feature_indices=(0, 2, 4, 6, 8, 10),
                sample_count=8,
            )
        self.assertTrue(report["mock_validation"]["passed"])
        self.assertLessEqual(
            report["mock_validation"]["max_absolute_mean_difference_mm"], 1e-5
        )
        self.assertIn("AWAC _logstd", report["not_copied"])

