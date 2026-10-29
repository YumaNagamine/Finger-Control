from __future__ import annotations

import sys
import tempfile
from dataclasses import asdict
from pathlib import Path
from unittest import TestCase

import torch

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from controller.rl.awac_transfer_v2 import validate_v2_awac_transfer
from controller.rl.bc_pretraining import StandardScalerState
from controller.rl.bc_pretraining_v2 import ActionTransform, BCActorV2


class AWACTransferV2Test(TestCase):
    def _checkpoint(self, root: Path, mode: str) -> Path:
        actor = BCActorV2(mode)
        obs = StandardScalerState((0.0,) * 6, (1.0,) * 6)
        transform = ActionTransform(mode, (0.0,) * 6, (10.0,) * 6)
        path = root / f"{mode}.pt"
        torch.save(
            {
                "output_mode": mode,
                "state_dict": actor.state_dict(),
                "observation_scaler": asdict(obs),
                "action_transform": asdict(transform),
            },
            path,
        )
        return path

    def test_linear_mode_is_explicitly_not_directly_compatible(self):
        with tempfile.TemporaryDirectory() as directory:
            report = validate_v2_awac_transfer(
                self._checkpoint(Path(directory), "linear_standard")
            )
        self.assertFalse(report["compatible"])
        self.assertIn("tanh", report["reason"])

    def test_wide_tanh_preserves_gaussian_mean(self):
        with tempfile.TemporaryDirectory() as directory:
            report = validate_v2_awac_transfer(
                self._checkpoint(Path(directory), "tanh_wide_standard")
            )
        self.assertTrue(report["compatible"])
        self.assertLessEqual(report["max_absolute_difference_mm"], 1e-5)
