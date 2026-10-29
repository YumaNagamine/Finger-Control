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

from controller.rl.bc_pretraining import (
    MinMaxActionScalerState,
    StandardScalerState,
    generate_synthetic_teacher_pool,
    load_bc_checkpoint,
    train_condition,
)
from controller.rl.legacy_bc_dataset import LegacyBCDatasetProcessor
from controller.rl.physics_teacher import MomentArmTeacherModel


MODEL_DIR = ROOT / "logs" / "dual_camera_motionBenchmark" / "moment_arm_average"
LOG_ROOT = ROOT / "logs" / "csv_player_gui"
CONFIG = SRC / "controller" / "rl" / "config_bc_pretraining.json"


class BCPretrainingTest(TestCase):
    @classmethod
    def setUpClass(cls):
        cls.model = MomentArmTeacherModel.from_directory(MODEL_DIR)
        dataset = LegacyBCDatasetProcessor(LOG_ROOT, cls.model).process()
        table = dataset.common6_completed
        cls.train = table.subset([row["split"] == "training" for row in table.metadata])
        cls.validation = table.subset([row["split"] == "validation" for row in table.metadata])
        cls.config = json.loads(CONFIG.read_text(encoding="utf-8"))

    def test_teacher_uses_training_sessions_and_is_novel_and_in_domain(self):
        pool = generate_synthetic_teacher_pool(
            self.train, self.model, 24, self.config["augmentation"]
        )
        training_sessions = {row["session_id"] for row in self.train.metadata}
        validation_sessions = {row["session_id"] for row in self.validation.metadata}
        self.assertEqual(pool.report["validation_pairs_used"], 0)
        self.assertEqual(pool.report["out_of_domain_pairs"], 0)
        self.assertGreaterEqual(
            pool.report["nearest_real_distance_deg"]["min"],
            self.config["augmentation"]["minimum_novelty_deg"],
        )
        for row in pool.metadata:
            self.assertIn(row["source_session_id"], training_sessions)
            self.assertNotIn(row["source_session_id"], validation_sessions)
            self.assertFalse(row["contains_measured_tension"])
            self.assertFalse(row["contains_next_state"])
            self.assertFalse(row["contains_reward"])

    def test_smoke_training_checkpoint_round_trip(self):
        observation_scaler = StandardScalerState.fit(self.train.observations)
        action_scaler = MinMaxActionScalerState.fit(self.train.actions_mm)
        training = dict(self.config["training"])
        training.update({"gradient_updates": 4, "evaluation_interval": 2, "batch_size": 16})
        with tempfile.TemporaryDirectory() as temporary_directory:
            output = Path(temporary_directory)
            metrics, prediction = train_condition(
                "smoke", 7, self.train, self.validation,
                np.empty((0, 6)), np.empty((0, 6)), 1.0,
                training, observation_scaler, action_scaler, output,
            )
            checkpoint = output / "smoke" / "seed_7" / "checkpoint.pt"
            actor, loaded_obs, loaded_action, payload = load_bc_checkpoint(checkpoint)
            self.assertEqual(prediction.shape, (155, 6))
            self.assertTrue(np.isfinite(metrics["rmse_mm"]))
            self.assertEqual(payload["architecture"]["hidden"], [128, 128])
            self.assertEqual(tuple(loaded_obs.mean), observation_scaler.mean)
            self.assertEqual(tuple(loaded_action.minimum), action_scaler.minimum)
            self.assertEqual(actor.mean.out_features, 6)

