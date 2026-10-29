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

from controller.rl.bc_pretraining_v2 import (
    ActionTransform,
    build_nested_balanced_order,
    _subset,
)
from controller.rl.legacy_bc_dataset import load_pair_table
from controller.rl.physics_teacher import MomentArmTeacherModel
from controller.rl.physics_teacher_sampling_v2 import (
    PassiveReachabilityModel,
    PhysicsTeacherSamplerV2,
)


DATASET = ROOT / "artifacts" / "bc_pretraining_20261028" / "dataset"
MODEL_DIR = ROOT / "logs" / "dual_camera_motionBenchmark" / "moment_arm_average"
PASSIVE_DIR = ROOT / "logs" / "dual_camera_motionBenchmark" / "processed_csv"
CONFIG = SRC / "controller" / "rl" / "config_bc_pretraining_v2.json"


class BCPretrainingV2Test(TestCase):
    @classmethod
    def setUpClass(cls):
        config = json.loads(CONFIG.read_text(encoding="utf-8"))
        table = load_pair_table(
            DATASET / "bc_common6_completed.npz",
            DATASET / "bc_common6_completed_metadata.jsonl",
        )
        cls.train = table.subset([row["split"] == "training" for row in table.metadata])
        cls.model = MomentArmTeacherModel.from_directory(MODEL_DIR)
        cls.reachability = PassiveReachabilityModel.from_directory(
            PASSIVE_DIR, cls.model, config["teacher"]["passive_training_families"], 12
        )
        cls.sampler = PhysicsTeacherSamplerV2(cls.model, cls.reachability)

    def test_linear_transform_is_unbounded_and_round_trips(self):
        transform = ActionTransform.fit(self.train.actions_mm[:50], "linear_standard")
        extreme = self.train.actions_mm[:3] * 100.0
        self.assertTrue(transform.representable(extreme).all())
        np.testing.assert_allclose(
            transform.decode(transform.target(extreme)), extreme, rtol=1e-12, atol=1e-12
        )
        self.assertTrue(all(value is None for value in transform.physical_output_range_mm().values()))

    def test_legacy_minmax_identifies_unrepresentable_targets(self):
        transform = ActionTransform.fit(self.train.actions_mm[:50], "tanh_minmax_legacy")
        extreme = self.train.actions_mm[:3] + 100.0
        self.assertFalse(transform.representable(extreme).any())

    def test_nested_50_pair_subset_has_all_training_families_and_many_sessions(self):
        subset = _subset(self.train, build_nested_balanced_order(self.train)[:50])
        self.assertEqual(len(subset.metadata), 50)
        self.assertEqual(
            {row["trajectory_family"] for row in subset.metadata},
            {"FDSFlexion", "FullFlexion", "MCPFlexion", "MIddleFlexion", "ShallowFlexion"},
        )
        self.assertGreaterEqual(len({row["session_id"] for row in subset.metadata}), 15)
        teacher = self.sampler.sample_method_a(subset, 12, 44)
        self.assertEqual(len(teacher.observations), 12)

    def test_passive_model_excludes_validation_families(self):
        families = set(self.reachability.families.tolist())
        self.assertNotIn("Edinburgh", families)
        self.assertNotIn("FullExtension", families)
        self.assertEqual(
            families,
            {"FDSFlexion", "FullFlexion", "MCPFlexion", "MIddleFlexion", "ShallowFlexion"},
        )

    def test_methods_b_and_c_are_in_domain_and_not_transitions(self):
        for dataset in (
            self.sampler.sample_method_b(20, 1),
            self.sampler.sample_method_c(20, 2),
        ):
            self.assertEqual(dataset.observations.shape, (20, 6))
            self.assertTrue(dataset.report["all_in_moment_arm_domain"])
            for row in dataset.metadata:
                self.assertFalse(row["contains_measured_tension"])
                self.assertFalse(row["contains_next_state"])
                self.assertFalse(row["contains_reward"])
                self.assertFalse(row["is_rl_transition"])

