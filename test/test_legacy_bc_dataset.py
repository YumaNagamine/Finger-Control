from __future__ import annotations

import sys
from pathlib import Path
from unittest import TestCase

import numpy as np

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.append(str(SRC))

from controller.rl.legacy_bc_dataset import LegacyBCDatasetProcessor
from controller.rl.physics_teacher import MomentArmTeacherModel


MODEL_DIR = ROOT / "logs" / "dual_camera_motionBenchmark" / "moment_arm_average"
LOG_ROOT = ROOT / "logs" / "csv_player_gui"


class LegacyBCDatasetTest(TestCase):
    @classmethod
    def setUpClass(cls):
        model = MomentArmTeacherModel.from_directory(MODEL_DIR)
        cls.dataset = LegacyBCDatasetProcessor(LOG_ROOT, model).process()

    def test_audited_pair_counts_are_reproduced(self):
        counts = self.dataset.report["counts"]
        self.assertEqual(counts["sensor15_all"], 1565)
        self.assertEqual(counts["sensor15_completed"], 742)
        self.assertEqual(counts["common6_completed"], 776)
        self.assertEqual(counts["training_common6"], 621)
        self.assertEqual(counts["validation_common6"], 155)

    def test_validation_families_and_sessions_do_not_leak(self):
        rows = self.dataset.common6_completed.metadata
        training = [row for row in rows if row["split"] == "training"]
        validation = [row for row in rows if row["split"] == "validation"]
        self.assertEqual({row["trajectory_family"] for row in validation}, {"Edinburgh", "FullExtension"})
        self.assertFalse(
            {row["session_id"] for row in training}
            & {row["session_id"] for row in validation}
        )

    def test_action_is_absolute_from_each_session_origin(self):
        table = self.dataset.common6_completed
        first_by_session = {}
        for action, row in zip(table.actions_mm, table.metadata):
            first_by_session.setdefault(row["session_id"], action)
            self.assertFalse(row["action_is_incremental"])
            self.assertEqual(row["q0_semantics"], "q0_plan; not q0_measured")
        for action in first_by_session.values():
            np.testing.assert_allclose(action, 0.0, atol=1e-12)

    def test_sensor_observations_are_causal(self):
        for row in self.dataset.sensor15_all.metadata:
            decision = row["decision_monotonic_s"]
            self.assertLessEqual(row["vision_capture_monotonic_s"], decision)
            self.assertLessEqual(row["vision_inference_finished_monotonic_s"], decision)
            self.assertLessEqual(row["servo_feedback_received_monotonic_s"], decision)

    def test_recorded_ff_matches_moment_arm_recalculation_in_domain(self):
        comparison = self.dataset.report["recorded_ff_vs_moment_arm_recalculation"]
        self.assertEqual(comparison["comparison_pairs"], 412)
        self.assertEqual(comparison["excluded_out_of_domain_pairs"], 364)
        self.assertGreaterEqual(comparison["rmse_range_mm"][0], 0.00011)
        self.assertLessEqual(comparison["rmse_range_mm"][1], 0.00029)

