from __future__ import annotations

import csv
import json
import math
import tempfile
import unittest
from pathlib import Path

import numpy as np

from controller.rl.physics_teacher import (
    JOINTS,
    MomentArmTeacherModel,
    PhysicsTeacherGenerator,
)
from controller.servo_mapping import TENDONS


PROJECT_ROOT = Path(__file__).resolve().parents[1]
REAL_MODEL_DIR = (
    PROJECT_ROOT
    / "logs"
    / "dual_camera_motionBenchmark"
    / "moment_arm_average"
)


def _write_constant_models(root: Path, *, direction_dependent=False, angle_unit="rad"):
    for joint_index, joint in enumerate(JOINTS):
        payload = {
            "joint": joint,
            "angle_unit": angle_unit,
            "moment_arm_unit": "mm/rad",
            "fit_degree": 0,
            "coefficient_order": ["moment_arm_c0"],
            "moment_arm_coefficients": {
                tendon: [float((joint_index + 1) * (tendon_index + 1))]
                for tendon_index, tendon in enumerate(TENDONS)
            },
            "prediction_direction_dependent": direction_dependent,
            "direction_policy": "same coefficients in both directions",
            "source_direction_by_tendon": {
                tendon: "flexion" for tendon in TENDONS
            },
            "evaluation_range_deg": [0.0, 90.0],
        }
        (root / f"moment_arm_{joint}_average.json").write_text(
            json.dumps(payload),
            encoding="utf-8",
        )


class PhysicsTeacherTest(unittest.TestCase):
    def test_absolute_action_is_integrated_from_each_session_origin(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_constant_models(root)
            model = MomentArmTeacherModel.from_directory(root)
            initial = np.deg2rad([10.0, 20.0, 30.0])
            target_a = np.deg2rad([20.0, 30.0, 40.0])
            target_b = np.deg2rad([30.0, 40.0, 50.0])
            action_a = model.action_from_session_origin(initial, target_a)
            action_b = model.action_from_session_origin(initial, target_b)

        joint_delta_a = target_a - initial
        joint_delta_b = target_b - initial
        expected_a = np.asarray(
            [
                sum(
                    (joint_index + 1) * (tendon_index + 1) * joint_delta_a[joint_index]
                    for joint_index in range(3)
                )
                for tendon_index in range(6)
            ]
        )
        expected_b = np.asarray(
            [
                sum(
                    (joint_index + 1) * (tendon_index + 1) * joint_delta_b[joint_index]
                    for joint_index in range(3)
                )
                for tendon_index in range(6)
            ]
        )
        np.testing.assert_allclose(action_a, expected_a, atol=1e-12)
        np.testing.assert_allclose(action_b, expected_b, atol=1e-12)
        self.assertFalse(np.allclose(action_b, action_b - action_a))

    def test_model_rejects_wrong_units_direction_dependence_and_range(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_constant_models(root, angle_unit="deg")
            with self.assertRaisesRegex(ValueError, "angle_unit"):
                MomentArmTeacherModel.from_directory(root)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_constant_models(root, direction_dependent=True)
            with self.assertRaisesRegex(ValueError, "Direction-dependent"):
                MomentArmTeacherModel.from_directory(root)

        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            _write_constant_models(root)
            model = MomentArmTeacherModel.from_directory(root)
            with self.assertRaisesRegex(ValueError, "outside"):
                model.action_from_session_origin(
                    np.deg2rad([0.0, 0.0, 0.0]),
                    np.deg2rad([91.0, 0.0, 0.0]),
                )

    def test_saved_teacher_data_does_not_invent_dynamics(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            models = root / "models"
            models.mkdir()
            _write_constant_models(models)
            model = MomentArmTeacherModel.from_directory(models)
            dataset = PhysicsTeacherGenerator(model).generate_pairs(
                [
                    (
                        np.deg2rad([0.0, 0.0, 0.0]),
                        np.deg2rad([10.0, 20.0, 30.0]),
                    )
                ]
            )
            paths = dataset.save(root / "output")
            manifest = json.loads(paths["manifest"].read_text(encoding="utf-8"))
            with paths["pairs"].open("r", encoding="utf-8", newline="") as handle:
                columns = next(csv.reader(handle))

        self.assertFalse(manifest["contains_tension"])
        self.assertFalse(manifest["contains_next_state"])
        self.assertFalse(manifest["contains_reward"])
        self.assertTrue(manifest["must_not_be_mixed_as_real_awac_transitions"])
        self.assertFalse(any("reward" in column.lower() for column in columns))
        self.assertFalse(any("tension" in column.lower() for column in columns))
        self.assertFalse(any("next" in column.lower() for column in columns))

    def test_repository_models_have_valid_units_ranges_and_integrals(self):
        model = MomentArmTeacherModel.from_directory(REAL_MODEL_DIR)
        self.assertEqual(tuple(model.joint_models), JOINTS)
        for joint in JOINTS:
            low, high = model.valid_ranges_rad[joint]
            self.assertLess(low, high)
            self.assertGreaterEqual(math.degrees(low), -10.0 - 1e-9)
            self.assertLessEqual(math.degrees(high), 100.0 + 1e-9)

        initial = np.deg2rad([10.0, 10.0, 10.0])
        target = np.deg2rad([20.0, 20.0, 20.0])
        action = model.action_from_session_origin(initial, target)
        self.assertEqual(action.shape, (len(TENDONS),))
        self.assertTrue(np.isfinite(action).all())


if __name__ == "__main__":
    unittest.main()

