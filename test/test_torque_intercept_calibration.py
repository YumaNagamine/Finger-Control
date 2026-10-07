from __future__ import annotations

import tempfile
import unittest
from pathlib import Path

from controller.csv_player.force_estimator import CalibratedForceEstimator
from controller.servo.torque_calibration.calibration import (
    calculate_calibration_point,
    fit_fixed_slope_intercept,
    load_servo_calibration_models,
)
from controller.servo.torque_calibration.storage import (
    intercept_for_model,
    load_torque_intercept_calibration,
    save_torque_intercept_calibration,
)


class TorqueInterceptCalibrationTest(unittest.TestCase):
    def setUp(self):
        self.model = {
            "servo_id": 1,
            "sensor_id": 1,
            "slope_nmm_per_deg": 2.0,
            "moment_arm_mm": 10.0,
            "pull_angle_deg": 90.0,
        }

    def test_fixed_slope_point_and_fit(self):
        point_a = calculate_calibration_point(
            self.model,
            force_n=1.0,
            angle_delta_deg=3.0,
            torque_direction=1,
        )
        point_b = calculate_calibration_point(
            self.model,
            force_n=2.0,
            angle_delta_deg=5.0,
            torque_direction=1,
        )

        self.assertAlmostEqual(point_a["torque_nmm"], 10.0, places=6)
        self.assertAlmostEqual(point_a["intercept_nmm"], 4.0, places=6)
        self.assertAlmostEqual(point_b["intercept_nmm"], 10.0, places=6)
        fit = fit_fixed_slope_intercept(self.model, [point_a, point_b])
        self.assertAlmostEqual(fit["intercept_nmm"], 7.0, places=6)
        self.assertEqual(fit["point_count"], 2)

    def test_saved_calibration_round_trips_and_is_used(self):
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "torque_intercept_calibration.json"
            records = {
                1: {
                    "servo_id": 1,
                    "encoder_sensor_id": 1,
                    "slope_nmm_per_deg": 2.0,
                    "intercept_nmm": 7.0,
                    "point_count": 2,
                }
            }
            save_torque_intercept_calibration(path, records)
            loaded = load_torque_intercept_calibration(path)

            self.assertAlmostEqual(intercept_for_model(loaded, self.model), 7.0)
            self.assertTrue(path.is_file())

    def test_saved_slope_mismatch_is_rejected(self):
        records = {
            1: {
                "servo_id": 1,
                "encoder_sensor_id": 1,
                "slope_nmm_per_deg": 3.0,
                "intercept_nmm": 7.0,
                "point_count": 2,
            }
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "torque_intercept_calibration.json"
            save_torque_intercept_calibration(path, records)
            loaded = load_torque_intercept_calibration(path)
            with self.assertRaisesRegex(ValueError, "different slope"):
                intercept_for_model(loaded, self.model)

    def test_force_estimator_prefers_saved_intercept_calibration(self):
        project_root = Path(__file__).resolve().parents[1]
        calibration_dir = project_root / "third_party" / "ServoHardwareAPI" / "calibration" / "current"
        model = load_servo_calibration_models(calibration_dir)[1]
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "torque_intercept_calibration.json"
            save_torque_intercept_calibration(
                path,
                {
                    1: {
                        "servo_id": 1,
                        "encoder_sensor_id": int(model["sensor_id"]),
                        "slope_nmm_per_deg": float(model["slope_nmm_per_deg"]),
                        "intercept_nmm": 123.456,
                        "point_count": 2,
                    }
                },
            )
            estimator = CalibratedForceEstimator(
                calibration_dir=calibration_dir,
                zero_offset_path=Path(directory) / "missing-zero-offsets.json",
                intercept_calibration_path=path,
                servo_ids=(1,),
                tendon_by_servo_id={1: "ED"},
            )
            self.assertAlmostEqual(estimator._models[1]["effective_intercept_nmm"], 123.456)


if __name__ == "__main__":
    unittest.main()
