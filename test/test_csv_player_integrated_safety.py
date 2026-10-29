from __future__ import annotations

import json
import math
import tempfile
import threading
import time
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from controller.csv_player.encoder_calibration import (
    encoder_conversion_is_p2p_approved,
    load_encoder_excursion_calibration,
)
from controller.csv_player.gui import (
    CsvPlayerApp,
    IntegratedControlRuntime,
    LatestJointState,
    JointMeasurementSnapshot,
    _collect_stable_initial_dlc,
    _build_integrated_control_runtime,
    _connect_moment_arm_extrapolation_logger,
    _make_feedback_action_provider,
    _validate_point_to_point_request,
    _validate_point_to_point_encoder_snapshot,
    _validate_point_to_point_runner_start_state,
    _configured_encoder_start_tolerance_counts,
)
from controller.csv_player.trajectory import ExcursionTrajectory
from controller.feedback.moment_arm_runtime import MomentArmRuntime
from controller.servo.hardware_session import HardwareFeedback
from controller.servo_mapping import (
    ENCODER_SENSOR_IDS_BY_SERVO_ID,
    SERVO_IDS_BY_TENDON,
    TENDONS,
)


class IntegratedCsvPlayerSafetyTest(unittest.TestCase):
    def test_paper_rom_reference_is_separate_from_unapproved_operating_limits(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "controller"
            / "config_dlc_excursion_feedback.json"
        )
        with config_path.open("r", encoding="utf-8") as handle:
            gui_config = json.load(handle)["gui_integration"]

        reference = gui_config["paper_joint_rom_reference_deg"]
        self.assertEqual(reference["source"], "HF3M Finger, Table VI")
        self.assertEqual(reference["joint_order"], ["DIP", "PIP", "MCP"])
        self.assertFalse(reference["is_current_individual_safety_limit"])
        self.assertEqual(
            reference["endpoints"],
            {
                "DIP": {"extension": -8.85, "flexion": 76.23},
                "PIP": {"extension": -24.07, "flexion": 81.66},
                "MCP": {"extension": -56.61, "flexion": 106.18},
            },
        )
        self.assertIsNone(gui_config["joint_angle_limits_deg"])

    def test_encoder_start_tolerance_configuration_is_integer_and_bounded(self):
        self.assertEqual(_configured_encoder_start_tolerance_counts({}), 1)
        self.assertEqual(
            _configured_encoder_start_tolerance_counts(
                {"encoder_start_position_tolerance_counts": 4}
            ),
            4,
        )
        for value in (-1, 1.5, 8193, True, "invalid"):
            with self.subTest(value=value), self.assertRaises(ValueError):
                _configured_encoder_start_tolerance_counts(
                    {"encoder_start_position_tolerance_counts": value}
                )

    def test_p2p_requires_safety_filter_measured_encoder_and_approved_limits(self):
        nominal = SimpleNamespace(calibration_status="nominal")
        with self.assertRaisesRegex(ValueError, "explicitly approved spool-radius"):
            _validate_point_to_point_request(
                (0.1, 0.2, 0.3),
                2.0,
                playback_mode="rl_ff_zoh",
                decision_hz=5.0,
                safety_filter_enabled=True,
                encoder_calibration=nominal,
            )
        with self.assertRaisesRegex(ValueError, "Safety Filter"):
            _validate_point_to_point_request(
                (0.1, 0.2, 0.3),
                2.0,
                playback_mode="rl_ff_zoh",
                decision_hz=5.0,
                safety_filter_enabled=False,
                encoder_calibration=SimpleNamespace(calibration_status="measured"),
            )

    def test_encoder_conversion_requires_explicit_geometry_approval(self):
        config_path = (
            Path(__file__).resolve().parents[1]
            / "src"
            / "controller"
            / "excursion_encoder_calibration.json"
        )
        with config_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        current = load_encoder_excursion_calibration(config_path)
        self.assertEqual(current.calibration_status, "nominal")
        self.assertEqual(current.calibration_basis, "unspecified")
        self.assertEqual(current.approval_status, "unapproved")
        self.assertFalse(encoder_conversion_is_p2p_approved(current))

        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "encoder-calibration.json"
            payload["metadata"].update(
                {
                    "calibration_basis": "spool_radius_geometry",
                    "approval_status": "approved",
                }
            )
            path.write_text(json.dumps(payload), encoding="utf-8")
            geometric = load_encoder_excursion_calibration(path)

            self.assertEqual(geometric.calibration_status, "nominal")
            self.assertEqual(geometric.spool_radius_mm, 20.0)
            self.assertTrue(encoder_conversion_is_p2p_approved(geometric))

            payload["metadata"]["approval_status"] = "pending"
            path.write_text(json.dumps(payload), encoding="utf-8")
            unapproved = load_encoder_excursion_calibration(path)
            self.assertFalse(encoder_conversion_is_p2p_approved(unapproved))

            payload["metadata"]["approval_status"] = "approved"
            payload["encoder_degrees_per_mm"]["FDP"] = 3.0
            path.write_text(json.dumps(payload), encoding="utf-8")
            inconsistent = load_encoder_excursion_calibration(path)
            self.assertFalse(encoder_conversion_is_p2p_approved(inconsistent))

    def test_integrated_runtime_gate_accepts_explicitly_approved_geometry(self):
        app = object.__new__(CsvPlayerApp)
        app.feedback_enabled_var = SimpleNamespace(get=lambda: True)
        app.safety_filter_enabled_var = SimpleNamespace(get=lambda: True)
        app._load_encoder_calibration = lambda: SimpleNamespace(
            calibration_status="nominal",
            calibration_basis="spool_radius_geometry",
            approval_status="approved",
            spool_radius_mm=20.0,
            degrees_per_mm=(2.8648,) * len(TENDONS),
        )
        trajectory = ExcursionTrajectory(
            elapsed_times_s=(0.0, 1.0),
            excursions_mm=((0.0,) * len(TENDONS), (1.0,) * len(TENDONS)),
        )
        expected_runtime = SimpleNamespace(active=True, require_rl_ff_zoh=True)
        with patch(
            "controller.csv_player.gui._build_integrated_control_runtime",
            return_value=expected_runtime,
        ) as build_runtime:
            actual = app._build_selected_integrated_runtime(
                trajectory,
                {tendon: 1.0 for tendon in TENDONS},
                "rl_ff_zoh",
            )
        self.assertIs(actual, expected_runtime)
        build_runtime.assert_called_once()

    def test_moment_arm_runtime_loads_declared_domain_and_provenance(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for joint in ("DIP", "PIP", "MCP"):
                path = root / f"moment_arm_{joint}_average.json"
                path.write_text(
                    json.dumps(
                        {
                            "source_id": "mock-finger",
                            "source_trial": "mock",
                            "moment_arm_fit_degree": 3,
                            "evaluation_range_deg": [0.0, 90.0],
                            "moment_arm_coefficients": {
                                tendon: [0.0, 0.0, 0.0, 1.0]
                                for tendon in TENDONS
                            },
                        }
                    ),
                    encoding="utf-8",
                )
            model = MomentArmRuntime.from_directory(root)

        self.assertAlmostEqual(model.evaluation_ranges_rad["DIP"][1], math.pi / 2)
        self.assertEqual(model.model_info["joints"]["DIP"]["source_id"], "mock-finger")
        self.assertEqual(len(model.model_info["joints"]["DIP"]["sha256"]), 64)
        ExcursionTrajectory.from_joint_point_to_point(
            (0.1, 0.2, 0.3),
            (0.1, 0.2, 0.3),
            1.0,
            model,
            sample_hz=1.0,
        )
        extrapolated = ExcursionTrajectory.from_joint_point_to_point(
            (math.radians(91), 0.2, 0.3),
            (math.radians(95), 0.2, 0.3),
            1.0,
            model,
            sample_hz=1.0,
        )
        self.assertAlmostEqual(extrapolated.target_angles_deg[0][0], 91.0)
        self.assertAlmostEqual(extrapolated.target_angles_deg[-1][0], 95.0)
        self.assertAlmostEqual(extrapolated.excursions_mm[-1][0], math.radians(4))

    def test_moment_arm_extrapolation_is_written_to_session_event_logger(self):
        model = MomentArmRuntime(
            {
                joint: {tendon: [1.0] for tendon in TENDONS}
                for joint in ("DIP", "PIP", "MCP")
            },
            evaluation_ranges_rad={
                joint: (0.0, 1.0) for joint in ("DIP", "PIP", "MCP")
            },
        )

        class CapturingLogger:
            def __init__(self):
                self.records = []

            def log(self, event_type, *, event_monotonic_s=None, payload=None):
                self.records.append(
                    (event_type, event_monotonic_s, dict(payload or {}))
                )

        logger = CapturingLogger()
        _connect_moment_arm_extrapolation_logger(model, logger)
        model.matrix((-0.2, 0.5, 0.5))

        self.assertEqual(len(logger.records), 1)
        event_type, event_time, payload = logger.records[0]
        self.assertEqual(event_type, "moment_arm_constant_extrapolation")
        self.assertIsInstance(event_time, float)
        self.assertEqual(payload["joint"], "DIP")
        self.assertAlmostEqual(payload["requested_angle_deg"], math.degrees(-0.2))
        self.assertEqual(
            payload["fitting_range_rad"], {"q_min": 0.0, "q_max": 1.0}
        )
        self.assertEqual(payload["used_endpoint"], "q_min")
        self.assertTrue(payload["constant_extrapolation_applied"])

    def test_moment_arm_runtime_uses_common_observed_fit_ranges_when_explicit_range_is_absent(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for joint, low in (("DIP", 0.0), ("PIP", 0.0), ("MCP", -10.0)):
                payload = {
                    "source_id": "mock-finger",
                    "source_trial": "mean_of_3",
                    "moment_arm_fit_degree": 3,
                    "moment_arm_coefficients": {
                        tendon: [0.0, 0.0, 0.0, 1.0]
                        for tendon in TENDONS
                    },
                    "fit_metrics": {
                        tendon: {
                            "angle_min_deg": low,
                            "angle_max_deg": 90.0,
                        }
                        for tendon in TENDONS
                    },
                }
                if joint == "DIP":
                    payload["evaluation_range_deg"] = [0.0, 90.0]
                (root / f"moment_arm_{joint}_average.json").write_text(
                    json.dumps(payload),
                    encoding="utf-8",
                )

            model = MomentArmRuntime.from_directory(root)

        expected = {"DIP": (0.0, 90.0), "PIP": (0.0, 90.0), "MCP": (-10.0, 90.0)}
        for joint, bounds in expected.items():
            actual = model.evaluation_ranges_rad[joint]
            self.assertAlmostEqual(math.degrees(actual[0]), bounds[0])
            self.assertAlmostEqual(math.degrees(actual[1]), bounds[1])
        self.assertEqual(
            model.model_info["joints"]["MCP"]["evaluation_range_source"],
            "intersection_of_all_tendon_fit_data_ranges",
        )
        ExcursionTrajectory.from_joint_point_to_point(
            (math.radians(10), math.radians(10), math.radians(-5)),
            (math.radians(10), math.radians(10), math.radians(-5)),
            1.0,
            model,
            sample_hz=1.0,
        )
        extrapolated = ExcursionTrajectory.from_joint_point_to_point(
            (math.radians(10), math.radians(10), math.radians(-11)),
            (math.radians(10), math.radians(10), math.radians(-5)),
            1.0,
            model,
            sample_hz=1.0,
        )
        self.assertAlmostEqual(extrapolated.target_angles_deg[0][2], -11.0)
        self.assertAlmostEqual(extrapolated.excursions_mm[-1][0], math.radians(6))

    def test_p2p_request_allows_fit_extrapolation_but_enforces_joint_limits(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            model_dir = root / "moment-arm"
            model_dir.mkdir()
            for joint in ("DIP", "PIP", "MCP"):
                (model_dir / f"moment_arm_{joint}_average.json").write_text(
                    json.dumps(
                        {
                            "evaluation_range_deg": [0.0, 90.0],
                            "moment_arm_coefficients": {
                                tendon: [0.0, 0.0, 0.0, 1.0]
                                for tendon in TENDONS
                            },
                        }
                    ),
                    encoding="utf-8",
                )
            config_path = root / "feedback.json"
            config = {
                "paths": {"moment_arm_directory": str(model_dir)},
                "gui_integration": {
                    "joint_angle_limits_deg": {
                        "DIP": [0.0, 90.0],
                        "PIP": [-10.0, 90.0],
                        "MCP": [-10.0, 90.0],
                    }
                },
            }
            config_path.write_text(json.dumps(config), encoding="utf-8")

            model = _validate_point_to_point_request(
                (math.radians(10.0), math.radians(-5.0), math.radians(20.0)),
                2.0,
                playback_mode="rl_ff_zoh",
                decision_hz=5.0,
                safety_filter_enabled=True,
                encoder_calibration=SimpleNamespace(
                    calibration_status="measured"
                ),
                config_path=config_path,
            )
            self.assertIsInstance(model, MomentArmRuntime)

            with self.assertRaisesRegex(ValueError, "q_goal exceeds approved PIP"):
                _validate_point_to_point_request(
                    (
                        math.radians(10.0),
                        math.radians(-11.0),
                        math.radians(20.0),
                    ),
                    2.0,
                    playback_mode="rl_ff_zoh",
                    decision_hz=5.0,
                    safety_filter_enabled=True,
                    encoder_calibration=SimpleNamespace(
                        calibration_status="measured"
                    ),
                    config_path=config_path,
                )

    def test_ff_and_ff_fb_runtimes_share_the_same_approved_joint_limits(self):
        project_root = Path(__file__).resolve().parents[1]
        source_config_path = (
            project_root / "src" / "controller" / "config_dlc_excursion_feedback.json"
        )
        with source_config_path.open("r", encoding="utf-8") as handle:
            config = json.load(handle)
        candidate_limits = {
            "DIP": [-3.0, 15.0],
            "PIP": [-10.0, 15.0],
            "MCP": [-10.0, 15.0],
        }
        config["gui_integration"]["joint_angle_limits_deg"] = candidate_limits
        model_dir = project_root / config["paths"]["moment_arm_directory"]
        moment_arm = MomentArmRuntime.from_directory(model_dir)
        trajectory = ExcursionTrajectory.from_joint_point_to_point(
            tuple(math.radians(value) for value in (10.0, -6.5, 10.0)),
            tuple(math.radians(value) for value in (15.0, 5.0, 15.0)),
            2.0,
            moment_arm,
            sample_hz=5.0,
        )

        with tempfile.TemporaryDirectory() as directory:
            temporary_config_path = Path(directory) / "feedback.json"
            temporary_config_path.write_text(
                json.dumps(config),
                encoding="utf-8",
            )
            limits_n = {tendon: 10.0 for tendon in TENDONS}
            ff_runtime = _build_integrated_control_runtime(
                trajectory,
                limits_n,
                feedback_enabled=False,
                safety_filter_enabled=True,
                config_path=temporary_config_path,
            )
            ff_fb_runtime = _build_integrated_control_runtime(
                trajectory,
                limits_n,
                feedback_enabled=True,
                safety_filter_enabled=True,
                config_path=temporary_config_path,
                moment_arm_override=moment_arm,
            )

        self.assertIsNotNone(ff_runtime.action_safety_filter)
        self.assertIsNotNone(ff_fb_runtime.action_safety_filter)
        self.assertIsNone(ff_runtime.feedback_controller)
        self.assertIsNotNone(ff_fb_runtime.feedback_controller)
        self.assertEqual(
            ff_runtime.action_safety_filter.config.joint_angle_limits_rad,
            ff_fb_runtime.action_safety_filter.config.joint_angle_limits_rad,
        )
        self.assertAlmostEqual(
            ff_runtime.action_safety_filter.config.joint_angle_limits_rad["PIP"][0],
            math.radians(-10.0),
        )

    def test_checked_in_moment_arm_ranges_match_source_data_coverage(self):
        root = (
            Path(__file__).resolve().parents[1]
            / "logs"
            / "dual_camera_motionBenchmark"
            / "moment_arm_average"
        )
        model = MomentArmRuntime.from_directory(root)
        for joint, (low, high) in {
            "DIP": (0.0, 90.0),
            "PIP": (0.0, 90.0),
            "MCP": (-10.0, 90.0),
        }.items():
            bounds = model.evaluation_ranges_rad[joint]
            self.assertAlmostEqual(math.degrees(bounds[0]), low)
            self.assertAlmostEqual(math.degrees(bounds[1]), high)

    def test_moment_arm_fit_range_fallback_fails_closed_when_tendon_metadata_is_missing(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            for joint in ("DIP", "PIP", "MCP"):
                metrics = {
                    tendon: {"angle_min_deg": 0.0, "angle_max_deg": 90.0}
                    for tendon in TENDONS
                }
                if joint == "PIP":
                    del metrics[TENDONS[-1]]
                (root / f"moment_arm_{joint}_average.json").write_text(
                    json.dumps(
                        {
                            "moment_arm_coefficients": {
                                tendon: [0.0, 0.0, 0.0, 1.0]
                                for tendon in TENDONS
                            },
                            "fit_metrics": metrics,
                        }
                    ),
                    encoding="utf-8",
                )
            with self.assertRaisesRegex(ValueError, "Missing fit range metadata"):
                MomentArmRuntime.from_directory(root)

    def test_initial_dlc_requires_distinct_stable_fresh_frames(self):
        now = time.monotonic()
        snapshots = tuple(
            JointMeasurementSnapshot(
                angles_rad=(0.1 + index * 0.001, 0.2, 0.3),
                captured_at=now + index * 0.01,
                inference_finished_at=now + index * 0.01,
                valid=True,
                held_last=False,
                reason=None,
            )
            for index in range(5)
        )

        class SequenceLatest:
            def __init__(self, values):
                self.values = iter(values)

            def fresh_snapshot(self, _max_age_s):
                return next(self.values, None)

        latest = SequenceLatest(snapshots)
        selected, collected, median, spread = _collect_stable_initial_dlc(
            latest,
            max_age_s=0.35,
            stop_event=threading.Event(),
        )

        self.assertIs(selected, snapshots[-1])
        self.assertEqual(len(collected), 5)
        self.assertAlmostEqual(median[0], 0.102)
        self.assertLess(spread[0], 2.0)

    def test_initial_dlc_refuses_stale_or_unstable_samples(self):
        class MissingLatest:
            @staticmethod
            def fresh_snapshot(_max_age_s):
                return None

        with self.assertRaisesRegex(RuntimeError, "fresh DLC frames"):
            _collect_stable_initial_dlc(
                MissingLatest(),
                max_age_s=0.1,
                stop_event=threading.Event(),
                timeout_s=0.01,
            )

        now = time.monotonic()
        values = [
            JointMeasurementSnapshot(
                angles_rad=(0.1 + math.radians(index), 0.2, 0.3),
                captured_at=now + index * 0.01,
                inference_finished_at=now + index * 0.01,
                valid=True,
                held_last=False,
                reason=None,
            )
            for index in range(5)
        ]

        class SequenceLatest:
            def __init__(self):
                self.values = iter(values)

            def fresh_snapshot(self, _max_age_s):
                return next(self.values, None)

        with self.assertRaisesRegex(RuntimeError, "not stationary"):
            _collect_stable_initial_dlc(
                SequenceLatest(),
                max_age_s=0.35,
                stop_event=threading.Event(),
            )

    def test_p2p_encoder_origin_requires_fresh_status_and_matching_state(self):
        now = time.monotonic()
        sensor_samples = {
            ENCODER_SENSOR_IDS_BY_SERVO_ID[servo_id]: {
                "status": "OK",
                "relative_deg": 100.0 + index,
                "raw": round((100.0 + index) * 16384 / 360) % 16384,
            }
            for index, servo_id in enumerate(SERVO_IDS_BY_TENDON)
        }
        sensor_times = {
            sensor_id: now - 0.01 for sensor_id in sensor_samples
        }
        feedback = HardwareFeedback(
            servo={"valid": [True] * 6},
            servo_received_at=now - 0.01,
            encoder_by_sensor=sensor_samples,
            encoder_received_at=sensor_times,
            sampled_at=now,
        )

        class FakeHardware:
            @staticmethod
            def latest_feedback(_max_age_s):
                return feedback

            @staticmethod
            def position_by_servo_id(_feedback):
                return {servo_id: 100.0 for servo_id in SERVO_IDS_BY_TENDON}

        q0 = JointMeasurementSnapshot(
            angles_rad=(0.1, 0.2, 0.3),
            captured_at=now - 0.02,
            inference_finished_at=now - 0.01,
            valid=True,
            held_last=False,
            reason=None,
        )
        result = _validate_point_to_point_encoder_snapshot(
            FakeHardware(),
            q0_snapshot=q0,
            max_servo_age_s=0.2,
            max_encoder_age_s=0.2,
            max_joint_age_s=0.2,
        )
        self.assertEqual(set(result["encoder_relative_deg_by_tendon"]), set(TENDONS))
        self.assertEqual(set(result["encoder_raw_count_by_tendon"]), set(TENDONS))

        stale_feedback = HardwareFeedback(
            servo=feedback.servo,
            servo_received_at=now - 1.0,
            encoder_by_sensor=sensor_samples,
            encoder_received_at={sensor_id: now - 1.0 for sensor_id in sensor_samples},
            sampled_at=now,
        )

        class StaleHardware:
            @staticmethod
            def latest_feedback(_max_age_s):
                return stale_feedback

            @staticmethod
            def position_by_servo_id(_feedback):
                return {servo_id: 100.0 for servo_id in SERVO_IDS_BY_TENDON}

        with self.assertRaisesRegex(TimeoutError, "servo feedback"):
            _validate_point_to_point_encoder_snapshot(
                StaleHardware(),
                q0_snapshot=q0,
                max_servo_age_s=0.2,
                max_encoder_age_s=0.2,
                max_joint_age_s=0.2,
            )

    def test_runner_start_state_matches_dlc_and_encoder_with_sensor_resolution_tolerances(self):
        now = time.monotonic()
        q0 = JointMeasurementSnapshot(
            angles_rad=(0.1, 0.2, 0.3),
            captured_at=now - 0.05,
            inference_finished_at=now - 0.05,
            valid=True,
            held_last=False,
            reason=None,
        )
        latest = LatestJointState()
        latest.update(
            SimpleNamespace(
                flexion_angles_rad=(0.1 + math.radians(0.1), 0.2, 0.3),
                captured_at=now - 0.003,
                inference_finished_at=now - 0.002,
                valid=True,
                held_last=False,
            )
        )
        directions = (1.0,) * len(TENDONS)
        pre_counts = {tendon: 4000 + index for index, tendon in enumerate(TENDONS)}
        start_counts = {tendon: value + 1 for tendon, value in pre_counts.items()}
        pre_relative = {
            tendon: count * 360.0 / 16384
            for tendon, count in pre_counts.items()
        }
        start_relative = {
            tendon: count * 360.0 / 16384
            for tendon, count in start_counts.items()
        }
        sensor_samples = {
            ENCODER_SENSOR_IDS_BY_SERVO_ID[servo_id]: {
                "status": "OK",
                "relative_deg": start_relative[tendon],
                "raw": start_counts[tendon],
            }
            for tendon, servo_id in zip(TENDONS, SERVO_IDS_BY_TENDON)
        }
        feedback = HardwareFeedback(
            servo={"positions": [101] * 6, "valid": [True] * 6},
            servo_received_at=now - 0.002,
            encoder_by_sensor=sensor_samples,
            encoder_received_at={
                sensor_id: now - 0.002 for sensor_id in sensor_samples
            },
            sampled_at=now - 0.002,
        )
        pre_snapshot = {
            "servo_feedback_received_at": now - 0.04,
            "servo_position_by_servo_id": {servo_id: 100 for servo_id in range(1, 7)},
            "encoder_raw_count_by_tendon": pre_counts,
            "encoder_relative_deg_by_tendon": pre_relative,
            "encoder_received_at_by_tendon": {
                tendon: now - 0.04 for tendon in TENDONS
            },
        }

        class FakeHardware:
            @staticmethod
            def position_by_servo_id(_feedback):
                return {servo_id: 101.0 for servo_id in range(1, 7)}

        runner_origin = {
            servo_id: start_relative[tendon]
            for tendon, servo_id in zip(TENDONS, SERVO_IDS_BY_TENDON)
        }
        result = _validate_point_to_point_runner_start_state(
            FakeHardware(),
            latest_joint_state=latest,
            q0_snapshot=q0,
            q0_spread_deg=(0.1, 0.1, 0.1),
            encoder_origin_snapshot=pre_snapshot,
            feedback=feedback,
            runner_encoder_origin_deg_by_servo_id=runner_origin,
            encoder_directions=directions,
            max_joint_age_s=0.35,
        )
        self.assertEqual(result["status"], "passed")
        self.assertTrue(all(value == 1 for value in result["encoder_raw_count_delta"].values()))

        # A four-count tolerance must cover both the raw circular count check
        # and the Runner-origin degree check, including a count wrap.
        wrapped_tendon = TENDONS[0]
        wrapped_servo_id = SERVO_IDS_BY_TENDON[0]
        count_deg = 360.0 / 16384
        wrapped_snapshot = dict(pre_snapshot)
        wrapped_counts = dict(pre_snapshot["encoder_raw_count_by_tendon"])
        wrapped_angles = dict(pre_snapshot["encoder_relative_deg_by_tendon"])
        wrapped_counts[wrapped_tendon] = 16383
        wrapped_angles[wrapped_tendon] = 16383 * count_deg
        wrapped_snapshot["encoder_raw_count_by_tendon"] = wrapped_counts
        wrapped_snapshot["encoder_relative_deg_by_tendon"] = wrapped_angles

        wrapped_samples = {
            sensor_id: dict(sample)
            for sensor_id, sample in feedback.encoder_by_sensor.items()
        }
        wrapped_sensor = ENCODER_SENSOR_IDS_BY_SERVO_ID[wrapped_servo_id]
        wrapped_samples[wrapped_sensor]["raw"] = 3
        wrapped_samples[wrapped_sensor]["relative_deg"] = 3 * count_deg
        wrapped_feedback = HardwareFeedback(
            servo=feedback.servo,
            servo_received_at=feedback.servo_received_at,
            encoder_by_sensor=wrapped_samples,
            encoder_received_at=feedback.encoder_received_at,
            sampled_at=feedback.sampled_at,
        )
        wrapped_runner_origin = dict(runner_origin)
        wrapped_runner_origin[wrapped_servo_id] = 3 * count_deg
        wrapped_result = _validate_point_to_point_runner_start_state(
            FakeHardware(),
            latest_joint_state=latest,
            q0_snapshot=q0,
            q0_spread_deg=(0.1, 0.1, 0.1),
            encoder_origin_snapshot=wrapped_snapshot,
            feedback=wrapped_feedback,
            runner_encoder_origin_deg_by_servo_id=wrapped_runner_origin,
            encoder_directions=directions,
            max_joint_age_s=0.35,
            encoder_start_tolerance_counts=4,
        )
        self.assertEqual(wrapped_result["encoder_raw_count_delta"][wrapped_tendon], 4)
        self.assertAlmostEqual(
            wrapped_result["encoder_origin_delta_deg"][wrapped_tendon],
            4 * count_deg,
        )
        self.assertEqual(wrapped_result["encoder_count_tolerance"], 4)
        self.assertEqual(wrapped_result["servo_position_count_tolerance"], 1)

        # Even when raw counts pass, an inconsistent Runner origin beyond the
        # same four-count angular allowance must still reject.
        mismatched_runner_origin = dict(wrapped_runner_origin)
        mismatched_runner_origin[wrapped_servo_id] = 4 * count_deg
        with self.assertRaisesRegex(TimeoutError, "Runner Encoder origin changed"):
            _validate_point_to_point_runner_start_state(
                FakeHardware(),
                latest_joint_state=latest,
                q0_snapshot=q0,
                q0_spread_deg=(0.1, 0.1, 0.1),
                encoder_origin_snapshot=wrapped_snapshot,
                feedback=wrapped_feedback,
                runner_encoder_origin_deg_by_servo_id=mismatched_runner_origin,
                encoder_directions=directions,
                max_joint_age_s=0.35,
                encoder_start_tolerance_counts=4,
            )

        over_limit_samples = {
            sensor_id: dict(sample)
            for sensor_id, sample in wrapped_samples.items()
        }
        over_limit_samples[wrapped_sensor]["raw"] = 4
        over_limit_samples[wrapped_sensor]["relative_deg"] = 4 * count_deg
        over_limit_feedback = HardwareFeedback(
            servo=feedback.servo,
            servo_received_at=feedback.servo_received_at,
            encoder_by_sensor=over_limit_samples,
            encoder_received_at=feedback.encoder_received_at,
            sampled_at=feedback.sampled_at,
        )
        over_limit_runner_origin = dict(wrapped_runner_origin)
        over_limit_runner_origin[wrapped_servo_id] = 4 * count_deg
        with self.assertRaisesRegex(TimeoutError, "Encoder origin changed"):
            _validate_point_to_point_runner_start_state(
                FakeHardware(),
                latest_joint_state=latest,
                q0_snapshot=q0,
                q0_spread_deg=(0.1, 0.1, 0.1),
                encoder_origin_snapshot=wrapped_snapshot,
                feedback=over_limit_feedback,
                runner_encoder_origin_deg_by_servo_id=over_limit_runner_origin,
                encoder_directions=directions,
                max_joint_age_s=0.35,
                encoder_start_tolerance_counts=4,
            )

        class MovedServoHardware:
            @staticmethod
            def position_by_servo_id(_feedback):
                return {
                    servo_id: 102.0 if servo_id == wrapped_servo_id else 101.0
                    for servo_id in range(1, 7)
                }

        with self.assertRaisesRegex(TimeoutError, "Servo position changed"):
            _validate_point_to_point_runner_start_state(
                MovedServoHardware(),
                latest_joint_state=latest,
                q0_snapshot=q0,
                q0_spread_deg=(0.1, 0.1, 0.1),
                encoder_origin_snapshot=wrapped_snapshot,
                feedback=wrapped_feedback,
                runner_encoder_origin_deg_by_servo_id=wrapped_runner_origin,
                encoder_directions=directions,
                max_joint_age_s=0.35,
                encoder_start_tolerance_counts=4,
            )

        changed_encoder_samples = {
            sensor_id: dict(sample)
            for sensor_id, sample in feedback.encoder_by_sensor.items()
        }
        changed_sensor = ENCODER_SENSOR_IDS_BY_SERVO_ID[SERVO_IDS_BY_TENDON[0]]
        changed_encoder_samples[changed_sensor]["raw"] = (
            start_counts[TENDONS[0]] + 2
        ) % 16384
        changed_feedback = HardwareFeedback(
            servo=feedback.servo,
            servo_received_at=feedback.servo_received_at,
            encoder_by_sensor=changed_encoder_samples,
            encoder_received_at=feedback.encoder_received_at,
            sampled_at=feedback.sampled_at,
        )
        with self.assertRaisesRegex(TimeoutError, "Encoder origin changed"):
            _validate_point_to_point_runner_start_state(
                FakeHardware(),
                latest_joint_state=latest,
                q0_snapshot=q0,
                q0_spread_deg=(0.1, 0.1, 0.1),
                encoder_origin_snapshot=pre_snapshot,
                feedback=changed_feedback,
                runner_encoder_origin_deg_by_servo_id=runner_origin,
                encoder_directions=directions,
                max_joint_age_s=0.35,
            )

        latest.update(
            SimpleNamespace(
                flexion_angles_rad=(0.1 + math.radians(0.2), 0.2, 0.3),
                captured_at=now - 0.003,
                inference_finished_at=now - 0.002,
                valid=True,
                held_last=False,
            )
        )
        with self.assertRaisesRegex(RuntimeError, "posture changed"):
            _validate_point_to_point_runner_start_state(
                FakeHardware(),
                latest_joint_state=latest,
                q0_snapshot=q0,
                q0_spread_deg=(0.1, 0.1, 0.1),
                encoder_origin_snapshot=pre_snapshot,
                feedback=feedback,
                runner_encoder_origin_deg_by_servo_id=runner_origin,
                encoder_directions=directions,
                max_joint_age_s=0.35,
            )

        latest.invalidate("test stale DLC")
        with self.assertRaisesRegex(RuntimeError, "new fresh DLC frame"):
            _validate_point_to_point_runner_start_state(
                FakeHardware(),
                latest_joint_state=latest,
                q0_snapshot=q0,
                q0_spread_deg=(0.1, 0.1, 0.1),
                encoder_origin_snapshot=pre_snapshot,
                feedback=feedback,
                runner_encoder_origin_deg_by_servo_id=runner_origin,
                encoder_directions=directions,
                max_joint_age_s=0.01,
            )

    def test_p2p_feedback_reference_uses_the_active_decision_time(self):
        now = time.monotonic()
        latest = LatestJointState()
        latest.update(
            SimpleNamespace(
                flexion_angles_rad=(0.0, 0.0, 0.0),
                captured_at=now,
                inference_finished_at=now,
                valid=True,
                held_last=False,
            )
        )

        class RecordingController:
            reference = None

            def compute(self, reference, _measured):
                self.reference = tuple(reference)
                return SimpleNamespace(
                    error_rad=(0.0, 0.0, 0.0),
                    joint_correction_rad=(0.0, 0.0, 0.0),
                    excursion_correction_mm=(0.0,) * len(TENDONS),
                )

        controller = RecordingController()
        runtime = IntegratedControlRuntime(
            feedback_enabled=True,
            safety_filter_enabled=True,
            require_rl_ff_zoh=True,
            max_joint_age_s=0.5,
            feedback_controller=controller,
            action_safety_filter=None,
            automatic_return_after_safety_stop=False,
            config_summary={},
        )
        trajectory = ExcursionTrajectory(
            elapsed_times_s=(0.0, 1.0),
            excursions_mm=((0.0,) * 6, (0.0,) * 6),
            target_angles_deg=((0.0, 0.0, 0.0), (90.0, 90.0, 90.0)),
        )
        provider = _make_feedback_action_provider(
            runtime,
            trajectory,
            latest,
            threading.Event(),
            None,
            decision_hz=5.0,
        )
        provider(0.399, (0.0,) * 6)

        self.assertAlmostEqual(controller.reference[0], math.radians(18.0))

    def test_latest_joint_state_rejects_stale_dlc_measurement(self):
        latest = LatestJointState()
        now = time.monotonic()
        latest.update(
            SimpleNamespace(
                flexion_angles_rad=(0.1, 0.2, 0.3),
                captured_at=now - 1.0,
                inference_finished_at=now - 0.9,
                valid=True,
                held_last=False,
                reason=None,
            )
        )

        self.assertIsNone(latest.fresh_snapshot(0.1, now=now))
        self.assertIsNone(latest.safety_state(0.1))

    def test_selected_runtime_refuses_nominal_encoder_conversion(self):
        app = object.__new__(CsvPlayerApp)
        app.feedback_enabled_var = SimpleNamespace(get=lambda: True)
        app.safety_filter_enabled_var = SimpleNamespace(get=lambda: True)
        app._load_encoder_calibration = lambda: SimpleNamespace(
            calibration_status="nominal"
        )
        trajectory = ExcursionTrajectory(
            elapsed_times_s=(0.0, 1.0),
            excursions_mm=((0.0,) * len(TENDONS), (1.0,) * len(TENDONS)),
        )

        with self.assertRaisesRegex(ValueError, "explicitly approved spool-radius"):
            app._build_selected_integrated_runtime(
                trajectory,
                {tendon: 1.0 for tendon in TENDONS},
                "rl_ff_zoh",
            )

    def test_integrated_start_refuses_unapproved_joint_limits(self):
        trajectory = ExcursionTrajectory(
            elapsed_times_s=(0.0, 1.0),
            excursions_mm=((0.0,) * len(TENDONS), (1.0,) * len(TENDONS)),
        )
        config = {
            "paths": {},
            "feedback": {},
            "gui_integration": {
                "max_joint_age_s": 0.2,
                "max_servo_age_s": 0.2,
                "max_encoder_age_s": 0.2,
                "max_deadline_lateness_s": 0.1,
                "action_envelope_margin_mm": {
                    tendon: 1.0 for tendon in TENDONS
                },
                "max_action_delta_mm": {tendon: 1.0 for tendon in TENDONS},
                "joint_angle_limits_deg": None,
                "automatic_return_after_safety_stop": False,
                "exploration_enabled": False,
            },
        }
        with tempfile.TemporaryDirectory() as directory:
            path = Path(directory) / "integration.json"
            path.write_text(json.dumps(config), encoding="utf-8")
            with self.assertRaisesRegex(ValueError, "joint_angle_limits_deg"):
                _build_integrated_control_runtime(
                    trajectory,
                    {tendon: 1.0 for tendon in TENDONS},
                    feedback_enabled=False,
                    safety_filter_enabled=True,
                    config_path=path,
                )


if __name__ == "__main__":
    unittest.main()
