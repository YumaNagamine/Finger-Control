from __future__ import annotations

import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from controller.rl.dataset_processor import (
    AWACDatasetProcessor,
    DatasetProcessorConfig,
    classify_episode_boundary,
)
from controller.csv_player.trajectory import ExcursionTrajectory
from controller.servo_mapping import TENDONS


JOINTS = ("DIP", "PIP", "MCP")


def _event(sequence, event_type, timestamp, payload):
    return {
        "schema_version": 1,
        "run_id": "episode-1",
        "event_sequence": sequence,
        "event_type": event_type,
        "event_monotonic_s": timestamp,
        "enqueued_monotonic_s": timestamp,
        "written_monotonic_s": timestamp + 0.001,
        "payload": payload,
    }


def _tick_payload(timestamp, decision_id, actual_offset=0.0):
    return {
        "phase": "playback",
        "decision_id": decision_id,
        "servo_feedback_age_s": 0.01,
        "sensor_snapshot_monotonic_s": timestamp,
        "tendons": {
            tendon: {
                "actual_encoder_deg": 100.0 + index + actual_offset,
                "force_n": 1.0 + index,
            }
            for index, tendon in enumerate(TENDONS)
        },
    }


def _vision_payload(frame, capture, inference, angles):
    return {
        "frame_idx": frame,
        "capture_monotonic_s": capture,
        "inference_finished_monotonic_s": inference,
        "inference_ms": (inference - capture) * 1000.0,
        "angles_deg": dict(zip(JOINTS, angles)),
        "min_likelihood": 0.9,
        "valid": True,
        "held_last": False,
        "reason": None,
        "source_elapsed_s_at_capture": capture - 1.0,
    }


def _create_session(
    root: Path,
    *,
    second_reward=True,
    termination="completed",
    reference_trajectory: ExcursionTrajectory | None = None,
) -> Path:
    session = root / "episode-1"
    session.mkdir()
    trajectory = session / "trajectory.csv"
    if reference_trajectory is None:
        with trajectory.open("w", newline="", encoding="utf-8") as handle:
            writer = csv.writer(handle)
            writer.writerow(
                [
                    "elapsed_s",
                    *[f"{tendon}_predicted_excursion_mm" for tendon in TENDONS],
                    "DIP_smoothed_angle_deg",
                    "PIP_smoothed_angle_deg",
                    "MCP_smoothed_angle_deg",
                ]
            )
            writer.writerow([0.0, *([0.0] * 6), 0.0, 0.0, 0.0])
            writer.writerow([1.0, *([10.0] * 6), 10.0, 20.0, 30.0])
    else:
        reference_trajectory.to_csv(trajectory)

    events = []
    events.append(
        _event(
            len(events),
            "control_session_start",
            0.8,
            {
                "session_encoder_origin_deg": {
                    tendon: 100.0 + index
                    for index, tendon in enumerate(TENDONS)
                },
                "calibration": {
                    "encoder_degrees_per_mm": {
                        tendon: 2.0 for tendon in TENDONS
                    }
                },
            },
        )
    )
    events.append(
        _event(
            len(events),
            "vision_observation",
            0.95,
            _vision_payload(1, 0.90, 0.95, (0.0, 0.0, 0.0)),
        )
    )
    # Captured before decision 0, but unavailable until after it.  It must not
    # leak into observation 0.
    events.append(
        _event(
            len(events),
            "vision_observation",
            1.05,
            _vision_payload(2, 0.99, 1.05, (9.0, 9.0, 9.0)),
        )
    )
    events.append(
        _event(
            len(events),
            "decision",
            1.0,
            {
                "decision_id": 0,
                "scheduled_elapsed_s": 0.0,
                "pid_target_reflected_monotonic_s": 1.0,
                "action_excursion_mm": {tendon: 0.0 for tendon in TENDONS},
            },
        )
    )
    events.append(_event(len(events), "pid_tick", 1.0, _tick_payload(1.0, 0)))
    # Capture belongs to action 0, while inference completes after its end.
    events.append(
        _event(
            len(events),
            "vision_observation",
            1.25,
            _vision_payload(3, 1.18, 1.25, (1.0, 1.0, 1.0)),
        )
    )
    events.append(
        _event(
            len(events),
            "decision",
            1.2,
            {
                "decision_id": 1,
                "scheduled_elapsed_s": 0.2,
                "pid_target_reflected_monotonic_s": 1.2,
                "action_excursion_mm": {tendon: 2.0 for tendon in TENDONS},
            },
        )
    )
    events.append(
        _event(len(events), "pid_tick", 1.2, _tick_payload(1.2, 1, 4.0))
    )
    if second_reward:
        events.append(
            _event(
                len(events),
                "vision_observation",
                1.39,
                _vision_payload(4, 1.38, 1.39, (2.0, 4.0, 6.0)),
            )
        )
    events.append(
        _event(len(events), "pid_tick", 1.4, _tick_payload(1.4, 1, 4.0))
    )
    events.append(_event(len(events), "playback_end", 1.4, {}))
    events.append(
        _event(
            len(events),
            "session_end",
            2.0,
            {"termination_reason": termination},
        )
    )
    (session / "raw_events.jsonl").write_text(
        "".join(json.dumps(event) + "\n" for event in events),
        encoding="utf-8",
    )
    manifest = {
        "run_id": "episode-1",
        "input_csv_path": str(trajectory.resolve()),
        "playback_mode": "rl_ff_zoh",
        "termination_reason": termination,
        "raw_logger_stats": {
            "complete": True,
            "enqueued_events": len(events),
            "written_events": len(events),
            "dropped_events": 0,
        },
    }
    (session / "session_manifest.json").write_text(
        json.dumps(manifest),
        encoding="utf-8",
    )
    return session


class AWACDatasetProcessorTest(unittest.TestCase):
    def test_generated_point_to_point_reference_drives_dataset_targets(self):
        class ConstantMomentArm:
            evaluation_ranges_rad = {
                joint: (0.0, 1.0) for joint in JOINTS
            }

            @staticmethod
            def matrix(_angles):
                return [[1.0, 0.0, 0.0] for _ in TENDONS]

        reference = ExcursionTrajectory.from_joint_point_to_point(
            (0.1, 0.2, 0.3),
            (0.5, 0.6, 0.7),
            1.0,
            ConstantMomentArm(),
            sample_hz=50.0,
        )
        with tempfile.TemporaryDirectory() as directory:
            session = _create_session(
                Path(directory), reference_trajectory=reference
            )
            dataset = AWACDatasetProcessor().process([session])

        self.assertEqual(dataset.report["transition_count"], 2)
        expected_target = np.deg2rad(reference.target_angles_deg[9])
        recorded_target = np.asarray(
            [dataset.metadata[0]["reward_target_joint_rad"][joint] for joint in JOINTS]
        )
        np.testing.assert_allclose(recorded_target, expected_target, atol=1e-12)

    def test_observation_is_causal_and_reward_uses_capture_period(self):
        with tempfile.TemporaryDirectory() as directory:
            session = _create_session(Path(directory))
            dataset = AWACDatasetProcessor().process([session])

        self.assertEqual(dataset.observations.shape, (2, 24))
        self.assertEqual(dataset.actions.shape, (2, 6))
        np.testing.assert_allclose(dataset.actions[0], np.zeros(6))
        np.testing.assert_allclose(dataset.actions[1], np.full(6, 2.0))
        self.assertEqual(dataset.metadata[0]["collection_condition"], "D0")
        self.assertEqual(dataset.metadata[0]["action_semantics"], "incremental_excursion_mm")
        self.assertEqual(
            dataset.metadata[0]["confirmed_applied_command"]["physical_applied_status"],
            "unobserved",
        )
        self.assertIn("sensor_freshness", dataset.metadata[0])
        self.assertIn("observed_joint_rad", dataset.metadata[0]["reward_timing"])
        self.assertEqual(dataset.metadata[0]["observation_timing"]["vision_frame_idx"], 1)
        self.assertEqual(dataset.metadata[0]["reward_timing"]["vision_frame_idx"], 3)
        self.assertAlmostEqual(
            dataset.metadata[0]["reward_timing"][
                "reward_available_after_period_end_s"
            ],
            0.05,
            places=6,
        )
        measured_rad = np.deg2rad([1.0, 1.0, 1.0])
        capture_target_rad = np.deg2rad([1.8, 3.6, 5.4])
        expected = -float(np.mean(np.square(measured_rad - capture_target_rad)))
        self.assertAlmostEqual(float(dataset.rewards[0]), expected, places=7)
        self.assertAlmostEqual(
            dataset.metadata[0]["reward_timing"][
                "source_elapsed_s_at_capture"
            ],
            0.18,
            places=6,
        )
        self.assertFalse(bool(dataset.terminated[-1]))
        self.assertTrue(bool(dataset.truncated[-1]))

    def test_missing_reward_capture_is_not_imputed(self):
        with tempfile.TemporaryDirectory() as directory:
            session = _create_session(Path(directory), second_reward=False)
            dataset = AWACDatasetProcessor().process([session])

        self.assertEqual(dataset.report["transition_count"], 1)
        self.assertEqual(
            dataset.report["rejection_counts"][
                "missing_reward_capture_in_action_period"
            ],
            1,
        )
        self.assertFalse(bool(dataset.terminated[0]))
        self.assertFalse(bool(dataset.truncated[0]))

    def test_safety_boundary_requires_explicit_semantics(self):
        unresolved = classify_episode_boundary(
            {"termination_reason": "force_hard_limit"},
            DatasetProcessorConfig(),
        )
        terminal = classify_episode_boundary(
            {"termination_reason": "force_hard_limit"},
            DatasetProcessorConfig(safety_constraint_is_terminal=True),
        )

        self.assertFalse(unresolved.known)
        self.assertFalse(unresolved.terminated)
        self.assertFalse(unresolved.truncated)
        self.assertTrue(terminal.known)
        self.assertTrue(terminal.terminated)
        self.assertFalse(terminal.truncated)

    def test_new_stop_reasons_are_truncated_and_drop_ambiguous_final_interval(self):
        for reason in (
            "safety_filter_rejection",
            "dlc_feedback_unavailable",
            "communication_timeout",
        ):
            boundary = classify_episode_boundary(
                {"termination_reason": reason},
                DatasetProcessorConfig(),
            )
            self.assertTrue(boundary.known, reason)
            self.assertFalse(boundary.terminated, reason)
            self.assertTrue(boundary.truncated, reason)

        with tempfile.TemporaryDirectory() as directory:
            session = _create_session(
                Path(directory),
                termination="communication_timeout",
            )
            dataset = AWACDatasetProcessor().process([session])

        self.assertEqual(dataset.report["transition_count"], 1)
        self.assertEqual(
            dataset.report["rejection_counts"]["unconfirmed_final_action_interval"],
            1,
        )
        self.assertEqual(dataset.metadata[0]["physical_servo_applied_time_status"], "unobserved")

    def test_missing_capture_to_source_time_mapping_is_not_guessed(self):
        with tempfile.TemporaryDirectory() as directory:
            session = _create_session(Path(directory))
            path = session / "raw_events.jsonl"
            events = [
                json.loads(line)
                for line in path.read_text(encoding="utf-8").splitlines()
            ]
            for event in events:
                if (
                    event["event_type"] == "vision_observation"
                    and event["payload"]["frame_idx"] == 3
                ):
                    del event["payload"]["source_elapsed_s_at_capture"]
            path.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            dataset = AWACDatasetProcessor().process([session])

        self.assertEqual(
            dataset.report["rejection_counts"][
                "missing_reward_capture_source_time"
            ],
            1,
        )

    def test_dataset_outputs_are_awac_ready_and_auditable(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            session = _create_session(root)
            dataset = AWACDatasetProcessor().process([session])
            paths = dataset.save(root / "output")
            with np.load(paths["arrays"]) as archive:
                observation_shape = archive["observations"].shape
                action_shape = archive["actions"].shape
            report = json.loads(paths["report"].read_text(encoding="utf-8"))

        self.assertEqual(observation_shape, (2, 24))
        self.assertEqual(action_shape, (2, 6))
        self.assertEqual(report["transition_count"], 2)
        self.assertIn("reward_definition", report)

    def test_broken_session_is_reported_without_imputation_or_batch_abort(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            valid_session = _create_session(root)
            broken_session = root / "broken-session"
            broken_session.mkdir()
            dataset = AWACDatasetProcessor().process(
                [broken_session, valid_session]
            )

        self.assertEqual(dataset.report["transition_count"], 2)
        self.assertEqual(dataset.report["session_count"], 2)
        self.assertEqual(
            dataset.report["rejection_counts"]["session_processing_error"],
            1,
        )
        self.assertEqual(
            dataset.report["sessions"][0]["transition_count"],
            0,
        )
        self.assertIn("processing_error", dataset.report["sessions"][0])


if __name__ == "__main__":
    unittest.main()

