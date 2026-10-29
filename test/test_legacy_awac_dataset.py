import csv
import json
import tempfile
import unittest
from pathlib import Path

import numpy as np

from controller.rl.legacy_awac_dataset import (
    LegacyAWACDatasetProcessor,
    OBSERVATION_FEATURES,
)
from controller.rl.offline_awac_trial import build_replay_buffer
from controller.servo_mapping import TENDONS


class LegacyAWACDatasetTest(unittest.TestCase):
    def test_native_tick_transition_is_causal_and_not_five_hz_zoh(self):
        with tempfile.TemporaryDirectory() as tmp:
            root = Path(tmp) / "logs" / "csv_player_gui"
            session = root / "Family" / "run"
            session.mkdir(parents=True)
            source = Path(tmp) / "source.csv"
            source_rows = []
            for index, elapsed in enumerate((0.0, 0.05, 0.10, 0.15)):
                row = {"elapsed_s": elapsed}
                for joint_offset, joint in enumerate(("DIP", "PIP", "MCP")):
                    row[f"{joint}_smoothed_angle_rad"] = 0.1 * index + joint_offset
                for tendon_index, tendon in enumerate(TENDONS):
                    row[f"{tendon}_predicted_excursion_mm"] = index * (tendon_index + 1) * 0.01
                source_rows.append(row)
            _write_csv(source, source_rows)

            controls = []
            for index, elapsed in enumerate((0.0, 0.05, 0.10, 0.15)):
                row = {
                    "phase": "playback",
                    "elapsed_s": elapsed,
                    "control_tick_monotonic_s": 10.0 + elapsed,
                    "command_submitted_monotonic_s": 10.0 + elapsed,
                    "servo_feedback_received_monotonic_s": 9.99 + elapsed,
                    "servo_telemetry_age_s": 0.01,
                }
                for tendon_index, tendon in enumerate(TENDONS):
                    action = index * (tendon_index + 1) * 0.01
                    row[f"{tendon}_target_excursion_mm"] = action
                    row[f"{tendon}_actual_encoder_deg"] = action * 2.8648
                    row[f"{tendon}_force_n"] = 1.0
                    row[f"{tendon}_encoder_received_monotonic_s"] = 9.995 + elapsed
                    row[f"{tendon}_force_status"] = "OK"
                controls.append(row)
            _write_csv(session / "control_trace.csv", controls)
            angles = [
                {
                    "frame_idx": 1,
                    "captured_at_monotonic": 9.99,
                    "inference_finished_at_monotonic": 9.995,
                    "valid": 1,
                    "held_last": 0,
                    "DIP_deg": 0,
                    "PIP_deg": 0,
                    "MCP_deg": 0,
                },
                {
                    "frame_idx": 2,
                    "captured_at_monotonic": 10.07,
                    # This reward measurement is not available at the next
                    # decision and therefore must not enter next_observation.
                    "inference_finished_at_monotonic": 10.12,
                    "valid": 1,
                    "held_last": 0,
                    "DIP_deg": 1,
                    "PIP_deg": 2,
                    "MCP_deg": 3,
                },
            ]
            _write_csv(session / "angle_trace.csv", angles)
            (session / "session_manifest.json").write_text(
                json.dumps(
                    {
                        "run_id": "run",
                        "completed": True,
                        "termination_reason": "completed",
                        "input_csv_path": str(source),
                    }
                ),
                encoding="utf-8",
            )
            dataset = LegacyAWACDatasetProcessor(root).process()
            self.assertEqual(len(dataset.metadata), 1)
            self.assertEqual(dataset.observations.shape, (1, len(OBSERVATION_FEATURES)))
            meta = dataset.metadata[0]
            self.assertFalse(meta["action_is_5hz_zoh"])
            self.assertAlmostEqual(meta["action_period_s"], 0.05)
            self.assertIsNone(meta["physical_servo_application_monotonic_s"])
            self.assertFalse(meta["reward"]["inference_available_by_next_decision"])
            self.assertFalse(meta["episode_boundary"]["terminated"])
            self.assertTrue(meta["episode_boundary"]["truncated"])
            # next_observation must still use frame 1, not future inference frame 2.
            self.assertAlmostEqual(dataset.next_observations[0, 6], 0.0)
            np.testing.assert_allclose(dataset.actions_mm[0], np.arange(1, 7) * 0.01)

    def test_replay_buffer_preserves_explicit_next_observation(self):
        data = {
            "observations": np.zeros((2, 30), dtype=np.float32),
            "actions_mm": np.ones((2, 6), dtype=np.float32),
            "rewards": np.asarray([[-1.0], [-2.0]], dtype=np.float32),
            "next_observations": np.full((2, 30), 2.0, dtype=np.float32),
        }
        replay = build_replay_buffer(data)
        self.assertEqual(replay.transition_count, 2)
        transition = replay.transition_picker(replay.episodes[0], 0)
        np.testing.assert_allclose(transition.next_observation, 2.0)
        self.assertEqual(transition.terminal, 0.0)


def _write_csv(path: Path, rows):
    with path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


if __name__ == "__main__":
    unittest.main()
