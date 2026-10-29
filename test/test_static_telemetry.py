from __future__ import annotations

import json
import sys
import tempfile
import time
import types
import unittest
from pathlib import Path
from types import SimpleNamespace
from unittest.mock import patch

from controller.csv_player.gui import (
    JointMeasurementSnapshot,
    LatestJointState,
    _validate_and_log_point_to_point_start_state,
)
from controller.csv_player.record_static_telemetry import (
    analyze_static_telemetry,
    circular_count_delta,
)
from controller.servo.hardware_session import HardwareFeedback, ServoHardwareSession
from controller.servo_mapping import (
    ENCODER_SENSOR_IDS_BY_SERVO_ID,
    SERVO_IDS,
    SERVO_IDS_BY_TENDON,
    SERVO_NAMES_BY_ID,
    TENDONS,
)


class StaticTelemetryTest(unittest.TestCase):
    def test_as5048a_count_delta_handles_wraparound(self):
        self.assertEqual(circular_count_delta(16383, 0, 16384), 1)
        self.assertEqual(circular_count_delta(0, 16383, 16384), 1)
        self.assertEqual(circular_count_delta(16383, 2, 16384), 3)

    def test_analysis_reports_lagged_count_deltas_dlc_drift_and_guard_rates(self):
        events = []
        for index, (servo_count, encoder_count) in enumerate(
            zip((100, 101, 103, 103, 104, 104, 105), (16381, 16383, 0, 1, 2, 3, 4))
        ):
            timestamp = index * 0.2
            events.append(
                {
                    "event_type": "static_telemetry_sample",
                    "payload": {
                        "signal": "servo",
                        "tendon": "FDP",
                        "received_monotonic_s": timestamp,
                        "position_count": servo_count,
                        "valid": True,
                    },
                }
            )
            events.append(
                {
                    "event_type": "static_telemetry_sample",
                    "payload": {
                        "signal": "encoder",
                        "tendon": "FDP",
                        "received_monotonic_s": timestamp,
                        "raw_count": encoder_count,
                        "valid": True,
                    },
                }
            )
            events.append(
                {
                    "event_type": "static_telemetry_sample",
                    "payload": {
                        "signal": "dlc",
                        "captured_at_monotonic_s": timestamp,
                        "angles_deg": {
                            "DIP": 10.0 + index * 0.1,
                            "PIP": 20.0,
                            "MCP": 30.0,
                        },
                        "valid": True,
                    },
                }
            )

        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "static.jsonl"
            source.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            report = analyze_static_telemetry(source, lags_s=(0.2,))

        encoder = report["encoder"]["FDP"]["lagged_position_deltas"][0]
        servo = report["servo"]["FDP"]["lagged_position_deltas"][0]
        dip = report["dlc"]["DIP"]
        self.assertGreater(encoder["pair_count"], 0)
        self.assertEqual(encoder["abs_delta_count_or_deg"]["max"], 2.0)
        self.assertIsNotNone(encoder["current_guard_rejection_fraction"])
        self.assertEqual(servo["abs_delta_count_or_deg"]["max"], 2.0)
        self.assertAlmostEqual(dip["linear_drift_deg_per_min"], 30.0, places=5)
        self.assertIsNotNone(dip["median_absolute_deviation_deg"])
        self.assertTrue(report["tolerance_decision"] is None)
        self.assertIn("current_q0_guard_static_rejection_estimate", report["dlc"])

    def test_analysis_uses_actual_lags_from_p2p_alignment_events(self):
        with tempfile.TemporaryDirectory() as directory:
            root = Path(directory)
            source = root / "static.jsonl"
            source.write_text(
                "\n".join(
                    json.dumps(
                        {
                            "event_type": "static_telemetry_sample",
                            "payload": {
                                "signal": "encoder",
                                "tendon": "FDP",
                                "received_monotonic_s": index * 0.1,
                                "raw_count": index,
                                "valid": True,
                            },
                        }
                    )
                    for index in range(10)
                )
                + "\n",
                encoding="utf-8",
            )
            p2p = root / "raw_events.jsonl"
            p2p.write_text(
                json.dumps(
                    {
                        "event_type": "point_to_point_start_alignment_check",
                        "payload": {
                            "time_deltas_s": {
                                "servo_by_tendon": {"FDP": 0.25},
                                "encoder_by_tendon": {"FDP": 0.3},
                                "dlc_q0_to_runner": 0.8,
                            },
                            "runner_dlc_snapshot_used_for_validation": True,
                        },
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            report = analyze_static_telemetry(
                source,
                p2p_event_paths=(p2p,),
            )

        self.assertEqual(
            report["encoder"]["FDP"]["lagged_position_deltas"][0]["requested_lag_s"],
            0.3,
        )
        self.assertEqual(
            report["servo"]["FDP"]["lagged_position_deltas"][0]["requested_lag_s"],
            0.25,
        )
        self.assertEqual(
            report["dlc"]["DIP"]["lagged_angle_deltas"][0]["requested_lag_s"],
            0.8,
        )

    def test_five_frame_spread_is_separate_from_q0_to_runner_change(self):
        events = []
        for index in range(7):
            dip = 10.2 if index >= 5 else 10.0
            events.append(
                {
                    "event_type": "static_telemetry_sample",
                    "payload": {
                        "signal": "dlc",
                        "captured_at_monotonic_s": index * 0.2,
                        "angles_deg": {"DIP": dip, "PIP": 20.0, "MCP": 30.0},
                        "p2p_angles_deg": {"DIP": dip, "PIP": 20.0, "MCP": 30.0},
                        "valid": True,
                        "p2p_valid": True,
                        "p2p_held_last": False,
                    },
                }
            )
        with tempfile.TemporaryDirectory() as directory:
            source = Path(directory) / "dlc.jsonl"
            source.write_text(
                "".join(json.dumps(event) + "\n" for event in events),
                encoding="utf-8",
            )
            report = analyze_static_telemetry(source, lags_s=(0.2,))

        result = report["dlc"]["current_q0_guard_static_rejection_estimate"][0]
        self.assertEqual(result["spread_gate_reject_count"], 0)
        self.assertEqual(result["q0_to_runner_gate_reject_count"], 1)
        self.assertEqual(result["combined_reject_count"], 1)

    def test_read_only_hardware_session_captures_samples_without_servo_writes(self):
        serial_transports = {}

        class FakeSerialTransport:
            def __init__(self, port, **_kwargs):
                self.port_name = port
                self.writes = []
                self.closed = False
                serial_transports[port] = self

            def open(self):
                return self

            def read_line(self):
                time.sleep(0.001)
                return self.port_name

            def write_line(self, line):
                self.writes.append(str(line))

            def close(self):
                self.closed = True

        class FakeServoSystem:
            def __init__(self, *, servo_ids, **_kwargs):
                self.servo_ids = tuple(servo_ids)
                self.count = len(self.servo_ids)

            def index_for_servo_id(self, servo_id):
                return self.servo_ids.index(servo_id)

        class FakeServoDriver:
            def __init__(self, _transport, servo_system):
                self.system = servo_system

            def parse_line(self, _line):
                return {
                    "kind": "telemetry",
                    "device_ms": 10,
                    "positions": [100] * 6,
                    "raw_positions": [20] * 6,
                    "loads": [0] * 6,
                    "speeds": [0] * 6,
                    "valid": [True] * 6,
                    "parts": ["10", "V2"],
                }

        class FakeEncoderDriver:
            def __init__(self, transport):
                self.transport = transport
                self.sensor_id = 0

            def stop_stream(self):
                self.transport.write_line("STOP")

            def set_rate_hz(self, hz):
                self.transport.write_line(f"RATE,{hz}")

            def start_all_stream(self):
                self.transport.write_line("START_ALL")

            def parse_line(self, _line):
                self.sensor_id = (self.sensor_id % 6) + 1
                return (
                    "sensor_data",
                    {
                        "sensor_id": self.sensor_id,
                        "device_ms": 11,
                        "raw": 100,
                        "angle_deg": 2.0,
                        "relative_deg": 2.0,
                        "zero_deg": 0.0,
                        "status": "OK",
                    },
                )

        drivers_module = types.ModuleType("drivers")
        drivers_module.AS5048ABridgeDriver = FakeEncoderDriver
        drivers_module.SerialTransport = FakeSerialTransport
        drivers_module.ServoControllerDriver = FakeServoDriver
        drivers_module.ServoSystem = FakeServoSystem
        with patch.dict(sys.modules, {"drivers": drivers_module}):
            hardware = ServoHardwareSession(
                servo_ids=SERVO_IDS,
                servo_names=SERVO_NAMES_BY_ID,
                encoder_sensor_ids=tuple(
                    ENCODER_SENSOR_IDS_BY_SERVO_ID[servo_id]
                    for servo_id in SERVO_IDS
                ),
                servo_port="MOCK_SERVO",
                encoder_port="MOCK_ENCODER",
                serial_timeout_s=0.01,
                read_only=True,
                capture_telemetry=True,
            )
            hardware.connect(startup_timeout_s=1.0)
            self.assertIsNone(hardware.dispatcher)
            self.assertIsNone(hardware.sender)
            self.assertTrue(hardware.drain_telemetry_samples())
            with self.assertRaisesRegex(RuntimeError, "disabled in read-only mode"):
                hardware.submit_speed_to_servos({servo_id: 1 for servo_id in SERVO_IDS})
            hardware.close()

        self.assertEqual(serial_transports["MOCK_SERVO"].writes, [])
        self.assertIn("START_ALL", serial_transports["MOCK_ENCODER"].writes)

    def test_p2p_alignment_audit_records_pass_and_rejection_with_deltas(self):
        now = time.monotonic()
        q0 = JointMeasurementSnapshot(
            angles_rad=(0.1, 0.2, 0.3),
            captured_at=now,
            inference_finished_at=now + 0.001,
            valid=True,
            held_last=False,
            reason=None,
        )
        latest = LatestJointState()
        latest.update(
            SimpleNamespace(
                flexion_angles_rad=q0.angles_rad,
                captured_at=now + 0.05,
                inference_finished_at=now + 0.051,
                valid=True,
                held_last=False,
                reason=None,
            )
        )
        pre_counts = {tendon: 16383 for tendon in TENDONS}
        pre_snapshot = {
            "servo_feedback_received_at": now + 0.01,
            "servo_position_by_servo_id": {servo_id: 100 for servo_id in SERVO_IDS},
            "encoder_raw_count_by_tendon": pre_counts,
            "encoder_relative_deg_by_tendon": {
                tendon: 16383 * 360.0 / 16384 for tendon in TENDONS
            },
            "encoder_received_at_by_tendon": {
                tendon: now + 0.02 for tendon in TENDONS
            },
        }
        samples = {
            ENCODER_SENSOR_IDS_BY_SERVO_ID[servo_id]: {
                "status": "OK",
                "relative_deg": 0.0,
                "raw": 0,
            }
            for servo_id in SERVO_IDS_BY_TENDON
        }
        feedback = HardwareFeedback(
            servo={"positions": [101] * 6, "valid": [True] * 6},
            servo_received_at=now + 0.10,
            encoder_by_sensor=samples,
            encoder_received_at={
                sensor_id: now + 0.09 for sensor_id in samples
            },
            sampled_at=now + 0.10,
        )

        class FakeHardware:
            @staticmethod
            def position_by_servo_id(_feedback):
                return {servo_id: 101.0 for servo_id in SERVO_IDS}

        class FakeRawLogger:
            def __init__(self):
                self.events = []

            def log(self, event_type, *, event_monotonic_s, payload):
                self.events.append(
                    (event_type, event_monotonic_s, dict(payload))
                )

        logger = FakeRawLogger()
        runner_origin = {servo_id: 0.0 for servo_id in SERVO_IDS}
        kwargs = {
            "latest_joint_state": latest,
            "q0_snapshot": q0,
            "q0_spread_deg": (0.1, 0.1, 0.1),
            "encoder_origin_snapshot": pre_snapshot,
            "feedback": feedback,
            "runner_encoder_origin_deg_by_servo_id": runner_origin,
            "encoder_directions": (1.0,) * len(TENDONS),
            "max_joint_age_s": 0.35,
            "raw_event_logger": logger,
        }
        passed = _validate_and_log_point_to_point_start_state(
            FakeHardware(), **kwargs
        )
        self.assertEqual(passed["status"], "passed")
        success_payload = logger.events[-1][2]
        self.assertEqual(success_payload["status"], "passed")
        self.assertEqual(success_payload["position_deltas"]["encoder_raw_count_circular_by_tendon"]["FDP"], 1)
        self.assertAlmostEqual(success_payload["time_deltas_s"]["encoder_by_tendon"]["FDP"], 0.07)

        changed_samples = {key: dict(value) for key, value in samples.items()}
        changed_samples[ENCODER_SENSOR_IDS_BY_SERVO_ID[SERVO_IDS_BY_TENDON[0]]]["raw"] = 9
        rejected_feedback = HardwareFeedback(
            servo=feedback.servo,
            servo_received_at=feedback.servo_received_at,
            encoder_by_sensor=changed_samples,
            encoder_received_at=feedback.encoder_received_at,
            sampled_at=feedback.sampled_at,
        )
        with self.assertRaisesRegex(TimeoutError, "Encoder origin changed"):
            _validate_and_log_point_to_point_start_state(
                FakeHardware(), **{**kwargs, "feedback": rejected_feedback}
            )
        failure_payload = logger.events[-1][2]
        self.assertEqual(failure_payload["status"], "rejected")
        self.assertIn("reason", failure_payload)
        self.assertEqual(
            failure_payload["position_deltas"]["encoder_raw_count_circular_by_tendon"]["FDP"],
            10,
        )


if __name__ == "__main__":  # pragma: no cover
    unittest.main()
