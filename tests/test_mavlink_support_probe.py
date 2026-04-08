from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.mavlink_real.support_probe import assert_actual_run_observability, build_actual_run_observability_report, collect_capture_drain_failures


class MavlinkSupportProbeTests(unittest.TestCase):
    def test_actual_run_observability_requires_wire_and_state_flags(self) -> None:
        run_rows = [
            {
                "send_id": "send-001",
                "source_service": "ops_primary",
                "target_service": "mavlink_vehicle",
                "command": "PARAM_REQUEST_READ",
                "meta_json": json.dumps({"class_label": 0, "intent_context": "benign_clean"}),
                "gds_accept": "1",
                "sat_success": "1",
                "timeout": "0",
                "reason": "completed",
            },
            {
                "send_id": "send-002",
                "source_service": "ops_primary",
                "target_service": "mavlink_vehicle",
                "command": "PARAM_REQUEST_READ",
                "meta_json": json.dumps({"class_label": 0, "intent_context": "benign_noisy"}),
                "gds_accept": "1",
                "sat_success": "0",
                "timeout": "0",
                "reason": "command_ack_reject",
            },
        ]
        observations = [
            {
                "row_index": 0,
                "request_wire_seen": True,
                "response_direction_seen": False,
                "telemetry_recent": True,
                "state_snapshot_seen": False,
            },
            {
                "row_index": 1,
                "request_wire_seen": True,
                "response_direction_seen": True,
                "telemetry_recent": True,
                "state_snapshot_seen": True,
            },
        ]
        report = build_actual_run_observability_report(run_rows, observations)
        self.assertEqual(report["summary"]["benign_rows"], 2)
        self.assertEqual(report["summary"]["observability_failed_rows"], 1)
        self.assertEqual(report["summary"]["nonclean_rows"], 1)
        failing = next(item for item in report["rows"] if item["send_id"] == "send-001")
        self.assertEqual(failing["missing_observability"], ["response_direction_seen", "state_snapshot_seen"])
        with self.assertRaises(SystemExit) as exc:
            assert_actual_run_observability(report)
        self.assertIn("Captured MAVLink benign run is missing required observability or manifest coverage", str(exc.exception))

    def test_collect_capture_drain_failures_marks_missing_fields(self) -> None:
        expected_rows = [
            {
                "send_id": "send-100",
                "command": "PARAM_REQUEST_READ",
                "source_service": "ops_primary",
                "target_service": "mavlink_vehicle",
                "timeout": 0,
            }
        ]
        observations = [
            {
                "send_id": "send-100",
                "request_wire_seen": True,
                "response_direction_seen": False,
                "telemetry_recent": False,
                "state_snapshot_seen": True,
            }
        ]
        failures = collect_capture_drain_failures(expected_rows, observations)
        self.assertEqual(
            failures,
            [
                {
                    "send_id": "send-100",
                    "command": "PARAM_REQUEST_READ",
                    "source_service": "ops_primary",
                    "target_service": "mavlink_vehicle",
                    "missing": ["response_direction_seen", "telemetry_recent"],
                }
            ],
        )


if __name__ == "__main__":
    unittest.main()
