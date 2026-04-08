from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.mavlink_real.run_mavlink_schedule import (
    ExpandedEvent,
    expand_schedule,
    load_schedule,
    parse_sender_output,
    run_one_event,
    validate_serialized_run_rows,
)
from tools.mavlink_real.runtime_layout import container_identity_logs_dir, container_identity_send_log_path
from tools.mavlink_real.schedule_profiles import build_benign_rows, write_schedule_csv


class MavlinkScheduleRunnerTests(unittest.TestCase):
    def test_load_schedule_round_trips_rows(self) -> None:
        rows = build_benign_rows(target_rows=12, seed=7)
        with tempfile.TemporaryDirectory() as tmpdir:
            schedule_path = Path(tmpdir) / "schedule.csv"
            write_schedule_csv(schedule_path, rows)
            loaded = load_schedule(schedule_path)
        self.assertEqual(len(loaded), len(rows))
        self.assertEqual(loaded[0].command_family, rows[0]["command_family"])
        self.assertEqual(loaded[0].target_endpoint, rows[0]["target_endpoint"])

    def test_expand_schedule_offsets_virtual_day(self) -> None:
        rows = build_benign_rows(target_rows=4, seed=7)
        with tempfile.TemporaryDirectory() as tmpdir:
            schedule_path = Path(tmpdir) / "schedule.csv"
            write_schedule_csv(schedule_path, rows)
            events = load_schedule(schedule_path)
        expanded = expand_schedule(events, cycles=2)
        self.assertEqual(len(expanded), 8)
        self.assertEqual(expanded[0].virtual_day, 0)
        self.assertEqual(expanded[4].virtual_day, 1)
        self.assertEqual(expanded[4].absolute_virtual_seconds - expanded[0].absolute_virtual_seconds, 86400)

    def test_parse_sender_output_reads_last_json_line(self) -> None:
        row = {"send_id": "abc123", "command": "PARAM_REQUEST_READ"}
        payload = "progress\n" + json.dumps(row) + "\n"
        self.assertEqual(parse_sender_output(payload), row)

    def test_run_one_event_passes_endpoint_and_identity_log_paths(self) -> None:
        event = ExpandedEvent(
            absolute_virtual_seconds=10,
            virtual_day=0,
            virtual_time="00:00:10",
            virtual_seconds=10,
            source_service="ops_primary",
            target_service="mavlink_vehicle",
            target_endpoint="tcp:mavlink_vehicle:5760",
            command="PARAM_REQUEST_READ",
            command_family="read_only_inspection",
            arguments={"param_id": "FENCE_RADIUS", "param_index": -1},
            meta={"class_name": "benign", "run_id": 4},
        )
        completed = mock.Mock()
        completed.returncode = 0
        completed.stdout = json.dumps({"send_id": "abc123", "command": event.command})
        completed.stderr = ""

        with mock.patch("tools.mavlink_real.run_mavlink_schedule.subprocess.run", return_value=completed) as run_mock:
            row = run_one_event(
                REPO_ROOT / "orchestration" / "docker-compose.mavlink-real.yml",
                8.0,
                "/workspace/tools/mavlink_real/send_mavlink_events.py",
                event,
                target_stream_id="mavlink_vehicle@tcp:mavlink_vehicle:5760",
                target_stream_index=2,
            )

        self.assertEqual(row["send_id"], "abc123")
        invoked = run_mock.call_args.args[0]
        self.assertIn("--target-endpoint", invoked)
        self.assertIn("tcp:mavlink_vehicle:5760", invoked)
        self.assertIn("--command-family", invoked)
        self.assertIn("read_only_inspection", invoked)
        self.assertIn(str(container_identity_logs_dir("ops_primary")), invoked)
        self.assertIn(str(container_identity_send_log_path("ops_primary")), invoked)

    def test_validate_serialized_run_rows_accepts_monotonic_rows(self) -> None:
        rows = [
            {
                "target_service": "mavlink_vehicle",
                "target_stream_id": "mavlink_vehicle@tcp:mavlink_vehicle:5760",
                "target_stream_index": "0",
                "command": "PARAM_REQUEST_READ",
                "send_start_ms": "1000",
                "send_end_ms": "1050",
            },
            {
                "target_service": "mavlink_vehicle",
                "target_stream_id": "mavlink_vehicle@tcp:mavlink_vehicle:5760",
                "target_stream_index": "1",
                "command": "MISSION_REQUEST_LIST",
                "send_start_ms": "1100",
                "send_end_ms": "1160",
            },
        ]
        validate_serialized_run_rows(rows)

    def test_validate_serialized_run_rows_rejects_overlap(self) -> None:
        rows = [
            {
                "target_service": "mavlink_vehicle",
                "target_stream_id": "mavlink_vehicle@tcp:mavlink_vehicle:5760",
                "target_stream_index": "0",
                "command": "PARAM_REQUEST_READ",
                "send_start_ms": "1000",
                "send_end_ms": "1100",
            },
            {
                "target_service": "mavlink_vehicle",
                "target_stream_id": "mavlink_vehicle@tcp:mavlink_vehicle:5760",
                "target_stream_index": "1",
                "command": "PARAM_SET",
                "send_start_ms": "1050",
                "send_end_ms": "1150",
            },
        ]
        with self.assertRaises(SystemExit) as exc:
            validate_serialized_run_rows(rows)
        self.assertIn("serialized-per-target invariant", str(exc.exception))


if __name__ == "__main__":
    unittest.main()
