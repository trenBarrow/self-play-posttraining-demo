from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime import FEATURE_NAMES, packets_to_transactions, transactions_to_rows
from tools.fprime_real.generate_dataset import build_shared_fprime_artifact_layers
from tools.fprime_real.runtime_layout import host_downlink_records_path, host_event_log_path


class RuntimePhase2Tests(unittest.TestCase):
    def test_transactions_snapshot_target_and_peer_telemetry(self) -> None:
        packets = [
            {
                "ts_ms": 1000,
                "packet_kind": "telemetry",
                "src": "fprime_a",
                "dst": "gds",
                "service": "systemResources",
                "payload": {"cpu_total_pct": 11.0},
                "node_service": "fprime_a",
            },
            {
                "ts_ms": 1050,
                "packet_kind": "telemetry",
                "src": "fprime_a",
                "dst": "gds",
                "service": "cmdDisp",
                "payload": {"cmds_dispatched_total": 4.0},
                "node_service": "fprime_a",
            },
            {
                "ts_ms": 1100,
                "packet_kind": "telemetry",
                "src": "fprime_b",
                "dst": "gds",
                "service": "systemResources",
                "payload": {"cpu_total_pct": 7.0},
                "node_service": "fprime_b",
            },
            {
                "ts_ms": 60000,
                "packet_kind": "telemetry",
                "src": "fprime_a",
                "dst": "gds",
                "service": "cmdDisp",
                "payload": {"cmds_dispatched_total": 9.0},
                "node_service": "fprime_a",
            },
            {
                "ts_ms": 61000,
                "packet_kind": "request",
                "src": "ops_b1",
                "dst": "fprime_a",
                "target_service": "fprime_a",
                "service": "cmdDisp",
                "command": "cmdDisp.CMD_NO_OP",
                "label": 0,
                "episode_label": 0,
                "episode_kind": "benign",
                "episode_id": 0,
                "session_id": "ops_b1-0000",
                "txn_id": "0-000001-ops_b1",
                "target_stream_id": "fprime_a:50050",
                "target_stream_index": 0,
                "attack_family": "none",
                "phase": "startup",
                "actor": "ops_b1",
                "actor_role": "ops_primary",
                "actor_trust": 0.97,
                "args": {},
                "bytes_on_wire": 40,
            },
            {
                "ts_ms": 61020,
                "packet_kind": "uplink",
                "src": "gds",
                "dst": "fprime_a",
                "target_service": "fprime_a",
                "service": "cmdDisp",
                "command": "cmdDisp.CMD_NO_OP",
                "label": 0,
                "episode_label": 0,
                "episode_kind": "benign",
                "episode_id": 0,
                "session_id": "ops_b1-0000",
                "txn_id": "0-000001-ops_b1",
                "target_stream_id": "fprime_a:50050",
                "target_stream_index": 0,
            },
            {
                "ts_ms": 61045,
                "packet_kind": "sat_response",
                "src": "fprime_a",
                "dst": "gds",
                "target_service": "fprime_a",
                "service": "cmdDisp",
                "command": "cmdDisp.CMD_NO_OP",
                "label": 0,
                "episode_label": 0,
                "episode_kind": "benign",
                "episode_id": 0,
                "session_id": "ops_b1-0000",
                "txn_id": "0-000001-ops_b1",
                "target_stream_id": "fprime_a:50050",
                "target_stream_index": 0,
                "sat_success": 1,
                "response_code": 0,
                "reason": "completed",
            },
            {
                "ts_ms": 61070,
                "packet_kind": "final",
                "src": "fprime_a",
                "dst": "ops_b1",
                "target_service": "fprime_a",
                "service": "cmdDisp",
                "command": "cmdDisp.CMD_NO_OP",
                "label": 0,
                "episode_label": 0,
                "episode_kind": "benign",
                "episode_id": 0,
                "session_id": "ops_b1-0000",
                "txn_id": "0-000001-ops_b1",
                "target_stream_id": "fprime_a:50050",
                "target_stream_index": 0,
                "bytes_on_wire": 12,
                "gds_accept": 1,
                "sat_success": 1,
                "timeout": 0,
                "response_code": 0,
                "reason": "completed",
                "response_direction_seen": 1,
                "final_observed_on_wire": 1,
                "txn_warning_events": 0,
                "txn_error_events": 0,
            },
        ]

        transactions = packets_to_transactions(packets)
        self.assertEqual(len(transactions), 1)
        tx = transactions[0]
        self.assertEqual(tx["target_service"], "fprime_a")
        self.assertEqual(tx["target_stream_id"], "fprime_a:50050")
        self.assertEqual(tx["target_stream_index"], 0.0)
        self.assertEqual(tx["target_cmds_dispatched_1m"], 5.0)
        self.assertEqual(tx["peer_cpu_total_pct"], 7.0)
        self.assertEqual(tx["request_to_uplink_ms"], 20.0)
        self.assertEqual(tx["uplink_to_sat_response_ms"], 25.0)
        self.assertEqual(tx["sat_response_to_final_ms"], 25.0)

        rows = transactions_to_rows(transactions)
        self.assertEqual(len(rows), 1)
        row = rows[0]
        self.assertEqual(row["target_node_id"], 1.0)
        self.assertEqual(row["target_stream_id"], "fprime_a:50050")
        self.assertEqual(row["target_stream_index"], 0)
        self.assertEqual(row["target_cmds_dispatched_1m"], 5.0)
        self.assertEqual(row["peer_cpu_total_pct"], 7.0)
        self.assertEqual(row["response_direction_seen"], 1.0)
        self.assertIn("target_cpu_total_pct", FEATURE_NAMES)
        self.assertIn("peer_cpu_total_pct", FEATURE_NAMES)

    def test_packets_to_transactions_rejects_same_target_overlap(self) -> None:
        packets = [
            {
                "ts_ms": 1000,
                "packet_kind": "request",
                "src": "ops_b1",
                "dst": "fprime_a",
                "target_service": "fprime_a",
                "target_stream_id": "fprime_a:50050",
                "target_stream_index": 0,
                "service": "cmdDisp",
                "command": "cmdDisp.CMD_NO_OP",
                "label": 0,
                "episode_label": 0,
                "episode_kind": "benign",
                "episode_id": 0,
                "session_id": "ops_b1-0000",
                "txn_id": "0-000001-ops_b1",
                "args": {},
                "bytes_on_wire": 40,
            },
            {
                "ts_ms": 1005,
                "packet_kind": "request",
                "src": "ops_a1",
                "dst": "fprime_a",
                "target_service": "fprime_a",
                "target_stream_id": "fprime_a:50050",
                "target_stream_index": 1,
                "service": "cmdDisp",
                "command": "cmdDisp.CMD_NO_OP",
                "label": 0,
                "episode_label": 0,
                "episode_kind": "benign",
                "episode_id": 1,
                "session_id": "ops_a1-0001",
                "txn_id": "0-000002-ops_a1",
                "args": {},
                "bytes_on_wire": 40,
            },
        ]
        with self.assertRaises(SystemExit) as exc:
            packets_to_transactions(packets)
        self.assertIn("serialized-per-target invariant", str(exc.exception))

    def test_build_shared_fprime_artifact_layers_emits_canonical_rows_with_real_history_context(self) -> None:
        packets = [
            {
                "ts_ms": 1000,
                "packet_kind": "telemetry",
                "src": "fprime_a",
                "dst": "gds",
                "service": "systemResources",
                "payload": {"cpu_total_pct": 11.0},
                "node_service": "fprime_a",
                "run_id": 0,
            },
            {
                "ts_ms": 61000,
                "packet_kind": "request",
                "src": "ops_b1",
                "dst": "fprime_a",
                "src_ip": "192.168.144.22",
                "dst_ip": "192.168.144.2",
                "src_port": 50101,
                "dst_port": 50050,
                "target_service": "fprime_a",
                "service": "cmdDisp",
                "command": "cmdDisp.CMD_NO_OP",
                "label": 0,
                "episode_label": 0,
                "episode_kind": "benign",
                "episode_id": 0,
                "run_id": 0,
                "session_id": "ops_b1-0000",
                "txn_id": "0-000001-ops_b1",
                "send_id": "send-0001",
                "target_stream_id": "fprime_a:50050",
                "target_stream_index": 0,
                "attack_family": "none",
                "phase": "startup",
                "actor": "ops_b1",
                "actor_role": "ops_primary",
                "actor_trust": 0.97,
                "args": {},
                "bytes_on_wire": 40,
                "observed_on_wire": 1,
                "ts_source": "pcap",
                "bytes_source": "pcap",
            },
            {
                "ts_ms": 61020,
                "packet_kind": "uplink",
                "src": "gds",
                "dst": "fprime_a",
                "src_ip": "192.168.144.100",
                "dst_ip": "192.168.144.2",
                "src_port": 50050,
                "dst_port": 50050,
                "target_service": "fprime_a",
                "service": "cmdDisp",
                "command": "cmdDisp.CMD_NO_OP",
                "label": 0,
                "episode_label": 0,
                "episode_kind": "benign",
                "episode_id": 0,
                "run_id": 0,
                "session_id": "ops_b1-0000",
                "txn_id": "0-000001-ops_b1",
                "send_id": "send-0001",
                "target_stream_id": "fprime_a:50050",
                "target_stream_index": 0,
                "observed_on_wire": 1,
                "ts_source": "pcap",
                "bytes_source": "pcap",
            },
            {
                "ts_ms": 61070,
                "packet_kind": "final",
                "src": "fprime_a",
                "dst": "ops_b1",
                "src_ip": "192.168.144.2",
                "dst_ip": "192.168.144.22",
                "src_port": 50050,
                "dst_port": 50101,
                "target_service": "fprime_a",
                "service": "cmdDisp",
                "command": "cmdDisp.CMD_NO_OP",
                "label": 0,
                "episode_label": 0,
                "episode_kind": "benign",
                "episode_id": 0,
                "run_id": 0,
                "session_id": "ops_b1-0000",
                "txn_id": "0-000001-ops_b1",
                "send_id": "send-0001",
                "target_stream_id": "fprime_a:50050",
                "target_stream_index": 0,
                "bytes_on_wire": 12,
                "gds_accept": 1,
                "sat_success": 1,
                "timeout": 0,
                "response_code": 0,
                "reason": "completed",
                "response_direction_seen": 1,
                "final_observed_on_wire": 1,
                "txn_warning_events": 0,
                "txn_error_events": 0,
                "observed_on_wire": 1,
                "ts_source": "pcap",
                "bytes_source": "pcap",
            },
        ]

        transactions = packets_to_transactions(packets, reset_key="run_id")
        feature_rows = transactions_to_rows(transactions, reset_key="run_id")

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            manifest_path = tmp_path / "manifest.json"
            manifest_path.write_text("[]\n", encoding="utf-8")
            schedule_path = tmp_path / "schedule.csv"
            run_log_path = tmp_path / "run_log.csv"
            command_log_path = tmp_path / "command.log"
            send_log_path = tmp_path / "send_log.jsonl"
            traffic_pcap = tmp_path / "traffic.pcap"
            runtime_root = tmp_path / "runtime_root"
            schedule_path.write_text("virtual_time,command\n00:00:20,cmdDisp.CMD_NO_OP\n", encoding="utf-8")
            run_log_path.write_text("command\ncmdDisp.CMD_NO_OP\n", encoding="utf-8")
            command_log_path.write_text("", encoding="utf-8")
            send_log_path.write_text("", encoding="utf-8")
            traffic_pcap.write_bytes(b"pcap")
            event_log_path = host_event_log_path(runtime_root, "fprime_a")
            downlink_path = host_downlink_records_path(runtime_root, "fprime_a")
            event_log_path.parent.mkdir(parents=True, exist_ok=True)
            downlink_path.parent.mkdir(parents=True, exist_ok=True)
            event_log_path.write_text("", encoding="utf-8")
            downlink_path.write_text("", encoding="utf-8")

            shared = build_shared_fprime_artifact_layers(
                packets,
                transactions,
                feature_rows,
                manifest_path=manifest_path,
                per_run_reports=[
                    {
                        "run_id": 0,
                        "schedule_path": str(schedule_path.resolve()),
                        "run_log_path": str(run_log_path.resolve()),
                        "runtime_root": str(runtime_root.resolve()),
                        "command_log": str(command_log_path.resolve()),
                        "send_log": str(send_log_path.resolve()),
                        "traffic_pcap": str(traffic_pcap.resolve()),
                    }
                ],
                capture_backend="local_docker_bridge",
                capture_interface="br-test",
            )

        self.assertEqual(len(shared["raw_packets"]), len(packets))
        self.assertEqual(len(shared["raw_transactions"]), 1)
        self.assertEqual(len(shared["canonical_command_rows"]), 1)
        raw_transaction = shared["raw_transactions"][0]
        canonical_row = shared["canonical_command_rows"][0]
        self.assertEqual(raw_transaction["provenance"]["capture_interface"], "br-test")
        self.assertEqual(raw_transaction["sender"]["network_endpoint"]["ip"], "192.168.144.22")
        self.assertTrue(any(path.endswith("traffic.pcap") for path in raw_transaction["provenance"]["source_artifact_paths"]))
        self.assertEqual(canonical_row["recent_behavior"]["command_rate_1m"], feature_rows[0]["command_rate_1m"])
        self.assertEqual(canonical_row["command_semantics"]["canonical_command_family"], "read_only_inspection")
        self.assertIn("target_command_activity_ratio", canonical_row["normalized_state"])
        self.assertIn("peer_telemetry_staleness_ratio", canonical_row["normalized_state"])
        self.assertTrue(any(path.endswith("manifest.json") for path in canonical_row["audit_context"]["source_artifact_paths"]))


if __name__ == "__main__":
    unittest.main()
