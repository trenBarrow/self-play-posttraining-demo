from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.fprime_real.generate_dataset import write_schema_report
from tools.shared.canonical_records import CANONICAL_COMMAND_ROW_SCHEMA_VERSION, build_canonical_command_row
from tools.shared.schema import (
    RAW_PACKET_SCHEMA_VERSION,
    RAW_TRANSACTION_SCHEMA_VERSION,
    RawArtifactValidationError,
    adapt_legacy_fprime_packet,
    adapt_legacy_fprime_transaction,
    load_raw_packet_schema,
    load_raw_transaction_schema,
    validate_legacy_fprime_raw_contract,
    validate_raw_packet,
    validate_raw_transaction,
)


def collect_property_names(node: object) -> set[str]:
    names: set[str] = set()
    if isinstance(node, dict):
        properties = node.get("properties")
        if isinstance(properties, dict):
            for name, child in properties.items():
                names.add(name)
                names.update(collect_property_names(child))
        defs = node.get("$defs")
        if isinstance(defs, dict):
            for child in defs.values():
                names.update(collect_property_names(child))
        items = node.get("items")
        if items is not None:
            names.update(collect_property_names(items))
        for list_key in ("anyOf", "oneOf", "allOf"):
            values = node.get(list_key)
            if isinstance(values, list):
                for child in values:
                    names.update(collect_property_names(child))
    elif isinstance(node, list):
        for child in node:
            names.update(collect_property_names(child))
    return names


class RawArtifactContractTests(unittest.TestCase):
    def make_request_packet(self) -> dict[str, object]:
        return {
            "ts_ms": 61000,
            "packet_kind": "request",
            "src": "ops_b1",
            "dst": "fprime_a",
            "src_ip": "192.168.144.22",
            "dst_ip": "192.168.144.2",
            "src_port": 50101,
            "dst_port": 50050,
            "target_service": "fprime_a",
            "target_stream_id": "fprime_a:50050",
            "target_stream_index": 0,
            "service": "cmdDisp",
            "command": "cmdDisp.CMD_NO_OP",
            "label": 0,
            "episode_id": 0,
            "episode_kind": "benign",
            "session_id": "ops_b1-0000",
            "txn_id": "0-000001-ops_b1",
            "send_id": "send-0001",
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
        }

    def make_telemetry_packet(self) -> dict[str, object]:
        return {
            "ts_ms": 1000,
            "packet_kind": "telemetry",
            "src": "fprime_a",
            "dst": "gds",
            "service": "systemResources",
            "payload": {"cpu_total_pct": 11.0},
            "node_service": "fprime_a",
            "label": 0,
            "attack_family": "none",
            "phase": "telemetry",
            "bytes_on_wire": 130,
            "observed_on_wire": 1,
            "ts_source": "gds_recv_bin",
            "bytes_source": "gds_recv_bin",
            "run_id": 0,
        }

    def make_transaction(self) -> dict[str, object]:
        return {
            "run_id": 0,
            "episode_id": 0,
            "episode_label": 0,
            "episode_kind": "benign",
            "label": 0,
            "label_name": "benign",
            "session_id": "ops_b1-0000",
            "txn_id": "0-000001-ops_b1",
            "send_id": "send-0001",
            "target_stream_id": "fprime_a:50050",
            "target_stream_index": 0.0,
            "attack_family": "none",
            "phase": "startup",
            "actor": "ops_b1",
            "actor_role": "ops_primary",
            "actor_trust": 0.97,
            "command": "cmdDisp.CMD_NO_OP",
            "service": "cmdDisp",
            "args": {},
            "target_service": "fprime_a",
            "target_node_id": 1.0,
            "request_ts_ms": 61000.0,
            "packet_gap_ms": 20.0,
            "req_bytes": 40.0,
            "resp_bytes": 12.0,
            "gds_accept": 1.0,
            "sat_success": 1.0,
            "timeout": 0.0,
            "response_code": 0.0,
            "reason": "completed",
            "uplink_ts_ms": 61020.0,
            "sat_response_ts_ms": 61045.0,
            "final_ts_ms": 61070.0,
            "request_to_uplink_ms": 20.0,
            "uplink_to_sat_response_ms": 25.0,
            "sat_response_to_final_ms": 25.0,
            "response_direction_seen": 1.0,
            "final_observed_on_wire": 1.0,
            "txn_warning_events": 0.0,
            "txn_error_events": 0.0,
            "target_cpu_total_pct": 11.0,
            "peer_cpu_total_pct": 7.0,
            "target_telemetry_age_ms": 180.0,
            "peer_telemetry_age_ms": 200.0,
            "latency_ms": 70.0,
        }

    def test_packet_schema_property_names_stay_protocol_neutral(self) -> None:
        names = {name.lower() for name in collect_property_names(load_raw_packet_schema())}
        self.assertFalse(any("fprime" in name for name in names))
        self.assertFalse(any("mavlink" in name for name in names))

    def test_transaction_schema_property_names_stay_protocol_neutral(self) -> None:
        names = {name.lower() for name in collect_property_names(load_raw_transaction_schema())}
        self.assertFalse(any("fprime" in name for name in names))
        self.assertFalse(any("mavlink" in name for name in names))

    def test_adapt_legacy_fprime_request_packet_validates(self) -> None:
        packet = self.make_request_packet()
        raw_packet = adapt_legacy_fprime_packet(packet, source_artifact_paths=["logs/send_log.jsonl"])
        validate_raw_packet(raw_packet)
        self.assertEqual(raw_packet["schema_version"], RAW_PACKET_SCHEMA_VERSION)
        self.assertEqual(raw_packet["protocol_family"], "fprime")
        self.assertEqual(raw_packet["message_family"], "request")
        self.assertEqual(raw_packet["command"]["raw_name"], "cmdDisp.CMD_NO_OP")
        self.assertEqual(raw_packet["native_fields"]["legacy_record"]["target_stream_id"], "fprime_a:50050")

    def test_adapt_legacy_fprime_telemetry_packet_validates_with_null_command(self) -> None:
        packet = self.make_telemetry_packet()
        raw_packet = adapt_legacy_fprime_packet(packet)
        validate_raw_packet(raw_packet)
        self.assertEqual(raw_packet["message_family"], "telemetry")
        self.assertIsNone(raw_packet["command"])
        self.assertEqual(raw_packet["native_payload"], {"cpu_total_pct": 11.0})

    def test_adapt_legacy_fprime_transaction_uses_related_packets(self) -> None:
        request_packet = self.make_request_packet()
        uplink_packet = {
            **request_packet,
            "packet_kind": "uplink",
            "src": "gds",
            "dst": "fprime_a",
            "src_ip": "192.168.144.100",
            "src_port": 50050,
            "dst_port": 50050,
            "ts_ms": 61020,
        }
        final_packet = {
            **request_packet,
            "packet_kind": "final",
            "src": "fprime_a",
            "dst": "ops_b1",
            "src_ip": "192.168.144.2",
            "dst_ip": "192.168.144.22",
            "src_port": 50050,
            "dst_port": 50101,
            "ts_ms": 61070,
            "gds_accept": 1,
            "sat_success": 1,
            "timeout": 0,
            "response_code": 0,
            "reason": "completed",
            "response_direction_seen": 1,
            "final_observed_on_wire": 1,
            "txn_warning_events": 0,
            "txn_error_events": 0,
            "event_name": "CMD_RESPONSE",
        }
        transaction = self.make_transaction()
        raw_transaction = adapt_legacy_fprime_transaction(
            transaction,
            related_packets=[request_packet, uplink_packet, final_packet],
            source_artifact_paths=["logs/run.csv", "pcap/run_0000.pcap"],
            capture_backend="local_docker_bridge",
            capture_interface="br-test",
        )
        validate_raw_transaction(raw_transaction)
        self.assertEqual(raw_transaction["schema_version"], RAW_TRANSACTION_SCHEMA_VERSION)
        self.assertEqual(raw_transaction["evidence"]["related_packet_count"], 3)
        self.assertIn("request", raw_transaction["evidence"]["observed_message_families"])
        self.assertIn("response", raw_transaction["evidence"]["observed_message_families"])
        self.assertTrue(raw_transaction["evidence"]["request_wire_observed"])
        self.assertTrue(raw_transaction["evidence"]["response_wire_observed"])
        self.assertEqual(raw_transaction["sender"]["network_endpoint"]["ip"], "192.168.144.22")
        self.assertEqual(raw_transaction["target"]["network_endpoint"]["ip"], "192.168.144.2")
        self.assertEqual(raw_transaction["provenance"]["capture_backend"], "local_docker_bridge")
        self.assertEqual(raw_transaction["provenance"]["capture_interface"], "br-test")
        self.assertEqual(raw_transaction["provenance"]["timestamp_source"], "pcap")
        self.assertEqual(raw_transaction["provenance"]["bytes_source"], "pcap")
        self.assertEqual(raw_transaction["native_state_snapshot"]["target_fields"]["cpu_total_pct"], 11.0)

    def test_validate_raw_packet_rejects_missing_required_shared_field(self) -> None:
        raw_packet = adapt_legacy_fprime_packet(self.make_request_packet())
        raw_packet.pop("protocol_family")
        with self.assertRaises(RawArtifactValidationError):
            validate_raw_packet(raw_packet)

    def test_validate_legacy_fprime_raw_contract_accepts_current_packet_and_transaction_shapes(self) -> None:
        packets = [self.make_request_packet(), self.make_telemetry_packet()]
        transactions = [self.make_transaction()]
        raw_packets, raw_transactions = validate_legacy_fprime_raw_contract(packets, transactions)
        self.assertEqual(len(raw_packets), 2)
        self.assertEqual(len(raw_transactions), 1)

    def test_write_schema_report_includes_shared_raw_schema_versions(self) -> None:
        with TemporaryDirectory() as tmpdir:
            report_dir = Path(tmpdir)
            raw_packet = adapt_legacy_fprime_packet(self.make_request_packet())
            raw_transaction = adapt_legacy_fprime_transaction(
                self.make_transaction(),
                related_packets=[self.make_request_packet()],
            )
            canonical_row = build_canonical_command_row(
                raw_transaction,
                recent_behavior={"command_rate_1m": 1.0, "error_rate_1m": 0.0},
            )
            path = write_schema_report(
                report_dir,
                packets=[self.make_request_packet()],
                transactions=[self.make_transaction()],
                history_featurization={"source": "transactions", "group_key": "run_id", "state_reset": "per_run"},
                raw_packets=[raw_packet],
                raw_transactions=[raw_transaction],
                canonical_rows=[canonical_row],
                artifact_paths={"canonical_command_rows": str((report_dir / "canonical_command_rows.jsonl").resolve())},
            )
            payload = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(payload["raw_packet_schema_version"], RAW_PACKET_SCHEMA_VERSION)
        self.assertEqual(payload["raw_transaction_schema_version"], RAW_TRANSACTION_SCHEMA_VERSION)
        self.assertEqual(payload["canonical_command_row_schema_version"], CANONICAL_COMMAND_ROW_SCHEMA_VERSION)
        self.assertIn("command_semantics.canonical_command_family", payload["canonical_command_row_fields"])
        self.assertIn("canonical_command_rows", payload["artifact_paths"])


if __name__ == "__main__":
    unittest.main()
