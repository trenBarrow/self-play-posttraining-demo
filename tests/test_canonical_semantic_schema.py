from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.shared.canonical_records import (
    CANONICAL_COMMAND_ROW_SCHEMA_VERSION,
    CanonicalRecordValidationError,
    build_canonical_command_row,
    canonicalize_legacy_fprime_transaction,
    load_canonical_command_row_schema,
    validate_canonical_command_row,
)
from tools.shared.schema import (
    RAW_TRANSACTION_SCHEMA_VERSION,
    adapt_legacy_fprime_transaction,
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


class CanonicalSemanticSchemaTests(unittest.TestCase):
    def make_legacy_transaction(self) -> dict[str, object]:
        return {
            "run_id": 7,
            "episode_id": 3,
            "episode_label": 0,
            "episode_kind": "benign",
            "label": 0,
            "label_name": "benign",
            "session_id": "ops_b1-0007",
            "txn_id": "7-000001-ops_b1",
            "send_id": "send-0007",
            "target_stream_id": "fprime_a:50050",
            "target_stream_index": 0.0,
            "attack_family": "none",
            "phase": "startup",
            "actor": "ops_b1",
            "actor_role": "ops_primary",
            "actor_trust": 0.97,
            "command": "cmdDisp.CMD_NO_OP",
            "service": "cmdDisp",
            "args": {"duration": 5, "arm": True, "mode": "SAFE"},
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
            "target_cpu_00_pct": 5.0,
            "target_cpu_01_pct": 6.0,
            "peer_cpu_total_pct": 7.0,
            "peer_cpu_00_pct": 3.0,
            "peer_cpu_01_pct": 4.0,
            "target_blockdrv_cycles_1m": 16.0,
            "peer_blockdrv_cycles_1m": 8.0,
            "target_cmd_errors_1m": 1.0,
            "peer_cmd_errors_1m": 0.0,
            "target_cmds_dispatched_1m": 5.0,
            "peer_cmds_dispatched_1m": 2.0,
            "target_filemanager_errors_1m": 2.0,
            "target_filedownlink_warnings_1m": 1.0,
            "peer_filemanager_errors_1m": 0.0,
            "peer_filedownlink_warnings_1m": 0.0,
            "target_hibuffs_total": 2.0,
            "peer_hibuffs_total": 1.0,
            "target_rg1_max_time_ms": 8.0,
            "target_rg2_max_time_ms": 7.0,
            "peer_rg1_max_time_ms": 4.0,
            "peer_rg2_max_time_ms": 5.0,
            "target_telemetry_age_ms": 180.0,
            "peer_telemetry_age_ms": 220.0,
        }

    def make_related_packets(self) -> list[dict[str, object]]:
        request_packet = {
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
            "episode_id": 3,
            "episode_kind": "benign",
            "session_id": "ops_b1-0007",
            "txn_id": "7-000001-ops_b1",
            "send_id": "send-0007",
            "attack_family": "none",
            "phase": "startup",
            "actor": "ops_b1",
            "actor_role": "ops_primary",
            "actor_trust": 0.97,
            "args": {"duration": 5, "arm": True, "mode": "SAFE"},
            "bytes_on_wire": 40,
            "observed_on_wire": 1,
            "ts_source": "pcap",
            "bytes_source": "pcap",
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
        return [request_packet, final_packet]

    def make_mavlink_raw_transaction(self) -> dict[str, object]:
        return {
            "schema_version": RAW_TRANSACTION_SCHEMA_VERSION,
            "record_kind": "raw_transaction",
            "protocol_family": "mavlink",
            "protocol_version": "2.0",
            "platform_family": "multirotor",
            "sender": {
                "logical_id": "gcs_primary",
                "role": "ops_primary",
                "trust_score": 0.9,
                "network_endpoint": None,
            },
            "target": {
                "logical_id": "vehicle_1",
                "role": "airframe",
                "stream_id": "udp:14550",
                "stream_index": 0,
                "network_endpoint": None,
            },
            "command": {
                "raw_name": "MAV_CMD_NAV_TAKEOFF",
                "raw_identifier": {
                    "service_name": "command_long",
                    "native_service_id": 76,
                    "native_command_id": 22,
                },
                "raw_arguments": [10.0, 1.5, True, "AUTO"],
                "raw_argument_representation": [10.0, 1.5, True, "AUTO"],
            },
            "timing": {
                "submitted_at_ms": 1000.0,
                "request_forwarded_at_ms": 1001.0,
                "protocol_response_at_ms": None,
                "finalized_at_ms": None,
                "latency_ms": None,
            },
            "transport": {
                "transport_family": "udp",
                "request_bytes_on_wire": 33.0,
                "response_bytes_on_wire": None,
            },
            "provenance": {
                "observed_on_wire": True,
                "capture_backend": "pcap",
                "capture_interface": "tap0",
                "timestamp_source": "pcap",
                "bytes_source": "pcap",
                "source_artifact_paths": ["logs/mavlink/session_0001.jsonl"],
            },
            "outcome": None,
            "correlation": {
                "run_id": 2,
                "episode_id": 1,
                "session_id": "gcs_primary-0001",
                "transaction_id": "txn-0001",
                "send_id": "send-0001",
                "stream_id": "udp:14550",
                "stream_index": 0,
            },
            "evaluation_context": {
                "label": 0,
                "label_name": "benign",
                "attack_family": "none",
                "phase": "takeoff",
                "actor_id": "gcs_primary",
                "actor_role": "ops_primary",
                "actor_trust": 0.9,
            },
            "evidence": {
                "related_packet_count": 1,
                "observed_message_families": ["request"],
                "observed_message_stages": ["request"],
                "packet_timestamp_sources": ["pcap"],
                "packet_byte_sources": ["pcap"],
                "request_wire_observed": True,
                "response_wire_observed": False,
                "log_correlation_mode": "session_txn_id",
                "source_artifact_paths": ["logs/mavlink/session_0001.jsonl"],
            },
            "native_state_snapshot": None,
            "native_fields": {
                "raw_frame_type": "COMMAND_LONG"
            },
        }

    def test_canonical_schema_property_names_stay_protocol_neutral(self) -> None:
        names = {name.lower() for name in collect_property_names(load_canonical_command_row_schema())}
        self.assertFalse(any("fprime" in name for name in names))
        self.assertFalse(any("mavlink" in name for name in names))

    def test_build_canonical_command_row_from_fprime_raw_transaction_validates(self) -> None:
        raw_transaction = adapt_legacy_fprime_transaction(
            self.make_legacy_transaction(),
            related_packets=self.make_related_packets(),
            source_artifact_paths=["logs/run.csv", "pcap/run_0007.pcap"],
        )
        row = build_canonical_command_row(
            raw_transaction,
            mission_context={"window_class": "startup_window"},
            recent_behavior={"command_rate_1m": 2.0, "error_rate_1m": 0.25},
        )
        validate_canonical_command_row(row)
        self.assertEqual(row["schema_version"], CANONICAL_COMMAND_ROW_SCHEMA_VERSION)
        self.assertEqual(row["protocol_family"], "fprime")
        self.assertEqual(row["command_semantics"]["canonical_command_name"], "noop_ping")
        self.assertEqual(row["command_semantics"]["canonical_command_family"], "read_only_inspection")
        self.assertEqual(row["command_semantics"]["mutation_scope"], "observation_only")
        self.assertEqual(row["actor_context"]["trust_class"], "high")
        self.assertEqual(row["normalized_state"]["target_compute_load_ratio"], 0.11)
        self.assertEqual(row["normalized_state"]["peer_compute_load_ratio"], 0.07)
        self.assertTrue(row["normalized_state"]["target_state_present"])
        self.assertTrue(row["normalized_state"]["peer_state_present"])
        self.assertAlmostEqual(row["normalized_state"]["target_compute_peak_load_ratio"], 0.06)
        self.assertAlmostEqual(row["normalized_state"]["target_compute_imbalance_ratio"], 0.01)
        self.assertAlmostEqual(row["normalized_state"]["target_storage_io_pressure_ratio"], 0.5)
        self.assertAlmostEqual(row["normalized_state"]["target_command_activity_ratio"], 5.0 / 12.0)
        self.assertAlmostEqual(row["normalized_state"]["target_command_error_ratio"], 0.25)
        self.assertAlmostEqual(row["normalized_state"]["target_service_issue_ratio"], 0.75)
        self.assertAlmostEqual(row["normalized_state"]["target_queue_pressure_ratio"], 0.5)
        self.assertAlmostEqual(row["normalized_state"]["target_scheduler_pressure_ratio"], 0.8)
        self.assertIsNone(row["normalized_state"]["target_link_pressure_ratio"])
        self.assertIsNone(row["normalized_state"]["target_power_pressure_ratio"])
        self.assertIsNone(row["normalized_state"]["target_control_instability_ratio"])
        self.assertIsNone(row["normalized_state"]["target_navigation_uncertainty_ratio"])
        self.assertAlmostEqual(row["normalized_state"]["target_telemetry_staleness_ratio"], 0.036)
        self.assertEqual(row["argument_profile"]["argument_leaf_count"], 3)
        self.assertEqual(row["argument_profile"]["numeric_argument_count"], 1)
        self.assertEqual(row["argument_profile"]["boolean_argument_count"], 1)
        self.assertEqual(row["audit_context"]["raw_command_name"], "cmdDisp.CMD_NO_OP")
        self.assertEqual(row["mission_context"]["window_class"], "startup_window")
        self.assertEqual(row["recent_behavior"]["command_rate_1m"], 2.0)
        self.assertTrue(row["observability"]["correlated_response_observed"])

    def test_build_canonical_command_row_supports_synthetic_mavlink_raw_transaction(self) -> None:
        raw_transaction = self.make_mavlink_raw_transaction()
        raw_transaction["native_state_snapshot"] = {
            "target_logical_id": "vehicle_1",
            "peer_logical_id": None,
            "snapshot_observed_at_ms": 900.0,
            "target_fields": {
                "sys_load_fraction": 0.42,
                "drop_rate_comm_fraction": 0.12,
                "battery_remaining_pct": 40.0,
                "battery_status_remaining_pct": 35.0,
                "power_vcc_v": 4.6,
                "power_servo_v": 4.9,
                "heartbeat_system_status": 5,
                "heartbeat_base_mode": 128,
                "gps_fix_type": 2,
                "gps_satellites_visible": 5,
                "onboard_control_sensors_enabled": 35,
                "onboard_control_sensors_health": 3,
            },
            "peer_fields": {},
        }
        validate_raw_transaction(raw_transaction)
        row = build_canonical_command_row(raw_transaction)
        validate_canonical_command_row(row)
        self.assertEqual(row["protocol_family"], "mavlink")
        self.assertEqual(row["platform_family"], "multirotor")
        self.assertEqual(row["command_semantics"]["canonical_command_name"], "takeoff")
        self.assertEqual(row["command_semantics"]["canonical_command_family"], "mission_sequence_control")
        self.assertEqual(row["command_semantics"]["safety_criticality"], "critical")
        self.assertTrue(row["normalized_state"]["state_available"])
        self.assertTrue(row["normalized_state"]["target_state_present"])
        self.assertFalse(row["normalized_state"]["peer_state_present"])
        self.assertAlmostEqual(row["normalized_state"]["target_compute_load_ratio"], 0.42)
        self.assertAlmostEqual(row["normalized_state"]["target_compute_peak_load_ratio"], 0.42)
        self.assertAlmostEqual(row["normalized_state"]["target_link_pressure_ratio"], 0.12)
        self.assertAlmostEqual(row["normalized_state"]["target_power_pressure_ratio"], 0.65)
        self.assertAlmostEqual(row["normalized_state"]["target_control_instability_ratio"], 0.8)
        self.assertAlmostEqual(row["normalized_state"]["target_navigation_uncertainty_ratio"], 5.0 / 6.0)
        self.assertAlmostEqual(row["normalized_state"]["target_telemetry_staleness_ratio"], 0.02)
        self.assertIsNone(row["normalized_state"]["target_storage_io_pressure_ratio"])
        self.assertIsNone(row["normalized_state"]["target_scheduler_pressure_ratio"])
        self.assertEqual(row["argument_profile"]["argument_leaf_count"], 4)
        self.assertEqual(row["argument_profile"]["numeric_argument_count"], 2)
        self.assertEqual(row["argument_profile"]["boolean_argument_count"], 1)
        self.assertEqual(row["argument_profile"]["string_argument_count"], 1)
        self.assertIsNone(row["recent_behavior"]["command_rate_1m"])
        self.assertEqual(row["audit_context"]["raw_command_name"], "MAV_CMD_NAV_TAKEOFF")

    def test_canonicalize_legacy_fprime_transaction_adapts_without_guessing_window(self) -> None:
        row = canonicalize_legacy_fprime_transaction(
            self.make_legacy_transaction(),
            related_packets=self.make_related_packets(),
            source_artifact_paths=["logs/run.csv"],
        )
        validate_canonical_command_row(row)
        self.assertEqual(row["command_semantics"]["canonical_command_name"], "noop_ping")
        self.assertEqual(row["command_semantics"]["canonical_command_family"], "read_only_inspection")
        self.assertEqual(row["mission_context"]["window_class"], "unspecified")
        self.assertEqual(row["audit_context"]["target_id"], "fprime_a")

    def test_validate_canonical_command_row_rejects_missing_semantics_group(self) -> None:
        raw_transaction = adapt_legacy_fprime_transaction(self.make_legacy_transaction())
        row = build_canonical_command_row(raw_transaction)
        row.pop("command_semantics")
        with self.assertRaises(CanonicalRecordValidationError):
            validate_canonical_command_row(row)


if __name__ == "__main__":
    unittest.main()
