from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.mavlink_real.packet_fidelity import build_raw_packet_record, build_raw_transaction_record
from tools.shared.canonical_records import (
    CANONICAL_COMMAND_ROW_SCHEMA_VERSION,
    CanonicalRecordValidationError,
    build_canonical_command_row,
    load_canonical_command_row_schema,
    validate_canonical_command_row,
    validate_canonical_command_rows,
)
from tools.shared.schema import (
    RAW_PACKET_SCHEMA_VERSION,
    RAW_TRANSACTION_SCHEMA_VERSION,
    RawArtifactValidationError,
    adapt_legacy_fprime_packet,
    adapt_legacy_fprime_transaction,
    load_raw_packet_schema,
    load_raw_transaction_schema,
    validate_raw_packet_records,
    validate_raw_transaction,
    validate_raw_transaction_records,
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


def recent_behavior_fixture() -> dict[str, object]:
    return {
        "command_rate_1m": 2.0,
        "error_rate_1m": 0.25,
        "repeat_command_count_10m": 1,
        "same_target_command_rate_1m": 1.0,
    }


def make_fprime_request_packet() -> dict[str, object]:
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
        "args": {"duration": 5, "arm": True, "mode": "SAFE"},
        "bytes_on_wire": 40,
        "observed_on_wire": 1,
        "ts_source": "pcap",
        "bytes_source": "pcap",
    }


def make_fprime_transaction() -> dict[str, object]:
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


def make_mavlink_request_packet() -> dict[str, object]:
    return {
        "ts_ms": 1000,
        "packet_kind": "request",
        "protocol_version": "2.0",
        "src": "ops_primary",
        "dst": "mavlink_vehicle",
        "src_ip": "192.168.164.12",
        "dst_ip": "192.168.164.2",
        "src_port": 50100,
        "dst_port": 5760,
        "target_service": "mavlink_vehicle",
        "target_stream_id": "mavlink_vehicle@tcp:mavlink_vehicle:5760",
        "target_stream_index": 0,
        "service": "command_long",
        "command": "REQUEST_AUTOPILOT_CAPABILITIES",
        "label": 0,
        "episode_id": 0,
        "episode_kind": "benign",
        "session_id": "ops_primary-0001",
        "txn_id": "txn-0001",
        "send_id": "send-0001",
        "attack_family": "none",
        "phase": "startup",
        "actor": "ops_primary",
        "actor_role": "ops_primary",
        "actor_trust": 0.98,
        "args": {"broadcast": False},
        "bytes_on_wire": 33,
        "observed_on_wire": 1,
        "ts_source": "pcap",
        "bytes_source": "pcap",
    }


def make_mavlink_final_packet() -> dict[str, object]:
    request = make_mavlink_request_packet()
    return {
        **request,
        "packet_kind": "final",
        "src": "mavlink_vehicle",
        "dst": "ops_primary",
        "src_ip": "192.168.164.2",
        "dst_ip": "192.168.164.12",
        "src_port": 5760,
        "dst_port": 50100,
        "ts_ms": 1016,
        "bytes_on_wire": 17,
        "gds_accept": 1,
        "sat_success": 1,
        "timeout": 0,
        "response_code": 0,
        "reason": "accepted",
        "response_direction_seen": 1,
        "final_observed_on_wire": 1,
        "txn_warning_events": 0,
        "txn_error_events": 0,
        "message_name": "COMMAND_ACK",
    }


def make_mavlink_transaction() -> dict[str, object]:
    return {
        "run_id": 0,
        "episode_id": 0,
        "label": 0,
        "label_name": "benign",
        "session_id": "ops_primary-0001",
        "txn_id": "txn-0001",
        "send_id": "send-0001",
        "target_stream_id": "mavlink_vehicle@tcp:mavlink_vehicle:5760",
        "target_stream_index": 0,
        "attack_family": "none",
        "phase": "startup",
        "actor": "ops_primary",
        "actor_role": "ops_primary",
        "actor_trust": 0.98,
        "command": "REQUEST_AUTOPILOT_CAPABILITIES",
        "service": "command_long",
        "args": {"broadcast": False},
        "source_service": "ops_primary",
        "target_service": "mavlink_vehicle",
        "source_ip": "192.168.164.12",
        "target_ip": "192.168.164.2",
        "request_ts_ms": 1000.0,
        "final_ts_ms": 1016.0,
        "latency_ms": 16.0,
        "req_bytes": 33.0,
        "resp_bytes": 17.0,
        "gds_accept": 1.0,
        "sat_success": 1.0,
        "timeout": 0.0,
        "response_code": 0.0,
        "reason": "accepted",
        "txn_warning_events": 0.0,
        "txn_error_events": 0.0,
        "response_direction_seen": 1.0,
        "final_observed_on_wire": 1.0,
        "response_name": "COMMAND_ACK",
        "native_state_snapshot": {
            "target_logical_id": "mavlink_vehicle",
            "peer_logical_id": None,
            "snapshot_observed_at_ms": 999.0,
            "target_fields": {
                "sys_load_fraction": 0.32,
                "battery_remaining_pct": 73.0,
            },
            "peer_fields": {},
        },
    }


class CanonicalSchemaContractTests(unittest.TestCase):
    def test_shared_schema_property_names_stay_protocol_neutral(self) -> None:
        for schema in (
            load_raw_packet_schema(),
            load_raw_transaction_schema(),
            load_canonical_command_row_schema(),
        ):
            names = {name.lower() for name in collect_property_names(schema)}
            self.assertFalse(any("fprime" in name for name in names))
            self.assertFalse(any("mavlink" in name for name in names))

    def test_shared_raw_validators_accept_fprime_and_mavlink_records(self) -> None:
        fprime_request_packet = make_fprime_request_packet()
        fprime_raw_packet = adapt_legacy_fprime_packet(
            fprime_request_packet,
            source_artifact_paths=["logs/fprime/send_log.jsonl"],
        )
        fprime_raw_transaction = adapt_legacy_fprime_transaction(
            make_fprime_transaction(),
            related_packets=[fprime_request_packet],
            source_artifact_paths=["logs/fprime/run.csv", "pcap/fprime_run.pcap"],
        )

        mavlink_request_packet = make_mavlink_request_packet()
        mavlink_final_packet = make_mavlink_final_packet()
        mavlink_raw_packet = build_raw_packet_record(
            mavlink_request_packet,
            source_artifact_paths=["logs/mavlink/send_log.jsonl", "pcap/mavlink_run.pcap"],
            capture_backend="pcap",
            capture_interface="tap0",
        )
        mavlink_raw_transaction = build_raw_transaction_record(
            make_mavlink_transaction(),
            related_packets=[mavlink_request_packet, mavlink_final_packet],
            source_artifact_paths=["logs/mavlink/run.csv", "pcap/mavlink_run.pcap"],
            capture_backend="pcap",
            capture_interface="tap0",
        )

        validated_packets = validate_raw_packet_records([fprime_raw_packet, mavlink_raw_packet])
        validated_transactions = validate_raw_transaction_records([fprime_raw_transaction, mavlink_raw_transaction])

        self.assertEqual([record["schema_version"] for record in validated_packets], [RAW_PACKET_SCHEMA_VERSION] * 2)
        self.assertEqual([record["schema_version"] for record in validated_transactions], [RAW_TRANSACTION_SCHEMA_VERSION] * 2)
        self.assertEqual(
            {record["protocol_family"] for record in validated_transactions},
            {"fprime", "mavlink"},
        )
        self.assertEqual(validated_transactions[1]["target"]["network_endpoint"]["ip"], "192.168.164.2")

    def test_shared_canonical_validator_accepts_rows_from_both_protocols(self) -> None:
        fprime_raw_transaction = adapt_legacy_fprime_transaction(
            make_fprime_transaction(),
            related_packets=[make_fprime_request_packet()],
            source_artifact_paths=["logs/fprime/run.csv", "pcap/fprime_run.pcap"],
        )
        mavlink_raw_transaction = build_raw_transaction_record(
            make_mavlink_transaction(),
            related_packets=[make_mavlink_request_packet(), make_mavlink_final_packet()],
            source_artifact_paths=["logs/mavlink/run.csv", "pcap/mavlink_run.pcap"],
            capture_backend="pcap",
            capture_interface="tap0",
        )

        rows = validate_canonical_command_rows(
            [
                build_canonical_command_row(
                    fprime_raw_transaction,
                    mission_context={"window_class": "startup_window"},
                    recent_behavior=recent_behavior_fixture(),
                ),
                build_canonical_command_row(
                    mavlink_raw_transaction,
                    mission_context={"window_class": "takeoff_window"},
                    recent_behavior=recent_behavior_fixture(),
                ),
            ]
        )

        self.assertEqual([row["schema_version"] for row in rows], [CANONICAL_COMMAND_ROW_SCHEMA_VERSION] * 2)
        self.assertEqual({row["protocol_family"] for row in rows}, {"fprime", "mavlink"})
        self.assertEqual(rows[0]["platform_family"], "spacecraft")
        self.assertEqual(rows[1]["platform_family"], "air_vehicle")
        self.assertEqual(rows[1]["command_semantics"]["canonical_command_family"], "read_only_inspection")
        self.assertEqual(rows[1]["normalized_state"]["target_power_pressure_ratio"], 0.27)

    def test_shared_validators_reject_missing_required_sections(self) -> None:
        raw_transaction = build_raw_transaction_record(
            make_mavlink_transaction(),
            related_packets=[make_mavlink_request_packet(), make_mavlink_final_packet()],
            source_artifact_paths=["logs/mavlink/run.csv"],
            capture_backend="pcap",
            capture_interface="tap0",
        )
        raw_transaction.pop("sender")
        with self.assertRaises(RawArtifactValidationError):
            validate_raw_transaction(raw_transaction)

        canonical_row = build_canonical_command_row(
            adapt_legacy_fprime_transaction(
                make_fprime_transaction(),
                related_packets=[make_fprime_request_packet()],
                source_artifact_paths=["logs/fprime/run.csv"],
            ),
            mission_context={"window_class": "startup_window"},
            recent_behavior=recent_behavior_fixture(),
        )
        canonical_row.pop("command_semantics")
        with self.assertRaises(CanonicalRecordValidationError):
            validate_canonical_command_row(canonical_row)


if __name__ == "__main__":
    unittest.main()
