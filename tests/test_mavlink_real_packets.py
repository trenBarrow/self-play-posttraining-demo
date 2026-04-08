from __future__ import annotations

import json
import socket
import struct
import sys
import tempfile
import unittest
from pathlib import Path

import dpkt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.mavlink_real.packet_fidelity import build_packets_from_real_artifacts, write_artifact_bundle
from tools.mavlink_real.support_probe import collect_capture_drain_failures
from tools.shared.schema import RAW_TRANSACTION_SCHEMA_VERSION, validate_raw_packet_records, validate_raw_transaction_records


def ip_bytes(value: str) -> bytes:
    return socket.inet_pton(socket.AF_INET, value)


def ethernet_tcp_packet(src_ip: str, dst_ip: str, src_port: int, dst_port: int, payload: bytes, *, flags: int = dpkt.tcp.TH_ACK) -> bytes:
    tcp = dpkt.tcp.TCP(sport=src_port, dport=dst_port, flags=flags, seq=1, ack=1, win=8192, data=payload)
    tcp.off = 5
    ip = dpkt.ip.IP(src=ip_bytes(src_ip), dst=ip_bytes(dst_ip), p=dpkt.ip.IP_PROTO_TCP, ttl=64)
    ip.data = tcp
    ip.len = 20 + len(tcp)
    eth = dpkt.ethernet.Ethernet(
        src=b"\xaa\xaa\xaa\xaa\xaa\xaa",
        dst=b"\xbb\xbb\xbb\xbb\xbb\xbb",
        type=dpkt.ethernet.ETH_TYPE_IP,
        data=ip,
    )
    return bytes(eth)


def ethernet_udp_packet(src_ip: str, dst_ip: str, src_port: int, dst_port: int, payload: bytes) -> bytes:
    udp = dpkt.udp.UDP(sport=src_port, dport=dst_port, data=payload)
    udp.ulen = 8 + len(payload)
    ip = dpkt.ip.IP(src=ip_bytes(src_ip), dst=ip_bytes(dst_ip), p=dpkt.ip.IP_PROTO_UDP, ttl=64)
    ip.data = udp
    ip.len = 20 + len(udp)
    eth = dpkt.ethernet.Ethernet(
        src=b"\xcc\xcc\xcc\xcc\xcc\xcc",
        dst=b"\xdd\xdd\xdd\xdd\xdd\xdd",
        type=dpkt.ethernet.ETH_TYPE_IP,
        data=ip,
    )
    return bytes(eth)


def write_pcap(path: Path, frames: list[tuple[float, bytes]]) -> None:
    with path.open("wb") as handle:
        writer = dpkt.pcap.Writer(handle, linktype=dpkt.pcap.DLT_EN10MB)
        for ts, frame in frames:
            writer.writepkt(frame, ts=ts)


def mavlink_v2_frame(message_id: int, payload: bytes, *, seq: int, sysid: int, compid: int) -> bytes:
    header = bytes(
        [
            0xFD,
            len(payload),
            0,
            0,
            seq & 0xFF,
            sysid & 0xFF,
            compid & 0xFF,
            message_id & 0xFF,
            (message_id >> 8) & 0xFF,
            (message_id >> 16) & 0xFF,
        ]
    )
    return header + payload + b"\x00\x00"


def cstring16(value: str) -> bytes:
    raw = value.encode("ascii", "ignore")[:16]
    return raw + (b"\x00" * (16 - len(raw)))


def heartbeat_payload(*, custom_mode: int = 4, vehicle_type: int = 2, autopilot: int = 3, base_mode: int = 81, system_status: int = 4) -> bytes:
    return struct.pack("<IBBBBB", custom_mode, vehicle_type, autopilot, base_mode, system_status, 3)


def sys_status_payload(
    *,
    onboard_control_sensors_present: int = 0,
    onboard_control_sensors_enabled: int = 0,
    onboard_control_sensors_health: int = 0,
    load: int = 320,
    voltage_battery: int = 11800,
    current_battery: int = 1250,
    drop_rate_comm: int = 0,
    errors_comm: int = 0,
    battery_remaining: int = 73,
) -> bytes:
    return struct.pack(
        "<IIIHHhHHHHHHb",
        onboard_control_sensors_present,
        onboard_control_sensors_enabled,
        onboard_control_sensors_health,
        load,
        voltage_battery,
        current_battery,
        drop_rate_comm,
        errors_comm,
        0,
        0,
        0,
        0,
        battery_remaining,
    )


def param_request_read_payload(param_id: str) -> bytes:
    return struct.pack("<hBB16s", -1, 1, 1, cstring16(param_id))


def param_value_payload(param_id: str, param_value: float, *, param_count: int = 20, param_index: int = 3, param_type: int = 9) -> bytes:
    return struct.pack("<fHH16sB", param_value, param_count, param_index, cstring16(param_id), param_type)


def meta_json(*, class_label: int = 0, class_name: str = "benign", phase: str = "startup") -> str:
    return json.dumps(
        {
            "class_label": class_label,
            "class_name": class_name,
            "attack_family": "none" if class_label == 0 else "intrusion",
            "actor_role": "ops_primary",
            "actor_trust": 0.98,
            "episode_id": 0,
            "phase": phase,
            "run_id": 0,
        },
        separators=(",", ":"),
        sort_keys=True,
    )


def benign_run_row() -> dict[str, str]:
    return {
        "real_iso": "2026-03-29T20:00:00Z",
        "real_ms": "1000010",
        "send_start_ms": "1000010",
        "send_end_ms": "1000020",
        "virtual_day": "0",
        "virtual_time": "00:00:01",
        "virtual_seconds": "1",
        "source_service": "ops_primary",
        "source_ip": "192.168.164.12",
        "source_system_id": "21",
        "source_component_id": "190",
        "target_service": "mavlink_vehicle",
        "target_ip": "192.168.164.2",
        "target_endpoint": "tcp:mavlink_vehicle:5760",
        "target_system_id": "1",
        "target_component_id": "1",
        "target_stream_id": "mavlink_vehicle@tcp:mavlink_vehicle:5760",
        "target_stream_index": "0",
        "send_id": "send-001",
        "args_sha256": "hash-001",
        "command": "PARAM_REQUEST_READ",
        "command_family": "read_only_inspection",
        "arguments_json": json.dumps({"param_id": "FRAME_CLASS", "param_index": -1}, separators=(",", ":"), sort_keys=True),
        "meta_json": meta_json(),
        "run_id": "0",
        "heartbeat_type": "2",
        "heartbeat_autopilot": "3",
        "heartbeat_base_mode": "81",
        "heartbeat_custom_mode": "4",
        "response_name": "PARAM_VALUE",
        "response_code": "9",
        "response_text": "FRAME_CLASS",
        "response_json": json.dumps({"param_id": "FRAME_CLASS", "param_type": 9, "param_value": 1.0}, separators=(",", ":"), sort_keys=True),
        "observed_messages_json": "[]",
        "timeout": "0",
        "latency_ms": "10",
        "send_exception": "",
    }


def timeout_run_row() -> dict[str, str]:
    row = benign_run_row()
    row.update(
        {
            "real_ms": "1001010",
            "send_start_ms": "1001010",
            "send_end_ms": "1009010",
            "virtual_time": "00:00:11",
            "virtual_seconds": "11",
            "send_id": "send-002",
            "args_sha256": "hash-002",
            "arguments_json": json.dumps({"param_id": "SYSID_THISMAV", "param_index": -1}, separators=(",", ":"), sort_keys=True),
            "response_name": "",
            "response_code": "",
            "response_text": "",
            "response_json": "{}",
            "timeout": "1",
            "latency_ms": "8000",
            "send_exception": "timeout",
        }
    )
    return row


class MavlinkRealPacketTests(unittest.TestCase):
    def test_write_artifact_bundle_persists_shared_mavlink_packet_and_provenance_artifacts(self) -> None:
        run_rows = [benign_run_row()]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            pcap_path = tmp_path / "mavlink.pcap"
            heartbeat = mavlink_v2_frame(0, heartbeat_payload(), seq=1, sysid=1, compid=1)
            sys_status = mavlink_v2_frame(1, sys_status_payload(), seq=2, sysid=1, compid=1)
            request = mavlink_v2_frame(20, param_request_read_payload("FRAME_CLASS"), seq=10, sysid=21, compid=190)
            response = mavlink_v2_frame(22, param_value_payload("FRAME_CLASS", 1.0), seq=11, sysid=1, compid=1)
            write_pcap(
                pcap_path,
                [
                    (999.900, ethernet_udp_packet("192.168.164.2", "192.168.164.3", 14550, 14550, heartbeat + sys_status)),
                    (1000.010, ethernet_tcp_packet("192.168.164.12", "192.168.164.2", 50100, 5760, request)),
                    (1000.020, ethernet_tcp_packet("192.168.164.2", "192.168.164.12", 5760, 50100, response)),
                ],
            )

            packet_result = build_packets_from_real_artifacts(
                run_rows,
                pcap_path=pcap_path,
                capture_interface="tap0",
                capture_backend="pcap",
                source_artifact_paths=[str(pcap_path.resolve())],
                strict=True,
            )
            validate_raw_packet_records(packet_result.raw_packets)
            validate_raw_transaction_records(packet_result.raw_transactions)

            self.assertEqual(len(packet_result.transactions), 1)
            self.assertTrue(packet_result.observations[0]["request_wire_seen"])
            self.assertTrue(packet_result.observations[0]["response_direction_seen"])
            self.assertTrue(packet_result.observations[0]["telemetry_recent"])
            self.assertTrue(packet_result.observations[0]["state_snapshot_seen"])

            output_dir = tmp_path / "out"
            write_artifact_bundle(
                output_dir=output_dir,
                packet_result=packet_result,
                run_rows=run_rows,
            )

            actual_run_report = json.loads((output_dir / "reports" / "actual_run_observability.json").read_text(encoding="utf-8"))
            provenance_summary = json.loads((output_dir / "reports" / "provenance_summary.json").read_text(encoding="utf-8"))
            with (output_dir / "data" / "raw_transactions.jsonl").open(encoding="utf-8") as handle:
                raw_transactions = [json.loads(line) for line in handle if line.strip()]

            self.assertEqual(actual_run_report["summary"]["benign_rows"], 1)
            self.assertEqual(actual_run_report["summary"]["observability_failed_rows"], 0)
            self.assertEqual(provenance_summary["protocol_family"], "mavlink")
            self.assertEqual(provenance_summary["request_row_count"], 1)
            self.assertEqual(provenance_summary["state_snapshot_seen_count"], 1)
            self.assertGreaterEqual(provenance_summary["telemetry_message_count"], 2)
            self.assertEqual(raw_transactions[0]["schema_version"], RAW_TRANSACTION_SCHEMA_VERSION)
            self.assertEqual(raw_transactions[0]["native_state_snapshot"]["target_fields"]["sys_load_fraction"], 0.32)

    def test_timeout_rows_keep_capture_drain_green_when_request_and_state_are_observed(self) -> None:
        run_rows = [timeout_run_row()]
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            pcap_path = tmp_path / "mavlink_timeout.pcap"
            heartbeat = mavlink_v2_frame(0, heartbeat_payload(), seq=1, sysid=1, compid=1)
            sys_status = mavlink_v2_frame(1, sys_status_payload(load=250, battery_remaining=68), seq=2, sysid=1, compid=1)
            request = mavlink_v2_frame(20, param_request_read_payload("SYSID_THISMAV"), seq=15, sysid=21, compid=190)
            write_pcap(
                pcap_path,
                [
                    (1008.900, ethernet_udp_packet("192.168.164.2", "192.168.164.3", 14550, 14550, heartbeat + sys_status)),
                    (1001.010, ethernet_tcp_packet("192.168.164.12", "192.168.164.2", 50101, 5760, request)),
                ],
            )

            packet_result = build_packets_from_real_artifacts(
                run_rows,
                pcap_path=pcap_path,
                capture_interface="tap0",
                capture_backend="pcap",
                source_artifact_paths=[str(pcap_path.resolve())],
                strict=False,
            )

            validate_raw_packet_records(packet_result.raw_packets)
            validate_raw_transaction_records(packet_result.raw_transactions)
            failures = collect_capture_drain_failures(run_rows, packet_result.observations)

            self.assertEqual(failures, [])
            self.assertTrue(packet_result.observations[0]["request_wire_seen"])
            self.assertFalse(packet_result.observations[0]["response_direction_seen"])
            self.assertTrue(packet_result.observations[0]["telemetry_recent"])
            self.assertTrue(packet_result.observations[0]["state_snapshot_seen"])
            self.assertEqual(packet_result.provenance_summary["missing_response_rows"], [])


if __name__ == "__main__":
    unittest.main()
