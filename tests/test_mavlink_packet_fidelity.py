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

from tools.mavlink_real.packet_fidelity import build_packets_from_real_artifacts
from tools.shared.schema import validate_raw_packet_records, validate_raw_transaction_records


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


def autopilot_version_payload(
    *,
    capabilities: int = 64495,
    flight_sw_version: int = 67503103,
    middleware_sw_version: int = 0,
    os_sw_version: int = 0,
    board_version: int = 0,
    vendor_id: int = 0,
    product_id: int = 0,
    uid: int = 0,
) -> bytes:
    return struct.pack(
        "<QIIIIHHQ",
        capabilities,
        flight_sw_version,
        middleware_sw_version,
        os_sw_version,
        board_version,
        vendor_id,
        product_id,
        uid,
    )


def param_request_read_payload(param_id: str) -> bytes:
    return struct.pack("<hBB16s", -1, 1, 1, cstring16(param_id))


def param_request_list_payload(*, target_system: int = 1, target_component: int = 1) -> bytes:
    return struct.pack("<BB", target_system, target_component)


def param_value_payload(param_id: str, param_value: float, *, param_count: int = 20, param_index: int = 3, param_type: int = 9) -> bytes:
    return struct.pack("<fHH16sB", param_value, param_count, param_index, cstring16(param_id), param_type)


def benign_meta_json() -> str:
    return json.dumps(
        {
            "class_label": 0,
            "class_name": "benign",
            "attack_family": "none",
            "actor_role": "ops_primary",
            "actor_trust": 0.98,
            "episode_id": 0,
            "phase": "startup",
            "run_id": 0,
        },
        separators=(",", ":"),
        sort_keys=True,
    )


class MavlinkPacketFidelityTests(unittest.TestCase):
    def test_build_packets_accepts_truncated_command_long_request(self) -> None:
        run_rows = [
            {
                "real_iso": "2026-03-28T20:00:05Z",
                "real_ms": "1500010",
                "send_start_ms": "1500010",
                "send_end_ms": "1500020",
                "virtual_day": "0",
                "virtual_time": "00:00:05",
                "virtual_seconds": "5",
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
                "send_id": "send-truncated-command-long",
                "args_sha256": "hash-truncated-command-long",
                "command": "REQUEST_AUTOPILOT_CAPABILITIES",
                "command_family": "read_only_inspection",
                "arguments_json": json.dumps({"param1": 1.0}, separators=(",", ":"), sort_keys=True),
                "meta_json": benign_meta_json(),
                "run_id": "0",
                "heartbeat_type": "2",
                "heartbeat_autopilot": "3",
                "heartbeat_base_mode": "81",
                "heartbeat_custom_mode": "4",
                "response_name": "AUTOPILOT_VERSION",
                "response_code": "",
                "response_text": "",
                "response_json": json.dumps(
                    {"capabilities": 64495, "flight_sw_version": 67503103},
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                "observed_messages_json": "[]",
                "timeout": "0",
                "latency_ms": "10",
                "send_exception": "",
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            pcap_path = tmp_path / "mavlink_truncated_command_long.pcap"
            heartbeat = mavlink_v2_frame(0, heartbeat_payload(), seq=1, sysid=1, compid=1)
            sys_status = mavlink_v2_frame(1, sys_status_payload(), seq=2, sysid=1, compid=1)
            full_request_payload = struct.pack("<7fHBBB", 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 520, 1, 1, 0)
            truncated_request = mavlink_v2_frame(76, full_request_payload[:-1], seq=10, sysid=21, compid=190)
            response = mavlink_v2_frame(148, autopilot_version_payload(), seq=11, sysid=1, compid=1)
            write_pcap(
                pcap_path,
                [
                    (1499.900, ethernet_udp_packet("192.168.164.2", "192.168.164.3", 14550, 14550, heartbeat + sys_status)),
                    (1500.005, ethernet_tcp_packet("192.168.164.2", "192.168.164.12", 5760, 50300, heartbeat)),
                    (1500.010, ethernet_tcp_packet("192.168.164.12", "192.168.164.2", 50300, 5760, truncated_request)),
                    (1500.020, ethernet_tcp_packet("192.168.164.2", "192.168.164.12", 5760, 50300, response)),
                ],
            )
            result = build_packets_from_real_artifacts(
                run_rows,
                pcap_path=pcap_path,
                capture_interface="bridge100",
                capture_backend="rdctl_vm_bridge",
                source_artifact_paths=[str(pcap_path.resolve())],
                strict=True,
            )

        self.assertEqual(len(result.transactions), 1)
        self.assertTrue(result.observations[0]["request_wire_seen"])
        self.assertTrue(result.observations[0]["response_direction_seen"])
        self.assertTrue(result.observations[0]["telemetry_recent"])
        self.assertTrue(result.observations[0]["state_snapshot_seen"])
        self.assertEqual(result.observations[0]["request_message_name"], "COMMAND_LONG")
        self.assertEqual(result.transactions[0]["request_payload"]["command"], 520)
        self.assertAlmostEqual(result.transactions[0]["request_payload"]["param1"], 1.0)

    def test_build_packets_accepts_mission_request_list_without_explicit_mission_type_field(self) -> None:
        run_rows = [
            {
                "real_iso": "2026-03-29T13:56:56Z",
                "real_ms": "1774792616769",
                "send_start_ms": "1774792616769",
                "send_end_ms": "1774792616830",
                "virtual_day": "0",
                "virtual_time": "00:13:14",
                "virtual_seconds": "794",
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
                "target_stream_index": "10",
                "send_id": "send-mission-list",
                "args_sha256": "hash-mission-list",
                "command": "MISSION_REQUEST_LIST",
                "command_family": "read_only_inspection",
                "arguments_json": json.dumps({"mission_type": 0}, separators=(",", ":"), sort_keys=True),
                "meta_json": benign_meta_json(),
                "run_id": "0",
                "heartbeat_type": "2",
                "heartbeat_autopilot": "3",
                "heartbeat_base_mode": "81",
                "heartbeat_custom_mode": "0",
                "response_name": "MISSION_COUNT",
                "response_code": "",
                "response_text": "",
                "response_json": json.dumps(
                    {"count": 0, "mission_type": 0, "target_component": 190, "target_system": 21},
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                "observed_messages_json": json.dumps([{"count": 0, "mission_type": 0, "type": "MISSION_COUNT"}]),
                "timeout": "0",
                "latency_ms": "61",
                "send_exception": "",
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            pcap_path = tmp_path / "mavlink_mission_request_list_without_type.pcap"
            request_payload = struct.pack("<BB", 1, 1)
            response_payload = struct.pack("<HHBB", 0, 21, 190, 0)
            request = mavlink_v2_frame(43, request_payload, seq=10, sysid=21, compid=190)
            response = mavlink_v2_frame(44, response_payload, seq=11, sysid=1, compid=1)
            heartbeat = mavlink_v2_frame(0, heartbeat_payload(base_mode=81), seq=1, sysid=1, compid=1)
            write_pcap(
                pcap_path,
                [
                    (1774792616.770, ethernet_tcp_packet("192.168.164.2", "192.168.164.12", 5760, 50300, heartbeat)),
                    (1774792616.826, ethernet_tcp_packet("192.168.164.12", "192.168.164.2", 50300, 5760, request)),
                    (1774792616.829, ethernet_tcp_packet("192.168.164.2", "192.168.164.12", 5760, 50300, response)),
                ],
            )
            result = build_packets_from_real_artifacts(
                run_rows,
                pcap_path=pcap_path,
                capture_interface="bridge100",
                capture_backend="rdctl_vm_bridge",
                source_artifact_paths=[str(pcap_path.resolve())],
                strict=True,
            )

        self.assertEqual(len(result.transactions), 1)
        self.assertTrue(result.observations[0]["request_wire_seen"])
        self.assertTrue(result.observations[0]["response_direction_seen"])
        self.assertEqual(result.observations[0]["request_message_name"], "MISSION_REQUEST_LIST")
        self.assertEqual(result.observations[0]["response_message_name"], "MISSION_COUNT")

    def test_build_packets_accepts_mission_ack_without_explicit_mission_type_field(self) -> None:
        run_rows = [
            {
                "real_iso": "2026-03-29T14:04:55Z",
                "real_ms": "1774793095205",
                "send_start_ms": "1774793095205",
                "send_end_ms": "1774793095260",
                "virtual_day": "0",
                "virtual_time": "00:04:59",
                "virtual_seconds": "299",
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
                "target_stream_index": "3",
                "send_id": "send-mission-clear",
                "args_sha256": "hash-mission-clear",
                "command": "MISSION_CLEAR_ALL",
                "command_family": "mission_sequence_control",
                "arguments_json": json.dumps({"mission_type": 0}, separators=(",", ":"), sort_keys=True),
                "meta_json": benign_meta_json(),
                "run_id": "0",
                "heartbeat_type": "2",
                "heartbeat_autopilot": "3",
                "heartbeat_base_mode": "81",
                "heartbeat_custom_mode": "0",
                "response_name": "MISSION_ACK",
                "response_code": "0",
                "response_text": "MAV_MISSION_ACCEPTED",
                "response_json": json.dumps(
                    {"mission_type": 0, "target_component": 190, "target_system": 21, "type": 0},
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                "observed_messages_json": json.dumps(
                    [{"mission_type": 0, "type": "MISSION_ACK", "type_code": 0, "type_name": "MAV_MISSION_ACCEPTED"}]
                ),
                "timeout": "0",
                "latency_ms": "55",
                "send_exception": "",
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            pcap_path = tmp_path / "mavlink_mission_ack_without_type.pcap"
            request_payload = struct.pack("<BB", 1, 1)
            response_payload = b""
            request = mavlink_v2_frame(45, request_payload, seq=10, sysid=21, compid=190)
            response = mavlink_v2_frame(47, response_payload, seq=11, sysid=1, compid=1)
            heartbeat = mavlink_v2_frame(0, heartbeat_payload(base_mode=81), seq=1, sysid=1, compid=1)
            write_pcap(
                pcap_path,
                [
                    (1774793095.210, ethernet_tcp_packet("192.168.164.2", "192.168.164.12", 5760, 50300, heartbeat)),
                    (1774793095.258, ethernet_tcp_packet("192.168.164.12", "192.168.164.2", 50300, 5760, request)),
                    (1774793095.260, ethernet_tcp_packet("192.168.164.2", "192.168.164.12", 5760, 50300, response)),
                ],
            )
            result = build_packets_from_real_artifacts(
                run_rows,
                pcap_path=pcap_path,
                capture_interface="bridge100",
                capture_backend="rdctl_vm_bridge",
                source_artifact_paths=[str(pcap_path.resolve())],
                strict=True,
            )

        self.assertEqual(len(result.transactions), 1)
        self.assertTrue(result.observations[0]["request_wire_seen"])
        self.assertTrue(result.observations[0]["response_direction_seen"])
        self.assertEqual(result.observations[0]["request_message_name"], "MISSION_CLEAR_ALL")
        self.assertEqual(result.observations[0]["response_message_name"], "MISSION_ACK")

    def test_build_packets_separates_adjacent_param_reads_from_same_identity(self) -> None:
        run_rows = [
            {
                "real_iso": "2026-03-28T20:00:00Z",
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
                "meta_json": benign_meta_json(),
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
            },
            {
                "real_iso": "2026-03-28T20:00:00Z",
                "real_ms": "1000011",
                "send_start_ms": "1000011",
                "send_end_ms": "1000021",
                "virtual_day": "0",
                "virtual_time": "00:00:02",
                "virtual_seconds": "2",
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
                "target_stream_index": "1",
                "send_id": "send-002",
                "args_sha256": "hash-002",
                "command": "PARAM_REQUEST_READ",
                "command_family": "read_only_inspection",
                "arguments_json": json.dumps({"param_id": "FENCE_RADIUS", "param_index": -1}, separators=(",", ":"), sort_keys=True),
                "meta_json": benign_meta_json(),
                "run_id": "0",
                "heartbeat_type": "2",
                "heartbeat_autopilot": "3",
                "heartbeat_base_mode": "81",
                "heartbeat_custom_mode": "4",
                "response_name": "PARAM_VALUE",
                "response_code": "9",
                "response_text": "FENCE_RADIUS",
                "response_json": json.dumps({"param_id": "FENCE_RADIUS", "param_type": 9, "param_value": 120.0}, separators=(",", ":"), sort_keys=True),
                "observed_messages_json": "[]",
                "timeout": "0",
                "latency_ms": "10",
                "send_exception": "",
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            pcap_path = tmp_path / "mavlink.pcap"
            heartbeat = mavlink_v2_frame(0, heartbeat_payload(), seq=1, sysid=1, compid=1)
            sys_status = mavlink_v2_frame(
                1,
                sys_status_payload(
                    onboard_control_sensors_present=35,
                    onboard_control_sensors_enabled=35,
                    onboard_control_sensors_health=3,
                    drop_rate_comm=1200,
                ),
                seq=2,
                sysid=1,
                compid=1,
            )
            request_1 = mavlink_v2_frame(20, param_request_read_payload("FRAME_CLASS"), seq=10, sysid=21, compid=190)
            request_2 = mavlink_v2_frame(20, param_request_read_payload("FENCE_RADIUS"), seq=11, sysid=21, compid=190)
            response_1 = mavlink_v2_frame(22, param_value_payload("FRAME_CLASS", 1.0), seq=12, sysid=1, compid=1)
            response_2 = mavlink_v2_frame(22, param_value_payload("FENCE_RADIUS", 120.0), seq=13, sysid=1, compid=1)
            write_pcap(
                pcap_path,
                [
                    (999.900, ethernet_udp_packet("192.168.164.2", "192.168.164.3", 14550, 14550, heartbeat + sys_status)),
                    (1000.005, ethernet_tcp_packet("192.168.164.2", "192.168.164.12", 5760, 50100, heartbeat)),
                    (1000.010, ethernet_tcp_packet("192.168.164.12", "192.168.164.2", 50100, 5760, request_1)),
                    (1000.011, ethernet_tcp_packet("192.168.164.2", "192.168.164.12", 5760, 50101, heartbeat)),
                    (1000.0115, ethernet_tcp_packet("192.168.164.12", "192.168.164.2", 50101, 5760, request_2)),
                    (1000.020, ethernet_tcp_packet("192.168.164.2", "192.168.164.12", 5760, 50100, response_1)),
                    (1000.021, ethernet_tcp_packet("192.168.164.2", "192.168.164.12", 5760, 50101, response_2)),
                ],
            )

            result = build_packets_from_real_artifacts(
                run_rows,
                pcap_path=pcap_path,
                capture_interface="bridge100",
                capture_backend="rdctl_vm_bridge",
                source_artifact_paths=[str(pcap_path.resolve())],
                strict=True,
            )

        self.assertEqual(len(result.transactions), 2)
        telemetry_packets = [packet for packet in result.packets if packet.get("packet_kind") == "telemetry"]
        self.assertGreaterEqual(len(telemetry_packets), 2)
        self.assertIn("SYS_STATUS", {str(packet.get("service")) for packet in telemetry_packets})
        tx_by_send_id = {str(item["send_id"]): item for item in result.transactions}
        raw_tx_by_send_id = {
            str(item["correlation"]["send_id"]): item for item in result.raw_transactions
        }
        self.assertEqual(tx_by_send_id["send-001"]["request_payload"]["param_id"], "FRAME_CLASS")
        self.assertEqual(tx_by_send_id["send-002"]["request_payload"]["param_id"], "FENCE_RADIUS")
        self.assertEqual(tx_by_send_id["send-001"]["response_payload"]["param_id"], "FRAME_CLASS")
        self.assertEqual(tx_by_send_id["send-002"]["response_payload"]["param_id"], "FENCE_RADIUS")
        self.assertEqual(tx_by_send_id["send-001"]["req_bytes"], len(request_1))
        self.assertEqual(tx_by_send_id["send-002"]["resp_bytes"], len(response_2))
        self.assertTrue(all(item["telemetry_recent"] == 1 for item in result.transactions))
        self.assertTrue(all(observation["state_snapshot_seen"] for observation in result.observations))
        self.assertEqual(
            raw_tx_by_send_id["send-001"]["native_state_snapshot"]["target_fields"]["onboard_control_sensors_enabled"],
            35,
        )
        self.assertEqual(
            raw_tx_by_send_id["send-001"]["native_state_snapshot"]["target_fields"]["onboard_control_sensors_health"],
            3,
        )
        self.assertAlmostEqual(
            raw_tx_by_send_id["send-001"]["native_state_snapshot"]["target_fields"]["drop_rate_comm_fraction"],
            0.12,
        )

    def test_build_packets_accepts_param_value_stream_slightly_before_request_timestamp(self) -> None:
        run_rows = [
            {
                "real_iso": "2026-03-28T20:00:00Z",
                "real_ms": "1000010",
                "send_start_ms": "1000010",
                "send_end_ms": "1000018",
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
                "command": "PARAM_REQUEST_LIST",
                "command_family": "read_only_inspection",
                "arguments_json": "{}",
                "meta_json": benign_meta_json(),
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
                "latency_ms": "8",
                "send_exception": "",
            },
            {
                "real_iso": "2026-03-28T20:00:00Z",
                "real_ms": "1000020",
                "send_start_ms": "1000020",
                "send_end_ms": "1000028",
                "virtual_day": "0",
                "virtual_time": "00:00:02",
                "virtual_seconds": "2",
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
                "target_stream_index": "1",
                "send_id": "send-002",
                "args_sha256": "hash-002",
                "command": "PARAM_REQUEST_LIST",
                "command_family": "read_only_inspection",
                "arguments_json": "{}",
                "meta_json": benign_meta_json(),
                "run_id": "0",
                "heartbeat_type": "2",
                "heartbeat_autopilot": "3",
                "heartbeat_base_mode": "81",
                "heartbeat_custom_mode": "4",
                "response_name": "PARAM_VALUE",
                "response_code": "9",
                "response_text": "FENCE_RADIUS",
                "response_json": json.dumps({"param_id": "FENCE_RADIUS", "param_type": 9, "param_value": 120.0}, separators=(",", ":"), sort_keys=True),
                "observed_messages_json": "[]",
                "timeout": "0",
                "latency_ms": "8",
                "send_exception": "",
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            pcap_path = tmp_path / "mavlink.pcap"
            heartbeat = mavlink_v2_frame(0, heartbeat_payload(), seq=1, sysid=1, compid=1)
            request_1 = mavlink_v2_frame(21, param_request_list_payload(), seq=10, sysid=21, compid=190)
            request_2 = mavlink_v2_frame(21, param_request_list_payload(), seq=11, sysid=21, compid=190)
            response_1 = mavlink_v2_frame(22, param_value_payload("FRAME_CLASS", 1.0, param_index=3), seq=12, sysid=1, compid=1)
            response_2 = mavlink_v2_frame(22, param_value_payload("FENCE_RADIUS", 120.0, param_index=4), seq=13, sysid=1, compid=1)
            write_pcap(
                pcap_path,
                [
                    (1000.005, ethernet_tcp_packet("192.168.164.2", "192.168.164.12", 5760, 50300, heartbeat)),
                    (1000.010, ethernet_tcp_packet("192.168.164.12", "192.168.164.2", 50300, 5760, request_1)),
                    (1000.018, ethernet_tcp_packet("192.168.164.2", "192.168.164.12", 5760, 50300, response_1)),
                    (1000.019, ethernet_tcp_packet("192.168.164.2", "192.168.164.12", 5760, 50300, response_2)),
                    (1000.020, ethernet_tcp_packet("192.168.164.12", "192.168.164.2", 50300, 5760, request_2)),
                ],
            )

            result = build_packets_from_real_artifacts(
                run_rows,
                pcap_path=pcap_path,
                capture_interface="bridge100",
                capture_backend="rdctl_vm_bridge",
                source_artifact_paths=[str(pcap_path.resolve())],
                strict=True,
            )

        self.assertEqual(len(result.transactions), 2)
        tx_by_send_id = {str(item["send_id"]): item for item in result.transactions}
        self.assertEqual(tx_by_send_id["send-002"]["response_payload"]["param_id"], "FENCE_RADIUS")
        self.assertTrue(all(observation["response_direction_seen"] for observation in result.observations))

    def test_build_packets_resyncs_after_stale_tcp_tail_before_command_ack(self) -> None:
        run_rows = [
            {
                "real_iso": "2026-03-28T20:00:00Z",
                "real_ms": "1000010",
                "send_start_ms": "1000010",
                "send_end_ms": "1000020",
                "virtual_day": "0",
                "virtual_time": "00:00:01",
                "virtual_seconds": "1",
                "source_service": "ops_secondary",
                "source_ip": "192.168.164.13",
                "source_system_id": "22",
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
                "command": "MAV_CMD_DO_SET_MODE",
                "command_family": "mission_sequence_control",
                "arguments_json": json.dumps({"base_mode": 1, "custom_mode": 5, "custom_submode": 0}, separators=(",", ":"), sort_keys=True),
                "meta_json": json.dumps(
                    {
                        "class_label": 0,
                        "class_name": "benign",
                        "attack_family": "mode_verification",
                        "actor_role": "ops_backup",
                        "actor_trust": 0.92,
                        "episode_id": 0,
                        "phase": "recovery",
                        "run_id": 0,
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                "run_id": "0",
                "heartbeat_type": "2",
                "heartbeat_autopilot": "3",
                "heartbeat_base_mode": "81",
                "heartbeat_custom_mode": "0",
                "response_name": "COMMAND_ACK",
                "response_code": "0",
                "response_text": "MAV_RESULT_ACCEPTED",
                "response_json": json.dumps(
                    {
                        "command": 176,
                        "result": 0,
                        "progress": 0,
                        "result_param2": 0,
                        "target_system": 22,
                        "target_component": 190,
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                "observed_messages_json": "[]",
                "timeout": "0",
                "latency_ms": "10",
                "send_exception": "",
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            pcap_path = tmp_path / "mavlink.pcap"
            request_payload = struct.pack(
                "<7fHBBB",
                1.0,
                5.0,
                0.0,
                0.0,
                0.0,
                0.0,
                0.0,
                176,
                1,
                1,
                0,
            )
            ack_payload = struct.pack("<HBBiBB", 176, 0, 0, 0, 22, 190)
            request = mavlink_v2_frame(76, request_payload, seq=10, sysid=22, compid=190)
            ack = mavlink_v2_frame(77, ack_payload, seq=11, sysid=1, compid=1)
            heartbeat = mavlink_v2_frame(0, heartbeat_payload(custom_mode=5, base_mode=89, system_status=3), seq=12, sysid=1, compid=1)
            stale_partial_tail = heartbeat[:5]
            write_pcap(
                pcap_path,
                [
                    (1000.005, ethernet_tcp_packet("192.168.164.2", "192.168.164.13", 5760, 49106, stale_partial_tail)),
                    (1000.010, ethernet_tcp_packet("192.168.164.13", "192.168.164.2", 49106, 5760, request)),
                    (1000.011, ethernet_tcp_packet("192.168.164.2", "192.168.164.13", 5760, 49106, heartbeat + ack)),
                ],
            )

            result = build_packets_from_real_artifacts(
                run_rows,
                pcap_path=pcap_path,
                capture_interface="bridge100",
                capture_backend="rdctl_vm_bridge",
                source_artifact_paths=[str(pcap_path.resolve())],
                strict=True,
            )

        self.assertEqual(len(result.transactions), 1)
        self.assertTrue(result.observations[0]["request_wire_seen"])
        self.assertTrue(result.observations[0]["response_direction_seen"])
        self.assertEqual(result.observations[0]["response_message_name"], "COMMAND_ACK")
        validate_raw_packet_records(result.raw_packets)
        validate_raw_transaction_records(result.raw_transactions)

    def test_build_packets_records_command_ack_failure_reason(self) -> None:
        run_rows = [
            {
                "real_iso": "2026-03-28T20:00:10Z",
                "real_ms": "2000010",
                "send_start_ms": "2000010",
                "send_end_ms": "2000020",
                "virtual_day": "0",
                "virtual_time": "00:00:10",
                "virtual_seconds": "10",
                "source_service": "red_primary",
                "source_ip": "192.168.164.22",
                "source_system_id": "31",
                "source_component_id": "191",
                "target_service": "mavlink_vehicle",
                "target_ip": "192.168.164.2",
                "target_endpoint": "tcp:mavlink_vehicle:5760",
                "target_system_id": "1",
                "target_component_id": "1",
                "target_stream_id": "mavlink_vehicle@tcp:mavlink_vehicle:5760",
                "target_stream_index": "0",
                "send_id": "send-ack-fail",
                "args_sha256": "hash-ack-fail",
                "command": "MAV_CMD_COMPONENT_ARM_DISARM",
                "command_family": "safety_critical_control",
                "arguments_json": json.dumps({"param1": 1.0, "param2": 0.0}, separators=(",", ":"), sort_keys=True),
                "meta_json": json.dumps(
                    {
                        "class_label": 1,
                        "class_name": "cyber",
                        "attack_family": "unsafe_control",
                        "actor_role": "external",
                        "actor_trust": 0.24,
                        "episode_id": 0,
                        "phase": "science",
                        "run_id": 0,
                    },
                    separators=(",", ":"),
                    sort_keys=True,
                ),
                "run_id": "0",
                "response_name": "COMMAND_ACK",
                "response_code": "4",
                "response_text": "FAILED",
                "response_json": json.dumps({"command": 400, "result": 4}, separators=(",", ":"), sort_keys=True),
                "observed_messages_json": "[]",
                "timeout": "0",
                "latency_ms": "10",
                "send_exception": "",
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            pcap_path = tmp_path / "mavlink_ack.pcap"
            request = mavlink_v2_frame(
                76,
                struct.pack("<7fHBBB", 1.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 400, 1, 1, 0),
                seq=20,
                sysid=31,
                compid=191,
            )
            ack = mavlink_v2_frame(77, struct.pack("<HB", 400, 4), seq=21, sysid=1, compid=1)
            write_pcap(
                pcap_path,
                [
                    (2000.000, ethernet_udp_packet("192.168.164.2", "192.168.164.3", 14550, 14550, mavlink_v2_frame(0, heartbeat_payload(), seq=1, sysid=1, compid=1))),
                    (2000.010, ethernet_tcp_packet("192.168.164.22", "192.168.164.2", 50200, 5760, request)),
                    (2000.020, ethernet_tcp_packet("192.168.164.2", "192.168.164.22", 5760, 50200, ack)),
                ],
            )
            result = build_packets_from_real_artifacts(
                run_rows,
                pcap_path=pcap_path,
                capture_interface="bridge100",
                capture_backend="rdctl_vm_bridge",
                source_artifact_paths=[str(pcap_path.resolve())],
                strict=True,
            )

        self.assertEqual(len(result.transactions), 1)
        transaction = result.transactions[0]
        self.assertEqual(transaction["sat_success"], 0)
        self.assertEqual(transaction["response_code"], 4)
        self.assertEqual(transaction["reason"], "command_ack_reject")
        self.assertEqual(result.raw_transactions[0]["outcome"]["raw_code"], 4.0)


if __name__ == "__main__":
    unittest.main()
