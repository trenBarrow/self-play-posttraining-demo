from __future__ import annotations

import json
import socket
import sys
import tempfile
import unittest
from pathlib import Path

import dpkt

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime import packets_to_transactions, transactions_to_rows
from tools.fprime_real.generate_dataset import build_shared_fprime_artifact_layers
from tools.fprime_real.packet_fidelity import build_packets_from_real_artifacts
from tools.fprime_real.runtime_layout import host_downlink_records_path, host_event_log_path


def packet_for(packets: list[dict[str, object]], command: str, packet_kind: str) -> dict[str, object]:
    for packet in packets:
        if packet.get("command") == command and packet.get("packet_kind") == packet_kind:
            return packet
    raise AssertionError(f"Missing packet for {command} / {packet_kind}")


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


def write_pcap(path: Path, frames: list[tuple[float, bytes]]) -> None:
    with path.open("wb") as handle:
        writer = dpkt.pcap.Writer(handle, linktype=dpkt.pcap.DLT_EN10MB)
        for ts, frame in frames:
            writer.writepkt(frame, ts=ts)


class PacketFidelityTests(unittest.TestCase):
    def test_build_packets_handles_multiple_commands_on_one_tcp_connection(self) -> None:
        run_rows = [
            {
                "real_iso": "2026-03-23T19:00:01Z",
                "real_ms": "1774290001000",
                "send_start_ms": "1774290001000",
                "send_end_ms": "1774290001010",
                "virtual_day": "0",
                "virtual_time": "00:00:20",
                "virtual_seconds": "20",
                "source_service": "ops_b1",
                "source_ip": "192.168.144.22",
                "target_service": "fprime_a",
                "target_ip": "192.168.144.2",
                "target_tts_port": "50050",
                "target_stream_id": "fprime_a:50050",
                "target_stream_index": "0",
                "tts_port": "50050",
                "send_id": "send-reuse-1",
                "args_sha256": "hash-1",
                "command": "cmdDisp.CMD_NO_OP",
                "arguments_json": "[]",
                "meta_json": json.dumps({"class_label": 0, "class_name": "benign", "attack_family": "none", "actor_role": "ops_primary", "actor_trust": 0.97, "episode_id": 0, "phase": "startup"}),
                "gds_accept": "1",
                "sat_success": "1",
                "timeout": "0",
                "response_code": "0",
                "reason": "completed",
                "latency_ms": "30",
                "event_names_json": "[]",
            },
            {
                "real_iso": "2026-03-23T19:00:01Z",
                "real_ms": "1774290001050",
                "send_start_ms": "1774290001050",
                "send_end_ms": "1774290001060",
                "virtual_day": "0",
                "virtual_time": "00:00:21",
                "virtual_seconds": "21",
                "source_service": "ops_b1",
                "source_ip": "192.168.144.22",
                "target_service": "fprime_a",
                "target_ip": "192.168.144.2",
                "target_tts_port": "50050",
                "target_stream_id": "fprime_a:50050",
                "target_stream_index": "1",
                "tts_port": "50050",
                "send_id": "send-reuse-2",
                "args_sha256": "hash-2",
                "command": "cmdDisp.CMD_CLEAR_TRACKING",
                "arguments_json": "[]",
                "meta_json": json.dumps({"class_label": 0, "class_name": "benign", "attack_family": "none", "actor_role": "ops_primary", "actor_trust": 0.97, "episode_id": 0, "phase": "startup"}),
                "gds_accept": "1",
                "sat_success": "1",
                "timeout": "0",
                "response_code": "0",
                "reason": "completed",
                "latency_ms": "30",
                "event_names_json": "[]",
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            send_log_path = tmp_path / "send_log.jsonl"
            pcap_path = tmp_path / "traffic.pcap"
            send_log_path.write_text("\n".join(json.dumps(row, separators=(",", ":")) for row in run_rows) + "\n", encoding="utf-8")
            (tmp_path / "command.log").write_text("", encoding="utf-8")
            (tmp_path / "node_a_event.log").write_text("", encoding="utf-8")
            (tmp_path / "node_b_event.log").write_text("", encoding="utf-8")
            write_pcap(
                pcap_path,
                [
                    (1774290001.001, ethernet_tcp_packet("192.168.144.22", "192.168.144.2", 50101, 50050, b"REQ1")),
                    (1774290001.010, ethernet_tcp_packet("192.168.144.2", "192.168.144.22", 50050, 50101, b"RESP1")),
                    (1774290001.055, ethernet_tcp_packet("192.168.144.22", "192.168.144.2", 50101, 50050, b"REQ2")),
                    (1774290001.066, ethernet_tcp_packet("192.168.144.2", "192.168.144.22", 50050, 50101, b"RESP2")),
                ],
            )

            result = build_packets_from_real_artifacts(
                run_rows,
                command_log_path=tmp_path / "command.log",
                event_log_paths={
                    "fprime_a": tmp_path / "node_a_event.log",
                    "fprime_b": tmp_path / "node_b_event.log",
                },
                telemetry_record_paths={
                    "fprime_a": tmp_path / "missing_a.jsonl",
                    "fprime_b": tmp_path / "missing_b.jsonl",
                },
                send_log_path=send_log_path,
                pcap_path=pcap_path,
                capture_interface="br-test",
                capture_backend="local_docker_bridge",
                strict=False,
            )

        first_request = next(packet for packet in result.packets if packet.get("send_id") == "send-reuse-1" and packet.get("packet_kind") == "request")
        second_request = next(packet for packet in result.packets if packet.get("send_id") == "send-reuse-2" and packet.get("packet_kind") == "request")
        first_final = next(packet for packet in result.packets if packet.get("send_id") == "send-reuse-1" and packet.get("packet_kind") == "final")
        second_final = next(packet for packet in result.packets if packet.get("send_id") == "send-reuse-2" and packet.get("packet_kind") == "final")

        self.assertEqual(first_request["bytes_on_wire"], 4)
        self.assertEqual(second_request["bytes_on_wire"], 4)
        self.assertEqual(first_final["bytes_on_wire"], 5)
        self.assertEqual(second_final["bytes_on_wire"], 5)
        self.assertLess(first_request["ts_ms"], second_request["ts_ms"])
        self.assertLess(first_final["ts_ms"], second_final["ts_ms"])

    def test_build_packets_from_logs_channels_and_pcap(self) -> None:
        run_rows = [
            {
                "real_iso": "2026-03-23T19:00:01Z",
                "real_ms": "1774290001000",
                "send_start_ms": "1774290001000",
                "send_end_ms": "1774290001095",
                "virtual_day": "0",
                "virtual_time": "00:00:20",
                "virtual_seconds": "20",
                "source_service": "ops_b1",
                "source_ip": "192.168.144.22",
                "target_service": "fprime_a",
                "target_ip": "192.168.144.2",
                "target_tts_port": "50050",
                "target_stream_id": "fprime_a:50050",
                "target_stream_index": "0",
                "tts_port": "50050",
                "send_id": "send-0001",
                "args_sha256": "hash-noop",
                "command": "cmdDisp.CMD_NO_OP",
                "arguments_json": "[]",
                "meta_json": json.dumps({"class_label": 0, "class_name": "benign", "attack_family": "none", "actor_role": "ops_primary", "actor_trust": 0.97, "episode_id": 0, "phase": "startup"}),
                "gds_accept": "1",
                "sat_success": "1",
                "timeout": "0",
                "response_code": "0",
                "reason": "completed",
                "latency_ms": "95",
                "event_names_json": "[]",
            },
            {
                "real_iso": "2026-03-23T19:00:02Z",
                "real_ms": "1774290002000",
                "send_start_ms": "1774290002000",
                "send_end_ms": "1774290002080",
                "virtual_day": "0",
                "virtual_time": "00:00:40",
                "virtual_seconds": "40",
                "source_service": "ops_a1",
                "source_ip": "192.168.144.12",
                "target_service": "fprime_b",
                "target_ip": "192.168.144.3",
                "target_tts_port": "50050",
                "target_stream_id": "fprime_b:50050",
                "target_stream_index": "0",
                "tts_port": "50050",
                "send_id": "send-0002",
                "args_sha256": "hash-version",
                "command": "systemResources.VERSION",
                "arguments_json": "[]",
                "meta_json": json.dumps({"class_label": 0, "class_name": "benign", "attack_family": "none", "actor_role": "ops_primary", "actor_trust": 0.98, "episode_id": 0, "phase": "science"}),
                "gds_accept": "1",
                "sat_success": "1",
                "timeout": "0",
                "response_code": "0",
                "reason": "completed",
                "latency_ms": "80",
                "event_names_json": "[]",
            },
            {
                "real_iso": "2026-03-23T19:00:03Z",
                "real_ms": "1774290003000",
                "send_start_ms": "1774290003000",
                "send_end_ms": "1774290018000",
                "virtual_day": "0",
                "virtual_time": "00:01:00",
                "virtual_seconds": "60",
                "source_service": "red_b1",
                "source_ip": "192.168.144.41",
                "target_service": "fprime_a",
                "target_ip": "192.168.144.2",
                "target_tts_port": "50050",
                "target_stream_id": "fprime_a:50050",
                "target_stream_index": "1",
                "tts_port": "50050",
                "send_id": "send-0003",
                "args_sha256": "hash-shell",
                "command": "fileManager.ShellCommand",
                "arguments_json": json.dumps(["echo BAD", "/tmp/bad.log"]),
                "meta_json": json.dumps({"class_label": 1, "class_name": "cyber", "attack_family": "masquerade_abuse", "actor_role": "external", "actor_trust": 0.20, "episode_id": 0, "phase": "downlink"}),
                "gds_accept": "1",
                "sat_success": "0",
                "timeout": "1",
                "response_code": "3",
                "reason": "timeout",
                "latency_ms": "15000",
                "event_names_json": "[]",
            },
            {
                "real_iso": "2026-03-23T19:00:04Z",
                "real_ms": "1774290004000",
                "send_start_ms": "1774290004000",
                "send_end_ms": "1774290004012",
                "virtual_day": "0",
                "virtual_time": "00:01:20",
                "virtual_seconds": "80",
                "source_service": "red_a1",
                "source_ip": "192.168.144.31",
                "target_service": "fprime_b",
                "target_ip": "192.168.144.3",
                "target_tts_port": "50050",
                "target_stream_id": "fprime_b:50050",
                "target_stream_index": "1",
                "tts_port": "50050",
                "send_id": "send-0004",
                "args_sha256": "hash-reject",
                "command": "cmdDisp.CMD_TEST_CMD_1",
                "arguments_json": json.dumps(["5000", "500.0", "999"]),
                "meta_json": json.dumps({"class_label": 1, "class_name": "cyber", "attack_family": "arg_bruteforce", "actor_role": "external", "actor_trust": 0.22, "episode_id": 0, "phase": "standby"}),
                "gds_accept": "0",
                "sat_success": "0",
                "timeout": "0",
                "response_code": "2",
                "reason": "arg_reject",
                "latency_ms": "12",
                "event_names_json": "[]",
            },
        ]

        command_log_text = "\n".join(
            [
                "2026-03-23 19:00:01,(2(0)-1774290001:5000),cmdDisp.CMD_NO_OP,1280,[ ]",
                "2026-03-23 19:00:02,(2(0)-1774290002:8000),systemResources.VERSION,18945,[ ]",
                "2026-03-23 19:00:03,(2(0)-1774290003:3000),fileManager.ShellCommand,2052,[ ' e c h o   B A D ' ,   ' / t m p / b a d . l o g ' ]",
            ]
        )
        node_a_events = "\n".join(
            [
                "2026-03-23 19:00:01,(2(0)-1774290001:30000),cmdDisp.OpCodeDispatched,1281,EventSeverity.COMMAND,Opcode 0x500 dispatched to port 0",
                "2026-03-23 19:00:01,(2(0)-1774290001:60000),cmdDisp.NoOpReceived,1287,EventSeverity.ACTIVITY_HI,Received a NO-OP command",
                "2026-03-23 19:00:01,(2(0)-1774290001:95000),cmdDisp.OpCodeCompleted,1282,EventSeverity.COMMAND,Opcode 0x500 completed",
                "2026-03-23 19:00:03,(2(0)-1774290003:10000),cmdDisp.OpCodeDispatched,1281,EventSeverity.COMMAND,Opcode 0x804 dispatched to port 4",
            ]
        )
        node_b_events = "\n".join(
            [
                "2026-03-23 19:00:02,(2(0)-1774290002:20000),cmdDisp.OpCodeDispatched,1281,EventSeverity.COMMAND,Opcode 0x4a01 dispatched to port 0",
                "2026-03-23 19:00:02,(2(0)-1774290002:80000),cmdDisp.OpCodeCompleted,1282,EventSeverity.COMMAND,Opcode 0x4a01 completed",
            ]
        )
        node_a_channels = "\n".join(
            [
                "2026-03-23 19:00:00,(2(0)-1774290000:900000),systemResources.CPU,18948,12.50 percent",
                "2026-03-23 19:00:00,(2(0)-1774290000:900100),systemResources.CPU_00,18949,9.00 percent",
                "2026-03-23 19:00:00,(2(0)-1774290000:900200),systemResources.CPU_01,18950,16.00 percent",
                "2026-03-23 19:00:00,(2(0)-1774290000:900300),cmdDisp.CommandsDispatched,1280,4",
                "2026-03-23 19:00:00,(2(0)-1774290000:900400),rateGroup1.RgMaxTime,3300,4200 us",
                "2026-03-23 19:00:00,(2(0)-1774290000:900450),systemResources.FRAMEWORK_VERSION,2000,v3.2.0",
                "2026-03-23 19:00:00,(2(0)-1774290000:900500),unknown.Channel,999,hello",
            ]
        )
        node_b_channels = "\n".join(
            [
                "2026-03-23 19:00:01,(2(0)-1774290001:900000),systemResources.CPU,18948,8.50 percent",
                "2026-03-23 19:00:01,(2(0)-1774290001:900100),blockDrv.BD_Cycles,256,7",
                "2026-03-23 19:00:01,(2(0)-1774290001:900200),fileDownlink.Warnings,1794,1",
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            command_log_path = tmp_path / "command.log"
            send_log_path = tmp_path / "send_log.jsonl"
            node_a_event_path = tmp_path / "node_a_event.log"
            node_b_event_path = tmp_path / "node_b_event.log"
            node_a_channel_path = tmp_path / "node_a_channel.log"
            node_b_channel_path = tmp_path / "node_b_channel.log"
            pcap_path = tmp_path / "traffic.pcap"
            command_log_path.write_text(command_log_text, encoding="utf-8")
            send_log_path.write_text("\n".join(json.dumps(row, separators=(",", ":")) for row in run_rows) + "\n", encoding="utf-8")
            node_a_event_path.write_text(node_a_events, encoding="utf-8")
            node_b_event_path.write_text(node_b_events, encoding="utf-8")
            node_a_channel_path.write_text(node_a_channels, encoding="utf-8")
            node_b_channel_path.write_text(node_b_channels, encoding="utf-8")
            write_pcap(
                pcap_path,
                [
                    (1774290001.006, ethernet_tcp_packet("192.168.144.22", "192.168.144.2", 50101, 50050, b"1234567890123")),
                    (1774290001.040, ethernet_tcp_packet("192.168.144.22", "192.168.144.2", 50101, 50050, b"abcdefghijklmnopqrstuvwxyzABCDEFGH")),
                    (1774290001.060, ethernet_tcp_packet("192.168.144.2", "192.168.144.22", 50050, 50101, b"x" * 130)),
                    (1774290001.090, ethernet_tcp_packet("192.168.144.2", "192.168.144.22", 50050, 50101, b"y" * 25)),
                    (1774290002.009, ethernet_tcp_packet("192.168.144.12", "192.168.144.3", 50202, 50050, b"ABCDEFGHIJKLM")),
                    (1774290002.070, ethernet_tcp_packet("192.168.144.3", "192.168.144.12", 50050, 50202, b"")),
                    (1774290003.004, ethernet_tcp_packet("192.168.144.41", "192.168.144.2", 50303, 50050, b"timeout-request")),
                ],
            )

            result = build_packets_from_real_artifacts(
                run_rows,
                command_log_path=command_log_path,
                event_log_paths={"fprime_a": node_a_event_path, "fprime_b": node_b_event_path},
                channel_log_paths={"fprime_a": node_a_channel_path, "fprime_b": node_b_channel_path},
                send_log_path=send_log_path,
                pcap_path=pcap_path,
                capture_backend="rdctl_vm_bridge",
                capture_interface="br-40f7ae8220ee",
                strict=False,
            )

        packets = result.packets
        self.assertEqual({packet["packet_kind"] for packet in packets}, {"telemetry", "request", "uplink", "sat_response", "final"})

        no_op_request = packet_for(packets, "cmdDisp.CMD_NO_OP", "request")
        no_op_uplink = packet_for(packets, "cmdDisp.CMD_NO_OP", "uplink")
        no_op_sat_response = packet_for(packets, "cmdDisp.CMD_NO_OP", "sat_response")
        no_op_final = packet_for(packets, "cmdDisp.CMD_NO_OP", "final")
        self.assertEqual(no_op_request["bytes_on_wire"], 47)
        self.assertEqual(no_op_request["ts_source"], "pcap")
        self.assertEqual(no_op_request["send_id"], "send-0001")
        self.assertEqual(no_op_request["target_stream_id"], "fprime_a:50050")
        self.assertEqual(no_op_request["target_stream_index"], 0)
        self.assertEqual(no_op_request["src_ip"], "192.168.144.22")
        self.assertEqual(no_op_request["dst_port"], 50050)
        self.assertEqual(no_op_uplink["event_name"], "cmdDisp.OpCodeDispatched")
        self.assertEqual(no_op_sat_response["event_name"], "cmdDisp.NoOpReceived")
        self.assertEqual(no_op_final["bytes_on_wire"], 155)
        self.assertEqual(no_op_final["final_observed_on_wire"], 1)
        self.assertEqual(no_op_final["src_ip"], "192.168.144.2")
        self.assertEqual(no_op_final["dst_ip"], "192.168.144.22")

        version_sat_response = packet_for(packets, "systemResources.VERSION", "sat_response")
        version_final = packet_for(packets, "systemResources.VERSION", "final")
        self.assertEqual(version_sat_response["event_name"], "cmdDisp.OpCodeCompleted")
        self.assertEqual(version_final["bytes_on_wire"], 0)
        self.assertEqual(version_final["final_observed_on_wire"], 1)
        self.assertEqual(version_final["response_direction_seen"], 1)

        shell_request = packet_for(packets, "fileManager.ShellCommand", "request")
        shell_final = packet_for(packets, "fileManager.ShellCommand", "final")
        self.assertEqual(shell_request["bytes_on_wire"], len(b"timeout-request"))
        self.assertEqual(shell_request["src_ip"], "192.168.144.41")
        self.assertEqual(shell_final["bytes_on_wire"], 0)
        self.assertEqual(shell_final["timeout"], 1)
        self.assertEqual(shell_final["final_observed_on_wire"], 0)

        reject_request = packet_for(packets, "cmdDisp.CMD_TEST_CMD_1", "request")
        reject_final = packet_for(packets, "cmdDisp.CMD_TEST_CMD_1", "final")
        self.assertEqual(reject_request["observed_on_wire"], 0)
        self.assertEqual(reject_request["ts_source"], "send_log")
        self.assertNotIn("src_ip", reject_request)
        self.assertEqual(reject_final["bytes_on_wire"], 0)
        self.assertEqual(reject_final["ts_source"], "send_row")

        telemetry_packets = [packet for packet in packets if packet["packet_kind"] == "telemetry"]
        self.assertGreaterEqual(len(telemetry_packets), 8)
        self.assertIn("systemResources.FRAMEWORK_VERSION", result.channel_inventory["nodes"]["fprime_a"]["inventory_only"])
        self.assertIn("unknown.Channel", result.channel_inventory["nodes"]["fprime_a"]["unknown"])
        self.assertEqual(result.provenance_summary["schema_version"], "real_fprime_v2")
        self.assertEqual(result.provenance_summary["capture_backend"], "rdctl_vm_bridge")
        self.assertEqual(result.provenance_summary["capture_interface"], "br-40f7ae8220ee")
        self.assertEqual(result.provenance_summary["pcap_identity_mode"], "bridge_ip_5tuple")
        self.assertEqual(result.provenance_summary["serialization_violations"], 0)
        self.assertEqual(result.provenance_summary["send_log_records"], 4)
        self.assertEqual(result.provenance_summary["rows_with_send_id"], 4)
        self.assertEqual(result.provenance_summary["rows_with_target_stream_id"], 4)
        self.assertEqual(result.provenance_summary["rows_with_source_ip"], 4)
        self.assertEqual(result.provenance_summary["rows_with_target_ip"], 4)
        self.assertEqual(result.provenance_summary["request_anchor_sources"]["pcap"], 3)
        self.assertEqual(result.provenance_summary["request_anchor_sources"]["send_log"], 1)
        self.assertEqual(result.provenance_summary["pcap_5tuple_matches"], 3)
        self.assertEqual(len(result.observations), 4)
        shell_observation = next(item for item in result.observations if item["command"] == "fileManager.ShellCommand")
        self.assertTrue(shell_observation["request_wire_seen"])
        self.assertFalse(shell_observation["response_direction_seen"])
        reject_observation = next(item for item in result.observations if item["command"] == "cmdDisp.CMD_TEST_CMD_1")
        self.assertFalse(reject_observation["request_wire_seen"])

    def test_build_packets_prefers_decoded_downlink_records_for_telemetry(self) -> None:
        run_rows = [
            {
                "real_iso": "2026-03-23T20:00:01Z",
                "real_ms": "1774293601000",
                "send_start_ms": "1774293601000",
                "send_end_ms": "1774293601040",
                "virtual_day": "0",
                "virtual_time": "00:20:00",
                "virtual_seconds": "1200",
                "source_service": "ops_b1",
                "source_ip": "192.168.144.22",
                "target_service": "fprime_a",
                "target_ip": "192.168.144.2",
                "target_tts_port": "50050",
                "target_stream_id": "fprime_a:50050",
                "target_stream_index": "0",
                "tts_port": "50050",
                "send_id": "send-telemetry",
                "args_sha256": "hash-noop",
                "command": "cmdDisp.CMD_NO_OP",
                "arguments_json": "[]",
                "meta_json": json.dumps({"class_label": 0, "class_name": "benign", "attack_family": "none", "actor_role": "ops_primary", "actor_trust": 1.0, "episode_id": 0, "phase": "startup"}),
                "gds_accept": "1",
                "sat_success": "1",
                "timeout": "0",
                "response_code": "0",
                "reason": "completed",
                "latency_ms": "60",
            }
        ]
        command_log_text = "2026-03-23 20:00:01,(2(0)-1774293601:10000),cmdDisp.CMD_NO_OP,1280,[ ]\n"
        node_a_events = "\n".join(
            [
                "2026-03-23 20:00:01,(2(0)-1774293601:20000),cmdDisp.OpCodeDispatched,1281,EventSeverity.COMMAND,Opcode 0x500 dispatched to port 0",
                "2026-03-23 20:00:01,(2(0)-1774293601:60000),cmdDisp.OpCodeCompleted,1282,EventSeverity.COMMAND,Opcode 0x500 completed",
            ]
        )

        node_a_downlink = "\n".join(
            [
                json.dumps(
                    {
                        "kind": "telemetry",
                        "node_service": "fprime_a",
                        "raw_time": "(2(0)-1774293600:950000)",
                        "bytes_on_wire": 144,
                        "channels": [
                            {"name": "systemResources.CPU", "display_text": "12.50 percent"},
                            {"name": "cmdDisp.CommandsDispatched", "display_text": "4"},
                        ],
                    },
                    separators=(",", ":"),
                ),
                json.dumps(
                    {
                        "kind": "telemetry",
                        "node_service": "fprime_b",
                        "raw_time": "(2(0)-1774293600:955000)",
                        "bytes_on_wire": 96,
                        "channels": [
                            {"name": "systemResources.CPU", "display_text": "8.25 percent"},
                        ],
                    },
                    separators=(",", ":"),
                ),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            command_log_path = tmp_path / "command.log"
            send_log_path = tmp_path / "send_log.jsonl"
            node_a_event_path = tmp_path / "node_a_event.log"
            node_b_event_path = tmp_path / "node_b_event.log"
            node_a_downlink_path = tmp_path / "node_a_downlink.jsonl"
            node_b_downlink_path = tmp_path / "node_b_downlink.jsonl"
            pcap_path = tmp_path / "traffic.pcap"
            command_log_path.write_text(command_log_text, encoding="utf-8")
            send_log_path.write_text(json.dumps(run_rows[0], separators=(",", ":")) + "\n", encoding="utf-8")
            node_a_event_path.write_text(node_a_events, encoding="utf-8")
            node_b_event_path.write_text("", encoding="utf-8")
            node_a_downlink_path.write_text(node_a_downlink.splitlines()[0] + "\n", encoding="utf-8")
            node_b_downlink_path.write_text(node_a_downlink.splitlines()[1] + "\n", encoding="utf-8")
            write_pcap(
                pcap_path,
                [
                    (1774293601.005, ethernet_tcp_packet("192.168.144.22", "192.168.144.2", 50101, 50050, b"request")),
                    (1774293601.050, ethernet_tcp_packet("192.168.144.2", "192.168.144.22", 50050, 50101, b"reply")),
                ],
            )

            result = build_packets_from_real_artifacts(
                run_rows,
                command_log_path=command_log_path,
                event_log_paths={"fprime_a": node_a_event_path, "fprime_b": node_b_event_path},
                telemetry_record_paths={"fprime_a": node_a_downlink_path, "fprime_b": node_b_downlink_path},
                send_log_path=send_log_path,
                pcap_path=pcap_path,
                capture_backend="rdctl_vm_bridge",
                capture_interface="br-40f7ae8220ee",
                strict=True,
            )

        telemetry_packets = [packet for packet in result.packets if packet["packet_kind"] == "telemetry"]
        self.assertEqual(len(telemetry_packets), 2)
        self.assertEqual(telemetry_packets[0]["observed_on_wire"], 1)
        self.assertEqual(telemetry_packets[0]["ts_source"], "gds_recv_bin")
        self.assertGreater(telemetry_packets[0]["bytes_on_wire"], 0)
        self.assertIn("cpu_total_pct", telemetry_packets[0]["payload"])
        self.assertEqual(result.provenance_summary["telemetry_sources"], {"gds_recv_bin": 2})

    def test_same_command_different_targets_match_by_source_ip_and_send_id(self) -> None:
        run_rows = [
            {
                "real_iso": "2026-03-23T19:10:00Z",
                "real_ms": "1774290600000",
                "send_start_ms": "1774290600000",
                "send_end_ms": "1774290600060",
                "virtual_day": "0",
                "virtual_time": "00:10:00",
                "virtual_seconds": "600",
                "source_service": "ops_b1",
                "source_ip": "192.168.144.22",
                "target_service": "fprime_a",
                "target_ip": "192.168.144.2",
                "target_tts_port": "50050",
                "target_stream_id": "fprime_a:50050",
                "target_stream_index": "0",
                "tts_port": "50050",
                "send_id": "send-a",
                "args_sha256": "hash-a",
                "command": "cmdDisp.CMD_NO_OP",
                "arguments_json": "[]",
                "meta_json": json.dumps({"class_label": 0, "class_name": "benign", "attack_family": "none", "actor_role": "ops_primary", "actor_trust": 0.97, "episode_id": 0, "phase": "startup"}),
                "gds_accept": "1",
                "sat_success": "1",
                "timeout": "0",
                "response_code": "0",
                "reason": "completed",
                "latency_ms": "70",
            },
            {
                "real_iso": "2026-03-23T19:10:00Z",
                "real_ms": "1774290600020",
                "send_start_ms": "1774290600020",
                "send_end_ms": "1774290600080",
                "virtual_day": "0",
                "virtual_time": "00:10:00",
                "virtual_seconds": "600",
                "source_service": "red_b1",
                "source_ip": "192.168.144.41",
                "target_service": "fprime_b",
                "target_ip": "192.168.144.3",
                "target_tts_port": "50050",
                "target_stream_id": "fprime_b:50050",
                "target_stream_index": "0",
                "tts_port": "50050",
                "send_id": "send-b",
                "args_sha256": "hash-b",
                "command": "cmdDisp.CMD_NO_OP",
                "arguments_json": "[]",
                "meta_json": json.dumps({"class_label": 1, "class_name": "cyber", "attack_family": "masquerade_abuse", "actor_role": "external", "actor_trust": 0.3, "episode_id": 1, "phase": "science"}),
                "gds_accept": "1",
                "sat_success": "1",
                "timeout": "0",
                "response_code": "0",
                "reason": "completed",
                "latency_ms": "80",
            },
        ]
        command_log_text = "\n".join(
            [
                "2026-03-23 19:10:00,(2(0)-1774290600:15000),cmdDisp.CMD_NO_OP,1280,[ ]",
                "2026-03-23 19:10:00,(2(0)-1774290600:26000),cmdDisp.CMD_NO_OP,1280,[ ]",
            ]
        )
        node_a_events = "\n".join(
            [
                "2026-03-23 19:10:00,(2(0)-1774290600:30000),cmdDisp.OpCodeDispatched,1281,EventSeverity.COMMAND,Opcode 0x500 dispatched to port 0",
                "2026-03-23 19:10:00,(2(0)-1774290600:70000),cmdDisp.OpCodeCompleted,1282,EventSeverity.COMMAND,Opcode 0x500 completed",
            ]
        )
        node_b_events = "\n".join(
            [
                "2026-03-23 19:10:00,(2(0)-1774290600:40000),cmdDisp.OpCodeDispatched,1281,EventSeverity.COMMAND,Opcode 0x500 dispatched to port 0",
                "2026-03-23 19:10:00,(2(0)-1774290600:90000),cmdDisp.OpCodeCompleted,1282,EventSeverity.COMMAND,Opcode 0x500 completed",
            ]
        )
        node_a_channels = "2026-03-23 19:09:59,(2(0)-1774290599:950000),systemResources.CPU,18948,10.00 percent\n"
        node_b_channels = "2026-03-23 19:09:59,(2(0)-1774290599:960000),systemResources.CPU,18948,11.00 percent\n"

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            command_log_path = tmp_path / "command.log"
            send_log_path = tmp_path / "send_log.jsonl"
            node_a_event_path = tmp_path / "node_a_event.log"
            node_b_event_path = tmp_path / "node_b_event.log"
            node_a_channel_path = tmp_path / "node_a_channel.log"
            node_b_channel_path = tmp_path / "node_b_channel.log"
            pcap_path = tmp_path / "traffic.pcap"
            command_log_path.write_text(command_log_text, encoding="utf-8")
            send_log_path.write_text("\n".join(json.dumps(row, separators=(",", ":")) for row in run_rows) + "\n", encoding="utf-8")
            node_a_event_path.write_text(node_a_events, encoding="utf-8")
            node_b_event_path.write_text(node_b_events, encoding="utf-8")
            node_a_channel_path.write_text(node_a_channels, encoding="utf-8")
            node_b_channel_path.write_text(node_b_channels, encoding="utf-8")
            write_pcap(
                pcap_path,
                [
                    (1774290600.010, ethernet_tcp_packet("192.168.144.22", "192.168.144.2", 50101, 50050, b"ops")),
                    (1774290600.035, ethernet_tcp_packet("192.168.144.2", "192.168.144.22", 50050, 50101, b"r")),
                    (1774290600.025, ethernet_tcp_packet("192.168.144.41", "192.168.144.3", 50202, 50050, b"red")),
                    (1774290600.050, ethernet_tcp_packet("192.168.144.3", "192.168.144.41", 50050, 50202, b"r")),
                ],
            )

            result = build_packets_from_real_artifacts(
                run_rows,
                command_log_path=command_log_path,
                event_log_paths={"fprime_a": node_a_event_path, "fprime_b": node_b_event_path},
                channel_log_paths={"fprime_a": node_a_channel_path, "fprime_b": node_b_channel_path},
                send_log_path=send_log_path,
                pcap_path=pcap_path,
                capture_backend="rdctl_vm_bridge",
                capture_interface="br-40f7ae8220ee",
                strict=True,
            )

        request_a = next(packet for packet in result.packets if packet.get("send_id") == "send-a" and packet.get("packet_kind") == "request")
        request_b = next(packet for packet in result.packets if packet.get("send_id") == "send-b" and packet.get("packet_kind") == "request")
        final_a = next(packet for packet in result.packets if packet.get("send_id") == "send-a" and packet.get("packet_kind") == "final")
        final_b = next(packet for packet in result.packets if packet.get("send_id") == "send-b" and packet.get("packet_kind") == "final")
        self.assertEqual(request_a["dst"], "fprime_a")
        self.assertEqual(request_b["dst"], "fprime_b")
        self.assertEqual(request_a["src_ip"], "192.168.144.22")
        self.assertEqual(request_b["src_ip"], "192.168.144.41")
        self.assertEqual(request_a["ts_ms"], 1774290600010)
        self.assertEqual(request_b["ts_ms"], 1774290600025)
        self.assertEqual(request_a["target_stream_index"], 0)
        self.assertEqual(request_b["target_stream_index"], 0)
        self.assertEqual(final_a["dst_ip"], "192.168.144.22")
        self.assertEqual(final_b["dst_ip"], "192.168.144.41")

    def test_reconstruction_rejects_overlapping_target_windows(self) -> None:
        run_rows = [
            {
                "real_iso": "2026-03-23T19:10:00Z",
                "real_ms": "1774290600000",
                "send_start_ms": "1774290600000",
                "send_end_ms": "1774290600060",
                "virtual_day": "0",
                "virtual_time": "00:10:00",
                "virtual_seconds": "600",
                "source_service": "ops_b1",
                "source_ip": "192.168.144.22",
                "target_service": "fprime_a",
                "target_ip": "192.168.144.2",
                "target_tts_port": "50050",
                "target_stream_id": "fprime_a:50050",
                "target_stream_index": "0",
                "tts_port": "50050",
                "send_id": "send-a",
                "args_sha256": "hash-a",
                "command": "cmdDisp.CMD_NO_OP",
                "arguments_json": "[]",
                "meta_json": json.dumps({"class_label": 0, "class_name": "benign", "attack_family": "none", "actor_role": "ops_primary", "actor_trust": 0.97, "episode_id": 0, "phase": "startup"}),
                "gds_accept": "1",
                "sat_success": "1",
                "timeout": "0",
                "response_code": "0",
                "reason": "completed",
                "latency_ms": "70",
            },
            {
                "real_iso": "2026-03-23T19:10:00Z",
                "real_ms": "1774290600020",
                "send_start_ms": "1774290600020",
                "send_end_ms": "1774290600080",
                "virtual_day": "0",
                "virtual_time": "00:10:00",
                "virtual_seconds": "600",
                "source_service": "ops_a1",
                "source_ip": "192.168.144.12",
                "target_service": "fprime_a",
                "target_ip": "192.168.144.2",
                "target_tts_port": "50050",
                "target_stream_id": "fprime_a:50050",
                "target_stream_index": "1",
                "tts_port": "50050",
                "send_id": "send-b",
                "args_sha256": "hash-b",
                "command": "cmdDisp.CMD_NO_OP",
                "arguments_json": "[]",
                "meta_json": json.dumps({"class_label": 0, "class_name": "benign", "attack_family": "none", "actor_role": "ops_secondary", "actor_trust": 0.95, "episode_id": 1, "phase": "science"}),
                "gds_accept": "1",
                "sat_success": "1",
                "timeout": "0",
                "response_code": "0",
                "reason": "completed",
                "latency_ms": "70",
            },
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            command_log_path = tmp_path / "command.log"
            send_log_path = tmp_path / "send_log.jsonl"
            command_log_path.write_text("", encoding="utf-8")
            send_log_path.write_text("\n".join(json.dumps(row, separators=(",", ":")) for row in run_rows) + "\n", encoding="utf-8")
            (tmp_path / "node_a_event.log").write_text("", encoding="utf-8")
            (tmp_path / "node_b_event.log").write_text("", encoding="utf-8")
            write_pcap(tmp_path / "traffic.pcap", [])
            with self.assertRaises(SystemExit) as exc:
                build_packets_from_real_artifacts(
                    run_rows,
                    command_log_path=command_log_path,
                    event_log_paths={"fprime_a": tmp_path / "node_a_event.log", "fprime_b": tmp_path / "node_b_event.log"},
                    send_log_path=send_log_path,
                    pcap_path=tmp_path / "traffic.pcap",
                    capture_backend="rdctl_vm_bridge",
                    capture_interface="br-test",
                    strict=True,
                )
        self.assertIn("serialized-per-target", str(exc.exception))

    def test_reverse_direction_before_request_does_not_anchor_final_early(self) -> None:
        run_rows = [
            {
                "real_iso": "2026-03-23T19:20:00Z",
                "real_ms": "1774291200000",
                "send_start_ms": "1774291200000",
                "send_end_ms": "1774291200060",
                "virtual_day": "0",
                "virtual_time": "00:20:00",
                "virtual_seconds": "1200",
                "source_service": "ops_b1",
                "source_ip": "192.168.144.22",
                "target_service": "fprime_a",
                "target_ip": "192.168.144.2",
                "target_tts_port": "50050",
                "target_stream_id": "fprime_a:50050",
                "target_stream_index": "0",
                "tts_port": "50050",
                "send_id": "send-early-final",
                "args_sha256": "hash",
                "command": "cmdDisp.CMD_NO_OP",
                "arguments_json": "[]",
                "meta_json": json.dumps({"class_label": 0, "class_name": "benign", "attack_family": "none", "actor_role": "ops_primary", "actor_trust": 1.0, "episode_id": 0, "phase": "startup"}),
                "gds_accept": "1",
                "sat_success": "1",
                "timeout": "0",
                "response_code": "0",
                "reason": "completed",
                "latency_ms": "70",
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            command_log_path = tmp_path / "command.log"
            send_log_path = tmp_path / "send_log.jsonl"
            node_a_event_path = tmp_path / "node_a_event.log"
            node_b_event_path = tmp_path / "node_b_event.log"
            node_a_channel_path = tmp_path / "node_a_channel.log"
            node_b_channel_path = tmp_path / "node_b_channel.log"
            pcap_path = tmp_path / "traffic.pcap"
            command_log_path.write_text("", encoding="utf-8")
            send_log_path.write_text(json.dumps(run_rows[0], separators=(",", ":")) + "\n", encoding="utf-8")
            node_a_event_path.write_text("", encoding="utf-8")
            node_b_event_path.write_text("", encoding="utf-8")
            node_a_channel_path.write_text("2026-03-23 19:19:59,(2(0)-1774291199:950000),systemResources.CPU,18948,10.00 percent\n", encoding="utf-8")
            node_b_channel_path.write_text("", encoding="utf-8")
            write_pcap(
                pcap_path,
                [
                    (1774291199.999, ethernet_tcp_packet("192.168.144.2", "192.168.144.22", 50050, 50101, b"")),
                    (1774291200.010, ethernet_tcp_packet("192.168.144.22", "192.168.144.2", 50101, 50050, b"request")),
                    (1774291200.040, ethernet_tcp_packet("192.168.144.2", "192.168.144.22", 50050, 50101, b"reply")),
                ],
            )
            result = build_packets_from_real_artifacts(
                run_rows,
                command_log_path=command_log_path,
                event_log_paths={"fprime_a": node_a_event_path, "fprime_b": node_b_event_path},
                channel_log_paths={"fprime_a": node_a_channel_path, "fprime_b": node_b_channel_path},
                send_log_path=send_log_path,
                pcap_path=pcap_path,
                capture_backend="rdctl_vm_bridge",
                capture_interface="br-test",
                strict=False,
            )
        request = next(packet for packet in result.packets if packet.get("packet_kind") == "request")
        final = next(packet for packet in result.packets if packet.get("packet_kind") == "final")
        self.assertGreaterEqual(final["ts_ms"], request["ts_ms"])

    def test_equal_timestamp_packets_remain_in_lifecycle_order(self) -> None:
        run_rows = [
            {
                "real_iso": "2026-03-23T19:30:00Z",
                "real_ms": "1774291800000",
                "send_start_ms": "1774291800000",
                "send_end_ms": "1774291800010",
                "virtual_day": "0",
                "virtual_time": "00:30:00",
                "virtual_seconds": "1800",
                "source_service": "ops_a1",
                "source_ip": "192.168.144.12",
                "target_service": "fprime_b",
                "target_ip": "192.168.144.3",
                "target_tts_port": "50050",
                "target_stream_id": "fprime_b:50050",
                "target_stream_index": "0",
                "tts_port": "50050",
                "send_id": "send-tied-ts",
                "args_sha256": "hash-tied",
                "command": "cmdDisp.CMD_NO_OP",
                "arguments_json": "[]",
                "meta_json": json.dumps({"class_label": 0, "class_name": "benign", "attack_family": "none", "actor_role": "ops_primary", "actor_trust": 1.0, "episode_id": 0, "phase": "startup"}),
                "gds_accept": "1",
                "sat_success": "1",
                "timeout": "0",
                "response_code": "0",
                "reason": "completed",
                "latency_ms": "0",
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            command_log_path = tmp_path / "command.log"
            send_log_path = tmp_path / "send_log.jsonl"
            node_a_event_path = tmp_path / "node_a_event.log"
            node_b_event_path = tmp_path / "node_b_event.log"
            node_a_channel_path = tmp_path / "node_a_channel.log"
            node_b_channel_path = tmp_path / "node_b_channel.log"
            pcap_path = tmp_path / "traffic.pcap"
            command_log_path.write_text("", encoding="utf-8")
            send_log_path.write_text(json.dumps(run_rows[0], separators=(",", ":")) + "\n", encoding="utf-8")
            node_a_event_path.write_text("", encoding="utf-8")
            node_b_event_path.write_text("", encoding="utf-8")
            node_a_channel_path.write_text("", encoding="utf-8")
            node_b_channel_path.write_text("2026-03-23 19:29:59,(2(0)-1774291799:950000),systemResources.CPU,18948,11.00 percent\n", encoding="utf-8")
            write_pcap(
                pcap_path,
                [
                    (1774291800.000, ethernet_tcp_packet("192.168.144.12", "192.168.144.3", 50202, 50050, b"request")),
                    (1774291800.000, ethernet_tcp_packet("192.168.144.3", "192.168.144.12", 50050, 50202, b"reply")),
                ],
            )
            result = build_packets_from_real_artifacts(
                run_rows,
                command_log_path=command_log_path,
                event_log_paths={"fprime_a": node_a_event_path, "fprime_b": node_b_event_path},
                channel_log_paths={"fprime_a": node_a_channel_path, "fprime_b": node_b_channel_path},
                send_log_path=send_log_path,
                pcap_path=pcap_path,
                capture_backend="rdctl_vm_bridge",
                capture_interface="br-test",
                strict=False,
            )
        lifecycle = [
            packet
            for packet in result.packets
            if packet.get("send_id") == "send-tied-ts" and packet.get("packet_kind") != "telemetry"
        ]
        self.assertEqual([packet["packet_kind"] for packet in lifecycle], ["request", "final"])

    def test_reconstructed_real_artifacts_feed_shared_raw_and_canonical_outputs(self) -> None:
        run_rows = [
            {
                "real_iso": "2026-03-23T19:45:00Z",
                "real_ms": "1774292700000",
                "send_start_ms": "1774292700000",
                "send_end_ms": "1774292700010",
                "virtual_day": "0",
                "virtual_time": "00:45:00",
                "virtual_seconds": "2700",
                "source_service": "ops_b1",
                "source_ip": "192.168.144.22",
                "target_service": "fprime_a",
                "target_ip": "192.168.144.2",
                "target_tts_port": "50050",
                "target_stream_id": "fprime_a:50050",
                "target_stream_index": "0",
                "tts_port": "50050",
                "send_id": "send-shared-1",
                "args_sha256": "hash-shared-1",
                "command": "cmdDisp.CMD_NO_OP",
                "arguments_json": "[]",
                "meta_json": json.dumps({"class_label": 0, "class_name": "benign", "attack_family": "none", "actor_role": "ops_primary", "actor_trust": 0.97, "episode_id": 0, "phase": "startup"}),
                "gds_accept": "1",
                "sat_success": "1",
                "timeout": "0",
                "response_code": "0",
                "reason": "completed",
                "latency_ms": "30",
                "event_names_json": "[]",
            }
        ]

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            manifest_path = tmp_path / "manifest.json"
            schedule_path = tmp_path / "schedule.csv"
            run_log_path = tmp_path / "run_log.csv"
            command_log_path = tmp_path / "command.log"
            send_log_path = tmp_path / "send_log.jsonl"
            node_a_event_path = tmp_path / "node_a_event.log"
            node_b_event_path = tmp_path / "node_b_event.log"
            node_a_channel_path = tmp_path / "node_a_channel.log"
            node_b_channel_path = tmp_path / "node_b_channel.log"
            pcap_path = tmp_path / "traffic.pcap"
            runtime_root = tmp_path / "runtime_root"

            manifest_path.write_text("[]\n", encoding="utf-8")
            schedule_path.write_text("virtual_time,command\n00:45:00,cmdDisp.CMD_NO_OP\n", encoding="utf-8")
            run_log_path.write_text("command\ncmdDisp.CMD_NO_OP\n", encoding="utf-8")
            command_log_path.write_text("", encoding="utf-8")
            send_log_path.write_text(json.dumps(run_rows[0], separators=(",", ":")) + "\n", encoding="utf-8")
            node_a_event_path.write_text("", encoding="utf-8")
            node_b_event_path.write_text("", encoding="utf-8")
            node_a_channel_path.write_text("2026-03-23 19:44:59,(2(0)-1774292699:950000),systemResources.CPU,18948,11.00 percent\n", encoding="utf-8")
            node_b_channel_path.write_text("", encoding="utf-8")
            write_pcap(
                pcap_path,
                [
                    (1774292700.000, ethernet_tcp_packet("192.168.144.22", "192.168.144.2", 50101, 50050, b"request")),
                    (1774292700.020, ethernet_tcp_packet("192.168.144.2", "192.168.144.22", 50050, 50101, b"reply")),
                ],
            )
            event_log_path = host_event_log_path(runtime_root, "fprime_a")
            downlink_path = host_downlink_records_path(runtime_root, "fprime_a")
            event_log_path.parent.mkdir(parents=True, exist_ok=True)
            downlink_path.parent.mkdir(parents=True, exist_ok=True)
            event_log_path.write_text("", encoding="utf-8")
            downlink_path.write_text("", encoding="utf-8")

            result = build_packets_from_real_artifacts(
                run_rows,
                command_log_path=command_log_path,
                event_log_paths={"fprime_a": node_a_event_path, "fprime_b": node_b_event_path},
                channel_log_paths={"fprime_a": node_a_channel_path, "fprime_b": node_b_channel_path},
                send_log_path=send_log_path,
                pcap_path=pcap_path,
                capture_backend="rdctl_vm_bridge",
                capture_interface="br-test",
                strict=False,
            )
            transactions = packets_to_transactions(result.packets, reset_key="run_id")
            feature_rows = transactions_to_rows(transactions, reset_key="run_id")
            shared = build_shared_fprime_artifact_layers(
                result.packets,
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
                        "traffic_pcap": str(pcap_path.resolve()),
                    }
                ],
                capture_backend="rdctl_vm_bridge",
                capture_interface="br-test",
            )

        self.assertEqual(len(shared["raw_transactions"]), 1)
        self.assertEqual(len(shared["canonical_command_rows"]), 1)
        canonical_row = shared["canonical_command_rows"][0]
        self.assertEqual(canonical_row["audit_context"]["raw_command_name"], "cmdDisp.CMD_NO_OP")
        self.assertEqual(canonical_row["command_semantics"]["canonical_command_family"], "read_only_inspection")
        self.assertTrue(canonical_row["normalized_state"]["state_available"])
        self.assertIn("target_scheduler_pressure_ratio", canonical_row["normalized_state"])
        self.assertIn("target_telemetry_staleness_ratio", canonical_row["normalized_state"])
        self.assertGreaterEqual(canonical_row["observability"]["related_packet_count"], 2)
        self.assertTrue(any(path.endswith("traffic.pcap") for path in canonical_row["audit_context"]["source_artifact_paths"]))
        self.assertTrue(any(path.endswith("command.log") for path in canonical_row["audit_context"]["source_artifact_paths"]))


if __name__ == "__main__":
    unittest.main()
