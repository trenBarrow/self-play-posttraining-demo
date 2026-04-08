#!/usr/bin/env python3
"""Reconstruct real MAVLink packets and transactions from run logs and pcap evidence."""

from __future__ import annotations

import argparse
import json
import socket
import struct
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Iterable, Mapping

import dpkt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.mavlink_real.command_catalog import (
    COMMAND_SPECS,
    command_spec_for_name,
    numeric_command_id,
    source_identity_for_service,
)
from tools.mavlink_real.log_ingest import load_runtime_run_rows, row_meta
from tools.mavlink_real.runtime_layout import VEHICLE_SERVICE
from tools.mavlink_real.telemetry_ingest import (
    TELEMETRY_RECENT_THRESHOLD_MS,
    TelemetryRecord,
    latest_native_state_snapshot,
    telemetry_recent_for_time,
    telemetry_records_from_wire_messages,
)
from tools.shared.schema import (
    RAW_PACKET_SCHEMA_VERSION,
    RAW_TRANSACTION_SCHEMA_VERSION,
    validate_raw_packet_records,
    validate_raw_transaction_records,
)

SCHEMA_VERSION = "real_mavlink_v1"
TARGET_TCP_PORT = 5760
TARGET_UDP_PORT = 14550
REQUEST_WINDOW_EARLY_MS = 2000
REQUEST_WINDOW_LATE_MS = 4000
RESPONSE_WINDOW_LATE_MS = 8000
STREAMING_RESPONSE_WINDOW_EARLY_MS = 100
MAVLINK_V1_MAGIC = 0xFE
MAVLINK_V2_MAGIC = 0xFD
MAV_RESULT_ACCEPTED = 0
MISSION_ACK_ACCEPTED = 0
PACKET_KIND_ORDER = {
    "telemetry": 0,
    "request": 1,
    "final": 2,
}

MAVLINK_MESSAGE_NAMES = {
    0: "HEARTBEAT",
    1: "SYS_STATUS",
    20: "PARAM_REQUEST_READ",
    21: "PARAM_REQUEST_LIST",
    22: "PARAM_VALUE",
    23: "PARAM_SET",
    24: "GPS_RAW_INT",
    43: "MISSION_REQUEST_LIST",
    44: "MISSION_COUNT",
    45: "MISSION_CLEAR_ALL",
    47: "MISSION_ACK",
    76: "COMMAND_LONG",
    77: "COMMAND_ACK",
    125: "POWER_STATUS",
    147: "BATTERY_STATUS",
    148: "AUTOPILOT_VERSION",
    253: "STATUSTEXT",
}

REQUEST_MESSAGE_BY_SEND_KIND = {
    "command_long": "COMMAND_LONG",
    "param_request_read": "PARAM_REQUEST_READ",
    "param_request_list": "PARAM_REQUEST_LIST",
    "param_set": "PARAM_SET",
    "mission_request_list": "MISSION_REQUEST_LIST",
    "mission_clear_all": "MISSION_CLEAR_ALL",
}


@dataclass(frozen=True)
class PcapPacket:
    ts_ms: int
    transport_family: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    payload: bytes
    tcp_flags: int = 0


@dataclass(frozen=True)
class WireMessage:
    ts_ms: int
    transport_family: str
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    source_system_id: int
    source_component_id: int
    protocol_version: str
    message_id: int
    message_name: str
    payload: dict[str, Any]
    frame_len: int
    connection_key: tuple[str, int, str, int] | None


@dataclass
class PacketBuildResult:
    packets: list[dict[str, Any]]
    transactions: list[dict[str, Any]]
    raw_packets: list[dict[str, Any]]
    raw_transactions: list[dict[str, Any]]
    observations: list[dict[str, Any]]
    telemetry_records: list[dict[str, Any]]
    provenance_summary: dict[str, Any]
    source_artifact_paths: list[str] = field(default_factory=list)


class ReconstructionError(ValueError):
    """Raised when MAVLink artifacts cannot be reconstructed safely."""


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def save_json(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")


def _optional_int(value: Any) -> int | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, int):
        return int(value)
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return None


def _optional_number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _bool_flag(value: Any) -> int:
    return 1 if bool(value) else 0


def _json_object(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    if value in (None, ""):
        return {}
    try:
        payload = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return {}
    return payload if isinstance(payload, dict) else {}


def _json_list(value: Any) -> list[Any]:
    if isinstance(value, list):
        return list(value)
    if value in (None, ""):
        return []
    try:
        payload = json.loads(str(value))
    except (TypeError, ValueError, json.JSONDecodeError):
        return []
    return payload if isinstance(payload, list) else []


def _strip_null_terminated(value: bytes | bytearray) -> str:
    return bytes(value).split(b"\x00", 1)[0].decode("ascii", "ignore").strip()


def _packet_sort_key(packet: dict[str, Any]) -> tuple[int, int, int]:
    return (
        _optional_int(packet.get("ts_ms")) or 0,
        PACKET_KIND_ORDER.get(str(packet.get("packet_kind", "")).strip().lower(), 99),
        _optional_int(packet.get("target_stream_index")) or 0,
    )


def _pack_cstring(text: str, width: int = 16) -> bytes:
    raw = text.encode("ascii", "ignore")[:width]
    return raw + (b"\x00" * (width - len(raw)))


def _ip_text(value: bytes) -> str:
    return str(socket.inet_ntoa(value))


def _pad_mavlink_payload(payload: bytes, *, minimum_len: int, padded_len: int) -> bytes | None:
    if len(payload) < minimum_len:
        return None
    if len(payload) >= padded_len:
        return payload[:padded_len]
    return payload + (b"\x00" * (padded_len - len(payload)))


def iter_mavlink_frames(buffer: bytes) -> Iterable[tuple[str, bytes]]:
    offset = 0
    while offset < len(buffer):
        magic = buffer[offset]
        if magic == MAVLINK_V2_MAGIC:
            if offset + 10 > len(buffer):
                break
            payload_len = buffer[offset + 1]
            incompat_flags = buffer[offset + 2]
            signature_len = 13 if (incompat_flags & 0x01) else 0
            frame_len = 10 + payload_len + 2 + signature_len
            if offset + frame_len > len(buffer):
                break
            yield "2", buffer[offset : offset + frame_len]
            offset += frame_len
            continue
        if magic == MAVLINK_V1_MAGIC:
            if offset + 6 > len(buffer):
                break
            payload_len = buffer[offset + 1]
            frame_len = 6 + payload_len + 2
            if offset + frame_len > len(buffer):
                break
            yield "1", buffer[offset : offset + frame_len]
            offset += frame_len
            continue
        offset += 1


def _decode_heartbeat(payload: bytes) -> dict[str, Any]:
    padded = _pad_mavlink_payload(payload, minimum_len=8, padded_len=9)
    if padded is None:
        return {}
    custom_mode, vehicle_type, autopilot, base_mode, system_status, mavlink_version = struct.unpack(
        "<IBBBBB",
        padded[:9],
    )
    return {
        "custom_mode": custom_mode,
        "type": vehicle_type,
        "autopilot": autopilot,
        "base_mode": base_mode,
        "system_status": system_status,
        "mavlink_version": mavlink_version,
    }


def _decode_sys_status(payload: bytes) -> dict[str, Any]:
    padded = _pad_mavlink_payload(payload, minimum_len=30, padded_len=31)
    if padded is None:
        return {}
    values = struct.unpack("<IIIHHhHHHHHHb", padded[:31])
    return {
        "onboard_control_sensors_present": values[0],
        "onboard_control_sensors_enabled": values[1],
        "onboard_control_sensors_health": values[2],
        "load": values[3],
        "voltage_battery": values[4],
        "current_battery": values[5],
        "drop_rate_comm": values[6],
        "errors_comm": values[7],
        "errors_count1": values[8],
        "errors_count2": values[9],
        "errors_count3": values[10],
        "errors_count4": values[11],
        "battery_remaining": values[12],
    }


def _decode_param_request_read(payload: bytes) -> dict[str, Any]:
    padded = _pad_mavlink_payload(payload, minimum_len=4, padded_len=20)
    if padded is None:
        return {}
    param_index, target_system, target_component = struct.unpack("<hBB", padded[:4])
    return {
        "param_index": param_index,
        "target_system": target_system,
        "target_component": target_component,
        "param_id": _strip_null_terminated(padded[4:20]),
    }


def _decode_param_request_list(payload: bytes) -> dict[str, Any]:
    if len(payload) < 2:
        return {}
    target_system, target_component = struct.unpack("<BB", payload[:2])
    return {
        "target_system": target_system,
        "target_component": target_component,
    }


def _decode_param_value(payload: bytes) -> dict[str, Any]:
    if len(payload) < 25:
        return {}
    param_value, param_count, param_index = struct.unpack("<fHH", payload[:8])
    return {
        "param_value": param_value,
        "param_count": param_count,
        "param_index": param_index,
        "param_id": _strip_null_terminated(payload[8:24]),
        "param_type": payload[24],
    }


def _decode_param_set(payload: bytes) -> dict[str, Any]:
    if len(payload) < 23:
        return {}
    param_value = struct.unpack("<f", payload[:4])[0]
    target_system, target_component = struct.unpack("<BB", payload[4:6])
    return {
        "param_value": param_value,
        "target_system": target_system,
        "target_component": target_component,
        "param_id": _strip_null_terminated(payload[6:22]),
        "param_type": payload[22],
    }


def _decode_gps_raw_int(payload: bytes) -> dict[str, Any]:
    padded = _pad_mavlink_payload(payload, minimum_len=29, padded_len=30)
    if padded is None:
        return {}
    values = struct.unpack("<QiiiHHHHBB", padded[:30])
    return {
        "time_usec": values[0],
        "lat": values[1],
        "lon": values[2],
        "alt": values[3],
        "eph": values[4],
        "epv": values[5],
        "vel": values[6],
        "cog": values[7],
        "fix_type": values[8],
        "satellites_visible": values[9],
    }


def _decode_mission_request_list(payload: bytes) -> dict[str, Any]:
    if len(payload) < 2:
        return {}
    target_system, target_component = struct.unpack("<BB", payload[:2])
    result = {
        "target_system": target_system,
        "target_component": target_component,
    }
    if len(payload) >= 3:
        result["mission_type"] = payload[2]
    return result


def _decode_mission_count(payload: bytes) -> dict[str, Any]:
    if len(payload) < 4:
        return {}
    count, target_system, target_component = struct.unpack("<HBB", payload[:4])
    result = {
        "count": count,
        "target_system": target_system,
        "target_component": target_component,
    }
    if len(payload) >= 5:
        result["mission_type"] = payload[4]
    return result


def _decode_mission_clear_all(payload: bytes) -> dict[str, Any]:
    if len(payload) < 2:
        return {}
    target_system, target_component = struct.unpack("<BB", payload[:2])
    result = {
        "target_system": target_system,
        "target_component": target_component,
    }
    if len(payload) >= 3:
        result["mission_type"] = payload[2]
    return result


def _decode_mission_ack(payload: bytes) -> dict[str, Any]:
    if len(payload) < 3:
        return {}
    target_system, target_component, ack_type = struct.unpack("<BBB", payload[:3])
    result = {
        "target_system": target_system,
        "target_component": target_component,
        "type": ack_type,
    }
    if len(payload) >= 4:
        result["mission_type"] = payload[3]
    return result


def _decode_command_long(payload: bytes) -> dict[str, Any]:
    padded = _pad_mavlink_payload(payload, minimum_len=30, padded_len=33)
    if padded is None:
        return {}
    values = struct.unpack("<7fHBBB", padded[:33])
    return {
        "param1": values[0],
        "param2": values[1],
        "param3": values[2],
        "param4": values[3],
        "param5": values[4],
        "param6": values[5],
        "param7": values[6],
        "command": values[7],
        "target_system": values[8],
        "target_component": values[9],
        "confirmation": values[10],
    }


def _decode_command_ack(payload: bytes) -> dict[str, Any]:
    if len(payload) < 3:
        return {}
    command, result = struct.unpack("<HB", payload[:3])
    decoded = {
        "command": command,
        "result": result,
    }
    if len(payload) >= 4:
        decoded["progress"] = payload[3]
    if len(payload) >= 8:
        decoded["result_param2"] = struct.unpack("<i", payload[4:8])[0]
    if len(payload) >= 10:
        decoded["target_system"] = payload[8]
        decoded["target_component"] = payload[9]
    return decoded


def _decode_power_status(payload: bytes) -> dict[str, Any]:
    padded = _pad_mavlink_payload(payload, minimum_len=4, padded_len=6)
    if padded is None:
        return {}
    vcc, vservo, flags = struct.unpack("<HHH", padded[:6])
    return {
        "Vcc": vcc,
        "Vservo": vservo,
        "flags": flags,
    }


def _decode_battery_status(payload: bytes) -> dict[str, Any]:
    padded = _pad_mavlink_payload(payload, minimum_len=34, padded_len=38)
    if padded is None:
        return {}
    current_consumed, energy_consumed, temperature = struct.unpack("<ihh", padded[:8])
    voltages = list(struct.unpack("<10H", padded[8:28]))
    current_battery = struct.unpack("<h", padded[28:30])[0]
    battery_id = padded[30]
    function = padded[31]
    battery_type = padded[32]
    temperature_ext = padded[33]
    fault_bitmask = struct.unpack("<I", padded[34:38])[0]
    return {
        "current_consumed": current_consumed,
        "energy_consumed": energy_consumed,
        "temperature": temperature,
        "voltages": voltages,
        "current_battery": current_battery,
        "battery_id": battery_id,
        "battery_function": function,
        "battery_type": battery_type,
        "battery_remaining": temperature_ext if temperature_ext <= 100 else None,
        "fault_bitmask": fault_bitmask,
    }


def _decode_autopilot_version(payload: bytes) -> dict[str, Any]:
    padded = _pad_mavlink_payload(payload, minimum_len=28, padded_len=36)
    if padded is None:
        return {}
    values = struct.unpack("<QIIIIHHQ", padded[:36])
    return {
        "capabilities": values[0],
        "flight_sw_version": values[1],
        "middleware_sw_version": values[2],
        "os_sw_version": values[3],
        "board_version": values[4],
        "vendor_id": values[5],
        "product_id": values[6],
        "uid": values[7],
    }


def _decode_statustext(payload: bytes) -> dict[str, Any]:
    if len(payload) < 51:
        return {}
    severity = payload[0]
    text = payload[1:51].split(b"\x00", 1)[0].decode("utf-8", "ignore")
    decoded = {
        "severity": severity,
        "text": text,
    }
    if len(payload) >= 54:
        decoded["id"] = struct.unpack("<H", payload[51:53])[0]
        decoded["chunk_seq"] = payload[53]
    return decoded


def decode_payload(message_id: int, payload: bytes) -> dict[str, Any]:
    if message_id == 0:
        return _decode_heartbeat(payload)
    if message_id == 1:
        return _decode_sys_status(payload)
    if message_id == 20:
        return _decode_param_request_read(payload)
    if message_id == 21:
        return _decode_param_request_list(payload)
    if message_id == 22:
        return _decode_param_value(payload)
    if message_id == 23:
        return _decode_param_set(payload)
    if message_id == 24:
        return _decode_gps_raw_int(payload)
    if message_id == 43:
        return _decode_mission_request_list(payload)
    if message_id == 44:
        return _decode_mission_count(payload)
    if message_id == 45:
        return _decode_mission_clear_all(payload)
    if message_id == 47:
        return _decode_mission_ack(payload)
    if message_id == 76:
        return _decode_command_long(payload)
    if message_id == 77:
        return _decode_command_ack(payload)
    if message_id == 125:
        return _decode_power_status(payload)
    if message_id == 147:
        return _decode_battery_status(payload)
    if message_id == 148:
        return _decode_autopilot_version(payload)
    if message_id == 253:
        return _decode_statustext(payload)
    return {}


def decode_mavlink_frame(
    *,
    ts_ms: int,
    frame: bytes,
    protocol_version: str,
    transport_family: str,
    src_ip: str,
    dst_ip: str,
    src_port: int,
    dst_port: int,
    connection_key: tuple[str, int, str, int] | None,
) -> WireMessage:
    if protocol_version == "2":
        payload_len = frame[1]
        source_system_id = frame[5]
        source_component_id = frame[6]
        message_id = frame[7] | (frame[8] << 8) | (frame[9] << 16)
        payload = frame[10 : 10 + payload_len]
    else:
        payload_len = frame[1]
        source_system_id = frame[3]
        source_component_id = frame[4]
        message_id = frame[5]
        payload = frame[6 : 6 + payload_len]
    return WireMessage(
        ts_ms=ts_ms,
        transport_family=transport_family,
        src_ip=src_ip,
        dst_ip=dst_ip,
        src_port=src_port,
        dst_port=dst_port,
        source_system_id=source_system_id,
        source_component_id=source_component_id,
        protocol_version=protocol_version,
        message_id=message_id,
        message_name=MAVLINK_MESSAGE_NAMES.get(message_id, f"MSG_{message_id}"),
        payload=decode_payload(message_id, payload),
        frame_len=len(frame),
        connection_key=connection_key,
    )


def parse_pcap_capture(path: Path) -> list[PcapPacket]:
    packets: list[PcapPacket] = []
    if not path.exists():
        raise ReconstructionError(f"Missing MAVLink capture at {path}")
    with path.open("rb") as handle:
        reader = dpkt.pcap.Reader(handle)
        for ts, frame in reader:
            try:
                eth = dpkt.ethernet.Ethernet(frame)
                if not isinstance(eth.data, dpkt.ip.IP):
                    continue
                ip = eth.data
            except (dpkt.UnpackError, ValueError):
                continue
            src_ip = _ip_text(ip.src)
            dst_ip = _ip_text(ip.dst)
            ts_ms = int(ts * 1000)
            if ip.p == dpkt.ip.IP_PROTO_TCP and isinstance(ip.data, dpkt.tcp.TCP):
                tcp = ip.data
                if (tcp.sport != TARGET_TCP_PORT and tcp.dport != TARGET_TCP_PORT) or not tcp.data:
                    continue
                packets.append(
                    PcapPacket(
                        ts_ms=ts_ms,
                        transport_family="tcp",
                        src_ip=src_ip,
                        dst_ip=dst_ip,
                        src_port=int(tcp.sport),
                        dst_port=int(tcp.dport),
                        payload=bytes(tcp.data),
                        tcp_flags=int(tcp.flags),
                    )
                )
            elif ip.p == dpkt.ip.IP_PROTO_UDP and isinstance(ip.data, dpkt.udp.UDP):
                udp = ip.data
                if (udp.sport != TARGET_UDP_PORT and udp.dport != TARGET_UDP_PORT) or not udp.data:
                    continue
                packets.append(
                    PcapPacket(
                        ts_ms=ts_ms,
                        transport_family="udp",
                        src_ip=src_ip,
                        dst_ip=dst_ip,
                        src_port=int(udp.sport),
                        dst_port=int(udp.dport),
                        payload=bytes(udp.data),
                    )
                )
    return packets


def decode_wire_messages(pcap_packets: list[PcapPacket]) -> list[WireMessage]:
    tcp_buffers: dict[tuple[str, int, str, int, str], bytearray] = {}
    messages: list[WireMessage] = []
    for packet in pcap_packets:
        if packet.transport_family == "udp":
            for protocol_version, frame in iter_mavlink_frames(packet.payload):
                messages.append(
                    decode_mavlink_frame(
                        ts_ms=packet.ts_ms,
                        frame=frame,
                        protocol_version=protocol_version,
                        transport_family="udp",
                        src_ip=packet.src_ip,
                        dst_ip=packet.dst_ip,
                        src_port=packet.src_port,
                        dst_port=packet.dst_port,
                        connection_key=None,
                    )
                )
            continue

        if packet.dst_port == TARGET_TCP_PORT:
            client_ip, client_port, server_ip, server_port = packet.src_ip, packet.src_port, packet.dst_ip, packet.dst_port
            direction = "c2s"
        else:
            client_ip, client_port, server_ip, server_port = packet.dst_ip, packet.dst_port, packet.src_ip, packet.src_port
            direction = "s2c"
        connection_key = (client_ip, client_port, server_ip, server_port)
        buffer_key = (*connection_key, direction)
        buffer = tcp_buffers.setdefault(buffer_key, bytearray())
        if buffer and packet.payload and packet.payload[0] in {MAVLINK_V1_MAGIC, MAVLINK_V2_MAGIC}:
            # A fresh TCP segment that already starts at a MAVLink frame boundary should
            # take precedence over stale trailing bytes from the previous segment. Keeping
            # the stale tail can misalign the next decode and hide real responses.
            buffer.clear()
        buffer.extend(packet.payload)
        consumed = 0
        for protocol_version, frame in iter_mavlink_frames(bytes(buffer)):
            consumed += len(frame)
            messages.append(
                decode_mavlink_frame(
                    ts_ms=packet.ts_ms,
                    frame=frame,
                    protocol_version=protocol_version,
                    transport_family="tcp",
                    src_ip=packet.src_ip,
                    dst_ip=packet.dst_ip,
                    src_port=packet.src_port,
                    dst_port=packet.dst_port,
                    connection_key=connection_key,
                )
            )
        if consumed:
            del buffer[:consumed]
    messages.sort(key=lambda item: (item.ts_ms, item.frame_len, item.message_name))
    return messages


def request_message_name_for_row(row: Mapping[str, Any]) -> str:
    return REQUEST_MESSAGE_BY_SEND_KIND[command_spec_for_name(str(row.get("command", ""))).send_kind]


def response_message_name_for_row(row: Mapping[str, Any]) -> str:
    return _optional_text(row.get("response_name")) or command_spec_for_name(str(row.get("command", ""))).response_name


def request_matches_row(row: Mapping[str, Any], message: WireMessage) -> bool:
    if message.message_name != request_message_name_for_row(row):
        return False
    if message.transport_family != "tcp":
        return False
    if _optional_text(row.get("source_ip")) and _optional_text(row.get("source_ip")) != message.src_ip:
        return False
    if _optional_text(row.get("target_ip")) and _optional_text(row.get("target_ip")) != message.dst_ip:
        return False
    expected_system_id = _optional_int(row.get("source_system_id"))
    if expected_system_id is not None and expected_system_id > 0 and message.source_system_id != expected_system_id:
        return False
    expected_component_id = _optional_int(row.get("source_component_id"))
    if expected_component_id is not None and expected_component_id > 0 and message.source_component_id != expected_component_id:
        return False
    arguments = _json_object(row.get("arguments_json"))
    command_name = str(row.get("command", ""))
    payload = message.payload
    if message.message_name == "COMMAND_LONG":
        if _optional_int(payload.get("command")) != numeric_command_id(command_name):
            return False
        if command_name == "REQUEST_AUTOPILOT_CAPABILITIES":
            return abs((_optional_number(payload.get("param1")) or 0.0) - float(arguments.get("param1", 1.0))) < 1e-6
        if command_name == "MAV_CMD_DO_SET_MODE":
            return (
                abs((_optional_number(payload.get("param1")) or 0.0) - float(arguments.get("base_mode", 0.0))) < 1e-6
                and abs((_optional_number(payload.get("param2")) or 0.0) - float(arguments.get("custom_mode", 0.0))) < 1e-6
            )
        if command_name == "MAV_CMD_COMPONENT_ARM_DISARM":
            return abs((_optional_number(payload.get("param1")) or 0.0) - float(arguments.get("param1", 0.0))) < 1e-6
        return True
    if message.message_name == "PARAM_REQUEST_READ":
        return _optional_text(payload.get("param_id")) == _optional_text(arguments.get("param_id"))
    if message.message_name == "PARAM_SET":
        return _optional_text(payload.get("param_id")) == _optional_text(arguments.get("param_id"))
    if message.message_name == "MISSION_REQUEST_LIST":
        mission_type = _optional_int(arguments.get("mission_type", 0))
        actual_type = _optional_int(payload.get("mission_type"))
        return actual_type is None or actual_type == mission_type
    if message.message_name == "MISSION_CLEAR_ALL":
        mission_type = _optional_int(arguments.get("mission_type", 0))
        actual_type = _optional_int(payload.get("mission_type"))
        return actual_type is None or actual_type == mission_type
    return True


def response_matches_row(row: Mapping[str, Any], message: WireMessage) -> bool:
    expected = response_message_name_for_row(row)
    if message.message_name != expected:
        return False
    arguments = _json_object(row.get("arguments_json"))
    if message.message_name == "COMMAND_ACK":
        return _optional_int(message.payload.get("command")) == numeric_command_id(str(row.get("command", "")))
    if message.message_name == "PARAM_VALUE":
        expected_param = _optional_text(arguments.get("param_id"))
        actual_param = _optional_text(message.payload.get("param_id"))
        return expected_param is None or expected_param == actual_param
    if message.message_name == "MISSION_ACK":
        mission_type = _optional_int(arguments.get("mission_type"))
        actual_type = _optional_int(message.payload.get("mission_type"))
        return mission_type is None or actual_type is None or mission_type == actual_type
    return True


def find_best_request_candidate(row: Mapping[str, Any], messages: list[WireMessage], used_indexes: set[int]) -> int | None:
    send_start_ms = _optional_int(row.get("send_start_ms")) or _optional_int(row.get("real_ms")) or 0
    send_end_ms = _optional_int(row.get("send_end_ms")) or send_start_ms
    candidates: list[tuple[int, int]] = []
    for index, message in enumerate(messages):
        if index in used_indexes:
            continue
        if not request_matches_row(row, message):
            continue
        if message.ts_ms < send_start_ms - REQUEST_WINDOW_EARLY_MS:
            continue
        if message.ts_ms > send_end_ms + REQUEST_WINDOW_LATE_MS:
            continue
        score = abs(message.ts_ms - send_start_ms)
        candidates.append((score, index))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def find_best_response_candidate(
    row: Mapping[str, Any],
    messages: list[WireMessage],
    used_indexes: set[int],
    *,
    request_message: WireMessage | None,
) -> int | None:
    send_start_ms = _optional_int(row.get("send_start_ms")) or _optional_int(row.get("real_ms")) or 0
    send_end_ms = _optional_int(row.get("send_end_ms")) or send_start_ms
    lower_bound = request_message.ts_ms if request_message is not None else send_start_ms
    if response_message_name_for_row(row) == "PARAM_VALUE":
        # PARAM_REQUEST_LIST can overlap with an already-draining PARAM_VALUE stream on the
        # same TCP connection, so accept a tiny pre-request tolerance instead of treating
        # those responses as missing entirely.
        lower_bound -= STREAMING_RESPONSE_WINDOW_EARLY_MS
    candidates: list[tuple[int, int]] = []
    for index, message in enumerate(messages):
        if index in used_indexes:
            continue
        if not response_matches_row(row, message):
            continue
        if message.ts_ms < lower_bound:
            continue
        if message.ts_ms > send_end_ms + RESPONSE_WINDOW_LATE_MS:
            continue
        if request_message is not None and request_message.connection_key is not None:
            if message.connection_key != request_message.connection_key:
                continue
        score = abs(message.ts_ms - send_end_ms)
        candidates.append((score, index))
    if not candidates:
        return None
    candidates.sort()
    return candidates[0][1]


def response_outcome(
    row: Mapping[str, Any],
    response_message: WireMessage | None,
) -> tuple[int, int, int | None, str]:
    timeout = _bool_flag(_optional_int(row.get("timeout")) == 1)
    if response_message is None:
        if timeout:
            return 1, 0, _optional_int(row.get("response_code")), "timeout"
        if _optional_text(row.get("send_exception")):
            return 0, 0, _optional_int(row.get("response_code")), "send_exception"
        return 0, 0, _optional_int(row.get("response_code")), "missing_wire_response"
    if response_message.message_name == "COMMAND_ACK":
        code = _optional_int(response_message.payload.get("result"))
        success = int(code == MAV_RESULT_ACCEPTED)
        return 1, success, code, "completed" if success else "command_ack_reject"
    if response_message.message_name == "MISSION_ACK":
        code = _optional_int(response_message.payload.get("type"))
        success = int(code == MISSION_ACK_ACCEPTED)
        return 1, success, code, "completed" if success else "mission_ack_reject"
    if response_message.message_name in {"PARAM_VALUE", "MISSION_COUNT", "AUTOPILOT_VERSION"}:
        return 1, 1, _optional_int(row.get("response_code")), "completed"
    return 1, 0, _optional_int(row.get("response_code")), response_message.message_name.lower()


def row_label_fields(row: Mapping[str, Any]) -> tuple[int | None, str | None, str | None, str | None, str | None, float | None]:
    meta = row_meta(dict(row))
    label = _optional_int(meta.get("class_label"))
    if label is None:
        label = _optional_int(meta.get("label"))
    label_name = _optional_text(meta.get("class_name"))
    attack_family = _optional_text(meta.get("attack_family"))
    phase = _optional_text(meta.get("phase"))
    actor_role = _optional_text(meta.get("actor_role"))
    actor_trust = _optional_number(meta.get("actor_trust"))
    return label, label_name, attack_family, phase, actor_role, actor_trust


def session_id_for_row(row: Mapping[str, Any]) -> str:
    return (
        f"{_optional_text(row.get('source_service')) or 'unknown'}->"
        f"{_optional_text(row.get('target_service')) or 'unknown'}@"
        f"{_optional_text(row.get('target_endpoint')) or 'unknown'}"
    )


def transaction_id_for_row(row: Mapping[str, Any]) -> str:
    send_id = _optional_text(row.get("send_id"))
    if send_id is not None:
        return send_id
    return session_id_for_row(row)


def telemetry_record_to_packet(record: TelemetryRecord, *, run_id: int | None = None) -> dict[str, Any]:
    return {
        "ts_ms": record.ts_ms,
        "packet_kind": "telemetry",
        "src": VEHICLE_SERVICE,
        "dst": "mavlink_gcs",
        "target_service": VEHICLE_SERVICE,
        "service": record.message_name,
        "command": None,
        "args": None,
        "payload": dict(record.payload),
        "node_service": record.target_logical_id,
        "run_id": run_id,
        "episode_id": None,
        "session_id": None,
        "txn_id": None,
        "send_id": None,
        "target_stream_id": None,
        "target_stream_index": None,
        "attack_family": None,
        "phase": None,
        "actor": None,
        "actor_role": None,
        "actor_trust": None,
        "src_ip": None,
        "dst_ip": None,
        "src_port": None,
        "dst_port": TARGET_UDP_PORT,
        "bytes_on_wire": record.bytes_on_wire,
        "observed_on_wire": 1,
        "ts_source": record.ts_source,
        "bytes_source": record.bytes_source,
        "protocol_version": None,
        "message_name": record.message_name,
    }


def build_request_packet(row: Mapping[str, Any], request_message: WireMessage | None) -> dict[str, Any]:
    label, label_name, attack_family, phase, actor_role, actor_trust = row_label_fields(row)
    arguments = _json_object(row.get("arguments_json"))
    return {
        "ts_ms": request_message.ts_ms if request_message is not None else (_optional_int(row.get("send_start_ms")) or 0),
        "packet_kind": "request",
        "src": _optional_text(row.get("source_service")) or "unknown",
        "dst": _optional_text(row.get("target_service")) or "unknown",
        "target_service": _optional_text(row.get("target_service")) or "unknown",
        "service": "mavlink",
        "command": _optional_text(row.get("command")),
        "args": arguments,
        "payload": {} if request_message is None else dict(request_message.payload),
        "run_id": _optional_int(row.get("run_id")),
        "episode_id": _optional_int(row_meta(dict(row)).get("episode_id")),
        "label": label,
        "episode_label": label,
        "episode_kind": label_name,
        "session_id": session_id_for_row(row),
        "txn_id": transaction_id_for_row(row),
        "send_id": _optional_text(row.get("send_id")),
        "target_stream_id": _optional_text(row.get("target_stream_id")),
        "target_stream_index": _optional_int(row.get("target_stream_index")),
        "attack_family": attack_family,
        "phase": phase,
        "actor": _optional_text(row.get("source_service")),
        "actor_role": actor_role,
        "actor_trust": actor_trust,
        "src_ip": _optional_text(row.get("source_ip")),
        "dst_ip": _optional_text(row.get("target_ip")),
        "src_port": None if request_message is None else request_message.src_port,
        "dst_port": TARGET_TCP_PORT,
        "bytes_on_wire": None if request_message is None else request_message.frame_len,
        "observed_on_wire": _bool_flag(request_message is not None),
        "ts_source": "pcap" if request_message is not None else "send_log",
        "bytes_source": "pcap" if request_message is not None else None,
        "protocol_version": None if request_message is None else request_message.protocol_version,
        "message_name": request_message.message_name if request_message is not None else request_message_name_for_row(row),
    }


def build_final_packet(
    row: Mapping[str, Any],
    response_message: WireMessage | None,
    *,
    response_code: int | None,
    reason: str,
    sat_success: int,
) -> dict[str, Any]:
    label, label_name, attack_family, phase, actor_role, actor_trust = row_label_fields(row)
    response_payload = _json_object(row.get("response_json"))
    if response_message is not None:
        response_payload = dict(response_message.payload)
    return {
        "ts_ms": response_message.ts_ms if response_message is not None else (_optional_int(row.get("send_end_ms")) or 0),
        "packet_kind": "final",
        "src": _optional_text(row.get("target_service")) or "unknown",
        "dst": _optional_text(row.get("source_service")) or "unknown",
        "target_service": _optional_text(row.get("target_service")) or "unknown",
        "service": "mavlink",
        "command": _optional_text(row.get("command")),
        "args": _json_object(row.get("arguments_json")),
        "payload": response_payload,
        "run_id": _optional_int(row.get("run_id")),
        "episode_id": _optional_int(row_meta(dict(row)).get("episode_id")),
        "label": label,
        "episode_label": label,
        "episode_kind": label_name,
        "session_id": session_id_for_row(row),
        "txn_id": transaction_id_for_row(row),
        "send_id": _optional_text(row.get("send_id")),
        "target_stream_id": _optional_text(row.get("target_stream_id")),
        "target_stream_index": _optional_int(row.get("target_stream_index")),
        "attack_family": attack_family,
        "phase": phase,
        "actor": _optional_text(row.get("source_service")),
        "actor_role": actor_role,
        "actor_trust": actor_trust,
        "src_ip": _optional_text(row.get("target_ip")),
        "dst_ip": _optional_text(row.get("source_ip")),
        "src_port": None if response_message is None else response_message.src_port,
        "dst_port": None if response_message is None else response_message.dst_port,
        "bytes_on_wire": None if response_message is None else response_message.frame_len,
        "observed_on_wire": _bool_flag(response_message is not None),
        "ts_source": "pcap" if response_message is not None else "send_log",
        "bytes_source": "pcap" if response_message is not None else None,
        "protocol_version": None if response_message is None else response_message.protocol_version,
        "message_name": _optional_text(row.get("response_name")) if response_message is None else response_message.message_name,
        "gds_accept": 1 if _optional_text(row.get("send_exception")) in (None, "") else 0,
        "sat_success": sat_success,
        "timeout": _bool_flag(_optional_int(row.get("timeout")) == 1),
        "response_code": response_code,
        "reason": reason,
        "response_direction_seen": _bool_flag(response_message is not None),
        "final_observed_on_wire": _bool_flag(response_message is not None),
        "txn_warning_events": 0,
        "txn_error_events": 0,
    }


def build_transaction_record(
    row: Mapping[str, Any],
    request_packet: Mapping[str, Any],
    final_packet: Mapping[str, Any],
    *,
    request_message: WireMessage | None,
    response_message: WireMessage | None,
    telemetry_records: list[TelemetryRecord],
    telemetry_recent: bool,
    state_snapshot: dict[str, Any] | None,
) -> dict[str, Any]:
    label, label_name, attack_family, phase, actor_role, actor_trust = row_label_fields(row)
    latency_ms = _optional_number(row.get("latency_ms"))
    if latency_ms is None:
        request_ts = _optional_number(request_packet.get("ts_ms"))
        final_ts = _optional_number(final_packet.get("ts_ms"))
        if request_ts is not None and final_ts is not None:
            latency_ms = max(0.0, final_ts - request_ts)
    return {
        "request_ts_ms": _optional_number(request_packet.get("ts_ms")),
        "final_ts_ms": _optional_number(final_packet.get("ts_ms")),
        "latency_ms": latency_ms,
        "target_service": _optional_text(row.get("target_service")) or "unknown",
        "target_stream_id": _optional_text(row.get("target_stream_id")),
        "target_stream_index": _optional_int(row.get("target_stream_index")),
        "service": "mavlink",
        "command": _optional_text(row.get("command")),
        "args": _json_object(row.get("arguments_json")),
        "run_id": _optional_int(row.get("run_id")),
        "episode_id": _optional_int(row_meta(dict(row)).get("episode_id")),
        "label": label,
        "label_name": label_name,
        "episode_label": label,
        "episode_kind": label_name,
        "session_id": session_id_for_row(row),
        "txn_id": transaction_id_for_row(row),
        "send_id": _optional_text(row.get("send_id")),
        "attack_family": attack_family,
        "phase": phase,
        "actor": _optional_text(row.get("source_service")),
        "actor_role": actor_role,
        "actor_trust": actor_trust,
        "source_service": _optional_text(row.get("source_service")),
        "source_ip": _optional_text(row.get("source_ip")),
        "source_system_id": _optional_int(row.get("source_system_id")),
        "source_component_id": _optional_int(row.get("source_component_id")),
        "target_ip": _optional_text(row.get("target_ip")),
        "target_endpoint": _optional_text(row.get("target_endpoint")),
        "target_system_id": _optional_int(row.get("target_system_id")),
        "target_component_id": _optional_int(row.get("target_component_id")),
        "req_bytes": _optional_number(request_packet.get("bytes_on_wire")),
        "resp_bytes": _optional_number(final_packet.get("bytes_on_wire")),
        "gds_accept": _optional_int(final_packet.get("gds_accept")),
        "sat_success": _optional_int(final_packet.get("sat_success")),
        "timeout": _optional_int(final_packet.get("timeout")),
        "response_code": _optional_int(final_packet.get("response_code")),
        "reason": _optional_text(final_packet.get("reason")),
        "response_direction_seen": _optional_int(final_packet.get("response_direction_seen")),
        "final_observed_on_wire": _optional_int(final_packet.get("final_observed_on_wire")),
        "txn_warning_events": 0,
        "txn_error_events": 0,
        "response_name": final_packet.get("message_name"),
        "response_payload": dict(final_packet.get("payload", {}) or {}),
        "request_payload": dict(request_packet.get("payload", {}) or {}),
        "telemetry_recent": _bool_flag(telemetry_recent),
        "telemetry_message_count": len(telemetry_records),
        "native_state_snapshot": state_snapshot,
        "request_message_protocol_version": None if request_message is None else request_message.protocol_version,
        "response_message_protocol_version": None if response_message is None else response_message.protocol_version,
    }


def build_raw_packet_record(
    packet: Mapping[str, Any],
    *,
    source_artifact_paths: Iterable[str],
    capture_backend: str | None,
    capture_interface: str | None,
) -> dict[str, Any]:
    outcome = None
    if str(packet.get("packet_kind", "")).strip().lower() == "final":
        outcome = {
            "accepted": bool(_optional_int(packet.get("gds_accept"))) if packet.get("gds_accept") not in (None, "") else None,
            "executed_successfully": bool(_optional_int(packet.get("sat_success"))) if packet.get("sat_success") not in (None, "") else None,
            "timed_out": bool(_optional_int(packet.get("timeout"))) if packet.get("timeout") not in (None, "") else None,
            "raw_code": _optional_number(packet.get("response_code")),
            "raw_reason": _optional_text(packet.get("reason")),
            "warning_count": _optional_number(packet.get("txn_warning_events")),
            "error_count": _optional_number(packet.get("txn_error_events")),
            "response_direction_seen": bool(_optional_int(packet.get("response_direction_seen")))
            if packet.get("response_direction_seen") not in (None, "")
            else None,
            "terminal_observed_on_wire": bool(_optional_int(packet.get("final_observed_on_wire")))
            if packet.get("final_observed_on_wire") not in (None, "")
            else None,
            "raw_event_name": _optional_text(packet.get("message_name")),
        }
    packet_kind = str(packet.get("packet_kind", "")).strip().lower()
    if packet_kind == "telemetry":
        message_family = "telemetry"
        message_stage = "telemetry"
    elif packet_kind == "request":
        message_family = "request"
        message_stage = "request"
    else:
        message_family = "response"
        message_stage = "terminal_response"
    return {
        "schema_version": RAW_PACKET_SCHEMA_VERSION,
        "record_kind": "raw_packet",
        "protocol_family": "mavlink",
        "protocol_version": _optional_text(packet.get("protocol_version")),
        "platform_family": "air_vehicle",
        "message_family": message_family,
        "message_stage": message_stage,
        "sender": {
            "logical_id": _optional_text(packet.get("src")) or "unknown",
            "role": _optional_text(packet.get("actor_role")),
            "trust_score": _optional_number(packet.get("actor_trust")),
            "network_endpoint": {
                "host": _optional_text(packet.get("src")),
                "ip": _optional_text(packet.get("src_ip")),
                "port": _optional_int(packet.get("src_port")),
                "transport_family": _optional_text("udp" if packet_kind == "telemetry" else "tcp"),
            },
        },
        "target": {
            "logical_id": _optional_text(packet.get("target_service")) or _optional_text(packet.get("dst")) or "unknown",
            "role": None,
            "stream_id": _optional_text(packet.get("target_stream_id")),
            "stream_index": _optional_int(packet.get("target_stream_index")),
            "network_endpoint": {
                "host": _optional_text(packet.get("dst")),
                "ip": _optional_text(packet.get("dst_ip")),
                "port": _optional_int(packet.get("dst_port")),
                "transport_family": _optional_text("udp" if packet_kind == "telemetry" else "tcp"),
            },
        },
        "command": None
        if packet_kind == "telemetry"
        else {
            "raw_name": _optional_text(packet.get("command")),
            "raw_identifier": {
                "service_name": _optional_text(packet.get("service")),
                "native_service_id": None,
                "native_command_id": None,
            },
            "raw_arguments": packet.get("args"),
            "raw_argument_representation": packet.get("args"),
        },
        "timing": {
            "observed_at_ms": _optional_number(packet.get("ts_ms")) or 0.0,
            "timestamp_source": _optional_text(packet.get("ts_source")),
        },
        "transport": {
            "transport_family": "udp" if packet_kind == "telemetry" else "tcp",
            "bytes_on_wire": _optional_number(packet.get("bytes_on_wire")),
            "bytes_source": _optional_text(packet.get("bytes_source")),
            "src_ip": _optional_text(packet.get("src_ip")),
            "src_port": _optional_int(packet.get("src_port")),
            "dst_ip": _optional_text(packet.get("dst_ip")),
            "dst_port": _optional_int(packet.get("dst_port")),
        },
        "provenance": {
            "observed_on_wire": bool(_optional_int(packet.get("observed_on_wire"))),
            "capture_backend": _optional_text(capture_backend),
            "capture_interface": _optional_text(capture_interface),
            "timestamp_source": _optional_text(packet.get("ts_source")),
            "bytes_source": _optional_text(packet.get("bytes_source")),
            "source_artifact_paths": list(source_artifact_paths),
        },
        "outcome": outcome,
        "correlation": {
            "run_id": _optional_int(packet.get("run_id")),
            "episode_id": _optional_int(packet.get("episode_id")),
            "session_id": _optional_text(packet.get("session_id")),
            "transaction_id": _optional_text(packet.get("txn_id")),
            "send_id": _optional_text(packet.get("send_id")),
            "stream_id": _optional_text(packet.get("target_stream_id")),
            "stream_index": _optional_int(packet.get("target_stream_index")),
        },
        "evaluation_context": {
            "label": _optional_int(packet.get("label")),
            "label_name": _optional_text(packet.get("episode_kind")),
            "attack_family": _optional_text(packet.get("attack_family")),
            "phase": _optional_text(packet.get("phase")),
            "actor_id": _optional_text(packet.get("actor")),
            "actor_role": _optional_text(packet.get("actor_role")),
            "actor_trust": _optional_number(packet.get("actor_trust")),
        },
        "native_payload": packet.get("payload"),
        "native_fields": {"legacy_record": dict(packet)},
    }


def build_raw_transaction_record(
    transaction: Mapping[str, Any],
    *,
    related_packets: list[dict[str, Any]],
    source_artifact_paths: Iterable[str],
    capture_backend: str | None,
    capture_interface: str | None,
) -> dict[str, Any]:
    request_packet = next((packet for packet in related_packets if packet.get("packet_kind") == "request"), None)
    timestamp_sources = []
    byte_sources = []
    observed_message_families = []
    observed_message_stages = []
    for packet in related_packets:
        ts_source = _optional_text(packet.get("ts_source"))
        if ts_source and ts_source not in timestamp_sources:
            timestamp_sources.append(ts_source)
        bytes_source = _optional_text(packet.get("bytes_source"))
        if bytes_source and bytes_source not in byte_sources:
            byte_sources.append(bytes_source)
        packet_kind = str(packet.get("packet_kind", "")).strip().lower()
        if packet_kind == "telemetry":
            family, stage = "telemetry", "telemetry"
        elif packet_kind == "request":
            family, stage = "request", "request"
        else:
            family, stage = "response", "terminal_response"
        if family not in observed_message_families:
            observed_message_families.append(family)
        if stage not in observed_message_stages:
            observed_message_stages.append(stage)
    return {
        "schema_version": RAW_TRANSACTION_SCHEMA_VERSION,
        "record_kind": "raw_transaction",
        "protocol_family": "mavlink",
        "protocol_version": _optional_text(transaction.get("request_message_protocol_version"))
        or _optional_text(transaction.get("response_message_protocol_version")),
        "platform_family": "air_vehicle",
        "sender": {
            "logical_id": _optional_text(transaction.get("actor")) or "unknown",
            "role": _optional_text(transaction.get("actor_role")),
            "trust_score": _optional_number(transaction.get("actor_trust")),
            "network_endpoint": {
                "host": _optional_text(transaction.get("source_service")),
                "ip": _optional_text(transaction.get("source_ip")),
                "port": None if request_packet is None else _optional_int(request_packet.get("src_port")),
                "transport_family": "tcp",
            },
        },
        "target": {
            "logical_id": _optional_text(transaction.get("target_service")) or "unknown",
            "role": None,
            "stream_id": _optional_text(transaction.get("target_stream_id")),
            "stream_index": _optional_int(transaction.get("target_stream_index")),
            "network_endpoint": {
                "host": _optional_text(transaction.get("target_service")),
                "ip": _optional_text(transaction.get("target_ip")),
                "port": TARGET_TCP_PORT,
                "transport_family": "tcp",
            },
        },
        "command": {
            "raw_name": _optional_text(transaction.get("command")),
            "raw_identifier": {
                "service_name": _optional_text(transaction.get("service")),
                "native_service_id": None,
                "native_command_id": None,
            },
            "raw_arguments": transaction.get("args"),
            "raw_argument_representation": transaction.get("args"),
        },
        "timing": {
            "submitted_at_ms": _optional_number(transaction.get("request_ts_ms")),
            "request_forwarded_at_ms": _optional_number(transaction.get("request_ts_ms")),
            "protocol_response_at_ms": _optional_number(transaction.get("final_ts_ms")),
            "finalized_at_ms": _optional_number(transaction.get("final_ts_ms")),
            "latency_ms": _optional_number(transaction.get("latency_ms")),
        },
        "transport": {
            "transport_family": "tcp",
            "request_bytes_on_wire": _optional_number(transaction.get("req_bytes")),
            "response_bytes_on_wire": _optional_number(transaction.get("resp_bytes")),
        },
        "provenance": {
            "observed_on_wire": bool(_optional_int(transaction.get("final_observed_on_wire")))
            or bool(_optional_number(transaction.get("req_bytes"))),
            "capture_backend": _optional_text(capture_backend),
            "capture_interface": _optional_text(capture_interface),
            "timestamp_source": "pcap",
            "bytes_source": "pcap",
            "source_artifact_paths": list(source_artifact_paths),
        },
        "outcome": {
            "accepted": bool(_optional_int(transaction.get("gds_accept"))) if transaction.get("gds_accept") not in (None, "") else None,
            "executed_successfully": bool(_optional_int(transaction.get("sat_success"))) if transaction.get("sat_success") not in (None, "") else None,
            "timed_out": bool(_optional_int(transaction.get("timeout"))) if transaction.get("timeout") not in (None, "") else None,
            "raw_code": _optional_number(transaction.get("response_code")),
            "raw_reason": _optional_text(transaction.get("reason")),
            "warning_count": _optional_number(transaction.get("txn_warning_events")),
            "error_count": _optional_number(transaction.get("txn_error_events")),
            "response_direction_seen": bool(_optional_int(transaction.get("response_direction_seen")))
            if transaction.get("response_direction_seen") not in (None, "")
            else None,
            "terminal_observed_on_wire": bool(_optional_int(transaction.get("final_observed_on_wire")))
            if transaction.get("final_observed_on_wire") not in (None, "")
            else None,
            "raw_event_name": _optional_text(transaction.get("response_name")),
        },
        "correlation": {
            "run_id": _optional_int(transaction.get("run_id")),
            "episode_id": _optional_int(transaction.get("episode_id")),
            "session_id": _optional_text(transaction.get("session_id")),
            "transaction_id": _optional_text(transaction.get("txn_id")),
            "send_id": _optional_text(transaction.get("send_id")),
            "stream_id": _optional_text(transaction.get("target_stream_id")),
            "stream_index": _optional_int(transaction.get("target_stream_index")),
        },
        "evaluation_context": {
            "label": _optional_int(transaction.get("label")),
            "label_name": _optional_text(transaction.get("label_name")),
            "attack_family": _optional_text(transaction.get("attack_family")),
            "phase": _optional_text(transaction.get("phase")),
            "actor_id": _optional_text(transaction.get("actor")),
            "actor_role": _optional_text(transaction.get("actor_role")),
            "actor_trust": _optional_number(transaction.get("actor_trust")),
        },
        "evidence": {
            "related_packet_count": len(related_packets),
            "observed_message_families": observed_message_families,
            "observed_message_stages": observed_message_stages,
            "packet_timestamp_sources": timestamp_sources,
            "packet_byte_sources": byte_sources,
            "request_wire_observed": bool(_optional_number(transaction.get("req_bytes"))),
            "response_wire_observed": bool(_optional_int(transaction.get("response_direction_seen"))),
            "log_correlation_mode": "send_id_and_pcap_match",
            "source_artifact_paths": list(source_artifact_paths),
        },
        "native_state_snapshot": transaction.get("native_state_snapshot"),
        "native_fields": {"legacy_record": dict(transaction)},
    }


def build_packets_from_real_artifacts(
    run_rows: list[dict[str, Any]],
    *,
    pcap_path: Path,
    capture_interface: str,
    capture_backend: str | None = None,
    source_artifact_paths: Iterable[str] | None = None,
    strict: bool = True,
) -> PacketBuildResult:
    pcap_packets = parse_pcap_capture(pcap_path)
    wire_messages = decode_wire_messages(pcap_packets)
    telemetry_records = telemetry_records_from_wire_messages(wire_messages, target_logical_id=VEHICLE_SERVICE)
    packets: list[dict[str, Any]] = []
    transactions: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    raw_packets: list[dict[str, Any]] = []
    raw_transactions: list[dict[str, Any]] = []
    used_request_indexes: set[int] = set()
    used_response_indexes: set[int] = set()

    default_run_id = None
    if run_rows:
        default_run_id = _optional_int(run_rows[0].get("run_id"))
    for telemetry_record in telemetry_records:
        packets.append(telemetry_record_to_packet(telemetry_record, run_id=default_run_id))

    transaction_packets_by_send_id: dict[str, list[dict[str, Any]]] = {}

    for row_index, row in enumerate(run_rows):
        request_index = find_best_request_candidate(row, wire_messages, used_request_indexes)
        request_message = None if request_index is None else wire_messages[request_index]
        if request_index is not None:
            used_request_indexes.add(request_index)

        response_index = find_best_response_candidate(
            row,
            wire_messages,
            used_response_indexes,
            request_message=request_message,
        )
        response_message = None if response_index is None else wire_messages[response_index]
        if response_index is not None:
            used_response_indexes.add(response_index)

        telemetry_for_row = [
            record
            for record in telemetry_records
            if (
                (_optional_int(row.get("send_end_ms")) or _optional_int(row.get("send_start_ms")) or 0) - record.ts_ms
            )
            in range(0, TELEMETRY_RECENT_THRESHOLD_MS + 1)
        ]
        observed_before_ms = _optional_int(row.get("send_end_ms")) or _optional_int(row.get("send_start_ms")) or 0
        telemetry_recent = telemetry_recent_for_time(
            telemetry_records,
            target_logical_id=VEHICLE_SERVICE,
            observed_before_ms=observed_before_ms,
        )
        state_snapshot = latest_native_state_snapshot(
            telemetry_records,
            target_logical_id=VEHICLE_SERVICE,
            observed_before_ms=observed_before_ms,
        )
        gds_accept, sat_success, response_code, reason = response_outcome(row, response_message)
        request_packet = build_request_packet(row, request_message)
        final_packet = build_final_packet(
            row,
            response_message,
            response_code=response_code,
            reason=reason,
            sat_success=sat_success,
        )
        final_packet["gds_accept"] = gds_accept
        packets.extend([request_packet, final_packet])
        transaction = build_transaction_record(
            row,
            request_packet,
            final_packet,
            request_message=request_message,
            response_message=response_message,
            telemetry_records=telemetry_for_row,
            telemetry_recent=telemetry_recent,
            state_snapshot=state_snapshot,
        )
        transactions.append(transaction)

        observation = {
            "row_index": row_index,
            "send_id": _optional_text(row.get("send_id")),
            "command": _optional_text(row.get("command")),
            "source_service": _optional_text(row.get("source_service")),
            "target_service": _optional_text(row.get("target_service")),
            "request_wire_seen": request_message is not None,
            "response_direction_seen": response_message is not None,
            "terminal_event_seen": response_message is not None,
            "telemetry_recent": telemetry_recent,
            "state_snapshot_seen": state_snapshot is not None,
            "request_message_name": None if request_message is None else request_message.message_name,
            "response_message_name": None if response_message is None else response_message.message_name,
            "request_ts_ms": None if request_message is None else request_message.ts_ms,
            "final_ts_ms": None if response_message is None else response_message.ts_ms,
        }
        observations.append(observation)
        send_id = _optional_text(row.get("send_id")) or transaction_id_for_row(row)
        transaction_packets_by_send_id[send_id] = [request_packet, final_packet]

    packets.sort(key=_packet_sort_key)
    raw_source_paths = list(source_artifact_paths or [])
    for packet in packets:
        raw_packets.append(
            build_raw_packet_record(
                packet,
                source_artifact_paths=raw_source_paths,
                capture_backend=capture_backend,
                capture_interface=capture_interface,
            )
        )
    for transaction in transactions:
        send_id = _optional_text(transaction.get("send_id")) or ""
        raw_transactions.append(
            build_raw_transaction_record(
                transaction,
                related_packets=transaction_packets_by_send_id.get(send_id, []),
                source_artifact_paths=raw_source_paths,
                capture_backend=capture_backend,
                capture_interface=capture_interface,
            )
        )

    validate_raw_packet_records(raw_packets)
    validate_raw_transaction_records(raw_transactions)

    missing_request_rows = [item["send_id"] for item in observations if not bool(item.get("request_wire_seen"))]
    missing_response_rows = [
        item["send_id"]
        for item in observations
        if not bool(item.get("response_direction_seen")) and not bool(_optional_int(run_rows[item["row_index"]].get("timeout")) == 1)
    ]
    if strict and missing_request_rows:
        raise ReconstructionError(
            "Could not match all MAVLink request sends to on-wire packets. "
            f"Missing: {missing_request_rows[:8]}"
        )
    if strict and missing_response_rows:
        raise ReconstructionError(
            "Could not match all completed MAVLink sends to on-wire responses. "
            f"Missing: {missing_response_rows[:8]}"
        )

    provenance_summary = {
        "generated_schema_version": SCHEMA_VERSION,
        "protocol_family": "mavlink",
        "capture_backend": capture_backend,
        "capture_interface": capture_interface,
        "pcap_packet_count": len(pcap_packets),
        "decoded_wire_message_count": len(wire_messages),
        "telemetry_message_count": len(telemetry_records),
        "request_row_count": len(run_rows),
        "packet_count": len(packets),
        "transaction_count": len(transactions),
        "raw_packet_count": len(raw_packets),
        "raw_transaction_count": len(raw_transactions),
        "request_wire_seen_count": sum(1 for item in observations if bool(item.get("request_wire_seen"))),
        "response_direction_seen_count": sum(1 for item in observations if bool(item.get("response_direction_seen"))),
        "telemetry_recent_count": sum(1 for item in observations if bool(item.get("telemetry_recent"))),
        "state_snapshot_seen_count": sum(1 for item in observations if bool(item.get("state_snapshot_seen"))),
        "missing_request_rows": missing_request_rows,
        "missing_response_rows": missing_response_rows,
        "source_artifact_paths": raw_source_paths,
    }

    return PacketBuildResult(
        packets=packets,
        transactions=transactions,
        raw_packets=raw_packets,
        raw_transactions=raw_transactions,
        observations=observations,
        telemetry_records=[
            {
                "ts_ms": record.ts_ms,
                "target_logical_id": record.target_logical_id,
                "message_name": record.message_name,
                "payload": dict(record.payload),
                "bytes_on_wire": record.bytes_on_wire,
                "observed_on_wire": record.observed_on_wire,
                "ts_source": record.ts_source,
                "bytes_source": record.bytes_source,
            }
            for record in telemetry_records
        ],
        provenance_summary=provenance_summary,
        source_artifact_paths=raw_source_paths,
    )


def write_artifact_bundle(
    *,
    output_dir: Path,
    packet_result: PacketBuildResult,
    run_rows: list[dict[str, Any]],
) -> tuple[Path, Path]:
    from tools.mavlink_real.support_probe import assert_actual_run_observability, build_actual_run_observability_report

    data_dir = output_dir / "data"
    report_dir = output_dir / "reports"
    write_jsonl(data_dir / "packets.jsonl", packet_result.packets)
    write_jsonl(data_dir / "transactions.jsonl", packet_result.transactions)
    write_jsonl(data_dir / "raw_packets.jsonl", packet_result.raw_packets)
    write_jsonl(data_dir / "raw_transactions.jsonl", packet_result.raw_transactions)
    actual_run_report = build_actual_run_observability_report(run_rows, packet_result.observations)
    save_json(report_dir / "actual_run_observability.json", actual_run_report)
    assert_actual_run_observability(actual_run_report)
    save_json(report_dir / "provenance_summary.json", packet_result.provenance_summary)
    save_json(
        report_dir / "schema.json",
        {
            "schema_version": SCHEMA_VERSION,
            "raw_packet_schema_version": RAW_PACKET_SCHEMA_VERSION,
            "raw_transaction_schema_version": RAW_TRANSACTION_SCHEMA_VERSION,
            "artifact_counts": {
                "packets": len(packet_result.packets),
                "transactions": len(packet_result.transactions),
                "raw_packets": len(packet_result.raw_packets),
                "raw_transactions": len(packet_result.raw_transactions),
            },
        },
    )
    return data_dir, report_dir


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--runtime-root", type=Path, required=True)
    parser.add_argument("--run-log", type=Path, required=True)
    parser.add_argument("--pcap", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--capture-interface", default="")
    parser.add_argument("--capture-backend", default="")
    parser.add_argument("--strict", action="store_true")
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    run_rows, source_artifact_paths = load_runtime_run_rows(args.run_log, runtime_root=args.runtime_root)
    if args.pcap.exists():
        source_artifact_paths = [*source_artifact_paths, str(args.pcap.resolve())]
    packet_result = build_packets_from_real_artifacts(
        run_rows,
        pcap_path=args.pcap,
        capture_interface=args.capture_interface,
        capture_backend=_optional_text(args.capture_backend),
        source_artifact_paths=source_artifact_paths,
        strict=bool(args.strict),
    )
    data_dir, report_dir = write_artifact_bundle(
        output_dir=args.output_dir,
        packet_result=packet_result,
        run_rows=run_rows,
    )
    print(
        json.dumps(
            {
                "data_dir": str(data_dir.resolve()),
                "report_dir": str(report_dir.resolve()),
                "packet_count": len(packet_result.packets),
                "transaction_count": len(packet_result.transactions),
                "raw_packet_count": len(packet_result.raw_packets),
                "raw_transaction_count": len(packet_result.raw_transactions),
            },
            separators=(",", ":"),
            sort_keys=True,
        )
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
