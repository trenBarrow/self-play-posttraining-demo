#!/usr/bin/env python3
"""Observation-first packet and lifecycle reconstruction for real F' runs."""

from __future__ import annotations

import json
import re
import socket
import struct
import sys
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

import dpkt

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime import COMMAND_SPECS, normalize_command_args
from tools.fprime_real.runtime_layout import TARGET_NODE_BY_SERVICE
from tools.fprime_real.telemetry_catalog import TELEMETRY_BY_CHANNEL, catalog_bucket, convert_numeric_entry

SCHEMA_VERSION = "real_fprime_v2"
TARGET_PORTS = {50050}
TERMINAL_EVENT_NAMES = {"cmdDisp.OpCodeCompleted", "cmdDisp.OpCodeError"}
TELEMETRY_RECENT_THRESHOLD_MS = 5000
REQUEST_WINDOW_EARLY_MS = 2000
REQUEST_WINDOW_LATE_MS = 5000
REQUEST_OVERLAP_LATE_MS = 1000
COMMAND_AUDIT_MATCH_SKEW_MS = REQUEST_WINDOW_EARLY_MS + REQUEST_WINDOW_LATE_MS
RAW_TS_RE = re.compile(r"-(?P<seconds>\d+):(?P<micros>\d+)\)$")

WARNING_EVENT_NAMES = {
    "fileDownlink.FileOpenError",
    "fileManager.FileMoveError",
    "fileManager.FileSizeError",
}
PACKET_KIND_ORDER = {
    "telemetry": 0,
    "request": 1,
    "uplink": 2,
    "sat_response": 3,
    "final": 4,
}


@dataclass(frozen=True)
class CommandLogEntry:
    ts_ms: int
    command: str
    raw_line: str


@dataclass(frozen=True)
class SendLogEntry:
    send_id: str
    send_start_ms: int
    send_end_ms: int
    source_service: str
    source_ip: str
    target_service: str
    target_ip: str
    target_tts_port: int
    target_stream_id: str
    target_stream_index: int
    tts_port: int
    command: str
    args_sha256: str
    gds_accept: int
    sat_success: int
    timeout: int
    response_code: int
    reason: str
    raw_payload: dict[str, Any]


@dataclass(frozen=True)
class EventLogEntry:
    ts_ms: int
    name: str
    severity: str
    display_text: str
    raw_line: str


@dataclass(frozen=True)
class TelemetryRecord:
    ts_ms: int
    node_service: str
    payload: dict[str, float]
    raw_values: dict[str, str]
    channel_names: list[str]
    bytes_on_wire: int
    observed_on_wire: bool
    ts_source: str
    bytes_source: str


@dataclass
class PcapIncrementalReader:
    offset: int = 0
    endian: str | None = None
    nanosecond_resolution: bool = False
    linktype: int | None = None
    partial: bytes = b""
    packets: list["PcapPacket"] = field(default_factory=list)
    flow_states: list[dict[str, Any]] = field(default_factory=list)
    current_flow_by_key: dict[tuple[str, int, str, int], dict[str, Any]] = field(default_factory=dict)

    def update(self, pcap_path: Path) -> None:
        if not pcap_path.exists():
            return
        with pcap_path.open("rb") as handle:
            handle.seek(self.offset)
            chunk = handle.read()
        if not chunk:
            return
        self.offset += len(chunk)
        buffer = self.partial + chunk
        start = 0
        if self.endian is None:
            if len(buffer) < 24:
                self.partial = buffer
                return
            magic = pcap_magic(buffer[:4])
            if magic is None:
                raise SystemExit(f"Unsupported pcap format for {pcap_path}")
            self.endian, self.nanosecond_resolution = magic
            try:
                _, _, _, _, _, _, self.linktype = struct.unpack(f"{self.endian}IHHIIII", buffer[:24])
            except struct.error as exc:
                raise SystemExit(f"Could not parse pcap header for {pcap_path}: {exc}") from None
            start = 24

        offset = start
        while offset + 16 <= len(buffer):
            try:
                ts_sec, ts_frac, incl_len, _ = struct.unpack(f"{self.endian}IIII", buffer[offset : offset + 16])
            except struct.error:
                break
            if offset + 16 + incl_len > len(buffer):
                break
            frame = buffer[offset + 16 : offset + 16 + incl_len]
            offset += 16 + incl_len
            packet = parse_pcap_frame(
                linktype=self.linktype,
                ts_sec=ts_sec,
                ts_frac=ts_frac,
                nanosecond_resolution=self.nanosecond_resolution,
                frame=frame,
            )
            if packet is None:
                continue
            self.packets.append(packet)
            update_pcap_flow_state(packet, self.flow_states, self.current_flow_by_key)
        self.partial = buffer[offset:]

    def parse_result(self) -> "PcapParseResult":
        return PcapParseResult(
            packet_count=len(self.packets),
            linktype=self.linktype,
            packets=list(self.packets),
            connections=connections_from_flow_states(self.flow_states),
        )


@dataclass(frozen=True)
class PcapFlow:
    client_ip: str
    server_ip: str
    target_port: int
    client_port: int
    request_first_ms: int
    request_last_ms: int
    response_first_ms: int | None
    response_last_ms: int | None
    request_bytes: int
    response_bytes: int
    saw_response_direction: bool


@dataclass(frozen=True)
class PcapPacket:
    ts_ms: int
    src_ip: str
    dst_ip: str
    src_port: int
    dst_port: int
    payload_len: int
    tcp_flags: int


@dataclass(frozen=True)
class PcapParseResult:
    packet_count: int
    linktype: int | None
    packets: list[PcapPacket]
    connections: list[PcapFlow]


@dataclass
class CommandObservation:
    row_index: int
    row: dict[str, str]
    request_entry: CommandLogEntry | None
    request_flow: PcapFlow | None
    dispatch_event: EventLogEntry | None
    sat_response_event: EventLogEntry | None
    terminal_event: EventLogEntry | None
    request_wire_seen: bool
    response_direction_seen: bool
    telemetry_recent: bool
    warning_events: int
    error_events: int
    lifecycle_event_names: list[str]


@dataclass
class PacketBuildResult:
    packets: list[dict[str, Any]]
    observations: list[dict[str, Any]]
    channel_inventory: dict[str, Any]
    provenance_summary: dict[str, Any]
    source_artifact_paths: list[str] = field(default_factory=list)


def resolve_source_artifact_paths(*groups: list[Path] | None) -> list[str]:
    values: list[str] = []
    for group in groups:
        if group is None:
            continue
        for path in group:
            resolved = str(path.resolve())
            if resolved not in values and path.exists():
                values.append(resolved)
    return values


def read_lines(path: Path) -> list[str]:
    if not path.exists():
        return []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        return [line.strip() for line in handle if line.strip()]


def parse_raw_ts_ms(raw_token: str) -> int:
    match = RAW_TS_RE.search(raw_token.strip())
    if match is None:
        raise ValueError(f"Could not parse log timestamp token: {raw_token!r}")
    seconds = int(match.group("seconds"))
    micros = int(match.group("micros"))
    return (seconds * 1000) + (micros // 1000)


def parse_command_log_line(line: str) -> CommandLogEntry:
    parts = line.split(",", 4)
    if len(parts) != 5:
        raise ValueError(f"Unexpected command log line: {line!r}")
    _, raw_ts, command, _, _ = parts
    return CommandLogEntry(ts_ms=parse_raw_ts_ms(raw_ts), command=command.strip(), raw_line=line)


def parse_event_log_line(line: str) -> EventLogEntry:
    parts = line.split(",", 5)
    if len(parts) != 6:
        raise ValueError(f"Unexpected event log line: {line!r}")
    _, raw_ts, name, _, severity, display_text = parts
    return EventLogEntry(
        ts_ms=parse_raw_ts_ms(raw_ts),
        name=name.strip(),
        severity=severity.strip(),
        display_text=display_text.strip(),
        raw_line=line,
    )


def canonicalize_channel_value(channel_name: str, raw_value: str) -> tuple[str, float] | None:
    entry = TELEMETRY_BY_CHANNEL.get(channel_name)
    if entry is None or not entry.enabled_for_model:
        return None
    value = convert_numeric_entry(entry, raw_value)
    if value is None:
        return None
    return entry.feature_name, float(value)


def update_channel_inventory(inventory: dict[str, Any], node_service: str, channel_name: str, raw_value: str, mapped_field: str | None) -> None:
    nodes = inventory.setdefault("nodes", {})
    entry = TELEMETRY_BY_CHANNEL.get(channel_name)
    bucket_name = catalog_bucket(channel_name)
    node_bucket = nodes.setdefault(
        node_service,
        {
            "modeled": {},
            "inventory_only": {},
            "unknown": {},
        },
    )
    bucket = node_bucket[bucket_name]
    entry = bucket.setdefault(
        channel_name,
        {
            "count": 0,
            "mapped_field": mapped_field or "",
            "catalog_feature": TELEMETRY_BY_CHANNEL.get(channel_name).feature_name if TELEMETRY_BY_CHANNEL.get(channel_name) else "",
            "catalog_kind": TELEMETRY_BY_CHANNEL.get(channel_name).kind if TELEMETRY_BY_CHANNEL.get(channel_name) else "",
            "enabled_for_model": bool(TELEMETRY_BY_CHANNEL.get(channel_name).enabled_for_model) if TELEMETRY_BY_CHANNEL.get(channel_name) else False,
            "examples": [],
        },
    )
    entry["count"] = int(entry.get("count", 0)) + 1
    examples = list(entry.get("examples", []))
    if raw_value not in examples and len(examples) < 3:
        examples.append(raw_value)
        entry["examples"] = examples


def finalize_channel_inventory(inventory: dict[str, Any]) -> None:
    nodes = inventory.setdefault("nodes", {})
    summary = {"modeled": 0, "inventory_only": 0, "unknown": 0}
    for node_bucket in nodes.values():
        for bucket_name in summary:
            summary[bucket_name] += len(node_bucket.get(bucket_name, {}))
    inventory["summary"] = summary


def parse_channel_log(path: Path, node_service: str, inventory: dict[str, Any]) -> list[TelemetryRecord]:
    samples: list[TelemetryRecord] = []
    for line in read_lines(path):
        parts = line.split(",", 4)
        if len(parts) != 5:
            continue
        _, raw_ts, channel_name, _, raw_value = parts
        mapped = canonicalize_channel_value(channel_name.strip(), raw_value)
        update_channel_inventory(inventory, node_service, channel_name.strip(), raw_value, mapped[0] if mapped else None)
        if mapped is None:
            continue
        field_name, value = mapped
        samples.append(
            TelemetryRecord(
                ts_ms=parse_raw_ts_ms(raw_ts),
                node_service=node_service,
                payload={field_name: value},
                raw_values={channel_name.strip(): raw_value.strip()},
                channel_names=[channel_name.strip()],
                bytes_on_wire=0,
                observed_on_wire=False,
                ts_source="channel_log",
                bytes_source="none",
            )
        )
    samples.sort(key=lambda sample: (sample.ts_ms, sample.node_service, tuple(sorted(sample.payload))))
    return samples


def parse_downlink_records(path: Path, node_service: str, inventory: dict[str, Any]) -> list[TelemetryRecord]:
    records: list[TelemetryRecord] = []
    if not path.exists():
        return records
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if str(payload.get("kind", "")) != "telemetry":
                continue
            raw_values: dict[str, str] = {}
            mapped_payload: dict[str, float] = {}
            for channel in payload.get("channels", []):
                channel_name = str(channel.get("name", "")).strip()
                raw_value = str(channel.get("display_text", "")).strip()
                if not channel_name:
                    continue
                raw_values[channel_name] = raw_value
                mapped = canonicalize_channel_value(channel_name, raw_value)
                update_channel_inventory(inventory, node_service, channel_name, raw_value, mapped[0] if mapped else None)
                if mapped is None:
                    continue
                field_name, value = mapped
                mapped_payload[field_name] = value
            if not mapped_payload:
                continue
            raw_time = str(payload.get("raw_time", "")).strip()
            ts_ms = int(payload.get("ts_ms", 0) or 0)
            if ts_ms <= 0 and raw_time:
                ts_ms = parse_raw_ts_ms(raw_time)
            records.append(
                TelemetryRecord(
                    ts_ms=ts_ms,
                    node_service=node_service,
                    payload=mapped_payload,
                    raw_values=raw_values,
                    channel_names=list(raw_values.keys()),
                    bytes_on_wire=int(payload.get("bytes_on_wire", payload.get("frame_bytes", 0)) or 0),
                    observed_on_wire=True,
                    ts_source="gds_recv_bin",
                    bytes_source="gds_recv_bin",
                )
            )
    records.sort(key=lambda record: (record.ts_ms, record.node_service, tuple(sorted(record.payload))))
    return records


def pcap_magic(bytez: bytes) -> tuple[str, bool] | None:
    if bytez == b"\xd4\xc3\xb2\xa1":
        return "<", False
    if bytez == b"\xa1\xb2\xc3\xd4":
        return ">", False
    if bytez == b"\x4d\x3c\xb2\xa1":
        return "<", True
    if bytez == b"\xa1\xb2\x3c\x4d":
        return ">", True
    return None


def packet_ts_ms(ts_sec: int, ts_frac: int, nanosecond_resolution: bool) -> int:
    divisor = 1_000_000.0 if nanosecond_resolution else 1_000.0
    return int((float(ts_sec) * 1000.0) + (float(ts_frac) / divisor))


def inet_ntop_safe(raw: bytes) -> str:
    if len(raw) == 4:
        return socket.inet_ntop(socket.AF_INET, raw)
    if len(raw) == 16:
        return socket.inet_ntop(socket.AF_INET6, raw)
    raise ValueError(f"Unsupported IP byte length: {len(raw)}")


def ip_packet_from_frame(linktype: int, frame: bytes) -> dpkt.ip.IP | dpkt.ip6.IP6 | None:
    try:
        if linktype == dpkt.pcap.DLT_EN10MB:
            return dpkt.ethernet.Ethernet(frame).data
        if linktype in {dpkt.pcap.DLT_NULL, dpkt.pcap.DLT_LOOP}:
            return dpkt.loopback.Loopback(frame).data
        if linktype == dpkt.pcap.DLT_RAW:
            version = frame[0] >> 4
            if version == 4:
                return dpkt.ip.IP(frame)
            if version == 6:
                return dpkt.ip6.IP6(frame)
            return None
        if linktype == dpkt.pcap.DLT_LINUX_SLL:
            return dpkt.sll.SLL(frame).data
    except (dpkt.dpkt.NeedData, dpkt.dpkt.UnpackError, ValueError, IndexError):
        return None
    return None


def tcp_packet_from_network(network: Any) -> tuple[str, str, dpkt.tcp.TCP] | None:
    if isinstance(network, dpkt.ip.IP):
        if network.p != dpkt.ip.IP_PROTO_TCP:
            return None
        if not isinstance(network.data, dpkt.tcp.TCP):
            return None
        return inet_ntop_safe(network.src), inet_ntop_safe(network.dst), network.data
    if isinstance(network, dpkt.ip6.IP6):
        if network.nxt != dpkt.ip.IP_PROTO_TCP:
            return None
        if not isinstance(network.data, dpkt.tcp.TCP):
            return None
        return inet_ntop_safe(network.src), inet_ntop_safe(network.dst), network.data
    return None


def parse_pcap_frame(
    *,
    linktype: int | None,
    ts_sec: int,
    ts_frac: int,
    nanosecond_resolution: bool,
    frame: bytes,
) -> PcapPacket | None:
    if linktype is None:
        return None
    network = ip_packet_from_frame(linktype, frame)
    tcp_tuple = tcp_packet_from_network(network)
    if tcp_tuple is None:
        return None
    src_ip, dst_ip, tcp_packet = tcp_tuple
    ts_ms = packet_ts_ms(ts_sec, ts_frac, nanosecond_resolution)
    payload_len = len(bytes(tcp_packet.data))
    return PcapPacket(
        ts_ms=ts_ms,
        src_ip=src_ip,
        dst_ip=dst_ip,
        src_port=int(tcp_packet.sport),
        dst_port=int(tcp_packet.dport),
        payload_len=payload_len,
        tcp_flags=int(tcp_packet.flags),
    )


def update_pcap_flow_state(
    packet: PcapPacket,
    flow_states: list[dict[str, Any]],
    current_flow_by_key: dict[tuple[str, int, str, int], dict[str, Any]],
) -> None:
    tcp_syn = bool(packet.tcp_flags & dpkt.tcp.TH_SYN)
    tcp_ack = bool(packet.tcp_flags & dpkt.tcp.TH_ACK)
    tcp_fin = bool(packet.tcp_flags & dpkt.tcp.TH_FIN)
    tcp_rst = bool(packet.tcp_flags & dpkt.tcp.TH_RST)

    if packet.dst_port in TARGET_PORTS and packet.src_port not in TARGET_PORTS:
        key = (packet.src_ip, packet.src_port, packet.dst_ip, packet.dst_port)
        flow = current_flow_by_key.get(key)
        if flow is None or flow.get("closed", False) or (tcp_syn and not tcp_ack):
            flow = {
                "client_ip": packet.src_ip,
                "client_port": packet.src_port,
                "server_ip": packet.dst_ip,
                "target_port": packet.dst_port,
                "request_first_ms": None,
                "request_last_ms": None,
                "response_first_ms": None,
                "response_last_ms": None,
                "request_bytes": 0,
                "response_bytes": 0,
                "saw_response_direction": False,
                "closed": False,
            }
            flow_states.append(flow)
            current_flow_by_key[key] = flow
        if packet.payload_len > 0:
            if flow["request_first_ms"] is None:
                flow["request_first_ms"] = packet.ts_ms
            flow["request_last_ms"] = packet.ts_ms
            flow["request_bytes"] += packet.payload_len
        if tcp_fin or tcp_rst:
            flow["closed"] = True
        return

    if packet.src_port in TARGET_PORTS and packet.dst_port not in TARGET_PORTS:
        key = (packet.dst_ip, packet.dst_port, packet.src_ip, packet.src_port)
        flow = current_flow_by_key.get(key)
        if flow is None or flow.get("closed", False):
            return
        if flow["request_first_ms"] is None:
            if tcp_fin or tcp_rst:
                flow["closed"] = True
            return
        flow["saw_response_direction"] = True
        if flow["response_first_ms"] is None:
            flow["response_first_ms"] = packet.ts_ms
        flow["response_last_ms"] = packet.ts_ms
        if packet.payload_len > 0:
            flow["response_bytes"] += packet.payload_len
        if tcp_fin or tcp_rst:
            flow["closed"] = True


def connections_from_flow_states(flow_states: list[dict[str, Any]]) -> list[PcapFlow]:
    connections = [
        PcapFlow(
            client_ip=str(payload["client_ip"]),
            server_ip=str(payload["server_ip"]),
            target_port=int(payload["target_port"]),
            client_port=int(payload["client_port"]),
            request_first_ms=int(payload["request_first_ms"]),
            request_last_ms=int(payload["request_last_ms"] or payload["request_first_ms"]),
            response_first_ms=int(payload["response_first_ms"]) if payload["response_first_ms"] is not None else None,
            response_last_ms=int(payload["response_last_ms"]) if payload["response_last_ms"] is not None else None,
            request_bytes=int(payload["request_bytes"]),
            response_bytes=int(payload["response_bytes"]),
            saw_response_direction=bool(payload["saw_response_direction"]),
        )
        for payload in flow_states
        if payload["request_first_ms"] is not None
    ]
    connections.sort(key=lambda flow: (flow.request_first_ms, flow.client_ip, flow.client_port, flow.server_ip, flow.target_port))
    return connections


def parse_pcap_capture(pcap_path: Path) -> PcapParseResult:
    if not pcap_path.exists():
        return PcapParseResult(packet_count=0, linktype=None, packets=[], connections=[])
    reader = PcapIncrementalReader()
    reader.update(pcap_path)
    return reader.parse_result()


def ordered_run_rows(run_rows: list[dict[str, str]]) -> list[tuple[int, dict[str, str]]]:
    indexed = list(enumerate(run_rows))
    indexed.sort(
        key=lambda item: (
            int(item[1].get("real_ms", "0") or "0"),
            int(item[1].get("virtual_day", "0") or "0"),
            int(item[1].get("virtual_seconds", "0") or "0"),
            item[1].get("source_service", ""),
            item[1].get("command", ""),
        )
    )
    return indexed


def row_real_ms(row: dict[str, str]) -> int:
    return int(row.get("real_ms", "0") or "0")


def row_send_id(row: dict[str, str]) -> str:
    return str(row.get("send_id", "")).strip()


def row_source_ip(row: dict[str, str]) -> str:
    return str(row.get("source_ip", "")).strip()


def row_target_ip(row: dict[str, str]) -> str:
    return str(row.get("target_ip", "")).strip()


def row_target_stream_id(row: dict[str, Any]) -> str:
    explicit = str(row.get("target_stream_id", "")).strip()
    if explicit:
        return explicit
    target_service = str(row.get("target_service", "")).strip()
    try:
        target_port = int(row.get("target_tts_port", row.get("tts_port", 50050)) or 50050)
    except (TypeError, ValueError):
        target_port = 50050
    return f"{target_service}:{target_port}" if target_service else ""


def row_target_stream_index(row: dict[str, Any]) -> int:
    try:
        return int(row.get("target_stream_index", -1) or -1)
    except (TypeError, ValueError):
        return -1


def row_run_id(row: dict[str, Any]) -> int:
    value = row.get("run_id")
    if value not in (None, ""):
        try:
            parsed = int(value)
            if parsed >= 0:
                return parsed
        except (TypeError, ValueError):
            pass
    try:
        meta = json.loads(row.get("meta_json", "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        meta = {}
    if isinstance(meta, dict):
        try:
            return int(meta.get("run_id", -1))
        except (TypeError, ValueError):
            return -1
    return -1


def row_latency_ms(row: dict[str, str]) -> int:
    return max(1, int(float(row.get("latency_ms", "1") or 1.0)))


def row_gds_accept(row: dict[str, str]) -> bool:
    return int(row.get("gds_accept", "0") or "0") == 1


def row_timeout(row: dict[str, str]) -> bool:
    return int(row.get("timeout", "0") or "0") == 1


def parse_command_log(path: Path) -> list[CommandLogEntry]:
    return [parse_command_log_line(line) for line in read_lines(path)]


def send_entry_from_payload(payload: dict[str, Any]) -> SendLogEntry:
    return SendLogEntry(
        send_id=str(payload.get("send_id", "")),
        send_start_ms=int(payload.get("send_start_ms", payload.get("real_ms", 0)) or 0),
        send_end_ms=int(payload.get("send_end_ms", payload.get("real_ms", 0)) or 0),
        source_service=str(payload.get("source_service", "")),
        source_ip=str(payload.get("source_ip", "")),
        target_service=str(payload.get("target_service", "")),
        target_ip=str(payload.get("target_ip", "")),
        target_tts_port=int(payload.get("target_tts_port", payload.get("tts_port", 0)) or 0),
        target_stream_id=row_target_stream_id(payload),
        target_stream_index=row_target_stream_index(payload),
        tts_port=int(payload.get("tts_port", payload.get("target_tts_port", 0)) or 0),
        command=str(payload.get("command", "")),
        args_sha256=str(payload.get("args_sha256", "")),
        gds_accept=int(payload.get("gds_accept", 0) or 0),
        sat_success=int(payload.get("sat_success", 0) or 0),
        timeout=int(payload.get("timeout", 0) or 0),
        response_code=int(payload.get("response_code", 0) or 0),
        reason=str(payload.get("reason", "")),
        raw_payload=dict(payload),
    )


def send_entry_from_row(row: dict[str, str], row_index: int) -> SendLogEntry:
    return SendLogEntry(
        send_id=row_send_id(row) or f"legacy-{row_index:06d}",
        send_start_ms=int(row.get("send_start_ms", row.get("real_ms", 0)) or 0),
        send_end_ms=int(row.get("send_end_ms", row.get("real_ms", 0)) or 0),
        source_service=str(row.get("source_service", "")),
        source_ip=str(row.get("source_ip", "")),
        target_service=str(row.get("target_service", "")),
        target_ip=str(row.get("target_ip", "")),
        target_tts_port=int(row.get("target_tts_port", row.get("tts_port", 0)) or 0),
        target_stream_id=row_target_stream_id(row),
        target_stream_index=row_target_stream_index(row),
        tts_port=int(row.get("tts_port", row.get("target_tts_port", 0)) or 0),
        command=str(row.get("command", "")),
        args_sha256=str(row.get("args_sha256", "")),
        gds_accept=int(row.get("gds_accept", 0) or 0),
        sat_success=int(row.get("sat_success", 0) or 0),
        timeout=int(row.get("timeout", 0) or 0),
        response_code=int(row.get("response_code", 0) or 0),
        reason=str(row.get("reason", "")),
        raw_payload=dict(row),
    )


def parse_send_log(path: Path | None) -> list[SendLogEntry]:
    if path is None or not path.exists():
        return []
    entries: list[SendLogEntry] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            entries.append(send_entry_from_payload(json.loads(text)))
    return entries


def parse_event_log(path: Path) -> list[EventLogEntry]:
    return [parse_event_log_line(line) for line in read_lines(path)]


def match_send_entries(
    ordered_rows: list[tuple[int, dict[str, str]]],
    send_entries: list[SendLogEntry],
    *,
    strict: bool,
) -> dict[int, SendLogEntry]:
    by_send_id = {entry.send_id: entry for entry in send_entries if entry.send_id}
    matches: dict[int, SendLogEntry] = {}
    for row_index, row in ordered_rows:
        send_id = row_send_id(row)
        if send_id:
            entry = by_send_id.get(send_id)
            if entry is None:
                if strict:
                    raise SystemExit(f"Missing sidecar send-log entry for send_id={send_id}")
                matches[row_index] = send_entry_from_row(row, row_index)
                continue
            matches[row_index] = entry
            continue
        if strict:
            raise SystemExit(f"Missing send_id in run row for command={row.get('command', '')}")
        matches[row_index] = send_entry_from_row(row, row_index)
    return matches


def collect_serialization_violations(
    ordered_rows: list[tuple[int, dict[str, str]]],
    send_matches: dict[int, SendLogEntry],
) -> list[dict[str, Any]]:
    violations: list[dict[str, Any]] = []
    by_stream: dict[str, list[tuple[int, dict[str, str], SendLogEntry]]] = defaultdict(list)
    by_target: dict[str, list[tuple[int, dict[str, str], SendLogEntry]]] = defaultdict(list)
    for row_index, row in ordered_rows:
        send_entry = send_matches.get(row_index)
        if send_entry is None:
            continue
        if send_entry.target_stream_id:
            by_stream[send_entry.target_stream_id].append((row_index, row, send_entry))
        if send_entry.target_service:
            by_target[send_entry.target_service].append((row_index, row, send_entry))

    for target_stream_id, stream_rows in by_stream.items():
        stream_rows.sort(key=lambda item: (item[2].send_start_ms, item[2].send_end_ms, item[0]))
        previous_end_ms: int | None = None
        previous_index: int | None = None
        for row_index, row, send_entry in stream_rows:
            if send_entry.send_end_ms < send_entry.send_start_ms:
                violations.append(
                    {
                        "scope": "target_stream",
                        "target_stream_id": target_stream_id,
                        "row_index": row_index,
                        "command": row.get("command", ""),
                        "reason": "negative_send_window",
                        "send_start_ms": send_entry.send_start_ms,
                        "send_end_ms": send_entry.send_end_ms,
                    }
                )
            if previous_end_ms is not None and send_entry.send_start_ms < previous_end_ms:
                violations.append(
                    {
                        "scope": "target_stream",
                        "target_stream_id": target_stream_id,
                        "row_index": row_index,
                        "command": row.get("command", ""),
                        "reason": "overlapping_send_window",
                        "send_start_ms": send_entry.send_start_ms,
                        "previous_end_ms": previous_end_ms,
                    }
                )
            if previous_index is not None and send_entry.target_stream_index >= 0 and send_entry.target_stream_index <= previous_index:
                violations.append(
                    {
                        "scope": "target_stream",
                        "target_stream_id": target_stream_id,
                        "row_index": row_index,
                        "command": row.get("command", ""),
                        "reason": "non_monotonic_stream_index",
                        "target_stream_index": send_entry.target_stream_index,
                        "previous_index": previous_index,
                    }
                )
            previous_end_ms = send_entry.send_end_ms
            if send_entry.target_stream_index >= 0:
                previous_index = send_entry.target_stream_index

    for target_service, target_rows in by_target.items():
        target_rows.sort(key=lambda item: (item[2].send_start_ms, item[2].send_end_ms, item[0]))
        previous_end_ms: int | None = None
        previous_stream: str | None = None
        for row_index, row, send_entry in target_rows:
            if previous_end_ms is not None and send_entry.send_start_ms < previous_end_ms:
                violations.append(
                    {
                        "scope": "target_service",
                        "target_service": target_service,
                        "target_stream_id": send_entry.target_stream_id,
                        "previous_target_stream_id": previous_stream or "",
                        "row_index": row_index,
                        "command": row.get("command", ""),
                        "reason": "overlapping_event_window",
                        "send_start_ms": send_entry.send_start_ms,
                        "previous_end_ms": previous_end_ms,
                    }
                )
            previous_end_ms = send_entry.send_end_ms
            previous_stream = send_entry.target_stream_id
    return violations


def match_command_entries(
    ordered_rows: list[tuple[int, dict[str, str]]],
    send_matches: dict[int, SendLogEntry],
    command_entries: list[CommandLogEntry],
) -> dict[int, CommandLogEntry]:
    used: set[int] = set()
    by_command: dict[str, list[int]] = defaultdict(list)
    for entry_index, entry in enumerate(command_entries):
        by_command[entry.command].append(entry_index)

    matches: dict[int, CommandLogEntry] = {}
    for row_index, row in ordered_rows:
        send_entry = send_matches.get(row_index)
        if send_entry is None:
            continue
        command = str(row.get("command", ""))
        target_ms = send_entry.send_start_ms
        best_index: int | None = None
        best_delta: int | None = None
        for entry_index in by_command.get(command, []):
            if entry_index in used:
                continue
            entry = command_entries[entry_index]
            delta = abs(entry.ts_ms - target_ms)
            if delta > COMMAND_AUDIT_MATCH_SKEW_MS:
                continue
            if best_delta is None or delta < best_delta or (
                delta == best_delta and best_index is not None and entry.ts_ms < command_entries[best_index].ts_ms
            ):
                best_index = entry_index
                best_delta = delta
        if best_index is None:
            continue
        used.add(best_index)
        matches[row_index] = command_entries[best_index]
    return matches


def match_pcap_flows(
    ordered_rows: list[tuple[int, dict[str, str]]],
    send_matches: dict[int, SendLogEntry],
    packets: list[PcapPacket],
    *,
    strict: bool,
) -> dict[int, PcapFlow]:
    request_start_indexes: dict[int, int] = {}
    last_request_index_by_identity: dict[tuple[str, str, int], int] = {}
    for row_index, row in ordered_rows:
        if not row_gds_accept(row):
            continue
        send_entry = send_matches.get(row_index)
        if send_entry is None:
            continue
        source_ip = send_entry.source_ip or row_source_ip(row)
        target_ip = send_entry.target_ip or row_target_ip(row)
        if not source_ip:
            if strict:
                raise SystemExit(
                    f"Accepted command {row.get('command', '')} is missing source_ip for send_id={row.get('send_id', '')}"
                )
            continue
        if not target_ip:
            if strict:
                raise SystemExit(
                    f"Accepted command {row.get('command', '')} is missing target_ip for send_id={row.get('send_id', '')}"
                )
            continue

        window_start_ms = send_entry.send_start_ms - REQUEST_WINDOW_EARLY_MS
        window_end_ms = send_entry.send_end_ms + REQUEST_OVERLAP_LATE_MS
        identity_key = (source_ip, target_ip, int(send_entry.target_tts_port))
        minimum_packet_index = last_request_index_by_identity.get(identity_key, -1)
        best: tuple[int, int, int] | None = None
        for packet_index, packet in enumerate(packets):
            if packet_index <= minimum_packet_index:
                continue
            if packet.payload_len <= 0:
                continue
            if packet.src_ip != source_ip:
                continue
            if packet.dst_ip != target_ip:
                continue
            if packet.dst_port != int(send_entry.target_tts_port):
                continue
            if packet.ts_ms < window_start_ms or packet.ts_ms > window_end_ms:
                continue
            overlaps_send_window = send_entry.send_start_ms <= packet.ts_ms <= send_entry.send_end_ms
            candidate = (
                0 if overlaps_send_window else 1,
                abs(packet.ts_ms - send_entry.send_start_ms),
                packet_index,
            )
            if best is None or candidate < best:
                best = candidate

        if best is None:
            if strict:
                raise SystemExit(
                    "Failed to match an identity-aware request pcap flow to an accepted run row. "
                    f"row={json.dumps({'target_service': row.get('target_service', ''), 'source_service': row.get('source_service', ''), 'source_ip': source_ip, 'target_ip': target_ip, 'command': row.get('command', ''), 'send_id': row.get('send_id', ''), 'request_window_start_ms': window_start_ms, 'request_window_end_ms': window_end_ms}, separators=(',', ':'))}"
                )
            continue

        _, _, packet_index = best
        request_start_indexes[row_index] = packet_index
        last_request_index_by_identity[identity_key] = packet_index

    if not request_start_indexes:
        return {}

    matches: dict[int, PcapFlow] = {}
    by_tuple: dict[tuple[str, int, str, int], list[tuple[int, int]]] = defaultdict(list)
    for row_index, packet_index in request_start_indexes.items():
        packet = packets[packet_index]
        key = (packet.src_ip, packet.src_port, packet.dst_ip, packet.dst_port)
        by_tuple[key].append((packet_index, row_index))

    for key, entries in by_tuple.items():
        client_ip, client_port, server_ip, target_port = key
        entries.sort()
        for entry_pos, (start_index, row_index) in enumerate(entries):
            next_start_index = entries[entry_pos + 1][0] if entry_pos + 1 < len(entries) else len(packets)
            request_first_ms = None
            request_last_ms = None
            response_first_ms = None
            response_last_ms = None
            request_bytes = 0
            response_bytes = 0
            saw_response_direction = False
            for packet in packets[start_index:next_start_index]:
                if (
                    packet.src_ip == client_ip
                    and packet.src_port == client_port
                    and packet.dst_ip == server_ip
                    and packet.dst_port == target_port
                ):
                    if packet.payload_len > 0:
                        if request_first_ms is None:
                            request_first_ms = packet.ts_ms
                        request_last_ms = packet.ts_ms
                        request_bytes += packet.payload_len
                    continue
                if (
                    packet.src_ip == server_ip
                    and packet.src_port == target_port
                    and packet.dst_ip == client_ip
                    and packet.dst_port == client_port
                ):
                    saw_response_direction = True
                    if response_first_ms is None:
                        response_first_ms = packet.ts_ms
                    response_last_ms = packet.ts_ms
                    if packet.payload_len > 0:
                        response_bytes += packet.payload_len
            if request_first_ms is None:
                continue
            chosen_flow = PcapFlow(
                client_ip=client_ip,
                server_ip=server_ip,
                target_port=target_port,
                client_port=client_port,
                request_first_ms=request_first_ms,
                request_last_ms=request_last_ms or request_first_ms,
                response_first_ms=response_first_ms,
                response_last_ms=response_last_ms,
                request_bytes=request_bytes,
                response_bytes=response_bytes,
                saw_response_direction=saw_response_direction,
            )
            row = dict(ordered_rows[row_index][1])
            if chosen_flow.request_bytes <= 0 and strict:
                raise SystemExit(
                    f"Matched pcap flow is missing request payload bytes for {row.get('command', '')} on port {chosen_flow.target_port}"
                )
            if not row_timeout(row) and not chosen_flow.saw_response_direction and strict:
                raise SystemExit(
                    f"Matched pcap flow is missing response-direction traffic for non-timeout command {row.get('command', '')} on port {chosen_flow.target_port}"
                )
            matches[row_index] = chosen_flow

    if strict:
        missing_rows = [
            row_index
            for row_index, row in ordered_rows
            if row_gds_accept(row) and row_index not in matches
        ]
        if missing_rows:
            raise SystemExit(
                "Failed to build per-command pcap exchanges for accepted run rows. "
                f"Missing row indexes: {missing_rows[:8]}"
            )
    return matches


def choose_events_for_row(
    target_rows: list[tuple[int, dict[str, str]]],
    send_matches: dict[int, SendLogEntry],
    event_entries: list[EventLogEntry],
) -> dict[int, tuple[EventLogEntry | None, EventLogEntry | None, EventLogEntry | None, int, int, list[str]]]:
    # Event attribution intentionally assumes each target node executes one command at a time.
    # The schedule runner serializes per target stream, and this windowing strategy relies on
    # that invariant until the stack exposes a stronger execution correlation key.
    result: dict[int, tuple[EventLogEntry | None, EventLogEntry | None, EventLogEntry | None, int, int, list[str]]] = {}
    event_entries = sorted(event_entries, key=lambda entry: entry.ts_ms)
    for position, (row_index, row) in enumerate(target_rows):
        send_entry = send_matches.get(row_index)
        if send_entry is None or not row_gds_accept(row):
            result[row_index] = (None, None, None, 0, 0, [])
            continue
        request_ts = send_entry.send_start_ms
        next_request_ts = None
        for next_row_index, _ in target_rows[position + 1 :]:
            next_entry = send_matches.get(next_row_index)
            if next_entry is not None and row_gds_accept(_):
                next_request_ts = next_entry.send_start_ms
                break

        window = [
            entry
            for entry in event_entries
            if entry.ts_ms >= request_ts and (next_request_ts is None or entry.ts_ms < next_request_ts)
        ]
        dispatch_event = next((entry for entry in window if entry.name == "cmdDisp.OpCodeDispatched"), None)
        terminal_event = next((entry for entry in window if entry.name in TERMINAL_EVENT_NAMES), None)

        sat_response_event = None
        if dispatch_event is not None:
            for entry in window:
                if entry.ts_ms < dispatch_event.ts_ms:
                    continue
                if entry.name == "cmdDisp.OpCodeDispatched":
                    continue
                sat_response_event = entry
                break
        if sat_response_event is None:
            sat_response_event = terminal_event

        warning_events = 0
        error_events = 0
        lifecycle_event_names: list[str] = []
        for entry in window:
            lifecycle_event_names.append(entry.name)
            if "WARNING" in entry.severity:
                warning_events += 1
            if "ERROR" in entry.severity or entry.name == "cmdDisp.OpCodeError":
                error_events += 1
        result[row_index] = (dispatch_event, sat_response_event, terminal_event, warning_events, error_events, lifecycle_event_names)
    return result


def latest_telemetry_ts(samples: list[TelemetryRecord], now_ms: int) -> int | None:
    latest: int | None = None
    for sample in samples:
        if sample.ts_ms > now_ms:
            break
        latest = sample.ts_ms
    return latest


def packet_base_from_row(row: dict[str, str], index: int) -> dict[str, Any]:
    meta = json.loads(row.get("meta_json", "{}"))
    command = str(row.get("command", ""))
    service = COMMAND_SPECS[command].service if command in COMMAND_SPECS else command.split(".")[0]
    source_service = str(row.get("source_service", "ops"))
    target_service = str(row.get("target_service", "fprime"))
    run_id = row_run_id(row)
    session_prefix = f"run-{run_id:04d}" if run_id >= 0 else source_service
    return {
        "run_id": run_id,
        "episode_id": int(meta.get("episode_id", 0)),
        "episode_label": int(meta.get("class_label", 0)),
        "episode_kind": str(meta.get("class_name", "benign")),
        "session_id": f"{session_prefix}-ep-{int(meta.get('episode_id', 0)):04d}",
        "txn_id": f"{row.get('virtual_day', '0')}-{index:06d}-{source_service}",
        "send_id": row_send_id(row),
        "target_tts_port": int(row.get("target_tts_port", row.get("tts_port", 0)) or 0),
        "target_stream_id": row_target_stream_id(row),
        "target_stream_index": row_target_stream_index(row),
        "service": service,
        "command": command,
        "label": int(meta.get("class_label", 0)),
        "attack_family": str(meta.get("attack_family", "none")),
        "phase": str(meta.get("phase", "")),
        "actor": source_service,
        "actor_role": str(meta.get("actor_role", "ops")),
        "actor_trust": float(meta.get("actor_trust", 1.0)),
        "target_service": target_service,
        "node_service": target_service,
    }


def observation_to_dict(observation: CommandObservation, request_ts: int, final_ts: int, final_observed_on_wire: bool) -> dict[str, Any]:
    row = observation.row
    return {
        "row_index": observation.row_index,
        "run_id": row_run_id(row),
        "send_id": row_send_id(row),
        "target_stream_id": row_target_stream_id(row),
        "target_stream_index": row_target_stream_index(row),
        "command": str(row.get("command", "")),
        "source_service": str(row.get("source_service", "")),
        "source_ip": str(row.get("source_ip", "")),
        "target_service": str(row.get("target_service", "")),
        "target_ip": str(row.get("target_ip", "")),
        "request_wire_seen": observation.request_wire_seen,
        "response_direction_seen": observation.response_direction_seen,
        "op_dispatched_seen": observation.dispatch_event is not None,
        "terminal_event_seen": observation.terminal_event is not None,
        "telemetry_recent": observation.telemetry_recent,
        "final_observed_on_wire": final_observed_on_wire,
        "warning_events": observation.warning_events,
        "error_events": observation.error_events,
        "request_ts_ms": request_ts,
        "dispatch_ts_ms": observation.dispatch_event.ts_ms if observation.dispatch_event is not None else None,
        "sat_response_ts_ms": observation.sat_response_event.ts_ms if observation.sat_response_event is not None else None,
        "terminal_ts_ms": observation.terminal_event.ts_ms if observation.terminal_event is not None else None,
        "final_ts_ms": final_ts,
        "latency_ms_observed": max(0, final_ts - request_ts),
        "event_names": list(observation.lifecycle_event_names),
        "reason": str(row.get("reason", "")),
        "gds_accept": int(row.get("gds_accept", "0") or "0"),
        "sat_success": int(row.get("sat_success", "0") or "0"),
        "timeout": int(row.get("timeout", "0") or "0"),
    }


def build_telemetry_packets(channel_samples: list[TelemetryRecord], *, run_id: int = -1) -> list[dict[str, Any]]:
    packets: list[dict[str, Any]] = []
    for sample in channel_samples:
        packets.append(
            {
                "ts_ms": sample.ts_ms,
                "packet_kind": "telemetry",
                "src": sample.node_service,
                "dst": "gds",
                "service": sample.channel_names[0].split(".", 1)[0] if sample.channel_names else "",
                "command": "",
                "label": 0,
                "attack_family": "none",
                "phase": "telemetry",
                "bytes_on_wire": sample.bytes_on_wire,
                "payload": dict(sample.payload),
                "channel_name": ",".join(sample.channel_names),
                "node_service": sample.node_service,
                "observed_on_wire": 1 if sample.observed_on_wire else 0,
                "ts_source": sample.ts_source,
                "bytes_source": sample.bytes_source,
                "event_name": "",
                "run_id": run_id,
            }
        )
    return packets


def build_packets_from_parsed_sources(
    run_rows: list[dict[str, str]],
    *,
    inventory: dict[str, Any],
    command_entries: list[CommandLogEntry],
    send_entries: list[SendLogEntry],
    event_entries_by_target: dict[str, list[EventLogEntry]],
    telemetry_records_by_target: dict[str, list[TelemetryRecord]],
    pcap_result: PcapParseResult,
    source_artifact_paths: list[str] | None = None,
    capture_interface: str | None = None,
    capture_backend: str | None = None,
    telemetry_recent_threshold_ms: int = TELEMETRY_RECENT_THRESHOLD_MS,
    strict: bool = True,
) -> PacketBuildResult:
    ordered = ordered_run_rows(run_rows)
    send_matches = match_send_entries(ordered, send_entries, strict=strict)
    serialization_violations = collect_serialization_violations(ordered, send_matches)
    if strict and serialization_violations:
        raise SystemExit(
            "Packet reconstruction would violate the serialized-per-target invariant. "
            f"Examples: {json.dumps(serialization_violations[:8], separators=(',', ':'))}"
        )
    command_matches = match_command_entries(ordered, send_matches, command_entries)
    flow_matches = match_pcap_flows(ordered, send_matches, pcap_result.packets, strict=strict)

    event_matches: dict[int, tuple[EventLogEntry | None, EventLogEntry | None, EventLogEntry | None, int, int, list[str]]] = {}
    ordered_by_target: dict[str, list[tuple[int, dict[str, str]]]] = defaultdict(list)
    for row_index, row in ordered:
        ordered_by_target[str(row.get("target_service", ""))].append((row_index, row))
    for target_service, target_rows in ordered_by_target.items():
        event_matches.update(
            choose_events_for_row(
                target_rows,
                send_matches,
                event_entries_by_target.get(target_service, []),
            )
        )

    all_telemetry_records = sorted(
        [sample for samples in telemetry_records_by_target.values() for sample in samples],
        key=lambda sample: (sample.ts_ms, sample.node_service, tuple(sorted(sample.payload))),
    )
    run_ids = {row_run_id(row) for _, row in ordered if row_run_id(row) >= 0}
    telemetry_run_id = next(iter(run_ids)) if len(run_ids) == 1 else -1
    packets = build_telemetry_packets(all_telemetry_records, run_id=telemetry_run_id)
    observations: list[dict[str, Any]] = []
    for packet_index, (row_index, row) in enumerate(ordered):
        packet_base = packet_base_from_row(row, packet_index)
        command = str(row.get("command", ""))
        source_service = str(row.get("source_service", "ops"))
        target_service = str(row.get("target_service", "fprime"))
        raw_args = json.loads(row.get("arguments_json", "[]"))
        request_entry = command_matches.get(row_index)
        send_entry = send_matches.get(row_index)
        request_flow = flow_matches.get(row_index)
        dispatch_event, sat_response_event, terminal_event, warning_events, error_events, lifecycle_event_names = event_matches.get(
            row_index,
            (None, None, None, 0, 0, []),
        )

        accepted = row_gds_accept(row)
        request_ts = row_real_ms(row)
        request_ts_source = "send_row"
        request_bytes = 0
        request_observed_on_wire = 0
        request_wire_fields: dict[str, Any] = {}
        if send_entry is not None:
            request_ts = send_entry.send_start_ms
            request_ts_source = "send_log"
        if accepted:
            if request_flow is not None:
                request_ts = request_flow.request_first_ms
                request_ts_source = "pcap"
                request_bytes = request_flow.request_bytes
                request_observed_on_wire = 1 if request_flow.request_bytes > 0 else 0
                request_wire_fields = {
                    "src_ip": request_flow.client_ip,
                    "dst_ip": request_flow.server_ip,
                    "src_port": request_flow.client_port,
                    "dst_port": request_flow.target_port,
                }
            elif strict:
                raise SystemExit(f"Accepted command {command} is missing both send-log and request-wire anchors")

        provisional_final_ts = request_ts + row_latency_ms(row)
        if terminal_event is not None:
            provisional_final_ts = terminal_event.ts_ms
        latest_target_tm = latest_telemetry_ts(telemetry_records_by_target.get(target_service, []), provisional_final_ts)
        telemetry_recent = latest_target_tm is not None and (provisional_final_ts - latest_target_tm) <= telemetry_recent_threshold_ms

        observation = CommandObservation(
            row_index=row_index,
            row=row,
            request_entry=request_entry,
            request_flow=request_flow,
            dispatch_event=dispatch_event,
            sat_response_event=sat_response_event,
            terminal_event=terminal_event,
            request_wire_seen=request_flow is not None and request_flow.request_bytes > 0,
            response_direction_seen=request_flow is not None and request_flow.saw_response_direction,
            telemetry_recent=telemetry_recent,
            warning_events=warning_events,
            error_events=error_events,
            lifecycle_event_names=lifecycle_event_names,
        )

        packets.append(
            {
                "ts_ms": request_ts,
                "packet_kind": "request",
                "src": source_service,
                "dst": target_service,
                "bytes_on_wire": request_bytes,
                "args": normalize_command_args(command, raw_args),
                "observed_on_wire": request_observed_on_wire,
                "ts_source": request_ts_source,
                "bytes_source": "pcap" if request_observed_on_wire else "none",
                "event_name": "",
                **request_wire_fields,
                **packet_base,
            }
        )

        if dispatch_event is not None:
            packets.append(
                {
                    "ts_ms": dispatch_event.ts_ms,
                    "packet_kind": "uplink",
                    "src": "gds",
                    "dst": target_service,
                    "bytes_on_wire": 0,
                    "observed_on_wire": 0,
                    "ts_source": "event_log",
                    "bytes_source": "none",
                    "event_name": dispatch_event.name,
                    **packet_base,
                }
            )

        if accepted and sat_response_event is not None:
            packets.append(
                {
                    "ts_ms": sat_response_event.ts_ms,
                    "packet_kind": "sat_response",
                    "src": target_service,
                    "dst": "gds",
                    "bytes_on_wire": 0,
                    "response_code": int(row.get("response_code", "0") or "0"),
                    "reason": str(row.get("reason", "")),
                    "sat_success": int(row.get("sat_success", "0") or "0"),
                    "observed_on_wire": 0,
                    "ts_source": "event_log",
                    "bytes_source": "none",
                    "event_name": sat_response_event.name,
                    **packet_base,
                }
            )

        final_ts = request_ts + row_latency_ms(row)
        final_ts_source = "send_row"
        final_bytes = 0
        final_bytes_source = "none"
        final_observed_on_wire = 0
        final_wire_fields: dict[str, Any] = {}
        response_direction_seen = request_flow is not None and request_flow.saw_response_direction
        if request_flow is not None and request_flow.response_first_ms is not None:
            final_ts = request_flow.response_first_ms
            final_ts_source = "pcap"
            final_bytes = request_flow.response_bytes
            final_bytes_source = "pcap"
            final_observed_on_wire = 1
            final_wire_fields = {
                "src_ip": request_flow.server_ip,
                "dst_ip": request_flow.client_ip,
                "src_port": request_flow.target_port,
                "dst_port": request_flow.client_port,
            }
        elif terminal_event is not None:
            final_ts = terminal_event.ts_ms
            final_ts_source = "event_log"
        if strict and accepted and not row_timeout(row) and not response_direction_seen:
            raise SystemExit(
                f"Non-timeout accepted command is missing response-direction traffic: {target_service} {command}"
            )

        packets.append(
            {
                "ts_ms": final_ts,
                "packet_kind": "final",
                "src": target_service,
                "dst": source_service,
                "bytes_on_wire": final_bytes,
                "response_code": int(row.get("response_code", "0") or "0"),
                "gds_accept": int(row.get("gds_accept", "0") or "0"),
                "sat_success": int(row.get("sat_success", "0") or "0"),
                "timeout": int(row.get("timeout", "0") or "0"),
                "reason": str(row.get("reason", "")),
                "response_direction_seen": 1 if response_direction_seen else 0,
                "final_observed_on_wire": final_observed_on_wire,
                "txn_warning_events": warning_events,
                "txn_error_events": error_events,
                "observed_on_wire": final_observed_on_wire,
                "ts_source": final_ts_source,
                "bytes_source": final_bytes_source,
                "event_name": terminal_event.name if terminal_event is not None else "",
                **final_wire_fields,
                **packet_base,
            }
        )

        observations.append(observation_to_dict(observation, request_ts, final_ts, bool(final_observed_on_wire)))

    packets.sort(
        key=lambda packet: (
            int(packet.get("ts_ms", 0)),
            str(packet.get("txn_id", "")),
            PACKET_KIND_ORDER.get(str(packet.get("packet_kind", "")), 99),
            str(packet.get("packet_kind", "")),
            str(packet.get("channel_name", "")),
        )
    )

    packet_kinds = Counter(str(packet.get("packet_kind", "")) for packet in packets)
    ts_sources = Counter(str(packet.get("ts_source", "")) for packet in packets)
    bytes_sources = Counter(str(packet.get("bytes_source", "")) for packet in packets)
    observed_counter = Counter(int(packet.get("observed_on_wire", 0) or 0) for packet in packets)
    request_anchor_sources = Counter()
    for packet in packets:
        if str(packet.get("packet_kind", "")) != "request":
            continue
        ts_source = str(packet.get("ts_source", "fallback"))
        anchor_source = ts_source if ts_source in {"send_log", "pcap", "event_log"} else "fallback"
        request_anchor_sources[anchor_source] += 1
    provenance_summary = {
        "schema_version": SCHEMA_VERSION,
        "capture_backend": capture_backend or "",
        "capture_interface": capture_interface or "",
        "pcap_identity_mode": "bridge_ip_5tuple",
        "event_attribution_mode": "serialized_per_target",
        "target_stream_serialization_invariant": True,
        "serialization_violations": len(serialization_violations),
        "packet_count": len(packets),
        "pcap_packet_count": pcap_result.packet_count,
        "packet_kinds": dict(packet_kinds),
        "ts_sources": dict(ts_sources),
        "bytes_sources": dict(bytes_sources),
        "observed_on_wire": {str(key): value for key, value in observed_counter.items()},
        "telemetry_packet_count": int(packet_kinds.get("telemetry", 0)),
        "command_rows": len(run_rows),
        "send_log_records": len(send_entries),
        "command_log_records": len(command_entries),
        "command_log_audit_matches": len(command_matches),
        "rows_with_send_id": sum(1 for row in run_rows if row_send_id(row)),
        "rows_with_source_ip": sum(1 for row in run_rows if row_source_ip(row)),
        "rows_with_target_ip": sum(1 for row in run_rows if row_target_ip(row)),
        "rows_with_target_stream_id": sum(1 for row in run_rows if row_target_stream_id(row)),
        "packets_with_send_id": sum(1 for packet in packets if str(packet.get("send_id", ""))),
        "packets_with_target_stream_id": sum(1 for packet in packets if str(packet.get("target_stream_id", ""))),
        "pcap_5tuple_matches": len(flow_matches),
        "pcap_fallback_rows": sum(1 for row in run_rows if row_gds_accept(row)) - len(flow_matches),
        "request_anchor_sources": {
            "send_log": int(request_anchor_sources.get("send_log", 0)),
            "pcap": int(request_anchor_sources.get("pcap", 0)),
            "event_log": int(request_anchor_sources.get("event_log", 0)),
            "fallback": int(request_anchor_sources.get("fallback", 0)),
        },
        "accepted_rows": sum(1 for row in run_rows if row_gds_accept(row)),
        "timeout_rows": sum(1 for row in run_rows if row_timeout(row)),
        "observations_with_recent_telemetry": sum(1 for item in observations if bool(item.get("telemetry_recent", False))),
        "telemetry_sources": dict(Counter(record.ts_source for record in all_telemetry_records)),
    }
    finalize_channel_inventory(inventory)
    return PacketBuildResult(
        packets=packets,
        observations=observations,
        channel_inventory=inventory,
        provenance_summary=provenance_summary,
        source_artifact_paths=list(source_artifact_paths or []),
    )


def build_packets_from_real_artifacts(
    run_rows: list[dict[str, str]],
    *,
    command_log_path: Path,
    event_log_paths: dict[str, Path],
    channel_log_paths: dict[str, Path] | None = None,
    telemetry_record_paths: dict[str, Path] | None = None,
    send_log_path: Path | None = None,
    pcap_path: Path,
    capture_interface: str | None = None,
    capture_backend: str | None = None,
    telemetry_recent_threshold_ms: int = TELEMETRY_RECENT_THRESHOLD_MS,
    strict: bool = True,
) -> PacketBuildResult:
    command_entries = parse_command_log(command_log_path)
    send_entries = parse_send_log(send_log_path)
    event_entries_by_target = {
        target_service: parse_event_log(path)
        for target_service, path in event_log_paths.items()
    }
    inventory: dict[str, Any] = {"schema_version": SCHEMA_VERSION}
    telemetry_records_by_target: dict[str, list[TelemetryRecord]] = {}
    if telemetry_record_paths is not None:
        telemetry_records_by_target = {
            target_service: parse_downlink_records(path, target_service, inventory)
            for target_service, path in telemetry_record_paths.items()
        }
    elif channel_log_paths is not None:
        telemetry_records_by_target = {
            target_service: parse_channel_log(path, target_service, inventory)
            for target_service, path in channel_log_paths.items()
        }
    else:
        telemetry_records_by_target = {target_service: [] for target_service in event_log_paths}

    source_artifact_paths = resolve_source_artifact_paths(
        [command_log_path, pcap_path],
        list(event_log_paths.values()),
        list(telemetry_record_paths.values()) if telemetry_record_paths is not None else None,
        list(channel_log_paths.values()) if channel_log_paths is not None else None,
        [send_log_path] if send_log_path is not None else None,
    )
    result = build_packets_from_parsed_sources(
        run_rows,
        inventory=inventory,
        command_entries=command_entries,
        send_entries=send_entries,
        event_entries_by_target=event_entries_by_target,
        telemetry_records_by_target=telemetry_records_by_target,
        pcap_result=parse_pcap_capture(pcap_path),
        source_artifact_paths=source_artifact_paths,
        capture_interface=capture_interface,
        capture_backend=capture_backend,
        telemetry_recent_threshold_ms=telemetry_recent_threshold_ms,
        strict=strict,
    )
    return result
