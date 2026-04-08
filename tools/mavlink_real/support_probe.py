#!/usr/bin/env python3
"""Validate MAVLink capture identity and actual-run observability."""

from __future__ import annotations

import json
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any

from tools.mavlink_real.command_catalog import COMMAND_SPECS, TARGET_ENDPOINT, TARGET_SERVICE, request_autopilot_capabilities_payload
from tools.mavlink_real.packet_fidelity import PacketBuildResult, build_packets_from_real_artifacts, parse_pcap_capture
from tools.mavlink_real.pcap_capture import (
    DEFAULT_CAPTURE_FILTER,
    capture_pcap,
    env_capture_backend,
    env_capture_interface,
    list_capture_interfaces,
    ordered_capture_interfaces,
    preferred_capture_interface,
    resolve_capture_backend,
)
from tools.mavlink_real.run_mavlink_schedule import ExpandedEvent, run_one_event
from tools.mavlink_real.runtime_layout import host_capture_pcap_path


@dataclass(frozen=True)
class ResolvedIdentityCapture:
    backend: str
    interface: str


def capture_validation_event() -> ExpandedEvent:
    return ExpandedEvent(
        absolute_virtual_seconds=-1,
        virtual_day=-1,
        virtual_time="00:00:00",
        virtual_seconds=0,
        source_service="ops_primary",
        target_service=TARGET_SERVICE,
        target_endpoint=TARGET_ENDPOINT,
        command="REQUEST_AUTOPILOT_CAPABILITIES",
        command_family=next(spec.canonical_command_family for spec in COMMAND_SPECS if spec.raw_command_name == "REQUEST_AUTOPILOT_CAPABILITIES"),
        arguments=request_autopilot_capabilities_payload(),
        meta={
            "class_label": 0,
            "class_name": "capture_validation",
            "attack_family": "capture_validation",
            "actor_role": "ops_probe",
            "actor_trust": 1.0,
            "episode_id": -1,
            "phase": "capture_validation",
            "run_id": -1,
        },
    )


def capture_drain_signature(packet_result: PacketBuildResult, *, pcap_size: int) -> tuple[int, int, int, int, int]:
    return (
        pcap_size,
        len(packet_result.packets),
        len(packet_result.transactions),
        sum(1 for item in packet_result.observations if bool(item.get("request_wire_seen"))),
        sum(1 for item in packet_result.observations if bool(item.get("response_direction_seen"))),
    )


def collect_capture_drain_failures(expected_rows: list[dict[str, Any]], observations: list[dict[str, Any]]) -> list[dict[str, Any]]:
    observations_by_send_id = {
        str(item.get("send_id", "")): item
        for item in observations
        if str(item.get("send_id", ""))
    }
    failures: list[dict[str, Any]] = []
    for row in expected_rows:
        send_id = str(row.get("send_id", ""))
        observation = observations_by_send_id.get(send_id, {})
        missing: list[str] = []
        if not bool(observation.get("request_wire_seen", False)):
            missing.append("request_wire_seen")
        if int(row.get("timeout", 0) or 0) != 1 and not bool(observation.get("response_direction_seen", False)):
            missing.append("response_direction_seen")
        if not bool(observation.get("telemetry_recent", False)):
            missing.append("telemetry_recent")
        if not bool(observation.get("state_snapshot_seen", False)):
            missing.append("state_snapshot_seen")
        if missing:
            failures.append(
                {
                    "send_id": send_id,
                    "command": str(row.get("command", "")),
                    "source_service": str(row.get("source_service", "")),
                    "target_service": str(row.get("target_service", "")),
                    "missing": missing,
                }
            )
    return failures


def wait_for_capture_drain(
    expected_rows: list[dict[str, Any]],
    *,
    pcap_path: Path,
    capture_interface: str,
    capture_backend: str | None = None,
    source_artifact_paths: list[str] | None = None,
    timeout_seconds: float,
    quiet_period_seconds: float = 0.5,
) -> PacketBuildResult:
    deadline = time.time() + max(10.0, timeout_seconds + 5.0)
    quiet_start: float | None = None
    quiet_signature: tuple[int, int, int, int, int] | None = None
    while time.time() < deadline:
        if not pcap_path.exists() or pcap_path.stat().st_size <= 0:
            time.sleep(0.2)
            continue
        packet_result = build_packets_from_real_artifacts(
            expected_rows,
            pcap_path=pcap_path,
            capture_interface=capture_interface,
            capture_backend=capture_backend,
            source_artifact_paths=source_artifact_paths,
            strict=False,
        )
        failures = collect_capture_drain_failures(expected_rows, packet_result.observations)
        signature = capture_drain_signature(packet_result, pcap_size=pcap_path.stat().st_size)
        if not failures:
            if quiet_signature != signature:
                quiet_signature = signature
                quiet_start = time.time()
            elif quiet_start is not None and (time.time() - quiet_start) >= quiet_period_seconds:
                return packet_result
        else:
            quiet_signature = None
            quiet_start = None
        time.sleep(0.2)
    if not pcap_path.exists() or pcap_path.stat().st_size <= 0:
        raise SystemExit(f"Timed out draining MAVLink capture before shutdown: empty pcap at {pcap_path}")
    packet_result = build_packets_from_real_artifacts(
        expected_rows,
        pcap_path=pcap_path,
        capture_interface=capture_interface,
        capture_backend=capture_backend,
        source_artifact_paths=source_artifact_paths,
        strict=False,
    )
    failures = collect_capture_drain_failures(expected_rows, packet_result.observations)
    raise SystemExit(
        "Timed out draining MAVLink capture before shutdown. "
        f"Pending: {json.dumps(failures[:8], separators=(',', ':'))}"
    )


def validate_capture_interface(
    compose_file: Path,
    *,
    timeout_seconds: float,
    runtime_root: Path,
    backend: str,
    interface: str,
) -> tuple[bool, dict[str, Any]]:
    validation_event = capture_validation_event()
    pcap_path = host_capture_pcap_path(runtime_root, f"identity_probe_{interface}")
    sender_script = "/workspace/tools/mavlink_real/send_mavlink_events.py"
    try:
        with capture_pcap(
            pcap_path,
            interface=interface,
            filter_expr=DEFAULT_CAPTURE_FILTER,
            backend=backend,
        ) as capture:
            row = run_one_event(
                compose_file=compose_file,
                timeout_seconds=max(5.0, timeout_seconds),
                sender_script=sender_script,
                event=validation_event,
                target_stream_id=f"{validation_event.target_service}@{validation_event.target_endpoint}",
                target_stream_index=0,
            )
            packet_result = wait_for_capture_drain(
                [row],
                pcap_path=pcap_path,
                capture_interface=capture.interface,
                capture_backend=backend,
                source_artifact_paths=[str(pcap_path.resolve())],
                timeout_seconds=max(5.0, timeout_seconds),
            )
    except SystemExit as exc:
        return False, {
            "interface": interface,
            "backend": backend,
            "reason": str(exc),
        }

    source_ip = str(row.get("source_ip", ""))
    target_ip = str(row.get("target_ip", ""))
    pcap_packets = parse_pcap_capture(pcap_path)
    matched = any(
        packet.src_ip == source_ip
        and packet.dst_ip == target_ip
        and packet.dst_port == 5760
        and packet.payload
        for packet in pcap_packets
    )
    detail = {
        "interface": interface,
        "backend": backend,
        "source_ip": source_ip,
        "target_ip": target_ip,
        "pcap_path": str(pcap_path.resolve()),
        "matched_identity_request": matched,
        "packet_count": len(pcap_packets),
        "observation_count": len(packet_result.observations),
    }
    return matched, detail


def resolve_identity_capture_target(
    repo_root: Path,
    compose_file: Path,
    *,
    timeout_seconds: float,
    runtime_root: Path,
) -> ResolvedIdentityCapture:
    del repo_root
    capture_backend = resolve_capture_backend(env_capture_backend())
    ordered_interfaces = ordered_capture_interfaces(
        list_capture_interfaces(capture_backend),
        env_capture_interface(),
        preferred_capture_interface(capture_backend),
    )
    attempts: list[dict[str, Any]] = []
    for interface in ordered_interfaces:
        valid, detail = validate_capture_interface(
            compose_file,
            timeout_seconds=timeout_seconds,
            runtime_root=runtime_root,
            backend=capture_backend,
            interface=interface,
        )
        attempts.append(detail)
        if valid:
            return ResolvedIdentityCapture(backend=capture_backend, interface=interface)
    raise SystemExit(
        "Could not resolve a non-loopback MAVLink capture interface that preserves sender identity. "
        f"Tried: {json.dumps(attempts, separators=(',', ':'))}. "
        "Set MAVLINK_CAPTURE_BACKEND and MAVLINK_CAPTURE_INTERFACE to a supported identity-bearing capture path and retry."
    )


def actual_run_intent_context(row: dict[str, Any]) -> str:
    try:
        meta = json.loads(row.get("meta_json", "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        meta = {}
    if not isinstance(meta, dict):
        meta = {}
    explicit = str(meta.get("intent_context", "")).strip()
    return explicit or "benign_clean"


def is_clean_success(row: dict[str, Any]) -> bool:
    return (
        int(row.get("gds_accept", 0) or 0) == 1
        and int(row.get("sat_success", 0) or 0) == 1
        and int(row.get("timeout", 0) or 0) == 0
        and str(row.get("reason", "")) == "completed"
    )


def build_actual_run_observability_report(
    run_rows: list[dict[str, Any]],
    observations: list[dict[str, Any]],
) -> dict[str, Any]:
    observation_by_row_index = {int(item.get("row_index", -1)): item for item in observations}
    supported_commands = {spec.raw_command_name for spec in COMMAND_SPECS}
    rows: list[dict[str, Any]] = []
    intent_context_rows: dict[str, int] = {}
    for row_index, row in enumerate(run_rows):
        meta = {}
        try:
            meta = json.loads(row.get("meta_json", "{}"))
        except (TypeError, ValueError, json.JSONDecodeError):
            meta = {}
        if not isinstance(meta, dict) or int(meta.get("class_label", -1)) != 0:
            continue
        observation = observation_by_row_index.get(row_index, {})
        missing: list[str] = []
        for field_name in ("request_wire_seen", "response_direction_seen", "telemetry_recent", "state_snapshot_seen"):
            if not bool(observation.get(field_name, False)):
                missing.append(field_name)
        intent_context = actual_run_intent_context(row)
        intent_context_rows[intent_context] = int(intent_context_rows.get(intent_context, 0)) + 1
        rows.append(
            {
                "row_index": row_index,
                "send_id": str(row.get("send_id", "")),
                "source_service": str(row.get("source_service", "")),
                "target_service": str(row.get("target_service", "")),
                "command": str(row.get("command", "")),
                "reason": str(row.get("reason", "")),
                "intent_context": intent_context,
                "manifest_known": str(row.get("command", "")) in supported_commands,
                "clean_success": is_clean_success(row),
                "observability_passed": not missing,
                "request_wire_seen": bool(observation.get("request_wire_seen", False)),
                "response_direction_seen": bool(observation.get("response_direction_seen", False)),
                "telemetry_recent": bool(observation.get("telemetry_recent", False)),
                "state_snapshot_seen": bool(observation.get("state_snapshot_seen", False)),
                "missing_observability": missing,
            }
        )

    return {
        "schema_version": "real_mavlink_v1",
        "observability_definition": {
            "manifest_known": True,
            "required_observability": {
                "request_wire_seen": True,
                "response_direction_seen": True,
                "telemetry_recent": True,
                "state_snapshot_seen": True,
            },
        },
        "clean_success_definition": {
            "gds_accept": 1,
            "sat_success": 1,
            "timeout": 0,
            "reason": "completed",
        },
        "summary": {
            "benign_rows": len(rows),
            "manifest_known_rows": sum(1 for row in rows if bool(row.get("manifest_known", False))),
            "observability_passed_rows": sum(1 for row in rows if bool(row.get("observability_passed", False))),
            "observability_failed_rows": sum(1 for row in rows if not bool(row.get("observability_passed", False))),
            "clean_success_rows": sum(1 for row in rows if bool(row.get("clean_success", False))),
            "nonclean_rows": sum(1 for row in rows if not bool(row.get("clean_success", False))),
            "intent_context_rows": dict(intent_context_rows),
        },
        "rows": rows,
    }


def collect_actual_run_failures(report: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for row in report.get("rows", []):
        if bool(row.get("manifest_known", False)) and bool(row.get("observability_passed", False)):
            continue
        failures.append(
            {
                "source_service": row.get("source_service", ""),
                "target_service": row.get("target_service", ""),
                "command": row.get("command", ""),
                "reason": row.get("reason", ""),
                "missing_observability": list(row.get("missing_observability", [])),
                "manifest_known": bool(row.get("manifest_known", False)),
                "intent_context": row.get("intent_context", ""),
            }
        )
    return failures


def assert_actual_run_observability(report: dict[str, Any]) -> None:
    failures = collect_actual_run_failures(report)
    if not failures:
        return
    raise SystemExit(
        "Captured MAVLink benign run is missing required observability or manifest coverage. "
        f"Examples: {json.dumps(failures[:6], separators=(',', ':'))}"
    )
