#!/usr/bin/env python3
"""Probe per-node benign command support and required observability."""

from __future__ import annotations

import json
import re
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.fprime_real.benign_catalog import BenignCommandSample, load_benign_command_samples
from tools.fprime_real.downlink_ingest import decode_runtime_downlink
from tools.fprime_real.packet_fidelity import PcapIncrementalReader, build_packets_from_parsed_sources, build_packets_from_real_artifacts, parse_event_log_line, parse_pcap_capture, send_entry_from_payload
from tools.fprime_real.pcap_capture import DEFAULT_CAPTURE_FILTER, capture_pcap, env_capture_backend, env_capture_interface, list_capture_interfaces, ordered_capture_interfaces, preferred_capture_interface, resolve_capture_backend
from tools.fprime_real.run_fprime_schedule import ExpandedEvent, HOST_READY_PORT_BY_SERVICE, run_one_event, to_container_path, wait_for_port
from tools.fprime_real.runtime_layout import CONTAINER_RUNTIME_ROOT, TARGET_NODE_BY_SERVICE, ensure_runtime_tree, host_command_log_path, host_event_log_path, host_recv_bin_path, host_send_log_path

DEFAULT_DICTIONARY = "/workspace/gds/fprime_project/FlightPair/DualLink/build-artifacts/Linux/dict/DualLinkTopologyAppDictionary.xml"
PROBE_TARGETS = (
    {
        "node_name": "node_a",
        "source_service": "ops_b1",
        "target_service": "fprime_a",
        "target_tts_port": 50050,
    },
    {
        "node_name": "node_b",
        "source_service": "ops_a1",
        "target_service": "fprime_b",
        "target_tts_port": 50050,
    },
)


@dataclass(frozen=True)
class ResolvedIdentityCapture:
    backend: str
    interface: str


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def support_logs_dir() -> str:
    return str(CONTAINER_RUNTIME_ROOT / "cli_logs")


class TextTailReader:
    def __init__(self, path: Path, parser):
        self.path = path
        self.parser = parser
        self.offset = 0
        self.buffer = ""
        self.items: list[Any] = []

    def update(self) -> list[Any]:
        if not self.path.exists():
            return []
        with self.path.open("r", encoding="utf-8", errors="replace") as handle:
            handle.seek(self.offset)
            chunk = handle.read()
            self.offset = handle.tell()
        if not chunk:
            return []
        text = self.buffer + chunk
        lines = text.splitlines(keepends=True)
        complete = lines
        if lines and not lines[-1].endswith("\n"):
            self.buffer = lines[-1]
            complete = lines[:-1]
        else:
            self.buffer = ""
        new_items: list[Any] = []
        for raw_line in complete:
            line = raw_line.strip()
            if not line:
                continue
            item = self.parser(line)
            self.items.append(item)
            new_items.append(item)
        return new_items


def wait_for_runtime_ready(runtime_root: Path, timeout_seconds: float = 45.0) -> None:
    deadline = time.time() + timeout_seconds
    recv_bin_paths = [host_recv_bin_path(runtime_root, target_service) for target_service in TARGET_NODE_BY_SERVICE]
    while time.time() < deadline:
        ready = True
        for path in recv_bin_paths:
            if not path.exists():
                ready = False
                break
            try:
                if path.stat().st_size <= 0:
                    ready = False
                    break
            except OSError:
                ready = False
                break
        if ready:
            return
        time.sleep(0.5)
    raise SystemExit(f"Timed out waiting for run-local telemetry readiness in {runtime_root}")


def sanitize_interface_name(name: str) -> str:
    return re.sub(r"[^A-Za-z0-9_.-]+", "_", name)


def capture_validation_event() -> ExpandedEvent:
    target = PROBE_TARGETS[0]
    return ExpandedEvent(
        absolute_virtual_seconds=-3,
        virtual_day=-3,
        virtual_time="00:00:00",
        virtual_seconds=0,
        source_service=str(target["source_service"]),
        target_service=str(target["target_service"]),
        target_tts_port=int(target["target_tts_port"]),
        command="cmdDisp.CMD_NO_OP",
        arguments=[],
        meta={
            "class_label": 0,
            "class_name": "capture_validation",
            "attack_family": "capture_validation",
            "actor_role": "ops_probe",
            "actor_trust": 1.0,
            "episode_id": -3,
            "phase": "capture_validation",
        },
    )


def event_target_stream_id(event: ExpandedEvent) -> str:
    return f"{event.target_service}:{event.target_tts_port}"


def capture_drain_signature(send_count: int, event_counts: dict[str, int], pcap_packet_count: int) -> tuple[int, int, tuple[tuple[str, int], ...]]:
    return send_count, pcap_packet_count, tuple(sorted(event_counts.items()))


def collect_capture_drain_failures(
    expected_rows: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    present_send_ids: set[str],
) -> list[dict[str, Any]]:
    observations_by_send_id = {
        str(item.get("send_id", "")): item
        for item in observations
        if str(item.get("send_id", ""))
    }
    failures: list[dict[str, Any]] = []
    for row in expected_rows:
        send_id = str(row.get("send_id", ""))
        missing: list[str] = []
        if not send_id or send_id not in present_send_ids:
            missing.append("send_log")
        observation = observations_by_send_id.get(send_id, {})
        accepted = int(row.get("gds_accept", 0) or 0) == 1
        timeout = int(row.get("timeout", 0) or 0) == 1
        if accepted and not bool(observation.get("request_wire_seen", False)):
            missing.append("request_wire")
        if accepted and not timeout and not (
            bool(observation.get("response_direction_seen", False))
            or bool(observation.get("terminal_event_seen", False))
        ):
            missing.append("response_or_terminal")
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
    event_log_paths: dict[str, Path],
    send_log_path: Path,
    pcap_path: Path,
    capture_interface: str,
    timeout_seconds: float,
    quiet_period_seconds: float = 0.5,
) -> None:
    if not expected_rows:
        return
    send_reader = TextTailReader(send_log_path, lambda line: send_entry_from_payload(json.loads(line)))
    event_readers = {
        target_service: TextTailReader(path, parse_event_log_line)
        for target_service, path in event_log_paths.items()
    }
    pcap_reader = PcapIncrementalReader()
    deadline = time.time() + max(10.0, timeout_seconds + 5.0)
    quiet_start: float | None = None
    quiet_signature: tuple[int, int, tuple[tuple[str, int], ...]] | None = None
    while time.time() < deadline:
        send_reader.update()
        for reader in event_readers.values():
            reader.update()
        pcap_reader.update(pcap_path)
        pcap_result = pcap_reader.parse_result()
        packet_result = build_packets_from_parsed_sources(
            expected_rows,
            inventory={"schema_version": "real_fprime_v2"},
            command_entries=[],
            send_entries=list(send_reader.items),
            event_entries_by_target={target_service: list(reader.items) for target_service, reader in event_readers.items()},
            telemetry_records_by_target={target_service: [] for target_service in event_log_paths},
            pcap_result=pcap_result,
            capture_interface=capture_interface,
            strict=False,
        )
        failures = collect_capture_drain_failures(
            expected_rows,
            packet_result.observations,
            {entry.send_id for entry in send_reader.items if entry.send_id},
        )
        signature = capture_drain_signature(
            len(send_reader.items),
            {target_service: len(reader.items) for target_service, reader in event_readers.items()},
            pcap_result.packet_count,
        )
        if not failures:
            if quiet_signature != signature:
                quiet_signature = signature
                quiet_start = time.time()
            elif quiet_start is not None and (time.time() - quiet_start) >= quiet_period_seconds:
                return
        else:
            quiet_signature = None
            quiet_start = None
        time.sleep(0.2)

    packet_result = build_packets_from_parsed_sources(
        expected_rows,
        inventory={"schema_version": "real_fprime_v2"},
        command_entries=[],
        send_entries=list(send_reader.items),
        event_entries_by_target={target_service: list(reader.items) for target_service, reader in event_readers.items()},
        telemetry_records_by_target={target_service: [] for target_service in event_log_paths},
        pcap_result=pcap_reader.parse_result(),
        capture_interface=capture_interface,
        strict=False,
    )
    failures = collect_capture_drain_failures(
        expected_rows,
        packet_result.observations,
        {entry.send_id for entry in send_reader.items if entry.send_id},
    )
    raise SystemExit(
        "Timed out draining capture before shutdown. "
        f"Pending: {json.dumps(failures[:8], separators=(',', ':'))}"
    )


def validate_capture_interface(
    compose_file: Path,
    dictionary_path: str,
    timeout_seconds: float,
    sender_script: str,
    runtime_root: Path,
    backend: str,
    interface: str,
) -> tuple[bool, dict[str, Any]]:
    validation_event = capture_validation_event()
    pcap_path = runtime_root / "pcap" / f"identity_probe_{sanitize_interface_name(interface)}.pcap"
    _, event_log_paths = real_log_paths(runtime_root)
    try:
        with capture_pcap(
            pcap_path,
            interface=interface,
            filter_expr=DEFAULT_CAPTURE_FILTER,
            backend=backend,
        ) as capture:
            row = run_one_event(
                compose_file=compose_file,
                dictionary=dictionary_path,
                timeout_seconds=max(5.0, timeout_seconds),
                sender_script=sender_script,
                event=validation_event,
                logs_dir=support_logs_dir(),
                target_stream_id=event_target_stream_id(validation_event),
                target_stream_index=0,
            )
            wait_for_capture_drain(
                [row],
                event_log_paths=event_log_paths,
                send_log_path=host_send_log_path(runtime_root),
                pcap_path=pcap_path,
                capture_interface=capture.interface,
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
    target_port = int(row.get("target_tts_port", 0) or 0)
    pcap_result = parse_pcap_capture(pcap_path)
    matched = any(
        packet.payload_len > 0
        and packet.src_ip == source_ip
        and packet.dst_ip == target_ip
        and packet.dst_port == target_port
        for packet in pcap_result.packets
    )
    detail = {
        "interface": interface,
        "backend": backend,
        "source_ip": source_ip,
        "target_ip": target_ip,
        "target_tts_port": target_port,
        "pcap_path": str(pcap_path.resolve()),
        "matched_identity_request": matched,
        "packet_count": pcap_result.packet_count,
    }
    if not matched:
        return False, detail
    return True, detail


def resolve_identity_capture_target(
    repo_root: Path,
    compose_file: Path,
    *,
    timeout_seconds: float,
    runtime_root: Path,
    dictionary: str = DEFAULT_DICTIONARY,
) -> ResolvedIdentityCapture:
    sender_script = "/workspace/tools/fprime_real/send_fprime_events.py"
    dictionary_path = dictionary if dictionary.startswith("/workspace") else to_container_path(Path(dictionary), repo_root)
    capture_backend = resolve_capture_backend(env_capture_backend())
    ordered_interfaces = ordered_capture_interfaces(
        list_capture_interfaces(capture_backend),
        env_capture_interface(),
        preferred_capture_interface(capture_backend),
    )
    attempts: list[dict[str, Any]] = []
    for interface in ordered_interfaces:
        valid, detail = validate_capture_interface(
            compose_file=compose_file,
            dictionary_path=dictionary_path,
            timeout_seconds=timeout_seconds,
            sender_script=sender_script,
            runtime_root=runtime_root,
            backend=capture_backend,
            interface=interface,
        )
        attempts.append(detail)
        if valid:
            return ResolvedIdentityCapture(backend=capture_backend, interface=interface)
    raise SystemExit(
        "Could not resolve a non-loopback capture interface that preserves sender identity. "
        f"Tried: {json.dumps(attempts, separators=(',', ':'))}. "
        "Set FPRIME_CAPTURE_BACKEND and FPRIME_CAPTURE_INTERFACE to a supported identity-bearing capture path and retry."
    )


def resolve_identity_capture_interface(
    repo_root: Path,
    compose_file: Path,
    *,
    timeout_seconds: float,
    runtime_root: Path,
    dictionary: str = DEFAULT_DICTIONARY,
) -> str:
    return resolve_identity_capture_target(
        repo_root,
        compose_file,
        timeout_seconds=timeout_seconds,
        runtime_root=runtime_root,
        dictionary=dictionary,
    ).interface


def observation_requirements(sample: BenignCommandSample) -> dict[str, bool]:
    required = sample.required_observability
    return {
        "request_wire_seen": bool(required.request_wire),
        "op_dispatched_seen": bool(required.op_dispatched),
        "terminal_event_seen": bool(required.terminal_event),
        "telemetry_recent": bool(required.telemetry_recent),
    }


def missing_required_observability(observation: dict[str, Any], sample: BenignCommandSample) -> list[str]:
    missing: list[str] = []
    for key, required_value in observation_requirements(sample).items():
        if required_value and not bool(observation.get(key, False)):
            missing.append(key)
    return missing


def is_supported_row(row: dict[str, Any]) -> bool:
    return (
        int(row.get("gds_accept", 0) or 0) == 1
        and int(row.get("sat_success", 0) or 0) == 1
        and int(row.get("timeout", 0) or 0) == 0
        and str(row.get("reason", "")) == "completed"
    )


def observability_ok(observation: dict[str, Any], sample: BenignCommandSample) -> bool:
    return not missing_required_observability(observation, sample)


def is_supported_result(row: dict[str, Any], observation: dict[str, Any], sample: BenignCommandSample) -> bool:
    return is_supported_row(row) and observability_ok(observation, sample)


def summarize_probe_result(sample: BenignCommandSample, row: dict[str, Any], observation: dict[str, Any]) -> dict[str, Any]:
    return {
        "command": sample.command,
        "arguments": list(sample.arguments),
        "required_observability": observation_requirements(sample),
        "supported": is_supported_result(row, observation, sample),
        "gds_accept": int(row.get("gds_accept", 0) or 0),
        "sat_success": int(row.get("sat_success", 0) or 0),
        "timeout": int(row.get("timeout", 0) or 0),
        "response_code": int(row.get("response_code", 0) or 0),
        "reason": str(row.get("reason", "")),
        "event_names_json": str(row.get("event_names_json", "")),
        "send_exception": str(row.get("send_exception", "")),
        "observed_events_json": str(row.get("observed_events_json", "")),
        "latency_ms": int(float(row.get("latency_ms", 0) or 0)),
        "latency_baseline_ms": int(observation.get("latency_ms_observed", 0) or 0),
        "request_wire_seen": bool(observation.get("request_wire_seen", False)),
        "response_direction_seen": bool(observation.get("response_direction_seen", False)),
        "op_dispatched_seen": bool(observation.get("op_dispatched_seen", False)),
        "terminal_event_seen": bool(observation.get("terminal_event_seen", False)),
        "telemetry_recent": bool(observation.get("telemetry_recent", False)),
        "request_ts_ms": observation.get("request_ts_ms"),
        "dispatch_ts_ms": observation.get("dispatch_ts_ms"),
        "terminal_ts_ms": observation.get("terminal_ts_ms"),
        "final_ts_ms": observation.get("final_ts_ms"),
        "missing_observability": missing_required_observability(observation, sample),
    }


def collect_unsupported_results(matrix: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for node_data in matrix.get("nodes", {}).values():
        for result in node_data.get("results", []):
            if result.get("supported", False):
                continue
            failures.append(
                {
                    "target_service": node_data.get("target_service", ""),
                    "source_service": node_data.get("source_service", ""),
                    "target_tts_port": node_data.get("target_tts_port", 0),
                    "command": result.get("command", ""),
                    "reason": result.get("reason", ""),
                    "request_wire_seen": bool(result.get("request_wire_seen", False)),
                    "op_dispatched_seen": bool(result.get("op_dispatched_seen", False)),
                    "terminal_event_seen": bool(result.get("terminal_event_seen", False)),
                    "telemetry_recent": bool(result.get("telemetry_recent", False)),
                    "send_exception": result.get("send_exception", ""),
                }
            )
    return failures


def assert_nominal_support(matrix: dict[str, Any]) -> None:
    failures = collect_unsupported_results(matrix)
    if not failures:
        return
    raise SystemExit(
        "Nominal support preflight failed; at least one benign command is unsupported or missing required observability. "
        f"Examples: {json.dumps(failures[:6], separators=(',', ':'))}"
    )


def sample_map(manifest_path: Path | None = None) -> dict[str, BenignCommandSample]:
    return {sample.command: sample for sample in load_benign_command_samples(manifest_path)}


def actual_run_intent_context(row: dict[str, Any]) -> str:
    try:
        meta = json.loads(row.get("meta_json", "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        meta = {}
    if not isinstance(meta, dict):
        meta = {}
    explicit = str(meta.get("intent_context", "")).strip()
    if explicit:
        return explicit
    return "benign_clean"


def summarize_actual_run_row(
    sample: BenignCommandSample | None,
    row_index: int,
    row: dict[str, Any],
    observation: dict[str, Any],
) -> dict[str, Any]:
    missing_flags = [] if sample is None else missing_required_observability(observation, sample)
    observability_passed = False if sample is None else observability_ok(observation, sample)
    clean_success = is_supported_row(row)
    supported = False if sample is None else clean_success and observability_passed
    return {
        "row_index": row_index,
        "send_id": str(row.get("send_id", "")),
        "source_service": str(row.get("source_service", "")),
        "target_service": str(row.get("target_service", "")),
        "target_tts_port": int(row.get("target_tts_port", 0) or 0),
        "command": str(row.get("command", "")),
        "reason": str(row.get("reason", "")),
        "intent_context": actual_run_intent_context(row),
        "gds_accept": int(row.get("gds_accept", 0) or 0),
        "sat_success": int(row.get("sat_success", 0) or 0),
        "timeout": int(row.get("timeout", 0) or 0),
        "supported": supported,
        "observability_passed": observability_passed,
        "clean_success": clean_success,
        "manifest_known": sample is not None,
        "request_wire_seen": bool(observation.get("request_wire_seen", False)),
        "op_dispatched_seen": bool(observation.get("op_dispatched_seen", False)),
        "terminal_event_seen": bool(observation.get("terminal_event_seen", False)),
        "telemetry_recent": bool(observation.get("telemetry_recent", False)),
        "response_direction_seen": bool(observation.get("response_direction_seen", False)),
        "missing_observability": missing_flags,
    }


def build_actual_run_observability_report(
    run_rows: list[dict[str, Any]],
    observations: list[dict[str, Any]],
    *,
    manifest_path: Path | None = None,
) -> dict[str, Any]:
    benign_samples = sample_map(manifest_path)
    observations_by_row_index = {int(item["row_index"]): item for item in observations}
    results: list[dict[str, Any]] = []
    intent_context_count: dict[str, int] = {}
    for row_index, row in enumerate(run_rows):
        meta = json.loads(row.get("meta_json", "{}"))
        if int(meta.get("class_label", -1)) != 0:
            continue
        sample = benign_samples.get(str(row.get("command", "")))
        observation = observations_by_row_index.get(row_index, {})
        summary_row = summarize_actual_run_row(sample, row_index, row, observation)
        intent_context = str(summary_row.get("intent_context", "benign_clean"))
        intent_context_count[intent_context] = int(intent_context_count.get(intent_context, 0)) + 1
        results.append(summary_row)

    summary = {
        "benign_rows": len(results),
        "manifest_known_rows": sum(1 for item in results if bool(item.get("manifest_known", False))),
        "observability_passed_rows": sum(1 for item in results if bool(item.get("observability_passed", False))),
        "observability_failed_rows": sum(1 for item in results if not bool(item.get("observability_passed", False))),
        "clean_success_rows": sum(1 for item in results if bool(item.get("clean_success", False))),
        "nonclean_rows": sum(1 for item in results if not bool(item.get("clean_success", False))),
        "intent_context_rows": dict(intent_context_count),
    }
    return {
        "generated_at": utc_now(),
        "schema_version": "real_fprime_v2",
        "observability_definition": {
            "manifest_known": True,
            "required_observability": {
                "request_wire_seen": True,
                "op_dispatched_seen": True,
                "terminal_event_seen": True,
                "telemetry_recent": True,
            },
        },
        "clean_success_definition": {
            "gds_accept": 1,
            "sat_success": 1,
            "timeout": 0,
            "reason": "completed",
        },
        "summary": summary,
        "rows": results,
    }


def collect_actual_run_failures(report: dict[str, Any]) -> list[dict[str, Any]]:
    failures: list[dict[str, Any]] = []
    for result in report.get("rows", []):
        if bool(result.get("manifest_known", False)) and bool(result.get("observability_passed", False)):
            continue
        failures.append(
            {
                "source_service": result.get("source_service", ""),
                "target_service": result.get("target_service", ""),
                "command": result.get("command", ""),
                "reason": result.get("reason", ""),
                "missing_observability": list(result.get("missing_observability", [])),
                "manifest_known": bool(result.get("manifest_known", False)),
                "intent_context": result.get("intent_context", ""),
            }
        )
    return failures


def assert_actual_run_observability(report: dict[str, Any]) -> None:
    failures = collect_actual_run_failures(report)
    if not failures:
        return
    raise SystemExit(
        "Captured benign run is missing required observability or manifest coverage. "
        f"Examples: {json.dumps(failures[:6], separators=(',', ':'))}"
    )


def build_probe_rows(samples: tuple[BenignCommandSample, ...]) -> list[ExpandedEvent]:
    rows: list[ExpandedEvent] = []
    for target_index, target in enumerate(PROBE_TARGETS):
        for command_index, sample in enumerate(samples):
            virtual_seconds = target_index * 100 + command_index
            rows.append(
                ExpandedEvent(
                    absolute_virtual_seconds=virtual_seconds,
                    virtual_day=-1,
                    virtual_time=f"00:{(virtual_seconds // 60) % 60:02d}:{virtual_seconds % 60:02d}",
                    virtual_seconds=virtual_seconds,
                    source_service=str(target["source_service"]),
                    target_service=str(target["target_service"]),
                    target_tts_port=int(target["target_tts_port"]),
                    command=sample.command,
                    arguments=list(sample.arguments),
                    meta={
                        "class_label": 0,
                        "class_name": "support_probe",
                        "attack_family": "support_preflight",
                        "actor_role": "ops_probe",
                        "actor_trust": 1.0,
                        "episode_id": -1,
                        "phase": "support",
                    },
                )
            )
    return rows


def warmup_runtime_targets(compose_file: Path, dictionary_path: str, timeout_seconds: float, sender_script: str) -> None:
    for index, target in enumerate(PROBE_TARGETS):
        warmup_event = ExpandedEvent(
            absolute_virtual_seconds=index,
            virtual_day=-2,
            virtual_time=f"00:00:0{index}",
            virtual_seconds=index,
            source_service=str(target["source_service"]),
            target_service=str(target["target_service"]),
            target_tts_port=int(target["target_tts_port"]),
            command="cmdDisp.CMD_NO_OP",
            arguments=[],
            meta={
                "class_label": 0,
                "class_name": "support_warmup",
                "attack_family": "support_warmup",
                "actor_role": "ops_probe",
                "actor_trust": 1.0,
                "episode_id": -2,
                "phase": "warmup",
            },
        )
        try:
            run_one_event(
                compose_file=compose_file,
                dictionary=dictionary_path,
                timeout_seconds=max(5.0, timeout_seconds),
                sender_script=sender_script,
                event=warmup_event,
                logs_dir=support_logs_dir(),
                target_stream_id=event_target_stream_id(warmup_event),
                target_stream_index=index,
            )
        except SystemExit:
            continue


def real_log_paths(runtime_root: Path) -> tuple[Path, dict[str, Path]]:
    command_log_path = host_command_log_path(runtime_root)
    event_log_paths = {
        target_service: host_event_log_path(runtime_root, target_service)
        for target_service in TARGET_NODE_BY_SERVICE
    }
    return command_log_path, event_log_paths


def run_nominal_support_probe(
    repo_root: Path,
    compose_file: Path,
    timeout_seconds: float,
    runtime_root: Path,
    *,
    capture_backend: str,
    capture_interface: str,
    manifest_path: Path | None = None,
    dictionary: str = DEFAULT_DICTIONARY,
) -> dict[str, Any]:
    ensure_runtime_tree(runtime_root)
    sender_script = "/workspace/tools/fprime_real/send_fprime_events.py"
    dictionary_path = dictionary if dictionary.startswith("/workspace") else to_container_path(Path(dictionary), repo_root)
    samples = load_benign_command_samples(manifest_path)

    for target in PROBE_TARGETS:
        host_port = HOST_READY_PORT_BY_SERVICE.get(str(target["target_service"]))
        if host_port is None:
            raise SystemExit(f"Missing published host readiness port for target service {target['target_service']}")
        if not wait_for_port("127.0.0.1", host_port, timeout_seconds=45.0):
            raise SystemExit(f"Timed out waiting for published host port {host_port} to accept connections for {target['target_service']}")
    wait_for_runtime_ready(runtime_root)
    warmup_runtime_targets(compose_file, dictionary_path, timeout_seconds, sender_script)

    probe_rows = build_probe_rows(samples)
    sender_rows: list[dict[str, Any]] = []
    pcap_path = runtime_root / "pcap" / "support_probe.pcap"
    command_log_path, event_log_paths = real_log_paths(runtime_root)
    with capture_pcap(pcap_path, interface=capture_interface, backend=capture_backend) as capture:
        for event in probe_rows:
            sender_rows.append(
                run_one_event(
                    compose_file=compose_file,
                    dictionary=dictionary_path,
                    timeout_seconds=timeout_seconds,
                    sender_script=sender_script,
                    event=event,
                    logs_dir=support_logs_dir(),
                    target_stream_id=event_target_stream_id(event),
                    target_stream_index=len(
                        [
                            row
                            for row in sender_rows
                            if str(row.get("target_stream_id", "")) == event_target_stream_id(event)
                        ]
                    ),
                )
            )
        wait_for_capture_drain(
            sender_rows,
            event_log_paths=event_log_paths,
            send_log_path=host_send_log_path(runtime_root),
            pcap_path=pcap_path,
            capture_interface=capture.interface,
            timeout_seconds=timeout_seconds,
        )
    decoded_paths = decode_runtime_downlink(
        repo_root,
        compose_file,
        runtime_root,
        dictionary_path=dictionary_path,
    )

    packet_result = build_packets_from_real_artifacts(
        sender_rows,
        command_log_path=command_log_path,
        event_log_paths=event_log_paths,
        telemetry_record_paths=decoded_paths,
        send_log_path=host_send_log_path(runtime_root),
        pcap_path=pcap_path,
        capture_interface=capture_interface,
        capture_backend=capture_backend,
        strict=False,
    )
    observations_by_row_index = {int(item["row_index"]): item for item in packet_result.observations}

    nodes: dict[str, Any] = {}
    row_cursor = 0
    for target in PROBE_TARGETS:
        results: list[dict[str, Any]] = []
        for sample in samples:
            row = sender_rows[row_cursor]
            observation = observations_by_row_index.get(row_cursor, {})
            results.append(summarize_probe_result(sample, row, observation))
            row_cursor += 1
        nodes[str(target["target_service"])] = {
            "node_name": str(target["node_name"]),
            "source_service": str(target["source_service"]),
            "target_service": str(target["target_service"]),
            "target_tts_port": int(target["target_tts_port"]),
            "results": results,
        }

    matrix = {
        "generated_at": utc_now(),
        "schema_version": "real_fprime_v2",
        "runtime_root": str(runtime_root.resolve()),
        "traffic_pcap": str(pcap_path.resolve()),
        "capture_backend": capture_backend,
        "capture_interface": capture_interface,
        "support_definition": {
            "gds_accept": 1,
            "sat_success": 1,
            "timeout": 0,
            "reason": "completed",
            "required_observability": {
                "request_wire_seen": True,
                "op_dispatched_seen": True,
                "terminal_event_seen": True,
                "telemetry_recent": True,
            },
        },
        "channel_inventory": packet_result.channel_inventory,
        "provenance_summary": packet_result.provenance_summary,
        "nodes": nodes,
    }
    matrix["unsupported"] = collect_unsupported_results(matrix)
    return matrix
