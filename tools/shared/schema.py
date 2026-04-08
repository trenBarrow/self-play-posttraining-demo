from __future__ import annotations

import copy
import json
import numbers
from pathlib import Path
from typing import Any, Iterable

RAW_PACKET_SCHEMA_VERSION = "raw_packet.v1"
RAW_TRANSACTION_SCHEMA_VERSION = "raw_transaction.v1"

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = REPO_ROOT / "schemas"
RAW_PACKET_SCHEMA_PATH = SCHEMA_DIR / "raw_packet.schema.json"
RAW_TRANSACTION_SCHEMA_PATH = SCHEMA_DIR / "raw_transaction.schema.json"


class RawArtifactValidationError(ValueError):
    """Raised when a raw artifact record does not satisfy the shared contract."""


def load_schema(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def load_raw_packet_schema() -> dict[str, Any]:
    return load_schema(RAW_PACKET_SCHEMA_PATH)


def load_raw_transaction_schema() -> dict[str, Any]:
    return load_schema(RAW_TRANSACTION_SCHEMA_PATH)


def _clone_json(value: Any) -> Any:
    return copy.deepcopy(value)


def _raise(path: str, message: str) -> None:
    raise RawArtifactValidationError(f"{path}: {message}")


def _is_number(value: Any) -> bool:
    return isinstance(value, numbers.Real) and not isinstance(value, bool)


def _require_mapping(value: Any, path: str, *, allow_none: bool = False) -> dict[str, Any] | None:
    if value is None:
        if allow_none:
            return None
        _raise(path, "expected object")
    if not isinstance(value, dict):
        _raise(path, "expected object")
    return value


def _require_string(value: Any, path: str, *, allow_none: bool = False, allow_empty: bool = False) -> str | None:
    if value is None:
        if allow_none:
            return None
        _raise(path, "expected string")
    if not isinstance(value, str):
        _raise(path, "expected string")
    if not allow_empty and not value.strip():
        _raise(path, "expected non-empty string")
    return value


def _require_number(value: Any, path: str, *, allow_none: bool = False) -> float | None:
    if value is None:
        if allow_none:
            return None
        _raise(path, "expected number")
    if not _is_number(value):
        _raise(path, "expected number")
    return float(value)


def _require_integer(value: Any, path: str, *, allow_none: bool = False) -> int | None:
    if value is None:
        if allow_none:
            return None
        _raise(path, "expected integer")
    if isinstance(value, bool) or not isinstance(value, int):
        _raise(path, "expected integer")
    return int(value)


def _require_boolean(value: Any, path: str, *, allow_none: bool = False) -> bool | None:
    if value is None:
        if allow_none:
            return None
        _raise(path, "expected boolean")
    if not isinstance(value, bool):
        _raise(path, "expected boolean")
    return value


def _require_list(value: Any, path: str) -> list[Any]:
    if not isinstance(value, list):
        _raise(path, "expected array")
    return value


def _require_string_list(value: Any, path: str) -> list[str]:
    items = _require_list(value, path)
    for index, item in enumerate(items):
        _require_string(item, f"{path}[{index}]")
    return [str(item) for item in items]


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _optional_number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if _is_number(value):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


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


def _optional_bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip() in {"0", "1"}:
        return value.strip() == "1"
    return None


def _source_artifact_paths(paths: Iterable[str] | None) -> list[str]:
    if paths is None:
        return []
    values: list[str] = []
    for item in paths:
        text = _optional_text(item)
        if text is not None:
            values.append(text)
    return values


def _packet_family_and_stage(packet_kind: str) -> tuple[str, str | None]:
    normalized = packet_kind.strip().lower()
    if normalized == "telemetry":
        return "telemetry", "telemetry"
    if normalized == "request":
        return "request", "request"
    if normalized == "uplink":
        return "request", "forwarded_request"
    if normalized == "sat_response":
        return "response", "intermediate_response"
    if normalized == "final":
        return "response", "terminal_response"
    if normalized == "event":
        return "event", "event"
    return "other", normalized or None


def _infer_transport_family(record: dict[str, Any]) -> str | None:
    if record.get("src_port") not in (None, "") or record.get("dst_port") not in (None, ""):
        return "tcp"
    return None


def _build_command_object(
    *,
    raw_name: Any,
    service_name: Any,
    raw_arguments: Any,
) -> dict[str, Any] | None:
    name = _optional_text(raw_name)
    service = _optional_text(service_name)
    if name is None and raw_arguments in (None, {}, [], ""):
        return None
    return {
        "raw_name": name,
        "raw_identifier": {
            "service_name": service,
            "native_service_id": None,
            "native_command_id": None,
        },
        "raw_arguments": _clone_json(raw_arguments) if raw_arguments not in (None, "") else None,
        "raw_argument_representation": _clone_json(raw_arguments) if raw_arguments not in (None, "") else None,
    }


def _build_correlation(record: dict[str, Any]) -> dict[str, Any]:
    return {
        "run_id": _optional_int(record.get("run_id")),
        "episode_id": _optional_int(record.get("episode_id")),
        "session_id": _optional_text(record.get("session_id")),
        "transaction_id": _optional_text(record.get("txn_id")),
        "send_id": _optional_text(record.get("send_id")),
        "stream_id": _optional_text(record.get("target_stream_id")),
        "stream_index": _optional_int(record.get("target_stream_index")),
    }


def _build_evaluation_context(record: dict[str, Any]) -> dict[str, Any]:
    label_name = _optional_text(record.get("label_name")) or _optional_text(record.get("episode_kind"))
    return {
        "label": _optional_int(record.get("label")),
        "label_name": label_name,
        "attack_family": _optional_text(record.get("attack_family")),
        "phase": _optional_text(record.get("phase")),
        "actor_id": _optional_text(record.get("actor")),
        "actor_role": _optional_text(record.get("actor_role")),
        "actor_trust": _optional_number(record.get("actor_trust")),
    }


def _build_network_endpoint_from_packet(packet: dict[str, Any], *, direction: str) -> dict[str, Any] | None:
    transport_family = _infer_transport_family(packet)
    if direction == "src":
        host = _optional_text(packet.get("src"))
        ip = _optional_text(packet.get("src_ip"))
        port = _optional_int(packet.get("src_port"))
    else:
        host = _optional_text(packet.get("dst"))
        ip = _optional_text(packet.get("dst_ip"))
        port = _optional_int(packet.get("dst_port"))
    if host is None and ip is None and port is None and transport_family is None:
        return None
    return {
        "host": host,
        "ip": ip,
        "port": port,
        "transport_family": transport_family,
    }


def _primary_transaction_packet(related_packets: list[dict[str, Any]]) -> dict[str, Any] | None:
    for packet in related_packets:
        if str(packet.get("packet_kind", "")).strip().lower() == "request":
            return packet
    for packet in related_packets:
        if _packet_family_and_stage(str(packet.get("packet_kind", "")))[0] == "request":
            return packet
    return related_packets[0] if related_packets else None


def _packet_source_value(related_packets: list[dict[str, Any]], field_name: str) -> str | None:
    primary_packet = _primary_transaction_packet(related_packets)
    if primary_packet is not None:
        value = _optional_text(primary_packet.get(field_name))
        if value is not None:
            return value
    for packet in related_packets:
        value = _optional_text(packet.get(field_name))
        if value is not None:
            return value
    return None


def _build_outcome(record: dict[str, Any]) -> dict[str, Any] | None:
    outcome = {
        "accepted": _optional_bool(record.get("gds_accept")),
        "executed_successfully": _optional_bool(record.get("sat_success")),
        "timed_out": _optional_bool(record.get("timeout")),
        "raw_code": _optional_number(record.get("response_code")),
        "raw_reason": _optional_text(record.get("reason")),
        "warning_count": _optional_number(record.get("txn_warning_events")),
        "error_count": _optional_number(record.get("txn_error_events")),
        "response_direction_seen": _optional_bool(record.get("response_direction_seen")),
        "terminal_observed_on_wire": _optional_bool(record.get("final_observed_on_wire")),
        "raw_event_name": _optional_text(record.get("event_name")),
    }
    if all(value is None for value in outcome.values()):
        return None
    return outcome


def _build_native_state_snapshot(transaction: dict[str, Any]) -> dict[str, Any] | None:
    target_fields = {
        key[len("target_") :]: _clone_json(value)
        for key, value in transaction.items()
        if key.startswith("target_") and key not in {"target_service", "target_node_id", "target_stream_id", "target_stream_index"}
    }
    peer_fields = {
        key[len("peer_") :]: _clone_json(value)
        for key, value in transaction.items()
        if key.startswith("peer_")
    }
    if not target_fields and not peer_fields:
        return None
    target_logical_id = _optional_text(transaction.get("target_service"))
    peer_logical_id = None
    if target_logical_id == "fprime_a":
        peer_logical_id = "fprime_b"
    elif target_logical_id == "fprime_b":
        peer_logical_id = "fprime_a"
    return {
        "target_logical_id": target_logical_id,
        "peer_logical_id": peer_logical_id,
        "snapshot_observed_at_ms": _optional_number(transaction.get("request_ts_ms")),
        "target_fields": target_fields,
        "peer_fields": peer_fields,
    }


def adapt_legacy_fprime_packet(
    packet: dict[str, Any],
    *,
    source_artifact_paths: Iterable[str] | None = None,
    capture_backend: str | None = None,
    capture_interface: str | None = None,
) -> dict[str, Any]:
    message_family, message_stage = _packet_family_and_stage(str(packet.get("packet_kind", "")))
    transport_family = _infer_transport_family(packet)
    return {
        "schema_version": RAW_PACKET_SCHEMA_VERSION,
        "record_kind": "raw_packet",
        "protocol_family": "fprime",
        "protocol_version": None,
        "platform_family": "spacecraft",
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
                "transport_family": transport_family,
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
                "transport_family": transport_family,
            },
        },
        "command": _build_command_object(
            raw_name=packet.get("command"),
            service_name=packet.get("service"),
            raw_arguments=packet.get("args"),
        ),
        "timing": {
            "observed_at_ms": _optional_number(packet.get("ts_ms")) or 0.0,
            "timestamp_source": _optional_text(packet.get("ts_source")),
        },
        "transport": {
            "transport_family": transport_family,
            "bytes_on_wire": _optional_number(packet.get("bytes_on_wire")),
            "bytes_source": _optional_text(packet.get("bytes_source")),
            "src_ip": _optional_text(packet.get("src_ip")),
            "src_port": _optional_int(packet.get("src_port")),
            "dst_ip": _optional_text(packet.get("dst_ip")),
            "dst_port": _optional_int(packet.get("dst_port")),
        },
        "provenance": {
            "observed_on_wire": _optional_bool(packet.get("observed_on_wire")) or False,
            "capture_backend": _optional_text(capture_backend),
            "capture_interface": _optional_text(capture_interface),
            "timestamp_source": _optional_text(packet.get("ts_source")),
            "bytes_source": _optional_text(packet.get("bytes_source")),
            "source_artifact_paths": _source_artifact_paths(source_artifact_paths),
        },
        "outcome": _build_outcome(packet),
        "correlation": _build_correlation(packet),
        "evaluation_context": _build_evaluation_context(packet),
        "native_payload": _clone_json(packet.get("payload")),
        "native_fields": {"legacy_record": _clone_json(packet)},
    }


def _related_packet_evidence(related_packets: list[dict[str, Any]], source_artifact_paths: Iterable[str] | None) -> dict[str, Any]:
    message_families: list[str] = []
    message_stages: list[str] = []
    timestamp_sources: list[str] = []
    byte_sources: list[str] = []
    request_wire_observed = False
    response_wire_observed = False
    for packet in related_packets:
        message_family, message_stage = _packet_family_and_stage(str(packet.get("packet_kind", "")))
        if message_family not in message_families:
            message_families.append(message_family)
        if message_stage and message_stage not in message_stages:
            message_stages.append(message_stage)
        ts_source = _optional_text(packet.get("ts_source"))
        if ts_source and ts_source not in timestamp_sources:
            timestamp_sources.append(ts_source)
        byte_source = _optional_text(packet.get("bytes_source"))
        if byte_source and byte_source not in byte_sources:
            byte_sources.append(byte_source)
        observed_on_wire = bool(_optional_bool(packet.get("observed_on_wire")))
        if message_family == "request" and observed_on_wire:
            request_wire_observed = True
        if message_family == "response" and observed_on_wire:
            response_wire_observed = True
    return {
        "related_packet_count": len(related_packets),
        "observed_message_families": message_families,
        "observed_message_stages": message_stages,
        "packet_timestamp_sources": timestamp_sources,
        "packet_byte_sources": byte_sources,
        "request_wire_observed": request_wire_observed,
        "response_wire_observed": response_wire_observed,
        "log_correlation_mode": "session_txn_id",
        "source_artifact_paths": _source_artifact_paths(source_artifact_paths),
    }


def adapt_legacy_fprime_transaction(
    transaction: dict[str, Any],
    *,
    related_packets: list[dict[str, Any]] | None = None,
    source_artifact_paths: Iterable[str] | None = None,
    capture_backend: str | None = None,
    capture_interface: str | None = None,
) -> dict[str, Any]:
    packet_list = list(related_packets or [])
    transport_family = None
    for packet in packet_list:
        transport_family = _infer_transport_family(packet)
        if transport_family is not None:
            break
    primary_packet = _primary_transaction_packet(packet_list)
    transaction_observed_on_wire = any(
        bool(_optional_bool(packet.get("observed_on_wire")))
        for packet in packet_list
    ) or bool(_optional_bool(transaction.get("final_observed_on_wire")))
    return {
        "schema_version": RAW_TRANSACTION_SCHEMA_VERSION,
        "record_kind": "raw_transaction",
        "protocol_family": "fprime",
        "protocol_version": None,
        "platform_family": "spacecraft",
        "sender": {
            "logical_id": _optional_text(transaction.get("actor")) or "unknown",
            "role": _optional_text(transaction.get("actor_role")),
            "trust_score": _optional_number(transaction.get("actor_trust")),
            "network_endpoint": None if primary_packet is None else _build_network_endpoint_from_packet(primary_packet, direction="src"),
        },
        "target": {
            "logical_id": _optional_text(transaction.get("target_service")) or "unknown",
            "role": None,
            "stream_id": _optional_text(transaction.get("target_stream_id")),
            "stream_index": _optional_int(transaction.get("target_stream_index")),
            "network_endpoint": None if primary_packet is None else _build_network_endpoint_from_packet(primary_packet, direction="dst"),
        },
        "command": _build_command_object(
            raw_name=transaction.get("command"),
            service_name=transaction.get("service"),
            raw_arguments=transaction.get("args"),
        ),
        "timing": {
            "submitted_at_ms": _optional_number(transaction.get("request_ts_ms")),
            "request_forwarded_at_ms": _optional_number(transaction.get("uplink_ts_ms")),
            "protocol_response_at_ms": _optional_number(transaction.get("sat_response_ts_ms")),
            "finalized_at_ms": _optional_number(transaction.get("final_ts_ms")),
            "latency_ms": _optional_number(transaction.get("latency_ms")),
        },
        "transport": {
            "transport_family": transport_family,
            "request_bytes_on_wire": _optional_number(transaction.get("req_bytes")),
            "response_bytes_on_wire": _optional_number(transaction.get("resp_bytes")),
        },
        "provenance": {
            "observed_on_wire": transaction_observed_on_wire,
            "capture_backend": _optional_text(capture_backend),
            "capture_interface": _optional_text(capture_interface),
            "timestamp_source": _packet_source_value(packet_list, "ts_source"),
            "bytes_source": _packet_source_value(packet_list, "bytes_source"),
            "source_artifact_paths": _source_artifact_paths(source_artifact_paths),
        },
        "outcome": _build_outcome(transaction),
        "correlation": _build_correlation(transaction),
        "evaluation_context": _build_evaluation_context(transaction),
        "evidence": _related_packet_evidence(packet_list, source_artifact_paths),
        "native_state_snapshot": _build_native_state_snapshot(transaction),
        "native_fields": {"legacy_record": _clone_json(transaction)},
    }


def related_packets_by_transaction(packets: Iterable[dict[str, Any]]) -> dict[tuple[str, str], list[dict[str, Any]]]:
    grouped: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for packet in packets:
        session_id = _optional_text(packet.get("session_id"))
        transaction_id = _optional_text(packet.get("txn_id"))
        if session_id is None or transaction_id is None:
            continue
        grouped.setdefault((session_id, transaction_id), []).append(packet)
    return grouped


def validate_raw_packet(record: dict[str, Any]) -> dict[str, Any]:
    root = _require_mapping(record, "raw_packet")
    if root.get("schema_version") != RAW_PACKET_SCHEMA_VERSION:
        _raise("raw_packet.schema_version", f"expected {RAW_PACKET_SCHEMA_VERSION}")
    if root.get("record_kind") != "raw_packet":
        _raise("raw_packet.record_kind", "expected raw_packet")
    _require_string(root.get("protocol_family"), "raw_packet.protocol_family")
    _require_string(root.get("message_family"), "raw_packet.message_family")
    _require_string(root.get("message_stage"), "raw_packet.message_stage", allow_none=True)
    _require_string(root.get("platform_family"), "raw_packet.platform_family", allow_none=True)
    _require_string(root.get("protocol_version"), "raw_packet.protocol_version", allow_none=True)
    _validate_identity(root.get("sender"), "raw_packet.sender")
    _validate_target(root.get("target"), "raw_packet.target")
    _validate_command(root.get("command"), "raw_packet.command")
    _validate_packet_timing(root.get("timing"), "raw_packet.timing")
    _validate_packet_transport(root.get("transport"), "raw_packet.transport")
    _validate_provenance(root.get("provenance"), "raw_packet.provenance")
    _validate_outcome(root.get("outcome"), "raw_packet.outcome")
    _validate_correlation(root.get("correlation"), "raw_packet.correlation")
    _validate_evaluation_context(root.get("evaluation_context"), "raw_packet.evaluation_context")
    _require_mapping(root.get("native_fields"), "raw_packet.native_fields")
    return root


def validate_raw_packet_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        try:
            validated.append(validate_raw_packet(record))
        except RawArtifactValidationError as exc:
            raise RawArtifactValidationError(f"raw_packet[{index}] {exc}") from None
    return validated


def validate_raw_transaction(record: dict[str, Any]) -> dict[str, Any]:
    root = _require_mapping(record, "raw_transaction")
    if root.get("schema_version") != RAW_TRANSACTION_SCHEMA_VERSION:
        _raise("raw_transaction.schema_version", f"expected {RAW_TRANSACTION_SCHEMA_VERSION}")
    if root.get("record_kind") != "raw_transaction":
        _raise("raw_transaction.record_kind", "expected raw_transaction")
    _require_string(root.get("protocol_family"), "raw_transaction.protocol_family")
    _require_string(root.get("platform_family"), "raw_transaction.platform_family", allow_none=True)
    _require_string(root.get("protocol_version"), "raw_transaction.protocol_version", allow_none=True)
    _validate_identity(root.get("sender"), "raw_transaction.sender")
    _validate_target(root.get("target"), "raw_transaction.target")
    _validate_command(root.get("command"), "raw_transaction.command")
    _validate_transaction_timing(root.get("timing"), "raw_transaction.timing")
    _validate_transaction_transport(root.get("transport"), "raw_transaction.transport")
    _validate_provenance(root.get("provenance"), "raw_transaction.provenance")
    _validate_outcome(root.get("outcome"), "raw_transaction.outcome")
    _validate_correlation(root.get("correlation"), "raw_transaction.correlation")
    _validate_evaluation_context(root.get("evaluation_context"), "raw_transaction.evaluation_context")
    _validate_evidence(root.get("evidence"), "raw_transaction.evidence")
    _validate_native_state_snapshot(root.get("native_state_snapshot"), "raw_transaction.native_state_snapshot")
    _require_mapping(root.get("native_fields"), "raw_transaction.native_fields")
    return root


def validate_raw_transaction_records(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        try:
            validated.append(validate_raw_transaction(record))
        except RawArtifactValidationError as exc:
            raise RawArtifactValidationError(f"raw_transaction[{index}] {exc}") from None
    return validated


def validate_legacy_fprime_raw_contract(
    packets: list[dict[str, Any]],
    transactions: list[dict[str, Any]],
    *,
    packet_source_artifact_paths: Iterable[str] | None = None,
    transaction_source_artifact_paths: Iterable[str] | None = None,
    capture_backend: str | None = None,
    capture_interface: str | None = None,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    packet_records = [
        adapt_legacy_fprime_packet(
            packet,
            source_artifact_paths=packet_source_artifact_paths,
            capture_backend=capture_backend,
            capture_interface=capture_interface,
        )
        for packet in packets
    ]
    packet_index = related_packets_by_transaction(packets)
    transaction_records = [
        adapt_legacy_fprime_transaction(
            transaction,
            related_packets=packet_index.get(
                (
                    _optional_text(transaction.get("session_id")) or "",
                    _optional_text(transaction.get("txn_id")) or "",
                ),
                [],
            ),
            source_artifact_paths=transaction_source_artifact_paths,
            capture_backend=capture_backend,
            capture_interface=capture_interface,
        )
        for transaction in transactions
    ]
    validate_raw_packet_records(packet_records)
    validate_raw_transaction_records(transaction_records)
    return packet_records, transaction_records


def _validate_network_endpoint(value: Any, path: str) -> None:
    endpoint = _require_mapping(value, path, allow_none=True)
    if endpoint is None:
        return
    _require_string(endpoint.get("host"), f"{path}.host", allow_none=True)
    _require_string(endpoint.get("ip"), f"{path}.ip", allow_none=True)
    _require_integer(endpoint.get("port"), f"{path}.port", allow_none=True)
    _require_string(endpoint.get("transport_family"), f"{path}.transport_family", allow_none=True)


def _validate_identity(value: Any, path: str) -> None:
    identity = _require_mapping(value, path)
    _require_string(identity.get("logical_id"), f"{path}.logical_id")
    _require_string(identity.get("role"), f"{path}.role", allow_none=True)
    _require_number(identity.get("trust_score"), f"{path}.trust_score", allow_none=True)
    _validate_network_endpoint(identity.get("network_endpoint"), f"{path}.network_endpoint")


def _validate_target(value: Any, path: str) -> None:
    target = _require_mapping(value, path)
    _require_string(target.get("logical_id"), f"{path}.logical_id")
    _require_string(target.get("role"), f"{path}.role", allow_none=True)
    _require_string(target.get("stream_id"), f"{path}.stream_id", allow_none=True)
    _require_integer(target.get("stream_index"), f"{path}.stream_index", allow_none=True)
    _validate_network_endpoint(target.get("network_endpoint"), f"{path}.network_endpoint")


def _validate_command(value: Any, path: str) -> None:
    command = _require_mapping(value, path, allow_none=True)
    if command is None:
        return
    _require_string(command.get("raw_name"), f"{path}.raw_name", allow_none=True)
    identifier = _require_mapping(command.get("raw_identifier"), f"{path}.raw_identifier")
    _require_string(identifier.get("service_name"), f"{path}.raw_identifier.service_name", allow_none=True)
    _require_number(identifier.get("native_service_id"), f"{path}.raw_identifier.native_service_id", allow_none=True)
    _require_number(identifier.get("native_command_id"), f"{path}.raw_identifier.native_command_id", allow_none=True)


def _validate_packet_timing(value: Any, path: str) -> None:
    timing = _require_mapping(value, path)
    _require_number(timing.get("observed_at_ms"), f"{path}.observed_at_ms")
    _require_string(timing.get("timestamp_source"), f"{path}.timestamp_source", allow_none=True)


def _validate_transaction_timing(value: Any, path: str) -> None:
    timing = _require_mapping(value, path)
    _require_number(timing.get("submitted_at_ms"), f"{path}.submitted_at_ms", allow_none=True)
    _require_number(timing.get("request_forwarded_at_ms"), f"{path}.request_forwarded_at_ms", allow_none=True)
    _require_number(timing.get("protocol_response_at_ms"), f"{path}.protocol_response_at_ms", allow_none=True)
    _require_number(timing.get("finalized_at_ms"), f"{path}.finalized_at_ms", allow_none=True)
    _require_number(timing.get("latency_ms"), f"{path}.latency_ms", allow_none=True)


def _validate_packet_transport(value: Any, path: str) -> None:
    transport = _require_mapping(value, path)
    _require_string(transport.get("transport_family"), f"{path}.transport_family", allow_none=True)
    _require_number(transport.get("bytes_on_wire"), f"{path}.bytes_on_wire", allow_none=True)
    _require_string(transport.get("bytes_source"), f"{path}.bytes_source", allow_none=True)
    _require_string(transport.get("src_ip"), f"{path}.src_ip", allow_none=True)
    _require_integer(transport.get("src_port"), f"{path}.src_port", allow_none=True)
    _require_string(transport.get("dst_ip"), f"{path}.dst_ip", allow_none=True)
    _require_integer(transport.get("dst_port"), f"{path}.dst_port", allow_none=True)


def _validate_transaction_transport(value: Any, path: str) -> None:
    transport = _require_mapping(value, path)
    _require_string(transport.get("transport_family"), f"{path}.transport_family", allow_none=True)
    _require_number(transport.get("request_bytes_on_wire"), f"{path}.request_bytes_on_wire", allow_none=True)
    _require_number(transport.get("response_bytes_on_wire"), f"{path}.response_bytes_on_wire", allow_none=True)


def _validate_provenance(value: Any, path: str) -> None:
    provenance = _require_mapping(value, path)
    _require_boolean(provenance.get("observed_on_wire"), f"{path}.observed_on_wire")
    _require_string(provenance.get("capture_backend"), f"{path}.capture_backend", allow_none=True)
    _require_string(provenance.get("capture_interface"), f"{path}.capture_interface", allow_none=True)
    _require_string(provenance.get("timestamp_source"), f"{path}.timestamp_source", allow_none=True)
    _require_string(provenance.get("bytes_source"), f"{path}.bytes_source", allow_none=True)
    _require_string_list(provenance.get("source_artifact_paths"), f"{path}.source_artifact_paths")


def _validate_outcome(value: Any, path: str) -> None:
    outcome = _require_mapping(value, path, allow_none=True)
    if outcome is None:
        return
    _require_boolean(outcome.get("accepted"), f"{path}.accepted", allow_none=True)
    _require_boolean(outcome.get("executed_successfully"), f"{path}.executed_successfully", allow_none=True)
    _require_boolean(outcome.get("timed_out"), f"{path}.timed_out", allow_none=True)
    _require_number(outcome.get("raw_code"), f"{path}.raw_code", allow_none=True)
    _require_string(outcome.get("raw_reason"), f"{path}.raw_reason", allow_none=True)
    _require_number(outcome.get("warning_count"), f"{path}.warning_count", allow_none=True)
    _require_number(outcome.get("error_count"), f"{path}.error_count", allow_none=True)
    _require_boolean(outcome.get("response_direction_seen"), f"{path}.response_direction_seen", allow_none=True)
    _require_boolean(outcome.get("terminal_observed_on_wire"), f"{path}.terminal_observed_on_wire", allow_none=True)
    _require_string(outcome.get("raw_event_name"), f"{path}.raw_event_name", allow_none=True)


def _validate_correlation(value: Any, path: str) -> None:
    correlation = _require_mapping(value, path)
    _require_integer(correlation.get("run_id"), f"{path}.run_id", allow_none=True)
    _require_integer(correlation.get("episode_id"), f"{path}.episode_id", allow_none=True)
    _require_string(correlation.get("session_id"), f"{path}.session_id", allow_none=True)
    _require_string(correlation.get("transaction_id"), f"{path}.transaction_id", allow_none=True)
    _require_string(correlation.get("send_id"), f"{path}.send_id", allow_none=True)
    _require_string(correlation.get("stream_id"), f"{path}.stream_id", allow_none=True)
    _require_integer(correlation.get("stream_index"), f"{path}.stream_index", allow_none=True)


def _validate_evaluation_context(value: Any, path: str) -> None:
    context = _require_mapping(value, path)
    _require_integer(context.get("label"), f"{path}.label", allow_none=True)
    _require_string(context.get("label_name"), f"{path}.label_name", allow_none=True)
    _require_string(context.get("attack_family"), f"{path}.attack_family", allow_none=True)
    _require_string(context.get("phase"), f"{path}.phase", allow_none=True)
    _require_string(context.get("actor_id"), f"{path}.actor_id", allow_none=True)
    _require_string(context.get("actor_role"), f"{path}.actor_role", allow_none=True)
    _require_number(context.get("actor_trust"), f"{path}.actor_trust", allow_none=True)


def _validate_evidence(value: Any, path: str) -> None:
    evidence = _require_mapping(value, path)
    _require_integer(evidence.get("related_packet_count"), f"{path}.related_packet_count")
    _require_string_list(evidence.get("observed_message_families"), f"{path}.observed_message_families")
    _require_string_list(evidence.get("observed_message_stages"), f"{path}.observed_message_stages")
    _require_string_list(evidence.get("packet_timestamp_sources"), f"{path}.packet_timestamp_sources")
    _require_string_list(evidence.get("packet_byte_sources"), f"{path}.packet_byte_sources")
    _require_boolean(evidence.get("request_wire_observed"), f"{path}.request_wire_observed")
    _require_boolean(evidence.get("response_wire_observed"), f"{path}.response_wire_observed")
    _require_string(evidence.get("log_correlation_mode"), f"{path}.log_correlation_mode")
    _require_string_list(evidence.get("source_artifact_paths"), f"{path}.source_artifact_paths")


def _validate_native_state_snapshot(value: Any, path: str) -> None:
    snapshot = _require_mapping(value, path, allow_none=True)
    if snapshot is None:
        return
    _require_string(snapshot.get("target_logical_id"), f"{path}.target_logical_id", allow_none=True)
    _require_string(snapshot.get("peer_logical_id"), f"{path}.peer_logical_id", allow_none=True)
    _require_number(snapshot.get("snapshot_observed_at_ms"), f"{path}.snapshot_observed_at_ms", allow_none=True)
    _require_mapping(snapshot.get("target_fields"), f"{path}.target_fields")
    _require_mapping(snapshot.get("peer_fields"), f"{path}.peer_fields")
