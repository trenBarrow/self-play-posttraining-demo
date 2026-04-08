from __future__ import annotations

import copy
import json
import numbers
from pathlib import Path
from typing import Any, Iterable, Mapping

from tools.shared.canonical_state import (
    CANONICAL_STATE_BOOLEAN_FIELDS,
    CANONICAL_STATE_RATIO_FIELDS,
    summarize_normalized_state,
)
from tools.shared.schema import adapt_legacy_fprime_transaction
from tools.shared.taxonomy import resolve_command_semantics

CANONICAL_COMMAND_ROW_SCHEMA_VERSION = "canonical_command_row.v3"

REPO_ROOT = Path(__file__).resolve().parents[2]
SCHEMA_DIR = REPO_ROOT / "schemas"
CANONICAL_COMMAND_ROW_SCHEMA_PATH = SCHEMA_DIR / "canonical_command_row.schema.json"


class CanonicalRecordValidationError(ValueError):
    """Raised when a canonical semantic row does not satisfy the shared contract."""


def load_canonical_command_row_schema() -> dict[str, Any]:
    return json.loads(CANONICAL_COMMAND_ROW_SCHEMA_PATH.read_text(encoding="utf-8"))


def _clone_json(value: Any) -> Any:
    return copy.deepcopy(value)


def _raise(path: str, message: str) -> None:
    raise CanonicalRecordValidationError(f"{path}: {message}")


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


def _require_string_list(value: Any, path: str) -> list[str]:
    if not isinstance(value, list):
        _raise(path, "expected array")
    values: list[str] = []
    for index, item in enumerate(value):
        values.append(_require_string(item, f"{path}[{index}]") or "")
    return values


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


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _is_empty_structure(value: Any) -> bool:
    if value in (None, ""):
        return True
    if isinstance(value, dict):
        return all(_is_empty_structure(item) for item in value.values())
    if isinstance(value, list):
        return all(_is_empty_structure(item) for item in value)
    return False


def _infer_trust_class(trust_score: float | None) -> str:
    if trust_score is None:
        return "unknown"
    if trust_score >= 0.9:
        return "high"
    if trust_score >= 0.6:
        return "medium"
    return "low"


def _flatten_argument_leaves(value: Any) -> list[Any]:
    if isinstance(value, dict):
        leaves: list[Any] = []
        for child in value.values():
            leaves.extend(_flatten_argument_leaves(child))
        return leaves
    if isinstance(value, list):
        leaves = []
        for child in value:
            leaves.extend(_flatten_argument_leaves(child))
        return leaves
    return [value]


def _count_collections(value: Any) -> int:
    if isinstance(value, dict):
        return 1 + sum(_count_collections(child) for child in value.values())
    if isinstance(value, list):
        return 1 + sum(_count_collections(child) for child in value)
    return 0


def summarize_arguments(raw_arguments: Any) -> dict[str, Any]:
    if _is_empty_structure(raw_arguments):
        return {
            "has_arguments": False,
            "argument_leaf_count": 0,
            "numeric_argument_count": 0,
            "string_argument_count": 0,
            "boolean_argument_count": 0,
            "collection_argument_count": 0,
            "null_argument_count": 0,
            "numeric_magnitude_l1": None,
        }

    leaves = _flatten_argument_leaves(raw_arguments)
    numeric_values: list[float] = []
    string_count = 0
    boolean_count = 0
    null_count = 0
    for leaf in leaves:
        if leaf is None:
            null_count += 1
            continue
        if isinstance(leaf, bool):
            boolean_count += 1
            continue
        if _is_number(leaf):
            numeric_values.append(float(leaf))
            continue
        text = _optional_text(leaf)
        if text is not None:
            string_count += 1
            continue
        string_count += 1

    return {
        "has_arguments": True,
        "argument_leaf_count": len(leaves),
        "numeric_argument_count": len(numeric_values),
        "string_argument_count": string_count,
        "boolean_argument_count": boolean_count,
        "collection_argument_count": _count_collections(raw_arguments),
        "null_argument_count": null_count,
        "numeric_magnitude_l1": sum(abs(value) for value in numeric_values) if numeric_values else None,
    }
def _build_command_semantics(
    raw_transaction: dict[str, Any],
    overrides: Mapping[str, Any] | None,
) -> dict[str, Any]:
    command = _as_mapping(raw_transaction.get("command"))
    values = resolve_command_semantics(
        _optional_text(raw_transaction.get("protocol_family")) or "unknown",
        _optional_text(command.get("raw_name")),
        allow_unknown=True,
    )
    values.update(dict(overrides or {}))
    return {
        "canonical_command_name": _optional_text(values.get("canonical_command_name")),
        "canonical_command_family": _optional_text(values.get("canonical_command_family")) or "other_or_unknown",
        "mutation_scope": _optional_text(values.get("mutation_scope")) or "unknown",
        "persistence_class": _optional_text(values.get("persistence_class")) or "unknown",
        "safety_criticality": _optional_text(values.get("safety_criticality")) or "unknown",
        "authority_level": _optional_text(values.get("authority_level")) or "unknown",
        "target_scope": _optional_text(values.get("target_scope")) or "unspecified",
    }


def _build_recent_behavior(raw_transaction: dict[str, Any], overrides: Mapping[str, Any] | None) -> dict[str, Any]:
    values = dict(overrides or {})
    legacy_record = _as_mapping(_as_mapping(raw_transaction.get("native_fields")).get("legacy_record"))
    return {
        "command_rate_1m": _optional_number(values.get("command_rate_1m"))
        if "command_rate_1m" in values
        else _optional_number(legacy_record.get("command_rate_1m")),
        "error_rate_1m": _optional_number(values.get("error_rate_1m"))
        if "error_rate_1m" in values
        else _optional_number(legacy_record.get("error_rate_1m")),
        "repeat_command_count_10m": _optional_int(values.get("repeat_command_count_10m")),
        "same_target_command_rate_1m": _optional_number(values.get("same_target_command_rate_1m")),
    }


def _merge_source_artifact_paths(*groups: Iterable[str] | None) -> list[str]:
    values: list[str] = []
    for group in groups:
        if group is None:
            continue
        for item in group:
            text = _optional_text(item)
            if text is not None and text not in values:
                values.append(text)
    return values


def build_canonical_command_row(
    raw_transaction: dict[str, Any],
    *,
    command_semantics: Mapping[str, Any] | None = None,
    mission_context: Mapping[str, Any] | None = None,
    recent_behavior: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    sender = _as_mapping(raw_transaction.get("sender"))
    target = _as_mapping(raw_transaction.get("target"))
    command = _as_mapping(raw_transaction.get("command"))
    raw_identifier = _as_mapping(command.get("raw_identifier"))
    correlation = _as_mapping(raw_transaction.get("correlation"))
    evaluation_context = _as_mapping(raw_transaction.get("evaluation_context"))
    evidence = _as_mapping(raw_transaction.get("evidence"))
    provenance = _as_mapping(raw_transaction.get("provenance"))
    outcome = _as_mapping(raw_transaction.get("outcome"))
    mission_overrides = dict(mission_context or {})

    return {
        "schema_version": CANONICAL_COMMAND_ROW_SCHEMA_VERSION,
        "record_kind": "canonical_command_row",
        "platform_family": _optional_text(raw_transaction.get("platform_family")) or "unknown",
        "protocol_family": _optional_text(raw_transaction.get("protocol_family")) or "unknown",
        "protocol_version": _optional_text(raw_transaction.get("protocol_version")),
        "run_id": _optional_int(correlation.get("run_id")),
        "episode_id": _optional_int(correlation.get("episode_id")),
        "actor_context": {
            "role": _optional_text(sender.get("role")),
            "trust_score": _optional_number(sender.get("trust_score")),
            "trust_class": _infer_trust_class(_optional_number(sender.get("trust_score"))),
        },
        "mission_context": {
            "mission_phase": _optional_text(mission_overrides.get("mission_phase"))
            or _optional_text(evaluation_context.get("phase")),
            "window_class": _optional_text(mission_overrides.get("window_class")) or "unspecified",
        },
        "command_semantics": _build_command_semantics(raw_transaction, command_semantics),
        "argument_profile": summarize_arguments(command.get("raw_arguments")),
        "normalized_state": summarize_normalized_state(raw_transaction),
        "recent_behavior": _build_recent_behavior(raw_transaction, recent_behavior),
        "observability": {
            "related_packet_count": _optional_int(evidence.get("related_packet_count")) or 0,
            "request_wire_observed": _optional_bool(evidence.get("request_wire_observed")),
            "response_wire_observed": _optional_bool(evidence.get("response_wire_observed")),
            "correlated_response_observed": bool(
                _optional_bool(evidence.get("response_wire_observed"))
                or _optional_bool(outcome.get("response_direction_seen"))
                or _optional_bool(outcome.get("terminal_observed_on_wire"))
            ),
            "observed_message_families": list(evidence.get("observed_message_families") or []),
            "observed_message_stages": list(evidence.get("observed_message_stages") or []),
        },
        "audit_context": {
            "actor_id": _optional_text(sender.get("logical_id")),
            "target_id": _optional_text(target.get("logical_id")),
            "session_id": _optional_text(correlation.get("session_id")),
            "transaction_id": _optional_text(correlation.get("transaction_id")),
            "send_id": _optional_text(correlation.get("send_id")),
            "raw_command_name": _optional_text(command.get("raw_name")),
            "raw_service_name": _optional_text(raw_identifier.get("service_name")),
            "source_artifact_paths": _merge_source_artifact_paths(
                provenance.get("source_artifact_paths"),
                evidence.get("source_artifact_paths"),
            ),
            "label": _optional_int(evaluation_context.get("label")),
            "label_name": _optional_text(evaluation_context.get("label_name")),
            "attack_family": _optional_text(evaluation_context.get("attack_family")),
        },
    }


def canonicalize_legacy_fprime_transaction(
    legacy_transaction: dict[str, Any],
    *,
    related_packets: list[dict[str, Any]] | None = None,
    source_artifact_paths: Iterable[str] | None = None,
    command_semantics: Mapping[str, Any] | None = None,
    mission_context: Mapping[str, Any] | None = None,
    recent_behavior: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    raw_transaction = adapt_legacy_fprime_transaction(
        legacy_transaction,
        related_packets=related_packets,
        source_artifact_paths=source_artifact_paths,
    )
    return build_canonical_command_row(
        raw_transaction,
        command_semantics=command_semantics,
        mission_context=mission_context,
        recent_behavior=recent_behavior,
    )


def validate_canonical_command_row(record: dict[str, Any]) -> dict[str, Any]:
    root = _require_mapping(record, "canonical_command_row")
    if root.get("schema_version") != CANONICAL_COMMAND_ROW_SCHEMA_VERSION:
        _raise(
            "canonical_command_row.schema_version",
            f"expected {CANONICAL_COMMAND_ROW_SCHEMA_VERSION}",
        )
    if root.get("record_kind") != "canonical_command_row":
        _raise("canonical_command_row.record_kind", "expected canonical_command_row")
    _require_string(root.get("platform_family"), "canonical_command_row.platform_family")
    _require_string(root.get("protocol_family"), "canonical_command_row.protocol_family")
    _require_string(root.get("protocol_version"), "canonical_command_row.protocol_version", allow_none=True)
    _require_integer(root.get("run_id"), "canonical_command_row.run_id", allow_none=True)
    _require_integer(root.get("episode_id"), "canonical_command_row.episode_id", allow_none=True)
    _validate_actor_context(root.get("actor_context"), "canonical_command_row.actor_context")
    _validate_mission_context(root.get("mission_context"), "canonical_command_row.mission_context")
    _validate_command_semantics(root.get("command_semantics"), "canonical_command_row.command_semantics")
    _validate_argument_profile(root.get("argument_profile"), "canonical_command_row.argument_profile")
    _validate_normalized_state(root.get("normalized_state"), "canonical_command_row.normalized_state")
    _validate_recent_behavior(root.get("recent_behavior"), "canonical_command_row.recent_behavior")
    _validate_observability(root.get("observability"), "canonical_command_row.observability")
    _validate_audit_context(root.get("audit_context"), "canonical_command_row.audit_context")
    return root


def validate_canonical_command_rows(records: Iterable[dict[str, Any]]) -> list[dict[str, Any]]:
    validated: list[dict[str, Any]] = []
    for index, record in enumerate(records):
        try:
            validated.append(validate_canonical_command_row(record))
        except CanonicalRecordValidationError as exc:
            raise CanonicalRecordValidationError(f"canonical_command_row[{index}] {exc}") from None
    return validated


def _validate_actor_context(value: Any, path: str) -> None:
    actor = _require_mapping(value, path)
    _require_string(actor.get("role"), f"{path}.role", allow_none=True)
    _require_number(actor.get("trust_score"), f"{path}.trust_score", allow_none=True)
    _require_string(actor.get("trust_class"), f"{path}.trust_class")


def _validate_mission_context(value: Any, path: str) -> None:
    mission = _require_mapping(value, path)
    _require_string(mission.get("mission_phase"), f"{path}.mission_phase", allow_none=True)
    _require_string(mission.get("window_class"), f"{path}.window_class")


def _validate_command_semantics(value: Any, path: str) -> None:
    semantics = _require_mapping(value, path)
    _require_string(semantics.get("canonical_command_name"), f"{path}.canonical_command_name", allow_none=True)
    _require_string(semantics.get("canonical_command_family"), f"{path}.canonical_command_family")
    _require_string(semantics.get("mutation_scope"), f"{path}.mutation_scope")
    _require_string(semantics.get("persistence_class"), f"{path}.persistence_class")
    _require_string(semantics.get("safety_criticality"), f"{path}.safety_criticality")
    _require_string(semantics.get("authority_level"), f"{path}.authority_level")
    _require_string(semantics.get("target_scope"), f"{path}.target_scope")


def _validate_argument_profile(value: Any, path: str) -> None:
    profile = _require_mapping(value, path)
    _require_boolean(profile.get("has_arguments"), f"{path}.has_arguments")
    _require_integer(profile.get("argument_leaf_count"), f"{path}.argument_leaf_count")
    _require_integer(profile.get("numeric_argument_count"), f"{path}.numeric_argument_count")
    _require_integer(profile.get("string_argument_count"), f"{path}.string_argument_count")
    _require_integer(profile.get("boolean_argument_count"), f"{path}.boolean_argument_count")
    _require_integer(profile.get("collection_argument_count"), f"{path}.collection_argument_count")
    _require_integer(profile.get("null_argument_count"), f"{path}.null_argument_count")
    _require_number(profile.get("numeric_magnitude_l1"), f"{path}.numeric_magnitude_l1", allow_none=True)


def _validate_normalized_state(value: Any, path: str) -> None:
    state = _require_mapping(value, path)
    for name in CANONICAL_STATE_BOOLEAN_FIELDS:
        _require_boolean(state.get(name), f"{path}.{name}")
    for name in CANONICAL_STATE_RATIO_FIELDS:
        _require_number(state.get(name), f"{path}.{name}", allow_none=True)


def _validate_recent_behavior(value: Any, path: str) -> None:
    behavior = _require_mapping(value, path)
    _require_number(behavior.get("command_rate_1m"), f"{path}.command_rate_1m", allow_none=True)
    _require_number(behavior.get("error_rate_1m"), f"{path}.error_rate_1m", allow_none=True)
    _require_integer(behavior.get("repeat_command_count_10m"), f"{path}.repeat_command_count_10m", allow_none=True)
    _require_number(behavior.get("same_target_command_rate_1m"), f"{path}.same_target_command_rate_1m", allow_none=True)


def _validate_observability(value: Any, path: str) -> None:
    observability = _require_mapping(value, path)
    _require_integer(observability.get("related_packet_count"), f"{path}.related_packet_count")
    _require_boolean(observability.get("request_wire_observed"), f"{path}.request_wire_observed", allow_none=True)
    _require_boolean(observability.get("response_wire_observed"), f"{path}.response_wire_observed", allow_none=True)
    _require_boolean(observability.get("correlated_response_observed"), f"{path}.correlated_response_observed")
    _require_string_list(observability.get("observed_message_families"), f"{path}.observed_message_families")
    _require_string_list(observability.get("observed_message_stages"), f"{path}.observed_message_stages")


def _validate_audit_context(value: Any, path: str) -> None:
    audit = _require_mapping(value, path)
    _require_string(audit.get("actor_id"), f"{path}.actor_id", allow_none=True)
    _require_string(audit.get("target_id"), f"{path}.target_id", allow_none=True)
    _require_string(audit.get("session_id"), f"{path}.session_id", allow_none=True)
    _require_string(audit.get("transaction_id"), f"{path}.transaction_id", allow_none=True)
    _require_string(audit.get("send_id"), f"{path}.send_id", allow_none=True)
    _require_string(audit.get("raw_command_name"), f"{path}.raw_command_name", allow_none=True)
    _require_string(audit.get("raw_service_name"), f"{path}.raw_service_name", allow_none=True)
    _require_string_list(audit.get("source_artifact_paths"), f"{path}.source_artifact_paths")
    _require_integer(audit.get("label"), f"{path}.label", allow_none=True)
    _require_string(audit.get("label_name"), f"{path}.label_name", allow_none=True)
    _require_string(audit.get("attack_family"), f"{path}.attack_family", allow_none=True)
