#!/usr/bin/env python3
from __future__ import annotations

import json
import warnings
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Iterable, Mapping

import numpy as np
import yaml
from sklearn.exceptions import ConvergenceWarning
from sklearn.metrics import accuracy_score, precision_recall_fscore_support
from sklearn.neural_network import MLPClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from runtime import SimpleMLP, export_sklearn_mlp, stable_token_id
from tools.fprime_real import schedule_profiles as fprime_schedule_profiles
from tools.mavlink_real import schedule_profiles as mavlink_schedule_profiles
from tools.shared.canonical_records import (
    CANONICAL_COMMAND_ROW_SCHEMA_VERSION,
    summarize_arguments,
    validate_canonical_command_row,
)
from tools.shared.canonical_state import empty_normalized_state
from tools.shared.taxonomy import resolve_command_semantics
from tools.train.red_transcript import (
    DEFAULT_RED_CONTEXT_BUDGET_PATH,
    RED_TRANSCRIPT_SCHEMA_VERSION,
    build_red_transcript,
    load_red_context_budget,
)

RED_POLICY_ACTION_SPACE_SCHEMA_VERSION = "red_action_space.v1"
RED_POLICY_MODEL_SCHEMA_VERSION = "red_policy_model.v1"
RED_POLICY_EXAMPLE_SCHEMA_VERSION = "red_policy_example.v1"
RED_POLICY_MODEL_FAMILY = "bounded_red_policy_mlp_v1"
RED_POLICY_FEATURE_TIER = "bounded_red_policy_context"
RED_POLICY_MODEL_ARTIFACT_NAME = "red_policy_model.json"
RED_POLICY_REPORT_ARTIFACT_NAME = "red_policy_report.json"
RED_POLICY_EXAMPLES_ARTIFACT_NAME = "red_policy_examples.jsonl"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RED_ACTION_SPACE_PATH = REPO_ROOT / "configs" / "red_model" / "action_space.yaml"

SUPPORTED_RED_POLICY_PROTOCOL_MODES = ("fprime", "mavlink", "mixed")
SUPPORTED_RED_POLICY_PROTOCOL_FAMILIES = ("fprime", "mavlink")


class RedPolicyActionSpaceError(ValueError):
    """Raised when the red policy action-space config is invalid."""


class RedPolicyTrainingError(ValueError):
    """Raised when the red policy cannot be trained safely."""


@dataclass(frozen=True)
class RedPolicyArchitectureConfig:
    hidden_layer_sizes: tuple[int, ...] = (96, 48)
    activation: str = "relu"
    transcript_encoder: str = "ordered_stable_token_slots"
    context_encoder: str = "stable_token_id_plus_numeric_fill"
    shared_backbone: str = "independent_head_mlps_over_shared_feature_vector"
    family: str = RED_POLICY_MODEL_FAMILY


@dataclass(frozen=True)
class RedPolicyTrainingConfig:
    max_iter: int = 180
    batch_size: int = 64
    learning_rate_init: float = 1e-3
    alpha: float = 5e-4
    holdout_ratio: float = 0.25
    warm_start_rows_per_protocol: int = 96
    protocol_mode: str = "mixed"
    warm_start_source: str = "seeded_cyber_schedule_generators"


DEFAULT_RED_POLICY_ARCHITECTURE = RedPolicyArchitectureConfig()
DEFAULT_RED_POLICY_TRAINING_CONFIG = RedPolicyTrainingConfig()


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RedPolicyActionSpaceError(f"{path} must contain a YAML object")
    return payload


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise RedPolicyTrainingError(f"{path}:{line_number} must contain JSON objects")
            records.append(payload)
    return records


def write_red_policy_examples_jsonl(path: Path, examples: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for example in examples:
            handle.write(json.dumps(dict(example), sort_keys=True))
            handle.write("\n")


def load_red_policy_examples_jsonl(path: Path) -> list[dict[str, Any]]:
    return _read_jsonl(path)


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _number(value: Any) -> float | None:
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


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _compact_category(value: Any, default: str = "unknown") -> str:
    text = (_text(value) or default).strip().lower()
    chars: list[str] = []
    last_sep = False
    for char in text:
        if char.isalnum():
            chars.append(char)
            last_sep = False
            continue
        if not last_sep:
            chars.append("_")
            last_sep = True
    compact = "".join(chars).strip("_")
    return compact or default


def _require_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RedPolicyActionSpaceError(f"{path} must be a non-empty string")
    return value


def _require_string_list(value: Any, path: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise RedPolicyActionSpaceError(f"{path} must be a non-empty list")
    items: list[str] = []
    for index, item in enumerate(value):
        items.append(_require_string(item, f"{path}[{index}]"))
    return items


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RedPolicyActionSpaceError(f"{path} must be an object")
    return value


def _parse_hms(value: str) -> int:
    parts = str(value).strip().split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"Invalid time_of_day value: {value!r}")
    hour = int(parts[0])
    minute = int(parts[1])
    second = int(parts[2]) if len(parts) == 3 else 0
    if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
        raise ValueError(f"Invalid time_of_day value: {value!r}")
    return hour * 3600 + minute * 60 + second


def _infer_trust_class(trust_score: float | None) -> str:
    if trust_score is None:
        return "unknown"
    if trust_score >= 0.9:
        return "high"
    if trust_score >= 0.6:
        return "medium"
    return "low"


def _platform_family_for_protocol(protocol_family: str) -> str:
    normalized = str(protocol_family).strip().lower()
    if normalized == "fprime":
        return "spacecraft"
    if normalized == "mavlink":
        return "multirotor"
    return "unknown"


def normalize_red_policy_protocol_mode(protocol_mode: str | None) -> str:
    normalized = str(protocol_mode or DEFAULT_RED_POLICY_TRAINING_CONFIG.protocol_mode).strip().lower()
    if normalized not in SUPPORTED_RED_POLICY_PROTOCOL_MODES:
        raise RedPolicyTrainingError(
            f"Unsupported red policy protocol mode {protocol_mode!r}; "
            f"supported={list(SUPPORTED_RED_POLICY_PROTOCOL_MODES)}"
        )
    return normalized


def load_red_action_space(path: Path | None = None) -> dict[str, Any]:
    config_path = (path or DEFAULT_RED_ACTION_SPACE_PATH).resolve()
    payload = _load_yaml(config_path)
    schema_version = _require_string(payload.get("schema_version"), "schema_version")
    if schema_version != RED_POLICY_ACTION_SPACE_SCHEMA_VERSION:
        raise RedPolicyActionSpaceError(
            f"schema_version must be {RED_POLICY_ACTION_SPACE_SCHEMA_VERSION!r}; got {schema_version!r}"
        )
    record_kind = _require_string(payload.get("record_kind"), "record_kind")
    if record_kind != "red_action_space":
        raise RedPolicyActionSpaceError("record_kind must be 'red_action_space'")
    transcript_schema_version = _require_string(
        payload.get("transcript_schema_version"),
        "transcript_schema_version",
    )
    if transcript_schema_version != RED_TRANSCRIPT_SCHEMA_VERSION:
        raise RedPolicyActionSpaceError(
            f"transcript_schema_version must be {RED_TRANSCRIPT_SCHEMA_VERSION!r}; "
            f"got {transcript_schema_version!r}"
        )
    protocol_modes = _require_string_list(payload.get("protocol_modes"), "protocol_modes")
    current_context_fields = _require_string_list(payload.get("current_context_fields"), "current_context_fields")
    if set(current_context_fields) != {"protocol_family", "platform_family", "mission_phase", "window_class"}:
        raise RedPolicyActionSpaceError(
            "current_context_fields must explicitly list protocol_family, platform_family, "
            "mission_phase, and window_class"
        )
    warm_start = _require_mapping(payload.get("warm_start"), "warm_start")
    _require_string(warm_start.get("source"), "warm_start.source")
    _require_string(warm_start.get("default_protocol_mode"), "warm_start.default_protocol_mode")
    default_rows = warm_start.get("default_rows_per_protocol")
    if isinstance(default_rows, bool) or not isinstance(default_rows, int) or default_rows <= 0:
        raise RedPolicyActionSpaceError("warm_start.default_rows_per_protocol must be a positive integer")

    action_heads = _require_mapping(payload.get("action_heads"), "action_heads")
    required_heads = {"command_family", "timing_bucket", "identity_bucket"}
    if set(action_heads) != required_heads:
        raise RedPolicyActionSpaceError(f"action_heads must be exactly {sorted(required_heads)}")

    command_family_head = _require_mapping(action_heads.get("command_family"), "action_heads.command_family")
    command_family_values = _require_string_list(
        command_family_head.get("allowed_values"),
        "action_heads.command_family.allowed_values",
    )
    if "other_or_unknown" in command_family_values:
        raise RedPolicyActionSpaceError("command_family.allowed_values must not include other_or_unknown")

    timing_head = _require_mapping(action_heads.get("timing_bucket"), "action_heads.timing_bucket")
    timing_values = _require_string_list(
        timing_head.get("allowed_values"),
        "action_heads.timing_bucket.allowed_values",
    )
    if timing_values != ["bootstrap", "rapid", "steady", "delayed"]:
        raise RedPolicyActionSpaceError(
            "timing_bucket.allowed_values must be ['bootstrap', 'rapid', 'steady', 'delayed']"
        )
    thresholds = _require_mapping(
        timing_head.get("thresholds_seconds"),
        "action_heads.timing_bucket.thresholds_seconds",
    )
    for name in ("rapid_max", "steady_max"):
        value = thresholds.get(name)
        if isinstance(value, bool) or not isinstance(value, int) or value < 0:
            raise RedPolicyActionSpaceError(f"action_heads.timing_bucket.thresholds_seconds.{name} must be >= 0")

    identity_head = _require_mapping(action_heads.get("identity_bucket"), "action_heads.identity_bucket")
    identity_values = _require_string_list(
        identity_head.get("allowed_values"),
        "action_heads.identity_bucket.allowed_values",
    )
    if identity_values != ["external_low", "shared_identity"]:
        raise RedPolicyActionSpaceError(
            "identity_bucket.allowed_values must be ['external_low', 'shared_identity']"
        )
    protocol_services = _require_mapping(
        identity_head.get("protocol_services"),
        "action_heads.identity_bucket.protocol_services",
    )
    for protocol_family in SUPPORTED_RED_POLICY_PROTOCOL_FAMILIES:
        service_mapping = _require_mapping(protocol_services.get(protocol_family), f"identity_bucket.protocol_services.{protocol_family}")
        for bucket_name in identity_values:
            _require_string_list(service_mapping.get(bucket_name), f"identity_bucket.protocol_services.{protocol_family}.{bucket_name}")

    window_class_by_phase = _require_mapping(payload.get("window_class_by_phase"), "window_class_by_phase")
    for required_phase in ("startup", "science", "mission", "downlink", "recovery", "standby"):
        _require_string(window_class_by_phase.get(required_phase), f"window_class_by_phase.{required_phase}")

    return {
        **payload,
        "config_path": str(config_path),
        "protocol_modes": protocol_modes,
        "current_context_fields": current_context_fields,
        "command_family_values": command_family_values,
        "timing_bucket_values": timing_values,
        "identity_bucket_values": identity_values,
    }


def _identity_services_by_protocol(action_space: Mapping[str, Any], protocol_family: str, identity_bucket: str) -> list[str]:
    services = (
        _as_mapping(_as_mapping(_as_mapping(action_space.get("action_heads")).get("identity_bucket")).get("protocol_services"))
        .get(str(protocol_family), {})
    )
    values = _as_mapping(services).get(str(identity_bucket), [])
    return [str(item) for item in values]


def allowed_identity_services(protocol_family: str, identity_bucket: str, *, action_space: Mapping[str, Any] | None = None) -> list[str]:
    resolved = action_space or load_red_action_space()
    return _identity_services_by_protocol(resolved, protocol_family, identity_bucket)


def infer_red_identity_bucket(
    *,
    protocol_family: str,
    source_service: str | None,
    actor_role: str | None,
    action_space: Mapping[str, Any] | None = None,
) -> str | None:
    resolved = action_space or load_red_action_space()
    normalized_protocol_family = _compact_category(protocol_family, "unknown")
    normalized_role = _compact_category(actor_role, "unknown")
    normalized_source_service = _text(source_service) or ""
    if normalized_source_service in _identity_services_by_protocol(resolved, normalized_protocol_family, "shared_identity"):
        return "shared_identity"
    if normalized_source_service in _identity_services_by_protocol(resolved, normalized_protocol_family, "external_low"):
        return "external_low"
    if normalized_role == "shared_identity" or normalized_role.startswith("ops_"):
        return "shared_identity"
    if normalized_role in {"external", "external_low"}:
        return "external_low"
    return None


def schedule_phase_to_window_class(phase: str | None, *, action_space: Mapping[str, Any] | None = None) -> str:
    resolved = action_space or load_red_action_space()
    window_mapping = _as_mapping(resolved.get("window_class_by_phase"))
    phase_name = _compact_category(phase, "unspecified")
    if phase_name in window_mapping:
        return _compact_category(window_mapping[phase_name], "unspecified")
    return "unspecified"


def _schedule_row_protocol_family(row: Mapping[str, Any]) -> str:
    meta = _as_mapping(row.get("meta"))
    explicit = _text(meta.get("protocol_family"))
    if explicit is not None:
        return _compact_category(explicit, "unknown")
    if "command_family" in row or "target_endpoint" in row:
        return "mavlink"
    return "fprime"


def _schedule_row_command_family(row: Mapping[str, Any], protocol_family: str) -> str:
    command = _text(row.get("command")) or "unknown"
    if protocol_family == "mavlink":
        explicit = _text(row.get("command_family"))
        if explicit is not None:
            return _compact_category(explicit, "other_or_unknown")
    semantics = resolve_command_semantics(protocol_family, command, allow_unknown=False)
    return _compact_category(semantics.get("canonical_command_family"), "other_or_unknown")


def schedule_row_identity_bucket(
    row: Mapping[str, Any],
    *,
    protocol_family: str,
    action_space: Mapping[str, Any] | None = None,
) -> str:
    meta = _as_mapping(row.get("meta"))
    actor_role = _text(meta.get("actor_role"))
    source_service = _text(row.get("source_service"))
    identity_bucket = infer_red_identity_bucket(
        protocol_family=protocol_family,
        source_service=source_service,
        actor_role=actor_role,
        action_space=action_space,
    )
    if identity_bucket is not None:
        return identity_bucket
    raise RedPolicyTrainingError(
        "Warm-start schedule row does not map to an allowed red identity bucket. "
        f"protocol_family={protocol_family!r} source_service={source_service!r} actor_role={actor_role!r}"
    )


def timing_bucket_from_gap_seconds(
    gap_seconds: float | None,
    *,
    action_space: Mapping[str, Any] | None = None,
) -> str:
    resolved = action_space or load_red_action_space()
    thresholds = _as_mapping(
        _as_mapping(_as_mapping(resolved.get("action_heads")).get("timing_bucket")).get("thresholds_seconds")
    )
    if gap_seconds is None:
        return "bootstrap"
    if gap_seconds <= float(thresholds.get("rapid_max", 5)):
        return "rapid"
    if gap_seconds <= float(thresholds.get("steady_max", 30)):
        return "steady"
    return "delayed"


def _empty_recent_behavior() -> dict[str, Any]:
    return {
        "command_rate_1m": None,
        "error_rate_1m": None,
        "repeat_command_count_10m": None,
        "same_target_command_rate_1m": None,
    }


def schedule_row_to_canonical_row(
    row: Mapping[str, Any],
    *,
    protocol_family: str | None = None,
    transaction_id: str | None = None,
    send_id: str | None = None,
    window_class: str | None = None,
) -> dict[str, Any]:
    meta = _as_mapping(row.get("meta"))
    resolved_protocol_family = protocol_family or _schedule_row_protocol_family(row)
    platform_family = _platform_family_for_protocol(resolved_protocol_family)
    command = _text(row.get("command")) or "unknown"
    source_service = _text(row.get("source_service")) or "unknown"
    target_service = _text(row.get("target_service")) or "unknown"
    actor_role = _text(meta.get("actor_role"))
    actor_trust = _number(meta.get("actor_trust"))
    episode_id = int(_number(meta.get("episode_id")) or 0)
    phase = _text(meta.get("phase"))
    resolved_window_class = _compact_category(
        window_class or schedule_phase_to_window_class(phase, action_space=load_red_action_space()),
        "unspecified",
    )
    semantics = resolve_command_semantics(resolved_protocol_family, command, allow_unknown=False)
    session_id = f"{resolved_protocol_family}:{episode_id}:{source_service}"
    txn_id = transaction_id or f"{session_id}:{command}"
    send_identifier = send_id or f"send:{txn_id}"
    raw_service_name = command.split(".", 1)[0] if "." in command else target_service
    canonical_row = {
        "schema_version": CANONICAL_COMMAND_ROW_SCHEMA_VERSION,
        "record_kind": "canonical_command_row",
        "platform_family": platform_family,
        "protocol_family": resolved_protocol_family,
        "protocol_version": "2.0" if resolved_protocol_family == "mavlink" else None,
        "run_id": None,
        "episode_id": episode_id,
        "actor_context": {
            "role": actor_role,
            "trust_score": actor_trust,
            "trust_class": _infer_trust_class(actor_trust),
        },
        "mission_context": {
            "mission_phase": phase,
            "window_class": resolved_window_class,
        },
        "command_semantics": {
            "canonical_command_name": _text(semantics.get("canonical_command_name")),
            "canonical_command_family": _text(semantics.get("canonical_command_family")) or "other_or_unknown",
            "mutation_scope": _text(semantics.get("mutation_scope")) or "unknown",
            "persistence_class": _text(semantics.get("persistence_class")) or "unknown",
            "safety_criticality": _text(semantics.get("safety_criticality")) or "unknown",
            "authority_level": _text(semantics.get("authority_level")) or "unknown",
            "target_scope": _text(semantics.get("target_scope")) or "unspecified",
        },
        "argument_profile": summarize_arguments(row.get("arguments")),
        "normalized_state": empty_normalized_state(),
        "recent_behavior": _empty_recent_behavior(),
        "observability": {
            "related_packet_count": 0,
            "request_wire_observed": None,
            "response_wire_observed": None,
            "correlated_response_observed": False,
            "observed_message_families": [],
            "observed_message_stages": [],
        },
        "audit_context": {
            "actor_id": source_service,
            "target_id": target_service,
            "session_id": session_id,
            "transaction_id": txn_id,
            "send_id": send_identifier,
            "raw_command_name": command,
            "raw_service_name": raw_service_name,
            "source_artifact_paths": [],
            "label": int(_number(meta.get("class_label")) or 0),
            "label_name": _text(meta.get("class_name")),
            "attack_family": _text(meta.get("attack_family")),
        },
    }
    return validate_canonical_command_row(canonical_row)


def red_policy_action_space_summary(action_space: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "config_path": str(action_space.get("config_path", DEFAULT_RED_ACTION_SPACE_PATH)),
        "current_context_fields": list(action_space.get("current_context_fields", [])),
        "action_heads": {
            "command_family": list(action_space.get("command_family_values", [])),
            "timing_bucket": list(action_space.get("timing_bucket_values", [])),
            "identity_bucket": list(action_space.get("identity_bucket_values", [])),
        },
    }


def validate_red_policy_action(action: Mapping[str, Any], *, action_space: Mapping[str, Any] | None = None) -> dict[str, str]:
    resolved = action_space or load_red_action_space()
    validated: dict[str, str] = {}
    for head_name, allowed_values in (
        ("command_family", resolved.get("command_family_values", [])),
        ("timing_bucket", resolved.get("timing_bucket_values", [])),
        ("identity_bucket", resolved.get("identity_bucket_values", [])),
    ):
        value = _compact_category(action.get(head_name), "unknown")
        if value not in allowed_values:
            raise RedPolicyTrainingError(
                f"Action head {head_name!r} received disallowed value {value!r}; allowed={list(allowed_values)}"
            )
        validated[head_name] = value
    return validated


def current_context_from_schedule_row(
    row: Mapping[str, Any],
    *,
    protocol_family: str,
    action_space: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    resolved = action_space or load_red_action_space()
    meta = _as_mapping(row.get("meta"))
    phase = _compact_category(meta.get("phase"), "unknown")
    return {
        "protocol_family": _compact_category(protocol_family, "unknown"),
        "platform_family": _compact_category(_platform_family_for_protocol(protocol_family), "unknown"),
        "mission_phase": phase,
        "window_class": schedule_phase_to_window_class(phase, action_space=resolved),
    }


def action_from_schedule_row(
    row: Mapping[str, Any],
    *,
    protocol_family: str,
    previous_self_row: Mapping[str, Any] | None,
    action_space: Mapping[str, Any] | None = None,
) -> dict[str, str]:
    resolved = action_space or load_red_action_space()
    current_seconds = _parse_hms(_text(row.get("time_of_day")) or "00:00:00")
    previous_seconds = None
    if previous_self_row is not None:
        previous_seconds = _parse_hms(_text(previous_self_row.get("time_of_day")) or "00:00:00")
    gap_seconds = None if previous_seconds is None else max(0, current_seconds - previous_seconds)
    action = {
        "command_family": _schedule_row_command_family(row, protocol_family),
        "timing_bucket": timing_bucket_from_gap_seconds(gap_seconds, action_space=resolved),
        "identity_bucket": schedule_row_identity_bucket(row, protocol_family=protocol_family, action_space=resolved),
    }
    return validate_red_policy_action(action, action_space=resolved)


def _example_group_key(protocol_family: str, row: Mapping[str, Any]) -> str:
    meta = _as_mapping(row.get("meta"))
    episode_id = int(_number(meta.get("episode_id")) or 0)
    actor_id = _text(row.get("source_service")) or "unknown"
    return f"{protocol_family}:{episode_id}:{actor_id}"


def _example_metadata(protocol_family: str, row: Mapping[str, Any], index: int) -> dict[str, Any]:
    meta = _as_mapping(row.get("meta"))
    return {
        "protocol_family": protocol_family,
        "platform_family": _platform_family_for_protocol(protocol_family),
        "episode_id": int(_number(meta.get("episode_id")) or 0),
        "actor_id": _text(row.get("source_service")) or "unknown",
        "target_id": _text(row.get("target_service")) or "unknown",
        "group_key": _example_group_key(protocol_family, row),
        "time_of_day": _text(row.get("time_of_day")) or "00:00:00",
        "raw_command_name": _text(row.get("command")) or "unknown",
        "attack_family": _text(meta.get("attack_family")) or "none",
        "sequence_index": index,
    }


def _schedule_rows_for_protocol(protocol_family: str, *, rows_per_protocol: int, seed: int) -> list[dict[str, Any]]:
    if protocol_family == "fprime":
        rows = fprime_schedule_profiles.build_cyber_rows(target_rows=rows_per_protocol, seed=seed)
    elif protocol_family == "mavlink":
        rows = mavlink_schedule_profiles.build_cyber_rows(target_rows=rows_per_protocol, seed=seed)
    else:
        raise RedPolicyTrainingError(f"Unsupported protocol_family {protocol_family!r}")
    return [dict(row) for row in rows]


def build_red_policy_warmstart_examples(
    *,
    protocol_mode: str = DEFAULT_RED_POLICY_TRAINING_CONFIG.protocol_mode,
    rows_per_protocol: int = DEFAULT_RED_POLICY_TRAINING_CONFIG.warm_start_rows_per_protocol,
    seed: int = 7,
    max_history_entries: int | None = None,
    action_space: Mapping[str, Any] | None = None,
    transcript_budget: Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    normalized_protocol_mode = normalize_red_policy_protocol_mode(protocol_mode)
    if rows_per_protocol <= 0:
        raise RedPolicyTrainingError("rows_per_protocol must be > 0")
    resolved_action_space = action_space or load_red_action_space()
    resolved_budget = transcript_budget or load_red_context_budget()
    protocol_families = ["fprime", "mavlink"] if normalized_protocol_mode == "mixed" else [normalized_protocol_mode]
    examples: list[dict[str, Any]] = []
    for protocol_family in protocol_families:
        actor_history: dict[tuple[int, str], list[dict[str, Any]]] = {}
        rows = _schedule_rows_for_protocol(protocol_family, rows_per_protocol=rows_per_protocol, seed=seed)
        for index, row in enumerate(rows):
            meta = _as_mapping(row.get("meta"))
            episode_id = int(_number(meta.get("episode_id")) or 0)
            actor_id = _text(row.get("source_service")) or "unknown"
            history_key = (episode_id, actor_id)
            previous_rows = list(actor_history.get(history_key, []))
            history_items = []
            for previous_index, previous_row in enumerate(previous_rows):
                previous_meta = _example_metadata(protocol_family, previous_row, previous_index)
                history_items.append(
                    {
                        "canonical_row": schedule_row_to_canonical_row(
                            previous_row,
                            protocol_family=protocol_family,
                            transaction_id=previous_meta["group_key"] + f":txn:{previous_index:04d}",
                            send_id=previous_meta["group_key"] + f":send:{previous_index:04d}",
                            window_class=current_context_from_schedule_row(previous_row, protocol_family=protocol_family, action_space=resolved_action_space)["window_class"],
                        )
                    }
                )
            transcript = build_red_transcript(
                history_items,
                actor_id=actor_id,
                max_history_entries=max_history_entries,
                budget=resolved_budget,
            )
            current_context = current_context_from_schedule_row(row, protocol_family=protocol_family, action_space=resolved_action_space)
            previous_self_row = previous_rows[-1] if previous_rows else None
            action = action_from_schedule_row(
                row,
                protocol_family=protocol_family,
                previous_self_row=previous_self_row,
                action_space=resolved_action_space,
            )
            metadata = _example_metadata(protocol_family, row, index)
            examples.append(
                {
                    "schema_version": RED_POLICY_EXAMPLE_SCHEMA_VERSION,
                    "record_kind": "red_policy_example",
                    "transcript": transcript,
                    "current_context": current_context,
                    "action": action,
                    "metadata": metadata,
                }
            )
            actor_history.setdefault(history_key, []).append(dict(row))
    return examples


def _fixed_token_slot_count(transcript_budget: Mapping[str, Any]) -> int:
    transcript_format = _as_mapping(transcript_budget.get("transcript_format"))
    limits = _as_mapping(transcript_budget.get("limits"))
    header_tokens = 0
    if bool(limits.get("include_header_tokens", True)):
        header_tokens = len(transcript_format.get("header_field_order") or []) + 2
    event_tokens = (len(transcript_format.get("per_event_field_order") or []) + 2) * int(limits.get("max_history_entries", 0))
    return header_tokens + event_tokens


def red_policy_feature_names(
    *,
    action_space: Mapping[str, Any] | None = None,
    transcript_budget: Mapping[str, Any] | None = None,
) -> list[str]:
    resolved_action_space = action_space or load_red_action_space()
    resolved_budget = transcript_budget or load_red_context_budget()
    slot_count = _fixed_token_slot_count(resolved_budget)
    feature_names = [f"transcript_token_slot_{index:03d}" for index in range(slot_count)]
    feature_names.extend(
        f"context_{field_name}_token"
        for field_name in resolved_action_space.get("current_context_fields", [])
    )
    feature_names.extend(
        [
            "context_history_fill_ratio",
            "context_flattened_token_fill_ratio",
        ]
    )
    return feature_names


def encode_red_policy_feature_mapping(
    transcript: Mapping[str, Any],
    current_context: Mapping[str, Any],
    *,
    action_space: Mapping[str, Any] | None = None,
    transcript_budget: Mapping[str, Any] | None = None,
) -> dict[str, float]:
    resolved_action_space = action_space or load_red_action_space()
    resolved_budget = transcript_budget or load_red_context_budget()
    feature_names = red_policy_feature_names(action_space=resolved_action_space, transcript_budget=resolved_budget)
    features = {name: 0.0 for name in feature_names}
    token_modulo = float(_as_mapping(_as_mapping(transcript).get("budget")).get("token_id_modulo") or _as_mapping(_as_mapping(resolved_budget).get("limits")).get("token_id_modulo") or 65535)
    flattened_ids = list(_as_mapping(transcript).get("flattened_token_ids") or [])
    slot_names = [name for name in feature_names if name.startswith("transcript_token_slot_")]
    for index, token_id in enumerate(flattened_ids[: len(slot_names)]):
        number = _number(token_id)
        if number is None:
            continue
        features[slot_names[index]] = (float(number) + 1.0) / max(1.0, token_modulo)

    for field_name in resolved_action_space.get("current_context_fields", []):
        token = stable_token_id(
            f"{field_name}:{_compact_category(current_context.get(field_name), 'unknown')}",
            int(token_modulo),
        )
        features[f"context_{field_name}_token"] = (float(token) + 1.0) / max(1.0, token_modulo)

    included_history_count = float(_number(_as_mapping(transcript).get("included_history_count")) or 0.0)
    max_history_entries = float(_as_mapping(_as_mapping(transcript).get("budget")).get("max_history_entries") or _as_mapping(_as_mapping(resolved_budget).get("limits")).get("max_history_entries") or 1)
    features["context_history_fill_ratio"] = included_history_count / max(1.0, max_history_entries)
    features["context_flattened_token_fill_ratio"] = float(len(flattened_ids)) / max(1.0, float(len(slot_names)))
    return features


def examples_to_feature_matrix(
    examples: list[Mapping[str, Any]],
    *,
    action_space: Mapping[str, Any] | None = None,
    transcript_budget: Mapping[str, Any] | None = None,
) -> tuple[np.ndarray, list[str]]:
    resolved_action_space = action_space or load_red_action_space()
    resolved_budget = transcript_budget or load_red_context_budget()
    feature_names = red_policy_feature_names(action_space=resolved_action_space, transcript_budget=resolved_budget)
    rows: list[list[float]] = []
    for example in examples:
        encoded = encode_red_policy_feature_mapping(
            _as_mapping(example.get("transcript")),
            _as_mapping(example.get("current_context")),
            action_space=resolved_action_space,
            transcript_budget=resolved_budget,
        )
        rows.append([float(encoded.get(name, 0.0)) for name in feature_names])
    return np.asarray(rows, dtype=float), feature_names


def _action_head_allowed_values(action_space: Mapping[str, Any], head_name: str) -> list[str]:
    if head_name == "command_family":
        return list(action_space.get("command_family_values", []))
    if head_name == "timing_bucket":
        return list(action_space.get("timing_bucket_values", []))
    if head_name == "identity_bucket":
        return list(action_space.get("identity_bucket_values", []))
    raise RedPolicyTrainingError(f"Unknown action head {head_name!r}")


def encode_action_targets(examples: list[Mapping[str, Any]], *, action_space: Mapping[str, Any]) -> dict[str, np.ndarray]:
    encoded: dict[str, np.ndarray] = {}
    for head_name in ("command_family", "timing_bucket", "identity_bucket"):
        allowed_values = _action_head_allowed_values(action_space, head_name)
        index_by_name = {name: index for index, name in enumerate(allowed_values)}
        values: list[int] = []
        for example in examples:
            action = validate_red_policy_action(_as_mapping(example.get("action")), action_space=action_space)
            values.append(index_by_name[action[head_name]])
        encoded[head_name] = np.asarray(values, dtype=int)
    return encoded


def split_red_policy_examples(
    examples: list[Mapping[str, Any]],
    *,
    holdout_ratio: float,
    seed: int,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    if not examples:
        raise RedPolicyTrainingError("No red policy examples were provided")
    if not (0.0 <= holdout_ratio < 1.0):
        raise RedPolicyTrainingError("holdout_ratio must be in [0.0, 1.0)")

    groups = sorted({
        _text(_as_mapping(example.get("metadata")).get("group_key")) or f"group_{index}"
        for index, example in enumerate(examples)
    })
    if len(groups) <= 1 or holdout_ratio <= 0.0:
        train_examples = [dict(example) for example in examples]
        return train_examples, [], {
            "group_count": len(groups),
            "holdout_group_count": 0,
            "holdout_ratio": holdout_ratio,
            "split_mode": "train_only",
        }

    rng = np.random.default_rng(seed)
    shuffled = list(groups)
    rng.shuffle(shuffled)
    holdout_group_count = max(1, min(len(groups) - 1, int(round(len(groups) * holdout_ratio))))
    holdout_groups = set(shuffled[:holdout_group_count])

    train_examples: list[dict[str, Any]] = []
    validation_examples: list[dict[str, Any]] = []
    for example in examples:
        group_key = _text(_as_mapping(example.get("metadata")).get("group_key")) or ""
        target = validation_examples if group_key in holdout_groups else train_examples
        target.append(dict(example))
    if not train_examples:
        train_examples = validation_examples[:-1]
        validation_examples = validation_examples[-1:]
    return train_examples, validation_examples, {
        "group_count": len(groups),
        "holdout_group_count": len(holdout_groups),
        "holdout_ratio": holdout_ratio,
        "split_mode": "group_holdout",
    }


class ConstantActionHeadModel:
    def __init__(self, label_index: int):
        self.label_index = int(label_index)
        self.classes_ = np.asarray([int(label_index)], dtype=int)

    def predict(self, X: np.ndarray) -> np.ndarray:
        return np.full((len(X),), self.label_index, dtype=int)

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        return np.ones((len(X), 1), dtype=float)


def _fit_action_head(
    X_train: np.ndarray,
    y_train: np.ndarray,
    *,
    seed: int,
    architecture: RedPolicyArchitectureConfig,
    training_config: RedPolicyTrainingConfig,
) -> ConstantActionHeadModel | Pipeline:
    unique_labels = sorted({int(value) for value in y_train.tolist()})
    if len(unique_labels) == 1:
        return ConstantActionHeadModel(unique_labels[0])

    pipeline = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=architecture.hidden_layer_sizes,
                    activation=architecture.activation,
                    alpha=training_config.alpha,
                    batch_size=max(1, min(int(training_config.batch_size), int(len(y_train)))),
                    learning_rate_init=training_config.learning_rate_init,
                    max_iter=training_config.max_iter,
                    random_state=seed,
                ),
            ),
        ]
    )
    with warnings.catch_warnings():
        warnings.simplefilter("ignore", category=ConvergenceWarning)
        pipeline.fit(np.asarray(X_train, dtype=float), y_train.astype(int))
    return pipeline


def _predict_action_head(model: ConstantActionHeadModel | Pipeline, X: np.ndarray) -> np.ndarray:
    return model.predict(np.asarray(X, dtype=float)).astype(int)


def _evaluate_action_head(
    model: ConstantActionHeadModel | Pipeline,
    X: np.ndarray,
    y_true: np.ndarray,
    *,
    allowed_values: list[str],
) -> dict[str, Any]:
    predictions = _predict_action_head(model, X)
    accuracy = float(accuracy_score(y_true.astype(int), predictions.astype(int))) if len(y_true) else 0.0
    _, _, f1_scores, support = precision_recall_fscore_support(
        y_true.astype(int),
        predictions.astype(int),
        labels=list(range(len(allowed_values))),
        zero_division=0,
    )
    macro_f1 = float(np.mean(f1_scores)) if len(f1_scores) else 0.0
    per_class = {}
    for index, value_name in enumerate(allowed_values):
        per_class[value_name] = {
            "f1": float(f1_scores[index]) if index < len(f1_scores) else 0.0,
            "support": int(support[index]) if index < len(support) else 0,
        }
    return {
        "accuracy": accuracy,
        "macro_f1": macro_f1,
        "support": int(len(y_true)),
        "per_class": per_class,
    }


def red_policy_architecture_report(
    feature_names: list[str],
    *,
    architecture: RedPolicyArchitectureConfig = DEFAULT_RED_POLICY_ARCHITECTURE,
    action_space: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_action_space = action_space or load_red_action_space()
    return {
        "family": architecture.family,
        "feature_count": len(feature_names),
        "feature_tier": RED_POLICY_FEATURE_TIER,
        "transcript_encoder": architecture.transcript_encoder,
        "context_encoder": architecture.context_encoder,
        "shared_backbone": architecture.shared_backbone,
        "backbone": {
            "type": "mlp",
            "hidden_layer_sizes": list(architecture.hidden_layer_sizes),
            "activation": architecture.activation,
        },
        "action_heads": red_policy_action_space_summary(resolved_action_space)["action_heads"],
    }


def fit_red_policy_model(
    examples: list[Mapping[str, Any]],
    *,
    seed: int,
    action_space: Mapping[str, Any] | None = None,
    transcript_budget: Mapping[str, Any] | None = None,
    architecture: RedPolicyArchitectureConfig = DEFAULT_RED_POLICY_ARCHITECTURE,
    training_config: RedPolicyTrainingConfig = DEFAULT_RED_POLICY_TRAINING_CONFIG,
) -> dict[str, Any]:
    if not examples:
        raise RedPolicyTrainingError("Red policy training requires at least one example")
    resolved_action_space = action_space or load_red_action_space()
    resolved_budget = transcript_budget or load_red_context_budget()
    train_examples, validation_examples, split_summary = split_red_policy_examples(
        examples,
        holdout_ratio=training_config.holdout_ratio,
        seed=seed,
    )
    X_train, feature_names = examples_to_feature_matrix(
        train_examples,
        action_space=resolved_action_space,
        transcript_budget=resolved_budget,
    )
    train_targets = encode_action_targets(train_examples, action_space=resolved_action_space)
    X_validation, _ = examples_to_feature_matrix(
        validation_examples,
        action_space=resolved_action_space,
        transcript_budget=resolved_budget,
    )
    validation_targets = encode_action_targets(validation_examples, action_space=resolved_action_space) if validation_examples else {}

    head_models: dict[str, ConstantActionHeadModel | Pipeline] = {}
    head_reports: dict[str, Any] = {}
    validation_predictions: dict[str, np.ndarray] = {}
    for offset, head_name in enumerate(("command_family", "timing_bucket", "identity_bucket")):
        model = _fit_action_head(
            X_train,
            train_targets[head_name],
            seed=seed + 101 * (offset + 1),
            architecture=architecture,
            training_config=training_config,
        )
        head_models[head_name] = model
        evaluation_X = X_validation if len(X_validation) else X_train
        evaluation_y = validation_targets.get(head_name, train_targets[head_name]) if validation_examples else train_targets[head_name]
        head_reports[head_name] = _evaluate_action_head(
            model,
            evaluation_X,
            evaluation_y,
            allowed_values=_action_head_allowed_values(resolved_action_space, head_name),
        )
        validation_predictions[head_name] = _predict_action_head(model, evaluation_X)

    evaluation_length = len(X_validation) if validation_examples else len(X_train)
    exact_matches = 0
    if evaluation_length > 0:
        for index in range(evaluation_length):
            if all(
                int(validation_predictions[head_name][index]) == int(
                    (validation_targets if validation_examples else train_targets)[head_name][index]
                )
                for head_name in ("command_family", "timing_bucket", "identity_bucket")
            ):
                exact_matches += 1
    evaluation_summary = {
        "split": split_summary,
        "evaluated_example_count": int(evaluation_length),
        "joint_exact_match_accuracy": float(exact_matches / evaluation_length) if evaluation_length else 0.0,
        "heads": head_reports,
    }

    warm_start_summary = summarize_red_policy_examples(examples)
    return {
        "feature_names": feature_names,
        "head_models": head_models,
        "architecture": red_policy_architecture_report(
            feature_names,
            architecture=architecture,
            action_space=resolved_action_space,
        ),
        "training_config": asdict(training_config),
        "evaluation": evaluation_summary,
        "warm_start": warm_start_summary,
        "action_space": red_policy_action_space_summary(resolved_action_space),
        "transcript_budget": {
            "config_path": str(resolved_budget.get("config_path", DEFAULT_RED_CONTEXT_BUDGET_PATH)),
            "max_history_entries": int(_as_mapping(resolved_budget.get("limits")).get("max_history_entries", 0)),
            "transcript_schema_version": RED_TRANSCRIPT_SCHEMA_VERSION,
        },
        "train_example_count": len(train_examples),
        "validation_example_count": len(validation_examples),
    }


def summarize_red_policy_examples(examples: list[Mapping[str, Any]]) -> dict[str, Any]:
    protocol_counts: dict[str, int] = {}
    command_family_counts: dict[str, int] = {}
    timing_bucket_counts: dict[str, int] = {}
    identity_bucket_counts: dict[str, int] = {}
    for example in examples:
        metadata = _as_mapping(example.get("metadata"))
        action = _as_mapping(example.get("action"))
        protocol_family = _compact_category(metadata.get("protocol_family"), "unknown")
        command_family = _compact_category(action.get("command_family"), "unknown")
        timing_bucket = _compact_category(action.get("timing_bucket"), "unknown")
        identity_bucket = _compact_category(action.get("identity_bucket"), "unknown")
        protocol_counts[protocol_family] = protocol_counts.get(protocol_family, 0) + 1
        command_family_counts[command_family] = command_family_counts.get(command_family, 0) + 1
        timing_bucket_counts[timing_bucket] = timing_bucket_counts.get(timing_bucket, 0) + 1
        identity_bucket_counts[identity_bucket] = identity_bucket_counts.get(identity_bucket, 0) + 1
    return {
        "example_count": len(examples),
        "protocol_counts": protocol_counts,
        "command_family_counts": command_family_counts,
        "timing_bucket_counts": timing_bucket_counts,
        "identity_bucket_counts": identity_bucket_counts,
    }


def _constant_head_payload(
    *,
    head_name: str,
    feature_names: list[str],
    allowed_values: list[str],
    label_index: int,
) -> dict[str, Any]:
    return {
        "model_type": "constant",
        "model_name": "constant_action_head",
        "feature_tier": RED_POLICY_FEATURE_TIER,
        "feature_names": list(feature_names),
        "class_labels": [int(label_index)],
        "selected_label": int(label_index),
        "action_head": head_name,
        "allowed_values": list(allowed_values),
    }


def export_red_policy_model_payload(
    trained: Mapping[str, Any],
    *,
    action_space: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_action_space = action_space or load_red_action_space()
    feature_names = list(trained.get("feature_names", []))
    head_payloads: dict[str, Any] = {}
    for head_name in ("command_family", "timing_bucket", "identity_bucket"):
        allowed_values = _action_head_allowed_values(resolved_action_space, head_name)
        model = trained["head_models"][head_name]
        if isinstance(model, ConstantActionHeadModel):
            head_payloads[head_name] = _constant_head_payload(
                head_name=head_name,
                feature_names=feature_names,
                allowed_values=allowed_values,
                label_index=int(model.label_index),
            )
        else:
            head_payloads[head_name] = export_sklearn_mlp(
                model,
                feature_names,
                RED_POLICY_FEATURE_TIER,
                extra_fields={
                    "model_name": f"red_policy_{head_name}",
                    "action_head": head_name,
                    "allowed_values": allowed_values,
                },
            )
    return {
        "schema_version": RED_POLICY_MODEL_SCHEMA_VERSION,
        "record_kind": "red_policy_model",
        "model_family": RED_POLICY_MODEL_FAMILY,
        "feature_tier": RED_POLICY_FEATURE_TIER,
        "feature_names": feature_names,
        "architecture": dict(trained.get("architecture", {})),
        "training_config": dict(trained.get("training_config", {})),
        "evaluation": dict(trained.get("evaluation", {})),
        "warm_start": dict(trained.get("warm_start", {})),
        "action_space": dict(resolved_action_space),
        "transcript_budget": dict(trained.get("transcript_budget", {})),
        "heads": head_payloads,
    }


class LoadedConstantRedPolicyHead:
    def __init__(self, payload: Mapping[str, Any]):
        self.selected_label = int(payload.get("selected_label", 0))
        self.class_labels = [int(value) for value in payload.get("class_labels", [self.selected_label])]

    def predict_proba_one(self, row: Mapping[str, Any]) -> list[float]:
        del row
        return [1.0]


def _load_red_policy_head(payload: Mapping[str, Any]) -> LoadedConstantRedPolicyHead | SimpleMLP:
    if str(payload.get("model_type")) == "constant":
        return LoadedConstantRedPolicyHead(payload)
    return SimpleMLP.from_payload(dict(payload))


class LoadedRedPolicyModel:
    def __init__(self, payload: Mapping[str, Any]):
        if str(payload.get("schema_version")) != RED_POLICY_MODEL_SCHEMA_VERSION:
            raise RedPolicyTrainingError(
                f"Expected {RED_POLICY_MODEL_SCHEMA_VERSION!r}; got {payload.get('schema_version')!r}"
            )
        self.payload = dict(payload)
        self.feature_names = list(payload.get("feature_names", []))
        self.action_space = dict(payload.get("action_space", {}))
        self.transcript_budget = dict(payload.get("transcript_budget", {}))
        self.heads = {
            head_name: {
                "payload": dict(head_payload),
                "model": _load_red_policy_head(dict(head_payload)),
                "allowed_values": list(head_payload.get("allowed_values", [])),
                "class_labels": [int(value) for value in head_payload.get("class_labels", [])],
            }
            for head_name, head_payload in dict(payload.get("heads", {})).items()
        }

    @classmethod
    def from_payload(cls, payload: Mapping[str, Any]) -> "LoadedRedPolicyModel":
        return cls(payload)

    @classmethod
    def from_path(cls, path: Path) -> "LoadedRedPolicyModel":
        return cls(json.loads(path.read_text(encoding="utf-8")))

    def predict_action(self, transcript: Mapping[str, Any], current_context: Mapping[str, Any]) -> dict[str, Any]:
        feature_row = encode_red_policy_feature_mapping(
            transcript,
            current_context,
            action_space=self.action_space,
            transcript_budget=load_red_context_budget(Path(self.transcript_budget.get("config_path", DEFAULT_RED_CONTEXT_BUDGET_PATH))),
        )
        action: dict[str, str] = {}
        probabilities: dict[str, dict[str, float]] = {}
        for head_name in ("command_family", "timing_bucket", "identity_bucket"):
            head = self.heads[head_name]
            model = head["model"]
            allowed_values = list(head["allowed_values"])
            class_labels = list(head["class_labels"])
            raw_probs = model.predict_proba_one(feature_row)
            head_probs = {value_name: 0.0 for value_name in allowed_values}
            for class_index, prob in zip(class_labels, raw_probs):
                if 0 <= int(class_index) < len(allowed_values):
                    head_probs[allowed_values[int(class_index)]] = float(prob)
            predicted_value = max(head_probs.items(), key=lambda item: item[1])[0]
            action[head_name] = predicted_value
            probabilities[head_name] = head_probs
        protocol_family = _compact_category(current_context.get("protocol_family"), "unknown")
        identity_bucket = action["identity_bucket"]
        return {
            "action": action,
            "head_probabilities": probabilities,
            "simulation_constraints": {
                "protocol_family": protocol_family,
                "platform_family": _compact_category(current_context.get("platform_family"), "unknown"),
                "allowed_source_services": allowed_identity_services(
                    protocol_family,
                    identity_bucket,
                    action_space=self.action_space,
                ),
            },
        }


def run_red_policy_warmstart_training(
    *,
    output_dir: Path,
    protocol_mode: str = DEFAULT_RED_POLICY_TRAINING_CONFIG.protocol_mode,
    rows_per_protocol: int = DEFAULT_RED_POLICY_TRAINING_CONFIG.warm_start_rows_per_protocol,
    seed: int = 7,
    max_history_entries: int | None = None,
    dump_examples: bool = True,
) -> dict[str, Any]:
    resolved_action_space = load_red_action_space()
    resolved_budget = load_red_context_budget()
    examples = build_red_policy_warmstart_examples(
        protocol_mode=protocol_mode,
        rows_per_protocol=rows_per_protocol,
        seed=seed,
        max_history_entries=max_history_entries,
        action_space=resolved_action_space,
        transcript_budget=resolved_budget,
    )
    trained = fit_red_policy_model(
        examples,
        seed=seed,
        action_space=resolved_action_space,
        transcript_budget=resolved_budget,
        training_config=RedPolicyTrainingConfig(
            protocol_mode=normalize_red_policy_protocol_mode(protocol_mode),
            warm_start_rows_per_protocol=rows_per_protocol,
        ),
    )
    payload = export_red_policy_model_payload(trained, action_space=resolved_action_space)
    report = {
        "schema_version": RED_POLICY_MODEL_SCHEMA_VERSION,
        "record_kind": "red_policy_training_report",
        "model_family": RED_POLICY_MODEL_FAMILY,
        "protocol_mode": normalize_red_policy_protocol_mode(protocol_mode),
        "seed": int(seed),
        "rows_per_protocol": int(rows_per_protocol),
        "action_space": trained["action_space"],
        "transcript_budget": trained["transcript_budget"],
        "warm_start": trained["warm_start"],
        "architecture": trained["architecture"],
        "training_config": trained["training_config"],
        "evaluation": trained["evaluation"],
        "artifacts": {
            "model_path": str((output_dir / RED_POLICY_MODEL_ARTIFACT_NAME).resolve()),
            "report_path": str((output_dir / RED_POLICY_REPORT_ARTIFACT_NAME).resolve()),
            "examples_path": str((output_dir / RED_POLICY_EXAMPLES_ARTIFACT_NAME).resolve()),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_json(output_dir / RED_POLICY_MODEL_ARTIFACT_NAME, payload)
    _save_json(output_dir / RED_POLICY_REPORT_ARTIFACT_NAME, report)
    if dump_examples:
        write_red_policy_examples_jsonl(output_dir / RED_POLICY_EXAMPLES_ARTIFACT_NAME, examples)
    return {
        "examples": examples,
        "trained": trained,
        "payload": payload,
        "report": report,
    }
