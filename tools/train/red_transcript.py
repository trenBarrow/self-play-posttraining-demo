#!/usr/bin/env python3
from __future__ import annotations

from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

from runtime import stable_token_id
from tools.shared.canonical_records import build_canonical_command_row, validate_canonical_command_row
from tools.shared.schema import adapt_legacy_fprime_transaction, validate_raw_transaction

RED_TRANSCRIPT_SCHEMA_VERSION = "red_command_transcript.v1"
RED_CONTEXT_BUDGET_SCHEMA_VERSION = "red_context_budget.v1"

REPO_ROOT = Path(__file__).resolve().parents[2]
DEFAULT_RED_CONTEXT_BUDGET_PATH = REPO_ROOT / "configs" / "red_model" / "context_budget.yaml"

INVALID_REASON_MARKERS = (
    "invalid",
    "out_of_range",
    "out of range",
    "serialize",
    "malformed",
    "unsupported",
    "unknown command",
    "unknown_command",
    "bad arguments",
)


class RedTranscriptConfigError(ValueError):
    """Raised when the red transcript budget/config is invalid."""


class RedTranscriptBuildError(ValueError):
    """Raised when a bounded red transcript cannot be built safely."""


def _load_yaml(path: Path) -> dict[str, Any]:
    payload = yaml.safe_load(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RedTranscriptConfigError(f"{path} must contain a YAML object")
    return payload


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


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


def _bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    if isinstance(value, str) and value.strip() in {"0", "1"}:
        return value.strip() == "1"
    return None


def _compact_category(value: Any, default: str) -> str:
    text = (_text(value) or default).strip().lower()
    chars: list[str] = []
    last_was_sep = False
    for char in text:
        if char.isalnum():
            chars.append(char)
            last_was_sep = False
            continue
        if not last_was_sep:
            chars.append("_")
            last_was_sep = True
    compact = "".join(chars).strip("_")
    return compact or default


def _require_string(value: Any, path: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise RedTranscriptConfigError(f"{path} must be a non-empty string")
    return value


def _require_string_list(value: Any, path: str) -> list[str]:
    if not isinstance(value, list) or not value:
        raise RedTranscriptConfigError(f"{path} must be a non-empty list")
    items = []
    for index, item in enumerate(value):
        items.append(_require_string(item, f"{path}[{index}]"))
    return items


def _require_mapping(value: Any, path: str) -> dict[str, Any]:
    if not isinstance(value, dict):
        raise RedTranscriptConfigError(f"{path} must be an object")
    return value


def load_red_context_budget(path: Path | None = None) -> dict[str, Any]:
    config_path = (path or DEFAULT_RED_CONTEXT_BUDGET_PATH).resolve()
    payload = _load_yaml(config_path)
    schema_version = _require_string(payload.get("schema_version"), "schema_version")
    if schema_version != RED_CONTEXT_BUDGET_SCHEMA_VERSION:
        raise RedTranscriptConfigError(
            f"schema_version must be {RED_CONTEXT_BUDGET_SCHEMA_VERSION!r}; got {schema_version!r}"
        )
    record_kind = _require_string(payload.get("record_kind"), "record_kind")
    if record_kind != "red_context_budget":
        raise RedTranscriptConfigError("record_kind must be 'red_context_budget'")
    transcript_schema_version = _require_string(
        payload.get("transcript_schema_version"),
        "transcript_schema_version",
    )
    if transcript_schema_version != RED_TRANSCRIPT_SCHEMA_VERSION:
        raise RedTranscriptConfigError(
            f"transcript_schema_version must be {RED_TRANSCRIPT_SCHEMA_VERSION!r}; "
            f"got {transcript_schema_version!r}"
        )

    transcript_format = _require_mapping(payload.get("transcript_format"), "transcript_format")
    _require_string(transcript_format.get("name"), "transcript_format.name")
    header_field_order = _require_string_list(
        transcript_format.get("header_field_order"),
        "transcript_format.header_field_order",
    )
    per_event_field_order = _require_string_list(
        transcript_format.get("per_event_field_order"),
        "transcript_format.per_event_field_order",
    )
    token_prefixes = _require_mapping(transcript_format.get("token_prefixes"), "transcript_format.token_prefixes")
    special_tokens = _require_mapping(transcript_format.get("special_tokens"), "transcript_format.special_tokens")
    for field_name in [*header_field_order, *per_event_field_order]:
        _require_string(token_prefixes.get(field_name), f"transcript_format.token_prefixes.{field_name}")
    for token_name in ("header_start", "header_end", "event_start", "event_end"):
        _require_string(special_tokens.get(token_name), f"transcript_format.special_tokens.{token_name}")

    limits = _require_mapping(payload.get("limits"), "limits")
    max_history_entries = limits.get("max_history_entries")
    if isinstance(max_history_entries, bool) or not isinstance(max_history_entries, int) or max_history_entries <= 0:
        raise RedTranscriptConfigError("limits.max_history_entries must be a positive integer")
    truncation_policy = _require_string(limits.get("truncation_policy"), "limits.truncation_policy")
    if truncation_policy != "keep_most_recent_preserve_order":
        raise RedTranscriptConfigError(
            "limits.truncation_policy must be 'keep_most_recent_preserve_order'"
        )
    if not isinstance(limits.get("include_header_tokens"), bool):
        raise RedTranscriptConfigError("limits.include_header_tokens must be a boolean")
    token_id_modulo = limits.get("token_id_modulo")
    if isinstance(token_id_modulo, bool) or not isinstance(token_id_modulo, int) or token_id_modulo <= 0:
        raise RedTranscriptConfigError("limits.token_id_modulo must be a positive integer")

    defaults = _require_mapping(payload.get("defaults"), "defaults")
    for field_name in ("unknown_category", "mixed_category"):
        _require_string(defaults.get(field_name), f"defaults.{field_name}")

    buckets = _require_mapping(payload.get("buckets"), "buckets")
    argument_leaf_count = _require_mapping(buckets.get("argument_leaf_count"), "buckets.argument_leaf_count")
    for field_name in ("none_max", "single_max", "few_max", "medium_max"):
        threshold = argument_leaf_count.get(field_name)
        if isinstance(threshold, bool) or not isinstance(threshold, int) or threshold < 0:
            raise RedTranscriptConfigError(f"buckets.argument_leaf_count.{field_name} must be a non-negative integer")
    argument_labels = _require_mapping(argument_leaf_count.get("labels"), "buckets.argument_leaf_count.labels")
    for field_name in ("none", "single", "few", "medium", "many"):
        _require_string(argument_labels.get(field_name), f"buckets.argument_leaf_count.labels.{field_name}")

    actor_identity = _require_mapping(buckets.get("actor_identity"), "buckets.actor_identity")
    for field_name in ("self_label", "ally_label", "other_label", "unknown_label"):
        _require_string(actor_identity.get(field_name), f"buckets.actor_identity.{field_name}")

    coarse_result_class = _require_mapping(buckets.get("coarse_result_class"), "buckets.coarse_result_class")
    for field_name in (
        "success_label",
        "warning_label",
        "rejected_label",
        "failed_label",
        "timeout_label",
        "invalid_label",
        "unknown_label",
    ):
        _require_string(coarse_result_class.get(field_name), f"buckets.coarse_result_class.{field_name}")

    return {
        **payload,
        "config_path": str(config_path),
    }


def _argument_size_bucket(argument_profile: Mapping[str, Any], budget: Mapping[str, Any]) -> str:
    thresholds = _as_mapping(_as_mapping(_as_mapping(budget.get("buckets")).get("argument_leaf_count")))
    labels = _as_mapping(thresholds.get("labels"))
    count = int(_number(argument_profile.get("argument_leaf_count")) or 0)
    if count <= int(thresholds.get("none_max", 0)):
        return _compact_category(labels.get("none"), "none")
    if count <= int(thresholds.get("single_max", 1)):
        return _compact_category(labels.get("single"), "single")
    if count <= int(thresholds.get("few_max", 3)):
        return _compact_category(labels.get("few"), "few")
    if count <= int(thresholds.get("medium_max", 6)):
        return _compact_category(labels.get("medium"), "medium")
    return _compact_category(labels.get("many"), "many")


def coarse_result_class_from_outcome(
    outcome: Mapping[str, Any] | None,
    *,
    budget: Mapping[str, Any] | None = None,
) -> str:
    resolved_budget = budget or load_red_context_budget()
    labels = _as_mapping(_as_mapping(_as_mapping(resolved_budget.get("buckets")).get("coarse_result_class")))
    if not outcome:
        return _compact_category(labels.get("unknown_label"), "unknown")

    accepted = _bool(outcome.get("accepted"))
    executed_successfully = _bool(outcome.get("executed_successfully"))
    timed_out = _bool(outcome.get("timed_out"))
    warning_count = _number(outcome.get("warning_count")) or 0.0
    error_count = _number(outcome.get("error_count")) or 0.0
    raw_reason = (_text(outcome.get("raw_reason")) or "").lower()
    raw_code = _number(outcome.get("raw_code"))

    if timed_out:
        return _compact_category(labels.get("timeout_label"), "timeout")
    if raw_reason and any(marker in raw_reason for marker in INVALID_REASON_MARKERS):
        return _compact_category(labels.get("invalid_label"), "invalid")
    if accepted is False:
        return _compact_category(labels.get("rejected_label"), "rejected")
    if executed_successfully is True:
        if warning_count > 0.0 or error_count > 0.0:
            return _compact_category(labels.get("warning_label"), "warning")
        return _compact_category(labels.get("success_label"), "success")
    if executed_successfully is False:
        return _compact_category(labels.get("failed_label"), "failed")
    if warning_count > 0.0:
        return _compact_category(labels.get("warning_label"), "warning")
    if error_count > 0.0:
        return _compact_category(labels.get("failed_label"), "failed")
    if raw_code is not None and raw_code != 0.0:
        return _compact_category(labels.get("failed_label"), "failed")
    return _compact_category(labels.get("unknown_label"), "unknown")


def _history_key_from_canonical(row: Mapping[str, Any]) -> tuple[str | None, str | None, str | None]:
    audit = _as_mapping(row.get("audit_context"))
    return (
        _text(audit.get("session_id")),
        _text(audit.get("transaction_id")),
        _text(audit.get("send_id")),
    )


def _history_key_from_raw(row: Mapping[str, Any]) -> tuple[str | None, str | None, str | None]:
    correlation = _as_mapping(row.get("correlation"))
    return (
        _text(correlation.get("session_id")),
        _text(correlation.get("transaction_id")),
        _text(correlation.get("send_id")),
    )


def _normalize_history_item(item: Mapping[str, Any], index: int) -> dict[str, Any]:
    if not isinstance(item, Mapping):
        raise RedTranscriptBuildError(f"history[{index}] must be a mapping")

    canonical_row: dict[str, Any] | None = None
    raw_transaction: dict[str, Any] | None = None
    override_result_class = _text(item.get("coarse_result_class") or item.get("result_class"))
    source_kind = "unknown"

    if any(key in item for key in ("canonical_row", "raw_transaction", "legacy_transaction")):
        if item.get("canonical_row") is not None:
            canonical_row = validate_canonical_command_row(dict(_as_mapping(item.get("canonical_row"))))
        if item.get("raw_transaction") is not None:
            raw_transaction = validate_raw_transaction(dict(_as_mapping(item.get("raw_transaction"))))
        if item.get("legacy_transaction") is not None:
            if raw_transaction is not None:
                raise RedTranscriptBuildError(
                    f"history[{index}] cannot provide both raw_transaction and legacy_transaction"
                )
            raw_transaction = validate_raw_transaction(
                adapt_legacy_fprime_transaction(dict(_as_mapping(item.get("legacy_transaction"))))
            )
        if canonical_row is None:
            if raw_transaction is None:
                raise RedTranscriptBuildError(
                    f"history[{index}] must include canonical_row, raw_transaction, or legacy_transaction"
                )
            canonical_row = build_canonical_command_row(raw_transaction)
        if raw_transaction is not None and _history_key_from_canonical(canonical_row) != _history_key_from_raw(raw_transaction):
            raise RedTranscriptBuildError(
                f"history[{index}] canonical_row and raw_transaction do not reference the same session/transaction/send ids"
            )
        source_kind = "wrapped_history"
    elif item.get("record_kind") == "canonical_command_row":
        canonical_row = validate_canonical_command_row(dict(item))
        source_kind = "canonical_command_row"
    elif item.get("record_kind") == "raw_transaction":
        raw_transaction = validate_raw_transaction(dict(item))
        canonical_row = build_canonical_command_row(raw_transaction)
        source_kind = "raw_transaction"
    else:
        raw_transaction = validate_raw_transaction(adapt_legacy_fprime_transaction(dict(item)))
        canonical_row = build_canonical_command_row(raw_transaction)
        source_kind = "legacy_transaction"

    return {
        "source_kind": source_kind,
        "sequence_index": index,
        "canonical_row": canonical_row,
        "raw_transaction": raw_transaction,
        "override_result_class": override_result_class,
    }


def _infer_actor_id(normalized_history: list[dict[str, Any]]) -> str:
    actor_ids = {
        _text(_as_mapping(_as_mapping(item["canonical_row"]).get("audit_context")).get("actor_id"))
        for item in normalized_history
    }
    actor_ids.discard(None)
    if len(actor_ids) == 1:
        return next(iter(actor_ids))
    if not actor_ids:
        raise RedTranscriptBuildError("Cannot infer actor_id because history does not carry actor identifiers")
    raise RedTranscriptBuildError(
        "Mixed-actor history requires an explicit actor_id so the red transcript knows whose commands to keep"
    )


def _identity_bucket(actor_id: str | None, perspective_actor_id: str, actor_aliases: set[str], budget: Mapping[str, Any]) -> str:
    labels = _as_mapping(_as_mapping(_as_mapping(budget.get("buckets")).get("actor_identity")))
    if actor_id is None:
        return _compact_category(labels.get("unknown_label"), "unknown")
    if actor_id == perspective_actor_id:
        return _compact_category(labels.get("self_label"), "self")
    if actor_id in actor_aliases:
        return _compact_category(labels.get("ally_label"), "ally")
    return _compact_category(labels.get("other_label"), "other")


def _eligible_identity_buckets(budget: Mapping[str, Any]) -> set[str]:
    labels = _as_mapping(_as_mapping(_as_mapping(budget.get("buckets")).get("actor_identity")))
    return {
        _compact_category(labels.get("self_label"), "self"),
        _compact_category(labels.get("ally_label"), "ally"),
    }


def build_red_transcript_event(
    canonical_row: Mapping[str, Any],
    *,
    actor_id: str,
    actor_aliases: Iterable[str] | None = None,
    coarse_result_class: str | None = None,
    sequence_index: int | None = None,
    budget: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_budget = budget or load_red_context_budget()
    validated = validate_canonical_command_row(dict(canonical_row))
    actor_alias_set = {str(value) for value in (actor_aliases or []) if _text(value)}
    transcript_format = _as_mapping(resolved_budget.get("transcript_format"))
    field_order = list(transcript_format.get("per_event_field_order") or [])
    token_prefixes = _as_mapping(transcript_format.get("token_prefixes"))
    defaults = _as_mapping(resolved_budget.get("defaults"))
    unknown_category = _compact_category(defaults.get("unknown_category"), "unknown")

    actor_context = _as_mapping(validated.get("actor_context"))
    mission_context = _as_mapping(validated.get("mission_context"))
    command_semantics = _as_mapping(validated.get("command_semantics"))
    argument_profile = _as_mapping(validated.get("argument_profile"))
    audit_context = _as_mapping(validated.get("audit_context"))
    source_actor_id = _text(audit_context.get("actor_id"))

    field_values = {
        "canonical_command_family": _compact_category(
            command_semantics.get("canonical_command_family"),
            "other_or_unknown",
        ),
        "actor_identity_bucket": _identity_bucket(
            source_actor_id,
            actor_id,
            actor_alias_set,
            resolved_budget,
        ),
        "actor_role": _compact_category(actor_context.get("role"), unknown_category),
        "trust_class": _compact_category(actor_context.get("trust_class"), unknown_category),
        "mission_phase": _compact_category(mission_context.get("mission_phase"), unknown_category),
        "window_class": _compact_category(mission_context.get("window_class"), "unspecified"),
        "coarse_result_class": _compact_category(coarse_result_class, unknown_category),
        "mutation_scope": _compact_category(command_semantics.get("mutation_scope"), unknown_category),
        "persistence_class": _compact_category(command_semantics.get("persistence_class"), unknown_category),
        "safety_criticality": _compact_category(command_semantics.get("safety_criticality"), unknown_category),
        "authority_level": _compact_category(command_semantics.get("authority_level"), unknown_category),
        "target_scope": _compact_category(command_semantics.get("target_scope"), "unspecified"),
        "argument_size_bucket": _argument_size_bucket(argument_profile, resolved_budget),
    }
    token_strings = [
        f"{token_prefixes[field_name]}:{field_values[field_name]}"
        for field_name in field_order
    ]
    token_modulo = int(_as_mapping(resolved_budget.get("limits")).get("token_id_modulo", 65535))

    return {
        "sequence_index": -1 if sequence_index is None else int(sequence_index),
        "session_id": _text(audit_context.get("session_id")),
        "transaction_id": _text(audit_context.get("transaction_id")),
        "send_id": _text(audit_context.get("send_id")),
        "actor_id": source_actor_id,
        "field_values": field_values,
        "token_strings": token_strings,
        "token_ids": [int(stable_token_id(token, token_modulo)) for token in token_strings],
        "serialized_line": " ".join(token_strings),
    }


def _header_field_values(events: list[dict[str, Any]], budget: Mapping[str, Any]) -> dict[str, str]:
    transcript_format = _as_mapping(budget.get("transcript_format"))
    header_order = list(transcript_format.get("header_field_order") or [])
    defaults = _as_mapping(budget.get("defaults"))
    mixed_category = _compact_category(defaults.get("mixed_category"), "mixed")
    unknown_category = _compact_category(defaults.get("unknown_category"), "unknown")

    canonical_rows = [_as_mapping(event.get("canonical_row")) for event in events]
    values: dict[str, str] = {}
    for field_name in header_order:
        root_values = {
            _compact_category(row.get(field_name), unknown_category)
            for row in canonical_rows
            if _text(row.get(field_name)) is not None
        }
        if not root_values:
            values[field_name] = unknown_category
        elif len(root_values) == 1:
            values[field_name] = next(iter(root_values))
        else:
            values[field_name] = mixed_category
    return values


def _flatten_transcript_tokens(
    *,
    header_tokens: list[str],
    event_tokens: list[list[str]],
    budget: Mapping[str, Any],
) -> list[str]:
    transcript_format = _as_mapping(budget.get("transcript_format"))
    special_tokens = _as_mapping(transcript_format.get("special_tokens"))
    limits = _as_mapping(budget.get("limits"))
    tokens: list[str] = []
    if bool(limits.get("include_header_tokens", True)) and header_tokens:
        tokens.extend(
            [
                str(special_tokens.get("header_start")),
                *header_tokens,
                str(special_tokens.get("header_end")),
            ]
        )
    for line_tokens in event_tokens:
        tokens.extend(
            [
                str(special_tokens.get("event_start")),
                *line_tokens,
                str(special_tokens.get("event_end")),
            ]
        )
    return tokens


def build_red_transcript(
    history: Iterable[Mapping[str, Any]],
    *,
    actor_id: str | None = None,
    actor_aliases: Iterable[str] | None = None,
    max_history_entries: int | None = None,
    budget: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_budget = budget or load_red_context_budget()
    normalized_history = [
        _normalize_history_item(item, index)
        for index, item in enumerate(history)
    ]
    perspective_actor_id = _text(actor_id) or _infer_actor_id(normalized_history)
    actor_alias_set = {str(value) for value in (actor_aliases or []) if _text(value)}
    limits = _as_mapping(resolved_budget.get("limits"))
    configured_history_budget = int(limits.get("max_history_entries", 8))
    history_budget = int(max_history_entries if max_history_entries is not None else configured_history_budget)
    if history_budget <= 0:
        raise RedTranscriptBuildError("max_history_entries must be positive")
    if history_budget > configured_history_budget:
        raise RedTranscriptBuildError(
            "max_history_entries exceeds configured transcript budget: "
            f"requested={history_budget} configured={configured_history_budget}"
        )

    eligible_history: list[dict[str, Any]] = []
    eligible_buckets = _eligible_identity_buckets(resolved_budget)
    for item in normalized_history:
        canonical_row = _as_mapping(item.get("canonical_row"))
        audit_context = _as_mapping(canonical_row.get("audit_context"))
        bucket = _identity_bucket(
            _text(audit_context.get("actor_id")),
            perspective_actor_id,
            actor_alias_set,
            resolved_budget,
        )
        if bucket in eligible_buckets:
            eligible_history.append(item)
    if history_budget < len(eligible_history):
        included_history = eligible_history[-history_budget:]
    else:
        included_history = list(eligible_history)

    header_fields = _header_field_values(included_history, resolved_budget)
    transcript_format = _as_mapping(resolved_budget.get("transcript_format"))
    token_prefixes = _as_mapping(transcript_format.get("token_prefixes"))
    header_tokens = [
        f"{token_prefixes[field_name]}:{header_fields[field_name]}"
        for field_name in transcript_format.get("header_field_order") or []
    ]

    events: list[dict[str, Any]] = []
    for item in included_history:
        raw_transaction = _as_mapping(item.get("raw_transaction"))
        outcome = _as_mapping(raw_transaction.get("outcome")) if raw_transaction else {}
        coarse_result_class = (
            _text(item.get("override_result_class"))
            or coarse_result_class_from_outcome(outcome, budget=resolved_budget)
        )
        event = build_red_transcript_event(
            _as_mapping(item.get("canonical_row")),
            actor_id=perspective_actor_id,
            actor_aliases=sorted(actor_alias_set),
            coarse_result_class=coarse_result_class,
            sequence_index=int(item.get("sequence_index", -1)),
            budget=resolved_budget,
        )
        event["source_kind"] = str(item.get("source_kind") or "unknown")
        event["submitted_at_ms"] = _number(_as_mapping(raw_transaction.get("timing")).get("submitted_at_ms"))
        events.append(event)

    flattened_tokens = _flatten_transcript_tokens(
        header_tokens=header_tokens,
        event_tokens=[list(event["token_strings"]) for event in events],
        budget=resolved_budget,
    )
    token_modulo = int(limits.get("token_id_modulo", 65535))

    return {
        "schema_version": RED_TRANSCRIPT_SCHEMA_VERSION,
        "record_kind": "red_command_transcript",
        "transcript_format": str(transcript_format.get("name")),
        "budget": {
            "config_path": str(resolved_budget.get("config_path") or DEFAULT_RED_CONTEXT_BUDGET_PATH),
            "max_history_entries": history_budget,
            "truncation_policy": str(limits.get("truncation_policy")),
            "include_header_tokens": bool(limits.get("include_header_tokens", True)),
            "token_id_modulo": token_modulo,
            "ordering_mode": "input_order_preserved",
        },
        "actor_perspective": {
            "logical_id": perspective_actor_id,
            "alias_ids": sorted(actor_alias_set),
        },
        "original_history_count": len(normalized_history),
        "eligible_history_count": len(eligible_history),
        "included_history_count": len(events),
        "truncated_event_count": max(0, len(eligible_history) - len(events)),
        "header_fields": header_fields,
        "header_token_strings": header_tokens,
        "header_token_ids": [int(stable_token_id(token, token_modulo)) for token in header_tokens],
        "events": events,
        "serialized_lines": [str(event["serialized_line"]) for event in events],
        "serialized_text": "\n".join(
            [
                " ".join(header_tokens),
                *[str(event["serialized_line"]) for event in events],
            ]
        ).strip(),
        "flattened_token_strings": flattened_tokens,
        "flattened_token_ids": [int(stable_token_id(token, token_modulo)) for token in flattened_tokens],
    }
