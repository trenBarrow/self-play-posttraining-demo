#!/usr/bin/env python3
from __future__ import annotations

import json
from pathlib import Path
from typing import Any

from runtime import class_name, stable_token_id
from tools.shared.canonical_records import (
    canonicalize_legacy_fprime_transaction,
    validate_canonical_command_row,
    validate_canonical_command_rows,
)
from tools.shared.feature_policy import (
    BLUE_FEATURE_POLICY_POSTER_DEFAULT,
    extract_blue_model_features,
    load_blue_feature_policy,
)

POSTER_DEFAULT_TRAINING_PATH_NAME = "poster_default_canonical"
POSTER_DEFAULT_TRAINING_PATH_LABEL = "poster-default canonical neural path"
POSTER_DEFAULT_PRIMARY_MODEL_FEATURE_TIER = "poster_canonical_request"

POSTER_DEFAULT_COMMAND_BASELINE_FEATURE_NAMES = [
    "command_semantics.canonical_command_name",
    "command_semantics.canonical_command_family",
    "command_semantics.mutation_scope",
    "command_semantics.authority_level",
    "command_semantics.target_scope",
    "argument_profile.argument_leaf_count",
]

POSTER_DEFAULT_REQUEST_TUPLE_PURITY_FEATURE_NAMES = [
    "command_semantics.canonical_command_family",
    "command_semantics.mutation_scope",
    "command_semantics.authority_level",
    "command_semantics.target_scope",
    "argument_profile.argument_leaf_count",
]

POSTER_DEFAULT_REQUEST_TUPLE_PURITY_BUCKETS = {
    "command_semantics.canonical_command_family": 1.0,
    "command_semantics.mutation_scope": 1.0,
    "command_semantics.authority_level": 1.0,
    "command_semantics.target_scope": 1.0,
    "argument_profile.argument_leaf_count": 1.0,
}


def poster_default_model_feature_names() -> list[str]:
    return list(load_blue_feature_policy(BLUE_FEATURE_POLICY_POSTER_DEFAULT)["allowed_features"])


POSTER_DEFAULT_REQUEST_ONLY_BASELINE_FEATURE_NAMES = poster_default_model_feature_names()


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    records: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise SystemExit(f"{path}:{line_number} must contain JSON objects")
            records.append(payload)
    return records


def canonical_training_record_path(dataset_path: Path) -> Path:
    return dataset_path.resolve().with_name("canonical_command_rows.jsonl")


def transaction_training_record_path(dataset_path: Path) -> Path:
    return dataset_path.resolve().with_name("transactions.jsonl")


def _recent_behavior_overrides_from_legacy_transaction(transaction: dict[str, Any]) -> dict[str, Any]:
    return {
        "command_rate_1m": transaction.get("command_rate_1m"),
        "error_rate_1m": transaction.get("error_rate_1m"),
        "repeat_command_count_10m": transaction.get("repeat_command_count_10m"),
        "same_target_command_rate_1m": transaction.get("same_target_command_rate_1m"),
    }


def load_canonical_training_records(dataset_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    dataset_path = dataset_path.resolve()
    canonical_path = dataset_path if dataset_path.name == "canonical_command_rows.jsonl" else canonical_training_record_path(dataset_path)
    if canonical_path.exists():
        rows = validate_canonical_command_rows(_read_jsonl(canonical_path))
        return rows, {
            "record_source": "canonical_command_rows",
            "record_path": str(canonical_path),
            "derived_from_legacy_transactions": False,
        }

    transaction_path = transaction_training_record_path(dataset_path)
    if not transaction_path.exists():
        raise SystemExit(
            "Poster-default training needs canonical_command_rows.jsonl or transactions.jsonl next to the dataset. "
            f"dataset={dataset_path}"
        )
    transactions = _read_jsonl(transaction_path)
    rows = [
        canonicalize_legacy_fprime_transaction(
            transaction,
            recent_behavior=_recent_behavior_overrides_from_legacy_transaction(transaction),
        )
        for transaction in transactions
    ]
    validate_canonical_command_rows(rows)
    return rows, {
        "record_source": "derived_from_legacy_transactions",
        "record_path": str(transaction_path),
        "derived_from_legacy_transactions": True,
    }


def encode_feature_value(value: Any) -> float:
    if value is None:
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float)):
        return float(value)
    return float(stable_token_id(str(value), 8192))


def encode_feature_mapping(mapping: dict[str, Any]) -> dict[str, float]:
    return {
        str(name): encode_feature_value(value)
        for name, value in mapping.items()
    }


def canonical_row_metadata(canonical_row: dict[str, Any]) -> dict[str, Any]:
    validated = validate_canonical_command_row(canonical_row)
    actor_context = dict(validated.get("actor_context") or {})
    mission_context = dict(validated.get("mission_context") or {})
    command_semantics = dict(validated.get("command_semantics") or {})
    audit_context = dict(validated.get("audit_context") or {})
    label = int(audit_context.get("label")) if audit_context.get("label") not in (None, "") else 0
    label_name = str(audit_context.get("label_name") or class_name(label))
    command_name = (
        command_semantics.get("canonical_command_name")
        or audit_context.get("raw_command_name")
        or command_semantics.get("canonical_command_family")
        or "unknown"
    )
    service_name = (
        command_semantics.get("canonical_command_family")
        or audit_context.get("raw_service_name")
        or "unknown"
    )
    return {
        "run_id": int(validated.get("run_id", -1) if validated.get("run_id") not in (None, "") else -1),
        "episode_id": int(validated.get("episode_id", -1) if validated.get("episode_id") not in (None, "") else -1),
        "episode_label": label,
        "episode_kind": label_name,
        "label": label,
        "label_name": label_name,
        "command": str(command_name),
        "service": str(service_name),
        "attack_family": str(audit_context.get("attack_family") or "none"),
        "phase": str(mission_context.get("mission_phase") or ""),
        "actor": str(audit_context.get("actor_id") or "unknown"),
        "actor_role": str(actor_context.get("role") or "unknown"),
        "session_id": str(audit_context.get("session_id") or ""),
        "txn_id": str(audit_context.get("transaction_id") or ""),
        "send_id": str(audit_context.get("send_id") or ""),
        "protocol_family": str(validated.get("protocol_family") or "unknown"),
        "platform_family": str(validated.get("platform_family") or "unknown"),
        "raw_command_name": str(audit_context.get("raw_command_name") or ""),
        "raw_service_name": str(audit_context.get("raw_service_name") or ""),
    }


def canonical_row_to_training_row(
    canonical_row: dict[str, Any],
    *,
    policy_name: str = BLUE_FEATURE_POLICY_POSTER_DEFAULT,
) -> dict[str, Any]:
    metadata = canonical_row_metadata(canonical_row)
    extracted = extract_blue_model_features(
        canonical_row,
        policy_name=policy_name,
        require_all_allowed=True,
    )
    return {
        **metadata,
        **encode_feature_mapping(extracted),
    }


def canonical_rows_to_training_rows(
    canonical_rows: list[dict[str, Any]],
    *,
    policy_name: str = BLUE_FEATURE_POLICY_POSTER_DEFAULT,
) -> list[dict[str, Any]]:
    return [
        canonical_row_to_training_row(canonical_row, policy_name=policy_name)
        for canonical_row in canonical_rows
    ]
