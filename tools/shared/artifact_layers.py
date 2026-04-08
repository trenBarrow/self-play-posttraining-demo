from __future__ import annotations

from typing import Any, Iterable, Mapping

from tools.shared.canonical_records import (
    build_canonical_command_row,
    validate_canonical_command_rows,
)


class ArtifactLayerError(ValueError):
    """Raised when shared artifact layers cannot be joined safely."""


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


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def artifact_join_key(record: Mapping[str, Any]) -> tuple[int | None, str | None, str | None, str | None]:
    correlation = record.get("correlation")
    if isinstance(correlation, Mapping):
        return (
            _optional_int(correlation.get("run_id")),
            _optional_text(correlation.get("session_id")),
            _optional_text(correlation.get("transaction_id")),
            _optional_text(correlation.get("send_id")),
        )
    return (
        _optional_int(record.get("run_id")),
        _optional_text(record.get("session_id")),
        _optional_text(record.get("transaction_id")) or _optional_text(record.get("txn_id")),
        _optional_text(record.get("send_id")),
    )


def _format_join_key(key: tuple[int | None, str | None, str | None, str | None]) -> str:
    run_id, session_id, transaction_id, send_id = key
    return (
        f"run_id={run_id!r}, "
        f"session_id={session_id!r}, "
        f"transaction_id={transaction_id!r}, "
        f"send_id={send_id!r}"
    )


def extract_recent_behavior_overrides(record: Mapping[str, Any]) -> dict[str, Any]:
    values: dict[str, Any] = {}
    for field_name in (
        "command_rate_1m",
        "error_rate_1m",
        "repeat_command_count_10m",
        "same_target_command_rate_1m",
    ):
        if field_name in record:
            values[field_name] = record.get(field_name)
    return values


def index_recent_behavior_rows(rows: Iterable[Mapping[str, Any]]) -> dict[tuple[int | None, str | None, str | None, str | None], dict[str, Any]]:
    index: dict[tuple[int | None, str | None, str | None, str | None], dict[str, Any]] = {}
    for row in rows:
        key = artifact_join_key(row)
        if key in index:
            raise ArtifactLayerError(
                "Duplicate recent-behavior row for canonical join key "
                f"{_format_join_key(key)}"
            )
        index[key] = extract_recent_behavior_overrides(row)
    return index


def build_canonical_rows_from_raw_transactions(
    raw_transactions: list[dict[str, Any]],
    *,
    recent_behavior_rows: Iterable[Mapping[str, Any]] | None = None,
    require_recent_behavior: bool = False,
) -> list[dict[str, Any]]:
    recent_behavior_index = index_recent_behavior_rows(recent_behavior_rows or [])
    canonical_rows: list[dict[str, Any]] = []
    for raw_transaction in raw_transactions:
        key = artifact_join_key(raw_transaction)
        recent_behavior = recent_behavior_index.get(key)
        if require_recent_behavior and recent_behavior is None:
            raise ArtifactLayerError(
                "Missing recent-behavior row for canonical join key "
                f"{_format_join_key(key)}"
            )
        canonical_rows.append(
            build_canonical_command_row(
                raw_transaction,
                recent_behavior=recent_behavior,
            )
        )
    validate_canonical_command_rows(canonical_rows)
    return canonical_rows
