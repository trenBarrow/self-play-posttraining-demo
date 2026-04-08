from __future__ import annotations

import heapq
from collections import deque
from typing import Any, Mapping


class RecentBehaviorError(ValueError):
    """Raised when request-time behavior summaries cannot be derived safely."""


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


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return dict(value)
    return {}


def _transaction_join_key(transaction: Mapping[str, Any]) -> tuple[int | None, str | None, str | None, str | None]:
    correlation = _as_mapping(transaction.get("correlation"))
    return (
        _optional_int(correlation.get("run_id")),
        _optional_text(correlation.get("session_id")),
        _optional_text(correlation.get("transaction_id")),
        _optional_text(correlation.get("send_id")),
    )


def _reset_value(transaction: Mapping[str, Any], reset_key: str) -> int | str | None:
    correlation = _as_mapping(transaction.get("correlation"))
    if reset_key == "run_id":
        return _optional_int(correlation.get("run_id"))
    if reset_key == "episode_id":
        return _optional_int(correlation.get("episode_id"))
    value = correlation.get(reset_key)
    text = _optional_text(value)
    if text is not None:
        return text
    return _optional_int(value)


def _submitted_at_ms(transaction: Mapping[str, Any]) -> float:
    timing = _as_mapping(transaction.get("timing"))
    return float(
        _optional_number(timing.get("submitted_at_ms"))
        or _optional_number(timing.get("request_forwarded_at_ms"))
        or _optional_number(timing.get("protocol_response_at_ms"))
        or _optional_number(timing.get("finalized_at_ms"))
        or 0.0
    )


def _finalized_at_ms(transaction: Mapping[str, Any]) -> float:
    timing = _as_mapping(transaction.get("timing"))
    submitted_at_ms = _submitted_at_ms(transaction)
    return float(
        _optional_number(timing.get("finalized_at_ms"))
        or _optional_number(timing.get("protocol_response_at_ms"))
        or submitted_at_ms
    )


def _error_flag(transaction: Mapping[str, Any]) -> int:
    outcome = _as_mapping(transaction.get("outcome"))
    accepted = outcome.get("accepted")
    executed = outcome.get("executed_successfully")
    timed_out = outcome.get("timed_out")
    raw_code = _optional_number(outcome.get("raw_code"))
    if accepted is False or executed is False or timed_out is True:
        return 1
    if raw_code not in (None, 0.0):
        return 1
    return 0


def _target_id(transaction: Mapping[str, Any]) -> str:
    target = _as_mapping(transaction.get("target"))
    return _optional_text(target.get("logical_id")) or "unknown"


def _command_name(transaction: Mapping[str, Any]) -> str:
    command = _as_mapping(transaction.get("command"))
    return _optional_text(command.get("raw_name")) or "unknown"


def _transaction_sort_key(index: int, transaction: Mapping[str, Any]) -> tuple[float, int, int, str, str, int]:
    correlation = _as_mapping(transaction.get("correlation"))
    return (
        _submitted_at_ms(transaction),
        _optional_int(correlation.get("run_id")) or -1,
        _optional_int(correlation.get("stream_index")) or -1,
        _optional_text(correlation.get("transaction_id")) or "",
        _optional_text(correlation.get("send_id")) or "",
        index,
    )


def build_recent_behavior_rows_from_raw_transactions(
    raw_transactions: list[dict[str, Any]],
    *,
    reset_key: str = "run_id",
) -> list[dict[str, Any]]:
    grouped: dict[int | str | None, list[tuple[int, dict[str, Any]]]] = {}
    group_order: list[int | str | None] = []
    for index, transaction in enumerate(raw_transactions):
        group_value = _reset_value(transaction, reset_key)
        if group_value not in grouped:
            grouped[group_value] = []
            group_order.append(group_value)
        grouped[group_value].append((index, transaction))

    rows: list[dict[str, Any]] = []
    for group_value in group_order:
        items = sorted(grouped[group_value], key=lambda item: _transaction_sort_key(item[0], item[1]))
        command_history: deque[float] = deque()
        target_history: deque[tuple[float, str]] = deque()
        repeat_history: deque[tuple[float, str]] = deque()
        error_history: deque[tuple[float, int]] = deque()
        pending_errors: list[tuple[float, int]] = []

        for _, transaction in items:
            submitted_at_ms = _submitted_at_ms(transaction)
            finalized_at_ms = _finalized_at_ms(transaction)

            while pending_errors and pending_errors[0][0] <= submitted_at_ms:
                completed_at_ms, error_flag = heapq.heappop(pending_errors)
                error_history.append((completed_at_ms, error_flag))

            while command_history and submitted_at_ms - command_history[0] > 60000.0:
                command_history.popleft()
            while target_history and submitted_at_ms - target_history[0][0] > 60000.0:
                target_history.popleft()
            while repeat_history and submitted_at_ms - repeat_history[0][0] > 600000.0:
                repeat_history.popleft()
            while error_history and submitted_at_ms - error_history[0][0] > 60000.0:
                error_history.popleft()

            target_id = _target_id(transaction)
            command_name = _command_name(transaction)

            command_history.append(submitted_at_ms)
            target_history.append((submitted_at_ms, target_id))
            repeat_history.append((submitted_at_ms, command_name))

            error_rate_1m = 0.0
            if error_history:
                error_rate_1m = float(sum(value for _, value in error_history) / len(error_history))

            repeat_command_count_10m = max(
                0,
                sum(1 for _, previous_command in repeat_history if previous_command == command_name) - 1,
            )
            same_target_command_rate_1m = float(
                sum(1 for _, previous_target in target_history if previous_target == target_id)
            )

            join_key = _transaction_join_key(transaction)
            rows.append(
                {
                    "run_id": join_key[0],
                    "session_id": join_key[1],
                    "transaction_id": join_key[2],
                    "send_id": join_key[3],
                    "command_rate_1m": float(len(command_history)),
                    "error_rate_1m": error_rate_1m,
                    "repeat_command_count_10m": int(repeat_command_count_10m),
                    "same_target_command_rate_1m": same_target_command_rate_1m,
                }
            )
            heapq.heappush(pending_errors, (finalized_at_ms, _error_flag(transaction)))

    return rows
