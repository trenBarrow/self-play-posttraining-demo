#!/usr/bin/env python3
"""Helpers for reading real MAVLink run logs and per-identity sender logs."""

from __future__ import annotations

import csv
import json
from pathlib import Path
from typing import Any

from tools.mavlink_real.runtime_layout import IDENTITY_SERVICES, host_identity_send_log_path


class LogIngestError(ValueError):
    """Raised when run-log and send-log evidence cannot be merged safely."""


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


def row_meta(row: dict[str, Any]) -> dict[str, Any]:
    raw = row.get("meta_json", "{}")
    try:
        payload = json.loads(raw)
    except (TypeError, ValueError, json.JSONDecodeError):
        payload = {}
    return payload if isinstance(payload, dict) else {}


def row_run_id(row: dict[str, Any]) -> int | None:
    top_level = _optional_int(row.get("run_id"))
    if top_level is not None and top_level >= 0:
        return top_level
    meta_value = _optional_int(row_meta(row).get("run_id"))
    if meta_value is not None and meta_value >= 0:
        return meta_value
    return top_level


def load_schedule_run_rows(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        raise LogIngestError(f"Missing MAVLink run log at {path}")
    with path.open("r", newline="", encoding="utf-8") as handle:
        rows = list(csv.DictReader(handle))
    if not rows:
        raise LogIngestError(f"MAVLink run log is empty at {path}")
    return rows


def load_send_log_rows(path: Path) -> list[dict[str, Any]]:
    if not path.exists():
        return []
    rows: list[dict[str, Any]] = []
    with path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise LogIngestError(f"Unexpected non-object send-log payload in {path}")
            rows.append(payload)
    return rows


def runtime_send_log_paths(runtime_root: Path) -> list[Path]:
    return [
        host_identity_send_log_path(runtime_root, identity_service)
        for identity_service in IDENTITY_SERVICES
    ]


def source_artifact_paths(*paths: Path) -> list[str]:
    values: list[str] = []
    for path in paths:
        if not path.exists():
            continue
        resolved = str(path.resolve())
        if resolved not in values:
            values.append(resolved)
    return values


def index_send_rows(send_rows: list[dict[str, Any]]) -> dict[str, dict[str, Any]]:
    by_send_id: dict[str, dict[str, Any]] = {}
    for row in send_rows:
        send_id = _optional_text(row.get("send_id"))
        if send_id is None:
            raise LogIngestError("Encountered send-log row without send_id")
        if send_id in by_send_id:
            raise LogIngestError(f"Duplicate send_id in sender logs: {send_id}")
        by_send_id[send_id] = row
    return by_send_id


def merge_run_and_send_rows(run_rows: list[dict[str, str]], send_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    send_rows_by_id = index_send_rows(send_rows)
    merged: list[dict[str, Any]] = []
    for run_row in run_rows:
        send_id = _optional_text(run_row.get("send_id"))
        if send_id is None:
            raise LogIngestError("Encountered run-log row without send_id")
        send_row = send_rows_by_id.get(send_id)
        if send_row is None:
            raise LogIngestError(f"Run log references send_id {send_id!r} without matching sender evidence")
        for key in ("source_service", "target_service", "command", "target_endpoint"):
            run_value = _optional_text(run_row.get(key))
            send_value = _optional_text(send_row.get(key))
            if run_value and send_value and run_value != send_value:
                raise LogIngestError(
                    f"Run log / sender log mismatch for send_id={send_id!r} field {key!r}: "
                    f"{run_value!r} != {send_value!r}"
                )
        payload = dict(run_row)
        payload.update(send_row)
        if "run_id" not in payload or _optional_int(payload.get("run_id")) is None:
            payload["run_id"] = row_run_id(payload)
        merged.append(payload)
    return merged


def load_runtime_run_rows(run_log_path: Path, *, runtime_root: Path | None = None) -> tuple[list[dict[str, Any]], list[str]]:
    run_rows = load_schedule_run_rows(run_log_path)
    send_log_paths = runtime_send_log_paths(runtime_root) if runtime_root is not None else []
    send_rows: list[dict[str, Any]] = []
    existing_paths = [run_log_path]
    for path in send_log_paths:
        rows = load_send_log_rows(path)
        if rows:
            send_rows.extend(rows)
            existing_paths.append(path)
    if not send_rows:
        raise LogIngestError(
            "No per-identity sender logs were found for the MAVLink run. "
            f"Checked: {[str(path) for path in send_log_paths]}"
        )
    return merge_run_and_send_rows(run_rows, send_rows), source_artifact_paths(*existing_paths)
