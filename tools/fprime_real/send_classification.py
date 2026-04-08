#!/usr/bin/env python3
"""Classify live F' command outcomes from sender exceptions and observed events."""

from __future__ import annotations

from typing import Any

TERMINAL_EVENT_NAMES = {"cmdDisp.OpCodeCompleted", "cmdDisp.OpCodeError"}
MISSING_ARTIFACT_EVENT_NAMES = {
    "fileDownlink.FileOpenError",
    "fileManager.DirectoryRemoveError",
    "fileManager.FileMoveError",
    "fileManager.FileSizeError",
}
SPECIAL_REASON_BY_EVENT = {
    "cmdSeq.CS_InvalidMode": "invalid_mode",
}


def classify_send_exception(message: str) -> tuple[int, int, int, int, str]:
    lowered = message.lower()
    if "wasn't in the dictionary" in lowered or "unknown command" in lowered:
        return 0, 0, 0, 1, "unknown_command"
    if "commandargumentsexception" in lowered or "out of range" in lowered or "serialize" in lowered:
        return 0, 0, 0, 2, "arg_reject"
    if "timeout" in lowered:
        return 1, 0, 1, 3, "timeout"
    return 0, 0, 0, 2, "send_error"


def infer_reason_from_events(events: list[dict[str, str]]) -> str:
    for event in events:
        name = event["name"]
        text = event["display_text"].lower()
        if name in MISSING_ARTIFACT_EVENT_NAMES:
            return "missing_artifact"
        if name in SPECIAL_REASON_BY_EVENT:
            return SPECIAL_REASON_BY_EVENT[name]
        if "could not open file" in text or "does not exist" in text or "no such file" in text:
            return "missing_artifact"

    for event in events:
        if event["name"] != "cmdDisp.OpCodeError":
            continue
        text = event["display_text"].lower()
        if "format_error" in text:
            return "format_error"
        if "execution_error" in text:
            return "execution_error"
        return "opcode_error"

    for event in events:
        severity = event["severity"].lower()
        if "warning" in severity or "fatal" in severity:
            return "warning_event"

    return "completed"


def classify_event_history(events: list[dict[str, str]]) -> tuple[int, int, int, int, str]:
    names = {event["name"] for event in events}
    has_error = "cmdDisp.OpCodeError" in names
    has_completed = "cmdDisp.OpCodeCompleted" in names
    reason = infer_reason_from_events(events)

    if has_error:
        return 1, 0, 0, 2, reason
    if has_completed:
        if reason != "completed":
            return 1, 0, 0, 2, reason
        return 1, 1, 0, 0, "completed"
    return 1, 0, 1, 3, "timeout"


def compact_event_names(events: list[dict[str, Any]]) -> list[str]:
    return [str(event.get("name", "")) for event in events]
