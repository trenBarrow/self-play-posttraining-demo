#!/usr/bin/env python3
"""Run scheduled real MAVLink traffic while keeping request histories attributable per target stream."""

from __future__ import annotations

import argparse
import csv
import json
import socket
import subprocess
import time
from collections import defaultdict
from concurrent.futures import ThreadPoolExecutor
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import sys

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.mavlink_real.runtime_layout import container_identity_logs_dir, container_identity_send_log_path
from tools.mavlink_real.schedule_profiles import load_schedule_csv, parse_hms

RUN_LOG_COLUMNS = [
    "real_iso",
    "real_ms",
    "send_start_ms",
    "send_end_ms",
    "virtual_day",
    "virtual_time",
    "virtual_seconds",
    "source_service",
    "source_ip",
    "source_system_id",
    "source_component_id",
    "target_service",
    "target_ip",
    "target_endpoint",
    "target_system_id",
    "target_component_id",
    "target_stream_id",
    "target_stream_index",
    "send_id",
    "args_sha256",
    "command",
    "command_family",
    "arguments_json",
    "meta_json",
    "run_id",
    "heartbeat_type",
    "heartbeat_autopilot",
    "heartbeat_base_mode",
    "heartbeat_custom_mode",
    "response_name",
    "response_code",
    "response_text",
    "response_json",
    "observed_messages_json",
    "timeout",
    "latency_ms",
    "send_exception",
]

HOST_READY_PORT_BY_SERVICE = {
    "mavlink_vehicle": 5760,
}


@dataclass
class ScheduleEvent:
    time_of_day: str
    virtual_seconds: int
    source_service: str
    target_service: str
    target_endpoint: str
    command: str
    command_family: str
    arguments: dict[str, Any]
    meta: dict[str, Any]


@dataclass
class ExpandedEvent:
    absolute_virtual_seconds: int
    virtual_day: int
    virtual_time: str
    virtual_seconds: int
    source_service: str
    target_service: str
    target_endpoint: str
    command: str
    command_family: str
    arguments: dict[str, Any]
    meta: dict[str, Any]


def wait_for_port(host: str, port: int, timeout_seconds: float) -> bool:
    deadline = time.time() + timeout_seconds
    while time.time() < deadline:
        sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        sock.settimeout(1.0)
        try:
            sock.connect((host, port))
            return True
        except OSError:
            time.sleep(0.5)
        finally:
            sock.close()
    return False


def load_schedule(path: Path) -> list[ScheduleEvent]:
    rows = load_schedule_csv(path)
    events: list[ScheduleEvent] = []
    for row in rows:
        events.append(
            ScheduleEvent(
                time_of_day=str(row["time_of_day"]),
                virtual_seconds=parse_hms(str(row["time_of_day"])),
                source_service=str(row["source_service"]),
                target_service=str(row["target_service"]),
                target_endpoint=str(row["target_endpoint"]),
                command=str(row["command"]),
                command_family=str(row["command_family"]),
                arguments=dict(row["arguments"]),
                meta=dict(row["meta"]),
            )
        )
    if not events:
        raise ValueError(f"No schedule rows found in {path}")
    return events


def expand_schedule(events: list[ScheduleEvent], cycles: int) -> list[ExpandedEvent]:
    expanded: list[ExpandedEvent] = []
    for cycle in range(cycles):
        day_offset = cycle * 86400
        for event in events:
            expanded.append(
                ExpandedEvent(
                    absolute_virtual_seconds=day_offset + event.virtual_seconds,
                    virtual_day=cycle,
                    virtual_time=event.time_of_day,
                    virtual_seconds=event.virtual_seconds,
                    source_service=event.source_service,
                    target_service=event.target_service,
                    target_endpoint=event.target_endpoint,
                    command=event.command,
                    command_family=event.command_family,
                    arguments=event.arguments,
                    meta=event.meta,
                )
            )
    return expanded


def format_duration(seconds: int) -> str:
    days = seconds // 86400
    rem = seconds % 86400
    hours = rem // 3600
    rem %= 3600
    minutes = rem // 60
    sec = rem % 60
    return f"{days}d {hours:02d}:{minutes:02d}:{sec:02d}"


def parse_sender_output(stdout: str) -> dict[str, Any]:
    lines = [line.strip() for line in stdout.splitlines() if line.strip()]
    if not lines:
        raise ValueError("sender produced no stdout")
    return json.loads(lines[-1])


def run_one_event(
    compose_file: Path,
    timeout_seconds: float,
    sender_script: str,
    event: ExpandedEvent,
    *,
    target_stream_id: str,
    target_stream_index: int,
) -> dict[str, Any]:
    cmd = [
        "docker",
        "compose",
        "-f",
        str(compose_file),
        "exec",
        "-T",
        event.source_service,
        "python3",
        sender_script,
        "--target-endpoint",
        event.target_endpoint,
        "--timeout-seconds",
        str(timeout_seconds),
        "--virtual-day",
        str(event.virtual_day),
        "--virtual-time",
        event.virtual_time,
        "--virtual-seconds",
        str(event.virtual_seconds),
        "--source-service",
        event.source_service,
        "--target-service",
        event.target_service,
        "--target-stream-id",
        target_stream_id,
        "--target-stream-index",
        str(target_stream_index),
        "--logs-dir",
        str(container_identity_logs_dir(event.source_service)),
        "--send-log-path",
        str(container_identity_send_log_path(event.source_service)),
        "--command",
        event.command,
        "--command-family",
        event.command_family,
        "--arguments-json",
        json.dumps(event.arguments, separators=(",", ":"), sort_keys=True),
        "--meta-json",
        json.dumps(event.meta, separators=(",", ":"), sort_keys=True),
    ]
    proc = subprocess.run(
        cmd,
        capture_output=True,
        text=True,
        timeout=max(90.0, timeout_seconds + 45.0),
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit(
            f"Sender failed for {event.source_service} -> {event.target_service} ({event.command}) "
            f"(exit {proc.returncode})\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        )
    try:
        row = parse_sender_output(proc.stdout)
    except Exception as exc:
        raise SystemExit(
            f"Sender returned unreadable output for {event.source_service} -> {event.target_service} "
            f"({event.command}): {exc}\nstdout:\n{proc.stdout}\nstderr:\n{proc.stderr}"
        ) from exc
    return row


def run_target_stream(
    compose_file: Path,
    sender_script: str,
    time_scale: float,
    timeout_seconds: float,
    events: list[ExpandedEvent],
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    current_virtual = 0
    if not events:
        return rows
    target_stream_id = f"{events[0].target_service}@{events[0].target_endpoint}"
    for target_stream_index, event in enumerate(events):
        wait_virtual = event.absolute_virtual_seconds - current_virtual
        if wait_virtual > 0:
            time.sleep(wait_virtual / time_scale)
        current_virtual = event.absolute_virtual_seconds
        rows.append(
            run_one_event(
                compose_file,
                timeout_seconds,
                sender_script,
                event,
                target_stream_id=target_stream_id,
                target_stream_index=target_stream_index,
            )
        )
    return rows


def validate_serialized_run_rows(rows: list[dict[str, Any]]) -> None:
    by_stream: dict[str, list[dict[str, Any]]] = defaultdict(list)
    by_target: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in rows:
        target_stream_id = str(row.get("target_stream_id", "")).strip()
        target_service = str(row.get("target_service", "")).strip()
        if target_stream_id:
            by_stream[target_stream_id].append(row)
        if target_service:
            by_target[target_service].append(row)

    violations: list[dict[str, Any]] = []
    for target_stream_id, stream_rows in by_stream.items():
        stream_rows.sort(key=lambda row: int(row.get("send_start_ms", row.get("real_ms", 0)) or 0))
        previous_end_ms: int | None = None
        previous_index: int | None = None
        for row in stream_rows:
            send_start_ms = int(row.get("send_start_ms", row.get("real_ms", 0)) or 0)
            send_end_ms = int(row.get("send_end_ms", row.get("real_ms", 0)) or 0)
            stream_index = int(row.get("target_stream_index", -1) or -1)
            if send_end_ms < send_start_ms:
                violations.append(
                    {
                        "scope": "target_stream",
                        "target_stream_id": target_stream_id,
                        "command": row.get("command", ""),
                        "send_start_ms": send_start_ms,
                        "send_end_ms": send_end_ms,
                        "reason": "negative_send_window",
                    }
                )
            if previous_end_ms is not None and send_start_ms < previous_end_ms:
                violations.append(
                    {
                        "scope": "target_stream",
                        "target_stream_id": target_stream_id,
                        "command": row.get("command", ""),
                        "send_start_ms": send_start_ms,
                        "previous_end_ms": previous_end_ms,
                        "reason": "overlapping_send_window",
                    }
                )
            if previous_index is not None and stream_index <= previous_index:
                violations.append(
                    {
                        "scope": "target_stream",
                        "target_stream_id": target_stream_id,
                        "command": row.get("command", ""),
                        "target_stream_index": stream_index,
                        "previous_index": previous_index,
                        "reason": "non_monotonic_stream_index",
                    }
                )
            previous_end_ms = send_end_ms
            previous_index = stream_index

    for target_service, target_rows in by_target.items():
        target_rows.sort(key=lambda row: int(row.get("send_start_ms", row.get("real_ms", 0)) or 0))
        previous_end_ms: int | None = None
        previous_stream: str | None = None
        for row in target_rows:
            send_start_ms = int(row.get("send_start_ms", row.get("real_ms", 0)) or 0)
            send_end_ms = int(row.get("send_end_ms", row.get("real_ms", 0)) or 0)
            target_stream_id = str(row.get("target_stream_id", ""))
            if previous_end_ms is not None and send_start_ms < previous_end_ms:
                violations.append(
                    {
                        "scope": "target_service",
                        "target_service": target_service,
                        "target_stream_id": target_stream_id,
                        "previous_target_stream_id": previous_stream or "",
                        "command": row.get("command", ""),
                        "send_start_ms": send_start_ms,
                        "previous_end_ms": previous_end_ms,
                        "reason": "overlapping_target_execution",
                    }
                )
            previous_end_ms = send_end_ms
            previous_stream = target_stream_id

    if violations:
        raise SystemExit(
            "Run log violates the serialized-per-target invariant. "
            f"Examples: {json.dumps(violations[:8], separators=(',', ':'))}"
        )


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--compose-file", default="orchestration/docker-compose.mavlink-real.yml")
    parser.add_argument("--schedule", required=True)
    parser.add_argument("--time-scale", type=float, default=7200.0)
    parser.add_argument("--cycles", type=int, default=1)
    parser.add_argument("--timeout-seconds", type=float, default=8.0)
    parser.add_argument("--sender-script", default="/workspace/tools/mavlink_real/send_mavlink_events.py")
    parser.add_argument("--output", required=True)
    args = parser.parse_args()

    if args.time_scale <= 0:
        raise SystemExit("--time-scale must be > 0")
    if args.cycles <= 0:
        raise SystemExit("--cycles must be >= 1")

    compose_file = Path(args.compose_file).resolve()
    schedule_path = Path(args.schedule).resolve()
    output_path = Path(args.output).resolve()

    events = load_schedule(schedule_path)
    expanded = expand_schedule(events, args.cycles)
    required_services = sorted({event.target_service for event in expanded})
    for service in required_services:
        host_port = HOST_READY_PORT_BY_SERVICE.get(service)
        if host_port is None:
            raise SystemExit(f"Missing published host readiness port for target service {service}")
        if not wait_for_port("127.0.0.1", host_port, timeout_seconds=45.0):
            raise SystemExit(f"Timed out waiting for published host port {host_port} to accept connections for {service}")

    grouped: dict[tuple[str, str], list[ExpandedEvent]] = defaultdict(list)
    for event in expanded:
        grouped[(event.target_service, event.target_endpoint)].append(event)
    for service_events in grouped.values():
        service_events.sort(
            key=lambda event: (
                event.absolute_virtual_seconds,
                event.source_service,
                event.command,
            )
        )

    merged_rows: list[dict[str, Any]] = []
    with ThreadPoolExecutor(max_workers=max(1, len(grouped))) as pool:
        futures = [
            pool.submit(
                run_target_stream,
                compose_file,
                args.sender_script,
                args.time_scale,
                args.timeout_seconds,
                service_events,
            )
            for _, service_events in sorted(grouped.items())
        ]
        for future in futures:
            merged_rows.extend(future.result())

    merged_rows.sort(
        key=lambda row: (
            int(row.get("real_ms", 0)),
            int(row.get("virtual_day", 0)),
            int(row.get("virtual_seconds", 0)),
            str(row.get("source_service", "")),
            str(row.get("command", "")),
        )
    )
    validate_serialized_run_rows(merged_rows)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    with output_path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=RUN_LOG_COLUMNS)
        writer.writeheader()
        for row in merged_rows:
            writer.writerow({column: row.get(column, "") for column in RUN_LOG_COLUMNS})

    first_virtual = expanded[0].absolute_virtual_seconds
    last_virtual = expanded[-1].absolute_virtual_seconds
    span = max(0, last_virtual - first_virtual)
    print(f"Wrote schedule run log to {output_path}")
    print(f"rows={len(merged_rows)} virtual_span={format_duration(span)} cycles={args.cycles}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
