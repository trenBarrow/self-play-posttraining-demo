#!/usr/bin/env python3
"""Send one real F' command from inside a source container and record its observed outcome."""

from __future__ import annotations

import argparse
import fcntl
import hashlib
import json
import os
import socket
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from fprime_gds.common.testing_fw.api import IntegrationTestAPI
from fprime_gds.executables.cli import ParserBase, StandardPipelineParser

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.fprime_real.runtime_layout import container_cli_logs_dir, container_send_log_path
from tools.fprime_real.send_classification import classify_event_history, classify_send_exception, compact_event_names, TERMINAL_EVENT_NAMES


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def unix_ms() -> int:
    return int(time.time() * 1000)


def resolve_source_ip(tts_addr: str, tts_port: int) -> str:
    last_error: Exception | None = None
    try:
        addrinfo = socket.getaddrinfo(tts_addr, tts_port, socket.AF_INET, socket.SOCK_STREAM)
    except OSError as exc:
        last_error = exc
        addrinfo = []
    for family, _, _, _, sockaddr in addrinfo:
        sock = socket.socket(family, socket.SOCK_DGRAM)
        try:
            sock.connect((sockaddr[0], sockaddr[1]))
            return str(sock.getsockname()[0])
        except OSError as exc:
            last_error = exc
        finally:
            sock.close()
    try:
        hostname = socket.gethostname()
        fallback_ip = socket.gethostbyname(hostname)
        if fallback_ip:
            return str(fallback_ip)
    except OSError as exc:
        last_error = exc
    if last_error is not None:
        raise SystemExit(f"Could not resolve source IP for sender traffic: {last_error}") from None
    raise SystemExit("Could not resolve source IP for sender traffic")


def resolve_target_ip(tts_addr: str, tts_port: int) -> str:
    last_error: Exception | None = None
    try:
        addrinfo = socket.getaddrinfo(tts_addr, tts_port, socket.AF_INET, socket.SOCK_STREAM)
    except OSError as exc:
        last_error = exc
        addrinfo = []
    for _, _, _, _, sockaddr in addrinfo:
        if sockaddr and sockaddr[0]:
            return str(sockaddr[0])
    if last_error is not None:
        raise SystemExit(f"Could not resolve target IP for sender traffic: {last_error}") from None
    raise SystemExit("Could not resolve target IP for sender traffic")


def stable_send_id(
    *,
    source_service: str,
    target_service: str,
    target_tts_port: int,
    command: str,
    arguments_json: str,
    virtual_day: int,
    virtual_seconds: int,
    send_start_ms: int,
) -> str:
    seed = "|".join(
        [
            source_service,
            target_service,
            str(target_tts_port),
            command,
            arguments_json,
            str(virtual_day),
            str(virtual_seconds),
            str(send_start_ms),
        ]
    )
    return hashlib.sha256(seed.encode("utf-8")).hexdigest()[:24]


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("a", encoding="utf-8") as handle:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX)
        handle.write(json.dumps(payload, separators=(",", ":")) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def parse_json_list(value: str) -> list[str]:
    payload = json.loads(value)
    if not isinstance(payload, list):
        raise SystemExit("--arguments-json must decode to a JSON list")
    return [str(item) for item in payload]


def parse_json_obj(value: str) -> dict[str, Any]:
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise SystemExit("--meta-json must decode to a JSON object")
    return payload


def setup_api(dictionary: str, tts_addr: str, tts_port: int, logs_dir: str) -> IntegrationTestAPI:
    args, _ = ParserBase.parse_args(
        [StandardPipelineParser],
        arguments=[
            "--dictionary",
            dictionary,
            "--tts-addr",
            tts_addr,
            "--tts-port",
            str(tts_port),
            "--logs",
            logs_dir,
            "--log-directly",
        ],
        client=True,
    )
    pipeline = StandardPipelineParser.pipeline_factory(args)
    api = IntegrationTestAPI(pipeline)
    api.setup()
    return api


def compact_events(api: IntegrationTestAPI) -> list[dict[str, str]]:
    events: list[dict[str, str]] = []
    for event in api.get_event_test_history().retrieve():
        payload = event.get_dict()
        events.append(
            {
                "name": str(payload.get("name", "")),
                "severity": str(payload.get("severity", "")),
                "display_text": str(payload.get("display_text", "")),
            }
        )
    return events

def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--dictionary", required=True)
    parser.add_argument("--tts-addr", default="host.docker.internal")
    parser.add_argument("--tts-port", type=int, required=True)
    parser.add_argument("--timeout-seconds", type=float, default=5.0)
    parser.add_argument("--virtual-day", type=int, required=True)
    parser.add_argument("--virtual-time", required=True)
    parser.add_argument("--virtual-seconds", type=int, required=True)
    parser.add_argument("--source-service", required=True)
    parser.add_argument("--target-service", required=True)
    parser.add_argument("--target-tts-port", type=int, required=True)
    parser.add_argument("--target-stream-id", required=True)
    parser.add_argument("--target-stream-index", type=int, required=True)
    parser.add_argument("--logs-dir", default=str(container_cli_logs_dir()))
    parser.add_argument("--send-log-path", default=str(container_send_log_path()))
    parser.add_argument("--command", required=True)
    parser.add_argument("--arguments-json", required=True)
    parser.add_argument("--meta-json", required=True)
    args = parser.parse_args()

    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be > 0")

    arguments = parse_json_list(args.arguments_json)
    meta = parse_json_obj(args.meta_json)
    api = setup_api(args.dictionary, args.tts_addr, args.tts_port, args.logs_dir)
    time.sleep(0.5)

    arguments_json = json.dumps(arguments, separators=(",", ":"))
    args_sha256 = hashlib.sha256(arguments_json.encode("utf-8")).hexdigest()
    real_iso = utc_now()
    send_start_ms = unix_ms()
    source_ip = resolve_source_ip(args.tts_addr, args.tts_port)
    target_ip = resolve_target_ip(args.tts_addr, args.tts_port)
    send_id = stable_send_id(
        source_service=args.source_service,
        target_service=args.target_service,
        target_tts_port=args.target_tts_port,
        command=args.command,
        arguments_json=arguments_json,
        virtual_day=args.virtual_day,
        virtual_seconds=args.virtual_seconds,
        send_start_ms=send_start_ms,
    )
    start = time.time()
    observed_events: list[dict[str, str]] = []
    send_exception = ""

    api.clear_histories()
    try:
        api.send_command(args.command, arguments)
    except Exception as exc:  # pragma: no cover - depends on live F' runtime
        send_exception = f"{type(exc).__name__}: {exc}"
        gds_accept, sat_success, timeout, response_code, reason = classify_send_exception(send_exception)
    else:
        deadline = time.time() + args.timeout_seconds
        while time.time() < deadline:
            observed_events = compact_events(api)
            if any(event["name"] in TERMINAL_EVENT_NAMES for event in observed_events):
                time.sleep(0.05)
                observed_events = compact_events(api)
                break
            time.sleep(0.05)
        else:
            observed_events = compact_events(api)
        gds_accept, sat_success, timeout, response_code, reason = classify_event_history(observed_events)

    send_end_ms = unix_ms()
    meta_run_id = meta.get("run_id", -1)
    row = {
        "real_iso": real_iso,
        "real_ms": send_start_ms,
        "virtual_day": args.virtual_day,
        "virtual_time": args.virtual_time,
        "virtual_seconds": args.virtual_seconds,
        "source_service": args.source_service,
        "source_ip": source_ip,
        "target_service": args.target_service,
        "target_ip": target_ip,
        "target_tts_port": args.target_tts_port,
        "target_stream_id": args.target_stream_id,
        "target_stream_index": args.target_stream_index,
        "tts_port": args.tts_port,
        "send_id": send_id,
        "args_sha256": args_sha256,
        "send_start_ms": send_start_ms,
        "send_end_ms": send_end_ms,
        "command": args.command,
        "arguments_json": arguments_json,
        "meta_json": json.dumps(meta, separators=(",", ":")),
        "run_id": int(meta_run_id) if meta_run_id not in (None, "") else -1,
        "gds_accept": gds_accept,
        "sat_success": sat_success,
        "timeout": timeout,
        "response_code": response_code,
        "reason": reason,
        "event_names_json": json.dumps(compact_event_names(observed_events), separators=(",", ":")),
        "observed_events_json": json.dumps(observed_events, separators=(",", ":")),
        "latency_ms": max(1, int((time.time() - start) * 1000.0)),
        "send_exception": send_exception,
    }
    append_jsonl(Path(args.send_log_path), row)
    print(json.dumps(row, separators=(",", ":")), flush=True)
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(0)


if __name__ == "__main__":
    raise SystemExit(main())
