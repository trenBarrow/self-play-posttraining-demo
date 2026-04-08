#!/usr/bin/env python3
"""Send one MAVLink action from inside a source container and record its observed outcome."""

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

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.mavlink_real.command_catalog import (
    COMMAND_SPEC_BY_NAME,
    command_family_for_name,
    command_spec_for_name,
    numeric_command_id,
    source_identity_for_service,
)
from tools.mavlink_real.runtime_layout import container_identity_logs_dir, container_identity_send_log_path

try:
    from pymavlink import mavutil

    PYMAVLINK_IMPORT_ERROR: Exception | None = None
except Exception as exc:  # pragma: no cover - depends on caller environment
    mavutil = None
    PYMAVLINK_IMPORT_ERROR = exc


def utc_now() -> str:
    return datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z")


def unix_ms() -> int:
    return int(time.time() * 1000)


def parse_target_endpoint(endpoint: str) -> tuple[str, str, int]:
    value = str(endpoint).strip()
    parts = value.split(":")
    if len(parts) != 3:
        raise SystemExit(f"Unsupported MAVLink target endpoint {endpoint!r}; expected scheme:host:port")
    scheme = parts[0].strip().lower()
    host = parts[1].strip()
    try:
        port = int(parts[2])
    except ValueError as exc:
        raise SystemExit(f"Unsupported MAVLink target endpoint {endpoint!r}; invalid port") from exc
    if scheme not in {"tcp", "tcpin", "tcpout", "udp", "udpin", "udpout"}:
        raise SystemExit(f"Unsupported MAVLink endpoint scheme {scheme!r}")
    if not host or port <= 0:
        raise SystemExit(f"Unsupported MAVLink target endpoint {endpoint!r}")
    return scheme, host, port


def resolve_source_ip(target_host: str, target_port: int) -> str:
    last_error: Exception | None = None
    try:
        addrinfo = socket.getaddrinfo(target_host, target_port, socket.AF_INET, socket.SOCK_STREAM)
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


def resolve_target_ip(target_host: str) -> str:
    last_error: Exception | None = None
    try:
        addrinfo = socket.getaddrinfo(target_host, 0, socket.AF_INET, socket.SOCK_STREAM)
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
    target_endpoint: str,
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
            target_endpoint,
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
        handle.write(json.dumps(payload, separators=(",", ":"), sort_keys=True) + "\n")
        handle.flush()
        os.fsync(handle.fileno())
        fcntl.flock(handle.fileno(), fcntl.LOCK_UN)


def parse_json_obj(value: str, *, label: str) -> dict[str, Any]:
    payload = json.loads(value)
    if not isinstance(payload, dict):
        raise SystemExit(f"{label} must decode to a JSON object")
    return payload


def enum_name(enum_key: str, value: Any) -> str:
    if mavutil is None:
        return ""
    try:
        entry = mavutil.mavlink.enums[enum_key][int(value)]
    except Exception:
        return ""
    return str(getattr(entry, "name", ""))


def compact_message(message_name: str, payload: dict[str, Any]) -> dict[str, Any]:
    summary: dict[str, Any] = {"type": message_name}
    if message_name == "COMMAND_ACK":
        result = payload.get("result")
        summary["command"] = payload.get("command")
        summary["result"] = result
        summary["result_name"] = enum_name("MAV_RESULT", result)
    elif message_name == "PARAM_VALUE":
        summary["param_id"] = str(payload.get("param_id", "")).strip("\x00")
        summary["param_value"] = payload.get("param_value")
        summary["param_type"] = payload.get("param_type")
    elif message_name == "MISSION_ACK":
        result = payload.get("type")
        summary["type_code"] = result
        summary["type_name"] = enum_name("MAV_MISSION_RESULT", result)
        summary["mission_type"] = payload.get("mission_type")
    elif message_name == "MISSION_COUNT":
        summary["count"] = payload.get("count")
        summary["mission_type"] = payload.get("mission_type")
    elif message_name == "AUTOPILOT_VERSION":
        summary["flight_sw_version"] = payload.get("flight_sw_version")
        summary["middleware_sw_version"] = payload.get("middleware_sw_version")
        summary["capabilities"] = payload.get("capabilities")
    elif message_name == "HEARTBEAT":
        summary["autopilot"] = payload.get("autopilot")
        summary["base_mode"] = payload.get("base_mode")
        summary["custom_mode"] = payload.get("custom_mode")
        summary["type_code"] = payload.get("type")
        summary["system_status"] = payload.get("system_status")
    return summary


def payload_to_command_long_params(command_name: str, arguments: dict[str, Any]) -> tuple[float, float, float, float, float, float, float]:
    if command_name == "REQUEST_AUTOPILOT_CAPABILITIES":
        return (
            float(arguments.get("param1", 1.0)),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
    if command_name == "MAV_CMD_DO_SET_MODE":
        return (
            float(arguments.get("base_mode", 1.0)),
            float(arguments.get("custom_mode", 0.0)),
            float(arguments.get("custom_submode", 0.0)),
            0.0,
            0.0,
            0.0,
            0.0,
        )
    if command_name == "MAV_CMD_COMPONENT_ARM_DISARM":
        return (
            float(arguments.get("param1", 0.0)),
            float(arguments.get("param2", 0.0)),
            0.0,
            0.0,
            0.0,
            0.0,
            0.0,
        )
    return (0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0)


def param_id_bytes(param_id: str) -> bytes:
    return str(param_id).encode("ascii", "ignore")


def open_connection(endpoint: str, source_service: str, timeout_seconds: float) -> tuple[Any, dict[str, Any]]:
    if PYMAVLINK_IMPORT_ERROR is not None:
        raise SystemExit(f"pymavlink is required inside the MAVLink sender container: {PYMAVLINK_IMPORT_ERROR}")
    identity = source_identity_for_service(source_service)
    master = mavutil.mavlink_connection(
        endpoint,
        source_system=identity.source_system_id,
        source_component=identity.source_component_id,
        autoreconnect=False,
        dialect="ardupilotmega",
    )
    heartbeat = master.wait_heartbeat(timeout=timeout_seconds)
    if heartbeat is None:
        raise RuntimeError(f"Timed out waiting for heartbeat from {endpoint}")
    payload = heartbeat.to_dict() if hasattr(heartbeat, "to_dict") else {}
    return master, payload


def send_request(master: Any, command_name: str, arguments: dict[str, Any], *, target_system: int, target_component: int) -> None:
    spec = command_spec_for_name(command_name)
    if spec.send_kind == "command_long":
        params = payload_to_command_long_params(command_name, arguments)
        master.mav.command_long_send(
            target_system,
            target_component,
            numeric_command_id(command_name),
            0,
            *params,
        )
        return
    if spec.send_kind == "param_request_read":
        master.mav.param_request_read_send(
            target_system,
            target_component,
            param_id_bytes(str(arguments.get("param_id", ""))),
            int(arguments.get("param_index", -1)),
        )
        return
    if spec.send_kind == "param_request_list":
        master.mav.param_request_list_send(target_system, target_component)
        return
    if spec.send_kind == "param_set":
        master.mav.param_set_send(
            target_system,
            target_component,
            param_id_bytes(str(arguments.get("param_id", ""))),
            float(arguments.get("param_value", 0.0)),
            int(arguments.get("param_type", 9)),
        )
        return
    if spec.send_kind == "mission_request_list":
        master.mav.mission_request_list_send(
            target_system,
            target_component,
            int(arguments.get("mission_type", 0)),
        )
        return
    if spec.send_kind == "mission_clear_all":
        master.mav.mission_clear_all_send(
            target_system,
            target_component,
            int(arguments.get("mission_type", 0)),
        )
        return
    raise RuntimeError(f"Unsupported send kind {spec.send_kind!r} for command {command_name!r}")


def matches_expected_response(message_name: str, payload: dict[str, Any], spec: Any, command_name: str, arguments: dict[str, Any]) -> bool:
    if message_name != spec.response_name:
        return False
    if message_name == "COMMAND_ACK":
        return int(payload.get("command", -1)) == numeric_command_id(command_name)
    if message_name == "PARAM_VALUE":
        expected_param_id = str(arguments.get("param_id", "")).strip()
        if expected_param_id:
            actual = str(payload.get("param_id", "")).strip("\x00")
            return actual == expected_param_id
        return True
    return True


def wait_for_response(master: Any, command_name: str, arguments: dict[str, Any], *, timeout_seconds: float) -> tuple[str, dict[str, Any], list[dict[str, Any]], bool]:
    spec = command_spec_for_name(command_name)
    deadline = time.time() + timeout_seconds
    observed_messages: list[dict[str, Any]] = []
    while time.time() < deadline:
        timeout = max(0.05, min(0.75, deadline - time.time()))
        message = master.recv_match(blocking=True, timeout=timeout)
        if message is None:
            continue
        message_name = message.get_type()
        if message_name in {"BAD_DATA", "UNKNOWN"}:
            continue
        payload = message.to_dict() if hasattr(message, "to_dict") else {}
        observed_messages.append(compact_message(message_name, payload))
        if matches_expected_response(message_name, payload, spec, command_name, arguments):
            return message_name, payload, observed_messages, False
    return "", {}, observed_messages, True


def response_code_and_text(message_name: str, payload: dict[str, Any]) -> tuple[str, str]:
    if message_name == "COMMAND_ACK":
        result = payload.get("result")
        return str(result), enum_name("MAV_RESULT", result)
    if message_name == "MISSION_ACK":
        result = payload.get("type")
        return str(result), enum_name("MAV_MISSION_RESULT", result)
    if message_name == "PARAM_VALUE":
        return str(payload.get("param_type", "")), str(payload.get("param_id", "")).strip("\x00")
    return "", ""


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--target-endpoint", required=True)
    parser.add_argument("--timeout-seconds", type=float, default=8.0)
    parser.add_argument("--virtual-day", type=int, required=True)
    parser.add_argument("--virtual-time", required=True)
    parser.add_argument("--virtual-seconds", type=int, required=True)
    parser.add_argument("--source-service", required=True)
    parser.add_argument("--target-service", required=True)
    parser.add_argument("--target-stream-id", required=True)
    parser.add_argument("--target-stream-index", type=int, required=True)
    parser.add_argument("--logs-dir", default="")
    parser.add_argument("--send-log-path", default="")
    parser.add_argument("--command", required=True)
    parser.add_argument("--command-family", required=True)
    parser.add_argument("--arguments-json", required=True)
    parser.add_argument("--meta-json", required=True)
    args = parser.parse_args()

    if args.timeout_seconds <= 0:
        raise SystemExit("--timeout-seconds must be > 0")

    command_name = str(args.command).strip()
    if command_name not in COMMAND_SPEC_BY_NAME:
        raise SystemExit(f"Unsupported MAVLink command {command_name!r}")
    if str(args.command_family).strip() != command_family_for_name(command_name):
        raise SystemExit(
            f"Command family mismatch for {command_name!r}: got {args.command_family!r}, expected {command_family_for_name(command_name)!r}"
        )

    arguments = parse_json_obj(args.arguments_json, label="--arguments-json")
    meta = parse_json_obj(args.meta_json, label="--meta-json")

    identity = source_identity_for_service(args.source_service)
    _, target_host, target_port = parse_target_endpoint(args.target_endpoint)
    logs_dir = Path(args.logs_dir) if args.logs_dir else container_identity_logs_dir(args.source_service)
    send_log_path = Path(args.send_log_path) if args.send_log_path else container_identity_send_log_path(args.source_service)
    logs_dir.mkdir(parents=True, exist_ok=True)

    arguments_json = json.dumps(arguments, separators=(",", ":"), sort_keys=True)
    args_sha256 = hashlib.sha256(arguments_json.encode("utf-8")).hexdigest()
    source_ip = resolve_source_ip(target_host, target_port)
    target_ip = resolve_target_ip(target_host)
    real_iso = utc_now()
    send_start_ms = unix_ms()
    send_id = stable_send_id(
        source_service=args.source_service,
        target_service=args.target_service,
        target_endpoint=args.target_endpoint,
        command=command_name,
        arguments_json=arguments_json,
        virtual_day=args.virtual_day,
        virtual_seconds=args.virtual_seconds,
        send_start_ms=send_start_ms,
    )

    heartbeat_payload: dict[str, Any] = {}
    response_name = ""
    response_payload: dict[str, Any] = {}
    observed_messages: list[dict[str, Any]] = []
    timeout = 0
    send_exception = ""
    target_system_id = 0
    target_component_id = 0
    master: Any | None = None

    try:
        master, heartbeat_payload = open_connection(args.target_endpoint, args.source_service, args.timeout_seconds)
        target_system_id = int(getattr(master, "target_system", 0) or 0)
        target_component_id = int(getattr(master, "target_component", 0) or 0)
        if target_system_id <= 0:
            target_system_id = 1
        if target_component_id <= 0:
            target_component_id = 1
        send_request(
            master,
            command_name,
            arguments,
            target_system=target_system_id,
            target_component=target_component_id,
        )
        response_name, response_payload, observed_messages, timed_out = wait_for_response(
            master,
            command_name,
            arguments,
            timeout_seconds=args.timeout_seconds,
        )
        timeout = int(timed_out)
    except Exception as exc:  # pragma: no cover - live-runtime dependent
        send_exception = f"{type(exc).__name__}: {exc}"
        timeout = 1
    finally:
        try:
            if master is not None:
                master.close()
        except Exception:
            pass

    send_end_ms = unix_ms()
    response_code, response_text = response_code_and_text(response_name, response_payload)
    meta_run_id = meta.get("run_id", -1)
    row = {
        "real_iso": real_iso,
        "real_ms": send_start_ms,
        "send_start_ms": send_start_ms,
        "send_end_ms": send_end_ms,
        "virtual_day": args.virtual_day,
        "virtual_time": args.virtual_time,
        "virtual_seconds": args.virtual_seconds,
        "source_service": args.source_service,
        "source_ip": source_ip,
        "source_system_id": identity.source_system_id,
        "source_component_id": identity.source_component_id,
        "target_service": args.target_service,
        "target_ip": target_ip,
        "target_endpoint": args.target_endpoint,
        "target_system_id": target_system_id,
        "target_component_id": target_component_id,
        "target_stream_id": args.target_stream_id,
        "target_stream_index": args.target_stream_index,
        "send_id": send_id,
        "args_sha256": args_sha256,
        "command": command_name,
        "command_family": args.command_family,
        "arguments_json": arguments_json,
        "meta_json": json.dumps(meta, separators=(",", ":"), sort_keys=True),
        "run_id": int(meta_run_id) if meta_run_id not in (None, "") else -1,
        "heartbeat_type": heartbeat_payload.get("type", ""),
        "heartbeat_autopilot": heartbeat_payload.get("autopilot", ""),
        "heartbeat_base_mode": heartbeat_payload.get("base_mode", ""),
        "heartbeat_custom_mode": heartbeat_payload.get("custom_mode", ""),
        "response_name": response_name,
        "response_code": response_code,
        "response_text": response_text,
        "response_json": json.dumps(response_payload, separators=(",", ":"), sort_keys=True),
        "observed_messages_json": json.dumps(observed_messages, separators=(",", ":"), sort_keys=True),
        "timeout": timeout,
        "latency_ms": max(0, send_end_ms - send_start_ms),
        "send_exception": send_exception,
    }
    append_jsonl(send_log_path, row)
    print(json.dumps(row, separators=(",", ":"), sort_keys=True))
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
