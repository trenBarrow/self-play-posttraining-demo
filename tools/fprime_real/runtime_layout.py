#!/usr/bin/env python3
"""Shared host/container runtime layout for real F' runs."""

from __future__ import annotations

from pathlib import Path

CONTAINER_RUNTIME_ROOT = Path("/runtime_root")
TARGET_NODE_BY_SERVICE = {
    "fprime_a": "node_a",
    "fprime_b": "node_b",
}


def default_host_runtime_root(repo_root: Path) -> Path:
    return repo_root / "gds" / "fprime_runtime"


def runtime_root_for_output(output_dir: Path) -> Path:
    return output_dir / "fprime_real"


def container_cli_logs_dir() -> Path:
    return CONTAINER_RUNTIME_ROOT / "cli_logs"


def container_logs_dir() -> Path:
    return CONTAINER_RUNTIME_ROOT / "logs"


def container_send_log_path() -> Path:
    return container_logs_dir() / "send_log.jsonl"


def container_node_log_dir(node_name: str) -> Path:
    return CONTAINER_RUNTIME_ROOT / node_name / "logs"


def host_cli_logs_dir(runtime_root: Path) -> Path:
    return runtime_root / "cli_logs"


def host_logs_dir(runtime_root: Path) -> Path:
    return runtime_root / "logs"


def host_send_log_path(runtime_root: Path) -> Path:
    return host_logs_dir(runtime_root) / "send_log.jsonl"


def host_command_log_path(runtime_root: Path) -> Path:
    return host_cli_logs_dir(runtime_root) / "command.log"


def host_node_log_dir(runtime_root: Path, node_name: str) -> Path:
    return runtime_root / node_name / "logs"


def host_node_out_dir(runtime_root: Path, node_name: str) -> Path:
    return runtime_root / node_name


def host_event_log_path(runtime_root: Path, target_service: str) -> Path:
    return host_node_log_dir(runtime_root, TARGET_NODE_BY_SERVICE[target_service]) / "event.log"


def host_channel_log_path(runtime_root: Path, target_service: str) -> Path:
    return host_node_log_dir(runtime_root, TARGET_NODE_BY_SERVICE[target_service]) / "channel.log"


def host_recv_bin_path(runtime_root: Path, target_service: str) -> Path:
    return host_node_log_dir(runtime_root, TARGET_NODE_BY_SERVICE[target_service]) / "recv.bin"


def host_downlink_records_path(runtime_root: Path, target_service: str) -> Path:
    return host_node_log_dir(runtime_root, TARGET_NODE_BY_SERVICE[target_service]) / "downlink_records.jsonl"


def ensure_runtime_tree(runtime_root: Path) -> None:
    host_cli_logs_dir(runtime_root).mkdir(parents=True, exist_ok=True)
    host_logs_dir(runtime_root).mkdir(parents=True, exist_ok=True)
    for node_name in TARGET_NODE_BY_SERVICE.values():
        host_node_log_dir(runtime_root, node_name).mkdir(parents=True, exist_ok=True)
