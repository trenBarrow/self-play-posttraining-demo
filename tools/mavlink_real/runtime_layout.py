#!/usr/bin/env python3
"""Shared host/container runtime layout for real MAVLink runs."""

from __future__ import annotations

from pathlib import Path

CONTAINER_RUNTIME_ROOT = Path("/runtime_root")
CONTAINER_AUTOPILOT_ROOT = Path("/ardupilot")

MAVLINK_NETWORK_NAME = "mavlink_real_net"
VEHICLE_SERVICE = "mavlink_vehicle"
GCS_SERVICE = "mavlink_gcs"
IDENTITY_SERVICES = (
    "ops_primary",
    "ops_secondary",
    "red_primary",
    "red_secondary",
)
SERVICE_IP_BY_NAME = {
    VEHICLE_SERVICE: "192.168.164.2",
    GCS_SERVICE: "192.168.164.3",
    "ops_primary": "192.168.164.12",
    "ops_secondary": "192.168.164.13",
    "red_primary": "192.168.164.22",
    "red_secondary": "192.168.164.23",
}


def default_host_runtime_root(repo_root: Path) -> Path:
    return repo_root / "gds" / "mavlink_runtime"


def runtime_root_for_output(output_dir: Path) -> Path:
    return output_dir / "mavlink_real"


def host_logs_dir(runtime_root: Path) -> Path:
    return runtime_root / "logs"


def host_captures_dir(runtime_root: Path) -> Path:
    return runtime_root / "captures"


def host_capture_pcap_path(runtime_root: Path, capture_name: str = "mavlink_run") -> Path:
    return host_captures_dir(runtime_root) / f"{capture_name}.pcap"


def host_schedules_dir(runtime_root: Path) -> Path:
    return runtime_root / "schedules"


def host_state_exports_dir(runtime_root: Path) -> Path:
    return runtime_root / "state_exports"


def host_metadata_dir(runtime_root: Path) -> Path:
    return runtime_root / "metadata"


def host_bootstrap_metadata_path(runtime_root: Path) -> Path:
    return host_metadata_dir(runtime_root) / "bootstrap_metadata.json"


def host_startup_metadata_path(runtime_root: Path) -> Path:
    return host_metadata_dir(runtime_root) / "startup_metadata.json"


def host_vehicle_dir(runtime_root: Path) -> Path:
    return runtime_root / "vehicle"


def host_vehicle_logs_dir(runtime_root: Path) -> Path:
    return host_vehicle_dir(runtime_root) / "logs"


def host_vehicle_state_dir(runtime_root: Path) -> Path:
    return host_vehicle_dir(runtime_root) / "state"


def host_vehicle_stdout_log_path(runtime_root: Path) -> Path:
    return host_vehicle_logs_dir(runtime_root) / "vehicle.stdout.log"


def host_gcs_dir(runtime_root: Path) -> Path:
    return runtime_root / "gcs"


def host_gcs_logs_dir(runtime_root: Path) -> Path:
    return host_gcs_dir(runtime_root) / "logs"


def host_gcs_state_dir(runtime_root: Path) -> Path:
    return host_gcs_dir(runtime_root) / "state"


def host_gcs_stdout_log_path(runtime_root: Path) -> Path:
    return host_gcs_logs_dir(runtime_root) / "mavproxy.stdout.log"


def host_identity_dir(runtime_root: Path, identity_service: str) -> Path:
    return runtime_root / "identities" / identity_service


def host_identity_logs_dir(runtime_root: Path, identity_service: str) -> Path:
    return host_identity_dir(runtime_root, identity_service) / "logs"


def host_identity_send_log_path(runtime_root: Path, identity_service: str) -> Path:
    return host_identity_logs_dir(runtime_root, identity_service) / "send_log.jsonl"


def host_schedule_run_logs_dir(runtime_root: Path) -> Path:
    return host_logs_dir(runtime_root) / "schedule_runs"


def container_logs_dir() -> Path:
    return CONTAINER_RUNTIME_ROOT / "logs"


def container_captures_dir() -> Path:
    return CONTAINER_RUNTIME_ROOT / "captures"


def container_schedules_dir() -> Path:
    return CONTAINER_RUNTIME_ROOT / "schedules"


def container_state_exports_dir() -> Path:
    return CONTAINER_RUNTIME_ROOT / "state_exports"


def container_metadata_dir() -> Path:
    return CONTAINER_RUNTIME_ROOT / "metadata"


def container_bootstrap_metadata_path() -> Path:
    return container_metadata_dir() / "bootstrap_metadata.json"


def container_startup_metadata_path() -> Path:
    return container_metadata_dir() / "startup_metadata.json"


def container_vehicle_dir() -> Path:
    return CONTAINER_RUNTIME_ROOT / "vehicle"


def container_vehicle_logs_dir() -> Path:
    return container_vehicle_dir() / "logs"


def container_vehicle_state_dir() -> Path:
    return container_vehicle_dir() / "state"


def container_vehicle_stdout_log_path() -> Path:
    return container_vehicle_logs_dir() / "vehicle.stdout.log"


def container_gcs_dir() -> Path:
    return CONTAINER_RUNTIME_ROOT / "gcs"


def container_gcs_logs_dir() -> Path:
    return container_gcs_dir() / "logs"


def container_gcs_state_dir() -> Path:
    return container_gcs_dir() / "state"


def container_gcs_stdout_log_path() -> Path:
    return container_gcs_logs_dir() / "mavproxy.stdout.log"


def container_identity_dir(identity_service: str) -> Path:
    return CONTAINER_RUNTIME_ROOT / "identities" / identity_service


def container_identity_logs_dir(identity_service: str) -> Path:
    return container_identity_dir(identity_service) / "logs"


def container_identity_send_log_path(identity_service: str) -> Path:
    return container_identity_logs_dir(identity_service) / "send_log.jsonl"


def container_schedule_run_logs_dir() -> Path:
    return container_logs_dir() / "schedule_runs"


def ensure_runtime_tree(runtime_root: Path) -> None:
    host_logs_dir(runtime_root).mkdir(parents=True, exist_ok=True)
    host_schedule_run_logs_dir(runtime_root).mkdir(parents=True, exist_ok=True)
    host_captures_dir(runtime_root).mkdir(parents=True, exist_ok=True)
    host_schedules_dir(runtime_root).mkdir(parents=True, exist_ok=True)
    host_state_exports_dir(runtime_root).mkdir(parents=True, exist_ok=True)
    host_metadata_dir(runtime_root).mkdir(parents=True, exist_ok=True)
    host_vehicle_logs_dir(runtime_root).mkdir(parents=True, exist_ok=True)
    host_vehicle_state_dir(runtime_root).mkdir(parents=True, exist_ok=True)
    host_gcs_logs_dir(runtime_root).mkdir(parents=True, exist_ok=True)
    host_gcs_state_dir(runtime_root).mkdir(parents=True, exist_ok=True)
    for identity_service in IDENTITY_SERVICES:
        host_identity_logs_dir(runtime_root, identity_service).mkdir(parents=True, exist_ok=True)
