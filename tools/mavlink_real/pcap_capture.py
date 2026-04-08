#!/usr/bin/env python3
"""Portable helpers for strict run-scoped MAVLink packet capture."""

from __future__ import annotations

import os
import re
import shlex
import shutil
import signal
import subprocess
import sys
import time
import uuid
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from typing import Iterator

from tools.mavlink_real.runtime_layout import MAVLINK_NETWORK_NAME

DEFAULT_CAPTURE_FILTER = "(tcp port 5760) or (udp port 14550)"
STARTUP_POLL_TIMEOUT_SECONDS = 3.0
STARTUP_POLL_INTERVAL_SECONDS = 0.05
INTERFACE_LINE_RE = re.compile(r"^\d+\.(?P<name>\S+)")
DEFAULT_VM_NETWORK = MAVLINK_NETWORK_NAME
CAPTURE_BACKEND_AUTO = "auto"
CAPTURE_BACKEND_RDCTL_VM_BRIDGE = "rdctl_vm_bridge"
CAPTURE_BACKEND_LOCAL_DOCKER_BRIDGE = "local_docker_bridge"
SUPPORTED_CAPTURE_BACKENDS = {
    CAPTURE_BACKEND_AUTO,
    CAPTURE_BACKEND_RDCTL_VM_BRIDGE,
    CAPTURE_BACKEND_LOCAL_DOCKER_BRIDGE,
}


@dataclass
class CaptureSession:
    path: Path
    interface: str
    filter_expr: str
    process: subprocess.Popen[bytes] | None = None
    backend: str = CAPTURE_BACKEND_LOCAL_DOCKER_BRIDGE
    remote_pid_file: str | None = None
    remote_log_file: str | None = None


def parse_capture_interfaces(stdout: str) -> list[str]:
    interfaces: list[str] = []
    for raw_line in stdout.splitlines():
        line = raw_line.strip()
        if not line:
            continue
        match = INTERFACE_LINE_RE.match(line)
        if match is None:
            continue
        interfaces.append(match.group("name"))
    return interfaces


def env_capture_interface() -> str:
    return os.environ.get("MAVLINK_CAPTURE_INTERFACE", "").strip()


def env_capture_backend() -> str:
    return os.environ.get("MAVLINK_CAPTURE_BACKEND", CAPTURE_BACKEND_AUTO).strip() or CAPTURE_BACKEND_AUTO


def is_loopback_interface(name: str) -> bool:
    return name.lower().startswith("lo")


def rdctl_binary() -> str | None:
    return shutil.which("rdctl")


def has_rdctl_vm() -> bool:
    return rdctl_binary() is not None


def docker_binary() -> str | None:
    return shutil.which("docker")


def run_rdctl(args: list[str], *, check: bool = False, text: bool = True) -> subprocess.CompletedProcess[str]:
    rdctl = rdctl_binary()
    if rdctl is None:
        raise SystemExit("Strict identity capture requires Rancher Desktop rdctl, but rdctl is not installed")
    proc = subprocess.run(
        [rdctl, "shell", "--", *args],
        capture_output=True,
        text=text,
        check=False,
    )
    if check and proc.returncode != 0:
        stderr = (proc.stderr or "").strip()
        raise SystemExit(f"rdctl shell command failed: {' '.join(args)}{': ' + stderr if stderr else ''}")
    return proc


def ensure_vm_tcpdump() -> None:
    proc = run_rdctl(["sh", "-lc", "command -v tcpdump >/dev/null 2>&1 || sudo apk add --no-cache tcpdump"], check=False)
    if proc.returncode != 0:
        stderr = (proc.stderr or proc.stdout or "").strip()
        raise SystemExit(f"tcpdump is required inside the Rancher VM for strict capture: {stderr}")


def default_vm_bridge_interface(network_name: str = DEFAULT_VM_NETWORK) -> str | None:
    proc = run_rdctl(
        [
            "sh",
            "-lc",
            f"docker network inspect {shlex.quote(network_name)} --format '{{{{.Id}}}}' 2>/dev/null | cut -c1-12",
        ],
        check=False,
    )
    bridge_id = (proc.stdout or "").strip()
    if not bridge_id:
        return None
    return f"br-{bridge_id}"


def default_local_bridge_interface(network_name: str = DEFAULT_VM_NETWORK) -> str | None:
    if not sys.platform.startswith("linux"):
        return None
    docker = docker_binary()
    if docker is None:
        return None
    proc = subprocess.run(
        [docker, "network", "inspect", network_name, "--format", "{{.Id}}"],
        capture_output=True,
        text=True,
        check=False,
    )
    network_id = (proc.stdout or "").strip()
    if proc.returncode != 0 or not network_id:
        return None
    return f"br-{network_id[:12]}"


def local_docker_bridge_available(network_name: str = DEFAULT_VM_NETWORK) -> tuple[bool, str]:
    if not sys.platform.startswith("linux"):
        return False, "native Docker bridge capture is only supported on Linux hosts"
    if docker_binary() is None:
        return False, "docker is not installed"
    if shutil.which("tcpdump") is None:
        return False, "tcpdump is not installed on the host"
    if os.geteuid() != 0 and shutil.which("sudo") is None:
        return False, "tcpdump capture on the local Docker bridge requires root or passwordless sudo"
    bridge = default_local_bridge_interface(network_name)
    if bridge is None:
        return False, f"docker network inspect could not resolve a host-visible bridge for {network_name}"
    return True, bridge


def local_capture_command(path: Path, interface: str, filter_expr: str) -> list[str]:
    cmd = ["tcpdump", "-i", interface, "-U", "-w", str(path), *shlex.split(filter_expr)]
    if os.geteuid() == 0:
        return cmd
    if shutil.which("sudo") is None:
        raise SystemExit("tcpdump capture on the local Docker bridge requires root or passwordless sudo")
    return ["sudo", "-n", *cmd]


def resolve_capture_backend(backend: str | None = None) -> str:
    chosen = (backend or env_capture_backend()).strip() or CAPTURE_BACKEND_AUTO
    if chosen not in SUPPORTED_CAPTURE_BACKENDS:
        raise SystemExit(
            f"Unsupported MAVLINK_CAPTURE_BACKEND={chosen!r}. "
            f"Expected one of {sorted(SUPPORTED_CAPTURE_BACKENDS)}."
        )
    if chosen == CAPTURE_BACKEND_RDCTL_VM_BRIDGE:
        if not has_rdctl_vm():
            raise SystemExit("MAVLINK_CAPTURE_BACKEND=rdctl_vm_bridge requires Rancher Desktop rdctl")
        return chosen
    if chosen == CAPTURE_BACKEND_LOCAL_DOCKER_BRIDGE:
        available, reason = local_docker_bridge_available(DEFAULT_VM_NETWORK)
        if not available:
            raise SystemExit(f"MAVLINK_CAPTURE_BACKEND=local_docker_bridge is unavailable: {reason}")
        return chosen
    if has_rdctl_vm():
        return CAPTURE_BACKEND_RDCTL_VM_BRIDGE
    available, _ = local_docker_bridge_available(DEFAULT_VM_NETWORK)
    if available:
        return CAPTURE_BACKEND_LOCAL_DOCKER_BRIDGE
    if sys.platform == "darwin":
        raise SystemExit(
            "Strict identity capture supports Rancher Desktop via rdctl or native Docker bridge capture on Linux. "
            "Docker Desktop on macOS without rdctl is unsupported in this phase. "
            "Set MAVLINK_CAPTURE_BACKEND and MAVLINK_CAPTURE_INTERFACE explicitly if you have a supported backend."
        )
    raise SystemExit(
        "Could not resolve a supported strict capture backend. "
        "Expected Rancher Desktop via rdctl or a Linux host with a visible Docker bridge interface."
    )


def preferred_capture_interface(backend: str) -> str | None:
    if backend == CAPTURE_BACKEND_RDCTL_VM_BRIDGE:
        return default_vm_bridge_interface(DEFAULT_VM_NETWORK)
    if backend == CAPTURE_BACKEND_LOCAL_DOCKER_BRIDGE:
        return default_local_bridge_interface(DEFAULT_VM_NETWORK)
    return None


def list_capture_interfaces(backend: str | None = None) -> list[str]:
    resolved_backend = resolve_capture_backend(backend)
    if resolved_backend == CAPTURE_BACKEND_RDCTL_VM_BRIDGE:
        ensure_vm_tcpdump()
        proc = run_rdctl(["tcpdump", "-D"], check=False)
        if proc.returncode != 0:
            stderr = (proc.stderr or "").strip()
            raise SystemExit(f"tcpdump is required in the Rancher VM but interface enumeration failed: {stderr}")
        interfaces = parse_capture_interfaces(proc.stdout)
        if not interfaces:
            raise SystemExit("tcpdump inside the Rancher VM did not report any capture interfaces")
        return interfaces

    proc = subprocess.run(
        ["tcpdump", "-D"],
        capture_output=True,
        text=True,
        check=False,
    )
    if proc.returncode != 0:
        raise SystemExit("tcpdump is required for MAVLink capture but could not enumerate interfaces")
    interfaces = parse_capture_interfaces(proc.stdout)
    if not interfaces:
        raise SystemExit("tcpdump did not report any capture interfaces")
    return interfaces


def ordered_capture_interfaces(
    interfaces: list[str],
    override: str | None = None,
    preferred_interface: str | None = None,
) -> list[str]:
    chosen_override = (override or "").strip()
    if chosen_override:
        return [chosen_override]

    ranked: list[tuple[int, int, str]] = []
    for index, name in enumerate(interfaces):
        lower = name.lower()
        if is_loopback_interface(lower):
            continue
        if preferred_interface and name == preferred_interface:
            bucket = 0
        elif lower == "bridge100":
            bucket = 1
        elif lower.startswith("vmenet"):
            bucket = 2
        elif lower == "docker0":
            bucket = 3
        elif lower.startswith("br-"):
            bucket = 4
        elif lower.startswith("cni"):
            bucket = 5
        else:
            bucket = 6
        ranked.append((bucket, index, name))
    ranked.sort(key=lambda item: (item[0], item[1], item[2]))
    return [name for _, _, name in ranked]


def start_vm_capture(path: Path, interface: str, filter_expr: str) -> CaptureSession:
    ensure_vm_tcpdump()
    path.parent.mkdir(parents=True, exist_ok=True)
    remote_path = str(path.resolve())
    capture_id = uuid.uuid4().hex[:12]
    remote_pid_file = f"/tmp/anomaly-capture-{capture_id}.pid"
    remote_log_file = f"/tmp/anomaly-capture-{capture_id}.log"
    tcpdump_cmd = shlex.join(["tcpdump", "-i", interface, "-U", "-w", remote_path, *shlex.split(filter_expr)])
    launch_cmd = (
        f"rm -f {shlex.quote(remote_pid_file)} {shlex.quote(remote_log_file)} && "
        f"sudo rm -f {shlex.quote(remote_path)} && "
        f"sudo sh -lc {shlex.quote(f'{tcpdump_cmd} > {shlex.quote(remote_log_file)} 2>&1 & echo $! > {shlex.quote(remote_pid_file)}')}"
    )
    proc = run_rdctl(["sh", "-lc", launch_cmd], check=False)
    if proc.returncode != 0:
        stderr = (proc.stderr or proc.stdout or "").strip()
        raise SystemExit(f"Failed to start Rancher VM capture on {interface}: {stderr}")

    deadline = time.time() + STARTUP_POLL_TIMEOUT_SECONDS
    started = False
    while time.time() < deadline:
        poll = run_rdctl(
            [
                "sh",
                "-lc",
                f"test -s {shlex.quote(remote_pid_file)} && sudo kill -0 $(cat {shlex.quote(remote_pid_file)}) 2>/dev/null",
            ],
            check=False,
        )
        if poll.returncode == 0:
            started = True
            break
        time.sleep(STARTUP_POLL_INTERVAL_SECONDS)
    if not started:
        log_proc = run_rdctl(["sh", "-lc", f"cat {shlex.quote(remote_log_file)} 2>/dev/null || true"], check=False)
        log_tail = (log_proc.stdout or log_proc.stderr or "").strip()
        raise SystemExit(
            f"tcpdump inside the Rancher VM exited before capture could begin for {path} on {interface}"
            + (f": {log_tail}" if log_tail else "")
        )
    return CaptureSession(
        path=path,
        interface=interface,
        filter_expr=filter_expr,
        process=None,
        backend=CAPTURE_BACKEND_RDCTL_VM_BRIDGE,
        remote_pid_file=remote_pid_file,
        remote_log_file=remote_log_file,
    )


def stop_vm_capture(session: CaptureSession) -> None:
    assert session.remote_pid_file is not None
    assert session.remote_log_file is not None
    stop_cmd = (
        f"if test -f {shlex.quote(session.remote_pid_file)}; then "
        f"sudo kill -INT $(cat {shlex.quote(session.remote_pid_file)}) 2>/dev/null || true; "
        f"for _ in $(seq 1 100); do "
        f"sudo kill -0 $(cat {shlex.quote(session.remote_pid_file)}) 2>/dev/null || exit 0; "
        f"sleep 0.1; "
        f"done; "
        f"sudo kill -KILL $(cat {shlex.quote(session.remote_pid_file)}) 2>/dev/null || true; "
        f"fi"
    )
    run_rdctl(["sh", "-lc", stop_cmd], check=False)
    if not session.path.exists() or session.path.stat().st_size <= 0:
        log_proc = run_rdctl(["sh", "-lc", f"cat {shlex.quote(session.remote_log_file)} 2>/dev/null || true"], check=False)
        log_tail = (log_proc.stdout or log_proc.stderr or "").strip()
        raise SystemExit(
            f"Rancher VM tcpdump capture failed or produced an empty file at {session.path}"
            + (f": {log_tail}" if log_tail else "")
        )


def start_local_capture(path: Path, interface: str, filter_expr: str) -> CaptureSession:
    path.parent.mkdir(parents=True, exist_ok=True)
    cmd = local_capture_command(path, interface, filter_expr)
    proc: subprocess.Popen[bytes] = subprocess.Popen(
        cmd,
        stdout=subprocess.DEVNULL,
        stderr=subprocess.DEVNULL,
    )
    deadline = time.time() + STARTUP_POLL_TIMEOUT_SECONDS
    started = False
    while time.time() < deadline:
        if proc.poll() is not None:
            raise SystemExit(f"tcpdump exited before capture could begin for {path} on {interface}")
        time.sleep(STARTUP_POLL_INTERVAL_SECONDS)
        started = True
        break
    if not started and proc.poll() is not None:
        raise SystemExit(f"tcpdump exited before capture could begin for {path} on {interface}")
    return CaptureSession(
        path=path,
        interface=interface,
        filter_expr=filter_expr,
        process=proc,
        backend=CAPTURE_BACKEND_LOCAL_DOCKER_BRIDGE,
    )


def stop_local_capture(session: CaptureSession) -> None:
    proc = session.process
    if proc is not None and proc.poll() is None:
        proc.send_signal(signal.SIGINT)
        try:
            proc.wait(timeout=10.0)
        except subprocess.TimeoutExpired:
            proc.kill()
            proc.wait(timeout=5.0)
    if not session.path.exists() or session.path.stat().st_size <= 0:
        raise SystemExit(f"tcpdump capture failed or produced an empty file at {session.path}")


@contextmanager
def capture_pcap(
    path: Path,
    *,
    interface: str | None = None,
    filter_expr: str = DEFAULT_CAPTURE_FILTER,
    backend: str | None = None,
) -> Iterator[CaptureSession]:
    resolved_backend = resolve_capture_backend(backend)
    chosen_interface = (interface or env_capture_interface()).strip()
    if not chosen_interface:
        raise SystemExit(
            "capture_pcap requires a validated non-loopback interface. "
            "Set MAVLINK_CAPTURE_INTERFACE or resolve one before starting capture."
        )

    if resolved_backend == CAPTURE_BACKEND_RDCTL_VM_BRIDGE:
        session = start_vm_capture(path, chosen_interface, filter_expr)
        try:
            yield session
        finally:
            stop_vm_capture(session)
        return

    session = start_local_capture(path, chosen_interface, filter_expr)
    try:
        yield session
    finally:
        stop_local_capture(session)
