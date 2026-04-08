#!/usr/bin/env python3
"""Host-side helpers for decoding GDS raw downlink artifacts inside the F' container image."""

from __future__ import annotations

import subprocess
from pathlib import Path

from tools.fprime_real.run_fprime_schedule import to_container_path
from tools.fprime_real.runtime_layout import CONTAINER_RUNTIME_ROOT, TARGET_NODE_BY_SERVICE, host_downlink_records_path


def decode_runtime_downlink(
    repo_root: Path,
    compose_file: Path,
    runtime_root: Path,
    *,
    dictionary_path: str,
) -> dict[str, Path]:
    script_path = "/workspace/tools/fprime_real/decode_gds_downlink.py"
    results: dict[str, Path] = {}
    for target_service, node_name in TARGET_NODE_BY_SERVICE.items():
        recv_bin_path = str(CONTAINER_RUNTIME_ROOT / node_name / "logs" / "recv.bin")
        output_path = str(CONTAINER_RUNTIME_ROOT / node_name / "logs" / "downlink_records.jsonl")
        cmd = [
            "docker",
            "compose",
            "-f",
            str(compose_file),
            "exec",
            "-T",
            target_service,
            "python3",
            script_path,
            "--dictionary",
            dictionary_path if dictionary_path.startswith("/workspace") else to_container_path(Path(dictionary_path), repo_root),
            "--recv-bin",
            recv_bin_path,
            "--node-service",
            target_service,
            "--output",
            output_path,
        ]
        try:
            subprocess.run(cmd, cwd=repo_root, check=True)
        except subprocess.CalledProcessError as exc:
            raise SystemExit(
                f"Failed to decode raw GDS downlink for {target_service}: exit {exc.returncode}"
            ) from None
        results[target_service] = host_downlink_records_path(runtime_root, target_service)
    return results
