#!/usr/bin/env python3
"""Generate a real F' dataset using run-local logs, telemetry, and packet capture."""

from __future__ import annotations

import argparse
import csv
import json
import math
import os
import random
import shutil
import subprocess
import sys
import time
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime import FEATURE_NAMES, FEATURE_TIER_FEATURE_NAMES, MODEL_FEATURE_LAYOUTS, NOVELTY_FEATURE_NAMES, PRIMARY_MODEL_FEATURE_NAMES, SCHEMA_VERSION, history_state_reset_mode, packets_to_transactions, save_json, transactions_to_rows
from tools.fprime_real.downlink_ingest import decode_runtime_downlink
from tools.fprime_real.packet_fidelity import build_packets_from_real_artifacts
from tools.fprime_real.pcap_capture import capture_pcap, env_capture_interface, preferred_capture_interface
from tools.fprime_real.runtime_layout import TARGET_NODE_BY_SERVICE, ensure_runtime_tree, host_command_log_path, host_downlink_records_path, host_event_log_path, host_send_log_path, runtime_root_for_output
from tools.fprime_real.schedule_profiles import assert_diverse_episode_signatures, build_benign_rows, build_command_family_overlap_report, build_cyber_rows, build_episode_signature_report, build_fault_rows, write_schedule_csv
from tools.fprime_real.support_probe import DEFAULT_DICTIONARY, assert_actual_run_observability, assert_nominal_support, build_actual_run_observability_report, resolve_identity_capture_target, run_nominal_support_probe, wait_for_capture_drain, wait_for_runtime_ready, warmup_runtime_targets
from tools.shared.artifact_layers import build_canonical_rows_from_raw_transactions
from tools.shared.canonical_records import CANONICAL_COMMAND_ROW_SCHEMA_VERSION
from tools.shared.run_manifest import build_class_overlap_report
from tools.shared.schema import RAW_PACKET_SCHEMA_VERSION, RAW_TRANSACTION_SCHEMA_VERSION, adapt_legacy_fprime_packet, adapt_legacy_fprime_transaction, related_packets_by_transaction, validate_raw_packet_records, validate_raw_transaction_records

EPISODE_ROW_SPAN = 24
DEVELOPMENT_RUN_EPISODE_ROWS_BY_CLASS = {
    "benign": 13,
    "cyber": 6,
    "fault": 4,
}
DEVELOPMENT_RUN_EPISODE_ACTIVATION_ROWS_BY_CLASS = {
    "benign": 48,
    "cyber": 24,
    "fault": 16,
}
LARGE_RUN_EPISODE_ROWS_BY_CLASS = {
    "benign": 18,
    "cyber": 10,
    "fault": 8,
}
LARGE_RUN_EPISODE_ACTIVATION_ROWS_BY_CLASS = {
    "benign": 96,
    "cyber": 64,
    "fault": 32,
}
BENIGN_CLEAN_INTENT_CONTEXT = "benign_clean"
BENIGN_NOISY_INTENT_CONTEXT = "benign_noisy"
MAX_BENIGN_NOISY_FRACTION = 0.25
DEFAULT_INTENT_CONTEXT_BY_CLASS = {
    "benign": BENIGN_CLEAN_INTENT_CONTEXT,
    "cyber": "malicious",
    "fault": "fault",
}
ALLOWED_BENIGN_NUISANCE_REASONS_BY_COMMAND = {
    "cmdSeq.CS_AUTO": {"invalid_mode", "execution_error", "warning_event"},
    "cmdSeq.CS_MANUAL": {"invalid_mode", "execution_error", "warning_event"},
    # Real F' FileSize misses surface as FORMAT_ERROR without a separate FileSizeError event.
    "fileManager.FileSize": {"missing_artifact", "format_error", "execution_error", "warning_event"},
    "fileManager.RemoveDirectory": {"missing_artifact", "execution_error", "warning_event"},
    "fileDownlink.Cancel": {"execution_error", "warning_event"},
}
RUNTIME_RESET_POLICY = "fresh_runtime_per_episode"


@dataclass(frozen=True)
class IndependentRunPlan:
    run_id: int
    run_order: int
    class_name: str
    label: int
    episode_id: int
    class_run_index: int
    schedule_rows: list[dict[str, Any]]


def _median_int(values: list[int]) -> float:
    if not values:
        return 0.0
    ordered = sorted(int(value) for value in values)
    middle = len(ordered) // 2
    if len(ordered) % 2 == 1:
        return float(ordered[middle])
    return float(ordered[middle - 1] + ordered[middle]) / 2.0


def summarize_rows_per_episode(
    execution_order: list[dict[str, Any]],
    *,
    class_row_targets: dict[str, int] | None = None,
    class_episode_targets: dict[str, int] | None = None,
) -> dict[str, dict[str, Any]]:
    rows_by_class: dict[str, list[int]] = {name: [] for name in DEFAULT_INTENT_CONTEXT_BY_CLASS}
    for entry in execution_order:
        class_name = _optional_text(entry.get("class_name"))
        if class_name not in rows_by_class:
            continue
        rows_by_class[class_name].append(int(entry.get("rows", 0) or 0))

    summary: dict[str, dict[str, Any]] = {}
    for class_name, row_counts in rows_by_class.items():
        total_rows = sum(row_counts)
        summary[class_name] = {
            "episodes": len(row_counts),
            "row_target": int((class_row_targets or {}).get(class_name, total_rows)),
            "episode_target": int((class_episode_targets or {}).get(class_name, len(row_counts))),
            "min_rows": min(row_counts) if row_counts else 0,
            "max_rows": max(row_counts) if row_counts else 0,
            "avg_rows": round(total_rows / len(row_counts), 2) if row_counts else 0.0,
            "median_rows": round(_median_int(row_counts), 2) if row_counts else 0.0,
        }
    return summary


def run(cmd: list[str], cwd: Path, *, env: dict[str, str] | None = None) -> None:
    try:
        subprocess.run(cmd, cwd=cwd, check=True, env=env)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Command failed ({' '.join(cmd)}): exit {exc.returncode}") from None


def safe_remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def clear_directory(path: Path) -> None:
    if not path.exists():
        return
    for child in path.iterdir():
        if child.is_dir():
            shutil.rmtree(child)
        else:
            child.unlink()


def shutdown_stack(repo_root: Path) -> None:
    run(
        [
            "bash",
            str(repo_root / "scripts" / "fprime_real" / "down.sh"),
        ],
        repo_root,
    )


def compose_env_for_runtime(runtime_root: Path) -> dict[str, str]:
    env = dict(os.environ)
    env["FPRIME_RUNTIME_HOST_ROOT"] = str(runtime_root)
    return env


def restart_target_nodes(repo_root: Path, compose_file: Path, runtime_root: Path) -> None:
    env = compose_env_for_runtime(runtime_root)
    run(
        [
            "docker",
            "compose",
            "-f",
            str(compose_file),
            "rm",
            "-sf",
            "fprime_a",
            "fprime_b",
        ],
        repo_root,
        env=env,
    )
    runtime_root.mkdir(parents=True, exist_ok=True)
    clear_directory(runtime_root)
    ensure_runtime_tree(runtime_root)
    run(
        [
            "docker",
            "compose",
            "-f",
            str(compose_file),
            "up",
            "-d",
            "fprime_a",
            "fprime_b",
        ],
        repo_root,
        env=env,
    )


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def read_csv_rows(path: Path) -> list[dict[str, str]]:
    with path.open("r", newline="") as handle:
        return list(csv.DictReader(handle))


def write_csv_rows(path: Path, rows: list[dict[str, str]]) -> None:
    if not rows:
        return
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)


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


def resolve_existing_artifact_paths(paths: list[str | Path]) -> list[str]:
    values: list[str] = []
    for item in paths:
        text = _optional_text(item)
        if text is None:
            continue
        path = Path(text).expanduser()
        if not path.exists():
            continue
        resolved = str(path.resolve())
        if resolved not in values:
            values.append(resolved)
    return values


def collect_record_field_paths(records: list[dict[str, Any]]) -> list[str]:
    paths: set[str] = set()

    def visit(value: Any, prefix: str = "") -> None:
        if isinstance(value, dict):
            for key, child in value.items():
                child_prefix = key if not prefix else f"{prefix}.{key}"
                paths.add(child_prefix)
                visit(child, child_prefix)
        elif isinstance(value, list):
            for child in value:
                if isinstance(child, dict):
                    visit(child, prefix)

    for record in records:
        visit(record)
    return sorted(paths)


def normalize_target_stream_id(row: dict[str, str]) -> str:
    explicit = str(row.get("target_stream_id", "")).strip()
    if explicit:
        return explicit
    return f"{row.get('target_service', '')}:{row.get('target_tts_port', row.get('tts_port', '50050'))}"


def normalize_target_stream_indices(rows: list[dict[str, str]]) -> list[dict[str, str]]:
    counters: dict[str, int] = {}
    normalized: list[dict[str, str]] = []
    for row in rows:
        updated = dict(row)
        target_stream_id = normalize_target_stream_id(updated)
        next_index = counters.get(target_stream_id, 0)
        counters[target_stream_id] = next_index + 1
        updated["target_stream_id"] = target_stream_id
        updated["target_stream_index"] = str(next_index)
        normalized.append(updated)
    return normalized


def rewrite_send_log_indices(send_log_path: Path, rows: list[dict[str, str]]) -> None:
    if not send_log_path.exists():
        return
    by_send_id = {
        str(row.get("send_id", "")): row
        for row in rows
        if str(row.get("send_id", ""))
    }
    rewritten: list[dict[str, Any]] = []
    with send_log_path.open("r", encoding="utf-8", errors="replace") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            replacement = by_send_id.get(str(payload.get("send_id", "")))
            if replacement is not None:
                payload["target_stream_id"] = str(replacement.get("target_stream_id", ""))
                payload["target_stream_index"] = int(replacement.get("target_stream_index", "0") or 0)
            rewritten.append(payload)
    with send_log_path.open("w", encoding="utf-8") as handle:
        for payload in rewritten:
            handle.write(json.dumps(payload, separators=(",", ":")) + "\n")


def rewrite_run_logs_with_stream_indices(run_logs: list[Path], rows: list[dict[str, str]]) -> None:
    by_send_id = {
        str(row.get("send_id", "")): row
        for row in rows
        if str(row.get("send_id", ""))
    }
    for run_log_path in run_logs:
        if not run_log_path.exists():
            continue
        run_rows = read_csv_rows(run_log_path)
        rewritten_rows: list[dict[str, str]] = []
        for row in run_rows:
            replacement = by_send_id.get(str(row.get("send_id", "")))
            rewritten_rows.append(dict(replacement) if replacement is not None else row)
        write_csv_rows(run_log_path, rewritten_rows)


def ensure_stack_up(repo_root: Path, compose_file: Path, runtime_root: Path) -> None:
    ensure_runtime_tree(runtime_root)
    run(
        [
            "bash",
            str(repo_root / "scripts" / "fprime_real" / "up.sh"),
            "--runtime-root",
            str(runtime_root),
        ],
        repo_root,
    )


def resolve_manifest_path() -> Path | None:
    override = os.environ.get("FPRIME_BENIGN_MANIFEST", "").strip()
    if not override:
        return None
    return Path(override).resolve()


def real_log_paths(runtime_root: Path) -> tuple[Path, dict[str, Path]]:
    command_log_path = host_command_log_path(runtime_root)
    event_log_paths = {
        target_service: host_event_log_path(runtime_root, target_service)
        for target_service in TARGET_NODE_BY_SERVICE
    }
    return command_log_path, event_log_paths


def group_schedule_rows_by_episode(rows: list[dict[str, Any]]) -> list[list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    order: list[int] = []
    for row in rows:
        meta = dict(row.get("meta", {}) or {})
        episode_id = int(meta.get("episode_id", -1))
        if episode_id not in grouped:
            grouped[episode_id] = []
            order.append(episode_id)
        grouped[episode_id].append(row)
    return [grouped[episode_id] for episode_id in order]


def annotate_schedule_rows_for_run(
    rows: list[dict[str, Any]],
    *,
    run_id: int,
    run_order: int,
    class_run_index: int,
) -> list[dict[str, Any]]:
    annotated: list[dict[str, Any]] = []
    for row in rows:
        meta = dict(row.get("meta", {}) or {})
        meta["run_id"] = run_id
        meta["run_order"] = run_order
        meta["class_run_index"] = class_run_index
        meta["runtime_reset_policy"] = RUNTIME_RESET_POLICY
        annotated.append(
            {
                **row,
                "arguments": list(row.get("arguments", []) or []),
                "meta": meta,
            }
        )
    return annotated


def build_independent_run_plans(
    *,
    nominal_target: int,
    cyber_target: int,
    fault_target: int,
    seed: int,
) -> tuple[list[IndependentRunPlan], dict[str, Any]]:
    class_specs = [
        ("benign", nominal_target, build_benign_rows, 0, seed, {"attack_family": "none"}),
        ("cyber", cyber_target, build_cyber_rows, 1, seed + 101, {}),
        ("fault", fault_target, build_fault_rows, 2, seed + 202, {}),
    ]
    episode_offset = 0
    unassigned_plans: list[dict[str, Any]] = []
    class_episode_counts = {name: 0 for name in DEFAULT_INTENT_CONTEXT_BY_CLASS}
    class_row_targets = {
        "benign": int(nominal_target),
        "cyber": int(cyber_target),
        "fault": int(fault_target),
    }
    class_episode_targets = {name: 0 for name in DEFAULT_INTENT_CONTEXT_BY_CLASS}
    for class_name, target_rows, builder, label, builder_seed, extra_kwargs in class_specs:
        if target_rows <= 0:
            continue
        episode_target = episode_count_target_for_class_rows(class_name, target_rows)
        class_episode_targets[class_name] = episode_target
        episode_span = episode_span_for_target_rows(target_rows, class_name=class_name)
        schedule_rows = builder(
            target_rows=target_rows,
            seed=builder_seed,
            label=label,
            class_name=class_name,
            episode_offset=episode_offset,
            episode_span=episode_span,
            **extra_kwargs,
        )
        episode_groups = group_schedule_rows_by_episode(schedule_rows)
        class_episode_counts[class_name] = len(episode_groups)
        for class_run_index, episode_rows in enumerate(episode_groups):
            first_meta = dict(episode_rows[0].get("meta", {}) or {})
            unassigned_plans.append(
                {
                    "class_name": class_name,
                    "label": label,
                    "episode_id": int(first_meta.get("episode_id", -1)),
                    "class_run_index": class_run_index,
                    "schedule_rows": episode_rows,
                }
            )
        episode_offset += len(episode_groups)

    rng = random.Random(seed + 404)
    rng.shuffle(unassigned_plans)

    plans: list[IndependentRunPlan] = []
    execution_order: list[dict[str, Any]] = []
    for run_order, payload in enumerate(unassigned_plans):
        run_id = run_order
        schedule_rows = annotate_schedule_rows_for_run(
            payload["schedule_rows"],
            run_id=run_id,
            run_order=run_order,
            class_run_index=int(payload["class_run_index"]),
        )
        plan = IndependentRunPlan(
            run_id=run_id,
            run_order=run_order,
            class_name=str(payload["class_name"]),
            label=int(payload["label"]),
            episode_id=int(payload["episode_id"]),
            class_run_index=int(payload["class_run_index"]),
            schedule_rows=schedule_rows,
        )
        plans.append(plan)
        execution_order.append(
            {
                "run_id": run_id,
                "run_order": run_order,
                "class_name": plan.class_name,
                "label": plan.label,
                "episode_id": plan.episode_id,
                "rows": len(plan.schedule_rows),
                "runtime_reset_policy": RUNTIME_RESET_POLICY,
            }
        )

    manifest = {
        "group_key": "run_id",
        "runtime_reset_policy": RUNTIME_RESET_POLICY,
        "run_count": len(plans),
        "class_row_targets": class_row_targets,
        "class_episode_counts": class_episode_counts,
        "class_episode_targets": class_episode_targets,
        "execution_order": execution_order,
        "class_order": [entry["class_name"] for entry in execution_order],
        "class_rows_per_episode_summary": summarize_rows_per_episode(
            execution_order,
            class_row_targets=class_row_targets,
            class_episode_targets=class_episode_targets,
        ),
        "episode_policy": {
            "default_episode_row_span": EPISODE_ROW_SPAN,
            "development_run_episode_rows_by_class": dict(DEVELOPMENT_RUN_EPISODE_ROWS_BY_CLASS),
            "development_run_activation_rows_by_class": dict(DEVELOPMENT_RUN_EPISODE_ACTIVATION_ROWS_BY_CLASS),
            "large_run_episode_rows_by_class": dict(LARGE_RUN_EPISODE_ROWS_BY_CLASS),
            "large_run_activation_rows_by_class": dict(LARGE_RUN_EPISODE_ACTIVATION_ROWS_BY_CLASS),
        },
    }
    return plans, manifest


def append_text_artifact(source: Path, destination: Path) -> None:
    if not source.exists():
        return
    destination.parent.mkdir(parents=True, exist_ok=True)
    with source.open("r", encoding="utf-8", errors="replace") as src_handle:
        content = src_handle.read()
    if not content:
        return
    with destination.open("a", encoding="utf-8") as dst_handle:
        dst_handle.write(content)
        if not content.endswith("\n"):
            dst_handle.write("\n")


def aggregate_runtime_artifacts(aggregate_root: Path, per_run_root: Path, *, run_id: int) -> dict[str, str]:
    ensure_runtime_tree(aggregate_root)
    append_text_artifact(host_command_log_path(per_run_root), host_command_log_path(aggregate_root))
    append_text_artifact(host_send_log_path(per_run_root), host_send_log_path(aggregate_root))
    for target_service in TARGET_NODE_BY_SERVICE:
        append_text_artifact(host_event_log_path(per_run_root, target_service), host_event_log_path(aggregate_root, target_service))
        append_text_artifact(host_downlink_records_path(per_run_root, target_service), host_downlink_records_path(aggregate_root, target_service))
    per_run_pcap = per_run_root / "pcap" / "traffic.pcap"
    aggregate_pcap = aggregate_root / "pcap" / f"run_{run_id:04d}.pcap"
    if per_run_pcap.exists():
        aggregate_pcap.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(per_run_pcap, aggregate_pcap)
    return {
        "command_log": str(host_command_log_path(per_run_root).resolve()),
        "send_log": str(host_send_log_path(per_run_root).resolve()),
        "traffic_pcap": str(aggregate_pcap.resolve()) if aggregate_pcap.exists() else "",
    }


def snapshot_runtime_artifacts(source_runtime_root: Path, destination_runtime_root: Path) -> None:
    ensure_runtime_tree(destination_runtime_root)
    for relative_path in (
        Path("cli_logs"),
        Path("logs"),
        Path("node_a"),
        Path("node_b"),
    ):
        source_path = source_runtime_root / relative_path
        destination_path = destination_runtime_root / relative_path
        if not source_path.exists():
            continue
        if destination_path.exists():
            if destination_path.is_dir():
                shutil.rmtree(destination_path)
            else:
                destination_path.unlink()
        if source_path.is_dir():
            shutil.copytree(source_path, destination_path)
        else:
            destination_path.parent.mkdir(parents=True, exist_ok=True)
            shutil.copy2(source_path, destination_path)


def source_artifact_paths_for_fprime_record(
    record: dict[str, Any],
    *,
    manifest_path: Path,
    run_reports_by_id: dict[int, dict[str, Any]],
) -> list[str]:
    run_id = _optional_int(record.get("run_id"))
    target_service = (
        _optional_text(record.get("target_service"))
        or _optional_text(record.get("node_service"))
        or ""
    )
    paths: list[str | Path] = [manifest_path]
    run_report = run_reports_by_id.get(run_id) if run_id is not None else None
    if run_report is None and len(run_reports_by_id) == 1:
        run_report = next(iter(run_reports_by_id.values()))
    if run_report is not None:
        paths.extend(
            [
                str(run_report.get("schedule_path", "")),
                str(run_report.get("run_log_path", "")),
                str(run_report.get("command_log", "")),
                str(run_report.get("send_log", "")),
                str(run_report.get("traffic_pcap", "")),
            ]
        )
        runtime_root = _optional_text(run_report.get("runtime_root"))
        if runtime_root and target_service in TARGET_NODE_BY_SERVICE:
            runtime_root_path = Path(runtime_root)
            paths.extend(
                [
                    host_event_log_path(runtime_root_path, target_service),
                    host_downlink_records_path(runtime_root_path, target_service),
                ]
            )
    return resolve_existing_artifact_paths(paths)


def build_shared_fprime_artifact_layers(
    packets: list[dict[str, Any]],
    transactions: list[dict[str, Any]],
    feature_rows: list[dict[str, Any]],
    *,
    manifest_path: Path,
    per_run_reports: list[dict[str, Any]],
    capture_backend: str,
    capture_interface: str,
) -> dict[str, list[dict[str, Any]]]:
    run_reports_by_id = {
        int(report["run_id"]): report
        for report in per_run_reports
        if "run_id" in report
    }
    packet_index = related_packets_by_transaction(packets)
    raw_packets = [
        adapt_legacy_fprime_packet(
            packet,
            source_artifact_paths=source_artifact_paths_for_fprime_record(
                packet,
                manifest_path=manifest_path,
                run_reports_by_id=run_reports_by_id,
            ),
            capture_backend=capture_backend,
            capture_interface=capture_interface,
        )
        for packet in packets
    ]
    raw_transactions = [
        adapt_legacy_fprime_transaction(
            transaction,
            related_packets=packet_index.get(
                (
                    _optional_text(transaction.get("session_id")) or "",
                    _optional_text(transaction.get("txn_id")) or "",
                ),
                [],
            ),
            source_artifact_paths=source_artifact_paths_for_fprime_record(
                transaction,
                manifest_path=manifest_path,
                run_reports_by_id=run_reports_by_id,
            ),
            capture_backend=capture_backend,
            capture_interface=capture_interface,
        )
        for transaction in transactions
    ]
    validate_raw_packet_records(raw_packets)
    validate_raw_transaction_records(raw_transactions)
    canonical_rows = build_canonical_rows_from_raw_transactions(
        raw_transactions,
        recent_behavior_rows=feature_rows,
        require_recent_behavior=True,
    )
    return {
        "raw_packets": raw_packets,
        "raw_transactions": raw_transactions,
        "canonical_command_rows": canonical_rows,
    }


def merge_counter_values(target: dict[str, int], source: dict[str, Any]) -> None:
    for key, value in source.items():
        target[str(key)] = int(target.get(str(key), 0)) + int(value)


def merge_channel_inventory(combined: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    nodes = combined.setdefault("nodes", {})
    for node_name, node_payload in incoming.get("nodes", {}).items():
        merged_node = nodes.setdefault(
            node_name,
            {
                "modeled": {},
                "inventory_only": {},
                "unknown": {},
            },
        )
        for bucket_name in ("modeled", "inventory_only", "unknown"):
            merged_bucket = merged_node[bucket_name]
            for channel_name, channel_payload in node_payload.get(bucket_name, {}).items():
                merged_payload = merged_bucket.setdefault(
                    channel_name,
                    {
                        "count": 0,
                        "mapped_field": "",
                        "catalog_feature": "",
                        "catalog_kind": "",
                        "enabled_for_model": False,
                        "examples": [],
                    },
                )
                merged_payload["count"] = int(merged_payload.get("count", 0)) + int(channel_payload.get("count", 0))
                for field_name in ("mapped_field", "catalog_feature", "catalog_kind"):
                    if not merged_payload.get(field_name):
                        merged_payload[field_name] = str(channel_payload.get(field_name, ""))
                merged_payload["enabled_for_model"] = bool(
                    merged_payload.get("enabled_for_model", False) or channel_payload.get("enabled_for_model", False)
                )
                examples = list(merged_payload.get("examples", []))
                for example in channel_payload.get("examples", []):
                    if example not in examples and len(examples) < 3:
                        examples.append(example)
                merged_payload["examples"] = examples
    summary = {"modeled": 0, "inventory_only": 0, "unknown": 0}
    for node_payload in nodes.values():
        for bucket_name in summary:
            summary[bucket_name] += len(node_payload.get(bucket_name, {}))
    combined["summary"] = summary
    return combined


def aggregate_provenance_summaries(
    summaries: list[dict[str, Any]],
    *,
    capture_backend: str,
    capture_interface: str,
) -> dict[str, Any]:
    totals = {
        "schema_version": SCHEMA_VERSION,
        "capture_backend": capture_backend,
        "capture_interface": capture_interface,
        "pcap_identity_mode": "bridge_ip_5tuple",
        "event_attribution_mode": "serialized_per_target",
        "target_stream_serialization_invariant": True,
        "serialization_violations": 0,
        "packet_count": 0,
        "pcap_packet_count": 0,
        "telemetry_packet_count": 0,
        "command_rows": 0,
        "send_log_records": 0,
        "command_log_records": 0,
        "command_log_audit_matches": 0,
        "rows_with_send_id": 0,
        "rows_with_source_ip": 0,
        "rows_with_target_ip": 0,
        "rows_with_target_stream_id": 0,
        "packets_with_send_id": 0,
        "packets_with_target_stream_id": 0,
        "pcap_5tuple_matches": 0,
        "pcap_fallback_rows": 0,
        "accepted_rows": 0,
        "timeout_rows": 0,
        "observations_with_recent_telemetry": 0,
        "packet_kinds": {},
        "ts_sources": {},
        "bytes_sources": {},
        "observed_on_wire": {},
        "request_anchor_sources": {},
        "telemetry_sources": {},
        "independent_runs": len(summaries),
        "runtime_reset_policy": RUNTIME_RESET_POLICY,
    }
    counter_fields = (
        "packet_kinds",
        "ts_sources",
        "bytes_sources",
        "observed_on_wire",
        "request_anchor_sources",
        "telemetry_sources",
    )
    numeric_fields = (
        "serialization_violations",
        "packet_count",
        "pcap_packet_count",
        "telemetry_packet_count",
        "command_rows",
        "send_log_records",
        "command_log_records",
        "command_log_audit_matches",
        "rows_with_send_id",
        "rows_with_source_ip",
        "rows_with_target_ip",
        "rows_with_target_stream_id",
        "packets_with_send_id",
        "packets_with_target_stream_id",
        "pcap_5tuple_matches",
        "pcap_fallback_rows",
        "accepted_rows",
        "timeout_rows",
        "observations_with_recent_telemetry",
    )
    for summary in summaries:
        totals["target_stream_serialization_invariant"] = bool(
            totals["target_stream_serialization_invariant"] and summary.get("target_stream_serialization_invariant", True)
        )
        for field_name in numeric_fields:
            totals[field_name] = int(totals[field_name]) + int(summary.get(field_name, 0) or 0)
        for field_name in counter_fields:
            merge_counter_values(totals[field_name], dict(summary.get(field_name, {}) or {}))
    return totals


def summarize_feature_rows(rows: list[dict[str, Any]], run_rows: list[dict[str, str]]) -> dict[str, Any]:
    label_count = Counter(str(row.get("label_name", row.get("label", ""))) for row in rows)
    command_count = Counter(str(row.get("command", "")) for row in rows)
    service_count = Counter(str(row.get("service", "")) for row in rows)
    scenario_count = Counter(str(row.get("attack_family", "none")) for row in rows)
    episode_kind_count = Counter(str(row.get("episode_kind", "")) for row in rows)
    failure_count = Counter(str(row.get("reason", "")) for row in rows if float(row.get("sat_success", 1.0)) < 0.5 or float(row.get("gds_accept", 1.0)) < 0.5)
    reason_count = Counter(str(run_row.get("reason", "")) for run_row in run_rows)
    status_count = Counter(
        "success" if str(run_row.get("gds_accept", "")) == "1" and str(run_row.get("sat_success", "")) == "1" and str(run_row.get("timeout", "")) == "0" else "anomalous"
        for run_row in run_rows
    )
    return {
        "rows": len(rows),
        "labels": dict(label_count),
        "episode_kinds": dict(episode_kind_count),
        "services": dict(service_count),
        "commands": dict(command_count),
        "scenario_families": dict(scenario_count),
        "failures": dict(failure_count),
        "run_reasons": dict(reason_count),
        "run_status": dict(status_count),
        "behavior_summary": summarize_behavior_rows(run_rows),
    }


def build_canonical_actor_context_overlap_report(canonical_rows: list[dict[str, Any]]) -> dict[str, Any]:
    return {
        "role": build_class_overlap_report(
            canonical_rows,
            value_path="actor_context.role",
            family_key="canonical_actor_role",
            item_key="role",
            items_key="roles",
        ),
        "trust_class": build_class_overlap_report(
            canonical_rows,
            value_path="actor_context.trust_class",
            family_key="canonical_actor_trust_class",
            item_key="trust_class",
            items_key="trust_classes",
        ),
    }


def parse_run_row_meta(row: dict[str, str]) -> dict[str, Any]:
    try:
        payload = json.loads(row.get("meta_json", "{}"))
    except (TypeError, ValueError, json.JSONDecodeError):
        payload = {}
    return payload if isinstance(payload, dict) else {}


def run_row_class_name(row: dict[str, str], meta: dict[str, Any] | None = None) -> str:
    payload = parse_run_row_meta(row) if meta is None else meta
    explicit = str(payload.get("class_name", "")).strip()
    if explicit:
        return explicit
    label = payload.get("class_label")
    if label in (0, "0"):
        return "benign"
    if label in (1, "1"):
        return "cyber"
    if label in (2, "2"):
        return "fault"
    return ""


def run_row_intent_context(row: dict[str, str], meta: dict[str, Any] | None = None) -> str:
    payload = parse_run_row_meta(row) if meta is None else meta
    explicit = str(payload.get("intent_context", "")).strip()
    if explicit:
        return explicit
    return DEFAULT_INTENT_CONTEXT_BY_CLASS.get(run_row_class_name(row, payload), "unknown")


def run_row_status(row: dict[str, str]) -> str:
    return (
        "success"
        if str(row.get("gds_accept", "")) == "1" and str(row.get("sat_success", "")) == "1" and str(row.get("timeout", "")) == "0"
        else "anomalous"
    )


def is_clean_benign_success(row: dict[str, str]) -> bool:
    return run_row_status(row) == "success" and str(row.get("reason", "")) == "completed"


def nominal_row_preview(row: dict[str, str], *, meta: dict[str, Any] | None = None) -> dict[str, str]:
    payload = parse_run_row_meta(row) if meta is None else meta
    return {
        "source_service": row.get("source_service", ""),
        "target_service": row.get("target_service", ""),
        "command": row.get("command", ""),
        "reason": row.get("reason", ""),
        "gds_accept": row.get("gds_accept", ""),
        "sat_success": row.get("sat_success", ""),
        "timeout": row.get("timeout", ""),
        "intent_context": run_row_intent_context(row, payload),
        "attack_family": str(payload.get("attack_family", "")),
        "event_names_json": row.get("event_names_json", ""),
        "send_exception": row.get("send_exception", ""),
    }


def allowed_benign_nuisance_reason(row: dict[str, str], *, meta: dict[str, Any] | None = None) -> str | None:
    payload = parse_run_row_meta(row) if meta is None else meta
    if run_row_intent_context(row, payload) != BENIGN_NOISY_INTENT_CONTEXT:
        return None
    if str(row.get("gds_accept", "0")) != "1":
        return None
    if str(row.get("timeout", "0")) == "1":
        return None
    reason = str(row.get("reason", "")).strip()
    command = str(row.get("command", "")).strip()
    allowed_reasons = ALLOWED_BENIGN_NUISANCE_REASONS_BY_COMMAND.get(command, set())
    return reason if reason in allowed_reasons else None


def summarize_behavior_rows(run_rows: list[dict[str, str]]) -> dict[str, Any]:
    class_rows = Counter()
    intent_rows = Counter()
    status_by_intent: dict[str, Counter[str]] = {}
    class_intent_rows: dict[str, Counter[str]] = {}
    reason_by_intent: dict[str, Counter[str]] = {}
    for row in run_rows:
        meta = parse_run_row_meta(row)
        class_name = run_row_class_name(row, meta)
        if not class_name:
            continue
        intent_context = run_row_intent_context(row, meta)
        status = run_row_status(row)
        reason = str(row.get("reason", "")).strip()
        class_rows[class_name] += 1
        intent_rows[intent_context] += 1
        status_by_intent.setdefault(intent_context, Counter())[status] += 1
        class_intent_rows.setdefault(class_name, Counter())[intent_context] += 1
        if reason:
            reason_by_intent.setdefault(intent_context, Counter())[reason] += 1
    return {
        "class_rows": dict(class_rows),
        "intent_context_rows": dict(intent_rows),
        "status_by_intent_context": {
            intent_context: dict(counter)
            for intent_context, counter in status_by_intent.items()
        },
        "class_intent_rows": {
            class_name: dict(counter)
            for class_name, counter in class_intent_rows.items()
        },
        "reason_by_intent_context": {
            intent_context: dict(counter)
            for intent_context, counter in reason_by_intent.items()
        },
    }


def summarize_nominal_policy(run_rows: list[dict[str, str]]) -> dict[str, Any]:
    intent_rows = Counter()
    status_by_intent: dict[str, Counter[str]] = {}
    allowed_reason_count = Counter()
    unexpected_reason_count = Counter()
    allowed_examples: list[dict[str, str]] = []
    unexpected_examples: list[dict[str, str]] = []
    clean_success_rows = 0
    benign_noisy_rows = 0
    unexpected_rows = 0

    for row in run_rows:
        meta = parse_run_row_meta(row)
        if run_row_class_name(row, meta) != "benign":
            continue
        intent_context = run_row_intent_context(row, meta)
        intent_rows[intent_context] += 1
        status_by_intent.setdefault(intent_context, Counter())[run_row_status(row)] += 1
        if is_clean_benign_success(row):
            clean_success_rows += 1
            continue
        allowed_reason = allowed_benign_nuisance_reason(row, meta=meta)
        if allowed_reason is not None:
            benign_noisy_rows += 1
            allowed_reason_count[allowed_reason] += 1
            if len(allowed_examples) < 5:
                allowed_examples.append(nominal_row_preview(row, meta=meta))
            continue
        unexpected_rows += 1
        unexpected_reason_count[str(row.get("reason", ""))] += 1
        if len(unexpected_examples) < 5:
            unexpected_examples.append(nominal_row_preview(row, meta=meta))

    benign_rows = clean_success_rows + benign_noisy_rows + unexpected_rows
    max_noisy_rows = 0 if benign_rows <= 0 else max(1, math.ceil(benign_rows * MAX_BENIGN_NOISY_FRACTION))
    return {
        "policy": {
            "clean_intent_context": BENIGN_CLEAN_INTENT_CONTEXT,
            "noisy_intent_context": BENIGN_NOISY_INTENT_CONTEXT,
            "max_benign_noisy_fraction": MAX_BENIGN_NOISY_FRACTION,
            "max_benign_noisy_rows": max_noisy_rows,
            "allowed_benign_nuisance_reasons_by_command": {
                command: sorted(reasons)
                for command, reasons in ALLOWED_BENIGN_NUISANCE_REASONS_BY_COMMAND.items()
            },
        },
        "summary": {
            "benign_rows": benign_rows,
            "clean_success_rows": clean_success_rows,
            "benign_noisy_rows": benign_noisy_rows,
            "unexpected_rows": unexpected_rows,
            "benign_noisy_fraction": 0.0 if benign_rows <= 0 else round(benign_noisy_rows / benign_rows, 4),
            "intent_rows": dict(intent_rows),
            "status_by_intent_context": {
                intent_context: dict(counter)
                for intent_context, counter in status_by_intent.items()
            },
            "allowed_reasons": dict(allowed_reason_count),
            "unexpected_reasons": dict(unexpected_reason_count),
        },
        "allowed_examples": allowed_examples,
        "unexpected_examples": unexpected_examples,
    }


def validate_nominal_runs(run_rows: list[dict[str, str]]) -> dict[str, Any]:
    report = summarize_nominal_policy(run_rows)
    summary = report["summary"]
    policy = report["policy"]
    if int(summary["unexpected_rows"]) <= 0 and int(summary["benign_noisy_rows"]) <= int(policy["max_benign_noisy_rows"]):
        return report
    raise SystemExit(
        "Nominal schedule exceeded the bounded benign-noise policy. "
        f"Summary: {json.dumps(summary, separators=(',', ':'))}. "
        f"Unexpected examples: {json.dumps(report['unexpected_examples'][:5], separators=(',', ':'))}"
    )


def episode_count_target_for_class_rows(class_name: str, class_rows: int) -> int:
    if class_rows <= 0:
        return 0
    base_target = max(3, math.ceil(class_rows / EPISODE_ROW_SPAN))
    development_activation_rows = int(
        DEVELOPMENT_RUN_EPISODE_ACTIVATION_ROWS_BY_CLASS.get(class_name, EPISODE_ROW_SPAN)
    )
    activation_rows = int(LARGE_RUN_EPISODE_ACTIVATION_ROWS_BY_CLASS.get(class_name, EPISODE_ROW_SPAN))
    if class_rows >= activation_rows:
        target_rows_per_episode = int(LARGE_RUN_EPISODE_ROWS_BY_CLASS.get(class_name, EPISODE_ROW_SPAN))
    elif class_rows >= development_activation_rows:
        target_rows_per_episode = int(DEVELOPMENT_RUN_EPISODE_ROWS_BY_CLASS.get(class_name, EPISODE_ROW_SPAN))
    else:
        return base_target
    run_target = max(1, math.ceil(class_rows / target_rows_per_episode))
    return max(base_target, run_target)


def episode_span_for_target_rows(class_rows: int, *, class_name: str) -> int:
    if class_rows <= 0:
        return EPISODE_ROW_SPAN
    base_target = max(3, math.ceil(class_rows / EPISODE_ROW_SPAN))
    episode_count_target = episode_count_target_for_class_rows(class_name, class_rows)
    if episode_count_target <= base_target:
        return max(1, math.ceil(class_rows / base_target))
    return max(1, math.ceil(class_rows / episode_count_target))


def write_schema_report(
    report_dir: Path,
    packets: list[dict[str, Any]],
    transactions: list[dict[str, Any]],
    history_featurization: dict[str, Any],
    *,
    raw_packets: list[dict[str, Any]] | None = None,
    raw_transactions: list[dict[str, Any]] | None = None,
    canonical_rows: list[dict[str, Any]] | None = None,
    artifact_paths: dict[str, str] | None = None,
) -> Path:
    packet_fields = sorted({key for packet in packets for key in packet.keys()})
    transaction_fields = sorted({key for tx in transactions for key in tx.keys()})
    raw_packet_records = list(raw_packets or [])
    raw_transaction_records = list(raw_transactions or [])
    canonical_command_rows = list(canonical_rows or [])
    payload = {
        "schema_version": SCHEMA_VERSION,
        "raw_packet_schema_version": RAW_PACKET_SCHEMA_VERSION,
        "raw_transaction_schema_version": RAW_TRANSACTION_SCHEMA_VERSION,
        "canonical_command_row_schema_version": CANONICAL_COMMAND_ROW_SCHEMA_VERSION,
        "packet_fields": packet_fields,
        "transaction_fields": transaction_fields,
        "legacy_packet_fields": packet_fields,
        "legacy_transaction_fields": transaction_fields,
        "raw_packet_fields": collect_record_field_paths(raw_packet_records),
        "raw_transaction_fields": collect_record_field_paths(raw_transaction_records),
        "canonical_command_row_fields": collect_record_field_paths(canonical_command_rows),
        "artifact_paths": dict(artifact_paths or {}),
        "artifact_row_counts": {
            "legacy_packets": len(packets),
            "legacy_transactions": len(transactions),
            "raw_packets": len(raw_packet_records),
            "raw_transactions": len(raw_transaction_records),
            "canonical_command_rows": len(canonical_command_rows),
        },
        "dataset_feature_names": list(FEATURE_NAMES),
        "history_featurization": dict(history_featurization),
        "feature_tiers": {name: list(feature_names) for name, feature_names in FEATURE_TIER_FEATURE_NAMES.items()},
        "model_feature_layouts": {
            model_name: {
                "feature_tier": str(layout["feature_tier"]),
                "feature_names": list(layout["feature_names"]),
            }
            for model_name, layout in MODEL_FEATURE_LAYOUTS.items()
        },
        "primary_model_feature_names": list(PRIMARY_MODEL_FEATURE_NAMES),
        "novelty_feature_names": list(NOVELTY_FEATURE_NAMES),
    }
    path = report_dir / "schema.json"
    save_json(path, payload)
    return path


def resolve_active_capture_interface(capture_backend: str, fallback_interface: str) -> str:
    explicit_interface = env_capture_interface()
    if explicit_interface:
        return explicit_interface
    refreshed_interface = preferred_capture_interface(capture_backend)
    if refreshed_interface:
        return refreshed_interface
    return fallback_interface


def execute_run_plan(
    *,
    plan: IndependentRunPlan,
    repo_root: Path,
    compose_file: Path,
    aggregate_runtime_root: Path,
    live_runtime_root: Path | None = None,
    timeout_seconds: float,
    time_scale: float,
    capture_backend: str,
    capture_interface: str,
) -> dict[str, Any]:
    run_name = f"run_{plan.run_id:04d}_{plan.class_name}_ep_{plan.episode_id:04d}"
    per_run_root = aggregate_runtime_root / "runs" / run_name
    schedule_path = aggregate_runtime_root / "schedule" / f"{run_name}.csv"
    run_log_path = aggregate_runtime_root / "logs" / f"{run_name}.csv"
    pcap_path = per_run_root / "pcap" / "traffic.pcap"
    safe_remove_tree(per_run_root)
    write_schedule_csv(schedule_path, plan.schedule_rows)
    active_runtime_root = live_runtime_root or per_run_root

    try:
        if live_runtime_root is None:
            shutdown_stack(repo_root)
            ensure_stack_up(repo_root, compose_file, per_run_root)
        else:
            restart_target_nodes(repo_root, compose_file, live_runtime_root)
        wait_for_runtime_ready(active_runtime_root)
        run_capture_interface = resolve_active_capture_interface(capture_backend, capture_interface)
        warmup_runtime_targets(
            compose_file=compose_file,
            dictionary_path=DEFAULT_DICTIONARY,
            timeout_seconds=timeout_seconds,
            sender_script="/workspace/tools/fprime_real/send_fprime_events.py",
        )
        command_log_path, event_log_paths = real_log_paths(active_runtime_root)
        send_log_path = host_send_log_path(active_runtime_root)
        with capture_pcap(pcap_path, interface=run_capture_interface, backend=capture_backend) as capture:
            run(
                [
                    "python3",
                    str(repo_root / "tools" / "fprime_real" / "run_fprime_schedule.py"),
                    "--compose-file",
                    str(compose_file),
                    "--schedule",
                    str(schedule_path),
                    "--time-scale",
                    str(time_scale),
                    "--timeout-seconds",
                    str(timeout_seconds),
                    "--logs-dir",
                    "/runtime_root/cli_logs",
                    "--output",
                    str(run_log_path),
                ],
                repo_root,
            )
            run_rows = read_csv_rows(run_log_path)
            wait_for_capture_drain(
                run_rows,
                event_log_paths=event_log_paths,
                send_log_path=send_log_path,
                pcap_path=pcap_path,
                capture_interface=capture.interface,
                timeout_seconds=timeout_seconds,
            )
        run_rows = normalize_target_stream_indices(run_rows)
        rewrite_run_logs_with_stream_indices([run_log_path], run_rows)
        rewrite_send_log_indices(send_log_path, run_rows)
        decoded_paths = decode_runtime_downlink(
            repo_root,
            compose_file,
            active_runtime_root,
            dictionary_path=DEFAULT_DICTIONARY,
        )
        if live_runtime_root is not None:
            snapshot_runtime_artifacts(live_runtime_root, per_run_root)
            command_log_path, event_log_paths = real_log_paths(per_run_root)
            send_log_path = host_send_log_path(per_run_root)
            decoded_paths = {
                target_service: host_downlink_records_path(per_run_root, target_service)
                for target_service in TARGET_NODE_BY_SERVICE
            }
        packet_result = build_packets_from_real_artifacts(
            run_rows,
            command_log_path=command_log_path,
            event_log_paths=event_log_paths,
            telemetry_record_paths=decoded_paths,
            send_log_path=send_log_path,
            pcap_path=pcap_path,
            capture_interface=capture.interface,
            capture_backend=capture_backend,
            strict=True,
        )
        aggregate_paths = aggregate_runtime_artifacts(aggregate_runtime_root, per_run_root, run_id=plan.run_id)
        return {
            "run_name": run_name,
            "run_rows": run_rows,
            "run_log_path": run_log_path,
            "schedule_path": schedule_path,
            "runtime_root": per_run_root,
            "pcap_path": pcap_path,
            "aggregate_paths": aggregate_paths,
            "packet_result": packet_result,
        }
    finally:
        if live_runtime_root is None:
            shutdown_stack(repo_root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--nominal-ratio", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--time-scale", type=float, default=7200.0)
    parser.add_argument("--timeout-seconds", type=float, default=15.0)
    args = parser.parse_args()

    if not 0.0 < args.nominal_ratio < 1.0:
        raise SystemExit("--nominal-ratio must be between 0 and 1")

    repo_root = REPO_ROOT
    compose_file = repo_root / "orchestration" / "docker-compose.fprime-real.yml"
    output_dir = Path(args.output_dir).resolve()
    data_dir = output_dir / "data"
    report_dir = output_dir / "reports"
    runtime_root = runtime_root_for_output(output_dir)
    schedule_dir = runtime_root / "schedule"
    log_dir = runtime_root / "logs"
    preflight_root = output_dir / "fprime_real_preflight"
    manifest_path = resolve_manifest_path()

    nominal_target = int(round(args.rows * args.nominal_ratio))
    anomaly_target = int(args.rows - nominal_target)
    cyber_target = int(round(anomaly_target * 0.65))
    fault_target = int(anomaly_target - cyber_target)

    safe_remove_tree(runtime_root)
    safe_remove_tree(preflight_root)
    report_dir.mkdir(parents=True, exist_ok=True)
    ensure_runtime_tree(runtime_root)

    capture_backend = ""
    capture_interface = ""
    try:
        shutdown_stack(repo_root)
        ensure_stack_up(repo_root, compose_file, preflight_root)
        wait_for_runtime_ready(preflight_root)
        capture_target = resolve_identity_capture_target(
            repo_root,
            compose_file,
            timeout_seconds=args.timeout_seconds,
            runtime_root=preflight_root,
        )
        capture_backend = capture_target.backend
        capture_interface = capture_target.interface
        support_matrix = run_nominal_support_probe(
            repo_root,
            compose_file,
            timeout_seconds=args.timeout_seconds,
            runtime_root=preflight_root,
            capture_backend=capture_backend,
            capture_interface=capture_interface,
            manifest_path=manifest_path,
        )
        save_json(report_dir / "support_matrix.json", support_matrix)
        assert_nominal_support(support_matrix)
    finally:
        shutdown_stack(repo_root)

    run_plans, run_manifest = build_independent_run_plans(
        nominal_target=nominal_target,
        cyber_target=cyber_target,
        fault_target=fault_target,
        seed=args.seed,
    )
    save_json(report_dir / "run_manifest.json", run_manifest)

    run_logs: list[Path] = []
    all_run_rows: list[dict[str, str]] = []
    all_packets: list[dict[str, Any]] = []
    all_observations: list[dict[str, Any]] = []
    benign_run_rows: list[dict[str, str]] = []
    per_run_reports: list[dict[str, Any]] = []
    channel_inventory: dict[str, Any] = {"schema_version": SCHEMA_VERSION, "nodes": {}}
    provenance_inputs: list[dict[str, Any]] = []
    live_runtime_root = runtime_root / "_live_runtime"

    try:
        safe_remove_tree(live_runtime_root)
        ensure_stack_up(repo_root, compose_file, live_runtime_root)
        for plan in run_plans:
            result = execute_run_plan(
                plan=plan,
                repo_root=repo_root,
                compose_file=compose_file,
                aggregate_runtime_root=runtime_root,
                live_runtime_root=live_runtime_root,
                timeout_seconds=args.timeout_seconds,
                time_scale=args.time_scale,
                capture_backend=capture_backend,
                capture_interface=capture_interface,
            )
            observations_offset = len(all_run_rows)
            adjusted_observations: list[dict[str, Any]] = []
            for observation in result["packet_result"].observations:
                updated = dict(observation)
                updated["row_index"] = int(updated.get("row_index", 0)) + observations_offset
                adjusted_observations.append(updated)
            all_observations.extend(adjusted_observations)
            all_run_rows.extend(result["run_rows"])
            all_packets.extend(result["packet_result"].packets)
            run_logs.append(Path(result["run_log_path"]))
            per_run_reports.append(
                {
                    "run_id": plan.run_id,
                    "run_order": plan.run_order,
                    "class_name": plan.class_name,
                    "label": plan.label,
                    "episode_id": plan.episode_id,
                    "rows": len(result["run_rows"]),
                    "schedule_path": str(Path(result["schedule_path"]).resolve()),
                    "run_log_path": str(Path(result["run_log_path"]).resolve()),
                    "runtime_root": str(Path(result["runtime_root"]).resolve()),
                    "traffic_pcap": str(Path(result["pcap_path"]).resolve()),
                    **result["aggregate_paths"],
                }
            )
            provenance_inputs.append(dict(result["packet_result"].provenance_summary))
            channel_inventory = merge_channel_inventory(channel_inventory, result["packet_result"].channel_inventory)
            if plan.class_name == "benign":
                benign_run_rows.extend(result["run_rows"])
    finally:
        shutdown_stack(repo_root)

    benign_nominal_policy_report = validate_nominal_runs(benign_run_rows)
    transactions = packets_to_transactions(all_packets, reset_key="run_id")
    history_reset_key = "run_id"
    history_featurization = {
        "source": "transactions",
        "group_key": history_reset_key,
        "state_reset": history_state_reset_mode(history_reset_key),
    }
    feature_rows = transactions_to_rows(transactions, reset_key=history_reset_key)
    shared_artifacts = build_shared_fprime_artifact_layers(
        all_packets,
        transactions,
        feature_rows,
        manifest_path=manifest_path,
        per_run_reports=per_run_reports,
        capture_backend=capture_backend,
        capture_interface=capture_interface,
    )

    artifact_paths = {
        "legacy_packets": str((data_dir / "packets.jsonl").resolve()),
        "legacy_transactions": str((data_dir / "transactions.jsonl").resolve()),
        "legacy_dataset": str((data_dir / "dataset.jsonl").resolve()),
        "raw_packets": str((data_dir / "raw_packets.jsonl").resolve()),
        "raw_transactions": str((data_dir / "raw_transactions.jsonl").resolve()),
        "canonical_command_rows": str((data_dir / "canonical_command_rows.jsonl").resolve()),
    }

    write_jsonl(data_dir / "packets.jsonl", all_packets)
    write_jsonl(data_dir / "transactions.jsonl", transactions)
    write_jsonl(data_dir / "dataset.jsonl", feature_rows)
    write_jsonl(data_dir / "raw_packets.jsonl", shared_artifacts["raw_packets"])
    write_jsonl(data_dir / "raw_transactions.jsonl", shared_artifacts["raw_transactions"])
    write_jsonl(data_dir / "canonical_command_rows.jsonl", shared_artifacts["canonical_command_rows"])
    write_csv_rows(log_dir / "all_runs.csv", all_run_rows)

    episode_signature_report = build_episode_signature_report(all_run_rows)
    assert_diverse_episode_signatures(episode_signature_report)
    actual_run_report = build_actual_run_observability_report(
        all_run_rows,
        all_observations,
        manifest_path=manifest_path,
    )
    save_json(report_dir / "actual_run_observability.json", actual_run_report)
    assert_actual_run_observability(actual_run_report)

    combined_provenance = aggregate_provenance_summaries(
        provenance_inputs,
        capture_backend=capture_backend,
        capture_interface=capture_interface,
    )
    save_json(report_dir / "channel_inventory.json", channel_inventory)
    save_json(report_dir / "provenance_summary.json", combined_provenance)
    command_overlap_path = report_dir / "command_family_overlap.json"
    save_json(command_overlap_path, build_command_family_overlap_report(feature_rows))
    canonical_actor_context_overlap = build_canonical_actor_context_overlap_report(
        shared_artifacts["canonical_command_rows"]
    )
    canonical_actor_context_overlap_path = report_dir / "canonical_actor_context_overlap.json"
    save_json(canonical_actor_context_overlap_path, canonical_actor_context_overlap)
    episode_signature_path = report_dir / "episode_signatures.json"
    save_json(episode_signature_path, episode_signature_report)
    behavior_summary = summarize_behavior_rows(all_run_rows)
    behavior_summary_path = report_dir / "behavior_summary.json"
    save_json(behavior_summary_path, behavior_summary)
    benign_nominal_policy_path = report_dir / "benign_nominal_policy.json"
    save_json(benign_nominal_policy_path, benign_nominal_policy_report)
    schema_path = write_schema_report(
        report_dir,
        all_packets,
        transactions,
        history_featurization,
        raw_packets=shared_artifacts["raw_packets"],
        raw_transactions=shared_artifacts["raw_transactions"],
        canonical_rows=shared_artifacts["canonical_command_rows"],
        artifact_paths=artifact_paths,
    )

    summary = summarize_feature_rows(feature_rows, all_run_rows)
    summary.update(
        {
            "schema_version": SCHEMA_VERSION,
            "packet_count": len(all_packets),
            "transaction_count": len(transactions),
            "raw_packet_count": len(shared_artifacts["raw_packets"]),
            "raw_transaction_count": len(shared_artifacts["raw_transactions"]),
            "canonical_command_row_count": len(shared_artifacts["canonical_command_rows"]),
            "raw_packet_schema_version": RAW_PACKET_SCHEMA_VERSION,
            "raw_transaction_schema_version": RAW_TRANSACTION_SCHEMA_VERSION,
            "canonical_command_row_schema_version": CANONICAL_COMMAND_ROW_SCHEMA_VERSION,
            "feature_names": list(FEATURE_NAMES),
            "history_featurization": dict(history_featurization),
            "data_artifacts": artifact_paths,
            "feature_tiers": {name: list(feature_names) for name, feature_names in FEATURE_TIER_FEATURE_NAMES.items()},
            "model_feature_layouts": {
                model_name: {
                    "feature_tier": str(layout["feature_tier"]),
                    "feature_names": list(layout["feature_names"]),
                }
                for model_name, layout in MODEL_FEATURE_LAYOUTS.items()
            },
            "primary_model_feature_names": list(PRIMARY_MODEL_FEATURE_NAMES),
            "novelty_feature_names": list(NOVELTY_FEATURE_NAMES),
            "run_logs": [str(path) for path in run_logs],
            "combined_run_log": str((log_dir / "all_runs.csv").resolve()),
            "schedule_dir": str(schedule_dir.resolve()),
            "runtime_root": str(runtime_root.resolve()),
            "run_manifest_report": str((report_dir / "run_manifest.json").resolve()),
            "independent_runs": run_manifest,
            "class_rows_per_episode_summary": dict(run_manifest.get("class_rows_per_episode_summary", {})),
            "per_run_reports": per_run_reports,
            "support_matrix": str((report_dir / "support_matrix.json").resolve()),
            "actual_run_observability_report": str((report_dir / "actual_run_observability.json").resolve()),
            "behavior_summary": behavior_summary,
            "behavior_summary_report": str(behavior_summary_path.resolve()),
            "episode_signature_summary": episode_signature_report["summary"],
            "episode_signature_report": str(episode_signature_path.resolve()),
            "canonical_actor_context_overlap": {
                key: dict(value.get("summary", {}))
                for key, value in canonical_actor_context_overlap.items()
            },
            "canonical_actor_context_overlap_report": str(canonical_actor_context_overlap_path.resolve()),
            "benign_nominal_policy": {
                "policy": benign_nominal_policy_report["policy"],
                "summary": benign_nominal_policy_report["summary"],
            },
            "benign_nominal_policy_report": str(benign_nominal_policy_path.resolve()),
            "traffic_pcap": "" if len(per_run_reports) != 1 else str(Path(per_run_reports[0]["traffic_pcap"]).resolve()),
            "traffic_pcaps": [str(item["traffic_pcap"]) for item in per_run_reports if str(item.get("traffic_pcap", "")).strip()],
            "capture_backend": capture_backend,
            "capture_interface": capture_interface,
            "send_log": str(host_send_log_path(runtime_root).resolve()),
            "schema_report": str(schema_path.resolve()),
            "command_family_overlap_report": str(command_overlap_path.resolve()),
            "channel_inventory_report": str((report_dir / "channel_inventory.json").resolve()),
            "provenance_summary_report": str((report_dir / "provenance_summary.json").resolve()),
        }
    )
    save_json(report_dir / "generation_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
