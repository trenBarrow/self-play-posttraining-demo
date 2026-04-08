#!/usr/bin/env python3
"""Generate a real MAVLink dataset using the headless SITL stack plus packet provenance reconstruction."""

from __future__ import annotations

import argparse
import json
import random
import shutil
import subprocess
import sys
from collections import Counter
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime import save_json
from tools.mavlink_real.log_ingest import load_runtime_run_rows
from tools.mavlink_real.packet_fidelity import build_packets_from_real_artifacts, write_jsonl
from tools.mavlink_real.pcap_capture import capture_pcap
from tools.mavlink_real.runtime_layout import (
    ensure_runtime_tree,
    host_capture_pcap_path,
    host_schedule_run_logs_dir,
    host_schedules_dir,
    runtime_root_for_output,
)
from tools.mavlink_real.schedule_profiles import (
    build_benign_rows,
    build_cyber_rows,
    build_fault_rows,
    write_schedule_csv,
)
from tools.mavlink_real.support_probe import (
    assert_actual_run_observability,
    build_actual_run_observability_report,
    resolve_identity_capture_target,
    wait_for_capture_drain,
)
from tools.shared.artifact_layers import build_canonical_rows_from_raw_transactions
from tools.shared.canonical_records import CANONICAL_COMMAND_ROW_SCHEMA_VERSION
from tools.shared.recent_behavior import build_recent_behavior_rows_from_raw_transactions
from tools.shared.run_manifest import class_row_targets
from tools.train.poster_default import canonical_rows_to_training_rows, poster_default_model_feature_names

RUNTIME_RESET_POLICY = "fresh_runtime_per_episode"
EPISODE_ROW_SPAN = 18


@dataclass(frozen=True)
class IndependentRunPlan:
    run_id: int
    run_order: int
    class_name: str
    label: int
    episode_id: int
    class_run_index: int
    schedule_rows: list[dict[str, Any]]


def run(cmd: list[str], cwd: Path) -> None:
    try:
        subprocess.run(cmd, cwd=cwd, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Command failed ({' '.join(cmd)}): exit {exc.returncode}") from None


def safe_remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def shutdown_stack(repo_root: Path) -> None:
    run(
        [
            "bash",
            str(repo_root / "scripts" / "mavlink_real" / "down.sh"),
        ],
        repo_root,
    )


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
                "arguments": dict(row.get("arguments", {}) or {}),
                "meta": meta,
            }
        )
    return annotated


def episode_span_for_target_rows(class_rows: int) -> int:
    if class_rows <= 0:
        return EPISODE_ROW_SPAN
    episode_count_target = max(3, round(class_rows / EPISODE_ROW_SPAN))
    return max(1, round(class_rows / max(1, episode_count_target)))


def build_independent_run_plans(
    *,
    nominal_target: int,
    cyber_target: int,
    fault_target: int,
    seed: int,
) -> tuple[list[IndependentRunPlan], dict[str, Any]]:
    class_specs = [
        ("benign", nominal_target, build_benign_rows, 0, seed),
        ("cyber", cyber_target, build_cyber_rows, 1, seed + 101),
        ("fault", fault_target, build_fault_rows, 2, seed + 202),
    ]
    episode_offset = 0
    unassigned_plans: list[dict[str, Any]] = []
    class_episode_counts = {"benign": 0, "cyber": 0, "fault": 0}
    for class_name, target_rows, builder, label, builder_seed in class_specs:
        if target_rows <= 0:
            continue
        episode_span = episode_span_for_target_rows(target_rows)
        schedule_rows = builder(
            target_rows=target_rows,
            seed=builder_seed,
            label=label,
            class_name=class_name,
            episode_offset=episode_offset,
            episode_span=episode_span,
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
                "run_id": plan.run_id,
                "run_order": plan.run_order,
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
        "class_episode_counts": class_episode_counts,
        "execution_order": execution_order,
        "class_order": [entry["class_name"] for entry in execution_order],
    }
    return plans, manifest


def summarize_canonical_rows(canonical_rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter()
    protocol_families = Counter()
    platform_families = Counter()
    command_families = Counter()
    phases = Counter()
    attack_families = Counter()
    for row in canonical_rows:
        audit = dict(row.get("audit_context") or {})
        semantics = dict(row.get("command_semantics") or {})
        mission = dict(row.get("mission_context") or {})
        labels[str(audit.get("label_name") or audit.get("label") or "unknown")] += 1
        protocol_families[str(row.get("protocol_family") or "unknown")] += 1
        platform_families[str(row.get("platform_family") or "unknown")] += 1
        command_families[str(semantics.get("canonical_command_family") or "other_or_unknown")] += 1
        phases[str(mission.get("mission_phase") or "unknown")] += 1
        attack_families[str(audit.get("attack_family") or "none")] += 1
    return {
        "rows": len(canonical_rows),
        "labels": dict(labels),
        "protocol_families": dict(protocol_families),
        "platform_families": dict(platform_families),
        "command_families": dict(command_families),
        "mission_phases": dict(phases),
        "attack_families": dict(attack_families),
    }


def _load_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    if not path.exists():
        return rows
    with path.open(encoding="utf-8") as handle:
        for line in handle:
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise SystemExit(f"{path} must contain JSON objects")
            rows.append(payload)
    return rows


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


def write_schema_report(
    report_dir: Path,
    *,
    packets: list[dict[str, Any]],
    transactions: list[dict[str, Any]],
    raw_packets: list[dict[str, Any]],
    raw_transactions: list[dict[str, Any]],
    canonical_rows: list[dict[str, Any]],
    dataset_rows: list[dict[str, Any]],
    artifact_paths: dict[str, str],
    history_featurization: dict[str, Any],
) -> Path:
    payload = {
        "schema_version": "real_mavlink_generation.v1",
        "raw_packet_schema_version": raw_packets[0]["schema_version"] if raw_packets else None,
        "raw_transaction_schema_version": raw_transactions[0]["schema_version"] if raw_transactions else None,
        "canonical_command_row_schema_version": CANONICAL_COMMAND_ROW_SCHEMA_VERSION,
        "artifact_paths": dict(artifact_paths),
        "artifact_row_counts": {
            "legacy_packets": len(packets),
            "legacy_transactions": len(transactions),
            "raw_packets": len(raw_packets),
            "raw_transactions": len(raw_transactions),
            "canonical_command_rows": len(canonical_rows),
            "dataset_rows": len(dataset_rows),
        },
        "legacy_packet_fields": collect_record_field_paths(packets),
        "legacy_transaction_fields": collect_record_field_paths(transactions),
        "raw_packet_fields": collect_record_field_paths(raw_packets),
        "raw_transaction_fields": collect_record_field_paths(raw_transactions),
        "canonical_command_row_fields": collect_record_field_paths(canonical_rows),
        "dataset_fields": collect_record_field_paths(dataset_rows),
        "history_featurization": dict(history_featurization),
        "poster_default_feature_names": list(poster_default_model_feature_names()),
    }
    path = report_dir / "schema.json"
    save_json(path, payload)
    return path


def execute_run_plan(
    *,
    plan: IndependentRunPlan,
    repo_root: Path,
    aggregate_runtime_root: Path,
    timeout_seconds: float,
    time_scale: float,
) -> dict[str, Any]:
    run_name = f"run_{plan.run_id:04d}_{plan.class_name}_ep_{plan.episode_id:04d}"
    per_run_root = aggregate_runtime_root / "runs" / run_name
    schedule_path = host_schedules_dir(aggregate_runtime_root) / f"{run_name}.csv"
    run_log_path = host_schedule_run_logs_dir(aggregate_runtime_root) / f"{run_name}.csv"
    pcap_path = host_capture_pcap_path(per_run_root, "traffic")
    compose_file = repo_root / "orchestration" / "docker-compose.mavlink-real.yml"

    safe_remove_tree(per_run_root)
    ensure_runtime_tree(per_run_root)
    write_schedule_csv(schedule_path, plan.schedule_rows)
    try:
        shutdown_stack(repo_root)
        run(
            [
                "bash",
                str(repo_root / "scripts" / "mavlink_real" / "up.sh"),
                "--runtime-root",
                str(per_run_root),
            ],
            repo_root,
        )
        resolved_capture = resolve_identity_capture_target(
            repo_root,
            compose_file,
            timeout_seconds=timeout_seconds,
            runtime_root=per_run_root,
        )
        with capture_pcap(
            pcap_path,
            interface=resolved_capture.interface,
            backend=resolved_capture.backend,
        ) as session:
            run(
                [
                    "python3",
                    str(repo_root / "tools" / "mavlink_real" / "run_mavlink_schedule.py"),
                    "--schedule",
                    str(schedule_path),
                    "--time-scale",
                    str(time_scale),
                    "--timeout-seconds",
                    str(timeout_seconds),
                    "--output",
                    str(run_log_path),
                ],
                repo_root,
            )
            run_rows, source_artifact_paths = load_runtime_run_rows(
                run_log_path,
                runtime_root=per_run_root,
            )
            source_artifact_paths.extend(
                [
                    str(schedule_path.resolve()),
                    str(pcap_path.resolve()),
                ]
            )
            packet_result = wait_for_capture_drain(
                run_rows,
                pcap_path=pcap_path,
                capture_interface=session.interface,
                capture_backend=resolved_capture.backend,
                source_artifact_paths=source_artifact_paths,
                timeout_seconds=timeout_seconds,
            )
        return {
            "run_name": run_name,
            "run_rows": run_rows,
            "run_log_path": run_log_path,
            "schedule_path": schedule_path,
            "runtime_root": per_run_root,
            "pcap_path": pcap_path,
            "capture_backend": resolved_capture.backend,
            "capture_interface": resolved_capture.interface,
            "packet_result": packet_result,
        }
    finally:
        shutdown_stack(repo_root)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--nominal-ratio", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--time-scale", type=float, default=14400.0)
    parser.add_argument("--timeout-seconds", type=float, default=8.0)
    args = parser.parse_args()

    if not 0.0 < args.nominal_ratio < 1.0:
        raise SystemExit("--nominal-ratio must be between 0 and 1")

    repo_root = REPO_ROOT
    output_dir = Path(args.output_dir).resolve()
    data_dir = output_dir / "data"
    report_dir = output_dir / "reports"
    runtime_root = runtime_root_for_output(output_dir)
    safe_remove_tree(runtime_root)
    report_dir.mkdir(parents=True, exist_ok=True)
    ensure_runtime_tree(runtime_root)

    targets = class_row_targets(rows=args.rows, nominal_ratio=args.nominal_ratio)
    run_plans, run_manifest = build_independent_run_plans(
        nominal_target=targets["benign"],
        cyber_target=targets["cyber"],
        fault_target=targets["fault"],
        seed=args.seed,
    )
    save_json(report_dir / "run_manifest.json", run_manifest)

    packets: list[dict[str, Any]] = []
    transactions: list[dict[str, Any]] = []
    raw_packets: list[dict[str, Any]] = []
    raw_transactions: list[dict[str, Any]] = []
    observations: list[dict[str, Any]] = []
    run_rows: list[dict[str, Any]] = []
    per_run_reports: list[dict[str, Any]] = []
    provenance_summaries: list[dict[str, Any]] = []
    capture_backend = ""
    capture_interface = ""

    for plan in run_plans:
        result = execute_run_plan(
            plan=plan,
            repo_root=repo_root,
            aggregate_runtime_root=runtime_root,
            timeout_seconds=args.timeout_seconds,
            time_scale=args.time_scale,
        )
        capture_backend = result["capture_backend"]
        capture_interface = result["capture_interface"]
        observation_offset = len(run_rows)
        for observation in result["packet_result"].observations:
            updated = dict(observation)
            updated["row_index"] = int(updated.get("row_index", 0)) + observation_offset
            observations.append(updated)
        run_rows.extend(result["run_rows"])
        packets.extend(result["packet_result"].packets)
        transactions.extend(result["packet_result"].transactions)
        raw_packets.extend(result["packet_result"].raw_packets)
        raw_transactions.extend(result["packet_result"].raw_transactions)
        provenance_summaries.append(dict(result["packet_result"].provenance_summary))
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
            }
        )

    actual_run_report = build_actual_run_observability_report(run_rows, observations)
    save_json(report_dir / "actual_run_observability.json", actual_run_report)
    assert_actual_run_observability(actual_run_report)

    recent_behavior_rows = build_recent_behavior_rows_from_raw_transactions(raw_transactions, reset_key="run_id")
    canonical_rows = build_canonical_rows_from_raw_transactions(
        raw_transactions,
        recent_behavior_rows=recent_behavior_rows,
        require_recent_behavior=True,
    )
    dataset_rows = canonical_rows_to_training_rows(canonical_rows)

    artifact_paths = {
        "legacy_packets": str((data_dir / "packets.jsonl").resolve()),
        "legacy_transactions": str((data_dir / "transactions.jsonl").resolve()),
        "legacy_dataset": str((data_dir / "dataset.jsonl").resolve()),
        "raw_packets": str((data_dir / "raw_packets.jsonl").resolve()),
        "raw_transactions": str((data_dir / "raw_transactions.jsonl").resolve()),
        "canonical_command_rows": str((data_dir / "canonical_command_rows.jsonl").resolve()),
    }

    write_jsonl(data_dir / "packets.jsonl", packets)
    write_jsonl(data_dir / "transactions.jsonl", transactions)
    write_jsonl(data_dir / "dataset.jsonl", dataset_rows)
    write_jsonl(data_dir / "raw_packets.jsonl", raw_packets)
    write_jsonl(data_dir / "raw_transactions.jsonl", raw_transactions)
    write_jsonl(data_dir / "canonical_command_rows.jsonl", canonical_rows)

    combined_provenance = {
        "schema_version": "real_mavlink_generation.v1",
        "protocol_family": "mavlink",
        "capture_backend": capture_backend,
        "capture_interface": capture_interface,
        "independent_runs": len(per_run_reports),
        "runtime_reset_policy": RUNTIME_RESET_POLICY,
        "packet_count": len(packets),
        "transaction_count": len(transactions),
        "raw_packet_count": len(raw_packets),
        "raw_transaction_count": len(raw_transactions),
        "request_wire_seen_count": sum(int(item.get("request_wire_seen", False)) for item in observations),
        "response_direction_seen_count": sum(int(item.get("response_direction_seen", False)) for item in observations),
        "telemetry_recent_count": sum(int(item.get("telemetry_recent", False)) for item in observations),
        "state_snapshot_seen_count": sum(int(item.get("state_snapshot_seen", False)) for item in observations),
        "pcap_packet_count": sum(int(item.get("pcap_packet_count", 0) or 0) for item in provenance_summaries),
        "decoded_wire_message_count": sum(int(item.get("decoded_wire_message_count", 0) or 0) for item in provenance_summaries),
        "telemetry_message_count": sum(int(item.get("telemetry_message_count", 0) or 0) for item in provenance_summaries),
        "request_row_count": sum(int(item.get("request_row_count", 0) or 0) for item in provenance_summaries),
    }
    save_json(report_dir / "provenance_summary.json", combined_provenance)

    history_featurization = {
        "source": "raw_transactions",
        "group_key": "run_id",
        "state_reset": RUNTIME_RESET_POLICY,
        "protocol_neutral_recent_behavior": True,
    }
    schema_path = write_schema_report(
        report_dir,
        packets=packets,
        transactions=transactions,
        raw_packets=raw_packets,
        raw_transactions=raw_transactions,
        canonical_rows=canonical_rows,
        dataset_rows=dataset_rows,
        artifact_paths=artifact_paths,
        history_featurization=history_featurization,
    )

    summary = summarize_canonical_rows(canonical_rows)
    summary.update(
        {
            "schema_version": "real_mavlink_generation.v1",
            "packet_count": len(packets),
            "transaction_count": len(transactions),
            "raw_packet_count": len(raw_packets),
            "raw_transaction_count": len(raw_transactions),
            "canonical_command_row_count": len(canonical_rows),
            "dataset_row_count": len(dataset_rows),
            "canonical_command_row_schema_version": CANONICAL_COMMAND_ROW_SCHEMA_VERSION,
            "data_artifacts": artifact_paths,
            "history_featurization": history_featurization,
            "runtime_root": str(runtime_root.resolve()),
            "run_manifest_report": str((report_dir / "run_manifest.json").resolve()),
            "independent_runs": run_manifest,
            "per_run_reports": per_run_reports,
            "actual_run_observability_report": str((report_dir / "actual_run_observability.json").resolve()),
            "provenance_summary_report": str((report_dir / "provenance_summary.json").resolve()),
            "schema_report": str(schema_path.resolve()),
        }
    )
    save_json(report_dir / "generation_summary.json", summary)
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
