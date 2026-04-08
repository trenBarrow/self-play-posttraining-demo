#!/usr/bin/env python3
"""Shared dataset orchestration for F´-only, MAVLink-only, or mixed poster runs."""

from __future__ import annotations

import argparse
import json
import shutil
import subprocess
import sys
from collections import Counter
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime import save_json
from tools.shared.artifact_layers import build_canonical_rows_from_raw_transactions
from tools.shared.recent_behavior import build_recent_behavior_rows_from_raw_transactions
from tools.shared.run_manifest import (
    build_multi_protocol_run_manifest,
    protocol_row_targets,
    protocol_seed_map,
)
from tools.train.poster_default import canonical_rows_to_training_rows, poster_default_model_feature_names


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


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
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


def write_jsonl(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def safe_remove_tree(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)


def protocol_generator_script(protocol_family: str) -> Path:
    if protocol_family == "fprime":
        return REPO_ROOT / "tools" / "fprime_real" / "generate_dataset.py"
    if protocol_family == "mavlink":
        return REPO_ROOT / "tools" / "mavlink_real" / "generate_dataset.py"
    raise SystemExit(f"Unsupported protocol family {protocol_family!r}")


def invoke_protocol_generator(
    *,
    protocol_family: str,
    rows: int,
    nominal_ratio: float,
    seed: int,
    output_dir: Path,
    passthrough_args: list[str] | None = None,
) -> None:
    script_path = protocol_generator_script(protocol_family)
    python_bin = sys.executable or "python3"
    cmd = [
        python_bin,
        str(script_path),
        "--rows",
        str(rows),
        "--nominal-ratio",
        str(nominal_ratio),
        "--seed",
        str(seed),
        "--output-dir",
        str(output_dir),
    ]
    cmd.extend(list(passthrough_args or []))
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"{protocol_family} dataset generation failed: {exc}") from None


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


def summarize_canonical_rows(canonical_rows: list[dict[str, Any]]) -> dict[str, Any]:
    labels = Counter()
    protocols = Counter()
    platforms = Counter()
    command_families = Counter()
    attack_families = Counter()
    for row in canonical_rows:
        audit = dict(row.get("audit_context") or {})
        semantics = dict(row.get("command_semantics") or {})
        labels[str(audit.get("label_name") or audit.get("label") or "unknown")] += 1
        protocols[str(row.get("protocol_family") or "unknown")] += 1
        platforms[str(row.get("platform_family") or "unknown")] += 1
        command_families[str(semantics.get("canonical_command_family") or "other_or_unknown")] += 1
        attack_families[str(audit.get("attack_family") or "none")] += 1
    return {
        "rows": len(canonical_rows),
        "labels": dict(labels),
        "protocol_families": dict(protocols),
        "platform_families": dict(platforms),
        "command_families": dict(command_families),
        "attack_families": dict(attack_families),
    }


def remap_record_ids(record: dict[str, Any], *, run_id_map: dict[int, int], episode_id_map: dict[int, int]) -> dict[str, Any]:
    updated = json.loads(json.dumps(record))
    if isinstance(updated.get("run_id"), (int, float)) and int(updated["run_id"]) in run_id_map:
        updated["run_id"] = run_id_map[int(updated["run_id"])]
    if isinstance(updated.get("episode_id"), (int, float)) and int(updated["episode_id"]) in episode_id_map:
        updated["episode_id"] = episode_id_map[int(updated["episode_id"])]
    correlation = updated.get("correlation")
    if isinstance(correlation, dict):
        if isinstance(correlation.get("run_id"), (int, float)) and int(correlation["run_id"]) in run_id_map:
            correlation["run_id"] = run_id_map[int(correlation["run_id"])]
        if isinstance(correlation.get("episode_id"), (int, float)) and int(correlation["episode_id"]) in episode_id_map:
            correlation["episode_id"] = episode_id_map[int(correlation["episode_id"])]
    return updated


def load_protocol_bundle(protocol_root: Path, protocol_family: str) -> dict[str, Any]:
    data_dir = protocol_root / "data"
    report_dir = protocol_root / "reports"
    return {
        "protocol_family": protocol_family,
        "output_dir": str(protocol_root.resolve()),
        "packets": read_jsonl(data_dir / "packets.jsonl"),
        "transactions": read_jsonl(data_dir / "transactions.jsonl"),
        "raw_packets": read_jsonl(data_dir / "raw_packets.jsonl"),
        "raw_transactions": read_jsonl(data_dir / "raw_transactions.jsonl"),
        "generation_summary": read_json(report_dir / "generation_summary.json"),
        "run_manifest": read_json(report_dir / "run_manifest.json"),
    }


def build_protocol_id_maps(bundles: list[dict[str, Any]]) -> tuple[dict[tuple[str, int], int], dict[tuple[str, int], int]]:
    run_id_map: dict[tuple[str, int], int] = {}
    episode_id_map: dict[tuple[str, int], int] = {}
    next_run_id = 0
    next_episode_id = 0
    for bundle in bundles:
        protocol_family = str(bundle["protocol_family"])
        generation_summary = dict(bundle["generation_summary"])
        for report in generation_summary.get("per_run_reports", []):
            local_run_id = _optional_int(report.get("run_id"))
            if local_run_id is not None and (protocol_family, local_run_id) not in run_id_map:
                run_id_map[(protocol_family, local_run_id)] = next_run_id
                next_run_id += 1
            local_episode_id = _optional_int(report.get("episode_id"))
            if local_episode_id is not None and (protocol_family, local_episode_id) not in episode_id_map:
                episode_id_map[(protocol_family, local_episode_id)] = next_episode_id
                next_episode_id += 1
    return run_id_map, episode_id_map


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
        "schema_version": "multi_protocol_generation.v1",
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


def merge_protocol_bundles(
    *,
    bundles: list[dict[str, Any]],
    protocol_mode: str,
    requested_rows: int,
    nominal_ratio: float,
    seed: int,
    mixed_fprime_ratio: float | None,
    protocol_targets: dict[str, int],
    output_dir: Path,
) -> Path:
    data_dir = output_dir / "data"
    report_dir = output_dir / "reports"
    run_id_map_by_protocol, episode_id_map_by_protocol = build_protocol_id_maps(bundles)

    packets: list[dict[str, Any]] = []
    transactions: list[dict[str, Any]] = []
    raw_packets: list[dict[str, Any]] = []
    raw_transactions: list[dict[str, Any]] = []
    run_entries: list[dict[str, Any]] = []

    for bundle in bundles:
        protocol_family = str(bundle["protocol_family"])
        run_id_map = {
            local_run_id: global_run_id
            for (family, local_run_id), global_run_id in run_id_map_by_protocol.items()
            if family == protocol_family
        }
        episode_id_map = {
            local_episode_id: global_episode_id
            for (family, local_episode_id), global_episode_id in episode_id_map_by_protocol.items()
            if family == protocol_family
        }
        packets.extend(remap_record_ids(record, run_id_map=run_id_map, episode_id_map=episode_id_map) for record in bundle["packets"])
        transactions.extend(remap_record_ids(record, run_id_map=run_id_map, episode_id_map=episode_id_map) for record in bundle["transactions"])
        raw_packets.extend(remap_record_ids(record, run_id_map=run_id_map, episode_id_map=episode_id_map) for record in bundle["raw_packets"])
        raw_transactions.extend(remap_record_ids(record, run_id_map=run_id_map, episode_id_map=episode_id_map) for record in bundle["raw_transactions"])
        generation_summary = dict(bundle["generation_summary"])
        protocol_seed = _optional_int(generation_summary.get("seed")) or _optional_int(generation_summary.get("requested_seed"))
        for report in generation_summary.get("per_run_reports", []):
            local_run_id = _optional_int(report.get("run_id"))
            local_episode_id = _optional_int(report.get("episode_id"))
            if local_run_id is None or local_episode_id is None:
                continue
            run_entries.append(
                {
                    "run_id": run_id_map[local_run_id],
                    "run_order": run_id_map[local_run_id],
                    "protocol_family": protocol_family,
                    "class_name": _optional_text(report.get("class_name")) or "unknown",
                    "label": _optional_int(report.get("label")),
                    "episode_id": episode_id_map[local_episode_id],
                    "rows": _optional_int(report.get("rows")) or 0,
                    "runtime_reset_policy": dict(bundle["run_manifest"]).get("runtime_reset_policy", "fresh_runtime_per_episode"),
                    "seed": protocol_seed,
                    "schedule_path": _optional_text(report.get("schedule_path")),
                    "generation_root": str(bundle["output_dir"]),
                }
            )

    recent_behavior_rows = build_recent_behavior_rows_from_raw_transactions(raw_transactions, reset_key="run_id")
    canonical_rows = build_canonical_rows_from_raw_transactions(
        raw_transactions,
        recent_behavior_rows=recent_behavior_rows,
        require_recent_behavior=True,
    )
    dataset_rows = canonical_rows_to_training_rows(canonical_rows)

    write_jsonl(data_dir / "packets.jsonl", packets)
    write_jsonl(data_dir / "transactions.jsonl", transactions)
    write_jsonl(data_dir / "raw_packets.jsonl", raw_packets)
    write_jsonl(data_dir / "raw_transactions.jsonl", raw_transactions)
    write_jsonl(data_dir / "canonical_command_rows.jsonl", canonical_rows)
    write_jsonl(data_dir / "dataset.jsonl", dataset_rows)

    protocol_execution_order = [bundle["protocol_family"] for bundle in bundles]
    run_manifest = build_multi_protocol_run_manifest(
        protocol_mode=protocol_mode,
        requested_rows=requested_rows,
        nominal_ratio=nominal_ratio,
        seed=seed,
        mixed_fprime_ratio=mixed_fprime_ratio,
        history_group_key="run_id",
        history_reset_policy="fresh_runtime_per_episode",
        protocol_execution_order=protocol_execution_order,
        protocol_row_targets_by_family=protocol_targets,
        runs=sorted(run_entries, key=lambda item: int(item["run_order"])),
    )
    save_json(report_dir / "run_manifest.json", run_manifest)

    artifact_paths = {
        "legacy_packets": str((data_dir / "packets.jsonl").resolve()),
        "legacy_transactions": str((data_dir / "transactions.jsonl").resolve()),
        "legacy_dataset": str((data_dir / "dataset.jsonl").resolve()),
        "raw_packets": str((data_dir / "raw_packets.jsonl").resolve()),
        "raw_transactions": str((data_dir / "raw_transactions.jsonl").resolve()),
        "canonical_command_rows": str((data_dir / "canonical_command_rows.jsonl").resolve()),
    }
    history_featurization = {
        "source": "raw_transactions",
        "group_key": "run_id",
        "state_reset": "fresh_runtime_per_episode",
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
            "schema_version": "multi_protocol_generation.v1",
            "protocol_mode": protocol_mode,
            "requested_rows": requested_rows,
            "nominal_ratio": nominal_ratio,
            "mixed_fprime_ratio": mixed_fprime_ratio,
            "packet_count": len(packets),
            "transaction_count": len(transactions),
            "raw_packet_count": len(raw_packets),
            "raw_transaction_count": len(raw_transactions),
            "canonical_command_row_count": len(canonical_rows),
            "dataset_row_count": len(dataset_rows),
            "history_featurization": history_featurization,
            "data_artifacts": artifact_paths,
            "protocol_artifact_roots": {
                bundle["protocol_family"]: str(bundle["output_dir"])
                for bundle in bundles
            },
            "run_manifest_report": str((report_dir / "run_manifest.json").resolve()),
            "schema_report": str(schema_path.resolve()),
            "per_protocol_generation_summaries": {
                bundle["protocol_family"]: str(
                    (Path(str(bundle["output_dir"])) / "reports" / "generation_summary.json").resolve()
                )
                for bundle in bundles
            },
            "per_protocol_episode_support": {
                str(bundle["protocol_family"]): {
                    "class_episode_counts": dict(
                        (
                            dict(bundle["run_manifest"]).get("class_episode_counts")
                            or dict(bundle["generation_summary"]).get("independent_runs", {}).get("class_episode_counts")
                            or {}
                        )
                    ),
                    "class_rows_per_episode_summary": dict(
                        (
                            dict(bundle["run_manifest"]).get("class_rows_per_episode_summary")
                            or dict(bundle["generation_summary"]).get("independent_runs", {}).get("class_rows_per_episode_summary")
                            or {}
                        )
                    ),
                    "episode_policy": dict(
                        (
                            dict(bundle["run_manifest"]).get("episode_policy")
                            or dict(bundle["generation_summary"]).get("independent_runs", {}).get("episode_policy")
                            or {}
                        )
                    ),
                }
                for bundle in bundles
            },
        }
    )
    save_json(report_dir / "generation_summary.json", summary)
    return data_dir / "dataset.jsonl"


def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--protocol-mode", choices=["fprime", "mavlink", "mixed"], default="fprime")
    parser.add_argument("--rows", type=int, required=True)
    parser.add_argument("--nominal-ratio", type=float, default=0.55)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--output-dir", required=True)
    parser.add_argument("--mixed-fprime-ratio", type=float, default=0.5)
    parser.add_argument("--fprime-time-scale", type=float, default=7200.0)
    parser.add_argument("--fprime-timeout-seconds", type=float, default=15.0)
    parser.add_argument("--mavlink-time-scale", type=float, default=14400.0)
    parser.add_argument("--mavlink-timeout-seconds", type=float, default=8.0)
    return parser.parse_args(argv)


def main(argv: list[str] | None = None) -> int:
    args = parse_args(argv)
    output_dir = Path(args.output_dir).resolve()
    protocol_mode = str(args.protocol_mode)
    targets = protocol_row_targets(
        protocol_mode=protocol_mode,
        rows=int(args.rows),
        mixed_fprime_ratio=float(args.mixed_fprime_ratio),
    )

    if protocol_mode == "fprime":
        invoke_protocol_generator(
            protocol_family="fprime",
            rows=targets["fprime"],
            nominal_ratio=float(args.nominal_ratio),
            seed=int(args.seed),
            output_dir=output_dir,
            passthrough_args=[
                "--time-scale",
                str(args.fprime_time_scale),
                "--timeout-seconds",
                str(args.fprime_timeout_seconds),
            ],
        )
        return 0

    safe_remove_tree(output_dir)
    protocol_root = output_dir / "protocol_runs"
    protocol_root.mkdir(parents=True, exist_ok=True)
    protocol_order = ["mavlink"] if protocol_mode == "mavlink" else ["fprime", "mavlink"]
    seeds = protocol_seed_map(seed=int(args.seed), protocol_families=protocol_order)
    bundles: list[dict[str, Any]] = []
    for protocol_family in protocol_order:
        protocol_rows = int(targets.get(protocol_family, 0) or 0)
        if protocol_rows <= 0:
            continue
        protocol_output_dir = protocol_root / protocol_family
        passthrough_args: list[str] = []
        if protocol_family == "fprime":
            passthrough_args = [
                "--time-scale",
                str(args.fprime_time_scale),
                "--timeout-seconds",
                str(args.fprime_timeout_seconds),
            ]
        if protocol_family == "mavlink":
            passthrough_args = [
                "--time-scale",
                str(args.mavlink_time_scale),
                "--timeout-seconds",
                str(args.mavlink_timeout_seconds),
            ]
        invoke_protocol_generator(
            protocol_family=protocol_family,
            rows=protocol_rows,
            nominal_ratio=float(args.nominal_ratio),
            seed=int(seeds[protocol_family]),
            output_dir=protocol_output_dir,
            passthrough_args=passthrough_args,
        )
        bundle = load_protocol_bundle(protocol_output_dir, protocol_family)
        bundle["generation_summary"]["requested_seed"] = int(seeds[protocol_family])
        bundles.append(bundle)

    merge_protocol_bundles(
        bundles=bundles,
        protocol_mode=protocol_mode,
        requested_rows=int(args.rows),
        nominal_ratio=float(args.nominal_ratio),
        seed=int(args.seed),
        mixed_fprime_ratio=None if protocol_mode != "mixed" else float(args.mixed_fprime_ratio),
        protocol_targets=targets,
        output_dir=output_dir,
    )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
