#!/usr/bin/env python3
"""Evaluate static and learned adversaries against poster blue bundles."""

from __future__ import annotations

import argparse
import json
import sys
from collections import Counter
from pathlib import Path
from typing import Any, Iterable, Mapping

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime import DEFAULT_MODEL_DIR, load_runtime_bundle
from tools.train.checkpointing import load_self_play_state
from tools.train.red_policy_model import RED_POLICY_MODEL_ARTIFACT_NAME, LoadedRedPolicyModel, load_red_action_space
from tools.train.red_reward import load_red_reward_spec
from tools.train.red_transcript import load_red_context_budget
from tools.train.run_self_play import (
    build_replay_examples_from_dataset,
    evaluate_red_policy_alignment,
    score_and_reward_replay_examples,
)

RED_BLUE_EVALUATION_SCHEMA_VERSION = "red_blue_evaluation.v1"
DEFAULT_REPORT_ARTIFACT_NAME = "red_blue_evaluation.json"
DEFAULT_SUMMARY_ARTIFACT_NAME = "red_blue_evaluation_summary.txt"
STATIC_ADVERSARY_KIND = "static_schedule_replay"
LEARNED_ADVERSARY_KIND = "learned_red_policy_retrieval"
BREAKDOWN_FIELDS = ("protocol_family", "command_family", "window_class")
SUCCESS_REWARD_CASES = frozenset({"unsafe_executed_undetected", "unsafe_accepted_undetected"})


class RedBlueEvaluationError(ValueError):
    """Raised when red-vs-blue evaluation inputs are invalid."""


def _clone_json(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _number(value: Any) -> float | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return None
    if isinstance(value, (int, float)):
        return float(value)
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _compact_category(value: Any, default: str = "unknown") -> str:
    text = (_text(value) or default).strip().lower()
    chars: list[str] = []
    last_sep = False
    for char in text:
        if char.isalnum():
            chars.append(char)
            last_sep = False
            continue
        if not last_sep:
            chars.append("_")
            last_sep = True
    compact = "".join(chars).strip("_")
    return compact or default


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _read_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise RedBlueEvaluationError(f"{path} must contain a JSON object")
    return payload


def _dedupe_entries(entries: Iterable[Mapping[str, Any]], path_key: str) -> list[dict[str, Any]]:
    deduped: list[dict[str, Any]] = []
    seen_paths: set[str] = set()
    for entry in entries:
        resolved_path = _text(entry.get(path_key))
        if resolved_path is None or resolved_path in seen_paths:
            continue
        seen_paths.add(resolved_path)
        deduped.append(dict(entry))
    return deduped


def _build_catalog_ids(entries: list[dict[str, Any]], prefix: str) -> list[dict[str, Any]]:
    catalog: list[dict[str, Any]] = []
    for index, entry in enumerate(entries, start=1):
        catalog_entry = dict(entry)
        catalog_entry[f"{prefix}_id"] = f"{prefix}_{index:02d}"
        catalog.append(catalog_entry)
    return catalog


def _generation_summary_path(dataset_path: Path) -> Path:
    return dataset_path.resolve().parents[1] / "reports" / "generation_summary.json"


def _load_generation_summary(dataset_path: Path) -> dict[str, Any] | None:
    summary_path = _generation_summary_path(dataset_path)
    if not summary_path.exists():
        return None
    return _read_json(summary_path)


def _resolve_transcript_lengths(
    requested_lengths: list[int] | None,
    transcript_budget: Mapping[str, Any],
) -> list[int]:
    configured_limit = int(_as_mapping(transcript_budget.get("limits")).get("max_history_entries", 0) or 0)
    lengths = list(requested_lengths or [])
    if not lengths:
        lengths = [configured_limit]
    normalized: list[int] = []
    for length in lengths:
        if int(length) <= 0:
            raise RedBlueEvaluationError("max_history_entries values must be positive")
        if configured_limit > 0 and int(length) > configured_limit:
            raise RedBlueEvaluationError(
                f"max_history_entries={length} exceeds configured transcript budget {configured_limit}"
            )
        if int(length) not in normalized:
            normalized.append(int(length))
    return normalized


def _dataset_label(path: Path, metadata: Mapping[str, Any]) -> str:
    source_kind = _text(metadata.get("source_kind"))
    if source_kind == "self_play_round":
        output_dir = Path(str(metadata.get("self_play_output_dir", path.parent))).name
        round_slug = _text(metadata.get("round_slug")) or "round"
        return f"{output_dir}:{round_slug}"
    return path.parent.parent.name or path.parent.name or path.name


def _blue_label(path: Path, metadata: Mapping[str, Any]) -> str:
    source_kind = _text(metadata.get("source_kind"))
    if source_kind == "default_deployment":
        return "deployment_default"
    if source_kind == "self_play_checkpoint":
        output_dir = Path(str(metadata.get("self_play_output_dir", path.parent))).name
        checkpoint_id = _text(metadata.get("checkpoint_id")) or path.name
        return f"{output_dir}:blue:{checkpoint_id}"
    return path.name


def _red_label(path: Path, metadata: Mapping[str, Any]) -> str:
    source_kind = _text(metadata.get("source_kind"))
    if source_kind == "self_play_checkpoint":
        output_dir = Path(str(metadata.get("self_play_output_dir", path.parent))).name
        checkpoint_id = _text(metadata.get("checkpoint_id")) or path.parent.name
        return f"{output_dir}:red:{checkpoint_id}"
    return path.parent.name if path.name == RED_POLICY_MODEL_ARTIFACT_NAME else path.name


def discover_self_play_resources(output_dir: Path) -> dict[str, list[dict[str, Any]]]:
    resolved_output_dir = output_dir.resolve()
    state = load_self_play_state(resolved_output_dir)
    datasets: list[dict[str, Any]] = []
    if state is not None:
        for round_summary in state.get("rounds", []):
            round_payload = _as_mapping(round_summary)
            dataset = _as_mapping(round_payload.get("dataset"))
            dataset_path = _text(dataset.get("dataset_path"))
            if dataset_path is None:
                continue
            datasets.append(
                {
                    "dataset_path": str(Path(dataset_path).resolve()),
                    "source_kind": "self_play_round",
                    "self_play_output_dir": str(resolved_output_dir),
                    "round_index": int(round_payload.get("round_index") or 0),
                    "round_slug": _text(round_payload.get("round_slug")) or "",
                    "source_mode": _text(dataset.get("source_mode")) or "unknown",
                }
            )
    checkpoints: dict[str, list[dict[str, Any]]] = {"blue": [], "red": []}
    for side in ("blue", "red"):
        checkpoint_root = resolved_output_dir / "checkpoints" / side
        if not checkpoint_root.exists():
            continue
        for checkpoint_json in sorted(checkpoint_root.glob("*/checkpoint.json")):
            payload = _read_json(checkpoint_json)
            artifact_dir = Path(str(payload.get("artifact_dir", ""))).resolve()
            if not artifact_dir.exists():
                continue
            checkpoints[side].append(
                {
                    f"{side}_path": str(artifact_dir),
                    "source_kind": "self_play_checkpoint",
                    "self_play_output_dir": str(resolved_output_dir),
                    "checkpoint_id": _text(payload.get("checkpoint_id")) or artifact_dir.name,
                    "metadata": dict(payload.get("metadata") or {}),
                }
            )
    return {
        "datasets": _dedupe_entries(datasets, "dataset_path"),
        "blue": _dedupe_entries(checkpoints["blue"], "blue_path"),
        "red": _dedupe_entries(checkpoints["red"], "red_path"),
    }


def build_dataset_catalog(
    dataset_paths: list[Path],
    *,
    self_play_output_dirs: list[Path] | None = None,
) -> list[dict[str, Any]]:
    discovered_entries: list[dict[str, Any]] = []
    for output_dir in self_play_output_dirs or []:
        discovered_entries.extend(discover_self_play_resources(output_dir)["datasets"])
    for dataset_path in dataset_paths:
        discovered_entries.append(
            {
                "dataset_path": str(dataset_path.resolve()),
                "source_kind": "manual_dataset",
            }
        )
    deduped = _dedupe_entries(discovered_entries, "dataset_path")
    if not deduped:
        raise RedBlueEvaluationError("At least one dataset path or self-play output with round datasets is required")
    catalog = _build_catalog_ids(deduped, "dataset")
    for entry in catalog:
        dataset_path = Path(str(entry["dataset_path"])).resolve()
        generation_summary = _load_generation_summary(dataset_path)
        entry["dataset_label"] = _dataset_label(dataset_path, entry)
        entry["generation_summary_path"] = (
            None if generation_summary is None else str(_generation_summary_path(dataset_path).resolve())
        )
        entry["protocol_mode"] = None if generation_summary is None else generation_summary.get("protocol_mode")
        entry["rows"] = None if generation_summary is None else generation_summary.get("rows")
        entry["run_count"] = None if generation_summary is None else generation_summary.get("run_count")
    return catalog


def build_blue_variant_catalog(
    blue_model_dirs: list[Path] | None = None,
    *,
    self_play_output_dirs: list[Path] | None = None,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for output_dir in self_play_output_dirs or []:
        entries.extend(discover_self_play_resources(output_dir)["blue"])
    for model_dir in blue_model_dirs or []:
        entries.append(
            {
                "blue_path": str(model_dir.resolve()),
                "source_kind": "manual_blue_model",
            }
        )
    if not entries:
        entries.append(
            {
                "blue_path": str(Path(DEFAULT_MODEL_DIR).resolve()),
                "source_kind": "default_deployment",
            }
        )
    catalog = _build_catalog_ids(_dedupe_entries(entries, "blue_path"), "blue")
    for entry in catalog:
        bundle_dir = Path(str(entry["blue_path"])).resolve()
        bundle = load_runtime_bundle(bundle_dir)
        manifest = dict(getattr(bundle, "manifest", {}) or {})
        entry["blue_label"] = _blue_label(bundle_dir, entry)
        entry["runtime_kind"] = _text(getattr(bundle, "runtime_kind", None)) or _text(manifest.get("runtime_kind"))
        entry["training_path"] = _text(manifest.get("training_path"))
        entry["feature_count"] = len(list(getattr(bundle, "feature_names", []) or []))
    return catalog


def build_red_variant_catalog(
    red_model_paths: list[Path] | None = None,
    *,
    self_play_output_dirs: list[Path] | None = None,
) -> list[dict[str, Any]]:
    entries: list[dict[str, Any]] = []
    for output_dir in self_play_output_dirs or []:
        entries.extend(discover_self_play_resources(output_dir)["red"])
    for red_path in red_model_paths or []:
        resolved_path = red_path.resolve()
        model_path = resolved_path if resolved_path.name == RED_POLICY_MODEL_ARTIFACT_NAME else resolved_path / RED_POLICY_MODEL_ARTIFACT_NAME
        entries.append(
            {
                "red_path": str(model_path.resolve()),
                "source_kind": "manual_red_model",
            }
        )
    catalog = _build_catalog_ids(_dedupe_entries(entries, "red_path"), "red")
    for entry in catalog:
        model_path = Path(str(entry["red_path"])).resolve()
        if model_path.is_dir():
            model_path = model_path / RED_POLICY_MODEL_ARTIFACT_NAME
        if not model_path.exists():
            raise RedBlueEvaluationError(f"Missing red policy model: {model_path}")
        payload = _read_json(model_path)
        entry["red_path"] = str(model_path.resolve())
        entry["red_label"] = _red_label(model_path, entry)
        entry["model_family"] = _text(payload.get("model_family")) or "unknown"
        transcript_budget = _as_mapping(payload.get("transcript_budget"))
        entry["model_transcript_budget"] = int(transcript_budget.get("max_history_entries", 0) or 0)
        evaluation = _as_mapping(payload.get("evaluation"))
        entry["joint_exact_match_accuracy"] = _number(evaluation.get("joint_exact_match_accuracy"))
    return catalog


def _attempt_slice_value(attempt: Mapping[str, Any], field_name: str) -> str:
    if field_name == "protocol_family":
        return _compact_category(_as_mapping(attempt.get("current_context")).get("protocol_family"), "unknown")
    if field_name == "command_family":
        return _compact_category(_as_mapping(attempt.get("action")).get("command_family"), "unknown")
    if field_name == "window_class":
        return _compact_category(_as_mapping(attempt.get("current_context")).get("window_class"), "unknown")
    raise RedBlueEvaluationError(f"Unsupported breakdown field {field_name!r}")


def summarize_evaluation_attempts(attempts: list[Mapping[str, Any]]) -> dict[str, Any]:
    source_count = len(attempts)
    matched_attempts = [attempt for attempt in attempts if bool(attempt.get("matched"))]
    rewards: list[float] = []
    reward_case_counts = Counter()
    sandbox_violation_counts = Counter()
    unsafe_count = 0
    detected_count = 0
    detected_unsafe_count = 0
    adversary_success_count = 0
    positive_reward_count = 0
    for attempt in matched_attempts:
        rewarded_record = _as_mapping(attempt.get("matched_rewarded_record"))
        reward_result = _as_mapping(rewarded_record.get("reward_result"))
        environment_feedback = _as_mapping(reward_result.get("environment_feedback"))
        blue_feedback = _as_mapping(reward_result.get("blue_feedback"))
        reward_value = float(reward_result.get("reward") or 0.0)
        rewards.append(reward_value)
        if reward_value > 0.0:
            positive_reward_count += 1
        reward_case = _text(reward_result.get("reward_case")) or "unknown"
        reward_case_counts[reward_case] += 1
        for reason in _as_mapping(reward_result.get("sandbox")).get("violation_reasons", []):
            sandbox_violation_counts[str(reason)] += 1
        unsafe_label = bool(environment_feedback.get("unsafe_label"))
        unsafe_detected = bool(blue_feedback.get("unsafe_detected"))
        if unsafe_label:
            unsafe_count += 1
            if unsafe_detected:
                detected_unsafe_count += 1
        if unsafe_detected:
            detected_count += 1
        if reward_case in SUCCESS_REWARD_CASES:
            adversary_success_count += 1

    matched_count = len(matched_attempts)
    benign_count = max(0, matched_count - unsafe_count)
    reward_sum = float(sum(rewards))
    return {
        "source_example_count": source_count,
        "example_count": matched_count,
        "matched_example_count": matched_count,
        "unmatched_prediction_count": max(0, source_count - matched_count),
        "retrieval_coverage_rate": float(matched_count / max(1, source_count)),
        "unsafe_example_count": unsafe_count,
        "benign_example_count": benign_count,
        "detected_example_count": detected_count,
        "detected_unsafe_count": detected_unsafe_count,
        "adversary_success_count": adversary_success_count,
        "adversary_success_rate": float(adversary_success_count / max(1, matched_count)),
        "coverage_adjusted_adversary_success_rate": float(adversary_success_count / max(1, source_count)),
        "blue_recall_under_attack": None if unsafe_count == 0 else float(detected_unsafe_count / unsafe_count),
        "blue_precision_under_attack": None if detected_count == 0 else float(detected_unsafe_count / detected_count),
        "blue_flag_rate": float(detected_count / max(1, matched_count)),
        "reward_sum": reward_sum,
        "reward_mean": float(reward_sum / max(1, matched_count)),
        "coverage_adjusted_reward_mean": float(reward_sum / max(1, source_count)),
        "reward_min": float(min(rewards)) if rewards else 0.0,
        "reward_max": float(max(rewards)) if rewards else 0.0,
        "positive_reward_rate": float(positive_reward_count / max(1, matched_count)),
        "reward_case_counts": dict(reward_case_counts),
        "sandbox_violation_counts": dict(sandbox_violation_counts),
    }


def _summary_row_from_attempts(
    *,
    row_id: str,
    evaluation_id: str,
    dataset_entry: Mapping[str, Any],
    blue_entry: Mapping[str, Any],
    red_entry: Mapping[str, Any] | None,
    adversary_kind: str,
    transcript_length: int,
    attempts: list[Mapping[str, Any]],
    alignment: Mapping[str, Any] | None = None,
    retrieval_summary: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    summary = summarize_evaluation_attempts(attempts)
    alignment_payload = _as_mapping(alignment)
    head_accuracy = _as_mapping(alignment_payload.get("head_accuracy"))
    retrieval_payload = _as_mapping(retrieval_summary)
    return {
        "row_id": row_id,
        "evaluation_id": evaluation_id,
        "scope": "overall",
        "dataset_id": dataset_entry["dataset_id"],
        "dataset_label": dataset_entry["dataset_label"],
        "dataset_path": dataset_entry["dataset_path"],
        "protocol_mode": dataset_entry.get("protocol_mode"),
        "blue_id": blue_entry["blue_id"],
        "blue_label": blue_entry["blue_label"],
        "blue_path": blue_entry["blue_path"],
        "red_id": None if red_entry is None else red_entry["red_id"],
        "red_label": None if red_entry is None else red_entry["red_label"],
        "red_path": None if red_entry is None else red_entry["red_path"],
        "adversary_kind": adversary_kind,
        "transcript_length": int(transcript_length),
        "source_example_count": int(summary["source_example_count"]),
        "example_count": int(summary["example_count"]),
        "matched_example_count": int(summary["matched_example_count"]),
        "unmatched_prediction_count": int(summary["unmatched_prediction_count"]),
        "retrieval_coverage_rate": float(summary["retrieval_coverage_rate"]),
        "unsafe_example_count": int(summary["unsafe_example_count"]),
        "benign_example_count": int(summary["benign_example_count"]),
        "detected_example_count": int(summary["detected_example_count"]),
        "detected_unsafe_count": int(summary["detected_unsafe_count"]),
        "adversary_success_count": int(summary["adversary_success_count"]),
        "adversary_success_rate": float(summary["adversary_success_rate"]),
        "coverage_adjusted_adversary_success_rate": float(summary["coverage_adjusted_adversary_success_rate"]),
        "blue_recall_under_attack": summary["blue_recall_under_attack"],
        "blue_precision_under_attack": summary["blue_precision_under_attack"],
        "blue_flag_rate": float(summary["blue_flag_rate"]),
        "reward_sum": float(summary["reward_sum"]),
        "reward_mean": float(summary["reward_mean"]),
        "coverage_adjusted_reward_mean": float(summary["coverage_adjusted_reward_mean"]),
        "reward_min": float(summary["reward_min"]),
        "reward_max": float(summary["reward_max"]),
        "positive_reward_rate": float(summary["positive_reward_rate"]),
        "alignment_joint_exact_match_accuracy": _number(alignment_payload.get("joint_exact_match_accuracy")),
        "alignment_command_family_accuracy": _number(head_accuracy.get("command_family")),
        "alignment_timing_bucket_accuracy": _number(head_accuracy.get("timing_bucket")),
        "alignment_identity_bucket_accuracy": _number(head_accuracy.get("identity_bucket")),
        "retrieval_match_key_count": _number(retrieval_payload.get("match_key_count")),
        "retrieval_candidate_pool_size": _number(retrieval_payload.get("candidate_pool_size")),
    }


def build_breakdown_rows(
    *,
    evaluation_id: str,
    dataset_entry: Mapping[str, Any],
    blue_entry: Mapping[str, Any],
    red_entry: Mapping[str, Any] | None,
    adversary_kind: str,
    transcript_length: int,
    attempts: list[Mapping[str, Any]],
    start_index: int = 1,
) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    row_index = start_index
    for field_name in BREAKDOWN_FIELDS:
        grouped: dict[str, list[Mapping[str, Any]]] = {}
        for attempt in attempts:
            grouped.setdefault(_attempt_slice_value(attempt, field_name), []).append(attempt)
        for value in sorted(grouped):
            summary = summarize_evaluation_attempts(grouped[value])
            rows.append(
                {
                    "row_id": f"breakdown_{row_index:04d}",
                    "evaluation_id": evaluation_id,
                    "scope": "breakdown",
                    "breakdown_field": field_name,
                    "breakdown_value": value,
                    "dataset_id": dataset_entry["dataset_id"],
                    "dataset_label": dataset_entry["dataset_label"],
                    "dataset_path": dataset_entry["dataset_path"],
                    "blue_id": blue_entry["blue_id"],
                    "blue_label": blue_entry["blue_label"],
                    "blue_path": blue_entry["blue_path"],
                    "red_id": None if red_entry is None else red_entry["red_id"],
                    "red_label": None if red_entry is None else red_entry["red_label"],
                    "red_path": None if red_entry is None else red_entry["red_path"],
                    "adversary_kind": adversary_kind,
                    "transcript_length": int(transcript_length),
                    "source_example_count": int(summary["source_example_count"]),
                    "example_count": int(summary["example_count"]),
                    "matched_example_count": int(summary["matched_example_count"]),
                    "unmatched_prediction_count": int(summary["unmatched_prediction_count"]),
                    "retrieval_coverage_rate": float(summary["retrieval_coverage_rate"]),
                    "unsafe_example_count": int(summary["unsafe_example_count"]),
                    "benign_example_count": int(summary["benign_example_count"]),
                    "detected_example_count": int(summary["detected_example_count"]),
                    "detected_unsafe_count": int(summary["detected_unsafe_count"]),
                    "adversary_success_count": int(summary["adversary_success_count"]),
                    "adversary_success_rate": float(summary["adversary_success_rate"]),
                    "coverage_adjusted_adversary_success_rate": float(summary["coverage_adjusted_adversary_success_rate"]),
                    "blue_recall_under_attack": summary["blue_recall_under_attack"],
                    "blue_precision_under_attack": summary["blue_precision_under_attack"],
                    "blue_flag_rate": float(summary["blue_flag_rate"]),
                    "reward_sum": float(summary["reward_sum"]),
                    "reward_mean": float(summary["reward_mean"]),
                    "coverage_adjusted_reward_mean": float(summary["coverage_adjusted_reward_mean"]),
                    "reward_min": float(summary["reward_min"]),
                    "reward_max": float(summary["reward_max"]),
                    "positive_reward_rate": float(summary["positive_reward_rate"]),
                }
            )
            row_index += 1
    return rows


def _direct_attempts_from_rewarded_examples(rewarded_examples: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    attempts: list[dict[str, Any]] = []
    for rewarded in rewarded_examples:
        attempts.append(
            {
                "current_context": _clone_json(rewarded.get("current_context")),
                "action": _clone_json(rewarded.get("action")),
                "metadata": _clone_json(rewarded.get("metadata")),
                "matched": True,
                "matched_rewarded_record": _clone_json(rewarded),
                "retrieval": {
                    "match_strategy": "direct_replay",
                    "candidate_count": 1,
                },
            }
        )
    return attempts


def _library_key(current_context: Mapping[str, Any], action: Mapping[str, Any]) -> tuple[str, str, str, str, str, str, str]:
    return (
        _compact_category(current_context.get("protocol_family"), "unknown"),
        _compact_category(current_context.get("platform_family"), "unknown"),
        _compact_category(current_context.get("mission_phase"), "unknown"),
        _compact_category(current_context.get("window_class"), "unknown"),
        _compact_category(action.get("command_family"), "unknown"),
        _compact_category(action.get("timing_bucket"), "unknown"),
        _compact_category(action.get("identity_bucket"), "unknown"),
    )


def _candidate_sort_key(source_example: Mapping[str, Any], rewarded_record: Mapping[str, Any]) -> tuple[Any, ...]:
    source_metadata = _as_mapping(source_example.get("metadata"))
    source_actor = _text(source_metadata.get("actor_id")) or ""
    source_target = _text(source_metadata.get("target_id")) or ""
    source_history_count = int(_number(_as_mapping(source_example.get("transcript")).get("included_history_count")) or 0)
    candidate_metadata = _as_mapping(rewarded_record.get("metadata"))
    candidate_actor = _text(candidate_metadata.get("actor_id")) or ""
    candidate_target = _text(candidate_metadata.get("target_id")) or ""
    candidate_history_count = int(_number(_as_mapping(rewarded_record.get("transcript")).get("included_history_count")) or 0)
    return (
        0 if source_actor == candidate_actor else 1,
        0 if source_target == candidate_target else 1,
        abs(source_history_count - candidate_history_count),
        abs(int(_number(source_metadata.get("sequence_index")) or 0) - int(_number(candidate_metadata.get("sequence_index")) or 0)),
        _text(candidate_metadata.get("group_key")) or "",
        int(_number(candidate_metadata.get("sequence_index")) or 0),
        _text(candidate_metadata.get("raw_command_name")) or "",
    )


def evaluate_learned_red_policy_retrieval(
    model_path: Path,
    source_examples: list[Mapping[str, Any]],
    rewarded_library: list[Mapping[str, Any]],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    model = LoadedRedPolicyModel.from_path(model_path.resolve())
    lookup: dict[tuple[str, str, str, str, str, str, str], list[dict[str, Any]]] = {}
    for rewarded_record in rewarded_library:
        key = _library_key(
            _as_mapping(rewarded_record.get("current_context")),
            _as_mapping(rewarded_record.get("action")),
        )
        lookup.setdefault(key, []).append(dict(rewarded_record))

    attempts: list[dict[str, Any]] = []
    matched_action_counts = Counter()
    unmatched_action_counts = Counter()
    for source_example in source_examples:
        predicted = model.predict_action(
            _as_mapping(source_example.get("transcript")),
            _as_mapping(source_example.get("current_context")),
        )
        predicted_action = _as_mapping(predicted.get("action"))
        current_context = _as_mapping(source_example.get("current_context"))
        retrieval_key = _library_key(current_context, predicted_action)
        candidates = list(lookup.get(retrieval_key, []))
        selected: dict[str, Any] | None = None
        if candidates:
            sorted_candidates = sorted(candidates, key=lambda record: _candidate_sort_key(source_example, record))
            source_sequence_index = int(_number(_as_mapping(source_example.get("metadata")).get("sequence_index")) or 0)
            selected = sorted_candidates[source_sequence_index % len(sorted_candidates)]
            matched_action_counts["|".join(retrieval_key[4:])] += 1
        else:
            unmatched_action_counts["|".join(retrieval_key[4:])] += 1
        attempts.append(
            {
                "current_context": _clone_json(current_context),
                "action": _clone_json(predicted_action),
                "metadata": _clone_json(source_example.get("metadata")),
                "matched": selected is not None,
                "matched_rewarded_record": None if selected is None else _clone_json(selected),
                "retrieval": {
                    "match_strategy": "action_context_retrieval",
                    "candidate_count": len(candidates),
                    "retrieval_key": list(retrieval_key),
                    "predicted_action": _clone_json(predicted_action),
                    "source_action": _clone_json(_as_mapping(source_example.get("action"))),
                    "matched_group_key": None if selected is None else _text(_as_mapping(selected.get("metadata")).get("group_key")),
                },
            }
        )

    matched_count = sum(1 for attempt in attempts if attempt["matched"])
    alignment = evaluate_red_policy_alignment(model_path.resolve(), source_examples)
    return attempts, {
        "source_example_count": len(source_examples),
        "matched_example_count": matched_count,
        "unmatched_prediction_count": len(source_examples) - matched_count,
        "retrieval_coverage_rate": float(matched_count / max(1, len(source_examples))),
        "match_key_count": len(lookup),
        "candidate_pool_size": len(rewarded_library),
        "matched_action_counts": dict(matched_action_counts),
        "unmatched_action_counts": dict(unmatched_action_counts),
        "alignment": alignment,
    }


def build_comparison_rows(
    summary_rows: list[Mapping[str, Any]],
    breakdown_rows: list[Mapping[str, Any]],
) -> list[dict[str, Any]]:
    comparison_rows: list[dict[str, Any]] = []
    static_overall_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    learned_overall_rows: list[dict[str, Any]] = []
    for row in summary_rows:
        normalized = dict(row)
        key = (
            normalized["dataset_id"],
            normalized["blue_id"],
            int(normalized["transcript_length"]),
        )
        if normalized["adversary_kind"] == STATIC_ADVERSARY_KIND:
            static_overall_by_key[key] = normalized
        elif normalized["adversary_kind"] == LEARNED_ADVERSARY_KIND:
            learned_overall_rows.append(normalized)

    comparison_index = 1
    for learned in learned_overall_rows:
        baseline = static_overall_by_key.get(
            (learned["dataset_id"], learned["blue_id"], int(learned["transcript_length"]))
        )
        if baseline is None:
            continue
        comparison_rows.append(
            {
                "comparison_id": f"comparison_{comparison_index:04d}",
                "scope": "overall",
                "dataset_id": learned["dataset_id"],
                "blue_id": learned["blue_id"],
                "red_id": learned["red_id"],
                "transcript_length": int(learned["transcript_length"]),
                "static_row_id": baseline["row_id"],
                "learned_row_id": learned["row_id"],
                "delta_adversary_success_rate": float(learned["adversary_success_rate"] - baseline["adversary_success_rate"]),
                "delta_coverage_adjusted_adversary_success_rate": float(
                    learned["coverage_adjusted_adversary_success_rate"] - baseline["coverage_adjusted_adversary_success_rate"]
                ),
                "delta_blue_recall_under_attack": (
                    None
                    if learned["blue_recall_under_attack"] is None or baseline["blue_recall_under_attack"] is None
                    else float(learned["blue_recall_under_attack"] - baseline["blue_recall_under_attack"])
                ),
                "delta_blue_precision_under_attack": (
                    None
                    if learned["blue_precision_under_attack"] is None or baseline["blue_precision_under_attack"] is None
                    else float(learned["blue_precision_under_attack"] - baseline["blue_precision_under_attack"])
                ),
                "delta_blue_flag_rate": float(learned["blue_flag_rate"] - baseline["blue_flag_rate"]),
                "delta_reward_mean": float(learned["reward_mean"] - baseline["reward_mean"]),
                "delta_coverage_adjusted_reward_mean": float(
                    learned["coverage_adjusted_reward_mean"] - baseline["coverage_adjusted_reward_mean"]
                ),
                "retrieval_coverage_rate": learned["retrieval_coverage_rate"],
                "alignment_joint_exact_match_accuracy": learned["alignment_joint_exact_match_accuracy"],
            }
        )
        comparison_index += 1

    static_breakdown_by_key: dict[tuple[Any, ...], dict[str, Any]] = {}
    learned_breakdowns: list[dict[str, Any]] = []
    for row in breakdown_rows:
        normalized = dict(row)
        key = (
            normalized["dataset_id"],
            normalized["blue_id"],
            int(normalized["transcript_length"]),
            normalized["breakdown_field"],
            normalized["breakdown_value"],
        )
        if normalized["adversary_kind"] == STATIC_ADVERSARY_KIND:
            static_breakdown_by_key[key] = normalized
        elif normalized["adversary_kind"] == LEARNED_ADVERSARY_KIND:
            learned_breakdowns.append(normalized)

    for learned in learned_breakdowns:
        baseline = static_breakdown_by_key.get(
            (
                learned["dataset_id"],
                learned["blue_id"],
                int(learned["transcript_length"]),
                learned["breakdown_field"],
                learned["breakdown_value"],
            )
        )
        if baseline is None:
            continue
        comparison_rows.append(
            {
                "comparison_id": f"comparison_{comparison_index:04d}",
                "scope": "breakdown",
                "dataset_id": learned["dataset_id"],
                "blue_id": learned["blue_id"],
                "red_id": learned["red_id"],
                "transcript_length": int(learned["transcript_length"]),
                "breakdown_field": learned["breakdown_field"],
                "breakdown_value": learned["breakdown_value"],
                "static_row_id": baseline["row_id"],
                "learned_row_id": learned["row_id"],
                "delta_adversary_success_rate": float(learned["adversary_success_rate"] - baseline["adversary_success_rate"]),
                "delta_coverage_adjusted_adversary_success_rate": float(
                    learned["coverage_adjusted_adversary_success_rate"] - baseline["coverage_adjusted_adversary_success_rate"]
                ),
                "delta_blue_recall_under_attack": (
                    None
                    if learned["blue_recall_under_attack"] is None or baseline["blue_recall_under_attack"] is None
                    else float(learned["blue_recall_under_attack"] - baseline["blue_recall_under_attack"])
                ),
                "delta_blue_precision_under_attack": (
                    None
                    if learned["blue_precision_under_attack"] is None or baseline["blue_precision_under_attack"] is None
                    else float(learned["blue_precision_under_attack"] - baseline["blue_precision_under_attack"])
                ),
                "delta_blue_flag_rate": float(learned["blue_flag_rate"] - baseline["blue_flag_rate"]),
                "delta_reward_mean": float(learned["reward_mean"] - baseline["reward_mean"]),
                "delta_coverage_adjusted_reward_mean": float(
                    learned["coverage_adjusted_reward_mean"] - baseline["coverage_adjusted_reward_mean"]
                ),
                "retrieval_coverage_rate": learned["retrieval_coverage_rate"],
            }
        )
        comparison_index += 1
    return comparison_rows


def render_red_blue_evaluation_summary(report: Mapping[str, Any]) -> str:
    datasets = list(report.get("datasets", []))
    blue_variants = list(report.get("blue_variants", []))
    red_variants = list(report.get("red_variants", []))
    transcript_lengths = list(report.get("transcript_lengths", []))
    summary_rows = [dict(row) for row in report.get("summary_rows", [])]
    comparison_rows = [dict(row) for row in report.get("comparison_rows", []) if row.get("scope") == "overall"]
    static_rows = [row for row in summary_rows if row.get("adversary_kind") == STATIC_ADVERSARY_KIND]
    learned_rows = [row for row in summary_rows if row.get("adversary_kind") == LEARNED_ADVERSARY_KIND]
    lines = [
        "Red-vs-blue evaluation",
        f"datasets: {len(datasets)}",
        f"blue_variants: {len(blue_variants)}",
        f"red_variants: {len(red_variants)}",
        f"transcript_lengths: {', '.join(str(value) for value in transcript_lengths) if transcript_lengths else 'none'}",
        f"static_evaluations: {len(static_rows)}",
        f"learned_retrieval_evaluations: {len(learned_rows)}",
    ]
    if comparison_rows:
        best = max(
            comparison_rows,
            key=lambda row: float(row.get("delta_coverage_adjusted_adversary_success_rate") or 0.0),
        )
        lines.extend(
            [
                (
                    "best_learned_delta: "
                    f"dataset={best['dataset_id']} blue={best['blue_id']} red={best['red_id']} "
                    f"n={best['transcript_length']} "
                    f"delta_success={best['delta_adversary_success_rate']:.4f} "
                    f"delta_coverage_adjusted_success={best['delta_coverage_adjusted_adversary_success_rate']:.4f}"
                ),
                (
                    "best_learned_alignment: "
                    f"joint_exact_match_accuracy={float(best.get('alignment_joint_exact_match_accuracy') or 0.0):.4f} "
                    f"retrieval_coverage_rate={float(best.get('retrieval_coverage_rate') or 0.0):.4f}"
                ),
            ]
        )
    elif red_variants:
        lines.append("learned_comparison: unavailable")
    else:
        lines.append("learned_comparison: no red variants were supplied or discovered")
    return "\n".join(lines) + "\n"


def evaluate_red_vs_blue(
    *,
    dataset_paths: list[Path],
    output_dir: Path,
    blue_model_dirs: list[Path] | None = None,
    red_model_paths: list[Path] | None = None,
    self_play_output_dirs: list[Path] | None = None,
    transcript_lengths: list[int] | None = None,
) -> dict[str, Any]:
    resolved_output_dir = output_dir.resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)

    action_space = load_red_action_space()
    transcript_budget = load_red_context_budget()
    reward_spec = load_red_reward_spec()
    resolved_transcript_lengths = _resolve_transcript_lengths(transcript_lengths, transcript_budget)

    datasets = build_dataset_catalog(dataset_paths, self_play_output_dirs=self_play_output_dirs)
    blue_variants = build_blue_variant_catalog(blue_model_dirs, self_play_output_dirs=self_play_output_dirs)
    red_variants = build_red_variant_catalog(red_model_paths, self_play_output_dirs=self_play_output_dirs)

    replay_cache: dict[tuple[str, int], tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    static_reward_cache: dict[tuple[str, int, str], tuple[list[dict[str, Any]], dict[str, Any]]] = {}
    evaluations: list[dict[str, Any]] = []
    summary_rows: list[dict[str, Any]] = []
    breakdown_rows: list[dict[str, Any]] = []
    evaluation_index = 1
    summary_index = 1
    breakdown_index = 1

    for dataset_entry in datasets:
        dataset_path = Path(str(dataset_entry["dataset_path"])).resolve()
        for transcript_length in resolved_transcript_lengths:
            replay_key = (str(dataset_path), int(transcript_length))
            if replay_key not in replay_cache:
                replay_cache[replay_key] = build_replay_examples_from_dataset(
                    dataset_path,
                    action_space=action_space,
                    transcript_budget=transcript_budget,
                    max_history_entries=transcript_length,
                )
            replay_examples, replay_summary = replay_cache[replay_key]
            for blue_entry in blue_variants:
                blue_path = Path(str(blue_entry["blue_path"])).resolve()
                static_key = (str(dataset_path), int(transcript_length), str(blue_path))
                if static_key not in static_reward_cache:
                    static_reward_cache[static_key] = score_and_reward_replay_examples(
                        examples=replay_examples,
                        blue_model_dir=blue_path,
                        action_space=action_space,
                        transcript_budget=transcript_budget,
                        reward_spec=reward_spec,
                    )
                static_rewarded_examples, static_reward_summary = static_reward_cache[static_key]
                static_attempts = _direct_attempts_from_rewarded_examples(static_rewarded_examples)

                static_evaluation_id = f"evaluation_{evaluation_index:04d}"
                evaluation_index += 1
                static_summary_row = _summary_row_from_attempts(
                    row_id=f"summary_{summary_index:04d}",
                    evaluation_id=static_evaluation_id,
                    dataset_entry=dataset_entry,
                    blue_entry=blue_entry,
                    red_entry=None,
                    adversary_kind=STATIC_ADVERSARY_KIND,
                    transcript_length=transcript_length,
                    attempts=static_attempts,
                )
                summary_index += 1
                static_breakdowns = build_breakdown_rows(
                    evaluation_id=static_evaluation_id,
                    dataset_entry=dataset_entry,
                    blue_entry=blue_entry,
                    red_entry=None,
                    adversary_kind=STATIC_ADVERSARY_KIND,
                    transcript_length=transcript_length,
                    attempts=static_attempts,
                    start_index=breakdown_index,
                )
                breakdown_index += len(static_breakdowns)
                summary_rows.append(static_summary_row)
                breakdown_rows.extend(static_breakdowns)
                evaluations.append(
                    {
                        "evaluation_id": static_evaluation_id,
                        "adversary_kind": STATIC_ADVERSARY_KIND,
                        "dataset_id": dataset_entry["dataset_id"],
                        "blue_id": blue_entry["blue_id"],
                        "red_id": None,
                        "transcript_length": int(transcript_length),
                        "replay_summary": _clone_json(replay_summary),
                        "reward_summary": _clone_json(static_reward_summary),
                        "summary_row_id": static_summary_row["row_id"],
                        "breakdown_row_ids": [row["row_id"] for row in static_breakdowns],
                    }
                )

                for red_entry in red_variants:
                    learned_evaluation_id = f"evaluation_{evaluation_index:04d}"
                    evaluation_index += 1
                    learned_attempts, retrieval_summary = evaluate_learned_red_policy_retrieval(
                        Path(str(red_entry["red_path"])).resolve(),
                        replay_examples,
                        static_rewarded_examples,
                    )
                    alignment = _as_mapping(retrieval_summary.get("alignment"))
                    learned_summary_row = _summary_row_from_attempts(
                        row_id=f"summary_{summary_index:04d}",
                        evaluation_id=learned_evaluation_id,
                        dataset_entry=dataset_entry,
                        blue_entry=blue_entry,
                        red_entry=red_entry,
                        adversary_kind=LEARNED_ADVERSARY_KIND,
                        transcript_length=transcript_length,
                        attempts=learned_attempts,
                        alignment=alignment,
                        retrieval_summary=retrieval_summary,
                    )
                    summary_index += 1
                    learned_breakdowns = build_breakdown_rows(
                        evaluation_id=learned_evaluation_id,
                        dataset_entry=dataset_entry,
                        blue_entry=blue_entry,
                        red_entry=red_entry,
                        adversary_kind=LEARNED_ADVERSARY_KIND,
                        transcript_length=transcript_length,
                        attempts=learned_attempts,
                        start_index=breakdown_index,
                    )
                    breakdown_index += len(learned_breakdowns)
                    summary_rows.append(learned_summary_row)
                    breakdown_rows.extend(learned_breakdowns)
                    evaluations.append(
                        {
                            "evaluation_id": learned_evaluation_id,
                            "adversary_kind": LEARNED_ADVERSARY_KIND,
                            "dataset_id": dataset_entry["dataset_id"],
                            "blue_id": blue_entry["blue_id"],
                            "red_id": red_entry["red_id"],
                            "transcript_length": int(transcript_length),
                            "replay_summary": _clone_json(replay_summary),
                            "static_reward_summary": _clone_json(static_reward_summary),
                            "retrieval_summary": _clone_json(retrieval_summary),
                            "alignment": _clone_json(alignment),
                            "summary_row_id": learned_summary_row["row_id"],
                            "breakdown_row_ids": [row["row_id"] for row in learned_breakdowns],
                        }
                    )

    comparison_rows = build_comparison_rows(summary_rows, breakdown_rows)
    report = {
        "schema_version": RED_BLUE_EVALUATION_SCHEMA_VERSION,
        "record_kind": "red_blue_evaluation",
        "mode": "offline_retrieval_based_adversary_vs_blue_v1",
        "datasets": datasets,
        "blue_variants": blue_variants,
        "red_variants": red_variants,
        "transcript_lengths": resolved_transcript_lengths,
        "evaluations": evaluations,
        "summary_rows": summary_rows,
        "breakdown_rows": breakdown_rows,
        "comparison_rows": comparison_rows,
    }
    summary_text = render_red_blue_evaluation_summary(report)
    _save_json(resolved_output_dir / "reports" / DEFAULT_REPORT_ARTIFACT_NAME, report)
    (resolved_output_dir / "reports" / DEFAULT_SUMMARY_ARTIFACT_NAME).write_text(summary_text, encoding="utf-8")
    return report


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Compare static replay adversaries and learned bounded red checkpoints against blue runtime "
            "bundles using offline retrieval-based evaluation."
        )
    )
    parser.add_argument("--dataset", action="append", default=[], help="Path to dataset.jsonl")
    parser.add_argument("--blue-model-dir", action="append", default=[], help="Path to a blue runtime bundle directory")
    parser.add_argument("--red-model-dir", action="append", default=[], help="Path to a red model directory or red_policy_model.json")
    parser.add_argument(
        "--self-play-output-dir",
        action="append",
        default=[],
        help="Self-play output directory to discover datasets and blue/red checkpoints from",
    )
    parser.add_argument(
        "--max-history-entries",
        action="append",
        type=int,
        default=[],
        help="Transcript history lengths to evaluate; defaults to the configured budget maximum",
    )
    parser.add_argument("--output-dir", type=Path, required=True, help="Directory for evaluation reports")
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = evaluate_red_vs_blue(
        dataset_paths=[Path(value).resolve() for value in args.dataset if _text(value)],
        output_dir=args.output_dir.resolve(),
        blue_model_dirs=[Path(value).resolve() for value in args.blue_model_dir if _text(value)],
        red_model_paths=[Path(value).resolve() for value in args.red_model_dir if _text(value)],
        self_play_output_dirs=[Path(value).resolve() for value in args.self_play_output_dir if _text(value)],
        transcript_lengths=list(args.max_history_entries or []),
    )
    print(
        json.dumps(
            {
                "schema_version": report["schema_version"],
                "record_kind": report["record_kind"],
                "evaluation_count": len(report["evaluations"]),
                "summary_row_count": len(report["summary_rows"]),
                "comparison_row_count": len(report["comparison_rows"]),
                "report_path": str((args.output_dir.resolve() / "reports" / DEFAULT_REPORT_ARTIFACT_NAME).resolve()),
                "summary_path": str((args.output_dir.resolve() / "reports" / DEFAULT_SUMMARY_ARTIFACT_NAME).resolve()),
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
