#!/usr/bin/env python3
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

from main import DEFAULT_OUTPUT_DIR, model_uses_canonical_features, run_generate, run_training
from runtime import load_runtime_bundle
from tools.shared.schema import adapt_legacy_fprime_transaction
from tools.train.checkpointing import (
    SELF_PLAY_REPORT_SCHEMA_VERSION,
    checkpoint_id_for_round,
    create_directory_checkpoint,
    initialize_self_play_state,
    load_self_play_state,
    round_dir,
    round_slug,
    update_self_play_state,
    write_self_play_report,
    write_self_play_state,
)
from tools.train.poster_default import (
    canonical_row_to_training_row,
    load_canonical_training_records,
    transaction_training_record_path,
)
from tools.train.red_policy_model import (
    RED_POLICY_EXAMPLES_ARTIFACT_NAME,
    RED_POLICY_MODEL_ARTIFACT_NAME,
    RED_POLICY_REPORT_ARTIFACT_NAME,
    DEFAULT_RED_POLICY_TRAINING_CONFIG,
    LoadedRedPolicyModel,
    RedPolicyTrainingConfig,
    build_red_policy_warmstart_examples,
    export_red_policy_model_payload,
    fit_red_policy_model,
    infer_red_identity_bucket,
    load_red_action_space,
    summarize_red_policy_examples,
    timing_bucket_from_gap_seconds,
    validate_red_policy_action,
    write_red_policy_examples_jsonl,
)
from tools.train.red_reward import compute_red_reward, load_red_reward_spec
from tools.train.red_transcript import build_red_transcript, load_red_context_budget

SELF_PLAY_ROUND_SCHEMA_VERSION = "self_play_round.v1"
SELF_PLAY_REPLAY_EXAMPLE_SCHEMA_VERSION = "red_self_play_example.v1"
SELF_PLAY_REWARDED_EXAMPLE_SCHEMA_VERSION = "red_self_play_rewarded_example.v1"
SELF_PLAY_MODE = "alternating_frozen_opponent_offline_auto_research_v1"

DEFAULT_SELF_PLAY_ROUNDS = 1
DEFAULT_SELF_PLAY_ROWS = 240
DEFAULT_SELF_PLAY_NOMINAL_RATIO = 0.55
DEFAULT_SELF_PLAY_PROTOCOL_MODE = "mixed"
DEFAULT_SELF_PLAY_MIXED_FPRIME_RATIO = 0.5
DEFAULT_RED_REPLAY_BUFFER_LIMIT = 512
DEFAULT_RED_WARMSTART_ROWS_PER_PROTOCOL = 96
DEFAULT_INITIAL_BLUE_MODEL_DIR = REPO_ROOT / "deployments" / "DetectorRB3" / "config"


class SelfPlayHarnessError(ValueError):
    """Raised when the self-play harness cannot proceed safely."""


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


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SelfPlayHarnessError(f"{path} must contain a JSON object")
    return payload


def _read_jsonl(path: Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with path.open(encoding="utf-8") as handle:
        for line_number, line in enumerate(handle, start=1):
            text = line.strip()
            if not text:
                continue
            payload = json.loads(text)
            if not isinstance(payload, dict):
                raise SelfPlayHarnessError(f"{path}:{line_number} must contain JSON objects")
            rows.append(payload)
    return rows


def _write_jsonl(path: Path, rows: Iterable[Mapping[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(dict(row), sort_keys=True))
            handle.write("\n")


def _round_seed(seed: int, round_index: int, offset: int) -> int:
    return int(seed) * 1009 + int(round_index) * 97 + int(offset)


def _canonical_row_key(canonical_row: Mapping[str, Any]) -> tuple[str | None, str | None, str | None]:
    audit = _as_mapping(canonical_row.get("audit_context"))
    return (
        _text(audit.get("session_id")),
        _text(audit.get("transaction_id")),
        _text(audit.get("send_id")),
    )


def _raw_transaction_key(raw_transaction: Mapping[str, Any]) -> tuple[str | None, str | None, str | None]:
    correlation = _as_mapping(raw_transaction.get("correlation"))
    return (
        _text(correlation.get("session_id")),
        _text(correlation.get("transaction_id")),
        _text(correlation.get("send_id")),
    )


def _unique_lookup(
    entries: Iterable[tuple[Any, dict[str, Any]]],
) -> dict[Any, dict[str, Any]]:
    lookup: dict[Any, dict[str, Any]] = {}
    ambiguous_keys: set[Any] = set()
    for key, record in entries:
        if key is None:
            continue
        if isinstance(key, tuple) and any(part is None for part in key):
            continue
        if key in ambiguous_keys:
            continue
        if key in lookup:
            ambiguous_keys.add(key)
            lookup.pop(key, None)
            continue
        lookup[key] = record
    return lookup


def _canonical_raw_fallback_key(canonical_row: Mapping[str, Any]) -> tuple[str | None, str | None]:
    audit = _as_mapping(canonical_row.get("audit_context"))
    return (
        _text(audit.get("transaction_id")),
        _text(audit.get("send_id")),
    )


def _raw_transaction_fallback_key(raw_transaction: Mapping[str, Any]) -> tuple[str | None, str | None]:
    correlation = _as_mapping(raw_transaction.get("correlation"))
    return (
        _text(correlation.get("transaction_id")),
        _text(correlation.get("send_id")),
    )


def _canonical_raw_transaction_id(canonical_row: Mapping[str, Any]) -> str | None:
    return _text(_as_mapping(canonical_row.get("audit_context")).get("transaction_id"))


def _raw_transaction_id(raw_transaction: Mapping[str, Any]) -> str | None:
    return _text(_as_mapping(raw_transaction.get("correlation")).get("transaction_id"))


def _load_generation_summary(dataset_path: Path) -> dict[str, Any] | None:
    report_path = dataset_path.resolve().parents[1] / "reports" / "generation_summary.json"
    if not report_path.exists():
        return None
    return _load_json(report_path)


def _infer_protocol_mode_from_examples(examples: list[Mapping[str, Any]]) -> str:
    protocol_families = sorted({
        _compact_category(_as_mapping(example.get("metadata")).get("protocol_family"), "unknown")
        for example in examples
    })
    if protocol_families == ["fprime"]:
        return "fprime"
    if protocol_families == ["mavlink"]:
        return "mavlink"
    if set(protocol_families) == {"fprime", "mavlink"}:
        return "mixed"
    return DEFAULT_SELF_PLAY_PROTOCOL_MODE


def _load_round_canonical_and_raw(
    dataset_path: Path,
) -> tuple[list[dict[str, Any]], dict[str, dict[Any, dict[str, Any]]], dict[str, Any]]:
    canonical_rows, provenance = load_canonical_training_records(dataset_path)
    raw_transaction_path = dataset_path.resolve().with_name("raw_transactions.jsonl")
    raw_transactions: list[dict[str, Any]]
    if raw_transaction_path.exists():
        raw_transactions = _read_jsonl(raw_transaction_path)
        raw_source = "raw_transactions"
    else:
        legacy_transaction_path = transaction_training_record_path(dataset_path)
        if not legacy_transaction_path.exists():
            raise SelfPlayHarnessError(
                "Self-play replay needs raw_transactions.jsonl or transactions.jsonl next to the dataset. "
                f"dataset={dataset_path}"
            )
        raw_transactions = [
            adapt_legacy_fprime_transaction(record)
            for record in _read_jsonl(legacy_transaction_path)
        ]
        raw_source = "adapted_legacy_transactions"
    raw_transaction_lookups = {
        "exact": _unique_lookup((_raw_transaction_key(record), record) for record in raw_transactions),
        "transaction_send": _unique_lookup((_raw_transaction_fallback_key(record), record) for record in raw_transactions),
        "transaction_only": _unique_lookup((_raw_transaction_id(record), record) for record in raw_transactions),
    }
    return canonical_rows, raw_transaction_lookups, {
        **dict(provenance),
        "raw_transaction_path": str(raw_transaction_path if raw_source == "raw_transactions" else legacy_transaction_path.resolve()),
        "raw_transaction_source": raw_source,
    }


def _match_raw_transaction(
    canonical_row: Mapping[str, Any],
    raw_transaction_lookups: Mapping[str, Mapping[Any, dict[str, Any]]],
) -> tuple[dict[str, Any] | None, str | None]:
    exact = raw_transaction_lookups.get("exact", {})
    record = exact.get(_canonical_row_key(canonical_row))
    if record is not None:
        return record, "exact"
    transaction_send = raw_transaction_lookups.get("transaction_send", {})
    record = transaction_send.get(_canonical_raw_fallback_key(canonical_row))
    if record is not None:
        return record, "transaction_send"
    transaction_only = raw_transaction_lookups.get("transaction_only", {})
    record = transaction_only.get(_canonical_raw_transaction_id(canonical_row))
    if record is not None:
        return record, "transaction_only"
    return None, None


def _aligned_raw_transaction_for_replay(
    canonical_row: Mapping[str, Any],
    raw_transaction: Mapping[str, Any],
) -> dict[str, Any]:
    aligned = _clone_json(raw_transaction)
    correlation = _as_mapping(aligned.get("correlation"))
    audit = _as_mapping(canonical_row.get("audit_context"))
    for field_name in ("session_id", "transaction_id", "send_id"):
        canonical_value = _text(audit.get(field_name))
        if canonical_value is not None:
            correlation[field_name] = canonical_value
    aligned["correlation"] = correlation
    return aligned


def _identity_bucket_for_canonical_row(
    canonical_row: Mapping[str, Any],
    *,
    action_space: Mapping[str, Any],
) -> str | None:
    audit = _as_mapping(canonical_row.get("audit_context"))
    actor_context = _as_mapping(canonical_row.get("actor_context"))
    protocol_family = _compact_category(canonical_row.get("protocol_family"), "unknown")
    return infer_red_identity_bucket(
        protocol_family=protocol_family,
        source_service=_text(audit.get("actor_id")),
        actor_role=_text(actor_context.get("role")),
        action_space=action_space,
    )


def _current_context_from_canonical_row(canonical_row: Mapping[str, Any]) -> dict[str, str]:
    mission_context = _as_mapping(canonical_row.get("mission_context"))
    return {
        "protocol_family": _compact_category(canonical_row.get("protocol_family"), "unknown"),
        "platform_family": _compact_category(canonical_row.get("platform_family"), "unknown"),
        "mission_phase": _compact_category(mission_context.get("mission_phase"), "unknown"),
        "window_class": _compact_category(mission_context.get("window_class"), "unspecified"),
    }


def _action_from_canonical_row(
    canonical_row: Mapping[str, Any],
    raw_transaction: Mapping[str, Any],
    previous_self_raw_transaction: Mapping[str, Any] | None,
    *,
    action_space: Mapping[str, Any],
) -> dict[str, str] | None:
    command_semantics = _as_mapping(canonical_row.get("command_semantics"))
    command_family = _compact_category(command_semantics.get("canonical_command_family"), "unknown")
    identity_bucket = _identity_bucket_for_canonical_row(canonical_row, action_space=action_space)
    if identity_bucket is None:
        return None
    current_ms = _number(_as_mapping(raw_transaction.get("timing")).get("submitted_at_ms"))
    previous_ms = None
    if previous_self_raw_transaction is not None:
        previous_ms = _number(_as_mapping(previous_self_raw_transaction.get("timing")).get("submitted_at_ms"))
    gap_seconds = None if current_ms is None or previous_ms is None else max(0.0, current_ms - previous_ms) / 1000.0
    return validate_red_policy_action(
        {
            "command_family": command_family,
            "timing_bucket": timing_bucket_from_gap_seconds(gap_seconds, action_space=action_space),
            "identity_bucket": identity_bucket,
        },
        action_space=action_space,
    )


def build_replay_examples_from_dataset(
    dataset_path: Path,
    *,
    action_space: Mapping[str, Any],
    transcript_budget: Mapping[str, Any],
    max_history_entries: int | None = None,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    canonical_rows, raw_transaction_lookups, provenance = _load_round_canonical_and_raw(dataset_path)
    actor_histories: dict[tuple[str, int, str], list[dict[str, Any]]] = {}
    skipped = Counter()
    match_modes = Counter()
    examples: list[dict[str, Any]] = []
    for index, canonical_row in enumerate(canonical_rows):
        audit = _as_mapping(canonical_row.get("audit_context"))
        current_context = _current_context_from_canonical_row(canonical_row)
        protocol_family = current_context["protocol_family"]
        actor_id = _text(audit.get("actor_id"))
        if actor_id is None:
            skipped["missing_actor_id"] += 1
            continue
        raw_transaction, match_mode = _match_raw_transaction(canonical_row, raw_transaction_lookups)
        if raw_transaction is None:
            skipped["missing_raw_transaction"] += 1
            continue
        match_modes[str(match_mode)] += 1
        aligned_raw_transaction = _aligned_raw_transaction_for_replay(canonical_row, raw_transaction)
        episode_id = int(_number(canonical_row.get("episode_id")) or 0)
        history_key = (protocol_family, episode_id, actor_id)
        previous_items = list(actor_histories.get(history_key, []))
        previous_self_raw = _as_mapping(previous_items[-1].get("raw_transaction")) if previous_items else None
        action = _action_from_canonical_row(
            canonical_row,
            aligned_raw_transaction,
            previous_self_raw if previous_items else None,
            action_space=action_space,
        )
        if action is None:
            skipped["unsupported_identity_bucket"] += 1
            continue
        transcript = build_red_transcript(
            previous_items,
            actor_id=actor_id,
            max_history_entries=max_history_entries,
            budget=transcript_budget,
        )
        metadata = {
            "protocol_family": protocol_family,
            "platform_family": current_context["platform_family"],
            "episode_id": episode_id,
            "actor_id": actor_id,
            "target_id": _text(audit.get("target_id")) or "unknown",
            "group_key": f"{protocol_family}:{episode_id}:{actor_id}",
            "submitted_at_ms": _number(_as_mapping(aligned_raw_transaction.get("timing")).get("submitted_at_ms")),
            "raw_command_name": _text(audit.get("raw_command_name")) or "unknown",
            "attack_family": _text(audit.get("attack_family")) or "none",
            "sequence_index": index,
            "source_dataset_path": str(dataset_path.resolve()),
            "raw_transaction_match_mode": str(match_mode),
        }
        example = {
            "schema_version": SELF_PLAY_REPLAY_EXAMPLE_SCHEMA_VERSION,
            "record_kind": "red_self_play_example",
            "transcript": transcript,
            "current_context": current_context,
            "action": action,
            "metadata": metadata,
            "canonical_row": _clone_json(canonical_row),
            "raw_transaction": aligned_raw_transaction,
        }
        examples.append(example)
        actor_histories.setdefault(history_key, []).append(
            {
                "canonical_row": _clone_json(canonical_row),
                "raw_transaction": aligned_raw_transaction,
            }
        )
    summary = summarize_red_policy_examples(examples) if examples else {"example_count": 0}
    summary.update(
        {
            "dataset_path": str(dataset_path.resolve()),
            "record_provenance": provenance,
            "skipped_counts": dict(skipped),
            "raw_transaction_match_modes": dict(match_modes),
            "raw_transaction_match_rate": float(sum(match_modes.values()) / max(1, len(canonical_rows))),
        }
    )
    return examples, summary


def evaluate_red_policy_alignment(
    model_path: Path,
    examples: list[Mapping[str, Any]],
) -> dict[str, Any]:
    if not examples:
        return {
            "evaluated_examples": 0,
            "joint_exact_match_accuracy": 0.0,
            "head_accuracy": {"command_family": 0.0, "timing_bucket": 0.0, "identity_bucket": 0.0},
            "predicted_action_counts": {},
        }
    model = LoadedRedPolicyModel.from_path(model_path)
    head_matches = Counter()
    exact_matches = 0
    predicted_action_counts = Counter()
    for example in examples:
        predicted = model.predict_action(
            _as_mapping(example.get("transcript")),
            _as_mapping(example.get("current_context")),
        )
        action = _as_mapping(example.get("action"))
        predicted_action = _as_mapping(predicted.get("action"))
        predicted_action_counts[
            "|".join(
                [
                    str(predicted_action.get("command_family", "unknown")),
                    str(predicted_action.get("timing_bucket", "unknown")),
                    str(predicted_action.get("identity_bucket", "unknown")),
                ]
            )
        ] += 1
        per_example_exact = True
        for head_name in ("command_family", "timing_bucket", "identity_bucket"):
            if predicted_action.get(head_name) == action.get(head_name):
                head_matches[head_name] += 1
            else:
                per_example_exact = False
        if per_example_exact:
            exact_matches += 1
    total = len(examples)
    return {
        "evaluated_examples": total,
        "joint_exact_match_accuracy": float(exact_matches / total),
        "head_accuracy": {
            head_name: float(head_matches[head_name] / total)
            for head_name in ("command_family", "timing_bucket", "identity_bucket")
        },
        "predicted_action_counts": dict(predicted_action_counts),
    }


def _reward_feedback_from_example(example: Mapping[str, Any]) -> dict[str, Any]:
    raw_transaction = _as_mapping(example.get("raw_transaction"))
    canonical_row = _as_mapping(example.get("canonical_row"))
    audit = _as_mapping(canonical_row.get("audit_context"))
    outcome = _as_mapping(raw_transaction.get("outcome"))
    return {
        "label": audit.get("label"),
        "label_name": audit.get("label_name"),
        "accepted": outcome.get("accepted"),
        "executed_successfully": outcome.get("executed_successfully"),
        "timed_out": outcome.get("timed_out"),
        "outcome": outcome,
    }


def score_and_reward_replay_examples(
    *,
    examples: list[Mapping[str, Any]],
    blue_model_dir: Path,
    action_space: Mapping[str, Any],
    transcript_budget: Mapping[str, Any],
    reward_spec: Mapping[str, Any],
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    runtime_bundle = load_runtime_bundle(blue_model_dir.resolve())
    rewarded_examples: list[dict[str, Any]] = []
    reward_case_counts = Counter()
    protocol_reward_totals = Counter()
    protocol_counts = Counter()
    sandbox_violation_counts = Counter()
    rewards: list[float] = []
    for example in examples:
        canonical_row = _as_mapping(example.get("canonical_row"))
        metadata = _as_mapping(example.get("metadata"))
        scoring_row = dict(canonical_row)
        if getattr(runtime_bundle, "model", None) is not None and model_uses_canonical_features(
            list(getattr(runtime_bundle, "feature_names", []))
        ):
            scoring_row = canonical_row_to_training_row(canonical_row)
        scored = runtime_bundle.score_row(scoring_row)
        blue_feedback = {
            "unsafe_risk": scored.get("unsafe_risk"),
            "panomaly": scored.get("panomaly"),
            "pcyber": scored.get("pcyber"),
            "predicted_class": scored.get("predicted_class"),
            "detector_reason": scored.get("detector_reason"),
        }
        reward_result = compute_red_reward(
            action=_as_mapping(example.get("action")),
            current_context=_as_mapping(example.get("current_context")),
            transcript=_as_mapping(example.get("transcript")),
            environment_feedback=_reward_feedback_from_example(example),
            blue_feedback=blue_feedback,
            source_service=_text(metadata.get("actor_id")),
            action_space=action_space,
            transcript_budget=transcript_budget,
            reward_spec=reward_spec,
        )
        reward_value = float(reward_result["reward"])
        rewards.append(reward_value)
        protocol_family = _compact_category(metadata.get("protocol_family"), "unknown")
        protocol_reward_totals[protocol_family] += reward_value
        protocol_counts[protocol_family] += 1
        reward_case_counts[str(reward_result["reward_case"])] += 1
        for reason in _as_mapping(reward_result.get("sandbox")).get("violation_reasons", []):
            sandbox_violation_counts[str(reason)] += 1
        rewarded_examples.append(
            {
                "schema_version": SELF_PLAY_REWARDED_EXAMPLE_SCHEMA_VERSION,
                "record_kind": "red_self_play_rewarded_example",
                "transcript": _clone_json(example.get("transcript")),
                "current_context": _clone_json(example.get("current_context")),
                "action": _clone_json(example.get("action")),
                "metadata": _clone_json(metadata),
                "blue_feedback": blue_feedback,
                "reward_result": reward_result,
            }
        )
    positive_count = sum(1 for value in rewards if value > 0.0)
    negative_count = sum(1 for value in rewards if value < 0.0)
    zero_count = len(rewards) - positive_count - negative_count
    return rewarded_examples, {
        "example_count": len(rewarded_examples),
        "reward_case_counts": dict(reward_case_counts),
        "sandbox_violation_counts": dict(sandbox_violation_counts),
        "positive_count": positive_count,
        "negative_count": negative_count,
        "zero_count": zero_count,
        "reward_mean": float(sum(rewards) / max(1, len(rewards))),
        "reward_min": float(min(rewards)) if rewards else 0.0,
        "reward_max": float(max(rewards)) if rewards else 0.0,
        "reward_by_protocol": {
            protocol_family: float(protocol_reward_totals[protocol_family] / max(1, protocol_counts[protocol_family]))
            for protocol_family in sorted(protocol_counts)
        },
    }


def _reward_repeat_count(reward_value: float) -> int:
    if reward_value >= 0.75:
        return 5
    if reward_value >= 0.40:
        return 4
    if reward_value >= 0.10:
        return 3
    if reward_value >= 0.0:
        return 2
    return 0


def _clone_training_example(record: Mapping[str, Any], *, source_kind: str, round_index: int) -> dict[str, Any]:
    metadata = _clone_json(_as_mapping(record.get("metadata")))
    metadata["source_kind"] = source_kind
    metadata["replay_round"] = int(round_index)
    return {
        "transcript": _clone_json(record.get("transcript")),
        "current_context": _clone_json(record.get("current_context")),
        "action": _clone_json(record.get("action")),
        "metadata": metadata,
        "schema_version": record.get("schema_version", "red_policy_example.v1"),
        "record_kind": "red_policy_example",
    }


def assemble_red_training_examples(
    *,
    warmstart_examples: list[Mapping[str, Any]],
    buffered_rewarded_examples: list[Mapping[str, Any]],
    buffer_limit: int,
    round_index: int,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    selected_buffer = list(buffered_rewarded_examples[-buffer_limit:]) if buffer_limit > 0 else []
    replay_examples: list[dict[str, Any]] = []
    reward_histogram = Counter()
    skipped_nonpositive = 0
    for record in selected_buffer:
        reward_value = float(_as_mapping(record.get("reward_result")).get("reward") or 0.0)
        repeat_count = _reward_repeat_count(reward_value)
        reward_histogram[str(repeat_count)] += 1
        if repeat_count <= 0:
            skipped_nonpositive += 1
            continue
        training_example = _clone_training_example(record, source_kind="reward_replay", round_index=round_index)
        for _ in range(repeat_count):
            replay_examples.append(_clone_json(training_example))
    combined_examples = [
        _clone_training_example(example, source_kind="warmstart_anchor", round_index=0)
        for example in warmstart_examples
    ]
    combined_examples.extend(replay_examples)
    return combined_examples, {
        "warmstart_anchor_examples": len(warmstart_examples),
        "buffered_reward_records": len(selected_buffer),
        "replay_examples_after_weighting": len(replay_examples),
        "skipped_nonpositive_records": skipped_nonpositive,
        "reward_repeat_histogram": dict(reward_histogram),
        "combined_training_examples": len(combined_examples),
    }


def _extract_blue_metric_summary(report: Mapping[str, Any]) -> dict[str, Any]:
    metrics = _as_mapping(report.get("metrics"))
    model_only = _as_mapping(metrics.get("model_only"))
    neural_net = _as_mapping(model_only.get("neural_net"))
    multiclass = _as_mapping(neural_net.get("multiclass_metrics"))
    cyber = _as_mapping(neural_net.get("cyber_binary_metrics"))
    anomaly = _as_mapping(neural_net.get("anomaly_binary_metrics"))
    return {
        "macro_f1": float(multiclass.get("macro_f1") or 0.0),
        "accuracy": float(multiclass.get("accuracy") or 0.0),
        "cyber_f1": float(cyber.get("f1") or 0.0),
        "anomaly_f1": float(anomaly.get("f1") or 0.0),
        "deployment_ready": bool(report.get("deployment_ready")),
        "deployment_blocked_reason": report.get("deployment_blocked_reason"),
    }


def _load_report_if_exists(path: Path) -> dict[str, Any] | None:
    if not path.exists():
        return None
    return _load_json(path)


def _resolve_blue_runtime_bundle_dir(
    blue_update_dir: Path,
    report: Mapping[str, Any] | None,
) -> tuple[Path | None, str | None]:
    deployed_model_dir = blue_update_dir / "models"
    if deployed_model_dir.exists():
        return deployed_model_dir, "deployable_models"
    analysis_runtime_bundle = _as_mapping(_as_mapping(report).get("analysis_runtime_bundle"))
    analysis_artifact_dir = _text(analysis_runtime_bundle.get("artifact_dir"))
    if analysis_artifact_dir is not None:
        candidate_dir = Path(analysis_artifact_dir)
        if candidate_dir.exists():
            return candidate_dir, "analysis_runtime_bundle"
    research_model_dir = blue_update_dir / "research_models"
    if research_model_dir.exists():
        return research_model_dir, "analysis_runtime_bundle"
    return None, None


def run_blue_update_round(
    *,
    dataset_path: Path,
    output_dir: Path,
    seed: int,
    make_plots: bool,
) -> tuple[dict[str, Any] | None, str | None]:
    try:
        report = run_training(
            dataset_path=dataset_path,
            output_dir=output_dir,
            seed=seed,
            make_plots=make_plots,
            blue_feature_policy_name=None,
        )
        return report, None
    except SystemExit as exc:
        report = _load_report_if_exists(output_dir / "reports" / "metrics.json")
        if report is not None:
            return report, str(exc)
        return None, str(exc)


def train_red_candidate(
    *,
    output_dir: Path,
    training_examples: list[Mapping[str, Any]],
    protocol_mode: str,
    rows_per_protocol: int,
    seed: int,
    action_space: Mapping[str, Any],
    transcript_budget: Mapping[str, Any],
    buffer_summary: Mapping[str, Any],
) -> dict[str, Any]:
    trained = fit_red_policy_model(
        list(training_examples),
        seed=seed,
        action_space=action_space,
        transcript_budget=transcript_budget,
        training_config=RedPolicyTrainingConfig(
            protocol_mode=protocol_mode,
            warm_start_rows_per_protocol=rows_per_protocol,
            holdout_ratio=DEFAULT_RED_POLICY_TRAINING_CONFIG.holdout_ratio,
            max_iter=DEFAULT_RED_POLICY_TRAINING_CONFIG.max_iter,
            batch_size=DEFAULT_RED_POLICY_TRAINING_CONFIG.batch_size,
            learning_rate_init=DEFAULT_RED_POLICY_TRAINING_CONFIG.learning_rate_init,
            alpha=DEFAULT_RED_POLICY_TRAINING_CONFIG.alpha,
        ),
    )
    payload = export_red_policy_model_payload(trained, action_space=action_space)
    report = {
        "schema_version": SELF_PLAY_ROUND_SCHEMA_VERSION,
        "record_kind": "red_self_play_update_report",
        "protocol_mode": protocol_mode,
        "seed": int(seed),
        "buffer_summary": dict(buffer_summary),
        "warm_start": dict(trained.get("warm_start", {})),
        "architecture": dict(trained.get("architecture", {})),
        "training_config": dict(trained.get("training_config", {})),
        "evaluation": dict(trained.get("evaluation", {})),
        "artifacts": {
            "model_path": str((output_dir / RED_POLICY_MODEL_ARTIFACT_NAME).resolve()),
            "report_path": str((output_dir / RED_POLICY_REPORT_ARTIFACT_NAME).resolve()),
            "examples_path": str((output_dir / RED_POLICY_EXAMPLES_ARTIFACT_NAME).resolve()),
        },
    }
    output_dir.mkdir(parents=True, exist_ok=True)
    _save_json(output_dir / RED_POLICY_MODEL_ARTIFACT_NAME, payload)
    _save_json(output_dir / RED_POLICY_REPORT_ARTIFACT_NAME, report)
    write_red_policy_examples_jsonl(output_dir / RED_POLICY_EXAMPLES_ARTIFACT_NAME, training_examples)
    return {
        "trained": trained,
        "payload": payload,
        "report": report,
    }


def _dataset_sources(values: list[str] | None) -> list[Path]:
    return [Path(value).resolve() for value in (values or []) if _text(value)]


def _resolve_round_dataset(
    *,
    round_index: int,
    output_dir: Path,
    dataset_sources: list[Path],
    rows: int,
    nominal_ratio: float,
    seed: int,
    protocol_mode: str,
    mixed_fprime_ratio: float,
) -> tuple[Path, dict[str, Any]]:
    if dataset_sources:
        selected = dataset_sources[(round_index - 1) % len(dataset_sources)]
        if not selected.exists():
            raise SelfPlayHarnessError(f"Missing replay dataset: {selected}")
        return selected, {
            "source_mode": "replay_dataset",
            "dataset_path": str(selected),
        }
    rollout_output_dir = output_dir / "rollout"
    dataset_path = run_generate(
        rollout_output_dir,
        rows,
        nominal_ratio,
        seed,
        protocol_mode=protocol_mode,
        mixed_fprime_ratio=mixed_fprime_ratio,
    )
    return dataset_path, {
        "source_mode": "fresh_rollout",
        "dataset_path": str(dataset_path.resolve()),
        "rollout_output_dir": str(rollout_output_dir.resolve()),
    }


def _collect_reward_buffer(state: Mapping[str, Any], current_round_rewarded_examples: list[Mapping[str, Any]]) -> list[dict[str, Any]]:
    buffered: list[dict[str, Any]] = []
    for round_summary in state.get("rounds", []):
        artifacts = _as_mapping(_as_mapping(round_summary).get("artifacts"))
        reward_examples_path = _text(artifacts.get("reward_examples_path"))
        if reward_examples_path is None:
            continue
        path = Path(reward_examples_path)
        if path.exists():
            buffered.extend(_read_jsonl(path))
    buffered.extend(_clone_json(list(current_round_rewarded_examples)))
    return buffered


def _resume_state_or_initialize(
    *,
    output_dir: Path,
    requested_config: Mapping[str, Any],
    initial_blue_model_dir: Path,
    initial_red_dir: Path | None,
    protocol_mode: str,
    red_warmstart_rows_per_protocol: int,
    seed: int,
    max_history_entries: int | None,
    action_space: Mapping[str, Any],
    transcript_budget: Mapping[str, Any],
) -> dict[str, Any]:
    existing = load_self_play_state(output_dir)
    if existing is not None:
        return existing

    blue_checkpoint = create_directory_checkpoint(
        output_dir=output_dir,
        side="blue",
        checkpoint_id=checkpoint_id_for_round(0),
        source_dir=initial_blue_model_dir.resolve(),
        metadata={
            "round_index": 0,
            "source": "initial_blue_model_dir",
            "model_dir": str(initial_blue_model_dir.resolve()),
        },
    )

    if initial_red_dir is None:
        bootstrap_dir = output_dir / "bootstrap" / "red_initial"
        bootstrap_dir.mkdir(parents=True, exist_ok=True)
        warmstart_examples = build_red_policy_warmstart_examples(
            protocol_mode=protocol_mode,
            rows_per_protocol=red_warmstart_rows_per_protocol,
            seed=seed,
            max_history_entries=max_history_entries,
            action_space=action_space,
            transcript_budget=transcript_budget,
        )
        train_red_candidate(
            output_dir=bootstrap_dir,
            training_examples=warmstart_examples,
            protocol_mode=protocol_mode,
            rows_per_protocol=red_warmstart_rows_per_protocol,
            seed=seed,
            action_space=action_space,
            transcript_budget=transcript_budget,
            buffer_summary={
                "warmstart_anchor_examples": len(warmstart_examples),
                "buffered_reward_records": 0,
                "replay_examples_after_weighting": 0,
                "skipped_nonpositive_records": 0,
                "reward_repeat_histogram": {},
                "combined_training_examples": len(warmstart_examples),
            },
        )
        initial_red_dir = bootstrap_dir
        red_source = "warmstart_bootstrap"
    else:
        red_source = "initial_red_dir"

    red_checkpoint = create_directory_checkpoint(
        output_dir=output_dir,
        side="red",
        checkpoint_id=checkpoint_id_for_round(0),
        source_dir=initial_red_dir.resolve(),
        metadata={
            "round_index": 0,
            "source": red_source,
            "artifact_dir": str(initial_red_dir.resolve()),
        },
    )
    return initialize_self_play_state(
        output_dir=output_dir,
        config=requested_config,
        blue_checkpoint=blue_checkpoint,
        red_checkpoint=red_checkpoint,
    )


def run_self_play(
    *,
    output_dir: Path,
    rounds: int,
    seed: int,
    rows: int,
    nominal_ratio: float,
    protocol_mode: str,
    mixed_fprime_ratio: float,
    dataset_sources: list[Path] | None = None,
    initial_blue_model_dir: Path = DEFAULT_INITIAL_BLUE_MODEL_DIR,
    initial_red_dir: Path | None = None,
    red_warmstart_rows_per_protocol: int = DEFAULT_RED_WARMSTART_ROWS_PER_PROTOCOL,
    red_replay_buffer_limit: int = DEFAULT_RED_REPLAY_BUFFER_LIMIT,
    max_history_entries: int | None = None,
    make_plots: bool = False,
) -> dict[str, Any]:
    resolved_output_dir = output_dir.resolve()
    resolved_output_dir.mkdir(parents=True, exist_ok=True)
    action_space = load_red_action_space()
    transcript_budget = load_red_context_budget()
    reward_spec = load_red_reward_spec()
    requested_config = {
        "mode": SELF_PLAY_MODE,
        "rounds": int(rounds),
        "seed": int(seed),
        "rows": int(rows),
        "nominal_ratio": float(nominal_ratio),
        "protocol_mode": protocol_mode,
        "mixed_fprime_ratio": float(mixed_fprime_ratio),
        "dataset_sources": [str(path.resolve()) for path in list(dataset_sources or [])],
        "initial_blue_model_dir": str(initial_blue_model_dir.resolve()),
        "initial_red_dir": None if initial_red_dir is None else str(initial_red_dir.resolve()),
        "red_warmstart_rows_per_protocol": int(red_warmstart_rows_per_protocol),
        "red_replay_buffer_limit": int(red_replay_buffer_limit),
        "max_history_entries": max_history_entries,
        "make_plots": bool(make_plots),
    }
    state = _resume_state_or_initialize(
        output_dir=resolved_output_dir,
        requested_config=requested_config,
        initial_blue_model_dir=initial_blue_model_dir,
        initial_red_dir=initial_red_dir,
        protocol_mode=protocol_mode,
        red_warmstart_rows_per_protocol=red_warmstart_rows_per_protocol,
        seed=seed,
        max_history_entries=max_history_entries,
        action_space=action_space,
        transcript_budget=transcript_budget,
    )
    if int(state.get("rounds_completed", 0)) >= int(rounds):
        report_payload = {
            "status": "completed",
            "mode": SELF_PLAY_MODE,
            "config": dict(state.get("config", {})),
            "rounds_completed": int(state.get("rounds_completed", 0)),
            "latest_blue_checkpoint": dict(state.get("latest_blue_checkpoint", {})),
            "latest_red_checkpoint": dict(state.get("latest_red_checkpoint", {})),
            "rounds": list(state.get("rounds", [])),
        }
        write_self_play_report(resolved_output_dir, report_payload)
        return report_payload

    latest_blue_checkpoint = dict(state.get("latest_blue_checkpoint", {}))
    latest_red_checkpoint = dict(state.get("latest_red_checkpoint", {}))
    for round_index in range(int(state.get("rounds_completed", 0)) + 1, int(rounds) + 1):
        current_round_dir = round_dir(resolved_output_dir, round_index)
        current_round_dir.mkdir(parents=True, exist_ok=True)
        frozen_blue_checkpoint = dict(latest_blue_checkpoint)
        frozen_red_checkpoint = dict(latest_red_checkpoint)
        dataset_path, dataset_source_summary = _resolve_round_dataset(
            round_index=round_index,
            output_dir=current_round_dir,
            dataset_sources=list(dataset_sources or []),
            rows=rows,
            nominal_ratio=nominal_ratio,
            seed=_round_seed(seed, round_index, 11),
            protocol_mode=protocol_mode,
            mixed_fprime_ratio=mixed_fprime_ratio,
        )
        replay_examples, replay_summary = build_replay_examples_from_dataset(
            dataset_path,
            action_space=action_space,
            transcript_budget=transcript_budget,
            max_history_entries=max_history_entries,
        )
        if not replay_examples:
            raise SelfPlayHarnessError(f"Round {round_index} produced no replay examples: {dataset_path}")

        frozen_blue_dir = Path(str(frozen_blue_checkpoint["artifact_dir"]))
        frozen_red_model_path = Path(str(frozen_red_checkpoint["artifact_dir"])) / RED_POLICY_MODEL_ARTIFACT_NAME
        frozen_red_alignment = evaluate_red_policy_alignment(
            frozen_red_model_path,
            replay_examples,
        )
        rewarded_examples, reward_summary = score_and_reward_replay_examples(
            examples=replay_examples,
            blue_model_dir=frozen_blue_dir,
            action_space=action_space,
            transcript_budget=transcript_budget,
            reward_spec=reward_spec,
        )
        reward_examples_path = current_round_dir / "reward" / "rewarded_examples.jsonl"
        _write_jsonl(reward_examples_path, rewarded_examples)
        _save_json(current_round_dir / "reward" / "reward_summary.json", reward_summary)

        replay_buffer = _collect_reward_buffer(state, rewarded_examples)
        warmstart_examples = build_red_policy_warmstart_examples(
            protocol_mode=_infer_protocol_mode_from_examples(replay_examples),
            rows_per_protocol=red_warmstart_rows_per_protocol,
            seed=_round_seed(seed, round_index, 19),
            max_history_entries=max_history_entries,
            action_space=action_space,
            transcript_budget=transcript_budget,
        )
        red_training_examples, red_buffer_summary = assemble_red_training_examples(
            warmstart_examples=warmstart_examples,
            buffered_rewarded_examples=replay_buffer,
            buffer_limit=red_replay_buffer_limit,
            round_index=round_index,
        )
        red_update_dir = current_round_dir / "red_update"
        red_update = train_red_candidate(
            output_dir=red_update_dir,
            training_examples=red_training_examples,
            protocol_mode=_infer_protocol_mode_from_examples(replay_examples),
            rows_per_protocol=red_warmstart_rows_per_protocol,
            seed=_round_seed(seed, round_index, 23),
            action_space=action_space,
            transcript_budget=transcript_budget,
            buffer_summary=red_buffer_summary,
        )
        candidate_red_alignment = evaluate_red_policy_alignment(
            red_update_dir / RED_POLICY_MODEL_ARTIFACT_NAME,
            replay_examples,
        )
        latest_red_checkpoint = create_directory_checkpoint(
            output_dir=resolved_output_dir,
            side="red",
            checkpoint_id=checkpoint_id_for_round(round_index),
            source_dir=red_update_dir,
            metadata={
                "round_index": round_index,
                "source": "reward_weighted_replay_update",
                "protocol_mode": _infer_protocol_mode_from_examples(replay_examples),
                "train_example_count": len(red_training_examples),
            },
        )

        blue_update_dir = current_round_dir / "blue_update"
        blue_report, blue_error = run_blue_update_round(
            dataset_path=dataset_path,
            output_dir=blue_update_dir,
            seed=_round_seed(seed, round_index, 29),
            make_plots=make_plots,
        )
        blue_model_dir, blue_bundle_source = _resolve_blue_runtime_bundle_dir(blue_update_dir, blue_report)
        blue_candidate_checkpoint = None
        if blue_report is not None and blue_model_dir is not None:
            blue_candidate_checkpoint = create_directory_checkpoint(
                output_dir=resolved_output_dir,
                side="blue",
                checkpoint_id=checkpoint_id_for_round(round_index),
                source_dir=blue_model_dir,
                metadata={
                    "round_index": round_index,
                    "source": "poster_blue_training_update" if blue_bundle_source == "deployable_models" else "poster_blue_analysis_runtime_bundle",
                    "deployment_ready": bool(blue_report.get("deployment_ready")),
                    "dataset_path": str(dataset_path.resolve()),
                    "runtime_bundle_source": blue_bundle_source,
                },
            )
            latest_blue_checkpoint = blue_candidate_checkpoint

        round_summary = {
            "schema_version": SELF_PLAY_ROUND_SCHEMA_VERSION,
            "record_kind": "self_play_round",
            "round_index": round_index,
            "round_slug": round_slug(round_index),
            "round_seed": _round_seed(seed, round_index, 0),
            "mode": SELF_PLAY_MODE,
            "dataset": {
                **dataset_source_summary,
                "generation_summary_path": None
                if _load_generation_summary(dataset_path) is None
                else str((dataset_path.resolve().parents[1] / "reports" / "generation_summary.json").resolve()),
            },
            "frozen_blue_checkpoint": frozen_blue_checkpoint,
            "frozen_red_checkpoint": frozen_red_checkpoint,
            "replay_examples": replay_summary,
            "reward_summary": reward_summary,
            "transcript_budget": {
                "max_history_entries": int(_as_mapping(transcript_budget.get("limits")).get("max_history_entries", 0)),
                "config_path": str(transcript_budget.get("config_path")),
            },
            "frozen_red_policy_alignment": frozen_red_alignment,
            "candidate_red_policy_alignment": candidate_red_alignment,
            "red_update": {
                "checkpoint": dict(latest_red_checkpoint),
                "buffer_summary": red_buffer_summary,
                "evaluation": dict(red_update["report"]["evaluation"]),
            },
            "blue_update": {
                "report_path": None if blue_report is None else str((blue_update_dir / "reports" / "metrics.json").resolve()),
                "summary": None if blue_report is None else _extract_blue_metric_summary(blue_report),
                "error": blue_error,
                "checkpoint": None if blue_candidate_checkpoint is None else dict(blue_candidate_checkpoint),
            },
            "artifacts": {
                "round_dir": str(current_round_dir.resolve()),
                "reward_examples_path": str(reward_examples_path.resolve()),
                "reward_summary_path": str((current_round_dir / "reward" / "reward_summary.json").resolve()),
                "red_model_path": str((red_update_dir / RED_POLICY_MODEL_ARTIFACT_NAME).resolve()),
                "red_report_path": str((red_update_dir / RED_POLICY_REPORT_ARTIFACT_NAME).resolve()),
                "blue_metrics_path": None if blue_report is None else str((blue_update_dir / "reports" / "metrics.json").resolve()),
                "round_summary_path": str((current_round_dir / "round_summary.json").resolve()),
            },
        }
        _save_json(current_round_dir / "round_summary.json", round_summary)
        state = update_self_play_state(
            state,
            round_summary=round_summary,
            latest_blue_checkpoint=latest_blue_checkpoint,
            latest_red_checkpoint=latest_red_checkpoint,
            status="running",
        )
        write_self_play_state(resolved_output_dir, state)

    state["status"] = "completed"
    write_self_play_state(resolved_output_dir, state)
    report_payload = {
        "status": "completed",
        "mode": SELF_PLAY_MODE,
        "config": dict(state.get("config", {})),
        "rounds_completed": int(state.get("rounds_completed", 0)),
        "latest_blue_checkpoint": dict(state.get("latest_blue_checkpoint", {})),
        "latest_red_checkpoint": dict(state.get("latest_red_checkpoint", {})),
        "rounds": list(state.get("rounds", [])),
    }
    write_self_play_report(resolved_output_dir, report_payload)
    return report_payload


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Run the alternating blue/red self-play auto-research harness with deterministic "
            "round logging, reward summaries, and checkpointed blue/red artifacts."
        )
    )
    parser.add_argument("--rounds", type=int, default=DEFAULT_SELF_PLAY_ROUNDS)
    parser.add_argument("--rows", type=int, default=DEFAULT_SELF_PLAY_ROWS)
    parser.add_argument("--nominal-ratio", type=float, default=DEFAULT_SELF_PLAY_NOMINAL_RATIO)
    parser.add_argument("--seed", type=int, default=7)
    parser.add_argument("--protocol-mode", choices=["fprime", "mavlink", "mixed"], default=DEFAULT_SELF_PLAY_PROTOCOL_MODE)
    parser.add_argument("--mixed-fprime-ratio", type=float, default=DEFAULT_SELF_PLAY_MIXED_FPRIME_RATIO)
    parser.add_argument("--dataset", action="append", default=[])
    parser.add_argument("--initial-blue-model-dir", default=str(DEFAULT_INITIAL_BLUE_MODEL_DIR))
    parser.add_argument("--initial-red-dir")
    parser.add_argument("--red-warmstart-rows-per-protocol", type=int, default=DEFAULT_RED_WARMSTART_ROWS_PER_PROTOCOL)
    parser.add_argument("--red-replay-buffer-limit", type=int, default=DEFAULT_RED_REPLAY_BUFFER_LIMIT)
    parser.add_argument("--max-history-entries", type=int)
    parser.add_argument("--make-plots", action="store_true")
    parser.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR / "self_play_latest"))
    return parser


def main() -> None:
    args = build_parser().parse_args()
    report = run_self_play(
        output_dir=Path(args.output_dir).resolve(),
        rounds=args.rounds,
        seed=args.seed,
        rows=args.rows,
        nominal_ratio=args.nominal_ratio,
        protocol_mode=args.protocol_mode,
        mixed_fprime_ratio=args.mixed_fprime_ratio,
        dataset_sources=_dataset_sources(args.dataset),
        initial_blue_model_dir=Path(args.initial_blue_model_dir).resolve(),
        initial_red_dir=None if args.initial_red_dir is None else Path(args.initial_red_dir).resolve(),
        red_warmstart_rows_per_protocol=args.red_warmstart_rows_per_protocol,
        red_replay_buffer_limit=args.red_replay_buffer_limit,
        max_history_entries=args.max_history_entries,
        make_plots=bool(args.make_plots),
    )
    print(
        json.dumps(
            {
                "schema_version": SELF_PLAY_REPORT_SCHEMA_VERSION,
                "status": report["status"],
                "rounds_completed": report["rounds_completed"],
                "latest_blue_checkpoint": report["latest_blue_checkpoint"],
                "latest_red_checkpoint": report["latest_red_checkpoint"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
