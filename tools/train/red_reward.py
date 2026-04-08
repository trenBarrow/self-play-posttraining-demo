#!/usr/bin/env python3
from __future__ import annotations

from typing import Any, Mapping

from tools.train.red_policy_model import (
    allowed_identity_services,
    load_red_action_space,
    validate_red_policy_action,
)
from tools.train.red_transcript import (
    RED_TRANSCRIPT_SCHEMA_VERSION,
    coarse_result_class_from_outcome,
    load_red_context_budget,
)

RED_REWARD_SPEC_SCHEMA_VERSION = "red_reward_spec.v1"
RED_REWARD_RESULT_SCHEMA_VERSION = "red_reward_result.v1"

RED_UNSAFE_LABEL_NAMES = ("cyber", "fault")
RED_BENIGN_LABEL_NAME = "benign"
RED_DEFAULT_BLUE_UNSAFE_THRESHOLD = 0.50


class RedRewardError(ValueError):
    """Raised when the red reward or sandbox inputs are invalid."""


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


def _bool(value: Any) -> bool | None:
    if value is None or value == "":
        return None
    if isinstance(value, bool):
        return value
    if value in (0, 1):
        return bool(value)
    if isinstance(value, str):
        normalized = value.strip().lower()
        if normalized in {"true", "1", "yes"}:
            return True
        if normalized in {"false", "0", "no"}:
            return False
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


def _append_reason(reasons: list[str], reason: str) -> None:
    if reason not in reasons:
        reasons.append(reason)


def _transcript_token_slot_count(transcript_budget: Mapping[str, Any]) -> int:
    transcript_format = _as_mapping(transcript_budget.get("transcript_format"))
    limits = _as_mapping(transcript_budget.get("limits"))
    max_history_entries = int(limits.get("max_history_entries", 0) or 0)
    header_tokens = 0
    if bool(limits.get("include_header_tokens", True)):
        header_tokens = len(transcript_format.get("header_field_order") or []) + 2
    per_event_tokens = len(transcript_format.get("per_event_field_order") or []) + 2
    return header_tokens + (per_event_tokens * max_history_entries)


def load_red_reward_spec() -> dict[str, Any]:
    return {
        "schema_version": RED_REWARD_SPEC_SCHEMA_VERSION,
        "record_kind": "red_reward_spec",
        "unsafe_label_policy": {
            "explicit_boolean_keys": ["unsafe_label", "unsafe_action", "unsafe_command"],
            "unsafe_label_names": list(RED_UNSAFE_LABEL_NAMES),
            "benign_label_name": RED_BENIGN_LABEL_NAME,
            "unsafe_numeric_labels": [1, 2],
        },
        "blue_detection_policy": {
            "explicit_boolean_keys": ["unsafe_detected", "detected_unsafe", "flagged_unsafe"],
            "preferred_score_keys": ["unsafe_risk", "panomaly", "risk", "anomaly_score", "pcyber"],
            "predicted_unsafe_classes": list(RED_UNSAFE_LABEL_NAMES),
            "default_unsafe_threshold": RED_DEFAULT_BLUE_UNSAFE_THRESHOLD,
            "benign_reason": "benign_baseline",
        },
        "outcome_rewards": {
            "unsafe_executed_undetected": 1.00,
            "unsafe_accepted_undetected": 0.60,
            "unsafe_executed_detected": 0.20,
            "unsafe_accepted_detected": 0.00,
            "unsafe_blocked_or_failed": -0.20,
            "benign_executed": -0.35,
            "benign_accepted": -0.25,
            "benign_blocked_or_failed": -0.10,
        },
        "penalties": {
            "invalid_action": -1.00,
            "unsupported_protocol_family": -1.00,
            "missing_current_context": -0.60,
            "invalid_transcript": -0.80,
            "transcript_budget_abuse": -0.75,
            "source_service_out_of_bucket": -0.80,
            "bootstrap_after_history": -0.40,
            "rapid_repeat_spam": -0.25,
            "repeated_ineffective": -0.35,
        },
        "spam_policy": {
            "rapid_repeat_same_family_threshold": 2,
            "ineffective_result_classes": ["rejected", "failed", "invalid", "timeout"],
        },
        "sandbox_policy": {
            "required_current_context_fields": [
                "protocol_family",
                "platform_family",
                "mission_phase",
                "window_class",
            ],
            "supported_protocol_families": ["fprime", "mavlink"],
            "enforce_bootstrap_only_without_history": True,
        },
    }


def normalize_red_environment_feedback(
    feedback: Mapping[str, Any],
    *,
    reward_spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(feedback, Mapping):
        raise RedRewardError("environment_feedback must be a mapping")
    resolved_spec = reward_spec or load_red_reward_spec()
    label_policy = _as_mapping(resolved_spec.get("unsafe_label_policy"))
    explicit_keys = [str(name) for name in label_policy.get("explicit_boolean_keys", [])]

    unsafe_label = None
    for key in explicit_keys:
        unsafe_label = _bool(feedback.get(key))
        if unsafe_label is not None:
            break

    outcome = _as_mapping(feedback.get("outcome"))
    label_name = _compact_category(
        feedback.get("label_name") or outcome.get("label_name"),
        "unknown",
    )
    label_value = _number(feedback.get("label"))

    if unsafe_label is None and label_name != "unknown":
        unsafe_label = label_name in {
            _compact_category(name, "unknown")
            for name in label_policy.get("unsafe_label_names", [])
        }
    if unsafe_label is None and label_value is not None:
        unsafe_label = int(label_value) in {
            int(value)
            for value in label_policy.get("unsafe_numeric_labels", [])
            if isinstance(value, int)
        }
    if unsafe_label is None:
        raise RedRewardError(
            "environment_feedback must carry unsafe_label or a label/label_name so the reward depends on real labels"
        )

    accepted = _bool(feedback.get("accepted"))
    if accepted is None:
        accepted = _bool(outcome.get("accepted"))
    executed_successfully = _bool(feedback.get("executed_successfully"))
    if executed_successfully is None:
        executed_successfully = _bool(outcome.get("executed_successfully"))
    timed_out = _bool(feedback.get("timed_out"))
    if timed_out is None:
        timed_out = _bool(outcome.get("timed_out"))

    result_class = _compact_category(
        feedback.get("coarse_result_class")
        or feedback.get("result_class")
        or coarse_result_class_from_outcome(outcome or feedback),
        "unknown",
    )

    return {
        "unsafe_label": bool(unsafe_label),
        "label_name": label_name,
        "label": None if label_value is None else int(label_value),
        "accepted": accepted,
        "executed_successfully": executed_successfully,
        "timed_out": timed_out,
        "coarse_result_class": result_class,
    }


def normalize_red_blue_feedback(
    feedback: Mapping[str, Any],
    *,
    reward_spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    if not isinstance(feedback, Mapping):
        raise RedRewardError("blue_feedback must be a mapping")
    resolved_spec = reward_spec or load_red_reward_spec()
    detection_policy = _as_mapping(resolved_spec.get("blue_detection_policy"))

    explicit_detected = None
    explicit_keys = [str(name) for name in detection_policy.get("explicit_boolean_keys", [])]
    for key in explicit_keys:
        explicit_detected = _bool(feedback.get(key))
        if explicit_detected is not None:
            break

    threshold = _number(feedback.get("unsafe_threshold"))
    if threshold is None:
        threshold = float(detection_policy.get("default_unsafe_threshold", RED_DEFAULT_BLUE_UNSAFE_THRESHOLD))

    score_value = None
    score_name = None
    for key in detection_policy.get("preferred_score_keys", []):
        number = _number(feedback.get(key))
        if number is None:
            continue
        score_name = str(key)
        score_value = float(number)
        break

    predicted_class = _compact_category(feedback.get("predicted_class"), "unknown")
    detector_reason = _compact_category(feedback.get("detector_reason"), "unknown")
    benign_reason = _compact_category(detection_policy.get("benign_reason"), "benign_baseline")
    predicted_unsafe = {
        _compact_category(name, "unknown")
        for name in detection_policy.get("predicted_unsafe_classes", [])
    }

    detection_source = "unknown"
    if explicit_detected is not None:
        unsafe_detected = bool(explicit_detected)
        detection_source = "explicit_boolean"
    elif score_value is not None:
        unsafe_detected = float(score_value) >= float(threshold)
        detection_source = f"{score_name}_threshold"
    elif predicted_class in predicted_unsafe:
        unsafe_detected = True
        detection_source = "predicted_class"
    elif predicted_class == RED_BENIGN_LABEL_NAME:
        unsafe_detected = False
        detection_source = "predicted_class"
    elif detector_reason == benign_reason:
        unsafe_detected = False
        detection_source = "detector_reason"
    elif detector_reason != "unknown":
        unsafe_detected = True
        detection_source = "detector_reason"
    else:
        raise RedRewardError(
            "blue_feedback must carry unsafe_detected, an unsafe score, predicted_class, or detector_reason"
        )

    return {
        "unsafe_detected": bool(unsafe_detected),
        "detection_source": detection_source,
        "unsafe_threshold": float(threshold),
        "score_name": score_name,
        "score_value": score_value,
        "predicted_class": predicted_class,
        "detector_reason": detector_reason,
    }


def inspect_red_action_sandbox(
    *,
    action: Mapping[str, Any],
    current_context: Mapping[str, Any],
    transcript: Mapping[str, Any],
    source_service: str | None = None,
    action_space: Mapping[str, Any] | None = None,
    transcript_budget: Mapping[str, Any] | None = None,
    reward_spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_action_space = action_space or load_red_action_space()
    resolved_budget = transcript_budget or load_red_context_budget()
    resolved_spec = reward_spec or load_red_reward_spec()
    sandbox_policy = _as_mapping(resolved_spec.get("sandbox_policy"))

    reasons: list[str] = []
    validated_action: dict[str, str] | None = None
    try:
        validated_action = validate_red_policy_action(action, action_space=resolved_action_space)
    except ValueError:
        _append_reason(reasons, "invalid_action")

    required_fields = [str(name) for name in sandbox_policy.get("required_current_context_fields", [])]
    missing_fields = [name for name in required_fields if _text(current_context.get(name)) is None]
    if missing_fields:
        _append_reason(reasons, "missing_current_context")

    protocol_family = _compact_category(current_context.get("protocol_family"), "unknown")
    supported_protocols = {
        _compact_category(name, "unknown")
        for name in sandbox_policy.get("supported_protocol_families", [])
    }
    if protocol_family not in supported_protocols:
        _append_reason(reasons, "unsupported_protocol_family")

    transcript_record_kind = _text(transcript.get("record_kind"))
    transcript_schema_version = _text(transcript.get("schema_version"))
    if transcript_record_kind != "red_command_transcript" or transcript_schema_version != RED_TRANSCRIPT_SCHEMA_VERSION:
        _append_reason(reasons, "invalid_transcript")

    configured_max_history_entries = int(_as_mapping(resolved_budget.get("limits")).get("max_history_entries", 0))
    declared_budget = _as_mapping(transcript.get("budget"))
    declared_max_history_entries = int(
        _number(declared_budget.get("max_history_entries")) or configured_max_history_entries
    )
    included_history_count = int(_number(transcript.get("included_history_count")) or 0)
    event_count = len(transcript.get("events") or [])
    flattened_token_count = len(transcript.get("flattened_token_ids") or [])
    if event_count != included_history_count:
        _append_reason(reasons, "invalid_transcript")
    if declared_max_history_entries > configured_max_history_entries or included_history_count > configured_max_history_entries:
        _append_reason(reasons, "transcript_budget_abuse")
    if included_history_count > declared_max_history_entries:
        _append_reason(reasons, "transcript_budget_abuse")
    if flattened_token_count > _transcript_token_slot_count(resolved_budget):
        _append_reason(reasons, "transcript_budget_abuse")

    allowed_source_services: list[str] = []
    normalized_source_service = _text(source_service)
    if validated_action is not None and protocol_family in supported_protocols:
        allowed_source_services = allowed_identity_services(
            protocol_family,
            validated_action["identity_bucket"],
            action_space=resolved_action_space,
        )
        if normalized_source_service is not None and normalized_source_service not in allowed_source_services:
            _append_reason(reasons, "source_service_out_of_bucket")
        if (
            bool(sandbox_policy.get("enforce_bootstrap_only_without_history", True))
            and validated_action["timing_bucket"] == "bootstrap"
            and included_history_count > 0
        ):
            _append_reason(reasons, "bootstrap_after_history")

    return {
        "allowed": not reasons,
        "violation_reasons": reasons,
        "validated_action": validated_action,
        "protocol_family": protocol_family,
        "source_service": normalized_source_service,
        "allowed_source_services": allowed_source_services,
        "missing_current_context_fields": missing_fields,
        "transcript_summary": {
            "record_kind": transcript_record_kind,
            "schema_version": transcript_schema_version,
            "configured_max_history_entries": configured_max_history_entries,
            "declared_max_history_entries": declared_max_history_entries,
            "included_history_count": included_history_count,
            "event_count": event_count,
            "flattened_token_count": flattened_token_count,
            "max_flattened_token_slots": _transcript_token_slot_count(resolved_budget),
        },
    }


def analyze_red_action_pattern(
    *,
    action: Mapping[str, Any],
    transcript: Mapping[str, Any],
    reward_spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_spec = reward_spec or load_red_reward_spec()
    validated_action = validate_red_policy_action(action, action_space=load_red_action_space())
    spam_policy = _as_mapping(resolved_spec.get("spam_policy"))
    ineffective_result_classes = {
        _compact_category(name, "unknown")
        for name in spam_policy.get("ineffective_result_classes", [])
    }
    repeat_threshold = int(spam_policy.get("rapid_repeat_same_family_threshold", 2) or 2)

    desired_family = validated_action["command_family"]
    events = list(transcript.get("events") or [])
    trailing_same_family = 0
    last_result_class = "unknown"
    for event in reversed(events):
        field_values = _as_mapping(_as_mapping(event).get("field_values"))
        observed_family = _compact_category(field_values.get("canonical_command_family"), "unknown")
        if observed_family != desired_family:
            break
        trailing_same_family += 1
        if trailing_same_family == 1:
            last_result_class = _compact_category(field_values.get("coarse_result_class"), "unknown")

    rapid_repeat_spam = validated_action["timing_bucket"] == "rapid" and trailing_same_family >= repeat_threshold
    repeated_ineffective = trailing_same_family >= 1 and last_result_class in ineffective_result_classes
    return {
        "trailing_same_family_count": trailing_same_family,
        "last_matching_result_class": last_result_class,
        "rapid_repeat_spam": rapid_repeat_spam,
        "repeated_ineffective": repeated_ineffective,
    }


def classify_red_reward_case(
    environment_feedback: Mapping[str, Any],
    blue_feedback: Mapping[str, Any],
) -> str:
    unsafe_label = bool(environment_feedback.get("unsafe_label"))
    accepted = environment_feedback.get("accepted") is True
    executed_successfully = environment_feedback.get("executed_successfully") is True
    unsafe_detected = bool(blue_feedback.get("unsafe_detected"))

    if unsafe_label:
        if executed_successfully:
            return "unsafe_executed_detected" if unsafe_detected else "unsafe_executed_undetected"
        if accepted:
            return "unsafe_accepted_detected" if unsafe_detected else "unsafe_accepted_undetected"
        return "unsafe_blocked_or_failed"
    if executed_successfully:
        return "benign_executed"
    if accepted:
        return "benign_accepted"
    return "benign_blocked_or_failed"


def compute_red_reward(
    *,
    action: Mapping[str, Any],
    current_context: Mapping[str, Any],
    transcript: Mapping[str, Any],
    environment_feedback: Mapping[str, Any] | None = None,
    blue_feedback: Mapping[str, Any] | None = None,
    source_service: str | None = None,
    action_space: Mapping[str, Any] | None = None,
    transcript_budget: Mapping[str, Any] | None = None,
    reward_spec: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_action_space = action_space or load_red_action_space()
    resolved_budget = transcript_budget or load_red_context_budget()
    resolved_spec = reward_spec or load_red_reward_spec()
    penalties = _as_mapping(resolved_spec.get("penalties"))
    outcome_rewards = _as_mapping(resolved_spec.get("outcome_rewards"))

    sandbox = inspect_red_action_sandbox(
        action=action,
        current_context=current_context,
        transcript=transcript,
        source_service=source_service,
        action_space=resolved_action_space,
        transcript_budget=resolved_budget,
        reward_spec=resolved_spec,
    )

    reward_components = {
        "outcome_reward": 0.0,
        "invalid_action": 0.0,
        "unsupported_protocol_family": 0.0,
        "missing_current_context": 0.0,
        "invalid_transcript": 0.0,
        "transcript_budget_abuse": 0.0,
        "source_service_out_of_bucket": 0.0,
        "bootstrap_after_history": 0.0,
        "rapid_repeat_spam": 0.0,
        "repeated_ineffective": 0.0,
    }

    environment_summary: dict[str, Any] | None = None
    blue_summary: dict[str, Any] | None = None
    reward_case = "sandbox_blocked"
    if sandbox["allowed"]:
        if environment_feedback is None or blue_feedback is None:
            raise RedRewardError(
                "environment_feedback and blue_feedback are required whenever the red action clears the sandbox"
            )
        environment_summary = normalize_red_environment_feedback(
            environment_feedback,
            reward_spec=resolved_spec,
        )
        blue_summary = normalize_red_blue_feedback(
            blue_feedback,
            reward_spec=resolved_spec,
        )
        reward_case = classify_red_reward_case(environment_summary, blue_summary)
        reward_components["outcome_reward"] = float(outcome_rewards.get(reward_case, 0.0))
    else:
        for reason in sandbox["violation_reasons"]:
            if reason in reward_components:
                reward_components[reason] = float(penalties.get(reason, 0.0))

    pattern = None
    if sandbox["validated_action"] is not None:
        pattern = analyze_red_action_pattern(
            action=sandbox["validated_action"],
            transcript=transcript,
            reward_spec=resolved_spec,
        )
        if pattern["rapid_repeat_spam"]:
            reward_components["rapid_repeat_spam"] = float(penalties.get("rapid_repeat_spam", 0.0))
        if pattern["repeated_ineffective"]:
            reward_components["repeated_ineffective"] = float(penalties.get("repeated_ineffective", 0.0))

    total_reward = float(sum(float(value) for value in reward_components.values()))
    return {
        "schema_version": RED_REWARD_RESULT_SCHEMA_VERSION,
        "record_kind": "red_reward_result",
        "reward": total_reward,
        "reward_case": reward_case,
        "reward_components": reward_components,
        "sandbox": sandbox,
        "pattern_analysis": pattern or {
            "trailing_same_family_count": 0,
            "last_matching_result_class": "unknown",
            "rapid_repeat_spam": False,
            "repeated_ineffective": False,
        },
        "environment_feedback": environment_summary,
        "blue_feedback": blue_summary,
        "spec_summary": {
            "schema_version": RED_REWARD_SPEC_SCHEMA_VERSION,
            "unsafe_label_names": list(RED_UNSAFE_LABEL_NAMES),
            "default_blue_unsafe_threshold": float(
                _as_mapping(resolved_spec.get("blue_detection_policy")).get(
                    "default_unsafe_threshold",
                    RED_DEFAULT_BLUE_UNSAFE_THRESHOLD,
                )
            ),
        },
    }
