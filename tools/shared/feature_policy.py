from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Iterable, Mapping

import yaml

BLUE_FEATURE_POLICY_POSTER_DEFAULT = "poster_blue_default"
BLUE_FEATURE_POLICY_LEGACY_REQUEST_TIME_BASELINE = "legacy_request_time_baseline"

REPO_ROOT = Path(__file__).resolve().parents[2]
FEATURE_POLICY_DIR = REPO_ROOT / "configs" / "feature_policies"
BLUE_ALLOWED_FEATURES_PATH = FEATURE_POLICY_DIR / "blue_allowed_features.yaml"
BLUE_FORBIDDEN_FEATURES_PATH = FEATURE_POLICY_DIR / "blue_forbidden_features.yaml"


class BlueFeaturePolicyError(ValueError):
    """Raised when a blue-model feature list violates the configured policy."""


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


def load_blue_allowed_feature_policies() -> dict[str, Any]:
    return _load_yaml(BLUE_ALLOWED_FEATURES_PATH)


def load_blue_forbidden_feature_policies() -> dict[str, Any]:
    return _load_yaml(BLUE_FORBIDDEN_FEATURES_PATH)


def available_blue_feature_policies() -> list[str]:
    allowed = set(load_blue_allowed_feature_policies().get("profiles", {}))
    forbidden = set(load_blue_forbidden_feature_policies().get("profiles", {}))
    names = sorted(allowed & forbidden)
    if not names:
        raise BlueFeaturePolicyError("No blue feature policies are available")
    missing_allowed = sorted(forbidden - allowed)
    missing_forbidden = sorted(allowed - forbidden)
    if missing_allowed or missing_forbidden:
        raise BlueFeaturePolicyError(
            "Blue feature policy config mismatch: "
            f"missing_allowed={missing_allowed} missing_forbidden={missing_forbidden}"
        )
    return names


def load_blue_feature_policy(policy_name: str) -> dict[str, Any]:
    available = set(available_blue_feature_policies())
    if policy_name not in available:
        raise BlueFeaturePolicyError(
            f"Unknown blue feature policy {policy_name!r}; available={sorted(available)}"
        )
    allowed_profile = load_blue_allowed_feature_policies()["profiles"][policy_name]
    forbidden_profile = load_blue_forbidden_feature_policies()["profiles"][policy_name]
    allowed_features = list(allowed_profile.get("allowed_features", []))
    forbidden_features = dict(forbidden_profile.get("forbidden_features", {}))
    forbidden_prefixes = dict(forbidden_profile.get("forbidden_prefixes", {}))
    if not allowed_features:
        raise BlueFeaturePolicyError(f"Blue feature policy {policy_name!r} has no allowed features")
    return {
        "policy_name": policy_name,
        "description": str(allowed_profile.get("description", "")),
        "allowed_features": allowed_features,
        "forbidden_features": forbidden_features,
        "forbidden_prefixes": forbidden_prefixes,
    }


def flatten_feature_mapping(
    record: Mapping[str, Any],
    *,
    prefix: str = "",
    skip_prefixes: Iterable[str] | None = None,
) -> dict[str, Any]:
    flattened: dict[str, Any] = {}
    skipped = tuple(skip_prefixes or ())
    for key, value in record.items():
        name = f"{prefix}.{key}" if prefix else str(key)
        if skipped and any(name == skip or name.startswith(skip) for skip in skipped):
            continue
        if isinstance(value, Mapping):
            flattened.update(flatten_feature_mapping(value, prefix=name, skip_prefixes=skipped))
        else:
            flattened[name] = value
    return flattened


def check_blue_feature_names(feature_names: Iterable[str], policy_name: str) -> dict[str, Any]:
    feature_names = [str(name) for name in feature_names]
    policy = load_blue_feature_policy(policy_name)
    allowed = set(policy["allowed_features"])
    forbidden_features = dict(policy["forbidden_features"])
    forbidden_prefixes = dict(policy["forbidden_prefixes"])
    seen: set[str] = set()
    duplicates: list[str] = []
    allowed_used: list[str] = []
    unexpected: list[str] = []
    forbidden: list[dict[str, str]] = []

    for name in feature_names:
        if name in seen and name not in duplicates:
            duplicates.append(name)
        seen.add(name)
        if name in forbidden_features:
            forbidden.append(
                {
                    "feature_name": name,
                    "reason": str(forbidden_features[name]),
                    "rule_type": "exact",
                }
            )
            continue
        matched_prefix = next((prefix for prefix in forbidden_prefixes if name.startswith(prefix)), None)
        if matched_prefix is not None:
            forbidden.append(
                {
                    "feature_name": name,
                    "reason": str(forbidden_prefixes[matched_prefix]),
                    "rule_type": "prefix",
                }
            )
            continue
        if name in allowed:
            allowed_used.append(name)
            continue
        unexpected.append(name)

    return {
        "policy_name": policy_name,
        "description": str(policy["description"]),
        "allowed_feature_names": list(policy["allowed_features"]),
        "selected_feature_names": list(feature_names),
        "allowed_used": sorted(allowed_used),
        "duplicate_feature_names": sorted(duplicates),
        "forbidden_violations": forbidden,
        "unexpected_feature_names": sorted(unexpected),
        "passed": not duplicates and not forbidden and not unexpected and bool(feature_names),
    }


def validate_blue_feature_names(feature_names: Iterable[str], policy_name: str) -> dict[str, Any]:
    report = check_blue_feature_names(feature_names, policy_name)
    if report["passed"]:
        return report
    raise BlueFeaturePolicyError(json.dumps(report, sort_keys=True))


def extract_blue_model_features(
    record: Mapping[str, Any],
    *,
    policy_name: str,
    require_all_allowed: bool = False,
) -> dict[str, Any]:
    policy = load_blue_feature_policy(policy_name)
    flattened = flatten_feature_mapping(record)
    selected = {name: flattened[name] for name in policy["allowed_features"] if name in flattened}
    validate_blue_feature_names(selected.keys(), policy_name)
    if require_all_allowed:
        missing = [name for name in policy["allowed_features"] if name not in selected]
        if missing:
            raise BlueFeaturePolicyError(
                f"Feature record is missing allowed fields for {policy_name}: {missing}"
            )
    return selected
