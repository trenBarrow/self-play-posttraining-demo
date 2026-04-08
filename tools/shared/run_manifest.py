from __future__ import annotations

from collections import Counter
from typing import Any, Iterable, Mapping

DEFAULT_CYBER_FRACTION_OF_ANOMALY = 0.65
MULTI_PROTOCOL_RUN_MANIFEST_SCHEMA_VERSION = "multi_protocol_run_manifest.v1"
SUPPORTED_PROTOCOL_FAMILIES = ("fprime", "mavlink")
SUPPORTED_CLASS_NAMES = ("benign", "cyber", "fault")


class RunManifestError(ValueError):
    """Raised when a shared run-manifest request cannot be satisfied safely."""


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


def _optional_number(value: Any) -> float | None:
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


def _optional_text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _resolve_mapping_path(record: Mapping[str, Any], path: str) -> Any:
    value: Any = record
    for segment in str(path).split("."):
        if not isinstance(value, Mapping) or segment not in value:
            return None
        value = value[segment]
    return value


def build_class_overlap_report(
    records: Iterable[Mapping[str, Any]],
    *,
    value_path: str,
    family_key: str,
    item_key: str,
    items_key: str,
    class_path: str = "audit_context.label_name",
) -> dict[str, Any]:
    counts: dict[str, dict[str, int]] = {}
    row_total = 0
    shared_rows = 0
    shared_any = 0
    shared_all = 0
    exclusive = 0
    max_dominant_share = 0.0
    rows_above_ninety_five = 0
    values_above_ninety_five = 0

    for record in records:
        value = _optional_text(_resolve_mapping_path(record, value_path))
        class_name = _optional_text(_resolve_mapping_path(record, class_path))
        if value is None or class_name not in SUPPORTED_CLASS_NAMES:
            continue
        class_counts = counts.setdefault(value, {name: 0 for name in SUPPORTED_CLASS_NAMES})
        class_counts[class_name] += 1
        row_total += 1

    items: list[dict[str, Any]] = []
    for value in sorted(counts):
        class_counts = counts[value]
        shared_classes = [name for name in SUPPORTED_CLASS_NAMES if int(class_counts.get(name, 0)) > 0]
        active_count = len(shared_classes)
        total = sum(class_counts.values())
        if active_count >= 2:
            shared_any += 1
            shared_rows += total
        if active_count == len(SUPPORTED_CLASS_NAMES):
            shared_all += 1
        if active_count == 1:
            exclusive += 1
        dominant = max(class_counts.values()) if class_counts else 0
        dominant_share = 0.0 if total <= 0 else round(dominant / total, 4)
        max_dominant_share = max(max_dominant_share, dominant_share)
        if dominant_share >= 0.95:
            values_above_ninety_five += 1
            rows_above_ninety_five += total
        items.append(
            {
                item_key: value,
                "classes": dict(class_counts),
                "shared_classes": shared_classes,
                "shared_class_count": active_count,
                "exclusive": active_count == 1,
                "dominant_class_share": dominant_share,
                "rows": total,
            }
        )

    total_values = len(counts)
    return {
        "family_key": family_key,
        "value_path": str(value_path),
        "class_path": str(class_path),
        "summary": {
            "rows": row_total,
            "class_rows": {
                class_name: sum(value_counts[class_name] for value_counts in counts.values())
                for class_name in SUPPORTED_CLASS_NAMES
            },
            "total_values": total_values,
            "values_shared_by_at_least_two_classes": shared_any,
            "values_shared_by_all_classes": shared_all,
            "exclusive_values": exclusive,
            "overlap_ratio": 0.0 if total_values == 0 else round(shared_any / total_values, 4),
            "shared_row_fraction": 0.0 if row_total == 0 else round(shared_rows / row_total, 4),
            "max_dominant_class_share": round(max_dominant_share, 4),
            "values_with_dominant_class_share_ge_0_95": values_above_ninety_five,
            "rows_in_values_with_dominant_class_share_ge_0_95": rows_above_ninety_five,
        },
        items_key: items,
    }


def class_row_targets(
    *,
    rows: int,
    nominal_ratio: float,
    cyber_fraction_of_anomaly: float = DEFAULT_CYBER_FRACTION_OF_ANOMALY,
) -> dict[str, int]:
    if rows <= 0:
        raise RunManifestError("rows must be > 0")
    if not 0.0 < nominal_ratio < 1.0:
        raise RunManifestError("nominal_ratio must be between 0 and 1")
    if not 0.0 < cyber_fraction_of_anomaly < 1.0:
        raise RunManifestError("cyber_fraction_of_anomaly must be between 0 and 1")
    benign_rows = int(round(rows * nominal_ratio))
    anomaly_rows = int(rows - benign_rows)
    cyber_rows = int(round(anomaly_rows * cyber_fraction_of_anomaly))
    fault_rows = int(anomaly_rows - cyber_rows)
    return {
        "benign": benign_rows,
        "cyber": cyber_rows,
        "fault": fault_rows,
    }


def protocol_row_targets(
    *,
    protocol_mode: str,
    rows: int,
    mixed_fprime_ratio: float = 0.5,
) -> dict[str, int]:
    normalized_mode = str(protocol_mode).strip().lower()
    if normalized_mode not in {"fprime", "mavlink", "mixed"}:
        raise RunManifestError(f"Unsupported protocol_mode {protocol_mode!r}")
    if rows <= 0:
        raise RunManifestError("rows must be > 0")
    if normalized_mode == "fprime":
        return {"fprime": rows, "mavlink": 0}
    if normalized_mode == "mavlink":
        return {"fprime": 0, "mavlink": rows}
    if not 0.0 < mixed_fprime_ratio < 1.0:
        raise RunManifestError("mixed_fprime_ratio must be between 0 and 1 for protocol_mode='mixed'")
    if rows < 2:
        raise RunManifestError("protocol_mode='mixed' requires at least 2 rows so both protocols are represented")
    fprime_rows = int(round(rows * mixed_fprime_ratio))
    fprime_rows = max(1, min(rows - 1, fprime_rows))
    mavlink_rows = int(rows - fprime_rows)
    return {"fprime": fprime_rows, "mavlink": mavlink_rows}


def protocol_seed_map(*, seed: int, protocol_families: Iterable[str]) -> dict[str, int]:
    values: dict[str, int] = {}
    for index, protocol_family in enumerate(protocol_families):
        normalized = str(protocol_family).strip().lower()
        if normalized not in SUPPORTED_PROTOCOL_FAMILIES:
            raise RunManifestError(f"Unsupported protocol family {protocol_family!r}")
        values[normalized] = int(seed + index * 1000)
    return values


def build_multi_protocol_run_manifest(
    *,
    protocol_mode: str,
    requested_rows: int,
    nominal_ratio: float,
    seed: int,
    mixed_fprime_ratio: float | None,
    history_group_key: str,
    history_reset_policy: str,
    protocol_execution_order: Iterable[str],
    protocol_row_targets_by_family: Mapping[str, int],
    runs: Iterable[Mapping[str, Any]],
) -> dict[str, Any]:
    rows_list: list[dict[str, Any]] = []
    class_counter = Counter()
    protocol_counter = Counter()
    protocol_class_counter: dict[str, Counter[str]] = {}
    runtime_reset_policies: set[str] = set()

    for run_order, run in enumerate(runs):
        protocol_family = _optional_text(run.get("protocol_family")) or "unknown"
        class_name = _optional_text(run.get("class_name")) or "unknown"
        rows = _optional_int(run.get("rows")) or 0
        runtime_reset_policy = _optional_text(run.get("runtime_reset_policy")) or history_reset_policy
        runtime_reset_policies.add(runtime_reset_policy)
        class_counter[class_name] += 1
        protocol_counter[protocol_family] += 1
        protocol_class_counter.setdefault(protocol_family, Counter())[class_name] += 1
        rows_list.append(
            {
                "run_id": _optional_int(run.get("run_id")),
                "run_order": _optional_int(run.get("run_order"))
                if _optional_int(run.get("run_order")) is not None
                else run_order,
                "protocol_family": protocol_family,
                "class_name": class_name,
                "label": _optional_int(run.get("label")),
                "episode_id": _optional_int(run.get("episode_id")),
                "rows": rows,
                "runtime_reset_policy": runtime_reset_policy,
                "seed": _optional_int(run.get("seed")),
                "schedule_path": _optional_text(run.get("schedule_path")),
                "generation_root": _optional_text(run.get("generation_root")),
            }
        )

    if not rows_list:
        raise RunManifestError("Cannot build a multi-protocol run manifest without at least one run")

    protocol_targets = {
        str(protocol_family): int(protocol_row_targets_by_family.get(protocol_family, 0) or 0)
        for protocol_family in SUPPORTED_PROTOCOL_FAMILIES
    }
    return {
        "schema_version": MULTI_PROTOCOL_RUN_MANIFEST_SCHEMA_VERSION,
        "protocol_mode": str(protocol_mode),
        "requested_rows": int(requested_rows),
        "nominal_ratio": float(nominal_ratio),
        "mixed_fprime_ratio": None if mixed_fprime_ratio is None else float(mixed_fprime_ratio),
        "history_group_key": str(history_group_key),
        "history_reset_policy": str(history_reset_policy),
        "protocol_execution_order": [str(item) for item in protocol_execution_order],
        "protocol_row_targets": protocol_targets,
        "run_count": len(rows_list),
        "class_run_counts": dict(class_counter),
        "protocol_run_counts": dict(protocol_counter),
        "protocol_class_run_counts": {
            protocol_family: dict(counter)
            for protocol_family, counter in protocol_class_counter.items()
        },
        "runtime_reset_policies": sorted(runtime_reset_policies),
        "runs": rows_list,
    }
