from __future__ import annotations

from typing import Any, Mapping

CANONICAL_STATE_BOOLEAN_FIELDS = (
    "state_available",
    "target_state_present",
    "peer_state_present",
)

CANONICAL_STATE_RATIO_FIELDS = (
    "target_compute_load_ratio",
    "peer_compute_load_ratio",
    "target_compute_peak_load_ratio",
    "peer_compute_peak_load_ratio",
    "target_compute_imbalance_ratio",
    "peer_compute_imbalance_ratio",
    "target_storage_io_pressure_ratio",
    "peer_storage_io_pressure_ratio",
    "target_command_activity_ratio",
    "peer_command_activity_ratio",
    "target_command_error_ratio",
    "peer_command_error_ratio",
    "target_service_issue_ratio",
    "peer_service_issue_ratio",
    "target_queue_pressure_ratio",
    "peer_queue_pressure_ratio",
    "target_scheduler_pressure_ratio",
    "peer_scheduler_pressure_ratio",
    "target_link_pressure_ratio",
    "peer_link_pressure_ratio",
    "target_power_pressure_ratio",
    "peer_power_pressure_ratio",
    "target_control_instability_ratio",
    "peer_control_instability_ratio",
    "target_navigation_uncertainty_ratio",
    "peer_navigation_uncertainty_ratio",
    "target_telemetry_staleness_ratio",
    "peer_telemetry_staleness_ratio",
)

CANONICAL_STATE_FIELDS = CANONICAL_STATE_BOOLEAN_FIELDS + CANONICAL_STATE_RATIO_FIELDS

FPRIME_STATE_NORMALIZATION_LIMITS = {
    "compute_pct": 100.0,
    "storage_io_cycles_1m": 32.0,
    "command_activity_1m": 12.0,
    "command_errors_1m": 4.0,
    "service_issues_1m": 4.0,
    "queue_pressure": 4.0,
    "scheduler_max_time_ms": 10.0,
    "telemetry_age_ms": 5000.0,
}


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


def _as_mapping(value: Any) -> dict[str, Any]:
    if isinstance(value, dict):
        return value
    return {}


def _bounded_ratio(value: Any, saturation: float) -> float | None:
    number = _optional_number(value)
    if number is None:
        return None
    if number <= 0.0:
        return 0.0
    if saturation <= 0.0:
        return 1.0
    return min(number / saturation, 1.0)


def _max_optional(*values: Any) -> float | None:
    numbers = [_optional_number(value) for value in values]
    present = [value for value in numbers if value is not None]
    if not present:
        return None
    return max(present)


def _min_optional(*values: Any) -> float | None:
    numbers = [_optional_number(value) for value in values]
    present = [value for value in numbers if value is not None]
    if not present:
        return None
    return min(present)


def _sum_optional(*values: Any) -> float | None:
    present = [_optional_number(value) for value in values]
    numbers = [value for value in present if value is not None]
    if not numbers:
        return None
    return sum(numbers)


def empty_normalized_state() -> dict[str, Any]:
    state: dict[str, Any] = {}
    for name in CANONICAL_STATE_BOOLEAN_FIELDS:
        state[name] = False
    for name in CANONICAL_STATE_RATIO_FIELDS:
        state[name] = None
    return state


def _entity_present(fields: Mapping[str, Any]) -> bool:
    return any(value not in (None, "") for value in fields.values())


def _summarize_fprime_entity(fields: Mapping[str, Any]) -> dict[str, float | None]:
    core_peak = _max_optional(fields.get("cpu_00_pct"), fields.get("cpu_01_pct"))
    core_min = _min_optional(fields.get("cpu_00_pct"), fields.get("cpu_01_pct"))
    service_issues = _sum_optional(
        fields.get("filemanager_errors_1m"),
        fields.get("filedownlink_warnings_1m"),
    )
    scheduler_max_time = _max_optional(fields.get("rg1_max_time_ms"), fields.get("rg2_max_time_ms"))

    compute_imbalance_ratio: float | None = None
    if core_peak is not None and core_min is not None:
        compute_imbalance_ratio = _bounded_ratio(
            core_peak - core_min,
            FPRIME_STATE_NORMALIZATION_LIMITS["compute_pct"],
        )

    return {
        "compute_load_ratio": _bounded_ratio(
            fields.get("cpu_total_pct"),
            FPRIME_STATE_NORMALIZATION_LIMITS["compute_pct"],
        ),
        "compute_peak_load_ratio": _bounded_ratio(
            core_peak if core_peak is not None else fields.get("cpu_total_pct"),
            FPRIME_STATE_NORMALIZATION_LIMITS["compute_pct"],
        ),
        "compute_imbalance_ratio": compute_imbalance_ratio,
        "storage_io_pressure_ratio": _bounded_ratio(
            fields.get("blockdrv_cycles_1m"),
            FPRIME_STATE_NORMALIZATION_LIMITS["storage_io_cycles_1m"],
        ),
        "command_activity_ratio": _bounded_ratio(
            fields.get("cmds_dispatched_1m"),
            FPRIME_STATE_NORMALIZATION_LIMITS["command_activity_1m"],
        ),
        "command_error_ratio": _bounded_ratio(
            fields.get("cmd_errors_1m"),
            FPRIME_STATE_NORMALIZATION_LIMITS["command_errors_1m"],
        ),
        "service_issue_ratio": _bounded_ratio(
            service_issues,
            FPRIME_STATE_NORMALIZATION_LIMITS["service_issues_1m"],
        ),
        "queue_pressure_ratio": _bounded_ratio(
            fields.get("hibuffs_total"),
            FPRIME_STATE_NORMALIZATION_LIMITS["queue_pressure"],
        ),
        "scheduler_pressure_ratio": _bounded_ratio(
            scheduler_max_time,
            FPRIME_STATE_NORMALIZATION_LIMITS["scheduler_max_time_ms"],
        ),
        "telemetry_staleness_ratio": _bounded_ratio(
            fields.get("telemetry_age_ms"),
            FPRIME_STATE_NORMALIZATION_LIMITS["telemetry_age_ms"],
        ),
    }


def summarize_fprime_normalized_state(snapshot: Mapping[str, Any]) -> dict[str, Any]:
    target_fields = _as_mapping(snapshot.get("target_fields"))
    peer_fields = _as_mapping(snapshot.get("peer_fields"))
    target_present = _entity_present(target_fields)
    peer_present = _entity_present(peer_fields)
    state = empty_normalized_state()
    state["state_available"] = target_present or peer_present
    state["target_state_present"] = target_present
    state["peer_state_present"] = peer_present

    target_summary = _summarize_fprime_entity(target_fields)
    peer_summary = _summarize_fprime_entity(peer_fields)
    for key, value in target_summary.items():
        state[f"target_{key}"] = value
    for key, value in peer_summary.items():
        state[f"peer_{key}"] = value
    return state


def summarize_normalized_state(raw_transaction: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = raw_transaction.get("native_state_snapshot")
    if not isinstance(snapshot, dict):
        return empty_normalized_state()
    protocol_family = str(raw_transaction.get("protocol_family", "")).strip().lower()
    if protocol_family == "fprime":
        return summarize_fprime_normalized_state(snapshot)
    if protocol_family == "mavlink":
        from tools.mavlink_real.canonical_state import summarize_mavlink_normalized_state

        return summarize_mavlink_normalized_state(raw_transaction)
    return empty_normalized_state()
