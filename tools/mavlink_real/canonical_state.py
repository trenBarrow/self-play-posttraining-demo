from __future__ import annotations

from typing import Any, Mapping

from tools.shared.canonical_state import (
    _as_mapping,
    _bounded_ratio,
    _entity_present,
    _max_optional,
    _min_optional,
    _optional_int,
    _optional_number,
    empty_normalized_state,
)

MAVLINK_TELEMETRY_STALENESS_LIMIT_MS = 5000.0
MAVLINK_MODE_FLAG_AUTO_ENABLED = 0x04
MAVLINK_MODE_FLAG_GUIDED_ENABLED = 0x08
MAVLINK_MODE_FLAG_STABILIZE_ENABLED = 0x10
MAVLINK_MODE_FLAG_MANUAL_INPUT_ENABLED = 0x40
MAVLINK_MODE_FLAG_SAFETY_ARMED = 0x80

MAV_STATE_PRESSURE = {
    0: 1.0,
    1: 0.3,
    2: 0.4,
    3: 0.05,
    4: 0.0,
    5: 0.8,
    6: 1.0,
    7: 1.0,
    8: 1.0,
}

MAVLINK_NAVIGATION_SENSOR_MASK = (
    0x01  # 3D gyro
    | 0x02  # 3D accel
    | 0x04  # 3D mag
    | 0x20  # GPS
    | 0x800  # attitude stabilization
    | 0x1000  # yaw position
    | 0x2000  # altitude control
    | 0x4000  # XY position control
)


def _clamp_zero_one(value: float) -> float:
    return min(max(value, 0.0), 1.0)


def _bounded_deficit(value: Any, *, nominal: float, floor: float) -> float | None:
    number = _optional_number(value)
    if number is None:
        return None
    if number >= nominal:
        return 0.0
    if number <= floor:
        return 1.0
    if nominal <= floor:
        return 1.0
    return _clamp_zero_one((nominal - number) / (nominal - floor))


def _battery_pressure(fields: Mapping[str, Any]) -> float | None:
    remaining_pct = _min_optional(
        fields.get("battery_remaining_pct"),
        fields.get("battery_status_remaining_pct"),
    )
    if remaining_pct is None:
        return None
    return _bounded_ratio(max(100.0 - remaining_pct, 0.0), 100.0)


def _control_mode_pressure(base_mode: Any) -> float | None:
    value = _optional_int(base_mode)
    if value is None:
        return None
    if value & (
        MAVLINK_MODE_FLAG_AUTO_ENABLED
        | MAVLINK_MODE_FLAG_GUIDED_ENABLED
        | MAVLINK_MODE_FLAG_STABILIZE_ENABLED
        | MAVLINK_MODE_FLAG_MANUAL_INPUT_ENABLED
    ):
        return 0.0
    if value & MAVLINK_MODE_FLAG_SAFETY_ARMED:
        return 0.25
    return 0.1


def _gps_fix_pressure(fix_type: Any) -> float | None:
    value = _optional_int(fix_type)
    if value is None:
        return None
    if value >= 3:
        return 0.0
    if value == 2:
        return 0.65
    if value == 1:
        return 0.9
    return 1.0


def _satellite_pressure(satellites_visible: Any) -> float | None:
    satellites = _optional_number(satellites_visible)
    if satellites is None:
        return None
    return _bounded_ratio(max(10.0 - satellites, 0.0), 6.0)


def _sensor_health_pressure(fields: Mapping[str, Any]) -> float | None:
    enabled_mask = _optional_int(fields.get("onboard_control_sensors_enabled"))
    health_mask = _optional_int(fields.get("onboard_control_sensors_health"))
    if enabled_mask is None or health_mask is None:
        return None
    relevant_enabled = enabled_mask & MAVLINK_NAVIGATION_SENSOR_MASK
    if relevant_enabled == 0:
        return None
    unhealthy = relevant_enabled & ~health_mask
    enabled_count = relevant_enabled.bit_count()
    if enabled_count <= 0:
        return None
    return unhealthy.bit_count() / float(enabled_count)


def _telemetry_staleness_ratio(
    *,
    submitted_at_ms: Any,
    snapshot_observed_at_ms: Any,
) -> float | None:
    submitted = _optional_number(submitted_at_ms)
    observed = _optional_number(snapshot_observed_at_ms)
    if submitted is None or observed is None:
        return None
    return _bounded_ratio(max(submitted - observed, 0.0), MAVLINK_TELEMETRY_STALENESS_LIMIT_MS)


def _summarize_mavlink_entity(
    fields: Mapping[str, Any],
    *,
    telemetry_staleness_ratio: float | None,
) -> dict[str, float | None]:
    compute_load_ratio = _bounded_ratio(fields.get("sys_load_fraction"), 1.0)
    system_status_pressure = MAV_STATE_PRESSURE.get(_optional_int(fields.get("heartbeat_system_status")))
    return {
        "compute_load_ratio": compute_load_ratio,
        "compute_peak_load_ratio": compute_load_ratio,
        "compute_imbalance_ratio": None,
        "storage_io_pressure_ratio": None,
        "command_activity_ratio": None,
        "command_error_ratio": None,
        "service_issue_ratio": None,
        "queue_pressure_ratio": None,
        "scheduler_pressure_ratio": None,
        "link_pressure_ratio": _bounded_ratio(fields.get("drop_rate_comm_fraction"), 1.0),
        "power_pressure_ratio": _max_optional(
            _battery_pressure(fields),
            _bounded_deficit(fields.get("power_vcc_v"), nominal=4.9, floor=4.2),
            _bounded_deficit(fields.get("power_servo_v"), nominal=4.9, floor=4.2),
        ),
        "control_instability_ratio": _max_optional(
            system_status_pressure,
            _control_mode_pressure(fields.get("heartbeat_base_mode")),
        ),
        "navigation_uncertainty_ratio": _max_optional(
            _gps_fix_pressure(fields.get("gps_fix_type")),
            _satellite_pressure(fields.get("gps_satellites_visible")),
            _sensor_health_pressure(fields),
        ),
        "telemetry_staleness_ratio": telemetry_staleness_ratio,
    }


def summarize_mavlink_normalized_state(raw_transaction: Mapping[str, Any]) -> dict[str, Any]:
    snapshot = _as_mapping(raw_transaction.get("native_state_snapshot"))
    target_fields = _as_mapping(snapshot.get("target_fields"))
    peer_fields = _as_mapping(snapshot.get("peer_fields"))
    target_present = _entity_present(target_fields)
    peer_present = _entity_present(peer_fields)
    state = empty_normalized_state()
    state["state_available"] = target_present or peer_present
    state["target_state_present"] = target_present
    state["peer_state_present"] = peer_present

    target_staleness = _telemetry_staleness_ratio(
        submitted_at_ms=_as_mapping(raw_transaction.get("timing")).get("submitted_at_ms"),
        snapshot_observed_at_ms=snapshot.get("snapshot_observed_at_ms"),
    )
    peer_staleness = target_staleness if peer_present else None

    target_summary = _summarize_mavlink_entity(
        target_fields,
        telemetry_staleness_ratio=target_staleness,
    )
    peer_summary = _summarize_mavlink_entity(
        peer_fields,
        telemetry_staleness_ratio=peer_staleness,
    )
    for key, value in target_summary.items():
        state[f"target_{key}"] = value
    for key, value in peer_summary.items():
        state[f"peer_{key}"] = value
    return state
