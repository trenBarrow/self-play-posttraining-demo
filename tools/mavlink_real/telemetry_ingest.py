#!/usr/bin/env python3
"""Map decoded MAVLink telemetry/state messages into reusable snapshot records."""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Iterable, Mapping

TELEMETRY_RECENT_THRESHOLD_MS = 5000
TELEMETRY_MESSAGE_NAMES = {
    "HEARTBEAT",
    "SYS_STATUS",
    "POWER_STATUS",
    "BATTERY_STATUS",
    "GPS_RAW_INT",
    "AUTOPILOT_VERSION",
}


@dataclass(frozen=True)
class TelemetryRecord:
    ts_ms: int
    target_logical_id: str
    message_name: str
    payload: dict[str, Any]
    bytes_on_wire: int
    observed_on_wire: bool
    ts_source: str
    bytes_source: str


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


def is_telemetry_message(message_name: str) -> bool:
    return str(message_name).strip().upper() in TELEMETRY_MESSAGE_NAMES


def telemetry_records_from_wire_messages(
    messages: Iterable[Mapping[str, Any]],
    *,
    target_logical_id: str,
) -> list[TelemetryRecord]:
    records: list[TelemetryRecord] = []
    for message in messages:
        if isinstance(message, Mapping):
            message_name = str(message.get("message_name", "")).strip().upper()
            payload = dict(message.get("payload", {}) or {})
            ts_ms = _optional_int(message.get("ts_ms")) or 0
            frame_len = _optional_int(message.get("frame_len")) or 0
        else:
            message_name = str(getattr(message, "message_name", "")).strip().upper()
            payload = dict(getattr(message, "payload", {}) or {})
            ts_ms = _optional_int(getattr(message, "ts_ms", None)) or 0
            frame_len = _optional_int(getattr(message, "frame_len", None)) or 0
        if not is_telemetry_message(message_name):
            continue
        records.append(
            TelemetryRecord(
                ts_ms=ts_ms,
                target_logical_id=target_logical_id,
                message_name=message_name,
                payload=payload,
                bytes_on_wire=frame_len,
                observed_on_wire=True,
                ts_source="pcap",
                bytes_source="pcap",
            )
        )
    return records


def telemetry_recent_for_time(
    telemetry_records: Iterable[TelemetryRecord],
    *,
    target_logical_id: str,
    observed_before_ms: int,
    max_age_ms: int = TELEMETRY_RECENT_THRESHOLD_MS,
) -> bool:
    for record in telemetry_records:
        if record.target_logical_id != target_logical_id:
            continue
        age_ms = observed_before_ms - record.ts_ms
        if 0 <= age_ms <= max_age_ms:
            return True
    return False


def _heartbeat_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "heartbeat_type": _optional_int(payload.get("type")),
        "heartbeat_autopilot": _optional_int(payload.get("autopilot")),
        "heartbeat_base_mode": _optional_int(payload.get("base_mode")),
        "heartbeat_custom_mode": _optional_int(payload.get("custom_mode")),
        "heartbeat_system_status": _optional_int(payload.get("system_status")),
    }


def _sys_status_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    load = _optional_number(payload.get("load"))
    drop_rate_comm = _optional_number(payload.get("drop_rate_comm"))
    voltage_battery = _optional_number(payload.get("voltage_battery"))
    current_battery = _optional_number(payload.get("current_battery"))
    return {
        "onboard_control_sensors_present": _optional_int(payload.get("onboard_control_sensors_present")),
        "onboard_control_sensors_enabled": _optional_int(payload.get("onboard_control_sensors_enabled")),
        "onboard_control_sensors_health": _optional_int(payload.get("onboard_control_sensors_health")),
        "sys_load_fraction": None if load is None else load / 1000.0,
        "battery_remaining_pct": _optional_number(payload.get("battery_remaining")),
        "drop_rate_comm_fraction": None if drop_rate_comm is None else drop_rate_comm / 10000.0,
        "errors_comm_total": _optional_number(payload.get("errors_comm")),
        "voltage_battery_v": None if voltage_battery is None else voltage_battery / 1000.0,
        "current_battery_a": None if current_battery is None else current_battery / 100.0,
    }


def _power_status_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    vcc = _optional_number(payload.get("Vcc"))
    vservo = _optional_number(payload.get("Vservo"))
    return {
        "power_vcc_v": None if vcc is None else vcc / 1000.0,
        "power_servo_v": None if vservo is None else vservo / 1000.0,
        "power_flags": _optional_int(payload.get("flags")),
    }


def _battery_status_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    current = _optional_number(payload.get("current_battery"))
    energy = _optional_number(payload.get("energy_consumed"))
    return {
        "battery_status_remaining_pct": _optional_number(payload.get("battery_remaining")),
        "battery_status_current_a": None if current is None else current / 100.0,
        "battery_status_energy_consumed_mah": energy,
    }


def _gps_raw_int_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "gps_fix_type": _optional_int(payload.get("fix_type")),
        "gps_satellites_visible": _optional_int(payload.get("satellites_visible")),
        "gps_altitude_m": None
        if _optional_number(payload.get("alt")) is None
        else _optional_number(payload.get("alt")) / 1000.0,
    }


def _autopilot_version_fields(payload: Mapping[str, Any]) -> dict[str, Any]:
    return {
        "autopilot_capabilities": _optional_number(payload.get("capabilities")),
        "flight_sw_version": _optional_number(payload.get("flight_sw_version")),
        "middleware_sw_version": _optional_number(payload.get("middleware_sw_version")),
        "os_sw_version": _optional_number(payload.get("os_sw_version")),
    }


def state_fields_for_record(record: TelemetryRecord) -> dict[str, Any]:
    if record.message_name == "HEARTBEAT":
        return _heartbeat_fields(record.payload)
    if record.message_name == "SYS_STATUS":
        return _sys_status_fields(record.payload)
    if record.message_name == "POWER_STATUS":
        return _power_status_fields(record.payload)
    if record.message_name == "BATTERY_STATUS":
        return _battery_status_fields(record.payload)
    if record.message_name == "GPS_RAW_INT":
        return _gps_raw_int_fields(record.payload)
    if record.message_name == "AUTOPILOT_VERSION":
        return _autopilot_version_fields(record.payload)
    return {}


def latest_native_state_snapshot(
    telemetry_records: Iterable[TelemetryRecord],
    *,
    target_logical_id: str,
    observed_before_ms: int,
    max_age_ms: int = TELEMETRY_RECENT_THRESHOLD_MS,
) -> dict[str, Any] | None:
    latest_by_message: dict[str, TelemetryRecord] = {}
    for record in telemetry_records:
        if record.target_logical_id != target_logical_id:
            continue
        age_ms = observed_before_ms - record.ts_ms
        if age_ms < 0 or age_ms > max_age_ms:
            continue
        previous = latest_by_message.get(record.message_name)
        if previous is None or record.ts_ms >= previous.ts_ms:
            latest_by_message[record.message_name] = record
    if not latest_by_message:
        return None

    target_fields: dict[str, Any] = {}
    snapshot_observed_at_ms = max(record.ts_ms for record in latest_by_message.values())
    for record in latest_by_message.values():
        for key, value in state_fields_for_record(record).items():
            if value is not None:
                target_fields[key] = value
        target_fields[f"native_{record.message_name.lower()}"] = dict(record.payload)

    return {
        "target_logical_id": target_logical_id,
        "peer_logical_id": None,
        "snapshot_observed_at_ms": float(snapshot_observed_at_ms),
        "target_fields": target_fields,
        "peer_fields": {},
    }
