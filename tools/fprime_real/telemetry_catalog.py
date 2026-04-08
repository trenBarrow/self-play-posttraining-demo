#!/usr/bin/env python3
"""Checked-in telemetry catalog for observed F' deployment channels."""

from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class TelemetryCatalogEntry:
    channel_name: str
    feature_name: str
    kind: str
    unit_policy: str
    enabled_for_model: bool
    semantic_category: str
    canonical_dimensions: tuple[str, ...]


TELEMETRY_CATALOG = [
    TelemetryCatalogEntry(
        "systemResources.CPU",
        "cpu_total_pct",
        "gauge",
        "numeric",
        True,
        "compute_load",
        ("compute_load_ratio",),
    ),
    TelemetryCatalogEntry(
        "systemResources.CPU_00",
        "cpu_00_pct",
        "gauge",
        "numeric",
        True,
        "compute_load",
        ("compute_peak_load_ratio", "compute_imbalance_ratio"),
    ),
    TelemetryCatalogEntry(
        "systemResources.CPU_01",
        "cpu_01_pct",
        "gauge",
        "numeric",
        True,
        "compute_load",
        ("compute_peak_load_ratio", "compute_imbalance_ratio"),
    ),
    TelemetryCatalogEntry(
        "blockDrv.BD_Cycles",
        "blockdrv_cycles_total",
        "counter",
        "numeric",
        True,
        "storage_io_pressure",
        ("storage_io_pressure_ratio",),
    ),
    TelemetryCatalogEntry(
        "cmdDisp.CommandsDispatched",
        "cmds_dispatched_total",
        "counter",
        "numeric",
        True,
        "command_activity",
        ("command_activity_ratio",),
    ),
    TelemetryCatalogEntry(
        "cmdDisp.CommandErrors",
        "cmd_errors_total",
        "counter",
        "numeric",
        True,
        "command_error_pressure",
        ("command_error_ratio",),
    ),
    TelemetryCatalogEntry(
        "fileDownlink.Warnings",
        "filedownlink_warnings_total",
        "counter",
        "numeric",
        True,
        "service_issue_pressure",
        ("service_issue_ratio",),
    ),
    TelemetryCatalogEntry(
        "fileManager.Errors",
        "filemanager_errors_total",
        "counter",
        "numeric",
        True,
        "service_issue_pressure",
        ("service_issue_ratio",),
    ),
    TelemetryCatalogEntry(
        "fileUplinkBufferManager.HiBuffs",
        "hibuffs_total",
        "counter",
        "numeric",
        True,
        "queue_pressure",
        ("queue_pressure_ratio",),
    ),
    TelemetryCatalogEntry(
        "rateGroup1.RgMaxTime",
        "rg1_max_time_ms",
        "duration_ms",
        "duration_to_ms",
        True,
        "scheduler_pressure",
        ("scheduler_pressure_ratio",),
    ),
    TelemetryCatalogEntry(
        "rateGroup2.RgMaxTime",
        "rg2_max_time_ms",
        "duration_ms",
        "duration_to_ms",
        True,
        "scheduler_pressure",
        ("scheduler_pressure_ratio",),
    ),
    TelemetryCatalogEntry(
        "systemResources.FRAMEWORK_VERSION",
        "framework_version_text",
        "inventory_only",
        "text",
        False,
        "inventory_only",
        (),
    ),
    TelemetryCatalogEntry(
        "systemResources.PROJECT_VERSION",
        "project_version_text",
        "inventory_only",
        "text",
        False,
        "inventory_only",
        (),
    ),
]

TELEMETRY_BY_CHANNEL = {entry.channel_name: entry for entry in TELEMETRY_CATALOG}
TELEMETRY_BY_FEATURE = {entry.feature_name: entry for entry in TELEMETRY_CATALOG}
MODELED_TELEMETRY_ENTRIES = [entry for entry in TELEMETRY_CATALOG if entry.enabled_for_model]
MODELED_TELEMETRY_FIELDS = [entry.feature_name for entry in MODELED_TELEMETRY_ENTRIES]
COUNTER_TELEMETRY_FIELDS = {
    entry.feature_name for entry in MODELED_TELEMETRY_ENTRIES if entry.kind == "counter"
}
MODELED_TELEMETRY_SEMANTIC_CATEGORIES = sorted({entry.semantic_category for entry in MODELED_TELEMETRY_ENTRIES})
MODELED_CANONICAL_STATE_DIMENSIONS = sorted(
    {
        dimension
        for entry in MODELED_TELEMETRY_ENTRIES
        for dimension in entry.canonical_dimensions
    }
)


def parse_numeric_value(raw_value: str) -> tuple[float, str] | None:
    text = raw_value.strip()
    if not text:
        return None
    parts = text.split()
    try:
        value = float(parts[0])
    except ValueError:
        return None
    unit = parts[1] if len(parts) > 1 else ""
    return value, unit


def convert_numeric_entry(entry: TelemetryCatalogEntry, raw_value: str) -> float | None:
    numeric = parse_numeric_value(raw_value)
    if numeric is None:
        return None
    value, unit = numeric
    if entry.unit_policy == "duration_to_ms":
        if unit == "us":
            value /= 1000.0
        elif unit == "ns":
            value /= 1_000_000.0
        elif unit == "s":
            value *= 1000.0
    return float(value)


def catalog_bucket(channel_name: str) -> str:
    entry = TELEMETRY_BY_CHANNEL.get(channel_name)
    if entry is None:
        return "unknown"
    if not entry.enabled_for_model or entry.kind == "inventory_only":
        return "inventory_only"
    return "modeled"
