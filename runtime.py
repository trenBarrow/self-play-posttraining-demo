#!/usr/bin/env python3
from __future__ import annotations

import json
import heapq
import math
from collections import deque
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from tools.fprime_real.telemetry_catalog import COUNTER_TELEMETRY_FIELDS, MODELED_TELEMETRY_FIELDS

SERVICE_NAMES = [
    "cmdDisp",
    "cmdSeq",
    "eventLogger",
    "prmDb",
    "fileManager",
    "fileDownlink",
    "systemResources",
    "blockDrv",
    "fileUplinkBufferManager",
    "rateGroup1",
    "rateGroup2",
    "gds",
    "ops",
    "red",
]
SERVICE_IDS = {name: idx + 1 for idx, name in enumerate(SERVICE_NAMES)}

CLASS_NAMES = ["benign", "cyber", "fault"]
CLASS_IDS = {name: idx for idx, name in enumerate(CLASS_NAMES)}
SCHEMA_VERSION = "real_fprime_v2"
NODE_SERVICE_IDS = {
    "fprime_a": 1,
    "fprime_b": 2,
}
BACKGROUND_DEFAULTS = {
    "queue_depth": 0.0,
    "battery_soc": 82.0,
    "bus_voltage_v": 27.8,
    "solar_array_current_a": 3.6,
    "battery_temp_c": 16.0,
    "payload_temp_c": 2.0,
    "attitude_error_deg": 0.35,
    "wheel_speed_rpm": 1500.0,
    "link_rssi_dbm": -96.0,
    "downlink_backlog_mb": 25.0,
    "storage_free_mb": 6144.0,
    "event_backlog": 12.0,
    "radiation_flux": 1.4,
    "payload_cover_open": 0.0,
    "heater_on": 1.0,
    "comms_window_open": 0.0,
}
RAW_NODE_TELEMETRY_FIELDS = [
    "cpu_total_pct",
    "cpu_00_pct",
    "cpu_01_pct",
    "blockdrv_cycles_total",
    "cmds_dispatched_total",
    "cmd_errors_total",
    "filedownlink_warnings_total",
    "filemanager_errors_total",
    "hibuffs_total",
    "rg1_max_time_ms",
    "rg2_max_time_ms",
]
missing_catalog_fields = set(RAW_NODE_TELEMETRY_FIELDS) - set(MODELED_TELEMETRY_FIELDS)
if missing_catalog_fields:
    raise RuntimeError(f"Telemetry catalog is missing modeled fields required by runtime: {sorted(missing_catalog_fields)}")
NODE_TELEMETRY_FIELDS = [
    "cpu_total_pct",
    "cpu_00_pct",
    "cpu_01_pct",
    "blockdrv_cycles_total",
    "blockdrv_cycles_1m",
    "cmds_dispatched_total",
    "cmds_dispatched_1m",
    "cmd_errors_total",
    "cmd_errors_1m",
    "filedownlink_warnings_total",
    "filedownlink_warnings_1m",
    "filemanager_errors_total",
    "filemanager_errors_1m",
    "hibuffs_total",
    "rg1_max_time_ms",
    "rg2_max_time_ms",
    "telemetry_age_ms",
]
NODE_TELEMETRY_DEFAULTS = {field: 0.0 for field in NODE_TELEMETRY_FIELDS}
COUNTER_TOTAL_TO_RATE = {
    "blockdrv_cycles_total": "blockdrv_cycles_1m",
    "cmds_dispatched_total": "cmds_dispatched_1m",
    "cmd_errors_total": "cmd_errors_1m",
    "filedownlink_warnings_total": "filedownlink_warnings_1m",
    "filemanager_errors_total": "filemanager_errors_1m",
}
missing_counter_fields = set(COUNTER_TOTAL_TO_RATE) - set(COUNTER_TELEMETRY_FIELDS)
if missing_counter_fields:
    raise RuntimeError(f"Telemetry catalog is missing counter fields required by runtime: {sorted(missing_counter_fields)}")


@dataclass(frozen=True)
class CommandSpec:
    name: str
    service: str
    arg_bounds: dict[str, tuple[float, float, float]]
    request_bytes: int
    response_bytes: int
    base_latency_ms: int
    timeout_ms: int = 1200


COMMAND_SPECS: dict[str, CommandSpec] = {
    "cmdDisp.CMD_NO_OP": CommandSpec("cmdDisp.CMD_NO_OP", "cmdDisp", {}, 52, 68, 40, 350),
    "cmdDisp.CMD_NO_OP_STRING": CommandSpec("cmdDisp.CMD_NO_OP_STRING", "cmdDisp", {"token_id": (0, 99999, 100)}, 60, 76, 48, 350),
    "cmdDisp.CMD_TEST_CMD_1": CommandSpec(
        "cmdDisp.CMD_TEST_CMD_1",
        "cmdDisp",
        {"i32_arg": (-2000, 2000, 0), "f32_arg": (-200.0, 200.0, 0.0), "u8_arg": (0, 255, 1)},
        68,
        88,
        75,
        600,
    ),
    "cmdDisp.CMD_UNKNOWN_PROBE": CommandSpec("cmdDisp.CMD_UNKNOWN_PROBE", "cmdDisp", {}, 52, 68, 60, 500),
    "cmdDisp.CMD_CLEAR_TRACKING": CommandSpec("cmdDisp.CMD_CLEAR_TRACKING", "cmdDisp", {}, 56, 72, 70, 500),
    "cmdSeq.CS_AUTO": CommandSpec("cmdSeq.CS_AUTO", "cmdSeq", {}, 52, 68, 55, 500),
    "cmdSeq.CS_MANUAL": CommandSpec("cmdSeq.CS_MANUAL", "cmdSeq", {}, 52, 68, 55, 500),
    "cmdSeq.CS_STEP": CommandSpec("cmdSeq.CS_STEP", "cmdSeq", {}, 52, 72, 85, 700),
    "cmdSeq.CS_START": CommandSpec("cmdSeq.CS_START", "cmdSeq", {}, 52, 76, 90, 800),
    "cmdSeq.CS_VALIDATE": CommandSpec("cmdSeq.CS_VALIDATE", "cmdSeq", {"path_id": (0, 4096, 1)}, 84, 74, 75, 700),
    "cmdSeq.CS_CANCEL": CommandSpec("cmdSeq.CS_CANCEL", "cmdSeq", {}, 52, 68, 60, 500),
    "cmdSeq.CS_JOIN_WAIT": CommandSpec("cmdSeq.CS_JOIN_WAIT", "cmdSeq", {}, 52, 72, 65, 550),
    "eventLogger.DUMP_FILTER_STATE": CommandSpec("eventLogger.DUMP_FILTER_STATE", "eventLogger", {}, 56, 92, 55, 500),
    "prmDb.PRM_SAVE_FILE": CommandSpec("prmDb.PRM_SAVE_FILE", "prmDb", {}, 56, 84, 110, 900),
    "systemResources.VERSION": CommandSpec("systemResources.VERSION", "systemResources", {}, 56, 84, 50, 450),
    "fileManager.CreateDirectory": CommandSpec("fileManager.CreateDirectory", "fileManager", {"path_id": (0, 4096, 12)}, 64, 76, 95, 850),
    "fileManager.ShellCommand": CommandSpec("fileManager.ShellCommand", "fileManager", {"op_id": (0, 4096, 20)}, 72, 88, 180, 1500),
    "fileManager.FileSize": CommandSpec("fileManager.FileSize", "fileManager", {"path_id": (0, 4096, 16)}, 64, 80, 85, 700),
    "fileManager.AppendFile": CommandSpec(
        "fileManager.AppendFile",
        "fileManager",
        {"src_id": (0, 4096, 18), "dst_id": (0, 4096, 19)},
        72,
        84,
        105,
        950,
    ),
    "fileManager.MoveFile": CommandSpec(
        "fileManager.MoveFile",
        "fileManager",
        {"src_id": (0, 4096, 18), "dst_id": (0, 4096, 24)},
        72,
        84,
        95,
        900,
    ),
    "fileManager.RemoveFile": CommandSpec("fileManager.RemoveFile", "fileManager", {"path_id": (0, 4096, 21)}, 64, 76, 90, 850),
    "fileManager.RemoveDirectory": CommandSpec("fileManager.RemoveDirectory", "fileManager", {"path_id": (0, 4096, 21)}, 64, 76, 95, 850),
    "fileDownlink.SendPartial": CommandSpec(
        "fileDownlink.SendPartial",
        "fileDownlink",
        {"file_id": (0, 4096, 20), "offset_kb": (0, 8192, 0), "length_kb": (1, 8192, 1024)},
        76,
        112,
        170,
        1600,
    ),
    "fileDownlink.SendFile": CommandSpec(
        "fileDownlink.SendFile",
        "fileDownlink",
        {"file_id": (0, 4096, 20), "dest_id": (0, 4096, 20)},
        72,
        104,
        200,
        1800,
    ),
    "fileDownlink.Cancel": CommandSpec("fileDownlink.Cancel", "fileDownlink", {}, 52, 68, 60, 600),
}
COMMAND_NAMES = list(COMMAND_SPECS)
COMMAND_IDS = {name: idx + 1 for idx, name in enumerate(COMMAND_NAMES)}

FEATURE_NAMES = [
    "service_id",
    "command_id",
    "target_node_id",
    "actor_trust",
    "arg_count",
    "arg_norm",
    "arg_out_of_range",
    "req_bytes",
    "resp_bytes",
    "latency_ms",
    "gds_accept",
    "sat_success",
    "timeout",
    "response_code",
    "request_to_uplink_ms",
    "uplink_to_sat_response_ms",
    "sat_response_to_final_ms",
    "response_direction_seen",
    "final_observed_on_wire",
    "txn_warning_events",
    "txn_error_events",
    *[f"target_{field}" for field in NODE_TELEMETRY_FIELDS],
    *[f"peer_{field}" for field in NODE_TELEMETRY_FIELDS],
    "packet_gap_ms",
    "command_rate_1m",
    "error_rate_1m",
]
REQUEST_CONTEXT_FEATURE_NAMES = [
    "service_id",
    "command_id",
    "target_node_id",
    "actor_trust",
    "arg_count",
    "arg_norm",
    "arg_out_of_range",
    "req_bytes",
]
REQUEST_TELEMETRY_FEATURE_NAMES = [
    *[f"target_{field}" for field in NODE_TELEMETRY_FIELDS],
    *[f"peer_{field}" for field in NODE_TELEMETRY_FIELDS],
]
REQUEST_FEATURE_NAMES = [
    *REQUEST_CONTEXT_FEATURE_NAMES,
    *REQUEST_TELEMETRY_FEATURE_NAMES,
]
TERMINAL_OUTCOME_FEATURE_NAMES = [
    "resp_bytes",
    "latency_ms",
    "gds_accept",
    "sat_success",
    "timeout",
    "response_code",
    "request_to_uplink_ms",
    "uplink_to_sat_response_ms",
    "sat_response_to_final_ms",
    "response_direction_seen",
    "final_observed_on_wire",
    "txn_warning_events",
    "txn_error_events",
]
RESPONSE_FEATURE_NAMES = list(TERMINAL_OUTCOME_FEATURE_NAMES)
ROLLING_CONTEXT_FEATURE_NAMES = [
    "packet_gap_ms",
    "command_rate_1m",
    "error_rate_1m",
]
HISTORY_FEATURE_NAMES = list(ROLLING_CONTEXT_FEATURE_NAMES)
FORBIDDEN_PRIMARY_MODEL_FEATURE_NAMES = set(RESPONSE_FEATURE_NAMES) | set(HISTORY_FEATURE_NAMES)
PRIMARY_MODEL_FEATURE_TIER = "primary_model"
NOVELTY_MODEL_FEATURE_TIER = "novelty_model"
PRIMARY_MODEL_FEATURE_NAMES = list(REQUEST_FEATURE_NAMES)
NOVELTY_FEATURE_NAMES = [*REQUEST_FEATURE_NAMES, *HISTORY_FEATURE_NAMES]

DEFAULT_MODEL_DIR = Path(__file__).resolve().parent / "deployments" / "DetectorRB3" / "config"
SENSITIVE_SERVICES = {"cmdSeq", "fileManager", "fileDownlink", "prmDb"}
DEFAULT_CALIBRATOR_FEATURES = ["panomaly", "pcyber", "rules", "novelty"]
FEATURE_TIER_FEATURE_NAMES = {
    "request": list(REQUEST_FEATURE_NAMES),
    "response": list(RESPONSE_FEATURE_NAMES),
    "history": list(HISTORY_FEATURE_NAMES),
    PRIMARY_MODEL_FEATURE_TIER: list(PRIMARY_MODEL_FEATURE_NAMES),
    NOVELTY_MODEL_FEATURE_TIER: list(NOVELTY_FEATURE_NAMES),
}
MODEL_FEATURE_LAYOUTS = {
    "random_forest": {"feature_tier": PRIMARY_MODEL_FEATURE_TIER, "feature_names": list(PRIMARY_MODEL_FEATURE_NAMES)},
    "neural_net": {"feature_tier": PRIMARY_MODEL_FEATURE_TIER, "feature_names": list(PRIMARY_MODEL_FEATURE_NAMES)},
    "novelty": {"feature_tier": NOVELTY_MODEL_FEATURE_TIER, "feature_names": list(NOVELTY_FEATURE_NAMES)},
    "calibrator": {"feature_tier": "calibrator_inputs", "feature_names": list(DEFAULT_CALIBRATOR_FEATURES)},
}
RUNTIME_BUNDLE_SCHEMA_VERSION = "detector_runtime_bundle.v1"
RUNTIME_BUNDLE_MANIFEST_NAME = "bundle_manifest.json"
POSTER_BLUE_RUNTIME_KIND = "poster_blue_single_model_v1"
POSTER_BLUE_RUNTIME_API = "poster_blue_runtime.v1"
POSTER_BLUE_MODEL_ARTIFACT_NAME = "blue_model.json"
LEGACY_STACKED_RUNTIME_KIND = "legacy_fprime_stacked_v1"
LEGACY_STACKED_RUNTIME_API = "legacy_runtime_bundle.v1"
LEGACY_PRIMARY_MODEL_ARTIFACT_NAME = "model.json"


def _runtime_score_output_names(*, include_legacy_components: bool) -> list[str]:
    outputs = [
        "pbenign",
        "pcyber",
        "pfault",
        "panomaly",
        "anomaly_score",
        "unsafe_risk",
        "risk",
        "predicted_class",
        "detector_reason",
    ]
    if include_legacy_components:
        outputs.extend(["rules", "novelty"])
    return outputs


def build_runtime_bundle_manifest(
    *,
    runtime_kind: str,
    runtime_api: str,
    training_path: str,
    primary_model_artifact: str,
    feature_tier: str,
    uses_rules: bool,
    uses_novelty: bool,
    uses_calibrator: bool,
    extra_artifacts: dict[str, str] | None = None,
) -> dict[str, Any]:
    manifest = {
        "bundle_schema_version": RUNTIME_BUNDLE_SCHEMA_VERSION,
        "runtime_kind": str(runtime_kind),
        "runtime_api": str(runtime_api),
        "training_path": str(training_path),
        "primary_model_artifact": str(primary_model_artifact),
        "feature_tier": str(feature_tier),
        "class_names": list(CLASS_NAMES),
        "score_outputs": _runtime_score_output_names(
            include_legacy_components=bool(uses_rules or uses_novelty or uses_calibrator)
        ),
        "uses_rules": bool(uses_rules),
        "uses_novelty": bool(uses_novelty),
        "uses_calibrator": bool(uses_calibrator),
    }
    if extra_artifacts:
        manifest["extra_artifacts"] = dict(extra_artifacts)
    return manifest


def build_poster_blue_runtime_manifest(
    *,
    training_path: str,
    feature_tier: str,
    primary_model_artifact: str = POSTER_BLUE_MODEL_ARTIFACT_NAME,
) -> dict[str, Any]:
    return build_runtime_bundle_manifest(
        runtime_kind=POSTER_BLUE_RUNTIME_KIND,
        runtime_api=POSTER_BLUE_RUNTIME_API,
        training_path=training_path,
        primary_model_artifact=primary_model_artifact,
        feature_tier=feature_tier,
        uses_rules=False,
        uses_novelty=False,
        uses_calibrator=False,
    )


def build_legacy_runtime_manifest(
    *,
    training_path: str,
    feature_tier: str,
    primary_model_artifact: str = LEGACY_PRIMARY_MODEL_ARTIFACT_NAME,
) -> dict[str, Any]:
    return build_runtime_bundle_manifest(
        runtime_kind=LEGACY_STACKED_RUNTIME_KIND,
        runtime_api=LEGACY_STACKED_RUNTIME_API,
        training_path=training_path,
        primary_model_artifact=primary_model_artifact,
        feature_tier=feature_tier,
        uses_rules=True,
        uses_novelty=True,
        uses_calibrator=True,
        extra_artifacts={
            "random_forest": "forest.json",
            "neural_net": "nn.json",
            "novelty": "novelty.cfg",
            "calibrator": "calibrator.json",
            "random_forest_calibrator": "calibrator_rf.json",
            "neural_net_calibrator": "calibrator_nn.json",
        },
    )


def _validate_feature_tiers() -> None:
    feature_name_set = set(FEATURE_NAMES)
    tier_sources = {
        "request": REQUEST_FEATURE_NAMES,
        "response": RESPONSE_FEATURE_NAMES,
        "history": HISTORY_FEATURE_NAMES,
    }
    assigned: set[str] = set()
    for tier_name, feature_names in tier_sources.items():
        current = set(feature_names)
        unknown = current - feature_name_set
        if unknown:
            raise RuntimeError(f"Feature tier {tier_name} contains unknown features: {sorted(unknown)}")
        overlap = assigned & current
        if overlap:
            raise RuntimeError(f"Feature tiers overlap for {tier_name}: {sorted(overlap)}")
        assigned.update(current)
    missing = feature_name_set - assigned
    if missing:
        raise RuntimeError(f"Feature tiers do not classify all dataset features: {sorted(missing)}")
    if set(REQUEST_FEATURE_NAMES) != set(REQUEST_CONTEXT_FEATURE_NAMES) | set(REQUEST_TELEMETRY_FEATURE_NAMES):
        raise RuntimeError("REQUEST_FEATURE_NAMES must contain exactly request-context and request-telemetry fields")
    if set(RESPONSE_FEATURE_NAMES) != set(TERMINAL_OUTCOME_FEATURE_NAMES):
        raise RuntimeError("RESPONSE_FEATURE_NAMES must contain exactly terminal outcome fields")
    if set(HISTORY_FEATURE_NAMES) != set(ROLLING_CONTEXT_FEATURE_NAMES):
        raise RuntimeError("HISTORY_FEATURE_NAMES must contain exactly rolling context fields")
    if set(PRIMARY_MODEL_FEATURE_NAMES) != set(REQUEST_FEATURE_NAMES):
        raise RuntimeError("PRIMARY_MODEL_FEATURE_NAMES must match request-scope features exactly")
    forbidden = set(PRIMARY_MODEL_FEATURE_NAMES) & FORBIDDEN_PRIMARY_MODEL_FEATURE_NAMES
    if forbidden:
        raise RuntimeError(f"PRIMARY_MODEL_FEATURE_NAMES contains forbidden response/history features: {sorted(forbidden)}")
    if set(NOVELTY_FEATURE_NAMES) != set(REQUEST_FEATURE_NAMES) | set(HISTORY_FEATURE_NAMES):
        raise RuntimeError("NOVELTY_FEATURE_NAMES must combine request and history tiers exactly")


_validate_feature_tiers()


def history_state_reset_mode(reset_key: str | None) -> str:
    if not reset_key:
        return "continuous"
    label = str(reset_key).strip()
    if not label:
        return "continuous"
    if label.endswith("_id"):
        label = label[:-3]
    return f"per_{label}"


def clip(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))


def sigmoid(z: float | np.ndarray) -> float | np.ndarray:
    return 1.0 / (1.0 + np.exp(-z))


def softmax(values: np.ndarray) -> np.ndarray:
    shifted = values - np.max(values)
    exp_values = np.exp(shifted)
    total = np.sum(exp_values)
    if total <= 0.0:
        return np.full_like(values, 1.0 / max(1, values.size), dtype=float)
    return exp_values / total


def class_name(label: int) -> str:
    idx = int(label)
    return CLASS_NAMES[idx] if 0 <= idx < len(CLASS_NAMES) else "unknown"


def save_json(path: str | Path, payload: Any) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        json.dump(payload, handle, indent=2)


def stable_token_id(value: Any, modulo: int = 4096) -> float:
    text = str(value)
    size = max(1, int(modulo))
    acc = 0
    for char in text:
        acc = (acc * 131 + ord(char)) % size
    return float(acc)


def normalize_command_args(command_name: str, raw_args: Any) -> dict[str, float]:
    if isinstance(raw_args, dict):
        normalized: dict[str, float] = {}
        for key, value in raw_args.items():
            try:
                normalized[str(key)] = float(value)
            except (TypeError, ValueError):
                continue
        return normalized
    if not isinstance(raw_args, (list, tuple)):
        return {}

    parts = [str(item) for item in raw_args]

    def num(index: int, default: float = 0.0) -> float:
        if index >= len(parts):
            return float(default)
        try:
            return float(parts[index])
        except (TypeError, ValueError):
            return float(default)

    if command_name == "cmdDisp.CMD_NO_OP_STRING":
        return {"token_id": stable_token_id(parts[0] if parts else "", 100000)}
    if command_name == "cmdDisp.CMD_TEST_CMD_1":
        return {
            "i32_arg": num(0, 0.0),
            "f32_arg": num(1, 0.0),
            "u8_arg": num(2, 0.0),
        }
    if command_name == "cmdSeq.CS_VALIDATE":
        return {"path_id": stable_token_id(parts[0] if parts else "", 4096)}
    if command_name == "fileManager.CreateDirectory":
        return {"path_id": stable_token_id(parts[0] if parts else "", 4096)}
    if command_name == "fileManager.ShellCommand":
        return {"op_id": stable_token_id("|".join(parts), 4096)}
    if command_name == "fileManager.FileSize":
        return {"path_id": stable_token_id(parts[0] if parts else "", 4096)}
    if command_name in {"fileManager.RemoveFile", "fileManager.RemoveDirectory"}:
        return {"path_id": stable_token_id(parts[0] if parts else "", 4096)}
    if command_name in {"fileManager.AppendFile", "fileManager.MoveFile"}:
        return {
            "src_id": stable_token_id(parts[0] if len(parts) > 0 else "", 4096),
            "dst_id": stable_token_id(parts[1] if len(parts) > 1 else "", 4096),
        }
    if command_name == "fileDownlink.SendPartial":
        source = parts[0] if len(parts) > 0 else ""
        dest = parts[1] if len(parts) > 1 else ""
        return {
            "file_id": stable_token_id(f"{source}|{dest}", 4096),
            "offset_kb": num(2, 0.0),
            "length_kb": num(3, 1024.0),
        }
    if command_name == "fileDownlink.SendFile":
        return {
            "file_id": stable_token_id(parts[0] if len(parts) > 0 else "", 4096),
            "dest_id": stable_token_id(parts[1] if len(parts) > 1 else "", 4096),
        }
    return {}


def vector_from_row(row: dict[str, Any], feature_names: list[str] | None = None) -> np.ndarray:
    names = feature_names or FEATURE_NAMES
    return np.array([float(row.get(name, 0.0)) for name in names], dtype=float)


def inspect_args(command_name: str, args: dict[str, Any]) -> tuple[float, float, float, float, str]:
    spec = COMMAND_SPECS.get(command_name)
    if spec is None:
        return float(len(args)), 4.0 + float(len(args)), 1.0, 1.0, "unknown_command"
    expected = set(spec.arg_bounds)
    seen = set(args)
    if seen != expected:
        delta = abs(len(expected) - len(seen)) + len(seen - expected)
        return float(len(args)), 2.0 + float(delta), 1.0, 1.0, "schema_mismatch"
    arg_norm = 0.0
    max_violation = 0.0
    for key, (low, high, default) in spec.arg_bounds.items():
        value = float(args[key])
        span = max(1.0, high - low if high != low else max(abs(default), 1.0))
        arg_norm += abs((value - default) / span)
        if value < low:
            max_violation = max(max_violation, (low - value) / span)
        elif value > high:
            max_violation = max(max_violation, (value - high) / span)
    return float(len(args)), float(arg_norm), 1.0 if max_violation > 0.0 else 0.0, float(max_violation), "ok"


def packet_target_stream_id(packet: dict[str, Any]) -> str:
    explicit = str(packet.get("target_stream_id", "")).strip()
    if explicit:
        return explicit
    target_service = str(packet.get("target_service", packet.get("dst", packet.get("src", "")))).strip()
    port_value = packet.get("target_tts_port")
    if port_value in (None, ""):
        kind = str(packet.get("packet_kind", ""))
        if kind == "request":
            port_value = packet.get("dst_port", "")
        elif kind in {"sat_response", "final"}:
            port_value = packet.get("src_port", "")
    try:
        port = int(port_value or 50050)
    except (TypeError, ValueError):
        port = 50050
    return f"{target_service}:{port}" if target_service else ""


def packet_target_stream_index(packet: dict[str, Any]) -> int | None:
    value = packet.get("target_stream_index")
    if value in (None, ""):
        return None
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


class NodeTelemetryState:
    def __init__(self):
        self.values = {field: float(default) for field, default in NODE_TELEMETRY_DEFAULTS.items()}
        self.last_seen_ms = 0.0
        self.counter_history: dict[str, deque[tuple[float, float]]] = {
            total_field: deque() for total_field in COUNTER_TOTAL_TO_RATE
        }

    def update(self, field_name: str, value: float, ts_ms: float) -> None:
        self.values[field_name] = float(value)
        self.last_seen_ms = max(self.last_seen_ms, float(ts_ms))
        if field_name in self.counter_history:
            history = self.counter_history[field_name]
            history.append((float(ts_ms), float(value)))
            while history and ts_ms - history[0][0] > 60000.0:
                history.popleft()

    def snapshot(self, now_ms: float) -> dict[str, float]:
        values = {field: float(self.values.get(field, 0.0)) for field in NODE_TELEMETRY_FIELDS}
        values["telemetry_age_ms"] = max(0.0, now_ms - self.last_seen_ms) if self.last_seen_ms > 0.0 else 600000.0
        for total_field, rate_field in COUNTER_TOTAL_TO_RATE.items():
            current_value = float(self.values.get(total_field, 0.0))
            history = self.counter_history.get(total_field, deque())
            baseline = current_value
            while history and now_ms - history[0][0] > 60000.0:
                history.popleft()
            if history:
                baseline = float(history[0][1])
            values[rate_field] = max(0.0, current_value - baseline)
        return values


class MirrorTransactionAssembler:
    def __init__(self):
        self.node_state = {node_service: NodeTelemetryState() for node_service in NODE_SERVICE_IDS}
        self.pending: dict[tuple[str, str], dict[str, Any]] = {}
        self.pending_target_services: dict[str, tuple[str, str]] = {}
        self.pending_streams: dict[str, tuple[str, str]] = {}
        self.last_stream_index_by_stream: dict[str, int] = {}
        self.last_packet_ts: float | None = None

    def _packet_gap(self, now_ms: float) -> float:
        gap = 0.0 if self.last_packet_ts is None else max(0.0, now_ms - self.last_packet_ts)
        self.last_packet_ts = now_ms
        return float(gap)

    def _snapshot_nodes(self, target_service: str, now_ms: float) -> dict[str, float]:
        target_snapshot = self.node_state.get(target_service, NodeTelemetryState()).snapshot(now_ms)
        peer_service = "fprime_b" if target_service == "fprime_a" else "fprime_a"
        peer_snapshot = self.node_state.get(peer_service, NodeTelemetryState()).snapshot(now_ms)
        values = {}
        for field_name, value in target_snapshot.items():
            values[f"target_{field_name}"] = float(value)
        for field_name, value in peer_snapshot.items():
            values[f"peer_{field_name}"] = float(value)
        return values

    def feed_packet(self, packet: dict[str, Any]) -> list[dict[str, Any]]:
        now_ms = float(packet.get("ts_ms", 0.0))
        packet_gap_ms = self._packet_gap(now_ms)
        kind = str(packet.get("packet_kind", ""))
        if kind == "telemetry":
            node_service = str(packet.get("node_service", ""))
            payload = packet.get("payload", {}) or {}
            state = self.node_state.get(node_service)
            if state is None:
                return []
            for key, value in payload.items():
                if key in NODE_TELEMETRY_DEFAULTS:
                    state.update(str(key), float(value), now_ms)
            return []

        label = int(packet.get("label", 0))
        episode_label = int(packet.get("episode_label", label))
        episode_kind = str(packet.get("episode_kind", class_name(episode_label)))
        key = (str(packet.get("session_id", "")), str(packet.get("txn_id", "")))
        if kind == "request":
            target_service = str(packet.get("target_service", packet.get("dst", "")))
            target_stream_id = packet_target_stream_id(packet)
            target_stream_index = packet_target_stream_index(packet)
            if target_service:
                pending_key = self.pending_target_services.get(target_service)
                if pending_key is not None and pending_key != key:
                    raise SystemExit(
                        "Packet stream violates the serialized-per-target invariant: "
                        f"target_service={target_service} current_txn={packet.get('txn_id', '')} pending_txn={pending_key[1]}"
                    )
            if target_stream_id:
                pending_stream_key = self.pending_streams.get(target_stream_id)
                if pending_stream_key is not None and pending_stream_key != key:
                    raise SystemExit(
                        "Packet stream violates the serialized target-stream invariant: "
                        f"target_stream_id={target_stream_id} current_txn={packet.get('txn_id', '')} pending_txn={pending_stream_key[1]}"
                    )
                if target_stream_index is not None:
                    previous_index = self.last_stream_index_by_stream.get(target_stream_id)
                    if previous_index is not None and target_stream_index <= previous_index:
                        raise SystemExit(
                            "Packet stream violates monotonically increasing target_stream_index: "
                            f"target_stream_id={target_stream_id} current_index={target_stream_index} previous_index={previous_index}"
                        )
            tx = {
                "run_id": int(packet.get("run_id", -1)),
                "episode_id": int(packet.get("episode_id", -1)),
                "episode_label": episode_label,
                "episode_kind": episode_kind,
                "label": label,
                "label_name": class_name(label),
                "session_id": str(packet.get("session_id", "")),
                "txn_id": str(packet.get("txn_id", "")),
                "send_id": str(packet.get("send_id", "")),
                "target_stream_id": target_stream_id,
                "target_stream_index": float(target_stream_index) if target_stream_index is not None else -1.0,
                "attack_family": str(packet.get("attack_family", "none")),
                "phase": str(packet.get("phase", "")),
                "actor": str(packet.get("actor", "unknown")),
                "actor_role": str(packet.get("actor_role", "unknown")),
                "actor_trust": float(packet.get("actor_trust", 1.0)),
                "command": str(packet.get("command", "unknown")),
                "service": str(packet.get("service", "unknown")),
                "args": dict(packet.get("args", {}) or {}),
                "target_service": target_service,
                "target_node_id": float(NODE_SERVICE_IDS.get(target_service, 0)),
                "request_ts_ms": now_ms,
                "packet_gap_ms": float(packet_gap_ms),
                "req_bytes": float(packet.get("bytes_on_wire", 0.0)),
                "resp_bytes": 0.0,
                "gds_accept": 0.0,
                "sat_success": 0.0,
                "timeout": 0.0,
                "response_code": 0.0,
                "reason": "pending",
                "uplink_ts_ms": None,
                "sat_response_ts_ms": None,
                "final_ts_ms": None,
                "request_to_uplink_ms": 0.0,
                "uplink_to_sat_response_ms": 0.0,
                "sat_response_to_final_ms": 0.0,
                "response_direction_seen": 0.0,
                "final_observed_on_wire": 0.0,
                "txn_warning_events": 0.0,
                "txn_error_events": 0.0,
                **self._snapshot_nodes(target_service, now_ms),
            }
            self.pending[key] = tx
            if target_service:
                self.pending_target_services[target_service] = key
            if target_stream_id:
                self.pending_streams[target_stream_id] = key
            return []

        tx = self.pending.get(key)
        if tx is None:
            return []
        if kind == "uplink":
            tx["gds_accept"] = 1.0
            if tx.get("uplink_ts_ms") is None:
                tx["uplink_ts_ms"] = now_ms
            return []
        if kind == "sat_response":
            if tx.get("sat_response_ts_ms") is None:
                tx["sat_response_ts_ms"] = now_ms
            tx["sat_success"] = float(packet.get("sat_success", tx.get("sat_success", 0.0)))
            tx["response_code"] = float(packet.get("response_code", tx.get("response_code", 0.0)))
            tx["reason"] = str(packet.get("reason", tx.get("reason", "completed")))
            return []
        if kind == "final":
            tx["final_ts_ms"] = now_ms
            tx["latency_ms"] = max(0.0, now_ms - float(tx["request_ts_ms"]))
            tx["gds_accept"] = float(packet.get("gds_accept", tx.get("gds_accept", 0.0)))
            tx["sat_success"] = float(packet.get("sat_success", tx.get("sat_success", 0.0)))
            tx["timeout"] = float(packet.get("timeout", 0.0))
            tx["response_code"] = float(packet.get("response_code", tx.get("response_code", 0.0)))
            tx["reason"] = str(packet.get("reason", tx.get("reason", "completed")))
            tx["resp_bytes"] = float(packet.get("bytes_on_wire", tx.get("resp_bytes", 0.0)))
            tx["response_direction_seen"] = float(packet.get("response_direction_seen", 0.0))
            tx["final_observed_on_wire"] = float(packet.get("final_observed_on_wire", packet.get("observed_on_wire", 0.0)))
            tx["txn_warning_events"] = float(packet.get("txn_warning_events", 0.0))
            tx["txn_error_events"] = float(packet.get("txn_error_events", 0.0))
            uplink_ts = float(tx.get("uplink_ts_ms") or tx["request_ts_ms"])
            sat_response_ts = float(tx.get("sat_response_ts_ms") or uplink_ts)
            tx["request_to_uplink_ms"] = max(0.0, uplink_ts - float(tx["request_ts_ms"])) if tx.get("uplink_ts_ms") is not None else 0.0
            tx["uplink_to_sat_response_ms"] = max(0.0, sat_response_ts - uplink_ts) if tx.get("sat_response_ts_ms") is not None and tx.get("uplink_ts_ms") is not None else 0.0
            tx["sat_response_to_final_ms"] = max(0.0, now_ms - sat_response_ts) if tx.get("sat_response_ts_ms") is not None else max(0.0, now_ms - uplink_ts)
            done = dict(tx)
            self.pending.pop(key, None)
            target_service = str(tx.get("target_service", ""))
            if target_service:
                self.pending_target_services.pop(target_service, None)
            target_stream_id = str(tx.get("target_stream_id", ""))
            if target_stream_id:
                self.pending_streams.pop(target_stream_id, None)
                target_stream_index = int(float(tx.get("target_stream_index", -1.0)))
                if target_stream_index >= 0:
                    self.last_stream_index_by_stream[target_stream_id] = target_stream_index
            return [done]
        return []

    def flush(self) -> list[dict[str, Any]]:
        completed: list[dict[str, Any]] = []
        for key, tx in list(self.pending.items()):
            spec = COMMAND_SPECS.get(str(tx.get("command", "")))
            timeout_ms = float(spec.timeout_ms if spec is not None else 1200.0)
            final_ts_ms = float(tx.get("request_ts_ms", 0.0)) + timeout_ms
            tx["final_ts_ms"] = final_ts_ms
            tx["latency_ms"] = max(0.0, final_ts_ms - float(tx["request_ts_ms"]))
            tx["timeout"] = 1.0
            tx["response_code"] = float(tx.get("response_code", 3.0) or 3.0)
            tx["reason"] = str(tx.get("reason", "stream_timeout"))
            tx["request_to_uplink_ms"] = 0.0
            tx["uplink_to_sat_response_ms"] = 0.0
            tx["sat_response_to_final_ms"] = max(0.0, final_ts_ms - float(tx.get("uplink_ts_ms") or tx["request_ts_ms"]))
            completed.append(dict(tx))
            self.pending.pop(key, None)
            target_service = str(tx.get("target_service", ""))
            if target_service:
                self.pending_target_services.pop(target_service, None)
            target_stream_id = str(tx.get("target_stream_id", ""))
            if target_stream_id:
                self.pending_streams.pop(target_stream_id, None)
                target_stream_index = int(float(tx.get("target_stream_index", -1.0)))
                if target_stream_index >= 0:
                    self.last_stream_index_by_stream[target_stream_id] = target_stream_index
        return completed


class TransactionFeaturizer:
    def __init__(self):
        self.command_history: deque[float] = deque()
        self.error_history: deque[tuple[float, int]] = deque()
        self.pending_error_updates: list[tuple[float, int]] = []
        self.last_request_ts_ms: float | None = None

    def reset(self) -> None:
        self.command_history.clear()
        self.error_history.clear()
        self.pending_error_updates.clear()
        self.last_request_ts_ms = None

    def _trim(self, now_ms: float) -> None:
        while self.command_history and now_ms - self.command_history[0] > 60000.0:
            self.command_history.popleft()
        while self.error_history and now_ms - self.error_history[0][0] > 60000.0:
            self.error_history.popleft()

    def _apply_completed_errors(self, request_ts_ms: float) -> None:
        while self.pending_error_updates and self.pending_error_updates[0][0] <= request_ts_ms:
            final_ts_ms, error_flag = heapq.heappop(self.pending_error_updates)
            self.error_history.append((final_ts_ms, error_flag))
        self._trim(request_ts_ms)

    def _packet_gap(self, request_ts_ms: float) -> float:
        gap = 0.0 if self.last_request_ts_ms is None else max(0.0, request_ts_ms - self.last_request_ts_ms)
        self.last_request_ts_ms = request_ts_ms
        return float(gap)

    def row_from_transaction(self, tx: dict[str, Any]) -> dict[str, Any]:
        request_ts_ms = float(tx.get("request_ts_ms", tx.get("final_ts_ms", 0.0)))
        final_ts_ms = float(tx.get("final_ts_ms", request_ts_ms))
        self._apply_completed_errors(request_ts_ms)
        arg_count, arg_norm, arg_out_of_range, _, _ = inspect_args(str(tx.get("command", "")), dict(tx.get("args", {}) or {}))
        gds_accept = float(tx.get("gds_accept", 0.0))
        sat_success = float(tx.get("sat_success", 0.0))
        timeout = float(tx.get("timeout", 0.0))
        response_code = float(tx.get("response_code", 0.0))
        error_flag = 0 if gds_accept > 0.5 and sat_success > 0.5 and timeout < 0.5 and response_code == 0.0 else 1
        packet_gap_ms = self._packet_gap(request_ts_ms)
        self.command_history.append(request_ts_ms)
        self._trim(request_ts_ms)
        error_rate_1m = 0.0
        if self.error_history:
            error_rate_1m = float(sum(value for _, value in self.error_history) / len(self.error_history))
        service = str(tx.get("service", str(tx.get("command", "unknown")).split(".")[0]))
        label = int(tx.get("label", 0))
        row = {
            "label": label,
            "label_name": class_name(label),
            "run_id": int(tx.get("run_id", -1)),
            "episode_id": int(tx.get("episode_id", -1)),
            "episode_label": int(tx.get("episode_label", label)),
            "episode_kind": str(tx.get("episode_kind", class_name(tx.get("episode_label", label)))),
            "session_id": str(tx.get("session_id", "")),
            "txn_id": str(tx.get("txn_id", "")),
            "send_id": str(tx.get("send_id", "")),
            "target_stream_id": str(tx.get("target_stream_id", "")),
            "target_stream_index": int(float(tx.get("target_stream_index", -1.0))) if float(tx.get("target_stream_index", -1.0)) >= 0.0 else "",
            "attack_family": str(tx.get("attack_family", "none")),
            "phase": str(tx.get("phase", "")),
            "actor": str(tx.get("actor", "unknown")),
            "actor_role": str(tx.get("actor_role", "unknown")),
            "command": str(tx.get("command", "unknown")),
            "service": service,
            "reason": str(tx.get("reason", "completed")),
            "request_ts_ms": request_ts_ms,
            "final_ts_ms": final_ts_ms,
            "service_id": float(SERVICE_IDS.get(service, 0)),
            "command_id": float(COMMAND_IDS.get(str(tx.get("command", "")), 0)),
            "target_node_id": float(tx.get("target_node_id", 0.0)),
            "actor_trust": float(tx.get("actor_trust", 1.0)),
            "arg_count": float(arg_count),
            "arg_norm": float(arg_norm),
            "arg_out_of_range": float(arg_out_of_range),
            "req_bytes": float(tx.get("req_bytes", 0.0)),
            "resp_bytes": float(tx.get("resp_bytes", 0.0)),
            "latency_ms": float(tx.get("latency_ms", 0.0)),
            "gds_accept": gds_accept,
            "sat_success": sat_success,
            "timeout": timeout,
            "response_code": response_code,
            "request_to_uplink_ms": float(tx.get("request_to_uplink_ms", 0.0)),
            "uplink_to_sat_response_ms": float(tx.get("uplink_to_sat_response_ms", 0.0)),
            "sat_response_to_final_ms": float(tx.get("sat_response_to_final_ms", 0.0)),
            "response_direction_seen": float(tx.get("response_direction_seen", 0.0)),
            "final_observed_on_wire": float(tx.get("final_observed_on_wire", 0.0)),
            "txn_warning_events": float(tx.get("txn_warning_events", 0.0)),
            "txn_error_events": float(tx.get("txn_error_events", 0.0)),
            "packet_gap_ms": packet_gap_ms,
            "command_rate_1m": float(len(self.command_history)),
            "error_rate_1m": error_rate_1m,
        }
        for field_name in NODE_TELEMETRY_FIELDS:
            row[f"target_{field_name}"] = float(tx.get(f"target_{field_name}", NODE_TELEMETRY_DEFAULTS[field_name]))
            row[f"peer_{field_name}"] = float(tx.get(f"peer_{field_name}", NODE_TELEMETRY_DEFAULTS[field_name]))
        heapq.heappush(self.pending_error_updates, (final_ts_ms, error_flag))
        return row


def packets_to_transactions(packets: list[dict[str, Any]], reset_key: str | None = None) -> list[dict[str, Any]]:
    transactions: list[dict[str, Any]] = []
    for group_items in _group_records_for_history(packets, reset_key):
        assembler = MirrorTransactionAssembler()
        for _, packet in _sort_packet_group(group_items):
            transactions.extend(assembler.feed_packet(packet))
        transactions.extend(assembler.flush())
    return transactions


def _group_records_for_history(records: list[dict[str, Any]], reset_key: str | None) -> list[list[tuple[int, dict[str, Any]]]]:
    indexed_records = list(enumerate(records))
    if reset_key is None:
        return [indexed_records]
    grouped: dict[Any, list[tuple[int, dict[str, Any]]]] = {}
    group_order: list[Any] = []
    for index, record in indexed_records:
        group_value = record.get(reset_key)
        if group_value not in grouped:
            grouped[group_value] = []
            group_order.append(group_value)
        grouped[group_value].append((index, record))
    return [grouped[group_value] for group_value in group_order]


def _optional_float(value: Any) -> float | None:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except (TypeError, ValueError):
        return None


def _sort_history_group(items: list[tuple[int, dict[str, Any]]]) -> list[tuple[int, dict[str, Any]]]:
    if not all(_optional_float(record.get("request_ts_ms")) is not None for _, record in items):
        return items
    return sorted(
        items,
        key=lambda item: (
            float(item[1].get("request_ts_ms", 0.0)),
            float(item[1].get("final_ts_ms", item[1].get("request_ts_ms", 0.0))),
            str(item[1].get("txn_id", "")),
            item[0],
        ),
    )


def _sort_packet_group(items: list[tuple[int, dict[str, Any]]]) -> list[tuple[int, dict[str, Any]]]:
    packet_kind_order = {
        "telemetry": 0,
        "request": 1,
        "uplink": 2,
        "sat_response": 3,
        "final": 4,
    }
    if not all(_optional_float(record.get("ts_ms")) is not None for _, record in items):
        return items
    return sorted(
        items,
        key=lambda item: (
            float(item[1].get("ts_ms", 0.0)),
            str(item[1].get("txn_id", "")),
            packet_kind_order.get(str(item[1].get("packet_kind", "")), 99),
            item[0],
        ),
    )


def transactions_to_rows(transactions: list[dict[str, Any]], reset_key: str | None = None) -> list[dict[str, Any]]:
    """Convert transactions into feature rows.

    When ``reset_key`` is set, rolling history features are recomputed with a
    fresh state per group (for example ``episode_id`` during evaluation), so
    held-out groups do not inherit command/error history from other splits.
    """

    rows: list[dict[str, Any]] = []
    for group_items in _group_records_for_history(transactions, reset_key):
        featurizer = TransactionFeaturizer()
        for _, tx in _sort_history_group(group_items):
            rows.append(featurizer.row_from_transaction(tx))
    return rows


def replay_history_features(rows: list[dict[str, Any]], reset_key: str | None = None) -> list[dict[str, Any]]:
    """Recompute rolling history features for existing rows.

    This keeps request/response/telemetry values intact while rebuilding
    ``packet_gap_ms``, ``command_rate_1m``, and ``error_rate_1m`` from a fresh
    per-group history state.
    """

    replayed_rows: list[dict[str, Any]] = []
    for group_items in _group_records_for_history(rows, reset_key):
        featurizer = TransactionFeaturizer()
        last_request_ts_ms: float | None = None
        for _, row in _sort_history_group(group_items):
            request_ts_ms = _optional_float(row.get("request_ts_ms"))
            if request_ts_ms is None:
                if last_request_ts_ms is None:
                    request_ts_ms = 0.0
                else:
                    request_ts_ms = last_request_ts_ms + max(0.0, float(row.get("packet_gap_ms", 0.0)))
            final_ts_ms = _optional_float(row.get("final_ts_ms"))
            if final_ts_ms is None:
                final_ts_ms = request_ts_ms + max(0.0, float(row.get("latency_ms", 0.0)))
            synthetic_tx = {
                "request_ts_ms": request_ts_ms,
                "final_ts_ms": final_ts_ms,
                "command": row.get("command", "unknown"),
                "args": {},
                "gds_accept": row.get("gds_accept", 0.0),
                "sat_success": row.get("sat_success", 0.0),
                "timeout": row.get("timeout", 0.0),
                "response_code": row.get("response_code", 0.0),
            }
            history_row = featurizer.row_from_transaction(synthetic_tx)
            updated = dict(row)
            updated["request_ts_ms"] = request_ts_ms
            updated["final_ts_ms"] = final_ts_ms
            updated["packet_gap_ms"] = history_row["packet_gap_ms"]
            updated["command_rate_1m"] = history_row["command_rate_1m"]
            updated["error_rate_1m"] = history_row["error_rate_1m"]
            replayed_rows.append(updated)
            last_request_ts_ms = request_ts_ms
    return replayed_rows


def rule_hits(row: dict[str, Any]) -> list[tuple[float, str]]:
    hits: list[tuple[float, str]] = []
    if float(row.get("arg_out_of_range", 0.0)) > 0.5:
        hits.append((0.30, "arg_range"))
    if float(row.get("timeout", 0.0)) > 0.5:
        hits.append((0.35, "timeout"))
    if float(row.get("actor_trust", 1.0)) < 0.45 and float(row.get("command_rate_1m", 0.0)) > 5.0:
        hits.append((0.20, "low_trust_burst"))
    if max(float(row.get("target_cpu_total_pct", 0.0)), float(row.get("target_cpu_00_pct", 0.0)), float(row.get("target_cpu_01_pct", 0.0))) > 90.0:
        hits.append((0.18, "high_cpu"))
    if float(row.get("target_cmd_errors_1m", 0.0)) > 0.0:
        hits.append((0.22, "cmd_error_burst"))
    if float(row.get("target_filemanager_errors_1m", 0.0)) > 0.0:
        hits.append((0.22, "filemanager_error_burst"))
    if float(row.get("target_filedownlink_warnings_1m", 0.0)) > 0.0:
        hits.append((0.15, "filedownlink_warning_burst"))
    if max(float(row.get("target_rg1_max_time_ms", 0.0)), float(row.get("target_rg2_max_time_ms", 0.0))) > 10.0:
        hits.append((0.18, "rate_group_excursion"))
    if float(row.get("target_hibuffs_total", 0.0)) > 2.0:
        hits.append((0.12, "uplink_buffer_pressure"))
    if float(row.get("response_direction_seen", 0.0)) < 0.5 and float(row.get("timeout", 0.0)) < 0.5:
        hits.append((0.25, "missing_response_wire"))
    if float(row.get("target_telemetry_age_ms", 0.0)) > 5000.0:
        hits.append((0.15, "telemetry_stale"))
    if float(row.get("error_rate_1m", 0.0)) > 0.40:
        hits.append((0.20, "error_burst"))
    if float(row.get("txn_error_events", 0.0)) > 0.0:
        hits.append((0.18, "txn_error_event"))
    return hits


def rule_score(row: dict[str, Any]) -> float:
    return min(1.0, sum(score for score, _ in rule_hits(row)))


@dataclass
class GaussianNovelty:
    mu: np.ndarray
    sigma_inv: np.ndarray
    feature_names: list[str]
    cov: np.ndarray | None = None

    @staticmethod
    def fit(X: np.ndarray, feature_names: list[str], eps: float = 1e-6) -> "GaussianNovelty":
        if X.ndim != 2 or X.shape[0] < 2:
            raise ValueError("Need at least two nominal rows to fit novelty")
        mu = X.mean(axis=0)
        centered = X - mu
        cov = (centered.T @ centered) / max(1, X.shape[0] - 1)
        cov = cov + eps * np.eye(cov.shape[0])
        sigma_inv = np.linalg.pinv(cov)
        return GaussianNovelty(mu=mu, sigma_inv=sigma_inv, feature_names=list(feature_names), cov=cov)

    def score(self, x: np.ndarray) -> float:
        delta = x.reshape(-1) - self.mu.reshape(-1)
        q = float(delta.T @ self.sigma_inv @ delta)
        value = 1.0 - math.exp(-0.5 * max(0.0, q))
        return float(min(max(value, 0.0), 0.999999))

    def update_for_drift(self, x: np.ndarray, alpha: float) -> None:
        if alpha <= 0.0:
            return
        cov = self.cov if self.cov is not None else np.linalg.pinv(self.sigma_inv)
        x = x.reshape(-1)
        old_mu = self.mu.copy()
        self.mu = (1.0 - alpha) * self.mu + alpha * x
        delta = (x - old_mu).reshape(-1, 1)
        cov = (1.0 - alpha) * cov + alpha * (delta @ delta.T)
        cov = cov + 1e-6 * np.eye(cov.shape[0])
        self.cov = cov
        self.sigma_inv = np.linalg.pinv(cov)

    def dump_cfg(self, path: str | Path) -> None:
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        lines = [
            "feature_tier=" + NOVELTY_MODEL_FEATURE_TIER,
            "feature_names=" + ",".join(self.feature_names),
            "mu=" + ",".join(f"{value:.12g}" for value in self.mu.tolist()),
            f"sigma_inv_rows={self.sigma_inv.shape[0]}",
        ]
        for row_idx, row in enumerate(self.sigma_inv.tolist()):
            lines.append(f"sigma_inv_{row_idx}=" + ",".join(f"{value:.12g}" for value in row))
        target.write_text("\n".join(lines) + "\n", encoding="utf-8")

    @staticmethod
    def load(path: str | Path) -> "GaussianNovelty":
        target = Path(path)
        if target.suffix == ".json":
            with target.open("r", encoding="utf-8") as handle:
                payload = json.load(handle)
            mu = np.array(payload["mu"], dtype=float)
            sigma_inv = np.array(payload["sigma_inv"], dtype=float)
            return GaussianNovelty(mu=mu, sigma_inv=sigma_inv, feature_names=list(payload.get("feature_names", NOVELTY_FEATURE_NAMES)))
        raw: dict[str, str] = {}
        with target.open("r", encoding="utf-8") as handle:
            for line in handle:
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                key, value = line.split("=", 1)
                raw[key.strip()] = value.strip()
        feature_names = [item for item in raw.get("feature_names", "").split(",") if item]
        mu = np.array([float(item) for item in raw.get("mu", "").split(",") if item], dtype=float)
        rows = int(raw.get("sigma_inv_rows", "0"))
        sigma_rows = []
        for row_idx in range(rows):
            sigma_rows.append([float(item) for item in raw.get(f"sigma_inv_{row_idx}", "").split(",") if item])
        sigma_inv = np.array(sigma_rows, dtype=float)
        if mu.size == 0 or sigma_inv.size == 0:
            raise ValueError(f"Invalid novelty config: {target}")
        return GaussianNovelty(mu=mu, sigma_inv=sigma_inv, feature_names=feature_names or NOVELTY_FEATURE_NAMES)


@dataclass
class Calibrator:
    weights: np.ndarray
    bias: float
    feature_names: list[str]

    def score(self, panomaly: float, pcyber: float, rules: float, novelty: float) -> float:
        values = {"panomaly": panomaly, "pcyber": pcyber, "rules": rules, "novelty": novelty}
        x = np.array([values[name] for name in self.feature_names], dtype=float)
        return float(sigmoid(self.weights.reshape(-1) @ x + self.bias))

    def dump_json(self, path: str | Path) -> None:
        save_json(
            path,
            {
                "weights": self.weights.reshape(-1).tolist(),
                "bias": float(self.bias),
                "feature_names": list(self.feature_names),
            },
        )

    @staticmethod
    def load(path: str | Path) -> "Calibrator":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        weights = np.array(payload["weights"], dtype=float)
        feature_names = payload.get("feature_names")
        if not feature_names:
            feature_names = ["pcyber", "rules", "novelty"] if weights.size == 3 else DEFAULT_CALIBRATOR_FEATURES
        return Calibrator(weights=weights, bias=float(payload.get("bias", 0.0)), feature_names=list(feature_names))


@dataclass
class Node:
    feature: int
    threshold: float
    left: int
    right: int
    is_leaf: bool
    probs: list[float]


class SimpleTree:
    def __init__(self, nodes: list[Node]):
        self.nodes = nodes

    def predict_proba_one(self, x: list[float], class_count: int) -> list[float]:
        idx = 0
        while True:
            node = self.nodes[idx]
            if node.is_leaf:
                values = list(node.probs[:class_count])
                if len(values) < class_count:
                    values.extend([0.0] * (class_count - len(values)))
                total = sum(values)
                if total <= 0.0:
                    return [1.0 / max(1, class_count)] * max(1, class_count)
                return [float(value / total) for value in values]
            value = x[node.feature] if 0 <= node.feature < len(x) else 0.0
            idx = node.left if value <= node.threshold else node.right


class SimpleForest:
    model_type = "forest"

    def __init__(self, trees: list[SimpleTree], feature_names: list[str], class_labels: list[int]):
        self.trees = trees
        self.feature_names = feature_names
        self.class_labels = class_labels or [0, 1]
        self.model_name = "random_forest"

    def predict_proba_one(self, row: dict[str, Any]) -> list[float]:
        x = [float(row.get(name, 0.0)) for name in self.feature_names]
        class_count = max(1, len(self.class_labels))
        if not self.trees:
            return [1.0 / class_count] * class_count
        acc = np.zeros(class_count, dtype=float)
        for tree in self.trees:
            acc += np.array(tree.predict_proba_one(x, class_count), dtype=float)
        acc /= float(len(self.trees))
        return acc.tolist()

    @staticmethod
    def from_payload(payload: dict[str, Any]) -> "SimpleForest":
        trees: list[SimpleTree] = []
        for tree_payload in payload.get("trees", []):
            nodes = [Node(**node_payload) for node_payload in tree_payload.get("nodes", [])]
            trees.append(SimpleTree(nodes))
        return SimpleForest(
            trees=trees,
            feature_names=list(payload.get("feature_names", PRIMARY_MODEL_FEATURE_NAMES)),
            class_labels=[int(value) for value in payload.get("class_labels", [0, 1])],
        )

    @staticmethod
    def load(path: str | Path) -> "SimpleForest":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return SimpleForest.from_payload(payload)


class SimpleMLP:
    model_type = "mlp"

    def __init__(
        self,
        feature_names: list[str],
        class_labels: list[int],
        scaler_mean: np.ndarray,
        scaler_scale: np.ndarray,
        coefs: list[np.ndarray],
        intercepts: list[np.ndarray],
        hidden_activation: str,
        *,
        feature_tier: str = PRIMARY_MODEL_FEATURE_TIER,
        architecture: dict[str, Any] | None = None,
        output_formulation: dict[str, Any] | None = None,
        training_config: dict[str, Any] | None = None,
        training_summary: dict[str, Any] | None = None,
    ):
        self.feature_names = feature_names
        self.class_labels = class_labels
        self.scaler_mean = scaler_mean
        self.scaler_scale = np.where(np.abs(scaler_scale) < 1e-12, 1.0, scaler_scale)
        self.coefs = coefs
        self.intercepts = intercepts
        self.hidden_activation = hidden_activation
        self.feature_tier = feature_tier
        self.architecture = dict(architecture or {})
        self.output_formulation = dict(output_formulation or {})
        self.training_config = dict(training_config or {})
        self.training_summary = dict(training_summary or {})
        self.model_name = "neural_net"

    def _activate(self, x: np.ndarray) -> np.ndarray:
        if self.hidden_activation == "relu":
            return np.maximum(x, 0.0)
        if self.hidden_activation == "tanh":
            return np.tanh(x)
        if self.hidden_activation == "logistic":
            return sigmoid(x)
        return x

    def predict_proba_one(self, row: dict[str, Any]) -> list[float]:
        x = np.array([float(row.get(name, 0.0)) for name in self.feature_names], dtype=float)
        x = (x - self.scaler_mean) / self.scaler_scale
        for layer_idx, (coef, intercept) in enumerate(zip(self.coefs, self.intercepts)):
            x = x @ coef + intercept
            if layer_idx < len(self.coefs) - 1:
                x = self._activate(x)
        if x.ndim == 0:
            x = np.array([float(x)], dtype=float)
        return softmax(np.array(x, dtype=float)).tolist()

    @staticmethod
    def from_payload(payload: dict[str, Any]) -> "SimpleMLP":
        return SimpleMLP(
            feature_names=list(payload.get("feature_names", PRIMARY_MODEL_FEATURE_NAMES)),
            class_labels=[int(value) for value in payload.get("class_labels", [0, 1])],
            scaler_mean=np.array(payload.get("scaler_mean", []), dtype=float),
            scaler_scale=np.array(payload.get("scaler_scale", []), dtype=float),
            coefs=[np.array(values, dtype=float) for values in payload.get("coefs", [])],
            intercepts=[np.array(values, dtype=float) for values in payload.get("intercepts", [])],
            hidden_activation=str(payload.get("hidden_activation", "relu")),
            feature_tier=str(payload.get("feature_tier", PRIMARY_MODEL_FEATURE_TIER)),
            architecture=dict(payload.get("architecture") or {}),
            output_formulation=dict(payload.get("output_formulation") or {}),
            training_config=dict(payload.get("training_config") or {}),
            training_summary=dict(payload.get("training_summary") or {}),
        )

    @staticmethod
    def load(path: str | Path) -> "SimpleMLP":
        with Path(path).open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return SimpleMLP.from_payload(payload)


def load_primary_model(path: str | Path) -> SimpleForest | SimpleMLP:
    with Path(path).open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    model_type = str(payload.get("model_type", "forest"))
    if model_type == "mlp":
        return SimpleMLP.from_payload(payload)
    return SimpleForest.from_payload(payload)


def _feature_names_use_canonical_surface(feature_names: list[str]) -> bool:
    canonical_prefixes = (
        "actor_context.",
        "mission_context.",
        "command_semantics.",
        "argument_profile.",
        "normalized_state.",
        "recent_behavior.",
    )
    return any(
        name == "platform_family" or any(name.startswith(prefix) for prefix in canonical_prefixes)
        for name in feature_names
    )


def _model_probabilities(model: SimpleForest | SimpleMLP | None, row: dict[str, Any]) -> dict[str, float]:
    if model is None:
        return {"pbenign": 1.0, "pcyber": 0.0, "pfault": 0.0}
    raw_probs = model.predict_proba_one(row)
    class_labels = getattr(model, "class_labels", [0, 1])
    values = {class_name(label): 0.0 for label in range(len(CLASS_NAMES))}
    for label, prob in zip(class_labels, raw_probs):
        values[class_name(int(label))] = float(prob)
    total = sum(values.values())
    if total <= 0.0:
        return {"pbenign": 1.0, "pcyber": 0.0, "pfault": 0.0}
    return {
        "pbenign": float(values.get("benign", 0.0) / total),
        "pcyber": float(values.get("cyber", 0.0) / total),
        "pfault": float(values.get("fault", 0.0) / total),
    }


def _load_runtime_manifest(model_dir: Path) -> dict[str, Any]:
    manifest_path = model_dir / RUNTIME_BUNDLE_MANIFEST_NAME
    if manifest_path.exists():
        with manifest_path.open("r", encoding="utf-8") as handle:
            manifest = json.load(handle)
        manifest.setdefault("bundle_schema_version", RUNTIME_BUNDLE_SCHEMA_VERSION)
        return manifest
    blue_model_path = model_dir / POSTER_BLUE_MODEL_ARTIFACT_NAME
    if blue_model_path.exists():
        with blue_model_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        return build_poster_blue_runtime_manifest(
            training_path="poster_default_canonical",
            feature_tier=str(payload.get("feature_tier", PRIMARY_MODEL_FEATURE_TIER)),
        )
    legacy_model_path = model_dir / LEGACY_PRIMARY_MODEL_ARTIFACT_NAME
    if legacy_model_path.exists():
        with legacy_model_path.open("r", encoding="utf-8") as handle:
            payload = json.load(handle)
        feature_tier = str(payload.get("feature_tier", PRIMARY_MODEL_FEATURE_TIER))
        feature_names = list(payload.get("feature_names", []))
        if not (model_dir / "novelty.cfg").exists() and not (model_dir / "calibrator.json").exists():
            if _feature_names_use_canonical_surface(feature_names):
                return build_poster_blue_runtime_manifest(
                    training_path="poster_default_canonical",
                    feature_tier=feature_tier,
                    primary_model_artifact=LEGACY_PRIMARY_MODEL_ARTIFACT_NAME,
                )
        return build_legacy_runtime_manifest(
            training_path="legacy_fprime_request_time",
            feature_tier=feature_tier,
        )
    if (model_dir / "forest.json").exists():
        return build_legacy_runtime_manifest(
            training_path="legacy_fprime_request_time",
            feature_tier=PRIMARY_MODEL_FEATURE_TIER,
            primary_model_artifact="forest.json",
        )
    return build_legacy_runtime_manifest(
        training_path="legacy_fprime_request_time",
        feature_tier=PRIMARY_MODEL_FEATURE_TIER,
    )


def export_sklearn_forest(rf: Any, feature_names: list[str], feature_tier: str) -> dict[str, Any]:
    trees: list[dict[str, Any]] = []
    for estimator in rf.estimators_:
        tree = estimator.tree_
        nodes: list[dict[str, Any]] = []
        for idx in range(tree.node_count):
            is_leaf = tree.children_left[idx] == -1
            value = tree.value[idx][0].tolist()
            if is_leaf:
                nodes.append({"feature": -1, "threshold": 0.0, "left": -1, "right": -1, "is_leaf": True, "probs": value})
            else:
                nodes.append(
                    {
                        "feature": int(tree.feature[idx]),
                        "threshold": float(tree.threshold[idx]),
                        "left": int(tree.children_left[idx]),
                        "right": int(tree.children_right[idx]),
                        "is_leaf": False,
                        "probs": value,
                    }
                )
        trees.append({"nodes": nodes})
    return {
        "model_type": "forest",
        "model_name": "random_forest",
        "feature_tier": str(feature_tier),
        "feature_names": list(feature_names),
        "class_labels": [int(value) for value in rf.classes_.tolist()],
        "trees": trees,
    }


def export_sklearn_mlp(
    pipeline: Any,
    feature_names: list[str],
    feature_tier: str,
    *,
    extra_fields: dict[str, Any] | None = None,
) -> dict[str, Any]:
    scaler = pipeline.named_steps["scale"]
    mlp = pipeline.named_steps["mlp"]
    payload = {
        "model_type": "mlp",
        "model_name": "neural_net",
        "feature_tier": str(feature_tier),
        "feature_names": list(feature_names),
        "class_labels": [int(value) for value in mlp.classes_.tolist()],
        "scaler_mean": np.array(scaler.mean_, dtype=float).tolist(),
        "scaler_scale": np.array(scaler.scale_, dtype=float).tolist(),
        "coefs": [np.array(values, dtype=float).tolist() for values in mlp.coefs_],
        "intercepts": [np.array(values, dtype=float).tolist() for values in mlp.intercepts_],
        "hidden_activation": str(mlp.activation),
    }
    if extra_fields:
        payload.update(dict(extra_fields))
    return payload


class PosterBlueRuntimeBundle:
    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR, *, manifest: dict[str, Any] | None = None):
        self.model_dir = Path(model_dir)
        self.manifest = dict(manifest or _load_runtime_manifest(self.model_dir))
        primary_artifact = str(self.manifest.get("primary_model_artifact", POSTER_BLUE_MODEL_ARTIFACT_NAME))
        model_path = self.model_dir / primary_artifact
        self.model = load_primary_model(model_path) if model_path.exists() else None
        self.runtime_kind = POSTER_BLUE_RUNTIME_KIND
        self.runtime_api = POSTER_BLUE_RUNTIME_API
        self.feature_names = list(getattr(self.model, "feature_names", []))

    def score_row(self, row: dict[str, Any]) -> dict[str, Any]:
        probs = _model_probabilities(self.model, row)
        pbenign = probs["pbenign"]
        pcyber = probs["pcyber"]
        pfault = probs["pfault"]
        panomaly = clip(pcyber + pfault, 0.0, 1.0)
        predicted_class = max(
            [("benign", pbenign), ("cyber", pcyber), ("fault", pfault)],
            key=lambda item: item[1],
        )[0]
        detector_reason = "benign_baseline" if predicted_class == "benign" else f"model_{predicted_class}"
        scored = dict(row)
        scored.update(
            {
                "runtime_kind": self.runtime_kind,
                "pbenign": pbenign,
                "pcyber": pcyber,
                "pfault": pfault,
                "panomaly": panomaly,
                "anomaly_score": panomaly,
                "unsafe_risk": panomaly,
                "risk": panomaly,
                "predicted_class": predicted_class,
                "detector_reason": detector_reason,
            }
        )
        return scored


class LegacyRuntimeBundle:
    def __init__(self, model_dir: str | Path = DEFAULT_MODEL_DIR, *, manifest: dict[str, Any] | None = None):
        self.model_dir = Path(model_dir)
        self.manifest = dict(manifest or _load_runtime_manifest(self.model_dir))
        primary_artifact = str(self.manifest.get("primary_model_artifact", LEGACY_PRIMARY_MODEL_ARTIFACT_NAME))
        self.model = None
        primary_path = self.model_dir / primary_artifact
        if primary_path.exists():
            self.model = load_primary_model(primary_path)
        elif (self.model_dir / "forest.json").exists():
            self.model = load_primary_model(self.model_dir / "forest.json")
        self.novelty = GaussianNovelty.load(self.model_dir / "novelty.cfg") if (self.model_dir / "novelty.cfg").exists() else None
        self.calibrator = Calibrator.load(self.model_dir / "calibrator.json") if (self.model_dir / "calibrator.json").exists() else None
        self.runtime_kind = LEGACY_STACKED_RUNTIME_KIND
        self.runtime_api = LEGACY_STACKED_RUNTIME_API
        self.feature_names = list(getattr(self.model, "feature_names", []))

    def score_row(self, row: dict[str, Any]) -> dict[str, Any]:
        probs = _model_probabilities(self.model, row)
        pbenign = probs["pbenign"]
        pcyber = probs["pcyber"]
        pfault = probs["pfault"]
        panomaly = clip(pcyber + pfault, 0.0, 1.0)
        rules = float(rule_score(row))
        novelty = 0.0 if self.novelty is None else float(self.novelty.score(vector_from_row(row, self.novelty.feature_names or NOVELTY_FEATURE_NAMES)))
        risk = max(panomaly, rules, novelty)
        if self.calibrator is not None:
            risk = float(self.calibrator.score(panomaly, pcyber, rules, novelty))
        predicted_class = max(
            [("benign", pbenign), ("cyber", pcyber), ("fault", pfault)],
            key=lambda item: item[1],
        )[0]
        if rules > 0.0:
            detector_reason = "rule_triggered"
        elif predicted_class != "benign":
            detector_reason = f"model_{predicted_class}"
        elif novelty >= 0.55 and risk >= 0.35:
            detector_reason = "novelty_drift"
        else:
            detector_reason = "benign_baseline"
        scored = dict(row)
        scored.update(
            {
                "runtime_kind": self.runtime_kind,
                "pbenign": pbenign,
                "pcyber": pcyber,
                "pfault": pfault,
                "panomaly": panomaly,
                "anomaly_score": panomaly,
                "unsafe_risk": risk,
                "predicted_class": predicted_class,
                "rules": rules,
                "novelty": novelty,
                "risk": risk,
                "detector_reason": detector_reason,
            }
        )
        return scored


def load_runtime_bundle(model_dir: str | Path = DEFAULT_MODEL_DIR) -> PosterBlueRuntimeBundle | LegacyRuntimeBundle:
    resolved_model_dir = Path(model_dir)
    manifest = _load_runtime_manifest(resolved_model_dir)
    runtime_kind = str(manifest.get("runtime_kind", LEGACY_STACKED_RUNTIME_KIND))
    if runtime_kind == POSTER_BLUE_RUNTIME_KIND:
        return PosterBlueRuntimeBundle(resolved_model_dir, manifest=manifest)
    return LegacyRuntimeBundle(resolved_model_dir, manifest=manifest)


class RuntimeBundle:
    def __new__(cls, model_dir: str | Path = DEFAULT_MODEL_DIR):
        return load_runtime_bundle(model_dir)
