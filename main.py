#!/usr/bin/env python3
from __future__ import annotations

import argparse
import json
import math
import random
import shutil
import sys
import subprocess
from collections import Counter, deque
from dataclasses import asdict, dataclass
from pathlib import Path
from typing import Any, Callable

TRAINING_IMPORT_ERROR: Exception | None = None
try:
    import matplotlib
    import numpy as np
    from sklearn.ensemble import RandomForestClassifier
    from sklearn.linear_model import LogisticRegression
    from sklearn.metrics import (
        accuracy_score,
        average_precision_score,
        confusion_matrix,
        precision_recall_curve,
        precision_recall_fscore_support,
        roc_auc_score,
        roc_curve,
    )
    from sklearn.neural_network import MLPClassifier
    from sklearn.pipeline import Pipeline
    from sklearn.preprocessing import StandardScaler
except Exception as exc:  # pragma: no cover - environment-dependent import path
    TRAINING_IMPORT_ERROR = exc
    matplotlib = None
    np = None

from runtime import (
    BACKGROUND_DEFAULTS,
    CLASS_NAMES,
    COMMAND_IDS,
    COMMAND_NAMES,
    COMMAND_SPECS,
    DEFAULT_MODEL_DIR,
    FEATURE_TIER_FEATURE_NAMES,
    FEATURE_NAMES,
    LEGACY_STACKED_RUNTIME_KIND,
    LEGACY_PRIMARY_MODEL_ARTIFACT_NAME,
    MODEL_FEATURE_LAYOUTS,
    NOVELTY_FEATURE_NAMES,
    POSTER_BLUE_RUNTIME_KIND,
    POSTER_BLUE_MODEL_ARTIFACT_NAME,
    PRIMARY_MODEL_FEATURE_NAMES,
    PRIMARY_MODEL_FEATURE_TIER,
    REQUEST_FEATURE_NAMES,
    RUNTIME_BUNDLE_MANIFEST_NAME,
    SCHEMA_VERSION,
    SERVICE_IDS,
    SENSITIVE_SERVICES,
    Calibrator,
    GaussianNovelty,
    build_legacy_runtime_manifest,
    build_poster_blue_runtime_manifest,
    class_name,
    export_sklearn_forest,
    export_sklearn_mlp,
    history_state_reset_mode,
    inspect_args,
    load_runtime_bundle,
    packets_to_transactions,
    rule_score,
    save_json,
    stable_token_id,
    transactions_to_rows,
    vector_from_row,
)
from tools.shared.feature_policy import (
    BlueFeaturePolicyError,
    BLUE_FEATURE_POLICY_LEGACY_REQUEST_TIME_BASELINE,
    BLUE_FEATURE_POLICY_POSTER_DEFAULT,
    available_blue_feature_policies,
    validate_blue_feature_names,
)
from tools.fprime_real.schedule_profiles import assert_diverse_episode_signatures, assert_split_episode_separation, build_command_family_overlap_report, build_episode_signature_report, has_structural_signature_signal
from tools.shared.canonical_records import canonicalize_legacy_fprime_transaction
from tools.train.poster_default import (
    POSTER_DEFAULT_COMMAND_BASELINE_FEATURE_NAMES,
    POSTER_DEFAULT_PRIMARY_MODEL_FEATURE_TIER,
    POSTER_DEFAULT_REQUEST_ONLY_BASELINE_FEATURE_NAMES,
    POSTER_DEFAULT_REQUEST_TUPLE_PURITY_BUCKETS,
    POSTER_DEFAULT_REQUEST_TUPLE_PURITY_FEATURE_NAMES,
    POSTER_DEFAULT_TRAINING_PATH_LABEL,
    POSTER_DEFAULT_TRAINING_PATH_NAME,
    canonical_row_to_training_row,
    canonical_rows_to_training_rows,
    load_canonical_training_records,
)
from tools.train.blue_model import (
    fit_poster_blue_model,
    export_poster_blue_model_payload,
)
if matplotlib is not None:
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
else:  # pragma: no cover - exercised only when optional deps are missing
    plt = None

DEFAULT_ROWS = 50000
DEFAULT_OUTPUT_DIR = Path("artifacts/latest")
ORBIT_SECONDS = 5400
ROWS_PER_EPISODE = 240
FAULT_LABEL = 2
FAULT_REASONS = {"downlink_empty", "downlink_not_ready", "flush_blocked_backlog"}
FAULT_FAMILIES = [
    ("power_sag", 0.22),
    ("thermal_excursion", 0.20),
    ("comms_degradation", 0.20),
    ("adcs_instability", 0.18),
    ("storage_pressure", 0.20),
]
SYNTHETIC_HISTORY_RESET_KEY = "episode_id"
TRAINING_GROUP_KEY = "run_id"
COMMAND_ONLY_BASELINE_FEATURE_NAMES = ["service_id", "command_id", "arg_count", "arg_norm", "arg_out_of_range"]
COMMAND_ONLY_BASELINE_NEAR_PERFECT_THRESHOLD = 0.95
REQUEST_ONLY_BASELINE_FEATURE_NAMES = list(REQUEST_FEATURE_NAMES)
REQUEST_ONLY_BASELINE_NEAR_PERFECT_THRESHOLD = 0.95
OUTCOME_ONLY_BASELINE_FEATURE_NAMES = ["gds_accept", "sat_success", "timeout", "response_code", "txn_warning_events", "txn_error_events"]
OUTCOME_ONLY_BASELINE_NEAR_PERFECT_THRESHOLD = 0.95
PROTOCOL_ONLY_BASELINE_FEATURE_NAMES = ["protocol_family"]
PROTOCOL_ONLY_BASELINE_NEAR_PERFECT_THRESHOLD = 0.90
RAW_PROTOCOL_SHORTCUT_BASELINE_FEATURE_NAMES = ["protocol_family", "raw_service_name", "raw_command_name"]
RAW_PROTOCOL_SHORTCUT_BASELINE_NEAR_PERFECT_THRESHOLD = 0.95
REQUEST_TUPLE_PURITY_FEATURE_NAMES = list(COMMAND_ONLY_BASELINE_FEATURE_NAMES)
REQUEST_TUPLE_PURITY_BUCKETS = {
    "service_id": 1.0,
    "command_id": 1.0,
    "arg_count": 1.0,
    "arg_norm": 0.1,
    "arg_out_of_range": 1.0,
}
DATASET_SANITY_MIN_CLASS_OVERLAP_RATIO = 0.5
DATASET_SANITY_MAX_REQUEST_TUPLE_PURE_ROW_FRACTION = 0.95
DATASET_SANITY_MAX_REQUEST_TUPLE_MAJORITY_ROW_FRACTION = 0.95
DEFAULT_EVALUATION_SEED_COUNT = 3
DEFAULT_GROUPED_CV_FOLDS = 3
TRAINING_PATH_LEGACY_FPRIME_BASELINE = "legacy_fprime_baseline"
DEFAULT_TRAINING_PATH_NAME = POSTER_DEFAULT_TRAINING_PATH_NAME
DEFAULT_GENERATION_PROTOCOL_MODE = "fprime"
SUPPORTED_GENERATION_PROTOCOL_MODES = ("fprime", "mavlink", "mixed")
DEFAULT_MIXED_FPRIME_RATIO = 0.5
GENERALIZATION_GATE_THRESHOLDS = {
    "grouped_cv": {
        "class_macro_f1": 0.60,
        "min_per_class_recall": 0.45,
        "anomaly_f1": 0.60,
        "cyber_f1": 0.55,
    },
    "scenario_family_holdout": {
        "class_macro_f1": 0.50,
        "min_per_class_recall": 0.35,
        "anomaly_f1": 0.55,
        "cyber_f1": 0.50,
    },
    "command_family_holdout": {
        "class_macro_f1": 0.50,
        "min_per_class_recall": 0.35,
        "anomaly_f1": 0.55,
        "cyber_f1": 0.50,
    },
    "protocol_family_holdout": {
        "class_macro_f1": 0.50,
        "min_per_class_recall": 0.35,
        "anomaly_f1": 0.55,
        "cyber_f1": 0.50,
    },
}
MODEL_ONLY_NAMESPACE = "model_only"
STACKED_DETECTOR_NAMESPACE = "stacked_detector"
NOVELTY_ADAPTATION_DISABLED = "disabled"
NOVELTY_ADAPTATION_DISABLED_REASON = "offline_evaluation_matches_runtime_non_adaptive_scoring"
SELECTION_METRIC_CLASS_MACRO_F1 = "model_only.multiclass_macro_f1"
SELECTION_METRIC_MIN_PER_CLASS_RECALL = "model_only.min_per_class_recall"
SELECTION_METRIC_STACKED_ANOMALY_F1 = "stacked_detector.anomaly_f1"
SELECTION_METRIC_MODEL_ANOMALY_F1 = "model_only.anomaly_f1"
SELECTION_METRIC_MODEL_CYBER_F1 = "model_only.cyber_f1"
SELECTION_METRIC_ORDER = [
    SELECTION_METRIC_CLASS_MACRO_F1,
    SELECTION_METRIC_MIN_PER_CLASS_RECALL,
    SELECTION_METRIC_STACKED_ANOMALY_F1,
    SELECTION_METRIC_MODEL_CYBER_F1,
]
POSTER_SELECTION_METRIC_ORDER = [
    SELECTION_METRIC_CLASS_MACRO_F1,
    SELECTION_METRIC_MIN_PER_CLASS_RECALL,
    SELECTION_METRIC_MODEL_ANOMALY_F1,
    SELECTION_METRIC_MODEL_CYBER_F1,
]


def is_legacy_training_path(training_path_name: str) -> bool:
    return str(training_path_name) == TRAINING_PATH_LEGACY_FPRIME_BASELINE


def default_blue_feature_policy_name(training_path_name: str) -> str:
    if is_legacy_training_path(training_path_name):
        return BLUE_FEATURE_POLICY_LEGACY_REQUEST_TIME_BASELINE
    return BLUE_FEATURE_POLICY_POSTER_DEFAULT


def active_selection_metric_order(training_path_name: str) -> list[str]:
    if is_legacy_training_path(training_path_name):
        return list(SELECTION_METRIC_ORDER)
    return list(POSTER_SELECTION_METRIC_ORDER)


def active_anomaly_selection_metric(training_path_name: str) -> str:
    if is_legacy_training_path(training_path_name):
        return SELECTION_METRIC_STACKED_ANOMALY_F1
    return SELECTION_METRIC_MODEL_ANOMALY_F1


def training_path_summary(training_path_name: str, *, blue_feature_policy_name: str) -> dict[str, Any]:
    if is_legacy_training_path(training_path_name):
        return {
            "name": TRAINING_PATH_LEGACY_FPRIME_BASELINE,
            "label": "legacy F-prime request-time baseline (comparison-only)",
            "default": False,
            "comparison_only": True,
            "feature_surface": "flat legacy request-time rows",
            "detector_stack": "random_forest_and_neural_plus_rules_novelty_calibrator",
            "blue_feature_policy_name": blue_feature_policy_name,
        }
    return {
        "name": POSTER_DEFAULT_TRAINING_PATH_NAME,
        "label": POSTER_DEFAULT_TRAINING_PATH_LABEL,
        "default": True,
        "comparison_only": False,
        "feature_surface": "canonical semantic rows",
        "detector_stack": "neural_only",
        "blue_feature_policy_name": blue_feature_policy_name,
    }


def model_uses_canonical_features(feature_names: list[str]) -> bool:
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


def resolve_feature_name_list(
    feature_names: list[str] | None,
    default_feature_names: list[str],
) -> list[str]:
    if feature_names is None:
        return list(default_feature_names)
    return list(feature_names)


def poster_history_featurization_report(record_provenance: dict[str, Any]) -> dict[str, Any]:
    return {
        "mode": "canonical_rows",
        "group_key": TRAINING_GROUP_KEY,
        "state_reset": history_state_reset_mode(TRAINING_GROUP_KEY),
        "record_source": str(record_provenance.get("record_source", "canonical_command_rows")),
        "record_path": str(record_provenance.get("record_path", "")),
        "derived_from_legacy_transactions": bool(record_provenance.get("derived_from_legacy_transactions")),
        "canonical_recent_behavior": True,
    }


def materialize_canonical_training_rows(
    records: list[dict[str, Any]],
    *,
    blue_feature_policy_name: str,
) -> list[dict[str, Any]]:
    return canonical_rows_to_training_rows(records, policy_name=blue_feature_policy_name)


def materialize_canonical_split_rows(
    base_records: list[dict[str, Any]],
    calib_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    group_key: str,
    *,
    blue_feature_policy_name: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    del group_key
    return (
        materialize_canonical_training_rows(base_records, blue_feature_policy_name=blue_feature_policy_name),
        materialize_canonical_training_rows(calib_records, blue_feature_policy_name=blue_feature_policy_name),
        materialize_canonical_training_rows(test_records, blue_feature_policy_name=blue_feature_policy_name),
    )


def require_training_deps() -> None:
    if TRAINING_IMPORT_ERROR is not None:
        raise SystemExit(f"Training dependencies unavailable: {TRAINING_IMPORT_ERROR}")


@dataclass(frozen=True)
class Actor:
    name: str
    trust: float
    role: str


@dataclass(frozen=True)
class CommandIntent:
    episode_id: int
    episode_label: int
    episode_kind: str
    step_id: int
    time_ms: int
    session_id: str
    txn_id: str
    actor: Actor
    command: str
    args: dict[str, float]
    attack_family: str
    phase: str


@dataclass(frozen=True)
class PacketFrame:
    ts_ms: int
    packet_kind: str
    src: str
    dst: str
    episode_id: int
    episode_label: int
    episode_kind: str
    session_id: str
    txn_id: str
    service: str
    command: str
    bytes_on_wire: int
    label: int
    attack_family: str
    phase: str
    actor: str
    actor_role: str
    actor_trust: float
    target_service: str = ""
    target_stream_id: str = ""
    target_stream_index: int = -1
    node_service: str = ""
    args: dict[str, float] | None = None
    payload: dict[str, float] | None = None
    response_code: int = 0
    gds_accept: int = 0
    sat_success: int = 0
    timeout: int = 0
    reason: str = ""
    response_direction_seen: int = 0
    final_observed_on_wire: int = 0
    txn_warning_events: int = 0
    txn_error_events: int = 0


@dataclass
class SatelliteState:
    queue_depth: float = 0.0
    battery_soc: float = 82.0
    bus_voltage_v: float = 27.8
    solar_array_current_a: float = 3.6
    battery_temp_c: float = 16.0
    payload_temp_c: float = 2.0
    attitude_error_deg: float = 0.35
    wheel_speed_rpm: float = 1500.0
    link_rssi_dbm: float = -96.0
    downlink_backlog_mb: float = 25.0
    storage_free_mb: float = 6144.0
    event_backlog: float = 12.0
    radiation_flux: float = 1.4
    payload_cover_open: float = 0.0
    heater_on: float = 1.0
    comms_window_open: float = 0.0


NOMINAL_ACTORS = [
    Actor("ops_a1", 0.98, "ops_primary"),
    Actor("ops_a2", 0.91, "ops_backup"),
    Actor("ops_b1", 0.97, "ops_primary"),
    Actor("ops_b2", 0.90, "ops_backup"),
]

ATTACK_ACTORS = [
    Actor("ops_a1", 0.58, "shared_identity"),
    Actor("ops_b1", 0.61, "shared_identity"),
    Actor("red_a1", 0.22, "external"),
    Actor("red_a2", 0.34, "external"),
    Actor("red_b1", 0.20, "external"),
    Actor("red_b2", 0.31, "external"),
]

ATTACK_FAMILIES = [
    ("opcode_bruteforce", 0.14),
    ("arg_bruteforce", 0.25),
    ("sequence_abuse", 0.18),
    ("state_tamper", 0.17),
    ("downlink_abuse", 0.16),
    ("masquerade_abuse", 0.10),
]

STARTUP_COMMAND_SEQUENCE = [
    "cmdDisp.CMD_NO_OP",
    "cmdDisp.CMD_NO_OP_STRING",
    "systemResources.VERSION",
    "eventLogger.DUMP_FILTER_STATE",
    "cmdSeq.CS_AUTO",
]
SCIENCE_COMMAND_SEQUENCE = [
    "cmdDisp.CMD_TEST_CMD_1",
    "prmDb.PRM_SAVE_FILE",
    "fileManager.CreateDirectory",
    "fileManager.AppendFile",
    "cmdSeq.CS_START",
]
DOWNLINK_COMMAND_SEQUENCE = [
    "fileDownlink.SendPartial",
    "fileDownlink.SendFile",
    "fileManager.FileSize",
    "fileDownlink.Cancel",
]
STANDBY_COMMAND_SEQUENCE = [
    "cmdDisp.CMD_CLEAR_TRACKING",
    "cmdSeq.CS_MANUAL",
    "cmdSeq.CS_JOIN_WAIT",
    "fileManager.MoveFile",
    "fileManager.RemoveFile",
]
UNKNOWN_ATTACK_COMMANDS = [
    "cmdDisp.CMD_UNKNOWN_PROBE",
    "cmdSeq.CS_HALT_BAD",
    "fileManager.FormatVolume",
    "fileDownlink.SendEverything",
]


def clip(value: float, low: float, high: float) -> float:
    return float(min(max(value, low), high))


def default_args_for_command(command_name: str, slot: int) -> dict[str, float]:
    spec = COMMAND_SPECS[command_name]
    args: dict[str, float] = {}
    for arg_name, (low, high, default) in spec.arg_bounds.items():
        if arg_name == "token_id":
            value = 100 + (slot % 400)
        elif arg_name == "i32_arg":
            value = -120 + (slot % 11) * 24
        elif arg_name == "f32_arg":
            value = -6.0 + (slot % 9) * 1.5
        elif arg_name == "u8_arg":
            value = 4 + (slot % 20)
        elif arg_name in {"path_id", "file_id"}:
            value = 32 + (slot % 96)
        elif arg_name in {"src_id", "dst_id", "dest_id"}:
            value = 80 + (slot % 120)
        elif arg_name == "op_id":
            value = 200 + (slot % 64)
        elif arg_name == "offset_kb":
            value = (slot % 8) * 128
        elif arg_name == "length_kb":
            value = 256 + (slot % 8) * 128
        elif arg_name == "enabled":
            value = 1 if (slot % 7) != 0 else 0
        else:
            value = default
        args[arg_name] = clip(float(value), low, high)
    return args


def target_service_for_actor(actor_name: str, slot: int = 0) -> str:
    if actor_name.startswith(("ops_a", "red_a")):
        return "fprime_b"
    if actor_name.startswith(("ops_b", "red_b")):
        return "fprime_a"
    return "fprime_a" if (slot % 2) == 0 else "fprime_b"


def phase_for_time(time_ms: int) -> str:
    orbit_time = (time_ms / 1000.0) % ORBIT_SECONDS
    if orbit_time < 900:
        return "startup"
    if orbit_time < 3000:
        return "science"
    if orbit_time < 4200:
        return "downlink"
    return "standby"


class MissionScheduler:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def _session_id(self, episode_id: int) -> str:
        return f"ep-{episode_id:04d}"

    def _default_args(self, command_name: str, slot: int) -> dict[str, float]:
        return default_args_for_command(command_name, slot)

    def _next_startup(self, slot: int) -> str:
        return STARTUP_COMMAND_SEQUENCE[slot % len(STARTUP_COMMAND_SEQUENCE)]

    def _next_science(self, slot: int) -> str:
        return SCIENCE_COMMAND_SEQUENCE[slot % len(SCIENCE_COMMAND_SEQUENCE)]

    def _next_downlink(self, slot: int) -> str:
        return DOWNLINK_COMMAND_SEQUENCE[slot % len(DOWNLINK_COMMAND_SEQUENCE)]

    def _next_standby(self, slot: int) -> str:
        return STANDBY_COMMAND_SEQUENCE[slot % len(STANDBY_COMMAND_SEQUENCE)]

    def build_nominal(self, count: int, start_ms: int, episode_id: int) -> list[CommandIntent]:
        intents: list[CommandIntent] = []
        current_ms = start_ms
        session_id = self._session_id(episode_id)
        for slot in range(count):
            phase = phase_for_time(current_ms)
            if phase == "startup":
                command_name = self._next_startup(slot)
            elif phase == "science":
                command_name = self._next_science(slot)
            elif phase == "downlink":
                command_name = self._next_downlink(slot)
            else:
                command_name = self._next_standby(slot)
            actor = NOMINAL_ACTORS[slot % len(NOMINAL_ACTORS)]
            intents.append(
                CommandIntent(
                    episode_id=episode_id,
                    episode_label=0,
                    episode_kind="benign",
                    step_id=slot,
                    time_ms=current_ms,
                    session_id=session_id,
                    txn_id=f"{episode_id:04d}-{slot:06d}",
                    actor=actor,
                    command=command_name,
                    args=self._default_args(command_name, slot),
                    attack_family="none",
                    phase=phase,
                )
            )
            if phase in {"science", "downlink"} and self.rng.random() < 0.22:
                current_ms += self.rng.randint(4500, 11000)
            else:
                current_ms += self.rng.randint(9000, 28000)
        return intents


class AttackFuzzer:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def _session_id(self, episode_id: int) -> str:
        return f"ep-{episode_id:04d}"

    def _pick_family(self) -> str:
        names = [name for name, _ in ATTACK_FAMILIES]
        weights = [weight for _, weight in ATTACK_FAMILIES]
        return self.rng.choices(names, weights=weights, k=1)[0]

    def _pick_actor(self, family: str) -> Actor:
        if family == "masquerade_abuse":
            pool = [ATTACK_ACTORS[0], ATTACK_ACTORS[1], ATTACK_ACTORS[2], NOMINAL_ACTORS[2]]
            weights = [0.25, 0.35, 0.20, 0.20]
            return self.rng.choices(pool, weights=weights, k=1)[0]
        if family in {"state_tamper", "downlink_abuse"}:
            pool = [ATTACK_ACTORS[0], ATTACK_ACTORS[2], ATTACK_ACTORS[3], ATTACK_ACTORS[4], ATTACK_ACTORS[5]]
            weights = [0.24, 0.18, 0.18, 0.28, 0.12]
            return self.rng.choices(pool, weights=weights, k=1)[0]
        return self.rng.choice(ATTACK_ACTORS)

    def _unknown_command(self) -> str:
        return self.rng.choice(UNKNOWN_ATTACK_COMMANDS)

    def _fuzz_args(self, command_name: str, family: str, slot: int) -> dict[str, float]:
        spec = COMMAND_SPECS[command_name]
        args: dict[str, float] = {}
        for arg_name, (low, high, default) in spec.arg_bounds.items():
            span = max(1.0, high - low if high != low else max(abs(default), 1.0))
            if family == "arg_bruteforce":
                if self.rng.random() < 0.5:
                    value = high + span * self.rng.uniform(0.8, 4.0)
                else:
                    value = low - span * self.rng.uniform(0.8, 3.0)
            elif family == "state_tamper" and arg_name in {"path_id", "file_id", "src_id", "dst_id", "dest_id", "op_id"}:
                value = default + span * self.rng.uniform(0.40, 0.95)
            elif family == "downlink_abuse" and arg_name in {"offset_kb", "length_kb"}:
                value = high + span * self.rng.uniform(2.0, 12.0)
            elif family == "sequence_abuse" and arg_name == "path_id":
                value = low
            elif family == "masquerade_abuse":
                value = default + self.rng.uniform(-0.15 * span, 0.15 * span)
            else:
                value = default
            args[arg_name] = float(value)
        if family == "arg_bruteforce" and spec.arg_bounds and self.rng.random() < 0.25:
            args.pop(next(iter(spec.arg_bounds)))
        if family == "arg_bruteforce" and self.rng.random() < 0.25:
            args["junk"] = float(self.rng.randint(1, 999))
        return args

    def _pick_command(self, family: str, slot: int, previous: CommandIntent | None) -> tuple[str, dict[str, float]]:
        if family == "opcode_bruteforce":
            return self._unknown_command(), {}
        if family == "state_tamper":
            command_name = self.rng.choice(["fileManager.ShellCommand", "fileManager.RemoveDirectory", "fileManager.RemoveFile", "prmDb.PRM_SAVE_FILE"])
        elif family == "downlink_abuse":
            command_name = self.rng.choice(["fileDownlink.SendPartial", "fileDownlink.SendFile", "fileDownlink.Cancel"])
        elif family == "sequence_abuse":
            command_name = self.rng.choice(["cmdSeq.CS_STEP", "cmdSeq.CS_START", "cmdSeq.CS_CANCEL", "cmdSeq.CS_JOIN_WAIT"])
        elif family == "masquerade_abuse" and previous is not None and self.rng.random() < 0.5:
            command_name = previous.command
        elif family == "masquerade_abuse":
            command_name = self.rng.choice(["cmdDisp.CMD_NO_OP_STRING", "cmdDisp.CMD_TEST_CMD_1", "fileManager.ShellCommand"])
        else:
            command_name = self.rng.choice(COMMAND_NAMES)
        return command_name, self._fuzz_args(command_name, family, slot) if command_name in COMMAND_SPECS else {}

    def build_attack(self, count: int, start_ms: int, episode_id: int) -> list[CommandIntent]:
        intents: list[CommandIntent] = []
        session_id = self._session_id(episode_id)
        current_ms = start_ms
        previous: CommandIntent | None = None
        for slot in range(count):
            family = self._pick_family()
            actor = self._pick_actor(family)
            command_name, args = self._pick_command(family, slot, previous)
            intent = CommandIntent(
                episode_id=episode_id,
                episode_label=1,
                episode_kind="cyber",
                step_id=slot,
                time_ms=current_ms,
                session_id=session_id,
                txn_id=f"{episode_id:04d}-{slot:06d}",
                actor=actor,
                command=command_name,
                args=args,
                attack_family=family,
                phase=phase_for_time(current_ms),
            )
            intents.append(intent)
            previous = intent
            if family == "masquerade_abuse":
                current_ms += self.rng.randint(1800, 7000)
            elif family in {"state_tamper", "downlink_abuse"}:
                current_ms += self.rng.randint(2200, 9000)
            else:
                current_ms += self.rng.randint(3500, 12000)
        return intents


class FaultFuzzer:
    def __init__(self, seed: int):
        self.rng = random.Random(seed)

    def _session_id(self, episode_id: int) -> str:
        return f"ep-{episode_id:04d}"

    def _pick_family(self) -> str:
        names = [name for name, _ in FAULT_FAMILIES]
        weights = [weight for _, weight in FAULT_FAMILIES]
        return self.rng.choices(names, weights=weights, k=1)[0]

    def _pick_actor(self) -> Actor:
        pool = [NOMINAL_ACTORS[0], NOMINAL_ACTORS[1], NOMINAL_ACTORS[2], NOMINAL_ACTORS[3], ATTACK_ACTORS[1]]
        weights = [0.24, 0.28, 0.18, 0.22, 0.08]
        return self.rng.choices(pool, weights=weights, k=1)[0]

    def _pick_command(self, family: str, phase: str, slot: int) -> str:
        if family == "power_sag":
            pool = ["prmDb.PRM_SAVE_FILE", "fileDownlink.SendFile", "cmdSeq.CS_START", "fileManager.AppendFile"]
        elif family == "thermal_excursion":
            pool = ["systemResources.VERSION", "eventLogger.DUMP_FILTER_STATE", "cmdDisp.CMD_TEST_CMD_1", "prmDb.PRM_SAVE_FILE"]
        elif family == "comms_degradation":
            pool = ["fileDownlink.SendPartial", "fileDownlink.SendFile", "fileDownlink.Cancel", "cmdSeq.CS_JOIN_WAIT"]
        elif family == "adcs_instability":
            pool = ["cmdSeq.CS_START", "cmdSeq.CS_STEP", "cmdDisp.CMD_TEST_CMD_1", "cmdDisp.CMD_CLEAR_TRACKING"]
        else:
            pool = ["fileManager.MoveFile", "fileManager.AppendFile", "fileManager.FileSize", "fileDownlink.SendPartial"]
        if phase == "startup":
            return self.rng.choice(["cmdDisp.CMD_NO_OP", "systemResources.VERSION", "eventLogger.DUMP_FILTER_STATE"])
        return pool[slot % len(pool)] if self.rng.random() < 0.55 else self.rng.choice(pool)

    def build_fault(self, count: int, start_ms: int, episode_id: int) -> tuple[str, list[CommandIntent]]:
        family = self._pick_family()
        intents: list[CommandIntent] = []
        session_id = self._session_id(episode_id)
        current_ms = start_ms
        for slot in range(count):
            phase = phase_for_time(current_ms)
            command_name = self._pick_command(family, phase, slot)
            intents.append(
                CommandIntent(
                    episode_id=episode_id,
                    episode_label=FAULT_LABEL,
                    episode_kind="fault",
                    step_id=slot,
                    time_ms=current_ms,
                    session_id=session_id,
                    txn_id=f"{episode_id:04d}-{slot:06d}",
                    actor=self._pick_actor(),
                    command=command_name,
                    args=default_args_for_command(command_name, slot),
                    attack_family=family,
                    phase=phase,
                )
            )
            if family == "comms_degradation":
                current_ms += self.rng.randint(7000, 18000)
            elif family == "storage_pressure":
                current_ms += self.rng.randint(5000, 14000)
            else:
                current_ms += self.rng.randint(8500, 22000)
        return family, intents


class GdsSatelliteSimulator:
    def __init__(self, seed: int, start_ms: int, fault_family: str = "none"):
        self.rng = random.Random(seed)
        self.fault_family = fault_family
        self.state = SatelliteState(
            queue_depth=self.rng.uniform(0.0, 2.5),
            battery_soc=self.rng.uniform(68.0, 90.0),
            bus_voltage_v=self.rng.uniform(27.0, 28.8),
            solar_array_current_a=self.rng.uniform(2.8, 4.8),
            battery_temp_c=self.rng.uniform(12.0, 20.0),
            payload_temp_c=self.rng.uniform(-3.0, 6.0),
            attitude_error_deg=self.rng.uniform(0.18, 0.65),
            wheel_speed_rpm=self.rng.uniform(1100.0, 1900.0),
            link_rssi_dbm=self.rng.uniform(-100.0, -90.0),
            downlink_backlog_mb=self.rng.uniform(8.0, 45.0),
            storage_free_mb=self.rng.uniform(4096.0, 7168.0),
            event_backlog=self.rng.uniform(3.0, 22.0),
            radiation_flux=self.rng.uniform(1.0, 2.2),
            payload_cover_open=0.0,
            heater_on=1.0,
            comms_window_open=0.0,
        )
        self.current_ms = start_ms
        self.sequence_auto = True
        self.sequence_running = False
        self.request_times: deque[float] = deque()
        self.request_signatures: deque[tuple[float, str]] = deque()
        self.recent_commands: deque[tuple[float, str]] = deque()
        self.telemetry_last_ms = {"fprime_a": 0, "fprime_b": 0}
        self.stream_indices = {"fprime_a": 0, "fprime_b": 0}
        self.node_load = {node: self.rng.uniform(0.15, 0.75) for node in ("fprime_a", "fprime_b")}
        self.node_metrics = {node: self._initial_node_metrics(node) for node in ("fprime_a", "fprime_b")}
        self._apply_fault_baseline()
        self._refresh_node_metrics(float(start_ms))

    def _jitter(self, sigma: float) -> float:
        return self.rng.gauss(0.0, sigma)

    def _orbit_angle(self, time_ms: int) -> float:
        return 2.0 * math.pi * (((time_ms / 1000.0) % ORBIT_SECONDS) / ORBIT_SECONDS)

    def _trim_histories(self, now_ms: float) -> None:
        while self.request_times and now_ms - self.request_times[0] > 60000.0:
            self.request_times.popleft()
        while self.request_signatures and now_ms - self.request_signatures[0][0] > 20000.0:
            self.request_signatures.popleft()
        while self.recent_commands and now_ms - self.recent_commands[0][0] > 12000.0:
            self.recent_commands.popleft()

    def _initial_node_metrics(self, node_service: str) -> dict[str, float]:
        del node_service
        return {
            "cpu_total_pct": self.rng.uniform(12.0, 26.0),
            "cpu_00_pct": self.rng.uniform(6.0, 13.0),
            "cpu_01_pct": self.rng.uniform(6.0, 13.0),
            "blockdrv_cycles_total": self.rng.uniform(120.0, 420.0),
            "cmds_dispatched_total": 0.0,
            "cmd_errors_total": 0.0,
            "filedownlink_warnings_total": 0.0,
            "filemanager_errors_total": 0.0,
            "hibuffs_total": 0.0,
            "rg1_max_time_ms": self.rng.uniform(1.4, 3.4),
            "rg2_max_time_ms": self.rng.uniform(1.8, 3.8),
        }

    def _peer_service(self, target_service: str) -> str:
        return "fprime_b" if target_service == "fprime_a" else "fprime_a"

    def _refresh_node_metrics(self, now_ms: float) -> None:
        angle = self._orbit_angle(int(now_ms))
        for node_idx, node_service in enumerate(("fprime_a", "fprime_b")):
            metrics = self.node_metrics[node_service]
            load = self.node_load[node_service]
            cpu_bias = 0.0
            if self.fault_family == "comms_degradation":
                cpu_bias += 7.0
            elif self.fault_family == "storage_pressure":
                cpu_bias += 4.0
            elif self.fault_family == "thermal_excursion":
                cpu_bias += 2.5
            cpu_total = clip(
                16.0
                + 9.0 * load
                + 0.55 * self.state.queue_depth
                + 1.5 * max(math.sin(angle + node_idx * 0.8), 0.0)
                + cpu_bias
                + self._jitter(1.0),
                2.0,
                99.0,
            )
            cpu_00 = clip(cpu_total * (0.50 + self.rng.uniform(-0.06, 0.06)), 0.0, 99.0)
            cpu_01 = clip(cpu_total - cpu_00, 0.0, 99.0)
            metrics["cpu_total_pct"] = cpu_total
            metrics["cpu_00_pct"] = cpu_00
            metrics["cpu_01_pct"] = cpu_01
            metrics["hibuffs_total"] = clip(0.35 * self.state.queue_depth + 1.8 * load, 0.0, 12.0)
            metrics["rg1_max_time_ms"] = clip(1.6 + 1.5 * load + 0.65 * self.state.queue_depth + abs(self._jitter(0.4)), 0.2, 40.0)
            metrics["rg2_max_time_ms"] = clip(2.0 + 1.3 * load + 0.55 * self.state.queue_depth + abs(self._jitter(0.5)), 0.2, 40.0)

    def _record_command_metrics(self, target_service: str, service: str, success: bool, timeout: bool, response_code: int, latency_ms: float) -> None:
        metrics = self.node_metrics[target_service]
        metrics["cmds_dispatched_total"] += 1.0
        metrics["blockdrv_cycles_total"] += 1.0 + max(0.0, latency_ms / 90.0)
        if service in {"fileManager", "fileDownlink", "prmDb"}:
            metrics["blockdrv_cycles_total"] += 2.0
        if (not success) or timeout or response_code != 0:
            metrics["cmd_errors_total"] += 1.0
        if service == "fileDownlink" and ((not success) or timeout or response_code != 0):
            metrics["filedownlink_warnings_total"] += 1.0
        if service == "fileManager" and ((not success) or timeout or response_code != 0):
            metrics["filemanager_errors_total"] += 1.0
        self.node_load[target_service] = clip(self.node_load[target_service] + 0.30 + min(1.6, latency_ms / 240.0), 0.05, 8.0)
        peer_service = self._peer_service(target_service)
        self.node_load[peer_service] = clip(self.node_load[peer_service] + 0.04, 0.05, 8.0)
        self._refresh_node_metrics(float(self.current_ms))

    def _apply_fault_baseline(self) -> None:
        if self.fault_family == "power_sag":
            self.state.battery_soc = self.rng.uniform(18.0, 34.0)
            self.state.bus_voltage_v = self.rng.uniform(24.0, 25.4)
            self.state.solar_array_current_a = self.rng.uniform(1.1, 2.6)
        elif self.fault_family == "thermal_excursion":
            self.state.battery_temp_c = self.rng.uniform(22.0, 29.0)
            self.state.payload_temp_c = self.rng.uniform(18.0, 28.0)
            self.state.heater_on = 1.0
        elif self.fault_family == "comms_degradation":
            self.state.link_rssi_dbm = self.rng.uniform(-109.0, -98.0)
            self.state.queue_depth = self.rng.uniform(4.0, 7.5)
        elif self.fault_family == "adcs_instability":
            self.state.attitude_error_deg = self.rng.uniform(1.4, 2.8)
            self.state.wheel_speed_rpm = self.rng.uniform(2600.0, 3600.0)
        elif self.fault_family == "storage_pressure":
            self.state.downlink_backlog_mb = self.rng.uniform(160.0, 240.0)
            self.state.storage_free_mb = self.rng.uniform(420.0, 1200.0)
            self.state.event_backlog = self.rng.uniform(48.0, 110.0)
            self.state.queue_depth = self.rng.uniform(3.0, 6.0)

    def _apply_fault_drift(self, dt_s: float) -> None:
        if self.fault_family == "power_sag":
            self.state.solar_array_current_a = clip(self.state.solar_array_current_a - 0.10 * dt_s + self._jitter(0.05), 0.0, 8.5)
            self.state.battery_soc = clip(self.state.battery_soc - 0.030 * dt_s + self._jitter(0.05), 5.0, 100.0)
            self.state.bus_voltage_v = clip(self.state.bus_voltage_v - 0.015 * dt_s + self._jitter(0.03), 21.0, 30.0)
        elif self.fault_family == "thermal_excursion":
            self.state.heater_on = 1.0
            self.state.battery_temp_c = clip(self.state.battery_temp_c + 0.018 * dt_s + self._jitter(0.08), -20.0, 50.0)
            self.state.payload_temp_c = clip(self.state.payload_temp_c + 0.026 * dt_s + self._jitter(0.10), -35.0, 75.0)
        elif self.fault_family == "comms_degradation":
            self.state.link_rssi_dbm = clip(self.state.link_rssi_dbm - 0.05 * dt_s + self._jitter(1.3), -120.0, -45.0)
            self.state.queue_depth = clip(self.state.queue_depth + 0.02 * dt_s + abs(self._jitter(0.07)), 0.0, 20.0)
        elif self.fault_family == "adcs_instability":
            self.state.wheel_speed_rpm = clip(self.state.wheel_speed_rpm + 14.0 * dt_s + self._jitter(4.0), 250.0, 6000.0)
            self.state.attitude_error_deg = clip(self.state.attitude_error_deg + 0.010 * dt_s + abs(self._jitter(0.08)), 0.05, 12.0)
        elif self.fault_family == "storage_pressure":
            self.state.downlink_backlog_mb = clip(self.state.downlink_backlog_mb + 0.65 * dt_s + abs(self._jitter(0.20)), 0.0, 500.0)
            self.state.storage_free_mb = clip(self.state.storage_free_mb - 0.45 * dt_s + self._jitter(0.15), 256.0, 8192.0)
            self.state.event_backlog = clip(self.state.event_backlog + 0.08 * dt_s + abs(self._jitter(0.08)), 0.0, 500.0)

    def advance_to(self, time_ms: int) -> None:
        if time_ms <= self.current_ms:
            self.current_ms = time_ms
            self._refresh_node_metrics(float(time_ms))
            return
        dt_s = (time_ms - self.current_ms) / 1000.0
        angle = self._orbit_angle(time_ms)
        sun = max(math.sin(angle), 0.0)
        self.state.solar_array_current_a = clip(0.4 + 6.1 * sun + self._jitter(0.10), 0.0, 8.5)
        load = 1.0 + 0.12 * self.state.heater_on + 0.08 * self.state.payload_cover_open + 0.13 * self.state.comms_window_open + 0.05 * self.state.queue_depth
        self.state.battery_soc = clip(self.state.battery_soc + dt_s * (0.020 * self.state.solar_array_current_a - 0.012 * load), 5.0, 100.0)
        self.state.bus_voltage_v = clip(23.8 + 0.055 * self.state.battery_soc - 0.13 * load + self._jitter(0.05), 21.0, 30.0)
        self.state.battery_temp_c = clip(
            self.state.battery_temp_c + dt_s * (0.008 * self.state.solar_array_current_a + 0.015 * self.state.heater_on - 0.003 * (self.state.battery_temp_c - 16.0)) + self._jitter(0.08),
            -20.0,
            50.0,
        )
        self.state.payload_temp_c = clip(
            self.state.payload_temp_c + dt_s * (0.005 * self.state.solar_array_current_a + 0.020 * self.state.heater_on + 0.020 * self.state.payload_cover_open - 0.004 * (self.state.payload_temp_c - 2.0)) + self._jitter(0.12),
            -35.0,
            75.0,
        )
        self.state.wheel_speed_rpm = clip(self.state.wheel_speed_rpm + dt_s * (2.0 * self.state.payload_cover_open - 1.0) + self._jitter(2.8), 250.0, 6000.0)
        self.state.attitude_error_deg = clip(0.20 + 0.00035 * self.state.wheel_speed_rpm + 0.018 * self.state.queue_depth + abs(self._jitter(0.05)), 0.05, 12.0)
        comms_contact = 1.0 if phase_for_time(time_ms) == "downlink" else 0.0
        self.state.link_rssi_dbm = clip(-98.0 + 24.0 * comms_contact - 0.18 * self.state.queue_depth + self._jitter(1.0), -120.0, -45.0)
        self.state.radiation_flux = clip(1.4 + 0.8 * (1.0 + math.sin(angle * 2.0 + 0.7)) + self._jitter(0.08), 0.2, 5.0)
        self.state.queue_depth = max(0.0, self.state.queue_depth - dt_s / 10.0)
        self.state.event_backlog = max(0.0, self.state.event_backlog + self._jitter(0.10) - 0.02 * dt_s)
        self.state.comms_window_open = 1.0 if phase_for_time(time_ms) == "downlink" else 0.0
        self._apply_fault_drift(dt_s)
        for node_service in self.node_load:
            self.node_load[node_service] = clip(self.node_load[node_service] - 0.015 * dt_s, 0.05, 8.0)
        self.current_ms = time_ms
        self._trim_histories(float(time_ms))
        self._refresh_node_metrics(float(time_ms))

    def emit_due_telemetry(self, episode_id: int, episode_label: int, episode_kind: str, session_id: str, now_ms: int) -> list[dict[str, Any]]:
        packets: list[dict[str, Any]] = []
        offset = 0
        self._refresh_node_metrics(float(now_ms))
        for node_service, interval in (("fprime_a", 8000), ("fprime_b", 9200)):
            effective_interval = interval * 2 if self.fault_family == "comms_degradation" else interval
            if now_ms - self.telemetry_last_ms[node_service] < effective_interval:
                continue
            self.telemetry_last_ms[node_service] = now_ms + offset
            payload = {key: float(value) for key, value in self.node_metrics[node_service].items()}
            packets.append(
                asdict(
                    PacketFrame(
                        ts_ms=now_ms + offset,
                        packet_kind="telemetry",
                        src=node_service,
                        dst="gds",
                        episode_id=episode_id,
                        episode_label=episode_label,
                        episode_kind=episode_kind,
                        session_id=session_id,
                        txn_id="",
                        service="systemResources",
                        command="systemResources.HK",
                        bytes_on_wire=64 + 8 * len(payload),
                        label=episode_label,
                        attack_family="none",
                        phase=phase_for_time(now_ms),
                        actor="telemetry",
                        actor_role="telemetry",
                        actor_trust=1.0,
                        node_service=node_service,
                        payload=payload,
                    )
                )
            )
            offset += 2
        return packets

    def _signature(self, intent: CommandIntent) -> str:
        return f"{intent.command}|{json.dumps(intent.args, sort_keys=True)}"

    def _gds_validate(self, intent: CommandIntent, now_ms: int) -> tuple[bool, int, str]:
        self._trim_histories(float(now_ms))
        _, _, arg_out, max_violation, reason = inspect_args(intent.command, intent.args)
        signature = self._signature(intent)
        recent_same = sum(1 for ts, sig in self.request_signatures if now_ms - ts < 12000.0 and sig == signature)
        burst_10s = sum(1 for ts in self.request_times if now_ms - ts < 10000.0)
        self.request_times.append(float(now_ms))
        self.request_signatures.append((float(now_ms), signature))
        self.recent_commands.append((float(now_ms), intent.command))
        if intent.command not in COMMAND_SPECS:
            return False, 1, "gds_unknown_command"
        if reason == "schema_mismatch":
            return False, 2, "gds_schema_reject"
        if max_violation > 1.5:
            return False, 2, "gds_range_reject"
        if intent.actor.trust < 0.25 and COMMAND_SPECS[intent.command].service in SENSITIVE_SERVICES and self.rng.random() < 0.90:
            return False, 1, "gds_auth_reject"
        if recent_same >= 2:
            return False, 4, "gds_replay_reject"
        if burst_10s >= 8 and intent.actor.trust < 0.70:
            return False, 4, "gds_rate_reject"
        if arg_out > 0.5 and self.rng.random() < 0.65:
            return False, 2, "gds_range_reject"
        return True, 0, "accepted"

    def _satellite_execute(self, intent: CommandIntent) -> tuple[bool, bool, str, int, int, int, float]:
        spec = COMMAND_SPECS[intent.command]
        service = spec.service
        repeated_recently = sum(1 for ts, command_name in self.recent_commands if self.current_ms - ts < 8000.0 and command_name == intent.command)
        self.state.queue_depth = clip(self.state.queue_depth + 1.0 + (1.2 if repeated_recently >= 2 else 0.0), 0.0, 20.0)
        base_latency = float(spec.base_latency_ms) + 12.0 * self.state.queue_depth + abs(self._jitter(6.0))
        if service == "fileDownlink":
            base_latency += 50.0 + 0.45 * self.state.downlink_backlog_mb
        elif service == "fileManager":
            base_latency += 30.0 + 0.12 * max(0.0, 2048.0 - self.state.storage_free_mb)
        elif service == "prmDb":
            base_latency += 24.0
        elif service == "cmdSeq":
            base_latency += 16.0
        if self.fault_family == "comms_degradation":
            base_latency += 140.0 + 18.0 * self.state.queue_depth
        elif self.fault_family == "storage_pressure":
            base_latency += 35.0 + 0.18 * self.state.downlink_backlog_mb
        elif self.fault_family == "power_sag":
            base_latency += 30.0 if self.state.battery_soc < 25.0 else 0.0
        retries = 1 if self.state.queue_depth > 6.5 or repeated_recently >= 3 else 0
        if self.fault_family == "comms_degradation" and self.rng.random() < 0.30:
            retries = max(retries, 1)
        if retries:
            base_latency += 55.0
        timeout = False
        response_code = 0
        reason = "completed"
        success = True
        response_bytes = int(spec.response_bytes)
        if self.state.queue_depth > 10.0 and self.rng.random() < 0.40:
            timeout = True
            success = False
            response_code = 3
            reason = "queue_timeout"
        elif service == "cmdDisp":
            if intent.command == "cmdDisp.CMD_NO_OP_STRING":
                response_bytes += 8
            elif intent.command == "cmdDisp.CMD_TEST_CMD_1":
                science_load = 4.0 + abs(float(intent.args.get("f32_arg", 0.0))) * 0.06 + float(intent.args.get("u8_arg", 0.0)) * 0.05
                if self.state.battery_soc < 12.0:
                    success = False
                    response_code = 3
                    timeout = True
                    reason = "science_power_timeout"
                elif self.fault_family == "adcs_instability" and self.state.attitude_error_deg > 2.2:
                    success = False
                    response_code = 2
                    reason = "adcs_unstable"
                else:
                    self.state.downlink_backlog_mb = clip(self.state.downlink_backlog_mb + 6.0 + science_load, 0.0, 500.0)
                    self.state.storage_free_mb = clip(self.state.storage_free_mb - 4.0 - 0.3 * science_load, 256.0, 8192.0)
                    self.state.event_backlog = clip(self.state.event_backlog + 0.9, 0.0, 500.0)
                    self.state.payload_temp_c = clip(self.state.payload_temp_c + 0.5, -35.0, 75.0)
                    self.state.attitude_error_deg = clip(self.state.attitude_error_deg + 0.12, 0.05, 12.0)
                    response_bytes += int(10.0 + science_load)
            elif intent.command == "cmdDisp.CMD_CLEAR_TRACKING":
                self.state.event_backlog = max(0.0, self.state.event_backlog - 2.5)
                self.state.attitude_error_deg = clip(self.state.attitude_error_deg * 0.85, 0.05, 12.0)
        elif service == "systemResources":
            response_bytes += 18
        elif service == "eventLogger":
            response_bytes += 12
            self.state.event_backlog = max(0.0, self.state.event_backlog - 1.0)
        elif service == "cmdSeq":
            if intent.command == "cmdSeq.CS_AUTO":
                self.sequence_auto = True
            elif intent.command == "cmdSeq.CS_MANUAL":
                self.sequence_auto = False
            elif intent.command == "cmdSeq.CS_START":
                self.sequence_running = True
                self.state.event_backlog = clip(self.state.event_backlog + 0.7, 0.0, 500.0)
            elif intent.command == "cmdSeq.CS_CANCEL":
                self.sequence_running = False
                self.state.queue_depth = max(0.0, self.state.queue_depth - 1.2)
            elif intent.command == "cmdSeq.CS_STEP":
                if self.sequence_auto:
                    success = False
                    response_code = 2
                    reason = "step_requires_manual"
                else:
                    self.sequence_running = True
                    self.state.event_backlog = clip(self.state.event_backlog + 0.4, 0.0, 500.0)
            elif intent.command == "cmdSeq.CS_VALIDATE":
                if self.fault_family == "storage_pressure" and self.state.storage_free_mb < 700.0:
                    success = False
                    response_code = 2
                    reason = "sequence_missing"
            elif intent.command == "cmdSeq.CS_JOIN_WAIT":
                base_latency += 35.0
        elif service == "prmDb":
            if self.state.downlink_backlog_mb > 200.0:
                success = False
                response_code = 2
                reason = "flush_blocked_backlog"
            elif self.state.storage_free_mb < 512.0:
                success = False
                response_code = 2
                reason = "param_store_full"
            else:
                self.state.storage_free_mb = clip(self.state.storage_free_mb - 6.0, 256.0, 8192.0)
                self.state.event_backlog = clip(self.state.event_backlog + 0.6, 0.0, 500.0)
        elif service == "fileManager":
            if intent.command == "fileManager.ShellCommand" and intent.actor.trust < 0.55 and self.rng.random() < 0.35:
                success = False
                response_code = 2
                reason = "shell_command_rejected"
            elif intent.command == "fileManager.AppendFile":
                if self.state.storage_free_mb < 700.0:
                    success = False
                    response_code = 2
                    reason = "append_no_space"
                else:
                    self.state.storage_free_mb = clip(self.state.storage_free_mb - 12.0, 256.0, 8192.0)
                    self.state.downlink_backlog_mb = clip(self.state.downlink_backlog_mb + 5.0, 0.0, 500.0)
            elif intent.command == "fileManager.MoveFile":
                if self.fault_family == "storage_pressure" and self.rng.random() < 0.35:
                    success = False
                    response_code = 2
                    reason = "move_missing"
                else:
                    self.state.downlink_backlog_mb = clip(self.state.downlink_backlog_mb + 2.0, 0.0, 500.0)
            elif intent.command == "fileManager.RemoveDirectory":
                if self.fault_family == "storage_pressure" and self.rng.random() < 0.35:
                    success = False
                    response_code = 2
                    reason = "cleanup_missing"
                else:
                    self.state.storage_free_mb = clip(self.state.storage_free_mb + 12.0, 256.0, 8192.0)
            elif intent.command == "fileManager.RemoveFile":
                self.state.storage_free_mb = clip(self.state.storage_free_mb + 6.0, 256.0, 8192.0)
            elif intent.command == "fileManager.CreateDirectory":
                self.state.event_backlog = clip(self.state.event_backlog + 0.3, 0.0, 500.0)
            elif intent.command == "fileManager.FileSize":
                response_bytes += 12
        elif service == "fileDownlink":
            requested_mb = 0.0
            if intent.command == "fileDownlink.SendPartial":
                requested_mb = max(1.0, float(intent.args.get("length_kb", 1024.0)) / 1024.0)
            elif intent.command == "fileDownlink.SendFile":
                requested_mb = max(4.0, min(64.0, self.state.downlink_backlog_mb * 0.45))
            if intent.command == "fileDownlink.Cancel":
                self.state.queue_depth = max(0.0, self.state.queue_depth - 1.8)
            elif self.state.comms_window_open < 0.5 or self.state.link_rssi_dbm < -88.0:
                success = False
                response_code = 3 if intent.actor.trust < 0.5 else 2
                timeout = response_code == 3
                reason = "downlink_not_ready"
            elif self.state.downlink_backlog_mb <= 0.5:
                success = False
                response_code = 2
                reason = "downlink_empty"
            elif self.fault_family == "comms_degradation" and self.state.link_rssi_dbm < -96.0:
                success = False
                response_code = 3
                timeout = True
                reason = "downlink_link_degraded"
            else:
                moved = min(requested_mb, self.state.downlink_backlog_mb, 72.0)
                self.state.downlink_backlog_mb = clip(self.state.downlink_backlog_mb - moved, 0.0, 500.0)
                self.state.storage_free_mb = clip(self.state.storage_free_mb + moved * 0.35, 256.0, 8192.0)
                response_bytes += int(moved * 2.2)
        if self.fault_family == "power_sag" and success and service in SENSITIVE_SERVICES and self.state.battery_soc < 20.0 and self.rng.random() < 0.40:
            success = False
            response_code = 3
            timeout = True
            reason = "power_sag_timeout"
        if self.fault_family == "thermal_excursion" and success and service in {"prmDb", "fileManager"} and self.state.payload_temp_c > 36.0 and self.rng.random() < 0.35:
            success = False
            response_code = 2
            reason = "thermal_guard"
        if self.fault_family == "adcs_instability" and success and service in {"cmdSeq", "cmdDisp"} and self.state.attitude_error_deg > 2.2 and self.rng.random() < 0.45:
            success = False
            response_code = 2
            reason = "adcs_unstable"
        latency_ms = base_latency + retries * 45.0 + (40.0 if timeout else 0.0)
        self.state.queue_depth = max(0.0, self.state.queue_depth - 0.6)
        return success, timeout, reason, response_code, response_bytes, retries, latency_ms

    def packets_for_intent(self, intent: CommandIntent) -> list[dict[str, Any]]:
        self.advance_to(intent.time_ms)
        packets = self.emit_due_telemetry(intent.episode_id, intent.episode_label, intent.episode_kind, intent.session_id, intent.time_ms)
        service = COMMAND_SPECS[intent.command].service if intent.command in COMMAND_SPECS else intent.command.split(".")[0]
        target_service = target_service_for_actor(intent.actor.name, intent.step_id)
        target_stream_id = f"{target_service}:50050"
        target_stream_index = self.stream_indices[target_service]
        self.stream_indices[target_service] += 1
        req_bytes = int((COMMAND_SPECS[intent.command].request_bytes if intent.command in COMMAND_SPECS else 48 + 2 * len(intent.command)) + 4 * len(intent.args))
        request_ts = intent.time_ms + self.rng.randint(0, 8)
        packets.append(
            asdict(
                PacketFrame(
                    ts_ms=request_ts,
                    packet_kind="request",
                    src=intent.actor.name,
                    dst=target_service,
                    episode_id=intent.episode_id,
                    episode_label=intent.episode_label,
                    episode_kind=intent.episode_kind,
                    session_id=intent.session_id,
                    txn_id=intent.txn_id,
                    service=service,
                    command=intent.command,
                    bytes_on_wire=req_bytes,
                    label=intent.episode_label,
                    attack_family=intent.attack_family,
                    phase=intent.phase,
                    actor=intent.actor.name,
                    actor_role=intent.actor.role,
                    actor_trust=intent.actor.trust,
                    target_service=target_service,
                    target_stream_id=target_stream_id,
                    target_stream_index=target_stream_index,
                    args=intent.args,
                )
            )
        )
        gds_accept, gds_code, gds_reason = self._gds_validate(intent, request_ts)
        if not gds_accept:
            final_ts = request_ts + self.rng.randint(12, 30)
            packets.append(
                asdict(
                    PacketFrame(
                        ts_ms=final_ts,
                        packet_kind="final",
                        src="gds",
                        dst=intent.actor.name,
                        episode_id=intent.episode_id,
                        episode_label=intent.episode_label,
                        episode_kind=intent.episode_kind,
                        session_id=intent.session_id,
                        txn_id=intent.txn_id,
                        service=service,
                        command=intent.command,
                        bytes_on_wire=56,
                        label=intent.episode_label,
                        attack_family=intent.attack_family,
                        phase=intent.phase,
                        actor=intent.actor.name,
                        actor_role=intent.actor.role,
                        actor_trust=intent.actor.trust,
                        target_service=target_service,
                        target_stream_id=target_stream_id,
                        target_stream_index=target_stream_index,
                        response_code=gds_code,
                        gds_accept=0,
                        sat_success=0,
                        timeout=0,
                        reason=gds_reason,
                    )
                )
            )
            self.current_ms = max(self.current_ms, final_ts)
            return packets
        uplink_ts = request_ts + self.rng.randint(8, 18)
        packets.append(
            asdict(
                PacketFrame(
                    ts_ms=uplink_ts,
                    packet_kind="uplink",
                    src="gds",
                    dst=target_service,
                    episode_id=intent.episode_id,
                    episode_label=intent.episode_label,
                    episode_kind=intent.episode_kind,
                    session_id=intent.session_id,
                    txn_id=intent.txn_id,
                    service=service,
                    command=intent.command,
                    bytes_on_wire=req_bytes + 6,
                    label=intent.episode_label,
                    attack_family=intent.attack_family,
                    phase=intent.phase,
                    actor=intent.actor.name,
                    actor_role=intent.actor.role,
                    actor_trust=intent.actor.trust,
                    target_service=target_service,
                    target_stream_id=target_stream_id,
                    target_stream_index=target_stream_index,
                )
            )
        )
        sat_success, timeout, sat_reason, response_code, response_bytes, retries, latency_ms = self._satellite_execute(intent)
        self._record_command_metrics(target_service, service, sat_success, timeout, response_code, latency_ms)
        for retry_idx in range(retries):
            retry_ts = uplink_ts + 45 * (retry_idx + 1) + self.rng.randint(4, 10)
            packets.append(
                asdict(
                    PacketFrame(
                        ts_ms=retry_ts,
                        packet_kind="uplink",
                        src="gds",
                        dst=target_service,
                        episode_id=intent.episode_id,
                        episode_label=intent.episode_label,
                        episode_kind=intent.episode_kind,
                        session_id=intent.session_id,
                        txn_id=intent.txn_id,
                        service=service,
                        command=intent.command,
                        bytes_on_wire=req_bytes + 6,
                        label=intent.episode_label,
                        attack_family=intent.attack_family,
                        phase=intent.phase,
                        actor=intent.actor.name,
                        actor_role=intent.actor.role,
                        actor_trust=intent.actor.trust,
                        target_service=target_service,
                        target_stream_id=target_stream_id,
                        target_stream_index=target_stream_index,
                    )
                )
            )
        if not timeout:
            sat_ts = uplink_ts + int(latency_ms)
            packets.append(
                asdict(
                    PacketFrame(
                        ts_ms=sat_ts,
                        packet_kind="sat_response",
                        src=target_service,
                        dst="gds",
                        episode_id=intent.episode_id,
                        episode_label=intent.episode_label,
                        episode_kind=intent.episode_kind,
                        session_id=intent.session_id,
                        txn_id=intent.txn_id,
                        service=service,
                        command=intent.command,
                        bytes_on_wire=response_bytes,
                        label=intent.episode_label,
                        attack_family=intent.attack_family,
                        phase=intent.phase,
                        actor=intent.actor.name,
                        actor_role=intent.actor.role,
                        actor_trust=intent.actor.trust,
                        target_service=target_service,
                        target_stream_id=target_stream_id,
                        target_stream_index=target_stream_index,
                        response_code=response_code,
                        sat_success=1 if sat_success else 0,
                        reason=sat_reason,
                    )
                )
            )
            final_ts = sat_ts + self.rng.randint(6, 18)
        else:
            final_ts = uplink_ts + int(latency_ms)
        final_bytes = max(56, int(response_bytes * 0.70)) if not timeout else 56
        packets.append(
            asdict(
                PacketFrame(
                    ts_ms=final_ts,
                    packet_kind="final",
                    src=target_service if gds_accept else "gds",
                    dst=intent.actor.name,
                    episode_id=intent.episode_id,
                    episode_label=intent.episode_label,
                    episode_kind=intent.episode_kind,
                    session_id=intent.session_id,
                    txn_id=intent.txn_id,
                    service=service,
                    command=intent.command,
                    bytes_on_wire=final_bytes,
                    label=intent.episode_label,
                    attack_family=intent.attack_family,
                    phase=intent.phase,
                    actor=intent.actor.name,
                    actor_role=intent.actor.role,
                    actor_trust=intent.actor.trust,
                    target_service=target_service,
                    target_stream_id=target_stream_id,
                    target_stream_index=target_stream_index,
                    response_code=response_code,
                    gds_accept=1,
                    sat_success=1 if sat_success else 0,
                    timeout=1 if timeout else 0,
                    reason=sat_reason,
                    response_direction_seen=0 if timeout else 1,
                    final_observed_on_wire=0 if timeout else 1,
                    txn_warning_events=1 if (service == "fileDownlink" and not sat_success and not timeout) else 0,
                    txn_error_events=0 if sat_success else 1,
                )
            )
        )
        self.current_ms = max(self.current_ms, final_ts)
        return packets


class EpisodeFactory:
    def __init__(self, seed: int):
        self.seed = seed

    def _episode_start_ms(self, episode_id: int) -> int:
        return episode_id * 7200000 + (episode_id % 3) * 180000

    def _build_one_episode(self, episode_id: int, label: int, count: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        start_ms = self._episode_start_ms(episode_id)
        if label == 0:
            intents = MissionScheduler(self.seed + episode_id).build_nominal(count, start_ms, episode_id)
            simulator = GdsSatelliteSimulator(self.seed + 101 * episode_id, start_ms)
        elif label == 1:
            intents = AttackFuzzer(self.seed + 5000 + episode_id).build_attack(count, start_ms, episode_id)
            simulator = GdsSatelliteSimulator(self.seed + 101 * episode_id, start_ms)
        else:
            fault_family, intents = FaultFuzzer(self.seed + 9000 + episode_id).build_fault(count, start_ms, episode_id)
            simulator = GdsSatelliteSimulator(self.seed + 101 * episode_id, start_ms, fault_family=fault_family)
        packets: list[dict[str, Any]] = []
        for intent in intents:
            packets.extend(simulator.packets_for_intent(intent))
        transactions = packets_to_transactions(packets)
        rows = transactions_to_rows(transactions, reset_key=SYNTHETIC_HISTORY_RESET_KEY)
        if label == 0:
            for row in rows:
                if str(row.get("reason", "")) in FAULT_REASONS:
                    row["label"] = FAULT_LABEL
                    row["label_name"] = class_name(FAULT_LABEL)
        return packets, transactions, rows

    def build_dataset(self, total_rows: int, nominal_ratio: float) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
        nominal_target = int(round(total_rows * nominal_ratio))
        anomaly_target = int(total_rows - nominal_target)
        cyber_target = int(round(anomaly_target * 0.65))
        fault_target = int(anomaly_target - cyber_target)
        packets: list[dict[str, Any]] = []
        transactions: list[dict[str, Any]] = []
        rows: list[dict[str, Any]] = []
        episode_id = 0
        episode_counts = {"benign": 0, "cyber": 0, "fault": 0}
        for label, target in [(0, nominal_target), (1, cyber_target), (FAULT_LABEL, fault_target)]:
            produced = 0
            while produced < target:
                count = min(ROWS_PER_EPISODE, target - produced)
                ep_packets, ep_transactions, ep_rows = self._build_one_episode(episode_id, label, count)
                packets.extend(ep_packets)
                transactions.extend(ep_transactions)
                rows.extend(ep_rows)
                produced += len(ep_rows)
                episode_counts[class_name(label)] += 1
                episode_id += 1
        episode_signature_report = build_episode_signature_report(transactions)
        assert_diverse_episode_signatures(episode_signature_report)
        summary = summarize_rows(rows)
        summary.update(
            {
                "packet_count": len(packets),
                "transaction_count": len(transactions),
                "nominal_target": nominal_target,
                "cyber_target": cyber_target,
                "fault_target": fault_target,
                "episode_counts": episode_counts,
                "episode_signature_summary": episode_signature_report["summary"],
                "group_key": "episode_id",
                "history_featurization": history_featurization_report("transactions", SYNTHETIC_HISTORY_RESET_KEY),
            }
        )
        return packets, transactions, rows, summary


def summarize_rows(rows: list[dict[str, Any]]) -> dict[str, Any]:
    label_count = Counter(class_name(int(row["label"])) for row in rows)
    command_count = Counter(str(row["command"]) for row in rows)
    service_count = Counter(str(row["service"]) for row in rows)
    scenario_count = Counter(str(row["attack_family"]) for row in rows)
    episode_kind_count = Counter(str(row.get("episode_kind", class_name(int(row["episode_label"])))) for row in rows)
    episode_ids = {int(row["episode_id"]) for row in rows}
    return {
        "rows": len(rows),
        "episodes": len(episode_ids),
        "labels": dict(label_count),
        "episode_kinds": dict(episode_kind_count),
        "services": dict(service_count),
        "commands": dict(command_count),
        "scenario_families": dict(scenario_count),
        "has_real_file_downlink": command_count.get("fileDownlink.SendPartial", 0) > 0 or command_count.get("fileDownlink.SendFile", 0) > 0,
    }


def write_jsonl(path: str | Path, rows: list[dict[str, Any]]) -> None:
    target = Path(path)
    target.parent.mkdir(parents=True, exist_ok=True)
    with target.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def read_jsonl(path: str | Path) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    with Path(path).open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def build_nominal_preview(rows: int, seed: int) -> list[dict[str, Any]]:
    preview: list[dict[str, Any]] = []
    scheduler = MissionScheduler(seed)
    for intent in scheduler.build_nominal(rows, 0, 0):
        preview.append(
            {
                "time_ms": intent.time_ms,
                "phase": intent.phase,
                "actor": intent.actor.name,
                "command": intent.command,
                "args": intent.args,
            }
        )
    return preview


def inspect_schedule(rows: int, seed: int, contains: str) -> dict[str, Any]:
    preview = build_nominal_preview(rows, seed)
    counts = Counter(item["command"] for item in preview)
    return {
        "rows": rows,
        "contains": contains,
        "contains_count": counts.get(contains, 0),
        "command_counts": dict(counts),
        "preview": preview[: min(24, len(preview))],
    }


def rows_to_matrix(rows: list[dict[str, Any]], feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X = np.array([vector_from_row(row, feature_names) for row in rows], dtype=float)
    y = np.array([int(row["label"]) for row in rows], dtype=int)
    return X, y


def poster_model_feature_layouts() -> dict[str, dict[str, Any]]:
    feature_names = list(POSTER_DEFAULT_REQUEST_ONLY_BASELINE_FEATURE_NAMES)
    return {
        "neural_net": {
            "feature_tier": POSTER_DEFAULT_PRIMARY_MODEL_FEATURE_TIER,
            "feature_names": feature_names,
        }
    }


def feature_sets_report(
    active_blue_feature_policy_name: str | None = None,
    *,
    training_path_name: str = DEFAULT_TRAINING_PATH_NAME,
) -> dict[str, Any]:
    resolved_policy_name = active_blue_feature_policy_name or default_blue_feature_policy_name(training_path_name)
    if is_legacy_training_path(training_path_name):
        primary_model_feature_names = list(PRIMARY_MODEL_FEATURE_NAMES)
        novelty_feature_names = list(NOVELTY_FEATURE_NAMES)
        tiers = {name: list(feature_names) for name, feature_names in FEATURE_TIER_FEATURE_NAMES.items()}
        models = {
            model_name: {
                "feature_tier": str(layout["feature_tier"]),
                "feature_names": list(layout["feature_names"]),
            }
            for model_name, layout in MODEL_FEATURE_LAYOUTS.items()
        }
        baselines = {
            "command_only": {
                "feature_names": list(COMMAND_ONLY_BASELINE_FEATURE_NAMES),
                "near_perfect_threshold": float(COMMAND_ONLY_BASELINE_NEAR_PERFECT_THRESHOLD),
            },
            "request_only": {
                "feature_names": list(REQUEST_ONLY_BASELINE_FEATURE_NAMES),
                "near_perfect_threshold": float(REQUEST_ONLY_BASELINE_NEAR_PERFECT_THRESHOLD),
            },
            "outcome_only": {
                "feature_names": list(OUTCOME_ONLY_BASELINE_FEATURE_NAMES),
                "near_perfect_threshold": float(OUTCOME_ONLY_BASELINE_NEAR_PERFECT_THRESHOLD),
            },
            "protocol_only": {
                "feature_names": list(PROTOCOL_ONLY_BASELINE_FEATURE_NAMES),
                "near_perfect_threshold": float(PROTOCOL_ONLY_BASELINE_NEAR_PERFECT_THRESHOLD),
            },
            "raw_protocol_shortcuts": {
                "feature_names": list(RAW_PROTOCOL_SHORTCUT_BASELINE_FEATURE_NAMES),
                "near_perfect_threshold": float(RAW_PROTOCOL_SHORTCUT_BASELINE_NEAR_PERFECT_THRESHOLD),
            },
        }
    else:
        primary_model_feature_names = list(POSTER_DEFAULT_REQUEST_ONLY_BASELINE_FEATURE_NAMES)
        novelty_feature_names = []
        tiers = {
            "request": list(primary_model_feature_names),
            POSTER_DEFAULT_PRIMARY_MODEL_FEATURE_TIER: list(primary_model_feature_names),
        }
        models = {
            model_name: {
                "feature_tier": str(layout["feature_tier"]),
                "feature_names": list(layout["feature_names"]),
            }
            for model_name, layout in poster_model_feature_layouts().items()
        }
        baselines = {
            "command_only": {
                "feature_names": list(POSTER_DEFAULT_COMMAND_BASELINE_FEATURE_NAMES),
                "near_perfect_threshold": float(COMMAND_ONLY_BASELINE_NEAR_PERFECT_THRESHOLD),
            },
            "request_only": {
                "feature_names": list(POSTER_DEFAULT_REQUEST_ONLY_BASELINE_FEATURE_NAMES),
                "near_perfect_threshold": float(REQUEST_ONLY_BASELINE_NEAR_PERFECT_THRESHOLD),
            },
            "protocol_only": {
                "feature_names": list(PROTOCOL_ONLY_BASELINE_FEATURE_NAMES),
                "near_perfect_threshold": float(PROTOCOL_ONLY_BASELINE_NEAR_PERFECT_THRESHOLD),
            },
            "raw_protocol_shortcuts": {
                "feature_names": list(RAW_PROTOCOL_SHORTCUT_BASELINE_FEATURE_NAMES),
                "near_perfect_threshold": float(RAW_PROTOCOL_SHORTCUT_BASELINE_NEAR_PERFECT_THRESHOLD),
            },
        }
    active_blue_feature_policy = validate_blue_feature_names(
        primary_model_feature_names,
        resolved_policy_name,
    )
    return {
        "tiers": tiers,
        "models": models,
        "baselines": baselines,
        "primary_model_feature_names": list(primary_model_feature_names),
        "novelty_feature_names": list(novelty_feature_names),
        "blue_feature_policies": {
            "available": available_blue_feature_policies(),
            "active": active_blue_feature_policy,
        },
        "training_path": training_path_summary(
            training_path_name,
            blue_feature_policy_name=resolved_policy_name,
        ),
    }


def novelty_adaptation_report() -> dict[str, Any]:
    return {
        "mode": NOVELTY_ADAPTATION_DISABLED,
        "label_blind": True,
        "uses_ground_truth_labels": False,
        "reason": NOVELTY_ADAPTATION_DISABLED_REASON,
    }


def episode_kind_for_records(records: list[dict[str, Any]]) -> str:
    if not records:
        return "unknown"
    return class_name_for_record(records[0])


def class_name_for_record(record: dict[str, Any]) -> str:
    episode_kind = str(record.get("episode_kind", "")).strip()
    if episode_kind in CLASS_NAMES:
        return episode_kind
    label_name = str(record.get("label_name", "")).strip()
    if label_name in CLASS_NAMES:
        return label_name
    audit_context = record.get("audit_context")
    if isinstance(audit_context, dict):
        audit_label_name = str(audit_context.get("label_name", "")).strip()
        if audit_label_name in CLASS_NAMES:
            return audit_label_name
        if audit_context.get("label") not in (None, ""):
            return class_name(int(audit_context["label"]))
    if "episode_label" in record:
        return class_name(int(record["episode_label"]))
    return class_name(int(record["label"]))


def build_field_overlap_report(
    rows: list[dict[str, Any]],
    *,
    field_name: str,
    family_key: str,
    item_key: str,
    items_key: str,
) -> dict[str, Any]:
    counts: dict[str, dict[str, int]] = {}
    row_total = 0
    for row in rows:
        field_value = str(row.get(field_name, "")).strip()
        if not field_value:
            continue
        class_name_value = class_name_for_record(row)
        class_counts = counts.setdefault(field_value, {name: 0 for name in CLASS_NAMES})
        if class_name_value in class_counts:
            class_counts[class_name_value] += 1
        row_total += 1

    shared_rows = 0
    shared_any = 0
    shared_all = 0
    exclusive = 0
    items: list[dict[str, Any]] = []
    for field_value in sorted(counts):
        class_counts = counts[field_value]
        shared_classes = [name for name in CLASS_NAMES if int(class_counts.get(name, 0)) > 0]
        active_count = len(shared_classes)
        total = sum(class_counts.values())
        if active_count >= 2:
            shared_any += 1
            shared_rows += total
        if active_count == len(CLASS_NAMES):
            shared_all += 1
        if active_count == 1:
            exclusive += 1
        dominant = max(class_counts.values()) if class_counts else 0
        items.append(
            {
                item_key: field_value,
                "classes": dict(class_counts),
                "shared_classes": shared_classes,
                "shared_class_count": active_count,
                "exclusive": active_count == 1,
                "dominant_class_share": 0.0 if total <= 0 else round(dominant / total, 4),
                "rows": total,
            }
        )

    total_items = len(counts)
    return {
        "family_key": family_key,
        "field_name": field_name,
        "summary": {
            "rows": row_total,
            "class_rows": {
                class_name_value: sum(item_counts[class_name_value] for item_counts in counts.values())
                for class_name_value in CLASS_NAMES
            },
            "total_values": total_items,
            "values_shared_by_at_least_two_classes": shared_any,
            "values_shared_by_all_classes": shared_all,
            "exclusive_values": exclusive,
            "overlap_ratio": 0.0 if total_items == 0 else round(shared_any / total_items, 4),
            "shared_row_fraction": 0.0 if row_total == 0 else round(shared_rows / row_total, 4),
        },
        items_key: items,
    }


def evaluate_episode_signature_diversity(report: dict[str, Any]) -> dict[str, Any]:
    violations: list[str] = []
    per_class_checks: dict[str, dict[str, Any]] = {}
    per_class = report.get("per_class", {})
    for class_name_value in ("cyber", "fault"):
        class_summary = dict(per_class.get(class_name_value, {}))
        episode_count = int(class_summary.get("episodes", 0))
        unique_count = int(class_summary.get("unique_signatures", 0))
        unique_ratio = float(class_summary.get("unique_ratio", 0.0))
        max_duplicate = int(class_summary.get("max_duplicate_group_count", 0))
        class_violations: list[str] = []
        if episode_count > 1 and unique_count <= 1:
            class_violations.append(f"{class_name_value} collapsed to one signature across {episode_count} groups")
        elif episode_count >= 4 and unique_ratio < 0.75:
            class_violations.append(f"{class_name_value} unique_ratio={unique_ratio:.2f} across {episode_count} groups")
        if episode_count >= 4 and max_duplicate > max(1, math.ceil(episode_count * 0.25)):
            class_violations.append(f"{class_name_value} max_duplicate_group_count={max_duplicate}")
        violations.extend(class_violations)
        per_class_checks[class_name_value] = {
            "episodes": episode_count,
            "unique_signatures": unique_count,
            "unique_ratio": unique_ratio,
            "max_duplicate_group_count": max_duplicate,
            "passed": not class_violations,
            "violations": class_violations,
        }
    return {
        "policy": "cyber_and_fault_must_not_collapse_to_one_signature_and_must_maintain_diversity",
        "passed": not violations,
        "violations": violations,
        "per_class": per_class_checks,
    }


def quantize_audit_value(value: Any, bucket_size: float) -> int | float:
    numeric = float(value)
    if bucket_size <= 0.0:
        return round(numeric, 6)
    bucketed = round(numeric / bucket_size) * bucket_size
    if math.isclose(bucket_size, round(bucket_size)):
        return int(round(bucketed))
    return round(bucketed, 6)


def build_request_tuple_purity_report(
    rows: list[dict[str, Any]],
    *,
    feature_names: list[str] = REQUEST_TUPLE_PURITY_FEATURE_NAMES,
    bucket_sizes: dict[str, float] | None = None,
) -> dict[str, Any]:
    buckets = dict(bucket_sizes or REQUEST_TUPLE_PURITY_BUCKETS)
    tuple_counts: dict[tuple[Any, ...], dict[str, int]] = {}
    tuple_values: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in rows:
        tuple_key = tuple(
            quantize_audit_value(row.get(feature_name, 0.0), float(buckets.get(feature_name, 0.1)))
            for feature_name in feature_names
        )
        tuple_values.setdefault(
            tuple_key,
            {
                feature_name: tuple_key[index]
                for index, feature_name in enumerate(feature_names)
            },
        )
        class_counts = tuple_counts.setdefault(tuple_key, {name: 0 for name in CLASS_NAMES})
        class_name_value = class_name_for_record(row)
        if class_name_value in class_counts:
            class_counts[class_name_value] += 1

    pure_tuple_count = 0
    pure_row_count = 0
    majority_row_count = 0
    tuple_rows: list[dict[str, Any]] = []
    total_rows = 0
    for tuple_key, class_counts in tuple_counts.items():
        rows_for_tuple = sum(class_counts.values())
        total_rows += rows_for_tuple
        dominant_class = max(class_counts, key=class_counts.get) if class_counts else "benign"
        dominant_rows = max(class_counts.values()) if class_counts else 0
        majority_row_count += dominant_rows
        shared_classes = [name for name in CLASS_NAMES if int(class_counts.get(name, 0)) > 0]
        pure = len(shared_classes) == 1
        if pure:
            pure_tuple_count += 1
            pure_row_count += rows_for_tuple
        tuple_rows.append(
            {
                "values": tuple_values[tuple_key],
                "classes": dict(class_counts),
                "rows": rows_for_tuple,
                "shared_classes": shared_classes,
                "pure": pure,
                "dominant_class": dominant_class,
                "dominant_class_share": 0.0 if rows_for_tuple <= 0 else round(dominant_rows / rows_for_tuple, 4),
            }
        )
    tuple_rows.sort(
        key=lambda item: (
            -int(item["rows"]),
            -float(item["dominant_class_share"]),
            json.dumps(item["values"], sort_keys=True),
        )
    )
    unique_tuples = len(tuple_counts)
    return {
        "feature_names": list(feature_names),
        "bucket_sizes": {
            feature_name: float(buckets.get(feature_name, 0.1))
            for feature_name in feature_names
        },
        "summary": {
            "rows": total_rows,
            "unique_tuples": unique_tuples,
            "pure_tuples": pure_tuple_count,
            "shared_tuples": unique_tuples - pure_tuple_count,
            "pure_tuple_ratio": 0.0 if unique_tuples == 0 else round(pure_tuple_count / unique_tuples, 4),
            "pure_row_fraction": 0.0 if total_rows == 0 else round(pure_row_count / total_rows, 4),
            "majority_row_fraction": 0.0 if total_rows == 0 else round(majority_row_count / total_rows, 4),
        },
        "top_tuples": tuple_rows[:25],
    }


def class_group_counts(group_ids: list[int], grouped: dict[int, list[dict[str, Any]]]) -> dict[str, int]:
    counts = {name: 0 for name in CLASS_NAMES}
    for group_id in group_ids:
        kind = episode_kind_for_records(grouped[group_id])
        counts[kind] = counts.get(kind, 0) + 1
    return counts


def group_records(records: list[dict[str, Any]], group_key: str) -> dict[int, list[dict[str, Any]]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(int(record[group_key]), []).append(record)
    return grouped


def grouped_ids_by_kind(grouped: dict[int, list[dict[str, Any]]]) -> dict[str, list[int]]:
    by_kind: dict[str, list[int]] = {name: [] for name in CLASS_NAMES}
    for group_id in sorted(grouped):
        kind = episode_kind_for_records(grouped[group_id])
        by_kind.setdefault(kind, []).append(group_id)
    return by_kind


def grouped_rows_for_ids(group_ids: list[int], grouped: dict[int, list[dict[str, Any]]]) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for group_id in group_ids:
        out.extend(grouped[group_id])
    return out


def evaluation_seed_sequence(seed: int, count: int = DEFAULT_EVALUATION_SEED_COUNT) -> list[int]:
    return [seed + offset for offset in range(count)]


def stable_text_seed(value: str) -> int:
    return sum((idx + 1) * ord(ch) for idx, ch in enumerate(value))


def grouped_record_split(
    records: list[dict[str, Any]],
    seed: int,
    group_key: str = "episode_id",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    grouped: dict[int, list[dict[str, Any]]] = {}
    for record in records:
        grouped.setdefault(int(record[group_key]), []).append(record)
    by_kind: dict[str, list[int]] = {name: [] for name in CLASS_NAMES}
    for group_id in sorted(grouped):
        kind = episode_kind_for_records(grouped[group_id])
        by_kind.setdefault(kind, []).append(group_id)

    missing_kinds = [name for name in CLASS_NAMES if not by_kind.get(name)]
    if missing_kinds:
        raise SystemExit(
            "Need grouped episodes spanning benign, cyber, and fault for evaluation. "
            f"episode_counts={json.dumps({name: len(by_kind.get(name, [])) for name in CLASS_NAMES}, separators=(',', ':'))} "
            f"missing={json.dumps(missing_kinds, separators=(',', ':'))}"
        )

    too_small = {name: len(group_ids) for name, group_ids in by_kind.items() if len(group_ids) < 3}
    if too_small:
        raise SystemExit(
            "Need at least 3 episodes per class for grouped evaluation. "
            f"episode_counts={json.dumps({name: len(by_kind.get(name, [])) for name in CLASS_NAMES}, separators=(',', ':'))} "
            f"insufficient={json.dumps(too_small, separators=(',', ':'))}"
        )

    rng = random.Random(seed)
    for group_ids in by_kind.values():
        rng.shuffle(group_ids)

    def heldout_group_target(group_count: int) -> int:
        if group_count >= 15:
            return 3
        if group_count >= 8:
            return 2
        return 1

    test_group_ids: list[int] = []
    calib_group_ids: list[int] = []
    base_group_ids: list[int] = []
    heldout_groups_per_class: dict[str, int] = {}
    for kind in CLASS_NAMES:
        group_ids = by_kind[kind]
        heldout_count = heldout_group_target(len(group_ids))
        heldout_groups_per_class[kind] = heldout_count
        test_group_ids.extend(group_ids[:heldout_count])
        calib_group_ids.extend(group_ids[heldout_count : heldout_count * 2])
        base_group_ids.extend(group_ids[heldout_count * 2 :])

    def rows_for(ids: list[int]) -> list[dict[str, Any]]:
        out: list[dict[str, Any]] = []
        for group_id in ids:
            out.extend(grouped[group_id])
        return out

    base_class_group_counts = class_group_counts(base_group_ids, grouped)
    calibration_class_group_counts = class_group_counts(calib_group_ids, grouped)
    test_class_group_counts = class_group_counts(test_group_ids, grouped)
    missing_split_support = {
        "base": [name for name, count in base_class_group_counts.items() if count < 1],
        "calibration": [name for name, count in calibration_class_group_counts.items() if count < 1],
        "test": [name for name, count in test_class_group_counts.items() if count < 1],
    }
    if any(missing_split_support.values()):
        raise SystemExit(
            "Grouped split could not allocate at least one benign, cyber, and fault episode to base, calibration, and test. "
            f"episode_counts={json.dumps({name: len(by_kind.get(name, [])) for name in CLASS_NAMES}, separators=(',', ':'))} "
            f"missing={json.dumps(missing_split_support, separators=(',', ':'))}"
        )

    summary = {
        "group_key": group_key,
        "base_groups": len(base_group_ids),
        "calibration_groups": len(calib_group_ids),
        "test_groups": len(test_group_ids),
        "base_rows": sum(len(grouped[group_id]) for group_id in base_group_ids),
        "calibration_rows": sum(len(grouped[group_id]) for group_id in calib_group_ids),
        "test_rows": sum(len(grouped[group_id]) for group_id in test_group_ids),
        "heldout_groups_per_class": heldout_groups_per_class,
        "base_class_group_counts": base_class_group_counts,
        "calibration_class_group_counts": calibration_class_group_counts,
        "test_class_group_counts": test_class_group_counts,
    }
    if group_key == "episode_id":
        summary["base_episodes"] = summary["base_groups"]
        summary["calibration_episodes"] = summary["calibration_groups"]
        summary["test_episodes"] = summary["test_groups"]
        summary["base_class_episode_counts"] = base_class_group_counts
        summary["calibration_class_episode_counts"] = calibration_class_group_counts
        summary["test_class_episode_counts"] = test_class_group_counts
    return rows_for(base_group_ids), rows_for(calib_group_ids), rows_for(test_group_ids), summary


def grouped_row_split(
    rows: list[dict[str, Any]],
    seed: int,
    group_key: str = "episode_id",
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any]]:
    return grouped_record_split(rows, seed=seed, group_key=group_key)


def history_featurization_report(source: str, group_key: str | None) -> dict[str, Any]:
    return {
        "source": source,
        "group_key": group_key,
        "state_reset": history_state_reset_mode(group_key),
    }


def _record_has_group_key(record: dict[str, Any], group_key: str) -> bool:
    value = record.get(group_key)
    if value in (None, ""):
        return False
    try:
        return int(value) >= 0
    except (TypeError, ValueError):
        return False


def _records_support_group_key(records: list[dict[str, Any]], group_key: str) -> bool:
    return bool(records) and all(_record_has_group_key(record, group_key) for record in records)


def training_transaction_path(dataset_path: Path) -> Path:
    return dataset_path.resolve().with_name("transactions.jsonl")


def dataset_report_path(dataset_path: Path, report_name: str) -> Path:
    return dataset_path.resolve().parents[1] / "reports" / report_name


def load_optional_dataset_report(dataset_path: Path, report_name: str) -> dict[str, Any] | None:
    path = dataset_report_path(dataset_path, report_name)
    if not path.exists():
        return None
    try:
        payload = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"Invalid dataset report {path}: {exc}") from None
    if not isinstance(payload, dict):
        raise SystemExit(f"Dataset report {path} must be a JSON object")
    return payload


def normalize_generation_protocol_mode(protocol_mode: str | None) -> str:
    normalized = str(protocol_mode or DEFAULT_GENERATION_PROTOCOL_MODE).strip().lower()
    if normalized not in SUPPORTED_GENERATION_PROTOCOL_MODES:
        supported = ", ".join(SUPPORTED_GENERATION_PROTOCOL_MODES)
        raise SystemExit(
            f"Unsupported generation protocol mode {protocol_mode!r}. "
            f"Expected one of: {supported}."
        )
    return normalized


def assert_legacy_generation_protocol_mode(protocol_mode: str | None) -> None:
    normalized = normalize_generation_protocol_mode(protocol_mode)
    if normalized != "fprime":
        raise SystemExit(
            "run-legacy only supports --protocol-mode fprime because the preserved legacy "
            "baseline remains F´-only."
        )


def assert_legacy_training_dataset_supported(dataset_path: Path) -> None:
    generation_summary = load_optional_dataset_report(dataset_path, "generation_summary.json") or {}
    protocol_mode = str(generation_summary.get("protocol_mode", "")).strip().lower()
    if protocol_mode and protocol_mode != "fprime":
        raise SystemExit(
            "train-legacy only supports F´-only datasets. "
            f"dataset protocol_mode={protocol_mode!r}"
        )

    protocol_families = generation_summary.get("protocol_families")
    if isinstance(protocol_families, dict):
        observed_protocols = sorted(
            str(name)
            for name, count in protocol_families.items()
            if isinstance(count, (int, float)) and float(count) > 0.0
        )
        if observed_protocols and any(name != "fprime" for name in observed_protocols):
            raise SystemExit(
                "train-legacy only supports F´-only datasets. "
                f"observed protocol_families={observed_protocols!r}"
            )


def load_training_transactions(dataset_path: Path) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    transaction_path = training_transaction_path(dataset_path)
    if not transaction_path.exists():
        raise SystemExit(
            "Missing transactions.jsonl next to the dataset. "
            "Training and novelty fitting now require transaction replay inputs."
        )
    transactions = read_jsonl(transaction_path)
    if not _records_support_group_key(transactions, TRAINING_GROUP_KEY):
        raise SystemExit(
            f"transactions.jsonl must include {TRAINING_GROUP_KEY} on every row for grouped training replay"
        )
    history_featurization = history_featurization_report("transactions", TRAINING_GROUP_KEY)
    history_featurization["transaction_path"] = str(transaction_path)
    return transactions, history_featurization


def materialize_training_rows(records: list[dict[str, Any]], group_key: str) -> list[dict[str, Any]]:
    return transactions_to_rows(records, reset_key=group_key)


def materialize_split_rows(
    base_records: list[dict[str, Any]],
    calib_records: list[dict[str, Any]],
    test_records: list[dict[str, Any]],
    group_key: str,
) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
    return (
        materialize_training_rows(base_records, group_key),
        materialize_training_rows(calib_records, group_key),
        materialize_training_rows(test_records, group_key),
    )


def prepare_training_splits(dataset_path: Path, seed: int) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]], dict[str, Any], dict[str, Any]]:
    records, history_featurization = load_training_transactions(dataset_path)
    group_key = TRAINING_GROUP_KEY
    base_records, calib_records, test_records, split_summary = grouped_record_split(
        records,
        seed=seed,
        group_key=group_key,
    )
    base_rows, calib_rows, test_rows = materialize_split_rows(
        base_records,
        calib_records,
        test_records,
        group_key,
    )
    split_summary = dict(split_summary)
    split_summary["history_featurization"] = dict(history_featurization)
    return base_rows, calib_rows, test_rows, split_summary, history_featurization


def build_grouped_cv_plan(
    records: list[dict[str, Any]],
    seed: int,
    group_key: str = "episode_id",
    fold_count: int = DEFAULT_GROUPED_CV_FOLDS,
) -> dict[str, Any]:
    grouped = group_records(records, group_key)
    by_kind = grouped_ids_by_kind(grouped)

    missing_kinds = [name for name in CLASS_NAMES if not by_kind.get(name)]
    if missing_kinds:
        raise SystemExit(
            "Need grouped episodes spanning benign, cyber, and fault for evaluation. "
            f"episode_counts={json.dumps({name: len(by_kind.get(name, [])) for name in CLASS_NAMES}, separators=(',', ':'))} "
            f"missing={json.dumps(missing_kinds, separators=(',', ':'))}"
        )

    min_groups = min(len(by_kind.get(name, [])) for name in CLASS_NAMES)
    if min_groups < 3:
        raise SystemExit(
            "Need at least 3 episodes per class for grouped evaluation. "
            f"episode_counts={json.dumps({name: len(by_kind.get(name, [])) for name in CLASS_NAMES}, separators=(',', ':'))} "
            f"insufficient={json.dumps({name: len(by_kind.get(name, [])) for name in CLASS_NAMES if len(by_kind.get(name, [])) < 3}, separators=(',', ':'))}"
        )

    resolved_fold_count = min(max(3, int(fold_count)), min_groups)
    rng = random.Random(seed)
    fold_group_ids: list[list[int]] = [[] for _ in range(resolved_fold_count)]
    for kind in CLASS_NAMES:
        group_ids = list(by_kind[kind])
        rng.shuffle(group_ids)
        for index, group_id in enumerate(group_ids):
            fold_group_ids[index % resolved_fold_count].append(group_id)

    fold_summaries: list[dict[str, Any]] = []
    for fold_index in range(resolved_fold_count):
        test_group_ids = sorted(fold_group_ids[fold_index])
        calib_group_ids = sorted(fold_group_ids[(fold_index + 1) % resolved_fold_count])
        base_group_ids = sorted(
            group_id
            for other_index, group_ids in enumerate(fold_group_ids)
            if other_index not in {fold_index, (fold_index + 1) % resolved_fold_count}
            for group_id in group_ids
        )
        fold_summaries.append(
            {
                "seed": seed,
                "fold_index": fold_index,
                "group_key": group_key,
                "base_group_ids": base_group_ids,
                "calibration_group_ids": calib_group_ids,
                "test_group_ids": test_group_ids,
                "summary": {
                    "seed": seed,
                    "fold_index": fold_index,
                    "group_key": group_key,
                    "fold_count": resolved_fold_count,
                    "base_groups": len(base_group_ids),
                    "calibration_groups": len(calib_group_ids),
                    "test_groups": len(test_group_ids),
                    "base_rows": sum(len(grouped[group_id]) for group_id in base_group_ids),
                    "calibration_rows": sum(len(grouped[group_id]) for group_id in calib_group_ids),
                    "test_rows": sum(len(grouped[group_id]) for group_id in test_group_ids),
                    "base_class_group_counts": class_group_counts(base_group_ids, grouped),
                    "calibration_class_group_counts": class_group_counts(calib_group_ids, grouped),
                    "test_class_group_counts": class_group_counts(test_group_ids, grouped),
                },
            }
        )
    return {
        "seed": seed,
        "group_key": group_key,
        "fold_count": resolved_fold_count,
        "class_group_counts": {name: len(by_kind.get(name, [])) for name in CLASS_NAMES},
        "folds": fold_summaries,
    }


def build_family_holdout_plans(
    records: list[dict[str, Any]],
    seed: int,
    family_field: str,
    group_key: str = "episode_id",
) -> dict[str, Any]:
    grouped = group_records(records, group_key)
    by_kind = grouped_ids_by_kind(grouped)
    family_to_group_ids: dict[str, set[int]] = {}
    for group_id, group_rows in grouped.items():
        family_values = {
            str(row.get(family_field, "")).strip()
            for row in group_rows
            if str(row.get(family_field, "")).strip()
        }
        for family_value in family_values:
            family_to_group_ids.setdefault(family_value, set()).add(group_id)

    evaluations: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    for family_value in sorted(family_to_group_ids):
        target_group_ids = sorted(family_to_group_ids[family_value])
        target_class_counts = class_group_counts(target_group_ids, grouped)
        target_classes = {name for name, count in target_class_counts.items() if count > 0}
        family_rng = random.Random(
            seed * 1009 + stable_text_seed(family_field) * 37 + stable_text_seed(family_value)
        )
        support_group_ids: list[int] = []
        used_group_ids = set(target_group_ids)
        feasibility_reasons: list[str] = []
        for class_name_value in CLASS_NAMES:
            remaining = [group_id for group_id in by_kind.get(class_name_value, []) if group_id not in used_group_ids]
            required_remaining = 2 if class_name_value in target_classes else 3
            if len(remaining) < required_remaining:
                feasibility_reasons.append(
                    f"{class_name_value}:need_{required_remaining}_remaining_groups_after_holdout_have_{len(remaining)}"
                )
                continue
            if class_name_value not in target_classes:
                choices = list(remaining)
                family_rng.shuffle(choices)
                chosen_support = choices[0]
                support_group_ids.append(chosen_support)
                used_group_ids.add(chosen_support)
        if feasibility_reasons:
            skipped.append(
                {
                    "seed": seed,
                    "family_field": family_field,
                    "heldout_value": family_value,
                    "target_class_group_counts": target_class_counts,
                    "reason": ";".join(feasibility_reasons),
                }
            )
            continue

        calibration_group_ids: list[int] = []
        for class_name_value in CLASS_NAMES:
            available = [group_id for group_id in by_kind.get(class_name_value, []) if group_id not in used_group_ids]
            choices = list(available)
            family_rng.shuffle(choices)
            calibration_group_id = choices[0]
            calibration_group_ids.append(calibration_group_id)
            used_group_ids.add(calibration_group_id)

        base_group_ids = sorted(group_id for group_id in grouped if group_id not in used_group_ids)
        test_group_ids = sorted(target_group_ids + support_group_ids)
        summary = {
            "seed": seed,
            "group_key": group_key,
            "family_field": family_field,
            "heldout_value": family_value,
            "target_group_ids": target_group_ids,
            "support_group_ids": sorted(support_group_ids),
            "base_groups": len(base_group_ids),
            "calibration_groups": len(calibration_group_ids),
            "test_groups": len(test_group_ids),
            "base_rows": sum(len(grouped[group_id]) for group_id in base_group_ids),
            "calibration_rows": sum(len(grouped[group_id]) for group_id in calibration_group_ids),
            "test_rows": sum(len(grouped[group_id]) for group_id in test_group_ids),
            "target_class_group_counts": target_class_counts,
            "base_class_group_counts": class_group_counts(base_group_ids, grouped),
            "calibration_class_group_counts": class_group_counts(calibration_group_ids, grouped),
            "test_class_group_counts": class_group_counts(test_group_ids, grouped),
        }
        if any(int(summary["base_class_group_counts"].get(name, 0)) < 1 for name in CLASS_NAMES):
            skipped.append(
                {
                    "seed": seed,
                    "family_field": family_field,
                    "heldout_value": family_value,
                    "target_class_group_counts": target_class_counts,
                    "reason": "base_split_missing_class_support_after_holdout",
                }
            )
            continue
        evaluations.append(
            {
                "heldout_value": family_value,
                "base_group_ids": base_group_ids,
                "calibration_group_ids": sorted(calibration_group_ids),
                "test_group_ids": test_group_ids,
                "summary": summary,
            }
        )
    return {
        "seed": seed,
        "group_key": group_key,
        "family_field": family_field,
        "evaluations": evaluations,
        "skipped": skipped,
    }


def score_novelty_rows(model: GaussianNovelty, rows: list[dict[str, Any]]) -> np.ndarray:
    feature_names = list(model.feature_names or NOVELTY_FEATURE_NAMES)
    return np.array([model.score(vector_from_row(row, feature_names)) for row in rows], dtype=float)


def fit_calibrator(panomaly: np.ndarray, pcyber: np.ndarray, rules: np.ndarray, novelty: np.ndarray, labels: np.ndarray) -> Calibrator:
    X = np.column_stack([panomaly, pcyber, rules, novelty])
    anomaly_labels = (labels != 0).astype(int)
    model = LogisticRegression(max_iter=500, random_state=0)
    model.fit(X, anomaly_labels)
    return Calibrator(weights=model.coef_[0].astype(float), bias=float(model.intercept_[0]), feature_names=["panomaly", "pcyber", "rules", "novelty"])


def choose_threshold(labels: np.ndarray, scores: np.ndarray) -> float:
    best_threshold = 0.5
    best_f1 = -1.0
    for threshold in np.linspace(0.05, 0.95, 181):
        preds = (scores >= threshold).astype(int)
        _, _, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
        if f1 > best_f1:
            best_f1 = float(f1)
            best_threshold = float(threshold)
    return best_threshold


def safe_roc_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    try:
        return float(roc_auc_score(labels, scores))
    except ValueError:
        return 0.5


def safe_pr_auc(labels: np.ndarray, scores: np.ndarray) -> float:
    try:
        return float(average_precision_score(labels, scores))
    except ValueError:
        return 0.0


def evaluate_binary_scores(labels: np.ndarray, scores: np.ndarray, threshold: float) -> dict[str, Any]:
    preds = (scores >= threshold).astype(int)
    precision, recall, f1, _ = precision_recall_fscore_support(labels, preds, average="binary", zero_division=0)
    cm = confusion_matrix(labels, preds, labels=[0, 1]).tolist()
    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(labels, preds)),
        "precision": float(precision),
        "recall": float(recall),
        "f1": float(f1),
        "roc_auc": safe_roc_auc(labels, scores),
        "pr_auc": safe_pr_auc(labels, scores),
        "confusion_matrix": cm,
    }


def evaluate_class_scores(labels: np.ndarray, probs: np.ndarray) -> dict[str, Any]:
    preds = np.argmax(probs, axis=1)
    precision, recall, f1, support = precision_recall_fscore_support(labels, preds, labels=[0, 1, 2], zero_division=0)
    per_class = {
        CLASS_NAMES[idx]: {
            "precision": float(precision[idx]),
            "recall": float(recall[idx]),
            "f1": float(f1[idx]),
            "support": int(support[idx]),
        }
        for idx in range(len(CLASS_NAMES))
    }
    return {
        "accuracy": float(accuracy_score(labels, preds)),
        "macro_precision": float(np.mean(precision)),
        "macro_recall": float(np.mean(recall)),
        "macro_f1": float(np.mean(f1)),
        "confusion_matrix": confusion_matrix(labels, preds, labels=[0, 1, 2]).tolist(),
        "per_class": per_class,
    }


def mean_scores_by_class(labels: np.ndarray, score_map: dict[str, np.ndarray]) -> dict[str, dict[str, float]]:
    return {
        CLASS_NAMES[label]: {
            score_name: float(np.mean(score_values[labels == label])) if np.any(labels == label) else 0.0
            for score_name, score_values in score_map.items()
        }
        for label in range(len(CLASS_NAMES))
    }


def calibrator_summary(calibrator: Calibrator) -> dict[str, Any]:
    return {
        "feature_names": list(calibrator.feature_names),
        "weights": {
            feature_name: float(calibrator.weights[index])
            for index, feature_name in enumerate(calibrator.feature_names)
        },
        "bias": float(calibrator.bias),
    }


def transpose_metric_namespaces(by_model: dict[str, dict[str, Any]]) -> dict[str, dict[str, Any]]:
    namespaces: dict[str, dict[str, Any]] = {}
    for model_name, payload in by_model.items():
        for namespace_name, namespace_payload in payload.items():
            namespaces.setdefault(str(namespace_name), {})
            namespaces[str(namespace_name)][model_name] = namespace_payload
    return namespaces


def aggregate_structure(values: list[Any]) -> Any:
    filtered = [value for value in values if value is not None]
    if not filtered:
        return None
    first = filtered[0]
    if isinstance(first, bool):
        true_count = sum(1 for value in filtered if bool(value))
        return {
            "true_count": int(true_count),
            "false_count": int(len(filtered) - true_count),
            "true_fraction": 0.0 if not filtered else float(true_count / len(filtered)),
            "count": int(len(filtered)),
        }
    if isinstance(first, (int, float, np.integer, np.floating)):
        array = np.array([float(value) for value in filtered], dtype=float)
        return {
            "mean": float(np.mean(array)),
            "std": float(np.std(array, ddof=0)),
            "min": float(np.min(array)),
            "max": float(np.max(array)),
            "count": int(len(array)),
        }
    if isinstance(first, dict):
        keys = [key for key in first if all(isinstance(value, dict) and key in value for value in filtered)]
        return {key: aggregate_structure([value[key] for value in filtered]) for key in keys}
    if isinstance(first, list):
        if all(isinstance(value, list) and len(value) == len(first) for value in filtered):
            return [aggregate_structure([value[index] for value in filtered]) for index in range(len(first))]
        return filtered
    if all(value == first for value in filtered):
        return first
    return sorted(str(value) for value in {str(value) for value in filtered})


def aggregate_mean(payload: dict[str, Any], *path: Any) -> float:
    current: Any = payload
    for key in path:
        if isinstance(current, dict):
            current = current.get(key)
        elif isinstance(current, list) and isinstance(key, int) and 0 <= key < len(current):
            current = current[key]
        else:
            return 0.0
    if isinstance(current, dict) and "mean" in current:
        return float(current["mean"])
    if isinstance(current, (int, float)):
        return float(current)
    return 0.0


def aggregate_stat(payload: dict[str, Any], stat: str, *path: Any) -> float:
    current: Any = payload
    for key in path:
        if isinstance(current, dict):
            current = current.get(key)
        elif isinstance(current, list) and isinstance(key, int) and 0 <= key < len(current):
            current = current[key]
        else:
            return 0.0
    if isinstance(current, dict) and stat in current:
        return float(current[stat])
    return 0.0


def aggregate_min_per_class_recall(model_aggregate: dict[str, Any]) -> float:
    return min(
        aggregate_mean(
            model_aggregate,
            MODEL_ONLY_NAMESPACE,
            "multiclass_metrics",
            "per_class",
            class_name_value,
            "recall",
        )
        for class_name_value in CLASS_NAMES
    )


def aggregate_evaluation_runs(runs: list[dict[str, Any]]) -> dict[str, Any]:
    if not runs:
        return {}
    model_names = list(runs[0]["models"])
    aggregated_models = {
        model_name: aggregate_structure([run["models"][model_name] for run in runs])
        for model_name in model_names
    }
    aggregated_thresholds = {
        model_name: aggregate_structure([run["thresholds"][model_name] for run in runs])
        for model_name in model_names
    }
    return {
        "models": aggregated_models,
        "metric_namespaces": transpose_metric_namespaces(aggregated_models),
        "thresholds": aggregated_thresholds,
        "threshold_namespaces": transpose_metric_namespaces(aggregated_thresholds),
        "simple_command_baseline": aggregate_structure([run["simple_command_baseline"] for run in runs]),
        "request_only_baseline": aggregate_structure([run["request_only_baseline"] for run in runs]),
        "outcome_only_baseline": aggregate_structure([run["outcome_only_baseline"] for run in runs]),
        "protocol_only_baseline": aggregate_structure([run["protocol_only_baseline"] for run in runs]),
        "raw_protocol_shortcuts_baseline": aggregate_structure([run["raw_protocol_shortcuts_baseline"] for run in runs]),
    }


def fit_and_evaluate_models(
    base_rows: list[dict[str, Any]],
    calib_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    seed: int,
    include_artifacts: bool = False,
    include_curves: bool = False,
) -> dict[str, Any]:
    X_base, y_base = rows_to_matrix(base_rows, PRIMARY_MODEL_FEATURE_NAMES)
    X_calib, y_calib = rows_to_matrix(calib_rows, PRIMARY_MODEL_FEATURE_NAMES)
    X_test, y_test = rows_to_matrix(test_rows, PRIMARY_MODEL_FEATURE_NAMES)
    nominal_base = [row for row in base_rows if int(row["label"]) == 0]
    novelty = GaussianNovelty.fit(
        np.array([vector_from_row(row, NOVELTY_FEATURE_NAMES) for row in nominal_base], dtype=float),
        NOVELTY_FEATURE_NAMES,
    )
    nov_calib = score_novelty_rows(novelty, calib_rows)
    nov_test = score_novelty_rows(novelty, test_rows)
    rule_calib = np.array([rule_score(row) for row in calib_rows], dtype=float)
    rule_test = np.array([rule_score(row) for row in test_rows], dtype=float)

    rf = RandomForestClassifier(
        n_estimators=260,
        max_depth=14,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=seed,
        n_jobs=-1,
    )
    rf.fit(X_base, y_base)

    nn = Pipeline(
        [
            ("scale", StandardScaler()),
            (
                "mlp",
                MLPClassifier(
                    hidden_layer_sizes=(64, 32),
                    activation="relu",
                    alpha=5e-4,
                    batch_size=256,
                    learning_rate_init=1e-3,
                    early_stopping=False,
                    max_iter=120,
                    random_state=seed,
                ),
            ),
        ]
    )
    nn.fit(X_base, y_base)

    rf_calib_probs = align_probabilities(rf.predict_proba(X_calib), rf.classes_)
    rf_test_probs = align_probabilities(rf.predict_proba(X_test), rf.classes_)
    nn_calib_probs = align_probabilities(nn.predict_proba(X_calib), nn.named_steps["mlp"].classes_)
    nn_test_probs = align_probabilities(nn.predict_proba(X_test), nn.named_steps["mlp"].classes_)

    rf_calib_pcyber = rf_calib_probs[:, 1]
    rf_test_pcyber = rf_test_probs[:, 1]
    nn_calib_pcyber = nn_calib_probs[:, 1]
    nn_test_pcyber = nn_test_probs[:, 1]

    rf_calib_panomaly = np.clip(rf_calib_probs[:, 1] + rf_calib_probs[:, 2], 0.0, 1.0)
    rf_test_panomaly = np.clip(rf_test_probs[:, 1] + rf_test_probs[:, 2], 0.0, 1.0)
    nn_calib_panomaly = np.clip(nn_calib_probs[:, 1] + nn_calib_probs[:, 2], 0.0, 1.0)
    nn_test_panomaly = np.clip(nn_test_probs[:, 1] + nn_test_probs[:, 2], 0.0, 1.0)

    rf_calibrator = fit_calibrator(rf_calib_panomaly, rf_calib_pcyber, rule_calib, nov_calib, y_calib)
    nn_calibrator = fit_calibrator(nn_calib_panomaly, nn_calib_pcyber, rule_calib, nov_calib, y_calib)

    rf_calib_risk = np.array(
        [rf_calibrator.score(rf_calib_panomaly[index], rf_calib_pcyber[index], rule_calib[index], nov_calib[index]) for index in range(len(y_calib))],
        dtype=float,
    )
    rf_test_risk = np.array(
        [rf_calibrator.score(rf_test_panomaly[index], rf_test_pcyber[index], rule_test[index], nov_test[index]) for index in range(len(y_test))],
        dtype=float,
    )
    nn_calib_risk = np.array(
        [nn_calibrator.score(nn_calib_panomaly[index], nn_calib_pcyber[index], rule_calib[index], nov_calib[index]) for index in range(len(y_calib))],
        dtype=float,
    )
    nn_test_risk = np.array(
        [nn_calibrator.score(nn_test_panomaly[index], nn_test_pcyber[index], rule_test[index], nov_test[index]) for index in range(len(y_test))],
        dtype=float,
    )

    cyber_calib_labels = (y_calib == 1).astype(int)
    cyber_test_labels = (y_test == 1).astype(int)
    anomaly_calib_labels = (y_calib != 0).astype(int)
    anomaly_test_labels = (y_test != 0).astype(int)

    model_probabilities = {
        "random_forest": {
            "calib": rf_calib_probs,
            "test": rf_test_probs,
            "calib_pcyber": rf_calib_pcyber,
            "test_pcyber": rf_test_pcyber,
            "calib_panomaly": rf_calib_panomaly,
            "test_panomaly": rf_test_panomaly,
            "calib_risk": rf_calib_risk,
            "test_risk": rf_test_risk,
            "calibrator": rf_calibrator,
        },
        "neural_net": {
            "calib": nn_calib_probs,
            "test": nn_test_probs,
            "calib_pcyber": nn_calib_pcyber,
            "test_pcyber": nn_test_pcyber,
            "calib_panomaly": nn_calib_panomaly,
            "test_panomaly": nn_test_panomaly,
            "calib_risk": nn_calib_risk,
            "test_risk": nn_test_risk,
            "calibrator": nn_calibrator,
        },
    }

    thresholds: dict[str, dict[str, dict[str, float]]] = {}
    metrics: dict[str, dict[str, Any]] = {}
    curves = {
        MODEL_ONLY_NAMESPACE: {
            "cyber_roc": {},
            "cyber_pr": {},
            "anomaly_roc": {},
            "anomaly_pr": {},
        },
        STACKED_DETECTOR_NAMESPACE: {
            "anomaly_roc": {},
            "anomaly_pr": {},
        },
    }
    for name, values in model_probabilities.items():
        cyber_threshold = choose_threshold(cyber_calib_labels, values["calib_pcyber"])
        model_anomaly_threshold = choose_threshold(anomaly_calib_labels, values["calib_panomaly"])
        stacked_anomaly_threshold = choose_threshold(anomaly_calib_labels, values["calib_risk"])
        thresholds[name] = {
            MODEL_ONLY_NAMESPACE: {
                "cyber": cyber_threshold,
                "anomaly": model_anomaly_threshold,
            },
            STACKED_DETECTOR_NAMESPACE: {
                "anomaly": stacked_anomaly_threshold,
            },
        }
        class_metrics = evaluate_class_scores(y_test, values["test"])
        cyber_metrics = evaluate_binary_scores(cyber_test_labels, values["test_pcyber"], cyber_threshold)
        model_anomaly_metrics = evaluate_binary_scores(anomaly_test_labels, values["test_panomaly"], model_anomaly_threshold)
        stacked_anomaly_metrics = evaluate_binary_scores(anomaly_test_labels, values["test_risk"], stacked_anomaly_threshold)
        model_layout = MODEL_FEATURE_LAYOUTS.get(name, {"feature_tier": "unknown", "feature_names": []})
        model_only_score_means = mean_scores_by_class(
            y_test,
            {
                "mean_pcyber": values["test_pcyber"],
                "mean_panomaly": values["test_panomaly"],
            },
        )
        stacked_component_means = mean_scores_by_class(
            y_test,
            {
                "mean_model_panomaly": values["test_panomaly"],
                "mean_model_pcyber": values["test_pcyber"],
                "mean_rule_score": rule_test,
                "mean_novelty_score": nov_test,
                "mean_risk": values["test_risk"],
            },
        )
        metrics[name] = {
            MODEL_ONLY_NAMESPACE: {
                "feature_tier": str(model_layout["feature_tier"]),
                "feature_names": list(model_layout["feature_names"]),
                "multiclass_metrics": class_metrics,
                "cyber_binary_metrics": cyber_metrics,
                "anomaly_binary_metrics": model_anomaly_metrics,
                "score_means": model_only_score_means,
            },
            STACKED_DETECTOR_NAMESPACE: {
                "input_feature_tier": str(MODEL_FEATURE_LAYOUTS["calibrator"]["feature_tier"]),
                "input_feature_names": list(MODEL_FEATURE_LAYOUTS["calibrator"]["feature_names"]),
                "components": ["model_panomaly", "model_pcyber", "rules", "novelty", "calibrator"],
                "calibrator": calibrator_summary(values["calibrator"]),
                "anomaly_binary_metrics": stacked_anomaly_metrics,
                "component_score_means": stacked_component_means,
                "lift_vs_model_only": {
                    "f1_delta": float(stacked_anomaly_metrics["f1"] - model_anomaly_metrics["f1"]),
                    "pr_auc_delta": float(stacked_anomaly_metrics["pr_auc"] - model_anomaly_metrics["pr_auc"]),
                    "roc_auc_delta": float(stacked_anomaly_metrics["roc_auc"] - model_anomaly_metrics["roc_auc"]),
                },
            },
        }
        if include_curves:
            fpr, tpr, _ = roc_curve(cyber_test_labels, values["test_pcyber"])
            prec, rec, _ = precision_recall_curve(cyber_test_labels, values["test_pcyber"])
            curves[MODEL_ONLY_NAMESPACE]["cyber_roc"][name] = (fpr, tpr)
            curves[MODEL_ONLY_NAMESPACE]["cyber_pr"][name] = (rec, prec)
            anomaly_fpr, anomaly_tpr, _ = roc_curve(anomaly_test_labels, values["test_panomaly"])
            anomaly_prec, anomaly_rec, _ = precision_recall_curve(anomaly_test_labels, values["test_panomaly"])
            curves[MODEL_ONLY_NAMESPACE]["anomaly_roc"][name] = (anomaly_fpr, anomaly_tpr)
            curves[MODEL_ONLY_NAMESPACE]["anomaly_pr"][name] = (anomaly_rec, anomaly_prec)
            stacked_fpr, stacked_tpr, _ = roc_curve(anomaly_test_labels, values["test_risk"])
            stacked_prec, stacked_rec, _ = precision_recall_curve(anomaly_test_labels, values["test_risk"])
            curves[STACKED_DETECTOR_NAMESPACE]["anomaly_roc"][name] = (stacked_fpr, stacked_tpr)
            curves[STACKED_DETECTOR_NAMESPACE]["anomaly_pr"][name] = (stacked_rec, stacked_prec)

    result: dict[str, Any] = {
        "thresholds": thresholds,
        "metrics": metrics,
    }
    if include_curves:
        result["curves"] = curves
    if include_artifacts:
        feature_importance = [
            {"feature": feature_name, "importance": float(importance)}
            for feature_name, importance in sorted(
                zip(PRIMARY_MODEL_FEATURE_NAMES, rf.feature_importances_),
                key=lambda item: item[1],
                reverse=True,
            )[:12]
        ]
        calibrators = {"random_forest": rf_calibrator, "neural_net": nn_calibrator}
        model_payloads = {
            "random_forest": export_sklearn_forest(rf, PRIMARY_MODEL_FEATURE_NAMES, PRIMARY_MODEL_FEATURE_TIER),
            "neural_net": export_sklearn_mlp(nn, PRIMARY_MODEL_FEATURE_NAMES, PRIMARY_MODEL_FEATURE_TIER),
        }
        artifact_sizes = {
            name: len(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
            for name, payload in model_payloads.items()
        }
        result.update(
            {
                "feature_importance": feature_importance,
                "calibrators": calibrators,
                "model_payloads": model_payloads,
                "artifact_sizes": artifact_sizes,
                "novelty": novelty,
            }
        )
    return result


def fit_and_evaluate_poster_model(
    base_rows: list[dict[str, Any]],
    calib_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    seed: int,
    include_artifacts: bool = False,
    include_curves: bool = False,
) -> dict[str, Any]:
    feature_names = list(POSTER_DEFAULT_REQUEST_ONLY_BASELINE_FEATURE_NAMES)
    X_base, y_base = rows_to_matrix(base_rows, feature_names)
    X_calib, y_calib = rows_to_matrix(calib_rows, feature_names)
    X_test, y_test = rows_to_matrix(test_rows, feature_names)

    blue_model_fit = fit_poster_blue_model(
        X_base,
        y_base,
        X_calib,
        y_calib,
        feature_names=feature_names,
        seed=seed,
    )
    nn = blue_model_fit["pipeline"]

    calib_probs = align_probabilities(nn.predict_proba(X_calib), nn.named_steps["mlp"].classes_)
    test_probs = align_probabilities(nn.predict_proba(X_test), nn.named_steps["mlp"].classes_)
    calib_pcyber = calib_probs[:, 1]
    test_pcyber = test_probs[:, 1]
    calib_panomaly = np.clip(calib_probs[:, 1] + calib_probs[:, 2], 0.0, 1.0)
    test_panomaly = np.clip(test_probs[:, 1] + test_probs[:, 2], 0.0, 1.0)

    cyber_calib_labels = (y_calib == 1).astype(int)
    cyber_test_labels = (y_test == 1).astype(int)
    anomaly_calib_labels = (y_calib != 0).astype(int)
    anomaly_test_labels = (y_test != 0).astype(int)

    cyber_threshold = choose_threshold(cyber_calib_labels, calib_pcyber)
    anomaly_threshold = choose_threshold(anomaly_calib_labels, calib_panomaly)
    metrics = {
        "neural_net": {
            MODEL_ONLY_NAMESPACE: {
                "feature_tier": POSTER_DEFAULT_PRIMARY_MODEL_FEATURE_TIER,
                "feature_names": list(feature_names),
                "multiclass_metrics": evaluate_class_scores(y_test, test_probs),
                "cyber_binary_metrics": evaluate_binary_scores(cyber_test_labels, test_pcyber, cyber_threshold),
                "anomaly_binary_metrics": evaluate_binary_scores(anomaly_test_labels, test_panomaly, anomaly_threshold),
                "score_means": mean_scores_by_class(
                    y_test,
                    {
                        "mean_pcyber": test_pcyber,
                        "mean_panomaly": test_panomaly,
                    },
                ),
            }
        }
    }
    thresholds = {
        "neural_net": {
            MODEL_ONLY_NAMESPACE: {
                "cyber": cyber_threshold,
                "anomaly": anomaly_threshold,
            }
        }
    }
    result: dict[str, Any] = {
        "thresholds": thresholds,
        "metrics": metrics,
        "blue_model": {
            "architecture": dict(blue_model_fit["architecture"]),
            "output_formulation": dict(blue_model_fit["output_formulation"]),
            "training_config": dict(blue_model_fit["training_config"]),
            "training_summary": dict(blue_model_fit["training_summary"]),
            "training_history": list(blue_model_fit["training_history"]),
        },
    }
    if include_curves:
        cyber_fpr, cyber_tpr, _ = roc_curve(cyber_test_labels, test_pcyber)
        cyber_prec, cyber_rec, _ = precision_recall_curve(cyber_test_labels, test_pcyber)
        anomaly_fpr, anomaly_tpr, _ = roc_curve(anomaly_test_labels, test_panomaly)
        anomaly_prec, anomaly_rec, _ = precision_recall_curve(anomaly_test_labels, test_panomaly)
        result["curves"] = {
            MODEL_ONLY_NAMESPACE: {
                "cyber_roc": {"neural_net": (cyber_fpr, cyber_tpr)},
                "cyber_pr": {"neural_net": (cyber_rec, cyber_prec)},
                "anomaly_roc": {"neural_net": (anomaly_fpr, anomaly_tpr)},
                "anomaly_pr": {"neural_net": (anomaly_rec, anomaly_prec)},
            }
        }
    if include_artifacts:
        payload = export_poster_blue_model_payload(
            nn,
            feature_names,
            POSTER_DEFAULT_PRIMARY_MODEL_FEATURE_TIER,
            architecture=blue_model_fit["architecture"],
            output_formulation=blue_model_fit["output_formulation"],
            training_config=blue_model_fit["training_config"],
            training_summary=blue_model_fit["training_summary"],
        )
        result.update(
            {
                "feature_importance": [],
                "model_payloads": {"neural_net": payload},
                "artifact_sizes": {
                    "neural_net": len(json.dumps(payload, separators=(",", ":")).encode("utf-8"))
                },
            }
        )
    return result


def run_repeated_grouped_cv(
    records: list[dict[str, Any]],
    group_key: str,
    evaluation_seeds: list[int],
    *,
    fit_and_evaluate_fn: Callable[..., dict[str, Any]] | None = None,
    materialize_split_rows_fn: Callable[..., tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]] | None = None,
    command_only_feature_names: list[str] | None = None,
    request_only_feature_names: list[str] | None = None,
    outcome_only_feature_names: list[str] | None = None,
    protocol_only_feature_names: list[str] | None = None,
    raw_protocol_shortcut_feature_names: list[str] | None = None,
) -> dict[str, Any]:
    fit_fn = fit_and_evaluate_fn or fit_and_evaluate_models
    materialize_fn = materialize_split_rows_fn or materialize_split_rows
    grouped = group_records(records, group_key)
    runs: list[dict[str, Any]] = []
    fold_count = 0
    class_group_counts: dict[str, int] | None = None
    for evaluation_seed in evaluation_seeds:
        cv_plan = build_grouped_cv_plan(records, evaluation_seed, group_key=group_key)
        fold_count = max(fold_count, int(cv_plan["fold_count"]))
        if class_group_counts is None:
            class_group_counts = {
                name: int(cv_plan.get("class_group_counts", {}).get(name, 0))
                for name in CLASS_NAMES
            }
        for fold in cv_plan["folds"]:
            base_records = grouped_rows_for_ids(fold["base_group_ids"], grouped)
            calib_records = grouped_rows_for_ids(fold["calibration_group_ids"], grouped)
            test_records = grouped_rows_for_ids(fold["test_group_ids"], grouped)
            base_rows, calib_rows, test_rows = materialize_fn(base_records, calib_records, test_records, group_key)
            model_evaluation = fit_fn(
                base_rows,
                calib_rows,
                test_rows,
                seed=evaluation_seed,
            )
            runs.append(
                {
                    "seed": evaluation_seed,
                    "fold_index": int(fold["fold_index"]),
                    "split_summary": dict(fold["summary"]),
                    "thresholds": model_evaluation["thresholds"],
                    "models": model_evaluation["metrics"],
                    "simple_command_baseline": evaluate_command_only_baseline(
                        base_rows,
                        test_rows,
                        evaluation_seed,
                        feature_names=command_only_feature_names,
                    ),
                    "request_only_baseline": evaluate_request_only_baseline(
                        base_rows,
                        test_rows,
                        evaluation_seed,
                        feature_names=request_only_feature_names,
                    ),
                    "outcome_only_baseline": evaluate_outcome_only_baseline(
                        base_rows,
                        test_rows,
                        evaluation_seed,
                        feature_names=outcome_only_feature_names,
                    ),
                    "protocol_only_baseline": evaluate_protocol_only_baseline(
                        base_rows,
                        test_rows,
                        evaluation_seed,
                        feature_names=protocol_only_feature_names,
                    ),
                    "raw_protocol_shortcuts_baseline": evaluate_raw_protocol_shortcuts_baseline(
                        base_rows,
                        test_rows,
                        evaluation_seed,
                        feature_names=raw_protocol_shortcut_feature_names,
                    ),
                }
            )
    return {
        "group_key": group_key,
        "seeds": list(evaluation_seeds),
        "fold_count": int(fold_count),
        "total_runs": len(runs),
        "class_group_counts": class_group_counts or {name: 0 for name in CLASS_NAMES},
        "support_summary": summarize_evaluation_split_support(runs),
        "runs": runs,
        "aggregate": aggregate_evaluation_runs(runs),
    }


def run_family_holdout_evaluation(
    records: list[dict[str, Any]],
    group_key: str,
    evaluation_seeds: list[int],
    family_field: str,
    evaluation_name: str,
    *,
    fit_and_evaluate_fn: Callable[..., dict[str, Any]] | None = None,
    materialize_split_rows_fn: Callable[..., tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]] | None = None,
    command_only_feature_names: list[str] | None = None,
    request_only_feature_names: list[str] | None = None,
    outcome_only_feature_names: list[str] | None = None,
    protocol_only_feature_names: list[str] | None = None,
    raw_protocol_shortcut_feature_names: list[str] | None = None,
) -> dict[str, Any]:
    fit_fn = fit_and_evaluate_fn or fit_and_evaluate_models
    materialize_fn = materialize_split_rows_fn or materialize_split_rows
    grouped = group_records(records, group_key)
    class_group_counts = {name: len(grouped_ids_by_kind(grouped).get(name, [])) for name in CLASS_NAMES}
    runs: list[dict[str, Any]] = []
    skipped: list[dict[str, Any]] = []
    evaluated_values: set[str] = set()
    for evaluation_seed in evaluation_seeds:
        holdout_plan = build_family_holdout_plans(
            records,
            evaluation_seed,
            family_field=family_field,
            group_key=group_key,
        )
        skipped.extend(holdout_plan["skipped"])
        for evaluation in holdout_plan["evaluations"]:
            base_records = grouped_rows_for_ids(evaluation["base_group_ids"], grouped)
            calib_records = grouped_rows_for_ids(evaluation["calibration_group_ids"], grouped)
            test_records = grouped_rows_for_ids(evaluation["test_group_ids"], grouped)
            base_rows, calib_rows, test_rows = materialize_fn(base_records, calib_records, test_records, group_key)
            model_evaluation = fit_fn(
                base_rows,
                calib_rows,
                test_rows,
                seed=evaluation_seed,
            )
            evaluated_values.add(str(evaluation["heldout_value"]))
            runs.append(
                {
                    "seed": evaluation_seed,
                    "heldout_value": str(evaluation["heldout_value"]),
                    "family_field": family_field,
                    "split_summary": dict(evaluation["summary"]),
                    "thresholds": model_evaluation["thresholds"],
                    "models": model_evaluation["metrics"],
                    "simple_command_baseline": evaluate_command_only_baseline(
                        base_rows,
                        test_rows,
                        evaluation_seed,
                        feature_names=command_only_feature_names,
                    ),
                    "request_only_baseline": evaluate_request_only_baseline(
                        base_rows,
                        test_rows,
                        evaluation_seed,
                        feature_names=request_only_feature_names,
                    ),
                    "outcome_only_baseline": evaluate_outcome_only_baseline(
                        base_rows,
                        test_rows,
                        evaluation_seed,
                        feature_names=outcome_only_feature_names,
                    ),
                    "protocol_only_baseline": evaluate_protocol_only_baseline(
                        base_rows,
                        test_rows,
                        evaluation_seed,
                        feature_names=protocol_only_feature_names,
                    ),
                    "raw_protocol_shortcuts_baseline": evaluate_raw_protocol_shortcuts_baseline(
                        base_rows,
                        test_rows,
                        evaluation_seed,
                        feature_names=raw_protocol_shortcut_feature_names,
                    ),
                }
            )
    return {
        "name": evaluation_name,
        "family_field": family_field,
        "group_key": group_key,
        "seeds": list(evaluation_seeds),
        "feasible": bool(runs),
        "class_group_counts": class_group_counts,
        "support_summary": summarize_evaluation_split_support(runs),
        "evaluated_values": sorted(evaluated_values),
        "total_runs": len(runs),
        "runs": runs,
        "skipped": skipped,
        "aggregate": aggregate_evaluation_runs(runs),
    }


def baseline_not_applicable(
    feature_names: list[str] | None,
    *,
    reason: str,
    observed_tuple_count: int = 0,
) -> dict[str, Any]:
    return {
        "feature_names": list(feature_names or []),
        "near_perfect_threshold": 0.0,
        "near_perfect": False,
        "near_perfect_metrics": [],
        "best_metric_path": None,
        "best_metric_value": 0.0,
        "observed_tuple_count": int(observed_tuple_count),
        "not_applicable": True,
        "reason": reason,
    }


def summarize_evaluation_split_support(runs: list[dict[str, Any]]) -> dict[str, Any]:
    split_fields = (
        "base_class_group_counts",
        "calibration_class_group_counts",
        "test_class_group_counts",
    )
    per_run_support: list[dict[str, Any]] = []
    for run in runs:
        split_summary = run.get("split_summary", {})
        item: dict[str, Any] = {}
        if "seed" in run:
            item["seed"] = int(run["seed"])
        if "fold_index" in run:
            item["fold_index"] = int(run["fold_index"])
        if "heldout_value" in run:
            item["heldout_value"] = str(run["heldout_value"])
        if "family_field" in run:
            item["family_field"] = str(run["family_field"])
        for field in split_fields:
            item[field] = {
                name: int(split_summary.get(field, {}).get(name, 0))
                for name in CLASS_NAMES
            }
        per_run_support.append(item)

    summary: dict[str, Any] = {
        "run_count": len(per_run_support),
        "per_run_support": per_run_support,
    }
    for field in split_fields:
        field_counts = {
            name: [int(item.get(field, {}).get(name, 0)) for item in per_run_support]
            for name in CLASS_NAMES
        }
        summary[field] = {
            "min": {name: (min(values) if values else 0) for name, values in field_counts.items()},
            "max": {name: (max(values) if values else 0) for name, values in field_counts.items()},
            "values": {name: sorted(set(values)) for name, values in field_counts.items()},
        }
    return summary


def _feature_value_missing(value: Any) -> bool:
    return value is None or value == ""


def encode_baseline_feature_value(value: Any) -> float:
    if value is None or value == "":
        return 0.0
    if isinstance(value, bool):
        return 1.0 if value else 0.0
    if isinstance(value, (int, float, np.integer, np.floating)):
        return float(value)
    return float(stable_token_id(str(value), 8192))


def baseline_feature_tuple(row: dict[str, Any], feature_names: list[str]) -> tuple[Any, ...] | None:
    values = tuple(row.get(name) for name in feature_names)
    if all(_feature_value_missing(value) for value in values):
        return None
    return values


def baseline_rows_to_matrix(rows: list[dict[str, Any]], feature_names: list[str]) -> tuple[np.ndarray, np.ndarray]:
    X = np.array(
        [
            [encode_baseline_feature_value(row.get(name)) for name in feature_names]
            for row in rows
        ],
        dtype=float,
    )
    y = np.array([int(row["label"]) for row in rows], dtype=int)
    return X, y


def evaluate_feature_baseline(
    base_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    *,
    seed: int,
    feature_names: list[str],
    random_state_offset: int,
    near_perfect_threshold: float,
) -> dict[str, Any]:
    if not feature_names:
        return baseline_not_applicable(feature_names, reason="no_feature_names_configured")
    observed_feature_tuples = {
        tuple_value
        for row in [*base_rows, *test_rows]
        for tuple_value in [baseline_feature_tuple(row, feature_names)]
        if tuple_value is not None
    }
    if not observed_feature_tuples:
        return baseline_not_applicable(
            feature_names,
            reason="configured_features_missing_from_rows",
            observed_tuple_count=0,
        )
    if len(observed_feature_tuples) < 2:
        return baseline_not_applicable(
            feature_names,
            reason="configured_features_lack_variation",
            observed_tuple_count=len(observed_feature_tuples),
        )
    X_base, y_base = baseline_rows_to_matrix(base_rows, feature_names)
    X_test, y_test = baseline_rows_to_matrix(test_rows, feature_names)
    model = RandomForestClassifier(
        n_estimators=96,
        max_depth=8,
        min_samples_leaf=2,
        class_weight="balanced_subsample",
        random_state=seed + random_state_offset,
        n_jobs=-1,
    )
    model.fit(X_base, y_base)
    base_probs = align_probabilities(model.predict_proba(X_base), model.classes_)
    probs = align_probabilities(model.predict_proba(X_test), model.classes_)
    class_metrics = evaluate_class_scores(y_test, probs)
    cyber_base_labels = (y_base == 1).astype(int)
    cyber_test_labels = (y_test == 1).astype(int)
    anomaly_base_labels = (y_base != 0).astype(int)
    anomaly_test_labels = (y_test != 0).astype(int)
    cyber_threshold = choose_threshold(cyber_base_labels, base_probs[:, 1])
    anomaly_threshold = choose_threshold(
        anomaly_base_labels,
        np.clip(base_probs[:, 1] + base_probs[:, 2], 0.0, 1.0),
    )
    cyber_binary_metrics = evaluate_binary_scores(cyber_test_labels, probs[:, 1], cyber_threshold)
    anomaly_binary_metrics = evaluate_binary_scores(
        anomaly_test_labels,
        np.clip(probs[:, 1] + probs[:, 2], 0.0, 1.0),
        anomaly_threshold,
    )
    accuracy = float(class_metrics["accuracy"])
    macro_f1 = float(class_metrics["macro_f1"])
    metric_candidates = {
        "class_metrics.accuracy": accuracy,
        "class_metrics.macro_f1": macro_f1,
        "cyber_binary_metrics.f1": float(cyber_binary_metrics["f1"]),
        "anomaly_binary_metrics.f1": float(anomaly_binary_metrics["f1"]),
    }
    near_perfect_metrics = sorted(
        metric_name
        for metric_name, metric_value in metric_candidates.items()
        if float(metric_value) >= near_perfect_threshold
    )
    best_metric_name, best_metric_value = max(
        metric_candidates.items(),
        key=lambda item: float(item[1]),
    )
    return {
        "feature_names": list(feature_names),
        "observed_tuple_count": int(len(observed_feature_tuples)),
        "class_metrics": class_metrics,
        "cyber_binary_metrics": cyber_binary_metrics,
        "anomaly_binary_metrics": anomaly_binary_metrics,
        "near_perfect_threshold": float(near_perfect_threshold),
        "near_perfect_metrics": near_perfect_metrics,
        "best_metric_path": str(best_metric_name),
        "best_metric_value": float(best_metric_value),
        "near_perfect": bool(near_perfect_metrics),
    }


def evaluate_command_only_baseline(
    base_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    seed: int,
    *,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    return evaluate_feature_baseline(
        base_rows,
        test_rows,
        seed=seed,
        feature_names=resolve_feature_name_list(feature_names, COMMAND_ONLY_BASELINE_FEATURE_NAMES),
        random_state_offset=71,
        near_perfect_threshold=COMMAND_ONLY_BASELINE_NEAR_PERFECT_THRESHOLD,
    )


def evaluate_request_only_baseline(
    base_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    seed: int,
    *,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    return evaluate_feature_baseline(
        base_rows,
        test_rows,
        seed=seed,
        feature_names=resolve_feature_name_list(feature_names, REQUEST_ONLY_BASELINE_FEATURE_NAMES),
        random_state_offset=79,
        near_perfect_threshold=REQUEST_ONLY_BASELINE_NEAR_PERFECT_THRESHOLD,
    )


def evaluate_outcome_only_baseline(
    base_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    seed: int,
    *,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    return evaluate_feature_baseline(
        base_rows,
        test_rows,
        seed=seed,
        feature_names=resolve_feature_name_list(feature_names, OUTCOME_ONLY_BASELINE_FEATURE_NAMES),
        random_state_offset=89,
        near_perfect_threshold=OUTCOME_ONLY_BASELINE_NEAR_PERFECT_THRESHOLD,
    )


def evaluate_protocol_only_baseline(
    base_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    seed: int,
    *,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    return evaluate_feature_baseline(
        base_rows,
        test_rows,
        seed=seed,
        feature_names=resolve_feature_name_list(feature_names, PROTOCOL_ONLY_BASELINE_FEATURE_NAMES),
        random_state_offset=97,
        near_perfect_threshold=PROTOCOL_ONLY_BASELINE_NEAR_PERFECT_THRESHOLD,
    )


def evaluate_raw_protocol_shortcuts_baseline(
    base_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    seed: int,
    *,
    feature_names: list[str] | None = None,
) -> dict[str, Any]:
    return evaluate_feature_baseline(
        base_rows,
        test_rows,
        seed=seed,
        feature_names=resolve_feature_name_list(feature_names, RAW_PROTOCOL_SHORTCUT_BASELINE_FEATURE_NAMES),
        random_state_offset=103,
        near_perfect_threshold=RAW_PROTOCOL_SHORTCUT_BASELINE_NEAR_PERFECT_THRESHOLD,
    )


def audit_dataset_sanity(
    rows: list[dict[str, Any]],
    base_rows: list[dict[str, Any]],
    test_rows: list[dict[str, Any]],
    *,
    seed: int,
    group_key: str,
    command_only_feature_names: list[str] | None = None,
    request_only_feature_names: list[str] | None = None,
    protocol_only_feature_names: list[str] | None = None,
    raw_protocol_shortcut_feature_names: list[str] | None = None,
    request_tuple_purity_feature_names: list[str] | None = None,
    request_tuple_purity_bucket_sizes: dict[str, float] | None = None,
) -> dict[str, Any]:
    command_overlap = build_command_family_overlap_report(rows)
    service_overlap = build_field_overlap_report(
        rows,
        field_name="service",
        family_key="exact_service",
        item_key="service",
        items_key="services",
    )
    episode_signatures = build_episode_signature_report(rows, group_key=group_key)
    episode_diversity = evaluate_episode_signature_diversity(episode_signatures)
    request_tuple_purity = build_request_tuple_purity_report(
        rows,
        feature_names=resolve_feature_name_list(
            request_tuple_purity_feature_names,
            REQUEST_TUPLE_PURITY_FEATURE_NAMES,
        ),
        bucket_sizes=dict(
            REQUEST_TUPLE_PURITY_BUCKETS
            if request_tuple_purity_bucket_sizes is None
            else request_tuple_purity_bucket_sizes
        ),
    )
    command_only_baseline = evaluate_command_only_baseline(
        base_rows,
        test_rows,
        seed,
        feature_names=command_only_feature_names,
    )
    request_only_baseline = evaluate_request_only_baseline(
        base_rows,
        test_rows,
        seed,
        feature_names=request_only_feature_names,
    )
    protocol_only_baseline = evaluate_protocol_only_baseline(
        base_rows,
        test_rows,
        seed,
        feature_names=protocol_only_feature_names,
    )
    raw_protocol_shortcuts_baseline = evaluate_raw_protocol_shortcuts_baseline(
        base_rows,
        test_rows,
        seed,
        feature_names=raw_protocol_shortcut_feature_names,
    )

    checks = [
        {
            "name": "command_overlap",
            "metric_path": "command_overlap.summary.overlap_ratio",
            "value": float(command_overlap["summary"]["overlap_ratio"]),
            "threshold": float(DATASET_SANITY_MIN_CLASS_OVERLAP_RATIO),
            "operator": ">=",
            "passed": bool(float(command_overlap["summary"]["overlap_ratio"]) >= DATASET_SANITY_MIN_CLASS_OVERLAP_RATIO),
            "reason": "command_vocabulary_mostly_exclusive",
        },
        {
            "name": "service_overlap",
            "metric_path": "service_overlap.summary.overlap_ratio",
            "value": float(service_overlap["summary"]["overlap_ratio"]),
            "threshold": float(DATASET_SANITY_MIN_CLASS_OVERLAP_RATIO),
            "operator": ">=",
            "passed": bool(float(service_overlap["summary"]["overlap_ratio"]) >= DATASET_SANITY_MIN_CLASS_OVERLAP_RATIO),
            "reason": "service_vocabulary_mostly_exclusive",
        },
        {
            "name": "episode_signature_diversity",
            "metric_path": "episode_signatures.diversity_check",
            "passed": bool(episode_diversity["passed"]),
            "violations": list(episode_diversity["violations"]),
            "reason": "episode_signatures_insufficiently_diverse",
        },
        {
            "name": "request_tuple_purity_pure_rows",
            "metric_path": "request_tuple_purity.summary.pure_row_fraction",
            "value": float(request_tuple_purity["summary"]["pure_row_fraction"]),
            "threshold": float(DATASET_SANITY_MAX_REQUEST_TUPLE_PURE_ROW_FRACTION),
            "operator": "<=",
            "passed": bool(
                float(request_tuple_purity["summary"]["pure_row_fraction"])
                <= DATASET_SANITY_MAX_REQUEST_TUPLE_PURE_ROW_FRACTION
            ),
            "reason": "request_tuple_purity_too_high",
        },
        {
            "name": "request_tuple_purity_majority_rows",
            "metric_path": "request_tuple_purity.summary.majority_row_fraction",
            "value": float(request_tuple_purity["summary"]["majority_row_fraction"]),
            "threshold": float(DATASET_SANITY_MAX_REQUEST_TUPLE_MAJORITY_ROW_FRACTION),
            "operator": "<=",
            "passed": bool(
                float(request_tuple_purity["summary"]["majority_row_fraction"])
                <= DATASET_SANITY_MAX_REQUEST_TUPLE_MAJORITY_ROW_FRACTION
            ),
            "reason": "request_tuple_lookup_baseline_too_strong",
        },
        {
            "name": "simple_command_baseline",
            "metric_path": str(command_only_baseline.get("best_metric_path") or "baselines.command_only"),
            "value": float(command_only_baseline.get("best_metric_value") or 0.0),
            "threshold": float(command_only_baseline["near_perfect_threshold"]),
            "operator": "<",
            "passed": not bool(command_only_baseline["near_perfect"]),
            "reason": "simple_command_baseline_near_perfect",
        },
        {
            "name": "request_only_baseline",
            "metric_path": str(request_only_baseline.get("best_metric_path") or "baselines.request_only"),
            "value": float(request_only_baseline.get("best_metric_value") or 0.0),
            "threshold": float(request_only_baseline["near_perfect_threshold"]),
            "operator": "<",
            "passed": not bool(request_only_baseline["near_perfect"]),
            "reason": "request_only_baseline_near_perfect",
        },
        {
            "name": "protocol_only_baseline",
            "metric_path": str(protocol_only_baseline.get("best_metric_path") or "baselines.protocol_only"),
            "value": float(protocol_only_baseline.get("best_metric_value") or 0.0),
            "threshold": float(protocol_only_baseline["near_perfect_threshold"]),
            "operator": "<",
            "passed": True if bool(protocol_only_baseline.get("not_applicable")) else not bool(protocol_only_baseline["near_perfect"]),
            "reason": "protocol_only_baseline_near_perfect",
            **({"not_applicable": True, "not_applicable_reason": str(protocol_only_baseline.get("reason") or "")} if bool(protocol_only_baseline.get("not_applicable")) else {}),
        },
        {
            "name": "raw_protocol_shortcuts_baseline",
            "metric_path": str(raw_protocol_shortcuts_baseline.get("best_metric_path") or "baselines.raw_protocol_shortcuts"),
            "value": float(raw_protocol_shortcuts_baseline.get("best_metric_value") or 0.0),
            "threshold": float(raw_protocol_shortcuts_baseline["near_perfect_threshold"]),
            "operator": "<",
            "passed": True if bool(raw_protocol_shortcuts_baseline.get("not_applicable")) else not bool(raw_protocol_shortcuts_baseline["near_perfect"]),
            "reason": "raw_protocol_shortcuts_baseline_near_perfect",
            **({"not_applicable": True, "not_applicable_reason": str(raw_protocol_shortcuts_baseline.get("reason") or "")} if bool(raw_protocol_shortcuts_baseline.get("not_applicable")) else {}),
        },
    ]
    blocking_issues = [
        {
            "name": str(check["name"]),
            "reason": str(check["reason"]),
            "metric_path": str(check["metric_path"]),
            **({"value": float(check["value"])} if "value" in check else {}),
            **({"threshold": float(check["threshold"])} if "threshold" in check else {}),
            **({"violations": list(check["violations"])} if "violations" in check else {}),
        }
        for check in checks
        if not bool(check.get("passed"))
    ]
    return {
        "version": 1,
        "group_key": group_key,
        "rows": len(rows),
        "thresholds": {
            "class_overlap_ratio_min": float(DATASET_SANITY_MIN_CLASS_OVERLAP_RATIO),
            "request_tuple_pure_row_fraction_max": float(DATASET_SANITY_MAX_REQUEST_TUPLE_PURE_ROW_FRACTION),
            "request_tuple_majority_row_fraction_max": float(DATASET_SANITY_MAX_REQUEST_TUPLE_MAJORITY_ROW_FRACTION),
            "command_only_near_perfect_threshold": float(COMMAND_ONLY_BASELINE_NEAR_PERFECT_THRESHOLD),
            "request_only_near_perfect_threshold": float(REQUEST_ONLY_BASELINE_NEAR_PERFECT_THRESHOLD),
            "protocol_only_near_perfect_threshold": float(PROTOCOL_ONLY_BASELINE_NEAR_PERFECT_THRESHOLD),
            "raw_protocol_shortcuts_near_perfect_threshold": float(RAW_PROTOCOL_SHORTCUT_BASELINE_NEAR_PERFECT_THRESHOLD),
        },
        "baseline_split": {
            "seed": int(seed),
            "base_rows": len(base_rows),
            "test_rows": len(test_rows),
        },
        "command_overlap": command_overlap,
        "service_overlap": service_overlap,
        "episode_signatures": {
            **episode_signatures,
            "diversity_check": episode_diversity,
        },
        "request_tuple_purity": request_tuple_purity,
        "baselines": {
            "command_only": command_only_baseline,
            "request_only": request_only_baseline,
            "protocol_only": protocol_only_baseline,
            "raw_protocol_shortcuts": raw_protocol_shortcuts_baseline,
        },
        "checks": checks,
        "blocking_issues": blocking_issues,
        "passed": not blocking_issues,
        "eligible_for_deployment": not blocking_issues,
    }


def min_per_class_recall(class_metrics: dict[str, Any]) -> float:
    per_class = class_metrics.get("per_class", {})
    recalls = [float(per_class.get(name, {}).get("recall", 0.0)) for name in CLASS_NAMES]
    return min(recalls) if recalls else 0.0


def has_full_class_support(class_metrics: dict[str, Any]) -> bool:
    per_class = class_metrics.get("per_class", {})
    return all(int(per_class.get(name, {}).get("support", 0)) > 0 for name in CLASS_NAMES)


def split_has_full_group_support(split_summary: dict[str, Any]) -> bool:
    for field_name in ("base_class_group_counts", "calibration_class_group_counts", "test_class_group_counts"):
        counts = split_summary.get(field_name, {})
        if any(int(counts.get(name, 0)) < 1 for name in CLASS_NAMES):
            return False
    return True


def build_generalization_metric_checks(
    evaluation_name: str,
    model_aggregate: dict[str, Any],
    *,
    anomaly_selection_metric: str = SELECTION_METRIC_STACKED_ANOMALY_F1,
) -> list[dict[str, Any]]:
    profile = GENERALIZATION_GATE_THRESHOLDS[evaluation_name]
    anomaly_metric_path = (
        (STACKED_DETECTOR_NAMESPACE, "anomaly_binary_metrics", "f1")
        if anomaly_selection_metric == SELECTION_METRIC_STACKED_ANOMALY_F1
        else (MODEL_ONLY_NAMESPACE, "anomaly_binary_metrics", "f1")
    )
    metric_definitions = {
        "class_macro_f1": {
            "label": SELECTION_METRIC_CLASS_MACRO_F1,
            "metric_path": f"{MODEL_ONLY_NAMESPACE}.multiclass_metrics.macro_f1",
            "value": aggregate_mean(model_aggregate, MODEL_ONLY_NAMESPACE, "multiclass_metrics", "macro_f1"),
        },
        "min_per_class_recall": {
            "label": SELECTION_METRIC_MIN_PER_CLASS_RECALL,
            "metric_path": f"{MODEL_ONLY_NAMESPACE}.multiclass_metrics.per_class.*.recall",
            "value": aggregate_min_per_class_recall(model_aggregate),
        },
        "anomaly_f1": {
            "label": anomaly_selection_metric,
            "metric_path": ".".join(anomaly_metric_path),
            "value": aggregate_mean(model_aggregate, *anomaly_metric_path),
        },
        "cyber_f1": {
            "label": SELECTION_METRIC_MODEL_CYBER_F1,
            "metric_path": f"{MODEL_ONLY_NAMESPACE}.cyber_binary_metrics.f1",
            "value": aggregate_mean(model_aggregate, MODEL_ONLY_NAMESPACE, "cyber_binary_metrics", "f1"),
        },
    }
    return [
        {
            "evaluation": evaluation_name,
            "metric": metric_name,
            "label": str(metric_definitions[metric_name]["label"]),
            "metric_path": str(metric_definitions[metric_name]["metric_path"]),
            "value": float(metric_definitions[metric_name]["value"]),
            "threshold": float(threshold),
            "passed": bool(float(metric_definitions[metric_name]["value"]) >= threshold),
        }
        for metric_name, threshold in profile.items()
    ]


def build_generalization_gate(
    repeated_grouped_cv: dict[str, Any],
    scenario_family_holdout: dict[str, Any],
    command_family_holdout: dict[str, Any],
    protocol_family_holdout: dict[str, Any],
    *,
    anomaly_selection_metric: str = SELECTION_METRIC_STACKED_ANOMALY_F1,
) -> dict[str, Any]:
    model_names = list(repeated_grouped_cv.get("aggregate", {}).get("models", {}))
    model_gate: dict[str, Any] = {}
    eligible_models: list[str] = []
    for model_name in model_names:
        checks: list[dict[str, Any]] = []
        grouped_cv_aggregate = repeated_grouped_cv.get("aggregate", {}).get("models", {}).get(model_name, {})
        checks.extend(
            build_generalization_metric_checks(
                "grouped_cv",
                grouped_cv_aggregate,
                anomaly_selection_metric=anomaly_selection_metric,
            )
        )
        for evaluation_name, evaluation_report in (
            ("scenario_family_holdout", scenario_family_holdout),
            ("command_family_holdout", command_family_holdout),
            ("protocol_family_holdout", protocol_family_holdout),
        ):
            if evaluation_report.get("feasible"):
                model_aggregate = evaluation_report.get("aggregate", {}).get("models", {}).get(model_name, {})
                checks.extend(
                    build_generalization_metric_checks(
                        evaluation_name,
                        model_aggregate,
                        anomaly_selection_metric=anomaly_selection_metric,
                    )
                )
            else:
                checks.append(
                    {
                        "evaluation": evaluation_name,
                        "metric": "skipped",
                        "skipped": True,
                        "passed": True,
                        "reason": "no_feasible_holdouts",
                    }
                )
        eligible = all(bool(check.get("passed")) for check in checks)
        model_gate[model_name] = {
            "eligible_for_deployment": eligible,
            "checks": checks,
        }
        if eligible:
            eligible_models.append(model_name)
    return {
        "thresholds": GENERALIZATION_GATE_THRESHOLDS,
        "models": model_gate,
        "eligible_models": eligible_models,
        "passed": bool(eligible_models),
    }


def ranking_payload(
    name: str,
    model_metrics: dict[str, Any],
    artifact_bytes: int,
    *,
    anomaly_selection_metric: str = SELECTION_METRIC_STACKED_ANOMALY_F1,
) -> dict[str, Any]:
    class_metrics = model_metrics[MODEL_ONLY_NAMESPACE]["multiclass_metrics"]
    cyber_metrics = model_metrics[MODEL_ONLY_NAMESPACE]["cyber_binary_metrics"]
    if anomaly_selection_metric == SELECTION_METRIC_STACKED_ANOMALY_F1:
        anomaly_metrics = model_metrics[STACKED_DETECTOR_NAMESPACE]["anomaly_binary_metrics"]
    else:
        anomaly_metrics = model_metrics[MODEL_ONLY_NAMESPACE]["anomaly_binary_metrics"]
    return {
        "name": name,
        "selection_metrics": {
            SELECTION_METRIC_CLASS_MACRO_F1: float(class_metrics["macro_f1"]),
            SELECTION_METRIC_MIN_PER_CLASS_RECALL: float(min_per_class_recall(class_metrics)),
            anomaly_selection_metric: float(anomaly_metrics["f1"]),
            SELECTION_METRIC_MODEL_CYBER_F1: float(cyber_metrics["f1"]),
        },
        "artifact_bytes": int(artifact_bytes),
    }


def aggregate_ranking_payload(
    name: str,
    model_aggregate: dict[str, Any],
    artifact_bytes: int,
    *,
    anomaly_selection_metric: str = SELECTION_METRIC_STACKED_ANOMALY_F1,
) -> dict[str, Any]:
    anomaly_metric_path = (
        (STACKED_DETECTOR_NAMESPACE, "anomaly_binary_metrics", "f1")
        if anomaly_selection_metric == SELECTION_METRIC_STACKED_ANOMALY_F1
        else (MODEL_ONLY_NAMESPACE, "anomaly_binary_metrics", "f1")
    )
    return {
        "name": name,
        "selection_metrics": {
            SELECTION_METRIC_CLASS_MACRO_F1: aggregate_mean(model_aggregate, MODEL_ONLY_NAMESPACE, "multiclass_metrics", "macro_f1"),
            SELECTION_METRIC_MIN_PER_CLASS_RECALL: aggregate_min_per_class_recall(model_aggregate),
            anomaly_selection_metric: aggregate_mean(model_aggregate, *anomaly_metric_path),
            SELECTION_METRIC_MODEL_CYBER_F1: aggregate_mean(model_aggregate, MODEL_ONLY_NAMESPACE, "cyber_binary_metrics", "f1"),
        },
        "artifact_bytes": int(artifact_bytes),
    }


def ranking_key(
    payload: dict[str, Any],
    selection_metric_order: list[str] | None = None,
) -> tuple[float, ...]:
    order = list(selection_metric_order or SELECTION_METRIC_ORDER)
    selection_metrics = payload["selection_metrics"]
    return (
        *[-float(selection_metrics.get(metric_name, 0.0)) for metric_name in order],
        int(payload["artifact_bytes"]),
        str(payload["name"]),
    )


def align_probabilities(raw_probs: np.ndarray, class_labels: np.ndarray) -> np.ndarray:
    aligned = np.zeros((raw_probs.shape[0], len(CLASS_NAMES)), dtype=float)
    for idx, label in enumerate(class_labels):
        aligned[:, int(label)] = raw_probs[:, idx]
    return aligned


def plot_comparison(
    metrics_by_model: dict[str, dict[str, Any]],
    report_dir: Path,
    metric_names: list[str],
    file_name: str = "comparison.png",
    title: str | None = None,
    error_by_model: dict[str, dict[str, Any]] | None = None,
) -> None:
    report_dir.mkdir(parents=True, exist_ok=True)
    model_names = list(metrics_by_model)
    x = np.arange(len(model_names))
    width = 0.16
    fig, ax = plt.subplots(figsize=(11, 5.5))
    for idx, metric_name in enumerate(metric_names):
        values = [metrics_by_model[name][metric_name] for name in model_names]
        errors = None
        if error_by_model is not None:
            errors = [error_by_model.get(name, {}).get(metric_name, 0.0) for name in model_names]
        ax.bar(
            x + (idx - (len(metric_names) - 1) / 2.0) * width,
            values,
            width=width,
            label=metric_name,
            yerr=errors,
            capsize=3 if errors is not None else 0,
        )
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, rotation=15, ha="right")
    ax.set_ylim(0.0, 1.02)
    ax.set_ylabel("score")
    if title:
        ax.set_title(title)
    ax.legend(loc="lower right")
    fig.tight_layout()
    fig.savefig(report_dir / file_name)
    plt.close(fig)


def plot_curve(
    curves: dict[str, tuple[np.ndarray, np.ndarray]],
    report_dir: Path,
    file_name: str,
    xlabel: str,
    ylabel: str,
    title: str | None = None,
) -> None:
    fig, ax = plt.subplots(figsize=(7, 5))
    for name, (x_values, y_values) in curves.items():
        ax.plot(x_values, y_values, label=name)
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    if title:
        ax.set_title(title)
    ax.legend(loc="best")
    fig.tight_layout()
    fig.savefig(report_dir / file_name)
    plt.close(fig)


def export_runtime_files(
    output_dir: Path,
    model_payloads: dict[str, dict[str, Any]],
    novelty: GaussianNovelty,
    calibrators: dict[str, Calibrator],
    winner_name: str,
) -> None:
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    save_json(model_dir / "forest.json", model_payloads["random_forest"])
    save_json(model_dir / "nn.json", model_payloads["neural_net"])
    save_json(model_dir / LEGACY_PRIMARY_MODEL_ARTIFACT_NAME, model_payloads[winner_name])
    novelty.dump_cfg(model_dir / "novelty.cfg")
    calibrators["random_forest"].dump_json(model_dir / "calibrator_rf.json")
    calibrators["neural_net"].dump_json(model_dir / "calibrator_nn.json")
    calibrators[winner_name].dump_json(model_dir / "calibrator.json")
    save_json(
        model_dir / RUNTIME_BUNDLE_MANIFEST_NAME,
        build_legacy_runtime_manifest(
            training_path=TRAINING_PATH_LEGACY_FPRIME_BASELINE,
            feature_tier=str(model_payloads[winner_name].get("feature_tier", PRIMARY_MODEL_FEATURE_TIER)),
            primary_model_artifact=LEGACY_PRIMARY_MODEL_ARTIFACT_NAME,
        ),
    )
    stale_pickle = model_dir / "nn.pkl"
    if stale_pickle.exists():
        stale_pickle.unlink()
    stale_blue_model = model_dir / POSTER_BLUE_MODEL_ARTIFACT_NAME
    if stale_blue_model.exists():
        stale_blue_model.unlink()
    deployment_dir = DEFAULT_MODEL_DIR
    deployment_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_dir / LEGACY_PRIMARY_MODEL_ARTIFACT_NAME, deployment_dir / LEGACY_PRIMARY_MODEL_ARTIFACT_NAME)
    shutil.copy2(model_dir / "novelty.cfg", deployment_dir / "novelty.cfg")
    shutil.copy2(model_dir / "calibrator.json", deployment_dir / "calibrator.json")
    shutil.copy2(model_dir / RUNTIME_BUNDLE_MANIFEST_NAME, deployment_dir / RUNTIME_BUNDLE_MANIFEST_NAME)
    stale_blue_deployment = deployment_dir / POSTER_BLUE_MODEL_ARTIFACT_NAME
    if stale_blue_deployment.exists():
        stale_blue_deployment.unlink()


def export_runtime_files_poster(
    output_dir: Path,
    model_payloads: dict[str, dict[str, Any]],
    winner_name: str,
) -> None:
    model_dir = output_dir / "models"
    model_dir.mkdir(parents=True, exist_ok=True)
    winner_payload = model_payloads[winner_name]
    save_json(model_dir / "nn.json", model_payloads["neural_net"])
    save_json(model_dir / POSTER_BLUE_MODEL_ARTIFACT_NAME, winner_payload)
    save_json(
        model_dir / RUNTIME_BUNDLE_MANIFEST_NAME,
        build_poster_blue_runtime_manifest(
            training_path=POSTER_DEFAULT_TRAINING_PATH_NAME,
            feature_tier=str(winner_payload.get("feature_tier", POSTER_DEFAULT_PRIMARY_MODEL_FEATURE_TIER)),
            primary_model_artifact=POSTER_BLUE_MODEL_ARTIFACT_NAME,
        ),
    )
    for stale_name in (
        "forest.json",
        LEGACY_PRIMARY_MODEL_ARTIFACT_NAME,
        "novelty.cfg",
        "calibrator_rf.json",
        "calibrator_nn.json",
        "calibrator.json",
        "nn.pkl",
    ):
        stale_path = model_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    deployment_dir = DEFAULT_MODEL_DIR
    deployment_dir.mkdir(parents=True, exist_ok=True)
    shutil.copy2(model_dir / POSTER_BLUE_MODEL_ARTIFACT_NAME, deployment_dir / POSTER_BLUE_MODEL_ARTIFACT_NAME)
    shutil.copy2(model_dir / RUNTIME_BUNDLE_MANIFEST_NAME, deployment_dir / RUNTIME_BUNDLE_MANIFEST_NAME)
    for stale_name in (LEGACY_PRIMARY_MODEL_ARTIFACT_NAME, "novelty.cfg", "calibrator.json"):
        stale_path = deployment_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()


def export_runtime_files_poster_research_candidate(
    output_dir: Path,
    model_payloads: dict[str, dict[str, Any]],
    *,
    model_name: str = "neural_net",
    deployment_blocked_reason: str | None = None,
) -> dict[str, Any]:
    model_dir = output_dir / "research_models"
    model_dir.mkdir(parents=True, exist_ok=True)
    if model_name not in model_payloads:
        raise KeyError(f"Poster research runtime export is missing model payload {model_name!r}")
    winner_payload = model_payloads[model_name]
    save_json(model_dir / POSTER_BLUE_MODEL_ARTIFACT_NAME, winner_payload)
    manifest = build_poster_blue_runtime_manifest(
        training_path=POSTER_DEFAULT_TRAINING_PATH_NAME,
        feature_tier=str(winner_payload.get("feature_tier", POSTER_DEFAULT_PRIMARY_MODEL_FEATURE_TIER)),
        primary_model_artifact=POSTER_BLUE_MODEL_ARTIFACT_NAME,
    )
    manifest["bundle_purpose"] = "research_only_non_deployable_candidate"
    manifest["deployment_ready"] = False
    if deployment_blocked_reason is not None:
        manifest["deployment_blocked_reason"] = deployment_blocked_reason
    save_json(model_dir / RUNTIME_BUNDLE_MANIFEST_NAME, manifest)
    for stale_name in (
        "forest.json",
        LEGACY_PRIMARY_MODEL_ARTIFACT_NAME,
        "novelty.cfg",
        "calibrator_rf.json",
        "calibrator_nn.json",
        "calibrator.json",
        "nn.pkl",
        "nn.json",
    ):
        stale_path = model_dir / stale_name
        if stale_path.exists():
            stale_path.unlink()
    return {
        "runtime_kind": POSTER_BLUE_RUNTIME_KIND,
        "bundle_purpose": "research_only_non_deployable_candidate",
        "model_name": model_name,
        "deployment_ready": False,
        "deployment_blocked_reason": deployment_blocked_reason,
        "model_path": str((model_dir / POSTER_BLUE_MODEL_ARTIFACT_NAME).resolve()),
        "manifest_path": str((model_dir / RUNTIME_BUNDLE_MANIFEST_NAME).resolve()),
        "artifact_dir": str(model_dir.resolve()),
    }


def canonical_rows_from_legacy_transactions(
    transactions: list[dict[str, Any]],
    legacy_rows: list[dict[str, Any]],
    *,
    source_artifact_paths: list[str] | None = None,
) -> list[dict[str, Any]]:
    source_paths = list(source_artifact_paths or [])
    canonical_rows: list[dict[str, Any]] = []
    for transaction, legacy_row in zip(transactions, legacy_rows):
        canonical_rows.append(
            canonicalize_legacy_fprime_transaction(
                transaction,
                source_artifact_paths=source_paths,
                recent_behavior={
                    "command_rate_1m": legacy_row.get("command_rate_1m"),
                    "error_rate_1m": legacy_row.get("error_rate_1m"),
                    "repeat_command_count_10m": legacy_row.get("repeat_command_count_10m"),
                    "same_target_command_rate_1m": legacy_row.get("same_target_command_rate_1m"),
                },
            )
        )
    return canonical_rows


def run_generate(
    output_dir: Path,
    rows: int,
    nominal_ratio: float,
    seed: int,
    *,
    protocol_mode: str = DEFAULT_GENERATION_PROTOCOL_MODE,
    mixed_fprime_ratio: float = DEFAULT_MIXED_FPRIME_RATIO,
) -> Path:
    script_path = Path(__file__).resolve().parent / "tools" / "shared" / "generate_dataset.py"
    python_bin = sys.executable or "python3"
    normalized_protocol_mode = normalize_generation_protocol_mode(protocol_mode)
    try:
        subprocess.run(
            [
                python_bin,
                str(script_path),
                "--protocol-mode",
                normalized_protocol_mode,
                "--rows",
                str(rows),
                "--nominal-ratio",
                str(nominal_ratio),
                "--seed",
                str(seed),
                "--output-dir",
                str(output_dir),
                "--mixed-fprime-ratio",
                str(mixed_fprime_ratio),
            ],
            check=True,
        )
    except subprocess.CalledProcessError as exc:
        raise SystemExit(f"Dataset generation failed: {exc}") from None
    return output_dir / "data" / "dataset.jsonl"


def run_train_novelty(dataset_path: Path, output_path: Path) -> None:
    require_training_deps()
    transactions, history_featurization = load_training_transactions(dataset_path)
    rows = materialize_training_rows(transactions, history_featurization["group_key"])
    rows = [row for row in rows if int(row.get("label", 1)) == 0]
    if len(rows) < 10:
        raise SystemExit("Need at least 10 nominal rows for novelty training")
    X = np.array([vector_from_row(row, NOVELTY_FEATURE_NAMES) for row in rows], dtype=float)
    novelty = GaussianNovelty.fit(X, NOVELTY_FEATURE_NAMES)
    novelty.dump_cfg(output_path)


def _run_training_legacy_baseline(
    dataset_path: Path,
    output_dir: Path,
    seed: int,
    make_plots: bool,
    blue_feature_policy_name: str,
) -> dict[str, Any]:
    training_path_name = TRAINING_PATH_LEGACY_FPRIME_BASELINE
    selection_metric_order = active_selection_metric_order(training_path_name)
    anomaly_selection_metric = active_anomaly_selection_metric(training_path_name)
    require_training_deps()
    try:
        blue_feature_policy = validate_blue_feature_names(
            PRIMARY_MODEL_FEATURE_NAMES,
            blue_feature_policy_name,
        )
    except BlueFeaturePolicyError as exc:
        raise SystemExit(
            f"Blue feature policy violation for {blue_feature_policy_name}: {exc}"
        ) from None
    novelty_adaptation = novelty_adaptation_report()
    source_records, history_featurization = load_training_transactions(dataset_path)
    training_group_key = TRAINING_GROUP_KEY
    if len(source_records) < 100:
        raise SystemExit("Need at least 100 generated rows before training")
    base_records, calib_records, test_records, split_summary = grouped_record_split(
        source_records,
        seed=seed,
        group_key=training_group_key,
    )
    base_rows, calib_rows, test_rows = materialize_split_rows(
        base_records,
        calib_records,
        test_records,
        training_group_key,
    )
    split_summary = dict(split_summary)
    split_summary["history_featurization"] = dict(history_featurization)
    rows = [*base_rows, *calib_rows, *test_rows]
    split_assignments: dict[int, str] = {}
    for split_name, split_rows in (("base", base_rows), ("calibration", calib_rows), ("test", test_rows)):
        for row in split_rows:
            split_assignments[int(row[training_group_key])] = split_name
    split_episode_signature_report = build_episode_signature_report(
        rows,
        group_key=training_group_key,
        split_assignments=split_assignments,
    )
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    split_episode_signature_path = report_dir / "split_episode_signatures.json"
    save_json(split_episode_signature_path, split_episode_signature_report)
    split_episode_signatures_enforced = has_structural_signature_signal(split_episode_signature_report)
    if split_episode_signatures_enforced:
        assert_split_episode_separation(split_episode_signature_report)
    dataset_sanity_report = audit_dataset_sanity(
        rows,
        base_rows,
        test_rows,
        seed=seed,
        group_key=training_group_key,
    )
    dataset_sanity_path = report_dir / "dataset_sanity.json"
    save_json(dataset_sanity_path, dataset_sanity_report)
    command_only_baseline = dict(dataset_sanity_report["baselines"]["command_only"])
    request_only_baseline = dict(dataset_sanity_report["baselines"]["request_only"])
    protocol_only_baseline = dict(dataset_sanity_report["baselines"]["protocol_only"])
    raw_protocol_shortcuts_baseline = dict(dataset_sanity_report["baselines"]["raw_protocol_shortcuts"])

    evaluation_seeds = evaluation_seed_sequence(seed)
    repeated_grouped_cv = run_repeated_grouped_cv(
        source_records,
        training_group_key,
        evaluation_seeds,
    )
    scenario_family_holdout = run_family_holdout_evaluation(
        source_records,
        training_group_key,
        evaluation_seeds,
        family_field="attack_family",
        evaluation_name="scenario_family_holdout",
    )
    command_family_holdout = run_family_holdout_evaluation(
        source_records,
        training_group_key,
        evaluation_seeds,
        family_field="command",
        evaluation_name="command_family_holdout",
    )
    protocol_family_holdout = run_family_holdout_evaluation(
        source_records,
        training_group_key,
        evaluation_seeds,
        family_field="protocol_family",
        evaluation_name="protocol_family_holdout",
    )
    generalization_gate = build_generalization_gate(
        repeated_grouped_cv,
        scenario_family_holdout,
        command_family_holdout,
        protocol_family_holdout,
        anomaly_selection_metric=anomaly_selection_metric,
    )

    outcome_only_baseline = evaluate_outcome_only_baseline(base_rows, test_rows, seed)
    deployment_fit = fit_and_evaluate_models(
        base_rows,
        calib_rows,
        test_rows,
        seed=seed,
        include_artifacts=True,
        include_curves=True,
    )
    thresholds_by_model = deployment_fit["thresholds"]
    metrics_by_model = deployment_fit["metrics"]
    thresholds = transpose_metric_namespaces(thresholds_by_model)
    metrics = transpose_metric_namespaces(metrics_by_model)
    curves = deployment_fit["curves"]
    feature_importance = deployment_fit["feature_importance"]
    model_payloads = deployment_fit["model_payloads"]
    artifact_sizes = deployment_fit["artifact_sizes"]
    novelty = deployment_fit["novelty"]
    calibrators = deployment_fit["calibrators"]
    aggregate_model_metrics = repeated_grouped_cv.get("aggregate", {}).get("models", {})
    ranking_inputs = {
        name: aggregate_ranking_payload(
            name,
            aggregate_model_metrics.get(name, {}),
            artifact_sizes[name],
            anomaly_selection_metric=anomaly_selection_metric,
        )
        for name in model_payloads
    }
    deployment_blocked_reason: str | None = None
    if not split_has_full_group_support(split_summary):
        deployment_blocked_reason = "split_summary_missing_class_group_support"
    elif not dataset_sanity_report["passed"]:
        deployment_blocked_reason = "dataset_sanity_trivial_task"
    elif not all(
        has_full_class_support(metrics_by_model[name][MODEL_ONLY_NAMESPACE]["multiclass_metrics"])
        for name in metrics_by_model
    ):
        deployment_blocked_reason = "test_split_missing_class_support"
    elif not generalization_gate["passed"]:
        deployment_blocked_reason = "generalization_metrics_below_threshold"

    winner_name: str | None = None
    if deployment_blocked_reason is None:
        eligible_models = [name for name in generalization_gate["eligible_models"] if name in ranking_inputs]
        winner_name = sorted(
            eligible_models,
            key=lambda name: ranking_key(
                ranking_inputs[name],
                selection_metric_order=selection_metric_order,
            ),
        )[0]

    if make_plots:
        comparison_metrics = {
            name: {
                SELECTION_METRIC_CLASS_MACRO_F1: aggregate_mean(
                    aggregate_model_metrics.get(name, {}),
                    MODEL_ONLY_NAMESPACE,
                    "multiclass_metrics",
                    "macro_f1",
                ),
                SELECTION_METRIC_MODEL_CYBER_F1: aggregate_mean(
                    aggregate_model_metrics.get(name, {}),
                    MODEL_ONLY_NAMESPACE,
                    "cyber_binary_metrics",
                    "f1",
                ),
                "model_only.anomaly_f1": aggregate_mean(
                    aggregate_model_metrics.get(name, {}),
                    MODEL_ONLY_NAMESPACE,
                    "anomaly_binary_metrics",
                    "f1",
                ),
                SELECTION_METRIC_STACKED_ANOMALY_F1: aggregate_mean(
                    aggregate_model_metrics.get(name, {}),
                    STACKED_DETECTOR_NAMESPACE,
                    "anomaly_binary_metrics",
                    "f1",
                ),
                "stacked_detector.anomaly_pr_auc": aggregate_mean(
                    aggregate_model_metrics.get(name, {}),
                    STACKED_DETECTOR_NAMESPACE,
                    "anomaly_binary_metrics",
                    "pr_auc",
                ),
            }
            for name in aggregate_model_metrics
        }
        comparison_errors = {
            name: {
                SELECTION_METRIC_CLASS_MACRO_F1: aggregate_stat(
                    aggregate_model_metrics.get(name, {}),
                    "std",
                    MODEL_ONLY_NAMESPACE,
                    "multiclass_metrics",
                    "macro_f1",
                ),
                SELECTION_METRIC_MODEL_CYBER_F1: aggregate_stat(
                    aggregate_model_metrics.get(name, {}),
                    "std",
                    MODEL_ONLY_NAMESPACE,
                    "cyber_binary_metrics",
                    "f1",
                ),
                "model_only.anomaly_f1": aggregate_stat(
                    aggregate_model_metrics.get(name, {}),
                    "std",
                    MODEL_ONLY_NAMESPACE,
                    "anomaly_binary_metrics",
                    "f1",
                ),
                SELECTION_METRIC_STACKED_ANOMALY_F1: aggregate_stat(
                    aggregate_model_metrics.get(name, {}),
                    "std",
                    STACKED_DETECTOR_NAMESPACE,
                    "anomaly_binary_metrics",
                    "f1",
                ),
                "stacked_detector.anomaly_pr_auc": aggregate_stat(
                    aggregate_model_metrics.get(name, {}),
                    "std",
                    STACKED_DETECTOR_NAMESPACE,
                    "anomaly_binary_metrics",
                    "pr_auc",
                ),
            }
            for name in aggregate_model_metrics
        }
        plot_comparison(
            comparison_metrics,
            report_dir,
            metric_names=[
                SELECTION_METRIC_CLASS_MACRO_F1,
                SELECTION_METRIC_MODEL_CYBER_F1,
                "model_only.anomaly_f1",
                SELECTION_METRIC_STACKED_ANOMALY_F1,
                "stacked_detector.anomaly_pr_auc",
            ],
            title="Model Only vs Stacked Detector",
            error_by_model=comparison_errors,
        )
        plot_curve(
            curves[MODEL_ONLY_NAMESPACE]["cyber_roc"],
            report_dir,
            "roc.png",
            "false positive rate",
            "true positive rate",
            title="model_only.cyber ROC",
        )
        plot_curve(
            curves[MODEL_ONLY_NAMESPACE]["cyber_pr"],
            report_dir,
            "pr.png",
            "recall",
            "precision",
            title="model_only.cyber PR",
        )
        plot_curve(
            curves[MODEL_ONLY_NAMESPACE]["anomaly_roc"],
            report_dir,
            "model_only_anomaly_roc.png",
            "false positive rate",
            "true positive rate",
            title="model_only.anomaly ROC",
        )
        plot_curve(
            curves[MODEL_ONLY_NAMESPACE]["anomaly_pr"],
            report_dir,
            "model_only_anomaly_pr.png",
            "recall",
            "precision",
            title="model_only.anomaly PR",
        )
        plot_curve(
            curves[STACKED_DETECTOR_NAMESPACE]["anomaly_roc"],
            report_dir,
            "stacked_detector_anomaly_roc.png",
            "false positive rate",
            "true positive rate",
            title="stacked_detector.anomaly ROC",
        )
        plot_curve(
            curves[STACKED_DETECTOR_NAMESPACE]["anomaly_pr"],
            report_dir,
            "stacked_detector_anomaly_pr.png",
            "recall",
            "precision",
            title="stacked_detector.anomaly PR",
        )

    class_counts = Counter(class_name(int(row["label"])) for row in rows)
    deployment_winner: dict[str, Any] | None = None
    if winner_name is not None:
        export_runtime_files(output_dir, model_payloads, novelty, calibrators, winner_name)
        deployment_winner = {
            "name": winner_name,
            "selection_basis": "evaluation.repeated_grouped_cv.aggregate.models",
            "selection_metric_order": list(selection_metric_order),
            "selection_metrics": dict(ranking_inputs[winner_name]["selection_metrics"]),
            "runtime_kind": LEGACY_STACKED_RUNTIME_KIND,
            "model_path": str((output_dir / "models" / LEGACY_PRIMARY_MODEL_ARTIFACT_NAME).resolve()),
            "manifest_path": str((output_dir / "models" / RUNTIME_BUNDLE_MANIFEST_NAME).resolve()),
            "calibrator_path": str((output_dir / "models" / "calibrator.json").resolve()),
            "artifact_bytes": int(artifact_sizes[winner_name]),
            "deployment_split_metrics": {
                SELECTION_METRIC_CLASS_MACRO_F1: float(
                    metrics_by_model[winner_name][MODEL_ONLY_NAMESPACE]["multiclass_metrics"]["macro_f1"]
                ),
                SELECTION_METRIC_MIN_PER_CLASS_RECALL: float(
                    min_per_class_recall(metrics_by_model[winner_name][MODEL_ONLY_NAMESPACE]["multiclass_metrics"])
                ),
                SELECTION_METRIC_MODEL_CYBER_F1: float(
                    metrics_by_model[winner_name][MODEL_ONLY_NAMESPACE]["cyber_binary_metrics"]["f1"]
                ),
                SELECTION_METRIC_STACKED_ANOMALY_F1: float(
                    metrics_by_model[winner_name][STACKED_DETECTOR_NAMESPACE]["anomaly_binary_metrics"]["f1"]
                ),
            },
        }
    report = {
        "dataset": str(dataset_path),
        "schema_version": SCHEMA_VERSION,
        "training_path": training_path_summary(
            training_path_name,
            blue_feature_policy_name=blue_feature_policy_name,
        ),
        "comparison_only": True,
        "rows": len(rows),
        "class_names": CLASS_NAMES,
        "class_counts": dict(class_counts),
        "feature_names": list(FEATURE_NAMES),
        "feature_sets": feature_sets_report(
            blue_feature_policy_name,
            training_path_name=training_path_name,
        ),
        "blue_feature_policy": blue_feature_policy,
        "history_featurization": history_featurization,
        "training_group_key": training_group_key,
        "novelty_adaptation": novelty_adaptation,
        "split_summary": split_summary,
        "split_episode_signatures": {
            "summary": split_episode_signature_report["summary"],
            "per_class": split_episode_signature_report["per_class"],
            "split_overlap": split_episode_signature_report["split_overlap"],
            "enforced": split_episode_signatures_enforced,
            "report_path": str(split_episode_signature_path.resolve()),
        },
        "dataset_sanity": {
            "passed": bool(dataset_sanity_report["passed"]),
            "eligible_for_deployment": bool(dataset_sanity_report["eligible_for_deployment"]),
            "checks": list(dataset_sanity_report["checks"]),
            "blocking_issues": list(dataset_sanity_report["blocking_issues"]),
            "command_overlap": dict(dataset_sanity_report["command_overlap"]["summary"]),
            "service_overlap": dict(dataset_sanity_report["service_overlap"]["summary"]),
            "episode_signatures": {
                "summary": dict(dataset_sanity_report["episode_signatures"]["summary"]),
                "per_class": dict(dataset_sanity_report["episode_signatures"]["per_class"]),
                "diversity_check": dict(dataset_sanity_report["episode_signatures"]["diversity_check"]),
            },
            "request_tuple_purity": {
                "feature_names": list(dataset_sanity_report["request_tuple_purity"]["feature_names"]),
                "bucket_sizes": dict(dataset_sanity_report["request_tuple_purity"]["bucket_sizes"]),
                "summary": dict(dataset_sanity_report["request_tuple_purity"]["summary"]),
            },
            "baselines": {
                "command_only": command_only_baseline,
                "request_only": request_only_baseline,
                "protocol_only": protocol_only_baseline,
                "raw_protocol_shortcuts": raw_protocol_shortcuts_baseline,
            },
            "report_path": str(dataset_sanity_path.resolve()),
        },
        "evaluation": {
            "seeds": evaluation_seeds,
            "repeated_grouped_cv": repeated_grouped_cv,
            "scenario_family_holdout": scenario_family_holdout,
            "command_family_holdout": command_family_holdout,
            "protocol_family_holdout": protocol_family_holdout,
            "deployment_gate": generalization_gate,
        },
        "thresholds": thresholds,
        "feature_importance": {"random_forest": feature_importance},
        "simple_command_baseline": command_only_baseline,
        "request_only_baseline": request_only_baseline,
        "outcome_only_baseline": outcome_only_baseline,
        "protocol_only_baseline": protocol_only_baseline,
        "raw_protocol_shortcuts_baseline": raw_protocol_shortcuts_baseline,
        "metrics": metrics,
        "ranking_inputs": ranking_inputs,
        "ranking_source": "evaluation.repeated_grouped_cv.aggregate.models",
        "ranking_metric_order": list(selection_metric_order),
        "deployment_ready": deployment_blocked_reason is None,
        "deployment_blocked_reason": deployment_blocked_reason,
        "deployment_winner": deployment_winner,
    }
    save_json(report_dir / "metrics.json", report)
    deployed_model = winner_name or "none"
    rf_grouped_cv_macro = aggregate_mean(
        aggregate_model_metrics.get("random_forest", {}),
        MODEL_ONLY_NAMESPACE,
        "multiclass_metrics",
        "macro_f1",
    )
    nn_grouped_cv_macro = aggregate_mean(
        aggregate_model_metrics.get("neural_net", {}),
        MODEL_ONLY_NAMESPACE,
        "multiclass_metrics",
        "macro_f1",
    )
    deployed_selection_metrics = {} if winner_name is None else dict(ranking_inputs[winner_name]["selection_metrics"])
    deployed_anomaly_f1 = 0.0 if winner_name is None else float(
        ranking_inputs[winner_name]["selection_metrics"][anomaly_selection_metric]
    )
    grouped_cv_class_group_counts = repeated_grouped_cv.get("class_group_counts", {})
    grouped_cv_test_support_values = (
        repeated_grouped_cv.get("support_summary", {})
        .get("test_class_group_counts", {})
        .get("values", {})
    )
    summary = (
        f"deployed_model={deployed_model}\n"
        f"training_path={training_path_name}\n"
        "comparison_only=true\n"
        f"runtime_kind={LEGACY_STACKED_RUNTIME_KIND}\n"
        f"runtime_primary_artifact={LEGACY_PRIMARY_MODEL_ARTIFACT_NAME}\n"
        f"selection_basis=evaluation.repeated_grouped_cv.aggregate.models\n"
        f"selection_metric_order={json.dumps(selection_metric_order, separators=(',', ':'))}\n"
        f"rf_grouped_cv_{SELECTION_METRIC_CLASS_MACRO_F1.replace('.', '_')}_mean={rf_grouped_cv_macro:.4f}\n"
        f"nn_grouped_cv_{SELECTION_METRIC_CLASS_MACRO_F1.replace('.', '_')}_mean={nn_grouped_cv_macro:.4f}\n"
        f"deployed_grouped_cv_{anomaly_selection_metric.replace('.', '_')}_mean={deployed_anomaly_f1:.4f}\n"
        f"deployed_selection_metrics={json.dumps(deployed_selection_metrics, sort_keys=True, separators=(',', ':'))}\n"
        f"group_split={training_group_key}\n"
        f"split_heldout_groups_per_class={json.dumps(split_summary.get('heldout_groups_per_class', {}), sort_keys=True, separators=(',', ':'))}\n"
        f"split_base_class_group_counts={json.dumps(split_summary.get('base_class_group_counts', {}), sort_keys=True, separators=(',', ':'))}\n"
        f"split_calibration_class_group_counts={json.dumps(split_summary.get('calibration_class_group_counts', {}), sort_keys=True, separators=(',', ':'))}\n"
        f"split_test_class_group_counts={json.dumps(split_summary.get('test_class_group_counts', {}), sort_keys=True, separators=(',', ':'))}\n"
        f"grouped_cv_class_group_counts={json.dumps(grouped_cv_class_group_counts, sort_keys=True, separators=(',', ':'))}\n"
        f"grouped_cv_test_class_group_count_values={json.dumps(grouped_cv_test_support_values, sort_keys=True, separators=(',', ':'))}\n"
        f"novelty_adaptation_mode={novelty_adaptation['mode']}\n"
        f"evaluation_seeds={json.dumps(evaluation_seeds, separators=(',', ':'))}\n"
        f"scenario_family_holdout_feasible={str(bool(scenario_family_holdout['feasible'])).lower()}\n"
        f"command_family_holdout_feasible={str(bool(command_family_holdout['feasible'])).lower()}\n"
        f"protocol_family_holdout_feasible={str(bool(protocol_family_holdout['feasible'])).lower()}\n"
        f"dataset_sanity_passed={str(bool(dataset_sanity_report['passed'])).lower()}\n"
        f"dataset_sanity_blocking_issues={json.dumps(dataset_sanity_report['blocking_issues'], sort_keys=True, separators=(',', ':'))}\n"
        f"class_counts={json.dumps(dict(class_counts), separators=(',', ':'))}\n"
    )
    if deployment_blocked_reason is not None:
        summary += f"deployment_blocked_reason={deployment_blocked_reason}\n"
    (report_dir / "summary.txt").write_text(summary, encoding="utf-8")
    if deployment_blocked_reason is not None:
        raise SystemExit(
            "Training completed for analysis but deployment/export was blocked. "
            f"reason={deployment_blocked_reason}"
        )
    return report


def _run_training_poster_default(
    dataset_path: Path,
    output_dir: Path,
    seed: int,
    make_plots: bool,
    blue_feature_policy_name: str,
) -> dict[str, Any]:
    training_path_name = POSTER_DEFAULT_TRAINING_PATH_NAME
    selection_metric_order = active_selection_metric_order(training_path_name)
    anomaly_selection_metric = active_anomaly_selection_metric(training_path_name)
    require_training_deps()
    try:
        blue_feature_policy = validate_blue_feature_names(
            POSTER_DEFAULT_REQUEST_ONLY_BASELINE_FEATURE_NAMES,
            blue_feature_policy_name,
        )
    except BlueFeaturePolicyError as exc:
        raise SystemExit(
            f"Blue feature policy violation for {blue_feature_policy_name}: {exc}"
        ) from None
    novelty_adaptation = novelty_adaptation_report()
    source_records, record_provenance = load_canonical_training_records(dataset_path)
    if not _records_support_group_key(source_records, TRAINING_GROUP_KEY):
        raise SystemExit(
            f"canonical command rows must include {TRAINING_GROUP_KEY} on every row for grouped training replay"
        )
    history_featurization = poster_history_featurization_report(record_provenance)
    training_group_key = TRAINING_GROUP_KEY
    if len(source_records) < 100:
        raise SystemExit("Need at least 100 generated rows before training")

    def canonical_materialize(
        base_records: list[dict[str, Any]],
        calib_records: list[dict[str, Any]],
        test_records: list[dict[str, Any]],
        group_key: str,
    ) -> tuple[list[dict[str, Any]], list[dict[str, Any]], list[dict[str, Any]]]:
        return materialize_canonical_split_rows(
            base_records,
            calib_records,
            test_records,
            group_key,
            blue_feature_policy_name=blue_feature_policy_name,
        )

    base_records, calib_records, test_records, split_summary = grouped_record_split(
        source_records,
        seed=seed,
        group_key=training_group_key,
    )
    base_rows, calib_rows, test_rows = canonical_materialize(
        base_records,
        calib_records,
        test_records,
        training_group_key,
    )
    split_summary = dict(split_summary)
    split_summary["history_featurization"] = dict(history_featurization)
    rows = [*base_rows, *calib_rows, *test_rows]
    split_assignments: dict[int, str] = {}
    for split_name, split_rows in (("base", base_rows), ("calibration", calib_rows), ("test", test_rows)):
        for row in split_rows:
            split_assignments[int(row[training_group_key])] = split_name
    split_episode_signature_report = build_episode_signature_report(
        rows,
        group_key=training_group_key,
        split_assignments=split_assignments,
    )
    report_dir = output_dir / "reports"
    report_dir.mkdir(parents=True, exist_ok=True)
    split_episode_signature_path = report_dir / "split_episode_signatures.json"
    save_json(split_episode_signature_path, split_episode_signature_report)
    split_episode_signatures_enforced = has_structural_signature_signal(split_episode_signature_report)
    if split_episode_signatures_enforced:
        assert_split_episode_separation(split_episode_signature_report)
    dataset_sanity_report = audit_dataset_sanity(
        rows,
        base_rows,
        test_rows,
        seed=seed,
        group_key=training_group_key,
        command_only_feature_names=POSTER_DEFAULT_COMMAND_BASELINE_FEATURE_NAMES,
        request_only_feature_names=POSTER_DEFAULT_REQUEST_ONLY_BASELINE_FEATURE_NAMES,
        protocol_only_feature_names=PROTOCOL_ONLY_BASELINE_FEATURE_NAMES,
        raw_protocol_shortcut_feature_names=RAW_PROTOCOL_SHORTCUT_BASELINE_FEATURE_NAMES,
        request_tuple_purity_feature_names=POSTER_DEFAULT_REQUEST_TUPLE_PURITY_FEATURE_NAMES,
        request_tuple_purity_bucket_sizes=POSTER_DEFAULT_REQUEST_TUPLE_PURITY_BUCKETS,
    )
    dataset_sanity_path = report_dir / "dataset_sanity.json"
    save_json(dataset_sanity_path, dataset_sanity_report)
    command_only_baseline = dict(dataset_sanity_report["baselines"]["command_only"])
    request_only_baseline = dict(dataset_sanity_report["baselines"]["request_only"])
    protocol_only_baseline = dict(dataset_sanity_report["baselines"]["protocol_only"])
    raw_protocol_shortcuts_baseline = dict(dataset_sanity_report["baselines"]["raw_protocol_shortcuts"])
    outcome_only_baseline = baseline_not_applicable(
        [],
        reason="terminal_outcomes_not_allowed_in_poster_path",
    )

    evaluation_seeds = evaluation_seed_sequence(seed)
    repeated_grouped_cv = run_repeated_grouped_cv(
        source_records,
        training_group_key,
        evaluation_seeds,
        fit_and_evaluate_fn=fit_and_evaluate_poster_model,
        materialize_split_rows_fn=canonical_materialize,
        command_only_feature_names=POSTER_DEFAULT_COMMAND_BASELINE_FEATURE_NAMES,
        request_only_feature_names=POSTER_DEFAULT_REQUEST_ONLY_BASELINE_FEATURE_NAMES,
        outcome_only_feature_names=[],
        protocol_only_feature_names=PROTOCOL_ONLY_BASELINE_FEATURE_NAMES,
        raw_protocol_shortcut_feature_names=RAW_PROTOCOL_SHORTCUT_BASELINE_FEATURE_NAMES,
    )
    scenario_family_holdout = run_family_holdout_evaluation(
        source_records,
        training_group_key,
        evaluation_seeds,
        family_field="attack_family",
        evaluation_name="scenario_family_holdout",
        fit_and_evaluate_fn=fit_and_evaluate_poster_model,
        materialize_split_rows_fn=canonical_materialize,
        command_only_feature_names=POSTER_DEFAULT_COMMAND_BASELINE_FEATURE_NAMES,
        request_only_feature_names=POSTER_DEFAULT_REQUEST_ONLY_BASELINE_FEATURE_NAMES,
        outcome_only_feature_names=[],
        protocol_only_feature_names=PROTOCOL_ONLY_BASELINE_FEATURE_NAMES,
        raw_protocol_shortcut_feature_names=RAW_PROTOCOL_SHORTCUT_BASELINE_FEATURE_NAMES,
    )
    command_family_holdout = run_family_holdout_evaluation(
        source_records,
        training_group_key,
        evaluation_seeds,
        family_field="command",
        evaluation_name="command_family_holdout",
        fit_and_evaluate_fn=fit_and_evaluate_poster_model,
        materialize_split_rows_fn=canonical_materialize,
        command_only_feature_names=POSTER_DEFAULT_COMMAND_BASELINE_FEATURE_NAMES,
        request_only_feature_names=POSTER_DEFAULT_REQUEST_ONLY_BASELINE_FEATURE_NAMES,
        outcome_only_feature_names=[],
        protocol_only_feature_names=PROTOCOL_ONLY_BASELINE_FEATURE_NAMES,
        raw_protocol_shortcut_feature_names=RAW_PROTOCOL_SHORTCUT_BASELINE_FEATURE_NAMES,
    )
    protocol_family_holdout = run_family_holdout_evaluation(
        source_records,
        training_group_key,
        evaluation_seeds,
        family_field="protocol_family",
        evaluation_name="protocol_family_holdout",
        fit_and_evaluate_fn=fit_and_evaluate_poster_model,
        materialize_split_rows_fn=canonical_materialize,
        command_only_feature_names=POSTER_DEFAULT_COMMAND_BASELINE_FEATURE_NAMES,
        request_only_feature_names=POSTER_DEFAULT_REQUEST_ONLY_BASELINE_FEATURE_NAMES,
        outcome_only_feature_names=[],
        protocol_only_feature_names=PROTOCOL_ONLY_BASELINE_FEATURE_NAMES,
        raw_protocol_shortcut_feature_names=RAW_PROTOCOL_SHORTCUT_BASELINE_FEATURE_NAMES,
    )
    generalization_gate = build_generalization_gate(
        repeated_grouped_cv,
        scenario_family_holdout,
        command_family_holdout,
        protocol_family_holdout,
        anomaly_selection_metric=anomaly_selection_metric,
    )

    deployment_fit = fit_and_evaluate_poster_model(
        base_rows,
        calib_rows,
        test_rows,
        seed=seed,
        include_artifacts=True,
        include_curves=True,
    )
    thresholds_by_model = deployment_fit["thresholds"]
    metrics_by_model = deployment_fit["metrics"]
    thresholds = transpose_metric_namespaces(thresholds_by_model)
    metrics = transpose_metric_namespaces(metrics_by_model)
    blue_model_report = dict(deployment_fit["blue_model"])
    curves = deployment_fit["curves"]
    feature_importance = deployment_fit["feature_importance"]
    model_payloads = deployment_fit["model_payloads"]
    artifact_sizes = deployment_fit["artifact_sizes"]
    aggregate_model_metrics = repeated_grouped_cv.get("aggregate", {}).get("models", {})
    ranking_inputs = {
        name: aggregate_ranking_payload(
            name,
            aggregate_model_metrics.get(name, {}),
            artifact_sizes[name],
            anomaly_selection_metric=anomaly_selection_metric,
        )
        for name in model_payloads
    }
    deployment_blocked_reason: str | None = None
    if not split_has_full_group_support(split_summary):
        deployment_blocked_reason = "split_summary_missing_class_group_support"
    elif not dataset_sanity_report["passed"]:
        deployment_blocked_reason = "dataset_sanity_trivial_task"
    elif not all(
        has_full_class_support(metrics_by_model[name][MODEL_ONLY_NAMESPACE]["multiclass_metrics"])
        for name in metrics_by_model
    ):
        deployment_blocked_reason = "test_split_missing_class_support"
    elif not generalization_gate["passed"]:
        deployment_blocked_reason = "generalization_metrics_below_threshold"

    winner_name: str | None = None
    if deployment_blocked_reason is None:
        eligible_models = [name for name in generalization_gate["eligible_models"] if name in ranking_inputs]
        winner_name = sorted(
            eligible_models,
            key=lambda name: ranking_key(
                ranking_inputs[name],
                selection_metric_order=selection_metric_order,
            ),
        )[0]

    if make_plots:
        comparison_metrics = {
            name: {
                SELECTION_METRIC_CLASS_MACRO_F1: aggregate_mean(
                    aggregate_model_metrics.get(name, {}),
                    MODEL_ONLY_NAMESPACE,
                    "multiclass_metrics",
                    "macro_f1",
                ),
                SELECTION_METRIC_MODEL_CYBER_F1: aggregate_mean(
                    aggregate_model_metrics.get(name, {}),
                    MODEL_ONLY_NAMESPACE,
                    "cyber_binary_metrics",
                    "f1",
                ),
                SELECTION_METRIC_MODEL_ANOMALY_F1: aggregate_mean(
                    aggregate_model_metrics.get(name, {}),
                    MODEL_ONLY_NAMESPACE,
                    "anomaly_binary_metrics",
                    "f1",
                ),
            }
            for name in aggregate_model_metrics
        }
        comparison_errors = {
            name: {
                SELECTION_METRIC_CLASS_MACRO_F1: aggregate_stat(
                    aggregate_model_metrics.get(name, {}),
                    "std",
                    MODEL_ONLY_NAMESPACE,
                    "multiclass_metrics",
                    "macro_f1",
                ),
                SELECTION_METRIC_MODEL_CYBER_F1: aggregate_stat(
                    aggregate_model_metrics.get(name, {}),
                    "std",
                    MODEL_ONLY_NAMESPACE,
                    "cyber_binary_metrics",
                    "f1",
                ),
                SELECTION_METRIC_MODEL_ANOMALY_F1: aggregate_stat(
                    aggregate_model_metrics.get(name, {}),
                    "std",
                    MODEL_ONLY_NAMESPACE,
                    "anomaly_binary_metrics",
                    "f1",
                ),
            }
            for name in aggregate_model_metrics
        }
        plot_comparison(
            comparison_metrics,
            report_dir,
            metric_names=[
                SELECTION_METRIC_CLASS_MACRO_F1,
                SELECTION_METRIC_MODEL_CYBER_F1,
                SELECTION_METRIC_MODEL_ANOMALY_F1,
            ],
            title="Poster Default Neural Model",
            error_by_model=comparison_errors,
        )
        plot_curve(
            curves[MODEL_ONLY_NAMESPACE]["cyber_roc"],
            report_dir,
            "roc.png",
            "false positive rate",
            "true positive rate",
            title="model_only.cyber ROC",
        )
        plot_curve(
            curves[MODEL_ONLY_NAMESPACE]["cyber_pr"],
            report_dir,
            "pr.png",
            "recall",
            "precision",
            title="model_only.cyber PR",
        )
        plot_curve(
            curves[MODEL_ONLY_NAMESPACE]["anomaly_roc"],
            report_dir,
            "model_only_anomaly_roc.png",
            "false positive rate",
            "true positive rate",
            title="model_only.anomaly ROC",
        )
        plot_curve(
            curves[MODEL_ONLY_NAMESPACE]["anomaly_pr"],
            report_dir,
            "model_only_anomaly_pr.png",
            "recall",
            "precision",
            title="model_only.anomaly PR",
        )

    class_counts = Counter(class_name(int(row["label"])) for row in rows)
    deployment_winner: dict[str, Any] | None = None
    if winner_name is not None:
        export_runtime_files_poster(output_dir, model_payloads, winner_name)
        deployment_winner = {
            "name": winner_name,
            "selection_basis": "evaluation.repeated_grouped_cv.aggregate.models",
            "selection_metric_order": list(selection_metric_order),
            "selection_metrics": dict(ranking_inputs[winner_name]["selection_metrics"]),
            "runtime_kind": POSTER_BLUE_RUNTIME_KIND,
            "model_path": str((output_dir / "models" / POSTER_BLUE_MODEL_ARTIFACT_NAME).resolve()),
            "manifest_path": str((output_dir / "models" / RUNTIME_BUNDLE_MANIFEST_NAME).resolve()),
            "artifact_bytes": int(artifact_sizes[winner_name]),
            "deployment_split_metrics": {
                SELECTION_METRIC_CLASS_MACRO_F1: float(
                    metrics_by_model[winner_name][MODEL_ONLY_NAMESPACE]["multiclass_metrics"]["macro_f1"]
                ),
                SELECTION_METRIC_MIN_PER_CLASS_RECALL: float(
                    min_per_class_recall(metrics_by_model[winner_name][MODEL_ONLY_NAMESPACE]["multiclass_metrics"])
                ),
                SELECTION_METRIC_MODEL_CYBER_F1: float(
                    metrics_by_model[winner_name][MODEL_ONLY_NAMESPACE]["cyber_binary_metrics"]["f1"]
                ),
                SELECTION_METRIC_MODEL_ANOMALY_F1: float(
                    metrics_by_model[winner_name][MODEL_ONLY_NAMESPACE]["anomaly_binary_metrics"]["f1"]
                ),
            },
        }
    analysis_runtime_bundle: dict[str, Any] | None = None
    if deployment_blocked_reason is not None:
        analysis_runtime_bundle = export_runtime_files_poster_research_candidate(
            output_dir,
            model_payloads,
            model_name="neural_net",
            deployment_blocked_reason=deployment_blocked_reason,
        )
    report = {
        "dataset": str(dataset_path),
        "schema_version": SCHEMA_VERSION,
        "training_path": training_path_summary(
            training_path_name,
            blue_feature_policy_name=blue_feature_policy_name,
        ),
        "comparison_only": False,
        "rows": len(rows),
        "class_names": CLASS_NAMES,
        "class_counts": dict(class_counts),
        "feature_names": list(POSTER_DEFAULT_REQUEST_ONLY_BASELINE_FEATURE_NAMES),
        "feature_sets": feature_sets_report(
            blue_feature_policy_name,
            training_path_name=training_path_name,
        ),
        "blue_feature_policy": blue_feature_policy,
        "blue_model": blue_model_report,
        "history_featurization": history_featurization,
        "training_group_key": training_group_key,
        "novelty_adaptation": novelty_adaptation,
        "split_summary": split_summary,
        "split_episode_signatures": {
            "summary": split_episode_signature_report["summary"],
            "per_class": split_episode_signature_report["per_class"],
            "split_overlap": split_episode_signature_report["split_overlap"],
            "enforced": split_episode_signatures_enforced,
            "report_path": str(split_episode_signature_path.resolve()),
        },
        "dataset_sanity": {
            "passed": bool(dataset_sanity_report["passed"]),
            "eligible_for_deployment": bool(dataset_sanity_report["eligible_for_deployment"]),
            "checks": list(dataset_sanity_report["checks"]),
            "blocking_issues": list(dataset_sanity_report["blocking_issues"]),
            "command_overlap": dict(dataset_sanity_report["command_overlap"]["summary"]),
            "service_overlap": dict(dataset_sanity_report["service_overlap"]["summary"]),
            "episode_signatures": {
                "summary": dict(dataset_sanity_report["episode_signatures"]["summary"]),
                "per_class": dict(dataset_sanity_report["episode_signatures"]["per_class"]),
                "diversity_check": dict(dataset_sanity_report["episode_signatures"]["diversity_check"]),
            },
            "request_tuple_purity": {
                "feature_names": list(dataset_sanity_report["request_tuple_purity"]["feature_names"]),
                "bucket_sizes": dict(dataset_sanity_report["request_tuple_purity"]["bucket_sizes"]),
                "summary": dict(dataset_sanity_report["request_tuple_purity"]["summary"]),
            },
            "baselines": {
                "command_only": command_only_baseline,
                "request_only": request_only_baseline,
                "protocol_only": protocol_only_baseline,
                "raw_protocol_shortcuts": raw_protocol_shortcuts_baseline,
            },
            "report_path": str(dataset_sanity_path.resolve()),
        },
        "evaluation": {
            "seeds": evaluation_seeds,
            "repeated_grouped_cv": repeated_grouped_cv,
            "scenario_family_holdout": scenario_family_holdout,
            "command_family_holdout": command_family_holdout,
            "protocol_family_holdout": protocol_family_holdout,
            "deployment_gate": generalization_gate,
        },
        "thresholds": thresholds,
        "feature_importance": {"neural_net": list(feature_importance)},
        "simple_command_baseline": command_only_baseline,
        "request_only_baseline": request_only_baseline,
        "outcome_only_baseline": outcome_only_baseline,
        "protocol_only_baseline": protocol_only_baseline,
        "raw_protocol_shortcuts_baseline": raw_protocol_shortcuts_baseline,
        "metrics": metrics,
        "ranking_inputs": ranking_inputs,
        "ranking_source": "evaluation.repeated_grouped_cv.aggregate.models",
        "ranking_metric_order": list(selection_metric_order),
        "deployment_ready": deployment_blocked_reason is None,
        "deployment_blocked_reason": deployment_blocked_reason,
        "deployment_winner": deployment_winner,
        "analysis_runtime_bundle": analysis_runtime_bundle,
    }
    save_json(report_dir / "metrics.json", report)
    deployed_model = winner_name or "none"
    nn_grouped_cv_macro = aggregate_mean(
        aggregate_model_metrics.get("neural_net", {}),
        MODEL_ONLY_NAMESPACE,
        "multiclass_metrics",
        "macro_f1",
    )
    deployed_selection_metrics = {} if winner_name is None else dict(ranking_inputs[winner_name]["selection_metrics"])
    deployed_anomaly_f1 = 0.0 if winner_name is None else float(
        ranking_inputs[winner_name]["selection_metrics"][anomaly_selection_metric]
    )
    grouped_cv_class_group_counts = repeated_grouped_cv.get("class_group_counts", {})
    grouped_cv_test_support_values = (
        repeated_grouped_cv.get("support_summary", {})
        .get("test_class_group_counts", {})
        .get("values", {})
    )
    summary = (
        f"deployed_model={deployed_model}\n"
        f"training_path={training_path_name}\n"
        "comparison_only=false\n"
        f"runtime_kind={POSTER_BLUE_RUNTIME_KIND}\n"
        f"runtime_primary_artifact={POSTER_BLUE_MODEL_ARTIFACT_NAME}\n"
        f"blue_model_family={str(blue_model_report['architecture']['family'])}\n"
        f"blue_model_best_epoch={int(blue_model_report['training_summary']['best_epoch'])}\n"
        f"blue_model_best_validation_cross_entropy={float(blue_model_report['training_summary']['best_validation_cross_entropy']):.6f}\n"
        f"selection_basis=evaluation.repeated_grouped_cv.aggregate.models\n"
        f"selection_metric_order={json.dumps(selection_metric_order, separators=(',', ':'))}\n"
        f"nn_grouped_cv_{SELECTION_METRIC_CLASS_MACRO_F1.replace('.', '_')}_mean={nn_grouped_cv_macro:.4f}\n"
        f"deployed_grouped_cv_{anomaly_selection_metric.replace('.', '_')}_mean={deployed_anomaly_f1:.4f}\n"
        f"deployed_selection_metrics={json.dumps(deployed_selection_metrics, sort_keys=True, separators=(',', ':'))}\n"
        f"group_split={training_group_key}\n"
        f"split_heldout_groups_per_class={json.dumps(split_summary.get('heldout_groups_per_class', {}), sort_keys=True, separators=(',', ':'))}\n"
        f"split_base_class_group_counts={json.dumps(split_summary.get('base_class_group_counts', {}), sort_keys=True, separators=(',', ':'))}\n"
        f"split_calibration_class_group_counts={json.dumps(split_summary.get('calibration_class_group_counts', {}), sort_keys=True, separators=(',', ':'))}\n"
        f"split_test_class_group_counts={json.dumps(split_summary.get('test_class_group_counts', {}), sort_keys=True, separators=(',', ':'))}\n"
        f"grouped_cv_class_group_counts={json.dumps(grouped_cv_class_group_counts, sort_keys=True, separators=(',', ':'))}\n"
        f"grouped_cv_test_class_group_count_values={json.dumps(grouped_cv_test_support_values, sort_keys=True, separators=(',', ':'))}\n"
        f"novelty_adaptation_mode={novelty_adaptation['mode']}\n"
        f"evaluation_seeds={json.dumps(evaluation_seeds, separators=(',', ':'))}\n"
        f"scenario_family_holdout_feasible={str(bool(scenario_family_holdout['feasible'])).lower()}\n"
        f"command_family_holdout_feasible={str(bool(command_family_holdout['feasible'])).lower()}\n"
        f"protocol_family_holdout_feasible={str(bool(protocol_family_holdout['feasible'])).lower()}\n"
        f"dataset_sanity_passed={str(bool(dataset_sanity_report['passed'])).lower()}\n"
        f"dataset_sanity_blocking_issues={json.dumps(dataset_sanity_report['blocking_issues'], sort_keys=True, separators=(',', ':'))}\n"
        f"class_counts={json.dumps(dict(class_counts), separators=(',', ':'))}\n"
    )
    if deployment_blocked_reason is not None:
        summary += f"deployment_blocked_reason={deployment_blocked_reason}\n"
        if analysis_runtime_bundle is not None:
            summary += f"analysis_runtime_bundle_dir={analysis_runtime_bundle['artifact_dir']}\n"
            summary += f"analysis_runtime_bundle_manifest={analysis_runtime_bundle['manifest_path']}\n"
    (report_dir / "summary.txt").write_text(summary, encoding="utf-8")
    if deployment_blocked_reason is not None:
        raise SystemExit(
            "Training completed for analysis but deployment/export was blocked. "
            f"reason={deployment_blocked_reason}"
        )
    return report


def run_training(
    dataset_path: Path,
    output_dir: Path,
    seed: int,
    make_plots: bool,
    blue_feature_policy_name: str | None = None,
    *,
    training_path_name: str = DEFAULT_TRAINING_PATH_NAME,
) -> dict[str, Any]:
    resolved_training_path_name = str(training_path_name)
    resolved_blue_feature_policy_name = blue_feature_policy_name or default_blue_feature_policy_name(
        resolved_training_path_name
    )
    if is_legacy_training_path(resolved_training_path_name):
        return _run_training_legacy_baseline(
            dataset_path=dataset_path,
            output_dir=output_dir,
            seed=seed,
            make_plots=make_plots,
            blue_feature_policy_name=resolved_blue_feature_policy_name,
        )
    if resolved_training_path_name == POSTER_DEFAULT_TRAINING_PATH_NAME:
        return _run_training_poster_default(
            dataset_path=dataset_path,
            output_dir=output_dir,
            seed=seed,
            make_plots=make_plots,
            blue_feature_policy_name=resolved_blue_feature_policy_name,
        )
    raise SystemExit(f"Unknown training path: {resolved_training_path_name}")


def score_packets(packet_path: Path, output_dir: Path, model_dir: Path) -> dict[str, Any]:
    packets = read_jsonl(packet_path)
    reset_key = "run_id" if _records_support_group_key(packets, "run_id") else None
    transactions = packets_to_transactions(packets, reset_key=reset_key)
    legacy_rows = transactions_to_rows(transactions, reset_key=reset_key)
    scored_rows = legacy_rows
    scoring_input_kind = "legacy_request_rows"
    canonical_rows: list[dict[str, Any]] = []
    runtime_kind = ""
    if model_dir.exists():
        bundle = load_runtime_bundle(model_dir)
        runtime_kind = str(getattr(bundle, "runtime_kind", ""))
        if getattr(bundle, "model", None) is not None and model_uses_canonical_features(list(getattr(bundle, "feature_names", []))):
            canonical_rows = canonical_rows_from_legacy_transactions(
                transactions,
                legacy_rows,
                source_artifact_paths=[str(packet_path.resolve())],
            )
            poster_rows = [
                canonical_row_to_training_row(
                    canonical_row,
                    policy_name=BLUE_FEATURE_POLICY_POSTER_DEFAULT,
                )
                for canonical_row in canonical_rows
            ]
            scored_rows = [bundle.score_row(row) for row in poster_rows]
            scoring_input_kind = "poster_canonical_request_rows"
        else:
            scored_rows = [bundle.score_row(row) for row in legacy_rows]
    scored_dir = output_dir / "scored"
    scored_dir.mkdir(parents=True, exist_ok=True)
    write_jsonl(scored_dir / "transactions.jsonl", transactions)
    if canonical_rows:
        write_jsonl(scored_dir / "canonical_rows.jsonl", canonical_rows)
    write_jsonl(scored_dir / "rows.jsonl", scored_rows)
    summary = {
        "packets": len(packets),
        "transactions": len(transactions),
        "rows": len(scored_rows),
        "model_dir": str(model_dir),
        "group_key": reset_key or "",
        "scoring_input_kind": scoring_input_kind,
        "runtime_kind": runtime_kind,
    }
    save_json(scored_dir / "summary.json", summary)
    return summary


def resolve_dataset_path(output_dir: Path, dataset: str | None) -> Path:
    return Path(dataset) if dataset else output_dir / "data" / "dataset.jsonl"


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Unified multi-protocol generation, train, and eval pipeline. "
            "Default train/run commands use the poster-default canonical neural path; "
            "legacy commands are comparison-only."
        )
    )
    sub = parser.add_subparsers(dest="cmd", required=True)

    gen = sub.add_parser("generate", help="generate shared artifacts for the default poster workflow")
    gen.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    gen.add_argument("--nominal-ratio", type=float, default=0.55)
    gen.add_argument("--seed", type=int, default=7)
    gen.add_argument("--protocol-mode", choices=SUPPORTED_GENERATION_PROTOCOL_MODES, default=DEFAULT_GENERATION_PROTOCOL_MODE)
    gen.add_argument("--mixed-fprime-ratio", type=float, default=DEFAULT_MIXED_FPRIME_RATIO)
    gen.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))

    train = sub.add_parser(
        "train",
        help="train the poster-default canonical neural detector",
        description="Train the poster-default canonical neural detector.",
    )
    train.add_argument("--dataset")
    train.add_argument("--seed", type=int, default=7)
    train.add_argument("--no-plots", action="store_true")
    train.add_argument("--blue-feature-policy")
    train.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))

    train_legacy = sub.add_parser(
        "train-legacy",
        help="train the legacy F-prime baseline (comparison-only)",
        description="Train the legacy F-prime baseline. This path is comparison-only.",
    )
    train_legacy.add_argument("--dataset")
    train_legacy.add_argument("--seed", type=int, default=7)
    train_legacy.add_argument("--no-plots", action="store_true")
    train_legacy.add_argument("--blue-feature-policy", default=BLUE_FEATURE_POLICY_LEGACY_REQUEST_TIME_BASELINE)
    train_legacy.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))

    run = sub.add_parser(
        "run",
        help="generate and train the poster-default canonical neural workflow",
        description="Generate data and train the poster-default canonical neural workflow.",
    )
    run.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    run.add_argument("--nominal-ratio", type=float, default=0.55)
    run.add_argument("--seed", type=int, default=7)
    run.add_argument("--protocol-mode", choices=SUPPORTED_GENERATION_PROTOCOL_MODES, default=DEFAULT_GENERATION_PROTOCOL_MODE)
    run.add_argument("--mixed-fprime-ratio", type=float, default=DEFAULT_MIXED_FPRIME_RATIO)
    run.add_argument("--skip-generate", action="store_true")
    run.add_argument("--no-plots", action="store_true")
    run.add_argument("--blue-feature-policy")
    run.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))

    run_legacy = sub.add_parser(
        "run-legacy",
        help="generate and train the legacy F-prime baseline (comparison-only)",
        description="Generate data and train the legacy F-prime baseline. This path is comparison-only.",
    )
    run_legacy.add_argument("--rows", type=int, default=DEFAULT_ROWS)
    run_legacy.add_argument("--nominal-ratio", type=float, default=0.55)
    run_legacy.add_argument("--seed", type=int, default=7)
    run_legacy.add_argument("--protocol-mode", choices=SUPPORTED_GENERATION_PROTOCOL_MODES, default=DEFAULT_GENERATION_PROTOCOL_MODE)
    run_legacy.add_argument("--mixed-fprime-ratio", type=float, default=DEFAULT_MIXED_FPRIME_RATIO)
    run_legacy.add_argument("--skip-generate", action="store_true")
    run_legacy.add_argument("--no-plots", action="store_true")
    run_legacy.add_argument("--blue-feature-policy", default=BLUE_FEATURE_POLICY_LEGACY_REQUEST_TIME_BASELINE)
    run_legacy.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))

    novelty = sub.add_parser(
        "train-novelty",
        help="fit the legacy novelty helper (comparison utility)",
        description="Fit the preserved legacy novelty helper. This is not the poster headline path.",
    )
    novelty.add_argument("--dataset")
    novelty.add_argument("--output", default=str(DEFAULT_MODEL_DIR / "novelty.cfg"))
    novelty.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))

    inspect = sub.add_parser("inspect-schedule")
    inspect.add_argument("--rows", type=int, default=320)
    inspect.add_argument("--seed", type=int, default=7)
    inspect.add_argument("--contains", default="cmdDisp.CMD_TEST_CMD_1")

    score = sub.add_parser("score-packets")
    score.add_argument("--packets", required=True)
    score.add_argument("--model-dir", default=str(DEFAULT_MODEL_DIR))
    score.add_argument("--output-dir", default=str(DEFAULT_OUTPUT_DIR))

    return parser


def main() -> None:
    args = build_parser().parse_args()
    output_dir = Path(getattr(args, "output_dir", DEFAULT_OUTPUT_DIR)).resolve()

    if args.cmd == "generate":
        dataset_path = run_generate(
            output_dir,
            args.rows,
            args.nominal_ratio,
            args.seed,
            protocol_mode=args.protocol_mode,
            mixed_fprime_ratio=args.mixed_fprime_ratio,
        )
        print(f"dataset={dataset_path}")
        return

    if args.cmd == "train":
        dataset_path = resolve_dataset_path(output_dir, args.dataset)
        report = run_training(
            dataset_path=dataset_path,
            output_dir=output_dir,
            seed=args.seed,
            make_plots=not args.no_plots,
            blue_feature_policy_name=args.blue_feature_policy,
            training_path_name=POSTER_DEFAULT_TRAINING_PATH_NAME,
        )
        print(json.dumps(report["metrics"], indent=2))
        return

    if args.cmd == "train-legacy":
        dataset_path = resolve_dataset_path(output_dir, args.dataset)
        assert_legacy_training_dataset_supported(dataset_path)
        report = run_training(
            dataset_path=dataset_path,
            output_dir=output_dir,
            seed=args.seed,
            make_plots=not args.no_plots,
            blue_feature_policy_name=args.blue_feature_policy,
            training_path_name=TRAINING_PATH_LEGACY_FPRIME_BASELINE,
        )
        print(json.dumps(report["metrics"], indent=2))
        return

    if args.cmd == "train-novelty":
        dataset_path = resolve_dataset_path(output_dir, args.dataset)
        run_train_novelty(dataset_path=dataset_path, output_path=Path(args.output).resolve())
        print(f"novelty={Path(args.output).resolve()}")
        return

    if args.cmd == "inspect-schedule":
        print(json.dumps(inspect_schedule(args.rows, args.seed, args.contains), indent=2))
        return

    if args.cmd == "score-packets":
        summary = score_packets(Path(args.packets).resolve(), output_dir=output_dir, model_dir=Path(args.model_dir).resolve())
        print(json.dumps(summary, indent=2))
        return

    if args.cmd == "run":
        dataset_path = resolve_dataset_path(output_dir, None)
        if not args.skip_generate:
            dataset_path = run_generate(
                output_dir,
                args.rows,
                args.nominal_ratio,
                args.seed,
                protocol_mode=args.protocol_mode,
                mixed_fprime_ratio=args.mixed_fprime_ratio,
            )
        elif not dataset_path.exists():
            raise SystemExit(f"Missing dataset for --skip-generate: {dataset_path}")
        report = run_training(
            dataset_path=dataset_path,
            output_dir=output_dir,
            seed=args.seed,
            make_plots=not args.no_plots,
            blue_feature_policy_name=args.blue_feature_policy,
            training_path_name=POSTER_DEFAULT_TRAINING_PATH_NAME,
        )
        print(json.dumps(report["metrics"], indent=2))
        return

    if args.cmd == "run-legacy":
        assert_legacy_generation_protocol_mode(args.protocol_mode)
        dataset_path = resolve_dataset_path(output_dir, None)
        if not args.skip_generate:
            dataset_path = run_generate(
                output_dir,
                args.rows,
                args.nominal_ratio,
                args.seed,
                protocol_mode=args.protocol_mode,
                mixed_fprime_ratio=args.mixed_fprime_ratio,
            )
        elif not dataset_path.exists():
            raise SystemExit(f"Missing dataset for --skip-generate: {dataset_path}")
        assert_legacy_training_dataset_supported(dataset_path)
        report = run_training(
            dataset_path=dataset_path,
            output_dir=output_dir,
            seed=args.seed,
            make_plots=not args.no_plots,
            blue_feature_policy_name=args.blue_feature_policy,
            training_path_name=TRAINING_PATH_LEGACY_FPRIME_BASELINE,
        )
        print(json.dumps(report["metrics"], indent=2))
        return


if __name__ == "__main__":
    main()
