from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import (
    CLASS_NAMES,
    COMMAND_ONLY_BASELINE_FEATURE_NAMES,
    DEFAULT_EVALUATION_SEED_COUNT,
    DEFAULT_GROUPED_CV_FOLDS,
    MODEL_ONLY_NAMESPACE,
    OUTCOME_ONLY_BASELINE_FEATURE_NAMES,
    PROTOCOL_ONLY_BASELINE_FEATURE_NAMES,
    PROTOCOL_ONLY_BASELINE_NEAR_PERFECT_THRESHOLD,
    RAW_PROTOCOL_SHORTCUT_BASELINE_FEATURE_NAMES,
    RAW_PROTOCOL_SHORTCUT_BASELINE_NEAR_PERFECT_THRESHOLD,
    REQUEST_ONLY_BASELINE_FEATURE_NAMES,
    SELECTION_METRIC_CLASS_MACRO_F1,
    SELECTION_METRIC_MODEL_CYBER_F1,
    SELECTION_METRIC_MIN_PER_CLASS_RECALL,
    SELECTION_METRIC_ORDER,
    SELECTION_METRIC_STACKED_ANOMALY_F1,
    TRAINING_PATH_LEGACY_FPRIME_BASELINE,
    EpisodeFactory,
    STACKED_DETECTOR_NAMESPACE,
    TRAINING_IMPORT_ERROR,
    audit_dataset_sanity,
    evaluate_outcome_only_baseline,
    evaluate_request_only_baseline,
    fit_and_evaluate_models,
    grouped_row_split,
    np as main_np,
    prepare_training_splits,
    ranking_key,
    ranking_payload,
    run_family_holdout_evaluation,
    run_training,
    score_packets,
    score_novelty_rows,
    write_jsonl,
)
from runtime import (
    FEATURE_NAMES,
    FORBIDDEN_PRIMARY_MODEL_FEATURE_NAMES,
    GaussianNovelty,
    HISTORY_FEATURE_NAMES,
    POSTER_BLUE_MODEL_ARTIFACT_NAME,
    POSTER_BLUE_RUNTIME_KIND,
    MODEL_FEATURE_LAYOUTS,
    NOVELTY_FEATURE_NAMES,
    NOVELTY_MODEL_FEATURE_TIER,
    PRIMARY_MODEL_FEATURE_NAMES,
    PRIMARY_MODEL_FEATURE_TIER,
    REQUEST_FEATURE_NAMES,
    RESPONSE_FEATURE_NAMES,
    RUNTIME_BUNDLE_MANIFEST_NAME,
    TERMINAL_OUTCOME_FEATURE_NAMES,
    history_state_reset_mode,
    load_runtime_bundle,
    replay_history_features,
    transactions_to_rows,
    vector_from_row,
)
from tools.shared.canonical_records import canonicalize_legacy_fprime_transaction
from tools.shared.feature_policy import BLUE_FEATURE_POLICY_POSTER_DEFAULT
from tools.train.blue_model import POSTER_BLUE_MODEL_FAMILY
from tools.train.poster_default import (
    POSTER_DEFAULT_COMMAND_BASELINE_FEATURE_NAMES,
    POSTER_DEFAULT_PRIMARY_MODEL_FEATURE_TIER,
    POSTER_DEFAULT_REQUEST_ONLY_BASELINE_FEATURE_NAMES,
    POSTER_DEFAULT_TRAINING_PATH_NAME,
)


def make_row(episode_id: int, episode_kind: str, label: int, index: int) -> dict[str, object]:
    return {
        "run_id": episode_id,
        "episode_id": episode_id,
        "episode_kind": episode_kind,
        "episode_label": label,
        "label": label,
        "command": "cmdDisp.CMD_NO_OP",
        "service": "cmdDisp",
        "attack_family": "none",
        "phase": "startup",
        "actor": "ops",
        "actor_role": "ops_primary",
        "actor_trust": 1.0,
        "txn_id": f"{episode_id}-{index}",
    }


def make_training_row(episode_id: int, episode_kind: str, label: int, index: int) -> dict[str, object]:
    row: dict[str, object] = {name: 0.0 for name in FEATURE_NAMES}
    request_ts_ms = float(episode_id * 120000 + index * 1700)
    row.update(
        {
            "run_id": episode_id,
            "episode_id": episode_id,
            "episode_kind": episode_kind,
            "episode_label": label,
            "label": label,
            "label_name": episode_kind,
            "command": "cmdDisp.CMD_NO_OP",
            "service": "cmdDisp",
            "attack_family": "none" if label == 0 else ("intrusion" if label == 1 else "ops_fault"),
            "phase": "startup",
            "actor": "ops",
            "actor_role": "ops_primary",
            "txn_id": f"{episode_id}-{index}",
            "send_id": f"send-{episode_id}-{index}",
            "session_id": f"ep-{episode_id}",
            "reason": "completed",
            "request_ts_ms": request_ts_ms,
        }
    )
    request_profiles = {
        0: {"service_id": 1.0, "command_id": 1.0, "target_node_id": 1.0, "actor_trust": 0.96, "arg_count": 0.0, "arg_norm": 0.05, "arg_out_of_range": 0.0, "req_bytes": 52.0},
        1: {"service_id": 9.0, "command_id": 17.0, "target_node_id": 2.0, "actor_trust": 0.18, "arg_count": 3.0, "arg_norm": 1.25, "arg_out_of_range": 1.0, "req_bytes": 96.0},
        2: {"service_id": 6.0, "command_id": 13.0, "target_node_id": 1.0, "actor_trust": 0.82, "arg_count": 2.0, "arg_norm": 0.55, "arg_out_of_range": 0.0, "req_bytes": 72.0},
    }
    response_profiles = {
        0: {"resp_bytes": 68.0, "latency_ms": 82.0, "gds_accept": 1.0, "sat_success": 1.0, "timeout": 0.0, "response_code": 0.0, "request_to_uplink_ms": 12.0, "uplink_to_sat_response_ms": 34.0, "sat_response_to_final_ms": 18.0, "response_direction_seen": 1.0, "final_observed_on_wire": 1.0, "txn_warning_events": 0.0, "txn_error_events": 0.0},
        1: {"resp_bytes": 56.0, "latency_ms": 420.0, "gds_accept": 1.0, "sat_success": 0.0, "timeout": 1.0, "response_code": 3.0, "request_to_uplink_ms": 20.0, "uplink_to_sat_response_ms": 0.0, "sat_response_to_final_ms": 400.0, "response_direction_seen": 0.0, "final_observed_on_wire": 0.0, "txn_warning_events": 2.0, "txn_error_events": 3.0},
        2: {"resp_bytes": 60.0, "latency_ms": 155.0, "gds_accept": 1.0, "sat_success": 0.0, "timeout": 0.0, "response_code": 2.0, "request_to_uplink_ms": 14.0, "uplink_to_sat_response_ms": 91.0, "sat_response_to_final_ms": 50.0, "response_direction_seen": 1.0, "final_observed_on_wire": 1.0, "txn_warning_events": 1.0, "txn_error_events": 1.0},
    }
    history_profiles = {
        0: {"packet_gap_ms": 1800.0, "command_rate_1m": 2.0, "error_rate_1m": 0.0},
        1: {"packet_gap_ms": 120.0, "command_rate_1m": 14.0, "error_rate_1m": 0.75},
        2: {"packet_gap_ms": 900.0, "command_rate_1m": 5.0, "error_rate_1m": 0.25},
    }
    telemetry_profiles = {
        0: {"target_cpu_total_pct": 24.0, "target_cpu_00_pct": 11.0, "target_cpu_01_pct": 13.0, "target_cmd_errors_1m": 0.0, "target_filemanager_errors_1m": 0.0, "target_filedownlink_warnings_1m": 0.0, "target_rg1_max_time_ms": 3.0, "target_rg2_max_time_ms": 3.5, "target_hibuffs_total": 0.0, "target_telemetry_age_ms": 180.0, "peer_cpu_total_pct": 21.0, "peer_cpu_00_pct": 10.0, "peer_cpu_01_pct": 11.0, "peer_telemetry_age_ms": 200.0},
        1: {"target_cpu_total_pct": 97.0, "target_cpu_00_pct": 93.0, "target_cpu_01_pct": 95.0, "target_cmd_errors_1m": 4.0, "target_filemanager_errors_1m": 3.0, "target_filedownlink_warnings_1m": 2.0, "target_rg1_max_time_ms": 16.0, "target_rg2_max_time_ms": 15.0, "target_hibuffs_total": 5.0, "target_telemetry_age_ms": 7200.0, "peer_cpu_total_pct": 76.0, "peer_cpu_00_pct": 38.0, "peer_cpu_01_pct": 38.0, "peer_telemetry_age_ms": 6500.0},
        2: {"target_cpu_total_pct": 61.0, "target_cpu_00_pct": 30.0, "target_cpu_01_pct": 31.0, "target_cmd_errors_1m": 1.0, "target_filemanager_errors_1m": 1.0, "target_filedownlink_warnings_1m": 0.0, "target_rg1_max_time_ms": 8.0, "target_rg2_max_time_ms": 8.5, "target_hibuffs_total": 1.0, "target_telemetry_age_ms": 1100.0, "peer_cpu_total_pct": 42.0, "peer_cpu_00_pct": 20.0, "peer_cpu_01_pct": 22.0, "peer_telemetry_age_ms": 950.0},
    }
    row.update(request_profiles[label])
    row.update(response_profiles[label])
    row.update(history_profiles[label])
    row.update(telemetry_profiles[label])
    row["actor_role"] = "external" if label == 1 else "ops_primary"
    row["actor"] = "red_team" if label == 1 else ("ops_fault" if label == 2 else "flight_primary")
    row["phase"] = "downlink" if label == 1 else ("standby" if label == 2 else "startup")
    row["target_stream_id"] = f"fprime_{'a' if int(row['target_node_id']) == 1 else 'b'}:50050"
    row["target_stream_index"] = index
    row["command_rate_1m"] = float(row["command_rate_1m"]) + (index % 3)
    row["packet_gap_ms"] = float(row["packet_gap_ms"]) + 17.0 * index
    row["arg_norm"] = float(row["arg_norm"]) + 0.01 * index
    row["req_bytes"] = float(row["req_bytes"]) + float(index % 4)
    row["target_cpu_total_pct"] = float(row["target_cpu_total_pct"]) + float(index % 3)
    row["peer_cpu_total_pct"] = float(row["peer_cpu_total_pct"]) + float(index % 2)
    row["final_ts_ms"] = request_ts_ms + float(row["latency_ms"])
    return row


def build_training_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    episode_id = 0
    for label, kind in ((0, "benign"), (1, "cyber"), (2, "fault")):
        for index in range(3):
            for step in range(12):
                rows.append(make_training_row(episode_id, kind, label, step))
            episode_id += 1
    return rows


def build_training_rows_with_benign_noise() -> list[dict[str, object]]:
    rows = build_training_rows()
    benign_rows_by_episode: dict[int, list[dict[str, object]]] = {}
    for row in rows:
        if int(row["label"]) != 0:
            continue
        benign_rows_by_episode.setdefault(int(row["episode_id"]), []).append(row)

    for episode_rows in benign_rows_by_episode.values():
        for index, row in enumerate(episode_rows[:4]):
            row["gds_accept"] = 1.0
            row["sat_success"] = 0.0
            row["timeout"] = 0.0
            row["response_code"] = 2.0
            row["txn_warning_events"] = 1.0
            row["txn_error_events"] = 1.0
            row["reason"] = "warning_event"
            row["latency_ms"] = 260.0 + 15.0 * index
    return rows


def build_family_holdout_regression_rows() -> list[dict[str, object]]:
    rows: list[dict[str, object]] = []
    family_profiles = {
        "routine_alpha": {
            "command": "cmdDisp.CMD_TEST_CMD_1",
            "service": "cmdDisp",
            "service_id": 1.0,
            "command_id": 10.0,
            "target_node_id": 1.0,
            "actor_trust": 0.96,
            "arg_count": 1.0,
            "arg_norm": 0.10,
            "arg_out_of_range": 0.0,
            "req_bytes": 56.0,
            "packet_gap_ms": 420.0,
            "command_rate_1m": 2.0,
            "error_rate_1m": 0.0,
            "target_cpu_total_pct": 28.0,
            "target_cpu_00_pct": 13.0,
            "target_cpu_01_pct": 15.0,
            "target_cmd_errors_1m": 0.0,
            "target_filemanager_errors_1m": 0.0,
            "target_filedownlink_warnings_1m": 0.0,
            "target_rg1_max_time_ms": 3.5,
            "target_rg2_max_time_ms": 4.0,
            "target_hibuffs_total": 0.0,
            "target_telemetry_age_ms": 180.0,
            "peer_cpu_total_pct": 18.0,
            "peer_cpu_00_pct": 9.0,
            "peer_cpu_01_pct": 9.0,
            "peer_telemetry_age_ms": 200.0,
        },
        "routine_beta": {
            "command": "fileManager.RemoveDirectory",
            "service": "fileManager",
            "service_id": 5.0,
            "command_id": 21.0,
            "target_node_id": 2.0,
            "actor_trust": 0.94,
            "arg_count": 2.0,
            "arg_norm": 0.16,
            "arg_out_of_range": 0.0,
            "req_bytes": 60.0,
            "packet_gap_ms": 390.0,
            "command_rate_1m": 3.0,
            "error_rate_1m": 0.0,
            "target_cpu_total_pct": 31.0,
            "target_cpu_00_pct": 15.0,
            "target_cpu_01_pct": 16.0,
            "target_cmd_errors_1m": 0.0,
            "target_filemanager_errors_1m": 0.0,
            "target_filedownlink_warnings_1m": 0.0,
            "target_rg1_max_time_ms": 3.8,
            "target_rg2_max_time_ms": 4.1,
            "target_hibuffs_total": 0.0,
            "target_telemetry_age_ms": 210.0,
            "peer_cpu_total_pct": 20.0,
            "peer_cpu_00_pct": 10.0,
            "peer_cpu_01_pct": 10.0,
            "peer_telemetry_age_ms": 220.0,
        },
        "intrusion_alpha": {
            "command": "fileDownlink.SendPartial",
            "service": "fileDownlink",
            "service_id": 7.0,
            "command_id": 32.0,
            "target_node_id": 2.0,
            "actor_trust": 0.18,
            "arg_count": 3.0,
            "arg_norm": 1.20,
            "arg_out_of_range": 1.0,
            "req_bytes": 92.0,
            "packet_gap_ms": 130.0,
            "command_rate_1m": 14.0,
            "error_rate_1m": 0.8,
            "target_cpu_total_pct": 94.0,
            "target_cpu_00_pct": 46.0,
            "target_cpu_01_pct": 48.0,
            "target_cmd_errors_1m": 3.0,
            "target_filemanager_errors_1m": 1.0,
            "target_filedownlink_warnings_1m": 2.0,
            "target_rg1_max_time_ms": 14.0,
            "target_rg2_max_time_ms": 15.0,
            "target_hibuffs_total": 4.0,
            "target_telemetry_age_ms": 6400.0,
            "peer_cpu_total_pct": 74.0,
            "peer_cpu_00_pct": 37.0,
            "peer_cpu_01_pct": 37.0,
            "peer_telemetry_age_ms": 6100.0,
        },
        "intrusion_beta": {},
        "ops_fault_alpha": {
            "command": "prmDb.PRM_SAVE_FILE",
            "service": "prmDb",
            "service_id": 4.0,
            "command_id": 27.0,
            "target_node_id": 1.0,
            "actor_trust": 0.80,
            "arg_count": 2.0,
            "arg_norm": 0.58,
            "arg_out_of_range": 0.0,
            "req_bytes": 68.0,
            "packet_gap_ms": 760.0,
            "command_rate_1m": 5.0,
            "error_rate_1m": 0.25,
            "target_cpu_total_pct": 58.0,
            "target_cpu_00_pct": 28.0,
            "target_cpu_01_pct": 30.0,
            "target_cmd_errors_1m": 1.0,
            "target_filemanager_errors_1m": 1.0,
            "target_filedownlink_warnings_1m": 0.0,
            "target_rg1_max_time_ms": 8.0,
            "target_rg2_max_time_ms": 8.6,
            "target_hibuffs_total": 1.0,
            "target_telemetry_age_ms": 1100.0,
            "peer_cpu_total_pct": 42.0,
            "peer_cpu_00_pct": 21.0,
            "peer_cpu_01_pct": 21.0,
            "peer_telemetry_age_ms": 960.0,
        },
        "ops_fault_beta": {},
    }
    family_profiles["intrusion_beta"] = dict(family_profiles["routine_alpha"])
    family_profiles["ops_fault_beta"] = dict(family_profiles["routine_beta"])
    class_families = {
        0: ["routine_alpha", "routine_beta"],
        1: ["intrusion_alpha", "intrusion_beta"],
        2: ["ops_fault_alpha", "ops_fault_beta"],
    }
    episode_id = 0
    for label, kind in ((0, "benign"), (1, "cyber"), (2, "fault")):
        for family_name in class_families[label]:
            profile = family_profiles[family_name]
            for copy_index in range(3):
                for step in range(8):
                    row = make_training_row(episode_id, kind, label, step)
                    row["attack_family"] = family_name
                    row.update({key: value for key, value in profile.items()})
                    row["actor_trust"] = float(row["actor_trust"]) + 0.005 * (step % 2)
                    row["arg_norm"] = float(row["arg_norm"]) + 0.005 * step
                    row["req_bytes"] = float(row["req_bytes"]) + float((step + copy_index) % 3)
                    row["packet_gap_ms"] = float(row["packet_gap_ms"]) + 3.0 * step
                    row["command_rate_1m"] = float(row["command_rate_1m"]) + float(step % 2)
                    row["phase"] = "science" if "beta" in family_name else "startup"
                    row["actor"] = f"{kind}_{family_name}"
                    row["actor_role"] = "external" if label == 1 else ("ops_fault" if label == 2 else "ops_primary")
                    rows.append(row)
                episode_id += 1
    return rows


def build_protocol_shortcut_row(
    run_id: int,
    label: int,
    episode_kind: str,
    step: int,
    *,
    protocol_family: str,
    raw_service_name: str,
    raw_command_name: str,
    phase: str,
) -> dict[str, object]:
    attack_family = "none" if label == 0 else ("intrusion" if label == 1 else "ops_fault")
    platform_family = "fprime_satellite" if protocol_family == "fprime" else "mavlink_vehicle"
    return {
        "run_id": run_id,
        "episode_id": run_id,
        "episode_kind": episode_kind,
        "episode_label": label,
        "label": label,
        "label_name": episode_kind,
        "command": "canonical.shared_command",
        "service": "canonical.shared_service",
        "attack_family": attack_family,
        "phase": phase,
        "actor": "red_team" if label == 1 else ("ops_fault" if label == 2 else "flight_primary"),
        "actor_role": "external" if label == 1 else "ops_primary",
        "actor_trust": 0.5,
        "txn_id": f"{run_id}-{step}",
        "request_ts_ms": float(run_id * 10000 + step * 125),
        "protocol_family": protocol_family,
        "platform_family": platform_family,
        "raw_service_name": raw_service_name,
        "raw_command_name": raw_command_name,
        "service_id": 1.0,
        "command_id": 1.0,
        "arg_count": 0.0,
        "arg_norm": 0.0,
        "arg_out_of_range": 0.0,
        "req_bytes": 64.0,
        "target_stream_id": f"{protocol_family}:14550",
        "args": {},
    }


def build_protocol_shortcut_fixture(
    protocol_sequences: dict[int, list[str]],
    raw_name_map: dict[int, tuple[str, str]],
    *,
    base_episode_count: int,
) -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    phase_pairs = [
        ("startup", "execute"),
        ("startup", "verify"),
        ("standby", "execute"),
        ("standby", "verify"),
        ("downlink", "execute"),
        ("downlink", "verify"),
    ]
    rows: list[dict[str, object]] = []
    base_rows: list[dict[str, object]] = []
    test_rows: list[dict[str, object]] = []
    run_id = 0
    for label, episode_kind in ((0, "benign"), (1, "cyber"), (2, "fault")):
        service_name, command_name = raw_name_map[label]
        for episode_index, protocol_family in enumerate(protocol_sequences[label]):
            phase_pair = phase_pairs[(label + episode_index) % len(phase_pairs)]
            episode_rows = [
                build_protocol_shortcut_row(
                    run_id,
                    label,
                    episode_kind,
                    step,
                    protocol_family=protocol_family,
                    raw_service_name=service_name,
                    raw_command_name=command_name,
                    phase=phase,
                )
                for step, phase in enumerate(phase_pair)
            ]
            rows.extend(episode_rows)
            if episode_index < base_episode_count:
                base_rows.extend(episode_rows)
            else:
                test_rows.extend(episode_rows)
            run_id += 1
    return rows, base_rows, test_rows


def build_protocol_only_leakage_fixture() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    return build_protocol_shortcut_fixture(
        {
            0: ["fprime", "fprime", "fprime"],
            1: ["mavlink", "mavlink", "mavlink"],
            2: ["mavlink", "mavlink", "mavlink"],
        },
        {
            0: ("shared_raw_service", "shared_raw_command"),
            1: ("shared_raw_service", "shared_raw_command"),
            2: ("shared_raw_service", "shared_raw_command"),
        },
        base_episode_count=2,
    )


def build_raw_protocol_shortcut_fixture() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    return build_protocol_shortcut_fixture(
        {
            0: ["fprime", "mavlink", "fprime", "mavlink"],
            1: ["fprime", "mavlink", "fprime", "mavlink"],
            2: ["fprime", "mavlink", "fprime", "mavlink"],
        },
        {
            0: ("ops_nominal", "cmd_safe"),
            1: ("ops_intrusion", "cmd_cyber"),
            2: ("ops_faults", "cmd_fault"),
        },
        base_episode_count=2,
    )


def build_protocol_holdout_records() -> list[dict[str, object]]:
    rows, _, _ = build_raw_protocol_shortcut_fixture()
    return rows


def make_training_transaction(episode_id: int, episode_kind: str, label: int, index: int) -> dict[str, object]:
    return training_transaction_from_row(make_training_row(episode_id, episode_kind, label, index))


def training_transaction_from_row(row: dict[str, object]) -> dict[str, object]:
    return {
        "run_id": int(row["run_id"]),
        "episode_id": int(row["episode_id"]),
        "episode_kind": str(row["episode_kind"]),
        "episode_label": int(row["episode_label"]),
        "label": int(row["label"]),
        "label_name": str(row["label_name"]),
        "session_id": str(row["session_id"]),
        "txn_id": str(row["txn_id"]),
        "send_id": str(row["send_id"]),
        "target_stream_id": str(row["target_stream_id"]),
        "target_stream_index": float(row["target_stream_index"]),
        "attack_family": str(row["attack_family"]),
        "phase": str(row["phase"]),
        "actor": str(row["actor"]),
        "actor_role": str(row["actor_role"]),
        "actor_trust": float(row["actor_trust"]),
        "command": str(row["command"]),
        "service": str(row["service"]),
        "reason": str(row["reason"]),
        "request_ts_ms": float(row["request_ts_ms"]),
        "final_ts_ms": float(row["final_ts_ms"]),
        "target_node_id": float(row["target_node_id"]),
        "req_bytes": float(row["req_bytes"]),
        "resp_bytes": float(row["resp_bytes"]),
        "latency_ms": float(row["latency_ms"]),
        "gds_accept": float(row["gds_accept"]),
        "sat_success": float(row["sat_success"]),
        "timeout": float(row["timeout"]),
        "response_code": float(row["response_code"]),
        "request_to_uplink_ms": float(row["request_to_uplink_ms"]),
        "uplink_to_sat_response_ms": float(row["uplink_to_sat_response_ms"]),
        "sat_response_to_final_ms": float(row["sat_response_to_final_ms"]),
        "response_direction_seen": float(row["response_direction_seen"]),
        "final_observed_on_wire": float(row["final_observed_on_wire"]),
        "txn_warning_events": float(row["txn_warning_events"]),
        "txn_error_events": float(row["txn_error_events"]),
        "args": {},
        **{name: float(row[name]) for name in FEATURE_NAMES if name.startswith(("target_", "peer_"))},
    }


def build_training_transactions() -> list[dict[str, object]]:
    transactions: list[dict[str, object]] = []
    episode_id = 0
    for label, kind in ((0, "benign"), (1, "cyber"), (2, "fault")):
        for _ in range(3):
            for step in range(12):
                transactions.append(make_training_transaction(episode_id, kind, label, step))
            episode_id += 1
    return transactions


def canonical_training_row_from_row(row: dict[str, object]) -> dict[str, object]:
    return canonicalize_legacy_fprime_transaction(
        training_transaction_from_row(row),
        recent_behavior={
            "command_rate_1m": row.get("command_rate_1m"),
            "error_rate_1m": row.get("error_rate_1m"),
            "repeat_command_count_10m": row.get("repeat_command_count_10m"),
            "same_target_command_rate_1m": row.get("same_target_command_rate_1m"),
        },
    )


def write_training_fixture(
    dataset_path: Path,
    rows: list[dict[str, object]],
    transactions: list[dict[str, object]] | None = None,
    canonical_rows: list[dict[str, object]] | None = None,
) -> None:
    write_jsonl(dataset_path, rows)
    resolved_transactions = transactions if transactions is not None else [training_transaction_from_row(row) for row in rows]
    write_jsonl(
        dataset_path.with_name("transactions.jsonl"),
        resolved_transactions,
    )
    write_jsonl(
        dataset_path.with_name("canonical_command_rows.jsonl"),
        canonical_rows if canonical_rows is not None else [canonical_training_row_from_row(row) for row in rows],
    )


def build_baseline_stub(feature_names: list[str], threshold: float) -> dict[str, object]:
    return {
        "feature_names": list(feature_names),
        "observed_tuple_count": 6,
        "class_metrics": {
            "accuracy": 0.5,
            "macro_precision": 0.5,
            "macro_recall": 0.5,
            "macro_f1": 0.5,
            "confusion_matrix": [[1, 0, 0], [0, 1, 0], [0, 0, 1]],
            "per_class": {
                "benign": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 1},
                "cyber": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 1},
                "fault": {"precision": 0.5, "recall": 0.5, "f1": 0.5, "support": 1},
            },
        },
        "cyber_binary_metrics": {
            "precision": 0.5,
            "recall": 0.5,
            "f1": 0.5,
            "auc_roc": 0.5,
            "auc_pr": 0.5,
            "threshold": 0.5,
            "tp": 1,
            "fp": 1,
            "tn": 1,
            "fn": 1,
        },
        "anomaly_binary_metrics": {
            "precision": 0.5,
            "recall": 0.5,
            "f1": 0.5,
            "auc_roc": 0.5,
            "auc_pr": 0.5,
            "threshold": 0.5,
            "tp": 1,
            "fp": 1,
            "tn": 1,
            "fn": 1,
        },
        "near_perfect_threshold": float(threshold),
        "near_perfect_metrics": [],
        "best_metric_path": "class_metrics.macro_f1",
        "best_metric_value": 0.5,
        "near_perfect": False,
        "not_applicable": False,
    }


def build_not_applicable_baseline_stub(feature_names: list[str], reason: str) -> dict[str, object]:
    return {
        "feature_names": list(feature_names),
        "near_perfect_threshold": 0.0,
        "near_perfect": False,
        "near_perfect_metrics": [],
        "best_metric_path": None,
        "best_metric_value": 0.0,
        "observed_tuple_count": 1,
        "not_applicable": True,
        "reason": reason,
    }


def build_passing_dataset_sanity_stub(
    *,
    command_only_feature_names: list[str] | None = None,
    request_only_feature_names: list[str] | None = None,
    protocol_only_feature_names: list[str] | None = None,
    raw_protocol_shortcut_feature_names: list[str] | None = None,
    request_tuple_purity_feature_names: list[str] | None = None,
    request_tuple_purity_bucket_sizes: dict[str, float] | None = None,
) -> dict[str, object]:
    resolved_command_only_feature_names = list(command_only_feature_names or COMMAND_ONLY_BASELINE_FEATURE_NAMES)
    resolved_request_only_feature_names = list(request_only_feature_names or REQUEST_ONLY_BASELINE_FEATURE_NAMES)
    resolved_protocol_only_feature_names = list(protocol_only_feature_names or PROTOCOL_ONLY_BASELINE_FEATURE_NAMES)
    resolved_raw_protocol_shortcut_feature_names = list(
        raw_protocol_shortcut_feature_names or RAW_PROTOCOL_SHORTCUT_BASELINE_FEATURE_NAMES
    )
    resolved_request_tuple_purity_feature_names = list(
        request_tuple_purity_feature_names or COMMAND_ONLY_BASELINE_FEATURE_NAMES
    )
    resolved_request_tuple_purity_bucket_sizes = dict(
        request_tuple_purity_bucket_sizes
        or {
            "service_id": 1.0,
            "command_id": 1.0,
            "arg_count": 1.0,
            "arg_norm": 0.1,
            "arg_out_of_range": 1.0,
        }
    )
    return {
        "version": 1,
        "group_key": "run_id",
        "rows": 0,
        "thresholds": {
            "command_only_near_perfect_threshold": 0.95,
            "request_only_near_perfect_threshold": 0.95,
            "protocol_only_near_perfect_threshold": PROTOCOL_ONLY_BASELINE_NEAR_PERFECT_THRESHOLD,
            "raw_protocol_shortcuts_near_perfect_threshold": RAW_PROTOCOL_SHORTCUT_BASELINE_NEAR_PERFECT_THRESHOLD,
        },
        "baseline_split": {"seed": 7, "base_rows": 0, "test_rows": 0},
        "command_overlap": {
            "summary": {
                "rows": 0,
                "class_rows": {"benign": 0, "cyber": 0, "fault": 0},
                "total_commands": 0,
                "commands_shared_by_at_least_two_classes": 0,
                "commands_shared_by_all_classes": 0,
                "exclusive_commands": 0,
                "overlap_ratio": 1.0,
                "shared_row_fraction": 1.0,
            },
            "commands": [],
        },
        "service_overlap": {
            "summary": {
                "rows": 0,
                "class_rows": {"benign": 0, "cyber": 0, "fault": 0},
                "total_values": 0,
                "values_shared_by_at_least_two_classes": 0,
                "values_shared_by_all_classes": 0,
                "exclusive_values": 0,
                "overlap_ratio": 1.0,
                "shared_row_fraction": 1.0,
            },
            "services": [],
        },
        "episode_signatures": {
            "summary": {
                "episodes": 0,
                "unique_signatures": 0,
                "unique_ratio": 1.0,
                "max_duplicate_group_count": 0,
                "duplicate_signatures": 0,
            },
            "per_class": {
                "benign": {"episodes": 0, "unique_signatures": 0, "unique_ratio": 1.0, "max_duplicate_group_count": 0},
                "cyber": {"episodes": 0, "unique_signatures": 0, "unique_ratio": 1.0, "max_duplicate_group_count": 0},
                "fault": {"episodes": 0, "unique_signatures": 0, "unique_ratio": 1.0, "max_duplicate_group_count": 0},
            },
            "diversity_check": {"policy": "stub", "passed": True, "violations": [], "per_class": {}},
        },
        "request_tuple_purity": {
            "feature_names": resolved_request_tuple_purity_feature_names,
            "bucket_sizes": resolved_request_tuple_purity_bucket_sizes,
            "summary": {
                "rows": 0,
                "unique_tuples": 0,
                "pure_tuples": 0,
                "shared_tuples": 0,
                "pure_tuple_ratio": 0.0,
                "pure_row_fraction": 0.0,
                "majority_row_fraction": 0.0,
            },
            "top_tuples": [],
        },
        "baselines": {
            "command_only": build_baseline_stub(resolved_command_only_feature_names, 0.95),
            "request_only": build_baseline_stub(resolved_request_only_feature_names, 0.95),
            "protocol_only": build_not_applicable_baseline_stub(
                resolved_protocol_only_feature_names,
                "configured_features_lack_variation",
            ),
            "raw_protocol_shortcuts": build_not_applicable_baseline_stub(
                resolved_raw_protocol_shortcut_feature_names,
                "configured_features_lack_variation",
            ),
        },
        "checks": [],
        "blocking_issues": [],
        "passed": True,
        "eligible_for_deployment": True,
    }


def build_poster_passing_dataset_sanity_stub() -> dict[str, object]:
    return build_passing_dataset_sanity_stub(
        command_only_feature_names=POSTER_DEFAULT_COMMAND_BASELINE_FEATURE_NAMES,
        request_only_feature_names=POSTER_DEFAULT_REQUEST_ONLY_BASELINE_FEATURE_NAMES,
        protocol_only_feature_names=PROTOCOL_ONLY_BASELINE_FEATURE_NAMES,
        raw_protocol_shortcut_feature_names=RAW_PROTOCOL_SHORTCUT_BASELINE_FEATURE_NAMES,
        request_tuple_purity_feature_names=POSTER_DEFAULT_COMMAND_BASELINE_FEATURE_NAMES,
        request_tuple_purity_bucket_sizes={
            "command_semantics.canonical_command_name": 1.0,
            "command_semantics.canonical_command_family": 1.0,
            "command_semantics.mutation_scope": 1.0,
            "command_semantics.authority_level": 1.0,
            "command_semantics.target_scope": 1.0,
            "argument_profile.argument_leaf_count": 1.0,
        },
    )


def build_score_packets_fixture() -> list[dict[str, object]]:
    return [
        {
            "ts_ms": 1000,
            "packet_kind": "telemetry",
            "src": "fprime_a",
            "dst": "gds",
            "service": "systemResources",
            "payload": {"cpu_total_pct": 11.0},
            "node_service": "fprime_a",
            "run_id": 0,
        },
        {
            "ts_ms": 1050,
            "packet_kind": "telemetry",
            "src": "fprime_a",
            "dst": "gds",
            "service": "cmdDisp",
            "payload": {"cmds_dispatched_total": 4.0},
            "node_service": "fprime_a",
            "run_id": 0,
        },
        {
            "ts_ms": 1100,
            "packet_kind": "telemetry",
            "src": "fprime_b",
            "dst": "gds",
            "service": "systemResources",
            "payload": {"cpu_total_pct": 7.0},
            "node_service": "fprime_b",
            "run_id": 0,
        },
        {
            "ts_ms": 60000,
            "packet_kind": "telemetry",
            "src": "fprime_a",
            "dst": "gds",
            "service": "cmdDisp",
            "payload": {"cmds_dispatched_total": 9.0},
            "node_service": "fprime_a",
            "run_id": 0,
        },
        {
            "ts_ms": 61000,
            "packet_kind": "request",
            "src": "ops_b1",
            "dst": "fprime_a",
            "target_service": "fprime_a",
            "service": "cmdDisp",
            "command": "cmdDisp.CMD_NO_OP",
            "label": 0,
            "episode_label": 0,
            "episode_kind": "benign",
            "episode_id": 0,
            "session_id": "ops_b1-0000",
            "txn_id": "0-000001-ops_b1",
            "send_id": "send-0000",
            "target_stream_id": "fprime_a:50050",
            "target_stream_index": 0,
            "attack_family": "none",
            "phase": "startup",
            "actor": "ops_b1",
            "actor_role": "ops_primary",
            "actor_trust": 0.97,
            "args": {},
            "bytes_on_wire": 40,
            "run_id": 0,
        },
        {
            "ts_ms": 61020,
            "packet_kind": "uplink",
            "src": "gds",
            "dst": "fprime_a",
            "target_service": "fprime_a",
            "service": "cmdDisp",
            "command": "cmdDisp.CMD_NO_OP",
            "label": 0,
            "episode_label": 0,
            "episode_kind": "benign",
            "episode_id": 0,
            "session_id": "ops_b1-0000",
            "txn_id": "0-000001-ops_b1",
            "send_id": "send-0000",
            "target_stream_id": "fprime_a:50050",
            "target_stream_index": 0,
            "run_id": 0,
        },
        {
            "ts_ms": 61045,
            "packet_kind": "sat_response",
            "src": "fprime_a",
            "dst": "gds",
            "target_service": "fprime_a",
            "service": "cmdDisp",
            "command": "cmdDisp.CMD_NO_OP",
            "label": 0,
            "episode_label": 0,
            "episode_kind": "benign",
            "episode_id": 0,
            "session_id": "ops_b1-0000",
            "txn_id": "0-000001-ops_b1",
            "send_id": "send-0000",
            "target_stream_id": "fprime_a:50050",
            "target_stream_index": 0,
            "sat_success": 1,
            "response_code": 0,
            "reason": "completed",
            "run_id": 0,
        },
        {
            "ts_ms": 61070,
            "packet_kind": "final",
            "src": "fprime_a",
            "dst": "ops_b1",
            "target_service": "fprime_a",
            "service": "cmdDisp",
            "command": "cmdDisp.CMD_NO_OP",
            "label": 0,
            "episode_label": 0,
            "episode_kind": "benign",
            "episode_id": 0,
            "session_id": "ops_b1-0000",
            "txn_id": "0-000001-ops_b1",
            "send_id": "send-0000",
            "target_stream_id": "fprime_a:50050",
            "target_stream_index": 0,
            "bytes_on_wire": 12,
            "gds_accept": 1,
            "sat_success": 1,
            "timeout": 0,
            "response_code": 0,
            "reason": "completed",
            "response_direction_seen": 1,
            "final_observed_on_wire": 1,
            "txn_warning_events": 0,
            "txn_error_events": 0,
            "run_id": 0,
        },
    ]


class TrainingHardeningTests(unittest.TestCase):
    def test_synthetic_episode_factory_uses_fprime_command_catalog_and_packet_shape(self) -> None:
        packets, transactions, rows, summary = EpisodeFactory(7).build_dataset(120, 0.55)

        self.assertEqual(len(transactions), len(rows))
        self.assertEqual(summary["rows"], len(rows))
        self.assertTrue(rows)
        self.assertTrue(all(float(row["target_node_id"]) in {1.0, 2.0} for row in rows))
        self.assertTrue(all(str(row["target_stream_id"]) for row in rows))
        self.assertTrue(any(float(row.get("target_cpu_total_pct", 0.0)) > 0.0 for row in rows))
        self.assertTrue(any(packet.get("packet_kind") == "telemetry" and packet.get("node_service") in {"fprime_a", "fprime_b"} for packet in packets))
        self.assertEqual(summary["history_featurization"]["group_key"], "episode_id")
        self.assertEqual(summary["history_featurization"]["state_reset"], "per_episode")
        self.assertIn("episode_signature_summary", summary)
        self.assertEqual(summary["episode_signature_summary"]["max_duplicate_group_count"], 1)
        self.assertTrue(
            all(
                str(row["command"]).startswith(("cmdDisp.", "cmdSeq.", "eventLogger.", "prmDb.", "systemResources.", "fileManager.", "fileDownlink."))
                for row in rows
            )
        )

    def test_grouped_split_allocates_each_class_to_each_split(self) -> None:
        rows = []
        episode_id = 0
        for label, kind in ((0, "benign"), (1, "cyber"), (2, "fault")):
            for _ in range(3):
                rows.append(make_row(episode_id, kind, label, episode_id))
                episode_id += 1

        base_rows, calib_rows, test_rows, summary = grouped_row_split(rows, seed=7)
        self.assertEqual(summary["base_class_episode_counts"], {"benign": 1, "cyber": 1, "fault": 1})
        self.assertEqual(summary["calibration_class_episode_counts"], {"benign": 1, "cyber": 1, "fault": 1})
        self.assertEqual(summary["test_class_episode_counts"], {"benign": 1, "cyber": 1, "fault": 1})
        self.assertEqual(len(base_rows), 3)
        self.assertEqual(len(calib_rows), 3)
        self.assertEqual(len(test_rows), 3)

    def test_grouped_split_fails_when_a_class_has_too_few_episodes(self) -> None:
        rows = []
        episode_id = 0
        for label, kind, episodes in ((0, "benign", 3), (1, "cyber", 3), (2, "fault", 2)):
            for _ in range(episodes):
                rows.append(make_row(episode_id, kind, label, episode_id))
                episode_id += 1

        with self.assertRaises(SystemExit) as exc:
            grouped_row_split(rows, seed=7)
        self.assertIn("Need at least 3 episodes per class", str(exc.exception))

    def test_grouped_split_scales_heldout_groups_when_class_support_is_large(self) -> None:
        rows = []
        episode_id = 0
        for label, kind, episodes in ((0, "benign", 30), (1, "cyber", 30), (2, "fault", 20)):
            for _ in range(episodes):
                rows.append(make_row(episode_id, kind, label, episode_id))
                episode_id += 1

        base_rows, calib_rows, test_rows, summary = grouped_row_split(rows, seed=7, group_key="run_id")

        self.assertEqual(summary["heldout_groups_per_class"], {"benign": 3, "cyber": 3, "fault": 3})
        self.assertEqual(summary["calibration_class_group_counts"], {"benign": 3, "cyber": 3, "fault": 3})
        self.assertEqual(summary["test_class_group_counts"], {"benign": 3, "cyber": 3, "fault": 3})
        self.assertEqual(summary["base_class_group_counts"], {"benign": 24, "cyber": 24, "fault": 14})
        self.assertEqual(len(calib_rows), 9)
        self.assertEqual(len(test_rows), 9)
        self.assertEqual(len(base_rows), 62)

    def test_prepare_training_splits_requires_transaction_replay_inputs(self) -> None:
        rows = build_training_rows()

        with TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "data" / "dataset.jsonl"
            write_jsonl(dataset_path, rows)

            with self.assertRaises(SystemExit) as exc:
                prepare_training_splits(dataset_path, seed=7)

        self.assertIn("Missing transactions.jsonl next to the dataset", str(exc.exception))

    def test_prepare_training_splits_requires_run_id_on_transactions(self) -> None:
        rows = build_training_rows()
        transactions = build_training_transactions()
        for tx in transactions:
            tx.pop("run_id", None)

        with TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "data" / "dataset.jsonl"
            write_training_fixture(dataset_path, rows, transactions)

            with self.assertRaises(SystemExit) as exc:
                prepare_training_splits(dataset_path, seed=7)

        self.assertIn("transactions.jsonl must include run_id on every row", str(exc.exception))

    def test_transactions_to_rows_resets_history_per_episode(self) -> None:
        leading = make_training_transaction(0, "benign", 0, 0)
        leading["request_ts_ms"] = 1000.0
        leading["final_ts_ms"] = 1060.0

        heldout_first = make_training_transaction(1, "cyber", 1, 0)
        heldout_first["request_ts_ms"] = 1100.0
        heldout_first["final_ts_ms"] = 1180.0

        heldout_second = make_training_transaction(1, "cyber", 1, 1)
        heldout_second["request_ts_ms"] = 2000.0
        heldout_second["final_ts_ms"] = 2080.0

        all_rows = transactions_to_rows([leading, heldout_first, heldout_second], reset_key="episode_id")
        heldout_rows = [row for row in all_rows if int(row["episode_id"]) == 1]
        isolated_rows = transactions_to_rows([heldout_first, heldout_second], reset_key="episode_id")

        self.assertEqual(
            [(row["packet_gap_ms"], row["command_rate_1m"], row["error_rate_1m"]) for row in heldout_rows],
            [(row["packet_gap_ms"], row["command_rate_1m"], row["error_rate_1m"]) for row in isolated_rows],
        )
        self.assertEqual(heldout_rows[0]["packet_gap_ms"], 0.0)
        self.assertEqual(heldout_rows[0]["command_rate_1m"], 1.0)
        self.assertEqual(heldout_rows[1]["error_rate_1m"], 1.0)

    def test_replay_history_features_resets_history_per_episode_for_row_fallback(self) -> None:
        leading = make_training_row(0, "benign", 0, 0)
        leading["request_ts_ms"] = 1000.0
        leading["final_ts_ms"] = 1060.0

        heldout_first = make_training_row(1, "cyber", 1, 0)
        heldout_first["request_ts_ms"] = 1100.0
        heldout_first["final_ts_ms"] = 1180.0

        heldout_second = make_training_row(1, "cyber", 1, 1)
        heldout_second["request_ts_ms"] = 2000.0
        heldout_second["final_ts_ms"] = 2080.0

        all_rows = replay_history_features([leading, heldout_first, heldout_second], reset_key="episode_id")
        heldout_rows = [row for row in all_rows if int(row["episode_id"]) == 1]
        isolated_rows = replay_history_features([heldout_first, heldout_second], reset_key="episode_id")

        self.assertEqual(
            [(row["packet_gap_ms"], row["command_rate_1m"], row["error_rate_1m"]) for row in heldout_rows],
            [(row["packet_gap_ms"], row["command_rate_1m"], row["error_rate_1m"]) for row in isolated_rows],
        )
        self.assertEqual(heldout_rows[0]["packet_gap_ms"], 0.0)
        self.assertEqual(heldout_rows[0]["command_rate_1m"], 1.0)
        self.assertEqual(heldout_rows[1]["error_rate_1m"], 1.0)

    def test_history_state_reset_mode_describes_episode_run_and_continuous_scope(self) -> None:
        self.assertEqual(history_state_reset_mode("episode_id"), "per_episode")
        self.assertEqual(history_state_reset_mode("run_id"), "per_run")
        self.assertEqual(history_state_reset_mode(None), "continuous")

    def test_prepare_training_splits_replays_history_from_transactions(self) -> None:
        rows = build_training_rows()
        for row in rows:
            row["packet_gap_ms"] = 999.0
            row["command_rate_1m"] = 999.0
            row["error_rate_1m"] = 999.0
        transactions = build_training_transactions()

        with TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "data" / "dataset.jsonl"
            write_training_fixture(dataset_path, rows, transactions)

            base_rows, calib_rows, test_rows, split_summary, history_featurization = prepare_training_splits(dataset_path, seed=7)

        all_split_rows = [*base_rows, *calib_rows, *test_rows]
        self.assertEqual(history_featurization["source"], "transactions")
        self.assertEqual(history_featurization["group_key"], "run_id")
        self.assertEqual(history_featurization["state_reset"], "per_run")
        self.assertEqual(split_summary["history_featurization"]["state_reset"], "per_run")
        self.assertTrue(all(float(row["command_rate_1m"]) < 999.0 for row in all_split_rows))
        self.assertTrue(all(float(row["error_rate_1m"]) < 999.0 for row in all_split_rows))

    def test_prepare_training_splits_groups_on_run_id_not_episode_id(self) -> None:
        rows = build_training_rows()
        transactions = build_training_transactions()
        for row in rows:
            original_episode = int(row["episode_id"])
            row["run_id"] = 100 + original_episode
            row["episode_id"] = int(row["label"])
        for tx in transactions:
            original_episode = int(tx["episode_id"])
            tx["run_id"] = 100 + original_episode
            tx["episode_id"] = int(tx["label"])

        with TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "data" / "dataset.jsonl"
            write_training_fixture(dataset_path, rows, transactions)

            base_rows, calib_rows, test_rows, split_summary, history_featurization = prepare_training_splits(dataset_path, seed=7)

        self.assertEqual(history_featurization["group_key"], "run_id")
        self.assertEqual(history_featurization["state_reset"], "per_run")
        self.assertEqual(split_summary["group_key"], "run_id")
        self.assertEqual(split_summary["base_class_group_counts"], {"benign": 1, "cyber": 1, "fault": 1})
        self.assertEqual(split_summary["calibration_class_group_counts"], {"benign": 1, "cyber": 1, "fault": 1})
        self.assertEqual(split_summary["test_class_group_counts"], {"benign": 1, "cyber": 1, "fault": 1})
        self.assertTrue(all("run_id" in row for row in [*base_rows, *calib_rows, *test_rows]))

    def test_ranking_prefers_macro_f1_over_cyber_only_strength(self) -> None:
        low_macro = {
            MODEL_ONLY_NAMESPACE: {
                "multiclass_metrics": {
                    "macro_f1": 0.55,
                    "per_class": {
                        "benign": {"recall": 1.0, "support": 4},
                        "cyber": {"recall": 0.9, "support": 4},
                        "fault": {"recall": 0.2, "support": 4},
                    },
                },
                "cyber_binary_metrics": {"f1": 0.98},
            },
            STACKED_DETECTOR_NAMESPACE: {"anomaly_binary_metrics": {"f1": 0.95}},
        }
        high_macro = {
            MODEL_ONLY_NAMESPACE: {
                "multiclass_metrics": {
                    "macro_f1": 0.82,
                    "per_class": {
                        "benign": {"recall": 0.8, "support": 4},
                        "cyber": {"recall": 0.78, "support": 4},
                        "fault": {"recall": 0.76, "support": 4},
                    },
                },
                "cyber_binary_metrics": {"f1": 0.70},
            },
            STACKED_DETECTOR_NAMESPACE: {"anomaly_binary_metrics": {"f1": 0.80}},
        }

        low_payload = ranking_payload("low_macro", low_macro, 500)
        high_payload = ranking_payload("high_macro", high_macro, 900)

        ordered = sorted([low_payload, high_payload], key=ranking_key)
        self.assertEqual(ordered[0]["name"], "high_macro")
        self.assertEqual(
            set(low_payload["selection_metrics"]),
            {
                SELECTION_METRIC_CLASS_MACRO_F1,
                SELECTION_METRIC_MIN_PER_CLASS_RECALL,
                SELECTION_METRIC_STACKED_ANOMALY_F1,
                SELECTION_METRIC_MODEL_CYBER_F1,
            },
        )

    def test_primary_model_feature_tier_excludes_forbidden_terminal_fields(self) -> None:
        self.assertEqual(set(PRIMARY_MODEL_FEATURE_NAMES), set(REQUEST_FEATURE_NAMES))
        self.assertEqual(set(RESPONSE_FEATURE_NAMES), set(TERMINAL_OUTCOME_FEATURE_NAMES))
        self.assertEqual(set(FORBIDDEN_PRIMARY_MODEL_FEATURE_NAMES), set(RESPONSE_FEATURE_NAMES) | set(HISTORY_FEATURE_NAMES))
        self.assertEqual(set(NOVELTY_FEATURE_NAMES), set(REQUEST_FEATURE_NAMES) | set(HISTORY_FEATURE_NAMES))
        self.assertFalse(set(REQUEST_FEATURE_NAMES) & set(RESPONSE_FEATURE_NAMES))
        self.assertFalse(set(REQUEST_FEATURE_NAMES) & set(HISTORY_FEATURE_NAMES))
        self.assertFalse(set(PRIMARY_MODEL_FEATURE_NAMES) & FORBIDDEN_PRIMARY_MODEL_FEATURE_NAMES)

    @unittest.skipIf(TRAINING_IMPORT_ERROR is not None, f"training dependencies unavailable: {TRAINING_IMPORT_ERROR}")
    def test_novelty_scoring_is_label_blind_with_labeled_rows_present(self) -> None:
        nominal_rows = [make_training_row(0, "benign", 0, step) for step in range(3)]
        novelty = GaussianNovelty.fit(
            main_np.array([vector_from_row(row, NOVELTY_FEATURE_NAMES) for row in nominal_rows], dtype=float),
            list(NOVELTY_FEATURE_NAMES),
        )
        probe = make_training_row(42, "cyber", 1, 0)
        unlabeled_probe = dict(probe)
        unlabeled_probe.pop("label")
        benign_labeled_probe = dict(probe)
        benign_labeled_probe["label"] = 0
        fault_labeled_probe = dict(probe)
        fault_labeled_probe["label"] = 2

        scores = score_novelty_rows(novelty, [unlabeled_probe, benign_labeled_probe, fault_labeled_probe])

        self.assertEqual(len(scores), 3)
        self.assertAlmostEqual(float(scores[0]), float(scores[1]))
        self.assertAlmostEqual(float(scores[1]), float(scores[2]))

    @unittest.skipIf(TRAINING_IMPORT_ERROR is not None, f"training dependencies unavailable: {TRAINING_IMPORT_ERROR}")
    def test_outcome_only_baseline_still_flags_terminal_shortcuts_with_benign_nuisance_variance(self) -> None:
        rows = build_training_rows_with_benign_noise()
        base_rows, _, test_rows, _ = grouped_row_split(rows, seed=7)

        baseline = evaluate_outcome_only_baseline(base_rows, test_rows, seed=7)

        self.assertEqual(baseline["feature_names"], list(OUTCOME_ONLY_BASELINE_FEATURE_NAMES))
        self.assertTrue(baseline["near_perfect"])
        self.assertEqual(baseline["best_metric_path"], "cyber_binary_metrics.f1")

    @unittest.skipIf(TRAINING_IMPORT_ERROR is not None, f"training dependencies unavailable: {TRAINING_IMPORT_ERROR}")
    def test_request_only_baseline_flags_trivial_request_time_fixture(self) -> None:
        rows = build_training_rows()
        base_rows, _, test_rows, _ = grouped_row_split(rows, seed=7)

        baseline = evaluate_request_only_baseline(base_rows, test_rows, seed=7)

        self.assertEqual(baseline["feature_names"], list(REQUEST_ONLY_BASELINE_FEATURE_NAMES))
        self.assertTrue(baseline["near_perfect"])

    @unittest.skipIf(TRAINING_IMPORT_ERROR is not None, f"training dependencies unavailable: {TRAINING_IMPORT_ERROR}")
    def test_dataset_sanity_flags_protocol_only_leakage_using_binary_metrics(self) -> None:
        rows, base_rows, test_rows = build_protocol_only_leakage_fixture()

        report = audit_dataset_sanity(
            rows,
            base_rows,
            test_rows,
            seed=7,
            group_key="run_id",
            protocol_only_feature_names=PROTOCOL_ONLY_BASELINE_FEATURE_NAMES,
            raw_protocol_shortcut_feature_names=RAW_PROTOCOL_SHORTCUT_BASELINE_FEATURE_NAMES,
        )

        self.assertTrue(report["baselines"]["protocol_only"]["near_perfect"])
        self.assertEqual(report["baselines"]["protocol_only"]["best_metric_path"], "anomaly_binary_metrics.f1")
        blocking_reasons = {issue["reason"] for issue in report["blocking_issues"]}
        self.assertIn("protocol_only_baseline_near_perfect", blocking_reasons)

    @unittest.skipIf(TRAINING_IMPORT_ERROR is not None, f"training dependencies unavailable: {TRAINING_IMPORT_ERROR}")
    def test_dataset_sanity_flags_raw_protocol_shortcuts_when_protocol_alone_is_not_predictive(self) -> None:
        rows, base_rows, test_rows = build_raw_protocol_shortcut_fixture()

        report = audit_dataset_sanity(
            rows,
            base_rows,
            test_rows,
            seed=7,
            group_key="run_id",
            protocol_only_feature_names=PROTOCOL_ONLY_BASELINE_FEATURE_NAMES,
            raw_protocol_shortcut_feature_names=RAW_PROTOCOL_SHORTCUT_BASELINE_FEATURE_NAMES,
        )

        self.assertFalse(report["baselines"]["protocol_only"]["near_perfect"])
        self.assertTrue(report["baselines"]["raw_protocol_shortcuts"]["near_perfect"])
        blocking_reasons = {issue["reason"] for issue in report["blocking_issues"]}
        self.assertIn("raw_protocol_shortcuts_baseline_near_perfect", blocking_reasons)

    @unittest.skipIf(TRAINING_IMPORT_ERROR is not None, f"training dependencies unavailable: {TRAINING_IMPORT_ERROR}")
    def test_protocol_family_holdout_evaluation_runs_on_mixed_protocol_records(self) -> None:
        records = build_protocol_holdout_records()

        def stub_fit_and_evaluate(
            base_rows: list[dict[str, object]],
            calib_rows: list[dict[str, object]],
            test_rows: list[dict[str, object]],
            *,
            seed: int,
        ) -> dict[str, object]:
            del base_rows, calib_rows, test_rows, seed
            return {
                "thresholds": {
                    "neural_net": {
                        MODEL_ONLY_NAMESPACE: {
                            "cyber": 0.5,
                            "anomaly": 0.5,
                        }
                    }
                },
                "metrics": {
                    "neural_net": {
                        MODEL_ONLY_NAMESPACE: {
                            "multiclass_metrics": {
                                "accuracy": 0.7,
                                "macro_precision": 0.7,
                                "macro_recall": 0.7,
                                "macro_f1": 0.7,
                                "confusion_matrix": [[2, 0, 0], [0, 2, 0], [0, 0, 2]],
                                "per_class": {
                                    "benign": {"precision": 0.7, "recall": 0.7, "f1": 0.7, "support": 2},
                                    "cyber": {"precision": 0.7, "recall": 0.7, "f1": 0.7, "support": 2},
                                    "fault": {"precision": 0.7, "recall": 0.7, "f1": 0.7, "support": 2},
                                },
                            },
                            "cyber_binary_metrics": {
                                "precision": 0.72,
                                "recall": 0.68,
                                "f1": 0.70,
                                "auc_roc": 0.75,
                                "auc_pr": 0.74,
                                "threshold": 0.5,
                                "tp": 2,
                                "fp": 1,
                                "tn": 3,
                                "fn": 1,
                            },
                            "anomaly_binary_metrics": {
                                "precision": 0.74,
                                "recall": 0.70,
                                "f1": 0.72,
                                "auc_roc": 0.77,
                                "auc_pr": 0.76,
                                "threshold": 0.5,
                                "tp": 4,
                                "fp": 1,
                                "tn": 2,
                                "fn": 1,
                            },
                        }
                    }
                },
            }

        report = run_family_holdout_evaluation(
            records,
            "run_id",
            [7],
            family_field="protocol_family",
            evaluation_name="protocol_family_holdout",
            fit_and_evaluate_fn=stub_fit_and_evaluate,
            materialize_split_rows_fn=lambda base_records, calib_records, test_records, group_key: (
                base_records,
                calib_records,
                test_records,
            ),
            protocol_only_feature_names=PROTOCOL_ONLY_BASELINE_FEATURE_NAMES,
            raw_protocol_shortcut_feature_names=RAW_PROTOCOL_SHORTCUT_BASELINE_FEATURE_NAMES,
        )

        self.assertTrue(report["feasible"])
        self.assertEqual(report["evaluated_values"], ["fprime", "mavlink"])
        self.assertEqual(report["total_runs"], 2)
        self.assertIn("protocol_only_baseline", report["aggregate"])
        self.assertIn("raw_protocol_shortcuts_baseline", report["aggregate"])
        self.assertIn(
            MODEL_ONLY_NAMESPACE,
            report["aggregate"]["models"]["neural_net"],
        )

    @unittest.skipIf(TRAINING_IMPORT_ERROR is not None, f"training dependencies unavailable: {TRAINING_IMPORT_ERROR}")
    def test_fit_and_evaluate_models_keeps_model_only_and_stacked_metrics_separate(self) -> None:
        rows = build_training_rows()
        base_rows, calib_rows, test_rows, _ = grouped_row_split(rows, seed=7)

        result = fit_and_evaluate_models(
            base_rows,
            calib_rows,
            test_rows,
            seed=7,
            include_artifacts=False,
            include_curves=False,
        )

        expected_model_only_keys = {
            "feature_tier",
            "feature_names",
            "multiclass_metrics",
            "cyber_binary_metrics",
            "anomaly_binary_metrics",
            "score_means",
        }
        expected_stacked_detector_keys = {
            "input_feature_tier",
            "input_feature_names",
            "components",
            "calibrator",
            "anomaly_binary_metrics",
            "component_score_means",
            "lift_vs_model_only",
        }
        expected_model_only_score_means = {"mean_pcyber", "mean_panomaly"}
        expected_stacked_component_means = {
            "mean_model_panomaly",
            "mean_model_pcyber",
            "mean_rule_score",
            "mean_novelty_score",
            "mean_risk",
        }

        for model_name, payload in result["metrics"].items():
            self.assertEqual(
                set(payload.keys()),
                {MODEL_ONLY_NAMESPACE, STACKED_DETECTOR_NAMESPACE},
                f"{model_name} must expose separate model_only and stacked_detector metric namespaces",
            )
            self.assertEqual(
                set(payload[MODEL_ONLY_NAMESPACE].keys()),
                expected_model_only_keys,
                f"{model_name} model_only metrics should only contain raw model outputs",
            )
            self.assertEqual(
                set(payload[STACKED_DETECTOR_NAMESPACE].keys()),
                expected_stacked_detector_keys,
                f"{model_name} stacked_detector metrics should only contain stacked anomaly outputs",
            )
            self.assertNotIn(
                "calibrator",
                payload[MODEL_ONLY_NAMESPACE],
                f"{model_name} model_only metrics must stay free of calibrator state",
            )
            self.assertNotIn(
                "multiclass_metrics",
                payload[STACKED_DETECTOR_NAMESPACE],
                f"{model_name} stacked_detector metrics must not present multiclass classifier scores",
            )
            self.assertNotIn(
                "cyber_binary_metrics",
                payload[STACKED_DETECTOR_NAMESPACE],
                f"{model_name} stacked_detector metrics must not present raw cyber-vs-non-cyber scores",
            )
            self.assertEqual(
                payload[STACKED_DETECTOR_NAMESPACE]["components"],
                ["model_panomaly", "model_pcyber", "rules", "novelty", "calibrator"],
                f"{model_name} stacked_detector metrics must report the full stack composition",
            )
            for class_name, score_means in payload[MODEL_ONLY_NAMESPACE]["score_means"].items():
                self.assertEqual(
                    set(score_means.keys()),
                    expected_model_only_score_means,
                    f"{model_name} model_only score means for {class_name} must stay limited to raw model outputs",
                )
            for class_name, component_means in payload[STACKED_DETECTOR_NAMESPACE]["component_score_means"].items():
                self.assertEqual(
                    set(component_means.keys()),
                    expected_stacked_component_means,
                    f"{model_name} stacked_detector component means for {class_name} must expose the stacked inputs explicitly",
                )
            self.assertEqual(
                set(result["thresholds"][model_name][MODEL_ONLY_NAMESPACE].keys()),
                {"cyber", "anomaly"},
                f"{model_name} model_only thresholds must keep cyber and anomaly decisions separate",
            )
            self.assertEqual(
                set(result["thresholds"][model_name][STACKED_DETECTOR_NAMESPACE].keys()),
                {"anomaly"},
                f"{model_name} stacked_detector thresholds must only cover the stacked anomaly decision",
            )

    @unittest.skipIf(TRAINING_IMPORT_ERROR is not None, f"training dependencies unavailable: {TRAINING_IMPORT_ERROR}")
    def test_training_report_records_feature_tiers_by_model(self) -> None:
        rows = build_training_rows()
        with TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "data" / "dataset.jsonl"
            output_dir = Path(tmpdir) / "artifacts"
            write_training_fixture(dataset_path, rows)
            with mock.patch("main.audit_dataset_sanity", autospec=True, return_value=build_passing_dataset_sanity_stub()):
                with mock.patch("main.export_runtime_files", autospec=True):
                    report = run_training(
                        dataset_path,
                        output_dir,
                        seed=7,
                        make_plots=False,
                        training_path_name=TRAINING_PATH_LEGACY_FPRIME_BASELINE,
                    )

            self.assertEqual(report["feature_sets"]["tiers"]["request"], list(REQUEST_FEATURE_NAMES))
            self.assertEqual(report["training_path"]["name"], TRAINING_PATH_LEGACY_FPRIME_BASELINE)
            self.assertTrue(report["training_path"]["comparison_only"])
            self.assertTrue(report["comparison_only"])
            self.assertEqual(report["feature_sets"]["models"]["random_forest"]["feature_tier"], PRIMARY_MODEL_FEATURE_TIER)
            self.assertEqual(report["feature_sets"]["models"]["random_forest"]["feature_names"], list(PRIMARY_MODEL_FEATURE_NAMES))
            self.assertEqual(report["feature_sets"]["models"]["neural_net"]["feature_tier"], PRIMARY_MODEL_FEATURE_TIER)
            self.assertEqual(report["feature_sets"]["baselines"]["request_only"]["feature_names"], list(REQUEST_ONLY_BASELINE_FEATURE_NAMES))
            self.assertEqual(report["feature_sets"]["baselines"]["protocol_only"]["feature_names"], list(PROTOCOL_ONLY_BASELINE_FEATURE_NAMES))
            self.assertEqual(
                report["feature_sets"]["baselines"]["raw_protocol_shortcuts"]["feature_names"],
                list(RAW_PROTOCOL_SHORTCUT_BASELINE_FEATURE_NAMES),
            )
            self.assertEqual(report["feature_sets"]["models"]["novelty"]["feature_names"], report["feature_sets"]["tiers"]["novelty_model"])
            self.assertEqual(report["history_featurization"]["group_key"], "run_id")
            self.assertEqual(report["history_featurization"]["state_reset"], "per_run")
            self.assertEqual(report["novelty_adaptation"]["mode"], "disabled")
            self.assertTrue(report["novelty_adaptation"]["label_blind"])
            self.assertFalse(report["novelty_adaptation"]["uses_ground_truth_labels"])
            self.assertNotIn("requested_forget_alpha", report["novelty_adaptation"])
            self.assertNotIn("effective_forget_alpha", report["novelty_adaptation"])
            self.assertFalse(report["split_episode_signatures"]["enforced"])
            self.assertTrue(report["dataset_sanity"]["passed"])
            self.assertEqual(report["metrics"][MODEL_ONLY_NAMESPACE]["random_forest"]["feature_tier"], PRIMARY_MODEL_FEATURE_TIER)
            self.assertEqual(report["metrics"][MODEL_ONLY_NAMESPACE]["neural_net"]["feature_tier"], PRIMARY_MODEL_FEATURE_TIER)
            self.assertEqual(
                report["metrics"][STACKED_DETECTOR_NAMESPACE]["random_forest"]["calibrator"]["feature_names"],
                ["panomaly", "pcyber", "rules", "novelty"],
            )
            self.assertIn("lift_vs_model_only", report["metrics"][STACKED_DETECTOR_NAMESPACE]["neural_net"])
            self.assertEqual(report["simple_command_baseline"]["feature_names"], list(COMMAND_ONLY_BASELINE_FEATURE_NAMES))
            self.assertEqual(report["request_only_baseline"]["feature_names"], list(REQUEST_ONLY_BASELINE_FEATURE_NAMES))
            self.assertEqual(report["outcome_only_baseline"]["feature_names"], list(OUTCOME_ONLY_BASELINE_FEATURE_NAMES))
            self.assertEqual(report["protocol_only_baseline"]["feature_names"], list(PROTOCOL_ONLY_BASELINE_FEATURE_NAMES))
            self.assertTrue(report["protocol_only_baseline"]["not_applicable"])
            self.assertEqual(
                report["raw_protocol_shortcuts_baseline"]["feature_names"],
                list(RAW_PROTOCOL_SHORTCUT_BASELINE_FEATURE_NAMES),
            )
            self.assertTrue(report["raw_protocol_shortcuts_baseline"]["not_applicable"])
            self.assertEqual(report["evaluation"]["seeds"], [7, 8, 9])
            self.assertEqual(report["evaluation"]["repeated_grouped_cv"]["fold_count"], DEFAULT_GROUPED_CV_FOLDS)
            self.assertEqual(
                report["evaluation"]["repeated_grouped_cv"]["total_runs"],
                DEFAULT_EVALUATION_SEED_COUNT * DEFAULT_GROUPED_CV_FOLDS,
            )
            expected_class_group_counts = {
                name: int(report["split_summary"]["base_class_group_counts"][name])
                + int(report["split_summary"]["calibration_class_group_counts"][name])
                + int(report["split_summary"]["test_class_group_counts"][name])
                for name in CLASS_NAMES
            }
            self.assertEqual(
                report["evaluation"]["repeated_grouped_cv"]["class_group_counts"],
                expected_class_group_counts,
            )
            support_summary = report["evaluation"]["repeated_grouped_cv"]["support_summary"]
            self.assertEqual(
                support_summary["run_count"],
                DEFAULT_EVALUATION_SEED_COUNT * DEFAULT_GROUPED_CV_FOLDS,
            )
            self.assertEqual(
                len(support_summary["per_run_support"]),
                DEFAULT_EVALUATION_SEED_COUNT * DEFAULT_GROUPED_CV_FOLDS,
            )
            self.assertEqual(
                support_summary["test_class_group_counts"]["min"],
                {name: 1 for name in CLASS_NAMES},
            )
            self.assertFalse(report["evaluation"]["protocol_family_holdout"]["feasible"])
            self.assertIn(
                "mean",
                report["evaluation"]["repeated_grouped_cv"]["aggregate"]["models"]["random_forest"][MODEL_ONLY_NAMESPACE]["multiclass_metrics"]["macro_f1"],
            )
            self.assertIn(
                "std",
                report["evaluation"]["repeated_grouped_cv"]["aggregate"]["models"]["random_forest"][MODEL_ONLY_NAMESPACE]["multiclass_metrics"]["macro_f1"],
            )
            self.assertIn(
                MODEL_ONLY_NAMESPACE,
                report["evaluation"]["repeated_grouped_cv"]["aggregate"]["metric_namespaces"],
            )
            self.assertEqual(report["ranking_source"], "evaluation.repeated_grouped_cv.aggregate.models")
            self.assertEqual(report["ranking_metric_order"], list(SELECTION_METRIC_ORDER))
            self.assertIn("selection_basis", report["deployment_winner"])
            self.assertEqual(report["deployment_winner"]["selection_metric_order"], list(SELECTION_METRIC_ORDER))
            self.assertIn(SELECTION_METRIC_STACKED_ANOMALY_F1, report["deployment_winner"]["selection_metrics"])

            metrics_path = output_dir / "reports" / "metrics.json"
            self.assertTrue(metrics_path.exists())
            saved_report = json.loads(metrics_path.read_text(encoding="utf-8"))
            summary_text = (output_dir / "reports" / "summary.txt").read_text(encoding="utf-8")
            self.assertEqual(saved_report["history_featurization"]["state_reset"], "per_run")
            self.assertEqual(saved_report["training_path"]["name"], TRAINING_PATH_LEGACY_FPRIME_BASELINE)
            self.assertTrue(saved_report["training_path"]["comparison_only"])
            self.assertTrue(saved_report["comparison_only"])
            self.assertFalse(saved_report["split_episode_signatures"]["enforced"])
            self.assertEqual(saved_report["feature_sets"]["models"]["random_forest"]["feature_tier"], PRIMARY_MODEL_FEATURE_TIER)
            self.assertEqual(saved_report["novelty_adaptation"]["mode"], "disabled")
            self.assertTrue(saved_report["novelty_adaptation"]["label_blind"])
            self.assertNotIn("requested_forget_alpha", saved_report["novelty_adaptation"])
            self.assertNotIn("effective_forget_alpha", saved_report["novelty_adaptation"])
            self.assertTrue(saved_report["dataset_sanity"]["passed"])
            self.assertEqual(saved_report["metrics"][MODEL_ONLY_NAMESPACE]["random_forest"]["feature_names"], list(PRIMARY_MODEL_FEATURE_NAMES))
            self.assertIn(STACKED_DETECTOR_NAMESPACE, saved_report["metrics"])
            self.assertEqual(
                set(saved_report["thresholds"][MODEL_ONLY_NAMESPACE]["random_forest"].keys()),
                {"cyber", "anomaly"},
            )
            self.assertEqual(
                set(saved_report["thresholds"][STACKED_DETECTOR_NAMESPACE]["random_forest"].keys()),
                {"anomaly"},
            )
            self.assertEqual(saved_report["feature_sets"]["baselines"]["command_only"]["feature_names"], list(COMMAND_ONLY_BASELINE_FEATURE_NAMES))
            self.assertEqual(saved_report["feature_sets"]["baselines"]["request_only"]["feature_names"], list(REQUEST_ONLY_BASELINE_FEATURE_NAMES))
            self.assertEqual(saved_report["feature_sets"]["baselines"]["outcome_only"]["feature_names"], list(OUTCOME_ONLY_BASELINE_FEATURE_NAMES))
            self.assertEqual(
                saved_report["feature_sets"]["baselines"]["protocol_only"]["feature_names"],
                list(PROTOCOL_ONLY_BASELINE_FEATURE_NAMES),
            )
            self.assertEqual(
                saved_report["feature_sets"]["baselines"]["raw_protocol_shortcuts"]["feature_names"],
                list(RAW_PROTOCOL_SHORTCUT_BASELINE_FEATURE_NAMES),
            )
            self.assertEqual(saved_report["simple_command_baseline"]["feature_names"], list(COMMAND_ONLY_BASELINE_FEATURE_NAMES))
            self.assertEqual(saved_report["request_only_baseline"]["feature_names"], list(REQUEST_ONLY_BASELINE_FEATURE_NAMES))
            self.assertEqual(saved_report["outcome_only_baseline"]["feature_names"], list(OUTCOME_ONLY_BASELINE_FEATURE_NAMES))
            self.assertTrue(saved_report["protocol_only_baseline"]["not_applicable"])
            self.assertTrue(saved_report["raw_protocol_shortcuts_baseline"]["not_applicable"])
            self.assertEqual(
                saved_report["evaluation"]["repeated_grouped_cv"]["total_runs"],
                DEFAULT_EVALUATION_SEED_COUNT * DEFAULT_GROUPED_CV_FOLDS,
            )
            self.assertEqual(
                saved_report["evaluation"]["repeated_grouped_cv"]["class_group_counts"],
                expected_class_group_counts,
            )
            self.assertEqual(
                saved_report["evaluation"]["repeated_grouped_cv"]["support_summary"]["test_class_group_counts"]["min"],
                {name: 1 for name in CLASS_NAMES},
            )
            self.assertFalse(saved_report["evaluation"]["protocol_family_holdout"]["feasible"])
            self.assertIn(
                "std",
                saved_report["evaluation"]["repeated_grouped_cv"]["aggregate"]["models"]["neural_net"][MODEL_ONLY_NAMESPACE]["cyber_binary_metrics"]["f1"],
            )
            self.assertEqual(saved_report["ranking_metric_order"], list(SELECTION_METRIC_ORDER))
            self.assertIn("novelty_adaptation_mode=disabled", summary_text)
            self.assertIn("comparison_only=true", summary_text)
            self.assertIn("selection_metric_order=", summary_text)
            self.assertIn("deployed_selection_metrics=", summary_text)
            self.assertIn("split_heldout_groups_per_class=", summary_text)
            self.assertIn("grouped_cv_class_group_counts=", summary_text)
            self.assertIn("protocol_family_holdout_feasible=false", summary_text)

    @unittest.skipIf(TRAINING_IMPORT_ERROR is not None, f"training dependencies unavailable: {TRAINING_IMPORT_ERROR}")
    def test_training_report_drops_legacy_forget_alpha_fields(self) -> None:
        rows = build_training_rows()
        with TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "data" / "dataset.jsonl"
            output_dir = Path(tmpdir) / "artifacts"
            write_training_fixture(dataset_path, rows)
            with mock.patch("main.audit_dataset_sanity", autospec=True, return_value=build_passing_dataset_sanity_stub()):
                with mock.patch("main.export_runtime_files", autospec=True):
                    report = run_training(
                        dataset_path,
                        output_dir,
                        seed=7,
                        make_plots=False,
                        training_path_name=TRAINING_PATH_LEGACY_FPRIME_BASELINE,
                    )

            self.assertNotIn("forget_alpha", report)
            self.assertEqual(report["novelty_adaptation"]["mode"], "disabled")
            self.assertNotIn("requested_forget_alpha", report["novelty_adaptation"])
            self.assertNotIn("effective_forget_alpha", report["novelty_adaptation"])

            saved_report = json.loads((output_dir / "reports" / "metrics.json").read_text(encoding="utf-8"))
            summary_text = (output_dir / "reports" / "summary.txt").read_text(encoding="utf-8")
            self.assertNotIn("forget_alpha", saved_report)
            self.assertNotIn("requested_forget_alpha", saved_report["novelty_adaptation"])
            self.assertNotIn("effective_forget_alpha", saved_report["novelty_adaptation"])
            self.assertNotIn("novelty_adaptation_requested_forget_alpha", summary_text)
            self.assertNotIn("novelty_adaptation_effective_forget_alpha", summary_text)

    @unittest.skipIf(TRAINING_IMPORT_ERROR is not None, f"training dependencies unavailable: {TRAINING_IMPORT_ERROR}")
    def test_training_blocks_deployment_when_dataset_sanity_detects_trivial_task(self) -> None:
        rows = build_training_rows()
        with TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "data" / "dataset.jsonl"
            output_dir = Path(tmpdir) / "artifacts"
            write_training_fixture(dataset_path, rows)

            with mock.patch("main.export_runtime_files", autospec=True) as export_mock:
                with self.assertRaises(SystemExit) as exc:
                    run_training(
                        dataset_path,
                        output_dir,
                        seed=7,
                        make_plots=False,
                        training_path_name=TRAINING_PATH_LEGACY_FPRIME_BASELINE,
                    )

            self.assertIn("dataset_sanity_trivial_task", str(exc.exception))
            export_mock.assert_not_called()

            metrics_path = output_dir / "reports" / "metrics.json"
            sanity_path = output_dir / "reports" / "dataset_sanity.json"
            self.assertTrue(metrics_path.exists())
            self.assertTrue(sanity_path.exists())

            report = json.loads(metrics_path.read_text(encoding="utf-8"))
            sanity_report = json.loads(sanity_path.read_text(encoding="utf-8"))

            self.assertFalse(report["deployment_ready"])
            self.assertEqual(report["deployment_blocked_reason"], "dataset_sanity_trivial_task")
            self.assertFalse(report["dataset_sanity"]["passed"])
            self.assertEqual(report["request_only_baseline"]["feature_names"], list(REQUEST_ONLY_BASELINE_FEATURE_NAMES))
            self.assertEqual(
                report["dataset_sanity"]["request_tuple_purity"]["feature_names"],
                list(COMMAND_ONLY_BASELINE_FEATURE_NAMES),
            )
            failed_checks = {check["reason"] for check in report["dataset_sanity"]["blocking_issues"]}
            self.assertIn("request_only_baseline_near_perfect", failed_checks)
            self.assertIn("episode_signatures_insufficiently_diverse", failed_checks)
            self.assertFalse(sanity_report["passed"])
            self.assertEqual(
                sanity_report["baselines"]["request_only"]["feature_names"],
                list(REQUEST_ONLY_BASELINE_FEATURE_NAMES),
            )
            self.assertGreaterEqual(
                sanity_report["request_tuple_purity"]["summary"]["rows"],
                len(rows),
            )

    @unittest.skipIf(TRAINING_IMPORT_ERROR is not None, f"training dependencies unavailable: {TRAINING_IMPORT_ERROR}")
    def test_training_exports_safe_feature_tiers_in_model_artifacts(self) -> None:
        rows = build_training_rows()
        with TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "data" / "dataset.jsonl"
            output_dir = Path(tmpdir) / "artifacts"
            deployed_model_dir = Path(tmpdir) / "deployments" / "DetectorRB3" / "config"
            write_training_fixture(dataset_path, rows)

            with mock.patch("main.audit_dataset_sanity", autospec=True, return_value=build_passing_dataset_sanity_stub()):
                with mock.patch("main.DEFAULT_MODEL_DIR", deployed_model_dir):
                    report = run_training(
                        dataset_path,
                        output_dir,
                        seed=7,
                        make_plots=False,
                        training_path_name=TRAINING_PATH_LEGACY_FPRIME_BASELINE,
                    )

            self.assertTrue(report["deployment_ready"])
            self.assertIsNotNone(report["deployment_winner"])
            self.assertTrue(report["dataset_sanity"]["passed"])

            winner_name = str(report["deployment_winner"]["name"])
            expected_layout = MODEL_FEATURE_LAYOUTS[winner_name]
            self.assertEqual(expected_layout["feature_tier"], PRIMARY_MODEL_FEATURE_TIER)

            for model_name, artifact_name in (("random_forest", "forest.json"), ("neural_net", "nn.json"), (winner_name, "model.json")):
                payload = json.loads((output_dir / "models" / artifact_name).read_text(encoding="utf-8"))
                self.assertEqual(payload["feature_tier"], PRIMARY_MODEL_FEATURE_TIER)
                self.assertEqual(payload["feature_names"], list(MODEL_FEATURE_LAYOUTS[model_name]["feature_names"]))
                self.assertFalse(set(payload["feature_names"]) & FORBIDDEN_PRIMARY_MODEL_FEATURE_NAMES)

            deployed_payload = json.loads((deployed_model_dir / "model.json").read_text(encoding="utf-8"))
            self.assertEqual(deployed_payload["feature_tier"], PRIMARY_MODEL_FEATURE_TIER)
            self.assertEqual(deployed_payload["feature_names"], list(PRIMARY_MODEL_FEATURE_NAMES))

            novelty_cfg = (output_dir / "models" / "novelty.cfg").read_text(encoding="utf-8")
            deployed_novelty_cfg = (deployed_model_dir / "novelty.cfg").read_text(encoding="utf-8")
            for cfg_text in (novelty_cfg, deployed_novelty_cfg):
                config = {}
                for line in cfg_text.splitlines():
                    if "=" not in line:
                        continue
                    key, value = line.split("=", 1)
                    config[key.strip()] = value.strip()
                self.assertEqual(config["feature_tier"], NOVELTY_MODEL_FEATURE_TIER)
                self.assertEqual(config["feature_names"].split(","), list(NOVELTY_FEATURE_NAMES))

    @unittest.skipIf(TRAINING_IMPORT_ERROR is not None, f"training dependencies unavailable: {TRAINING_IMPORT_ERROR}")
    def test_poster_default_training_uses_canonical_neural_only_path(self) -> None:
        rows = build_training_rows()
        with TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "data" / "dataset.jsonl"
            output_dir = Path(tmpdir) / "artifacts"
            deployed_model_dir = Path(tmpdir) / "deployments" / "DetectorRB3" / "config"
            write_training_fixture(dataset_path, rows)
            poster_dataset_sanity = build_poster_passing_dataset_sanity_stub()

            with mock.patch("main.audit_dataset_sanity", autospec=True, return_value=poster_dataset_sanity):
                with mock.patch("main.DEFAULT_MODEL_DIR", deployed_model_dir):
                    report = run_training(dataset_path, output_dir, seed=7, make_plots=False)

            self.assertEqual(report["training_path"]["name"], POSTER_DEFAULT_TRAINING_PATH_NAME)
            self.assertTrue(report["training_path"]["default"])
            self.assertFalse(report["training_path"]["comparison_only"])
            self.assertFalse(report["comparison_only"])
            self.assertEqual(report["blue_feature_policy"]["policy_name"], BLUE_FEATURE_POLICY_POSTER_DEFAULT)
            self.assertEqual(report["blue_model"]["architecture"]["family"], POSTER_BLUE_MODEL_FAMILY)
            self.assertEqual(
                report["blue_model"]["training_config"]["validation_source"],
                "calibration_rows",
            )
            self.assertEqual(
                report["blue_model"]["training_config"]["class_balance_strategy"],
                "deterministic_oversample_to_max_class",
            )
            self.assertGreaterEqual(int(report["blue_model"]["training_summary"]["best_epoch"]), 1)
            self.assertEqual(
                report["blue_model"]["output_formulation"]["unsafe_risk_score"],
                "1 - p(benign)",
            )
            self.assertEqual(report["feature_sets"]["training_path"]["name"], POSTER_DEFAULT_TRAINING_PATH_NAME)
            self.assertEqual(
                report["feature_sets"]["models"]["neural_net"]["feature_tier"],
                POSTER_DEFAULT_PRIMARY_MODEL_FEATURE_TIER,
            )
            self.assertEqual(
                report["feature_sets"]["models"]["neural_net"]["feature_names"],
                list(POSTER_DEFAULT_REQUEST_ONLY_BASELINE_FEATURE_NAMES),
            )
            self.assertEqual(set(report["feature_sets"]["models"].keys()), {"neural_net"})
            self.assertNotIn("novelty", report["feature_sets"]["models"])
            self.assertEqual(
                report["feature_sets"]["baselines"]["command_only"]["feature_names"],
                list(POSTER_DEFAULT_COMMAND_BASELINE_FEATURE_NAMES),
            )
            self.assertEqual(
                report["feature_sets"]["baselines"]["request_only"]["feature_names"],
                list(POSTER_DEFAULT_REQUEST_ONLY_BASELINE_FEATURE_NAMES),
            )
            self.assertEqual(
                report["feature_sets"]["baselines"]["protocol_only"]["feature_names"],
                list(PROTOCOL_ONLY_BASELINE_FEATURE_NAMES),
            )
            self.assertEqual(
                report["feature_sets"]["baselines"]["raw_protocol_shortcuts"]["feature_names"],
                list(RAW_PROTOCOL_SHORTCUT_BASELINE_FEATURE_NAMES),
            )
            self.assertEqual(report["history_featurization"]["mode"], "canonical_rows")
            self.assertEqual(report["history_featurization"]["group_key"], "run_id")
            self.assertTrue(report["history_featurization"]["canonical_recent_behavior"])
            self.assertNotIn(STACKED_DETECTOR_NAMESPACE, report["metrics"])
            self.assertNotIn(STACKED_DETECTOR_NAMESPACE, report["thresholds"])
            self.assertEqual(report["feature_importance"], {"neural_net": []})
            self.assertTrue(report["outcome_only_baseline"]["not_applicable"])
            self.assertEqual(
                report["outcome_only_baseline"]["reason"],
                "terminal_outcomes_not_allowed_in_poster_path",
            )
            self.assertTrue(report["protocol_only_baseline"]["not_applicable"])
            self.assertTrue(report["raw_protocol_shortcuts_baseline"]["not_applicable"])
            self.assertFalse(report["evaluation"]["protocol_family_holdout"]["feasible"])
            self.assertEqual(
                report["ranking_metric_order"],
                [
                    SELECTION_METRIC_CLASS_MACRO_F1,
                    SELECTION_METRIC_MIN_PER_CLASS_RECALL,
                    "model_only.anomaly_f1",
                    SELECTION_METRIC_MODEL_CYBER_F1,
                ],
            )
            self.assertIn("model_only.anomaly_f1", report["deployment_winner"]["selection_metrics"])
            self.assertNotIn(SELECTION_METRIC_STACKED_ANOMALY_F1, report["deployment_winner"]["selection_metrics"])
            forbidden_names = {
                "service_id",
                "command_id",
                "target_node_id",
                "target_cpu_total_pct",
                "resp_bytes",
                "timeout",
            }
            self.assertFalse(set(report["feature_names"]) & forbidden_names)

            model_dir = output_dir / "models"
            self.assertTrue((model_dir / "nn.json").exists())
            self.assertTrue((model_dir / POSTER_BLUE_MODEL_ARTIFACT_NAME).exists())
            self.assertTrue((model_dir / RUNTIME_BUNDLE_MANIFEST_NAME).exists())
            self.assertFalse((model_dir / "forest.json").exists())
            self.assertFalse((model_dir / "model.json").exists())
            self.assertFalse((model_dir / "novelty.cfg").exists())
            self.assertFalse((model_dir / "calibrator.json").exists())
            self.assertFalse((model_dir / "calibrator_nn.json").exists())
            self.assertFalse((model_dir / "calibrator_rf.json").exists())
            self.assertTrue((deployed_model_dir / POSTER_BLUE_MODEL_ARTIFACT_NAME).exists())
            self.assertTrue((deployed_model_dir / RUNTIME_BUNDLE_MANIFEST_NAME).exists())
            self.assertFalse((deployed_model_dir / "model.json").exists())
            self.assertFalse((deployed_model_dir / "novelty.cfg").exists())
            self.assertFalse((deployed_model_dir / "calibrator.json").exists())

            model_payload = json.loads((model_dir / POSTER_BLUE_MODEL_ARTIFACT_NAME).read_text(encoding="utf-8"))
            manifest_payload = json.loads((model_dir / RUNTIME_BUNDLE_MANIFEST_NAME).read_text(encoding="utf-8"))
            self.assertEqual(model_payload["feature_tier"], POSTER_DEFAULT_PRIMARY_MODEL_FEATURE_TIER)
            self.assertEqual(model_payload["feature_names"], list(POSTER_DEFAULT_REQUEST_ONLY_BASELINE_FEATURE_NAMES))
            self.assertEqual(model_payload["architecture"]["family"], POSTER_BLUE_MODEL_FAMILY)
            self.assertEqual(model_payload["output_formulation"]["unsafe_risk_score"], "1 - p(benign)")
            self.assertEqual(
                model_payload["training_config"]["validation_source"],
                "calibration_rows",
            )
            self.assertGreaterEqual(int(model_payload["training_summary"]["best_epoch"]), 1)
            self.assertEqual(manifest_payload["runtime_kind"], POSTER_BLUE_RUNTIME_KIND)
            self.assertEqual(manifest_payload["primary_model_artifact"], POSTER_BLUE_MODEL_ARTIFACT_NAME)
            self.assertFalse(manifest_payload["uses_rules"])
            self.assertFalse(manifest_payload["uses_novelty"])
            self.assertFalse(manifest_payload["uses_calibrator"])

            saved_report = json.loads((output_dir / "reports" / "metrics.json").read_text(encoding="utf-8"))
            summary_text = (output_dir / "reports" / "summary.txt").read_text(encoding="utf-8")
            self.assertEqual(saved_report["training_path"]["name"], POSTER_DEFAULT_TRAINING_PATH_NAME)
            self.assertFalse(saved_report["training_path"]["comparison_only"])
            self.assertFalse(saved_report["comparison_only"])
            self.assertEqual(saved_report["blue_feature_policy"]["policy_name"], BLUE_FEATURE_POLICY_POSTER_DEFAULT)
            self.assertEqual(saved_report["blue_model"]["architecture"]["family"], POSTER_BLUE_MODEL_FAMILY)
            self.assertEqual(saved_report["deployment_winner"]["runtime_kind"], POSTER_BLUE_RUNTIME_KIND)
            self.assertTrue(saved_report["deployment_winner"]["model_path"].endswith(POSTER_BLUE_MODEL_ARTIFACT_NAME))
            self.assertTrue(saved_report["deployment_winner"]["manifest_path"].endswith(RUNTIME_BUNDLE_MANIFEST_NAME))
            self.assertNotIn(STACKED_DETECTOR_NAMESPACE, saved_report["metrics"])
            self.assertFalse(saved_report["evaluation"]["protocol_family_holdout"]["feasible"])
            expected_class_group_counts = {
                name: int(saved_report["split_summary"]["base_class_group_counts"][name])
                + int(saved_report["split_summary"]["calibration_class_group_counts"][name])
                + int(saved_report["split_summary"]["test_class_group_counts"][name])
                for name in CLASS_NAMES
            }
            self.assertEqual(
                saved_report["evaluation"]["repeated_grouped_cv"]["class_group_counts"],
                expected_class_group_counts,
            )
            self.assertEqual(
                saved_report["evaluation"]["repeated_grouped_cv"]["support_summary"]["test_class_group_counts"]["min"],
                {name: 1 for name in CLASS_NAMES},
            )
            self.assertIn("blue_model_family=poster_blue_mlp_v1", summary_text)
            self.assertIn("comparison_only=false", summary_text)
            self.assertIn("training_path=poster_default_canonical", summary_text)
            self.assertIn("deployed_grouped_cv_model_only_anomaly_f1_mean=", summary_text)
            self.assertIn("split_heldout_groups_per_class=", summary_text)
            self.assertIn("grouped_cv_class_group_counts=", summary_text)
            self.assertIn("protocol_family_holdout_feasible=false", summary_text)

    @unittest.skipIf(TRAINING_IMPORT_ERROR is not None, f"training dependencies unavailable: {TRAINING_IMPORT_ERROR}")
    def test_score_packets_uses_canonical_rows_for_poster_model(self) -> None:
        rows = build_training_rows()
        with TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "data" / "dataset.jsonl"
            output_dir = Path(tmpdir) / "artifacts"
            deployed_model_dir = Path(tmpdir) / "deployments" / "DetectorRB3" / "config"
            packet_path = Path(tmpdir) / "data" / "packets.jsonl"
            write_training_fixture(dataset_path, rows)
            write_jsonl(packet_path, build_score_packets_fixture())

            with mock.patch("main.audit_dataset_sanity", autospec=True, return_value=build_poster_passing_dataset_sanity_stub()):
                with mock.patch("main.DEFAULT_MODEL_DIR", deployed_model_dir):
                    run_training(dataset_path, output_dir, seed=7, make_plots=False)

            summary = score_packets(packet_path, output_dir / "scored-output", output_dir / "models")
            self.assertEqual(summary["scoring_input_kind"], "poster_canonical_request_rows")
            self.assertTrue((output_dir / "scored-output" / "scored" / "canonical_rows.jsonl").exists())
            scored_rows = [json.loads(line) for line in (output_dir / "scored-output" / "scored" / "rows.jsonl").read_text(encoding="utf-8").splitlines() if line]
            self.assertEqual(len(scored_rows), 1)
            self.assertIn("risk", scored_rows[0])
            self.assertIn("unsafe_risk", scored_rows[0])
            self.assertEqual(scored_rows[0]["runtime_kind"], POSTER_BLUE_RUNTIME_KIND)
            self.assertNotIn("rules", scored_rows[0])
            self.assertNotIn("novelty", scored_rows[0])
            self.assertNotIn("service_id", scored_rows[0])
            self.assertIn("command_semantics.canonical_command_family", scored_rows[0])
            bundle = load_runtime_bundle(output_dir / "models")
            self.assertEqual(getattr(bundle, "runtime_kind", ""), POSTER_BLUE_RUNTIME_KIND)

    @unittest.skipIf(TRAINING_IMPORT_ERROR is not None, f"training dependencies unavailable: {TRAINING_IMPORT_ERROR}")
    def test_training_blocks_deployment_when_family_holdout_generalization_is_poor(self) -> None:
        rows = build_family_holdout_regression_rows()
        with TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "data" / "dataset.jsonl"
            output_dir = Path(tmpdir) / "artifacts"
            write_training_fixture(dataset_path, rows)

            with mock.patch("main.audit_dataset_sanity", autospec=True, return_value=build_passing_dataset_sanity_stub()):
                with mock.patch("main.export_runtime_files", autospec=True) as export_mock:
                    with self.assertRaises(SystemExit) as exc:
                        run_training(
                            dataset_path,
                            output_dir,
                            seed=7,
                            make_plots=False,
                            training_path_name=TRAINING_PATH_LEGACY_FPRIME_BASELINE,
                        )

            self.assertIn("deployment/export was blocked", str(exc.exception))
            export_mock.assert_not_called()

            metrics_path = output_dir / "reports" / "metrics.json"
            self.assertTrue(metrics_path.exists())
            report = json.loads(metrics_path.read_text(encoding="utf-8"))
            self.assertFalse(report["deployment_ready"])
            self.assertEqual(report["deployment_blocked_reason"], "generalization_metrics_below_threshold")
            self.assertTrue(report["evaluation"]["scenario_family_holdout"]["feasible"])
            self.assertTrue(report["evaluation"]["command_family_holdout"]["feasible"])
            self.assertFalse(report["evaluation"]["protocol_family_holdout"]["feasible"])
            gate_models = report["evaluation"]["deployment_gate"]["models"]
            self.assertTrue(any(not payload["eligible_for_deployment"] for payload in gate_models.values()))

    @unittest.skipIf(TRAINING_IMPORT_ERROR is not None, f"training dependencies unavailable: {TRAINING_IMPORT_ERROR}")
    def test_poster_training_exports_analysis_bundle_when_deployment_is_blocked(self) -> None:
        rows = build_family_holdout_regression_rows()
        with TemporaryDirectory() as tmpdir:
            dataset_path = Path(tmpdir) / "data" / "dataset.jsonl"
            output_dir = Path(tmpdir) / "artifacts"
            write_training_fixture(dataset_path, rows)

            with mock.patch("main.audit_dataset_sanity", autospec=True, return_value=build_poster_passing_dataset_sanity_stub()):
                with self.assertRaises(SystemExit) as exc:
                    run_training(
                        dataset_path,
                        output_dir,
                        seed=7,
                        make_plots=False,
                    )

            self.assertIn("deployment/export was blocked", str(exc.exception))
            report = json.loads((output_dir / "reports" / "metrics.json").read_text(encoding="utf-8"))
            self.assertFalse(report["deployment_ready"])
            self.assertEqual(report["deployment_blocked_reason"], "generalization_metrics_below_threshold")
            analysis_bundle = report["analysis_runtime_bundle"]
            self.assertIsNotNone(analysis_bundle)
            self.assertTrue(Path(analysis_bundle["artifact_dir"]).exists())
            self.assertTrue(Path(analysis_bundle["model_path"]).exists())
            self.assertTrue(Path(analysis_bundle["manifest_path"]).exists())
            runtime_bundle = load_runtime_bundle(Path(analysis_bundle["artifact_dir"]))
            self.assertEqual(getattr(runtime_bundle, "runtime_kind", ""), POSTER_BLUE_RUNTIME_KIND)


if __name__ == "__main__":
    unittest.main()
