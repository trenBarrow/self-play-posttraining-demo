from __future__ import annotations

import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import (
    TRAINING_IMPORT_ERROR,
    TRAINING_PATH_LEGACY_FPRIME_BASELINE,
    feature_sets_report,
    run_training,
)
from runtime import PRIMARY_MODEL_FEATURE_NAMES
from tools.train.poster_default import POSTER_DEFAULT_TRAINING_PATH_NAME
from tools.shared.canonical_records import canonicalize_legacy_fprime_transaction
from tools.shared.feature_policy import (
    BLUE_FEATURE_POLICY_LEGACY_REQUEST_TIME_BASELINE,
    BLUE_FEATURE_POLICY_POSTER_DEFAULT,
    BlueFeaturePolicyError,
    available_blue_feature_policies,
    extract_blue_model_features,
    load_blue_feature_policy,
    validate_blue_feature_names,
)


class CanonicalFeaturePolicyTests(unittest.TestCase):
    def make_legacy_transaction(self) -> dict[str, object]:
        return {
            "run_id": 5,
            "episode_id": 2,
            "episode_label": 0,
            "episode_kind": "benign",
            "label": 0,
            "label_name": "benign",
            "session_id": "ops_b1-0005",
            "txn_id": "5-000001-ops_b1",
            "send_id": "send-0005",
            "target_stream_id": "fprime_a:50050",
            "target_stream_index": 0.0,
            "attack_family": "none",
            "phase": "startup",
            "actor": "ops_b1",
            "actor_role": "ops_primary",
            "actor_trust": 0.97,
            "command": "cmdDisp.CMD_NO_OP",
            "service": "cmdDisp",
            "args": {"duration": 5, "arm": True, "mode": "SAFE"},
            "target_service": "fprime_a",
            "target_node_id": 1.0,
            "request_ts_ms": 61000.0,
            "packet_gap_ms": 20.0,
            "req_bytes": 40.0,
            "resp_bytes": 12.0,
            "gds_accept": 1.0,
            "sat_success": 1.0,
            "timeout": 0.0,
            "response_code": 0.0,
            "reason": "completed",
            "uplink_ts_ms": 61020.0,
            "sat_response_ts_ms": 61045.0,
            "final_ts_ms": 61070.0,
            "request_to_uplink_ms": 20.0,
            "uplink_to_sat_response_ms": 25.0,
            "sat_response_to_final_ms": 25.0,
            "response_direction_seen": 1.0,
            "final_observed_on_wire": 1.0,
            "txn_warning_events": 0.0,
            "txn_error_events": 0.0,
            "target_cpu_total_pct": 11.0,
            "target_cpu_00_pct": 5.0,
            "target_cpu_01_pct": 6.0,
            "peer_cpu_total_pct": 7.0,
            "peer_cpu_00_pct": 3.0,
            "peer_cpu_01_pct": 4.0,
            "target_blockdrv_cycles_1m": 16.0,
            "peer_blockdrv_cycles_1m": 8.0,
            "target_cmd_errors_1m": 1.0,
            "peer_cmd_errors_1m": 0.0,
            "target_cmds_dispatched_1m": 5.0,
            "peer_cmds_dispatched_1m": 2.0,
            "target_filemanager_errors_1m": 2.0,
            "target_filedownlink_warnings_1m": 1.0,
            "peer_filemanager_errors_1m": 0.0,
            "peer_filedownlink_warnings_1m": 0.0,
            "target_hibuffs_total": 2.0,
            "peer_hibuffs_total": 1.0,
            "target_rg1_max_time_ms": 8.0,
            "target_rg2_max_time_ms": 7.0,
            "peer_rg1_max_time_ms": 4.0,
            "peer_rg2_max_time_ms": 5.0,
            "target_telemetry_age_ms": 180.0,
            "peer_telemetry_age_ms": 220.0,
            "command_rate_1m": 2.0,
            "error_rate_1m": 0.25,
        }

    def make_related_packets(self) -> list[dict[str, object]]:
        request_packet = {
            "ts_ms": 61000,
            "packet_kind": "request",
            "src": "ops_b1",
            "dst": "fprime_a",
            "src_ip": "192.168.144.22",
            "dst_ip": "192.168.144.2",
            "src_port": 50101,
            "dst_port": 50050,
            "target_service": "fprime_a",
            "target_stream_id": "fprime_a:50050",
            "target_stream_index": 0,
            "service": "cmdDisp",
            "command": "cmdDisp.CMD_NO_OP",
            "label": 0,
            "episode_id": 2,
            "episode_kind": "benign",
            "session_id": "ops_b1-0005",
            "txn_id": "5-000001-ops_b1",
            "send_id": "send-0005",
            "attack_family": "none",
            "phase": "startup",
            "actor": "ops_b1",
            "actor_role": "ops_primary",
            "actor_trust": 0.97,
            "args": {"duration": 5, "arm": True, "mode": "SAFE"},
            "bytes_on_wire": 40,
            "observed_on_wire": 1,
            "ts_source": "pcap",
            "bytes_source": "pcap",
        }
        final_packet = {
            **request_packet,
            "packet_kind": "final",
            "src": "fprime_a",
            "dst": "ops_b1",
            "src_ip": "192.168.144.2",
            "dst_ip": "192.168.144.22",
            "src_port": 50050,
            "dst_port": 50101,
            "ts_ms": 61070,
            "gds_accept": 1,
            "sat_success": 1,
            "timeout": 0,
            "response_code": 0,
            "reason": "completed",
            "response_direction_seen": 1,
            "final_observed_on_wire": 1,
            "txn_warning_events": 0,
            "txn_error_events": 0,
            "event_name": "CMD_RESPONSE",
        }
        return [request_packet, final_packet]

    def make_canonical_row(self) -> dict[str, object]:
        return canonicalize_legacy_fprime_transaction(
            self.make_legacy_transaction(),
            related_packets=self.make_related_packets(),
            source_artifact_paths=["logs/run.csv", "pcap/run_0005.pcap"],
            mission_context={"window_class": "startup_window"},
        )

    def test_available_blue_feature_policies_include_legacy_and_poster_profiles(self) -> None:
        self.assertEqual(
            available_blue_feature_policies(),
            [
                BLUE_FEATURE_POLICY_LEGACY_REQUEST_TIME_BASELINE,
                BLUE_FEATURE_POLICY_POSTER_DEFAULT,
            ],
        )

    def test_poster_blue_policy_allowed_names_stay_protocol_neutral(self) -> None:
        policy = load_blue_feature_policy(BLUE_FEATURE_POLICY_POSTER_DEFAULT)
        names = {name.lower() for name in policy["allowed_features"]}
        self.assertFalse(any("fprime" in name for name in names))
        self.assertFalse(any("mavlink" in name for name in names))
        self.assertFalse(any("cpu_total_pct" in name for name in names))
        self.assertFalse(any("cmd_errors_1m" in name for name in names))
        self.assertFalse(any("filemanager_errors_1m" in name for name in names))
        self.assertFalse(any("rg1_max_time_ms" in name for name in names))

    def test_extract_blue_model_features_accepts_canonical_row_under_poster_policy(self) -> None:
        row = self.make_canonical_row()
        features = extract_blue_model_features(
            row,
            policy_name=BLUE_FEATURE_POLICY_POSTER_DEFAULT,
            require_all_allowed=True,
        )
        report = validate_blue_feature_names(features.keys(), BLUE_FEATURE_POLICY_POSTER_DEFAULT)
        policy = load_blue_feature_policy(BLUE_FEATURE_POLICY_POSTER_DEFAULT)
        self.assertTrue(report["passed"])
        self.assertEqual(set(features), set(policy["allowed_features"]))
        self.assertNotIn("audit_context.raw_command_name", features)
        self.assertEqual(features["command_semantics.canonical_command_family"], "read_only_inspection")
        self.assertEqual(features["normalized_state.target_compute_load_ratio"], 0.11)
        self.assertEqual(features["normalized_state.target_service_issue_ratio"], 0.75)
        self.assertEqual(features["normalized_state.target_scheduler_pressure_ratio"], 0.8)
        self.assertIsNone(features["normalized_state.target_link_pressure_ratio"])
        self.assertIsNone(features["normalized_state.target_power_pressure_ratio"])

    def test_poster_blue_policy_rejects_current_legacy_primary_features(self) -> None:
        with self.assertRaises(BlueFeaturePolicyError) as exc:
            validate_blue_feature_names(PRIMARY_MODEL_FEATURE_NAMES, BLUE_FEATURE_POLICY_POSTER_DEFAULT)
        message = str(exc.exception)
        self.assertIn("service_id", message)
        self.assertIn("target_cpu_total_pct", message)

    def test_legacy_request_time_policy_accepts_current_primary_features(self) -> None:
        report = validate_blue_feature_names(
            PRIMARY_MODEL_FEATURE_NAMES,
            BLUE_FEATURE_POLICY_LEGACY_REQUEST_TIME_BASELINE,
        )
        self.assertTrue(report["passed"])
        self.assertEqual(set(report["allowed_used"]), set(PRIMARY_MODEL_FEATURE_NAMES))

    def test_feature_sets_report_records_active_blue_policy(self) -> None:
        report = feature_sets_report(
            BLUE_FEATURE_POLICY_LEGACY_REQUEST_TIME_BASELINE,
            training_path_name=TRAINING_PATH_LEGACY_FPRIME_BASELINE,
        )
        self.assertEqual(
            report["blue_feature_policies"]["active"]["policy_name"],
            BLUE_FEATURE_POLICY_LEGACY_REQUEST_TIME_BASELINE,
        )
        self.assertEqual(report["training_path"]["name"], TRAINING_PATH_LEGACY_FPRIME_BASELINE)
        self.assertIn(
            BLUE_FEATURE_POLICY_POSTER_DEFAULT,
            report["blue_feature_policies"]["available"],
        )

    def test_feature_sets_report_defaults_to_poster_canonical_path(self) -> None:
        report = feature_sets_report(training_path_name=POSTER_DEFAULT_TRAINING_PATH_NAME)
        self.assertEqual(report["training_path"]["name"], POSTER_DEFAULT_TRAINING_PATH_NAME)
        self.assertEqual(report["training_path"]["blue_feature_policy_name"], BLUE_FEATURE_POLICY_POSTER_DEFAULT)
        self.assertEqual(report["blue_feature_policies"]["active"]["policy_name"], BLUE_FEATURE_POLICY_POSTER_DEFAULT)
        self.assertNotIn("service_id", report["primary_model_feature_names"])
        self.assertNotIn("target_cpu_total_pct", report["primary_model_feature_names"])

    @unittest.skipIf(TRAINING_IMPORT_ERROR is not None, f"training dependencies unavailable: {TRAINING_IMPORT_ERROR}")
    def test_legacy_run_training_rejects_poster_policy_before_loading_dataset(self) -> None:
        with TemporaryDirectory() as tmpdir:
            with self.assertRaises(SystemExit) as exc:
                run_training(
                    dataset_path=Path(tmpdir) / "missing.jsonl",
                    output_dir=Path(tmpdir),
                    seed=7,
                    make_plots=False,
                    blue_feature_policy_name=BLUE_FEATURE_POLICY_POSTER_DEFAULT,
                    training_path_name=TRAINING_PATH_LEGACY_FPRIME_BASELINE,
                )
        self.assertIn("Blue feature policy violation", str(exc.exception))
        self.assertNotIn("missing.jsonl", str(exc.exception))


if __name__ == "__main__":
    unittest.main()
