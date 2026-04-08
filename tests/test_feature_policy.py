from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.shared.canonical_records import canonicalize_legacy_fprime_transaction
from tools.shared.feature_policy import (
    BLUE_FEATURE_POLICY_POSTER_DEFAULT,
    BlueFeaturePolicyError,
    check_blue_feature_names,
    extract_blue_model_features,
    load_blue_feature_policy,
    load_blue_forbidden_feature_policies,
    validate_blue_feature_names,
)


def make_legacy_transaction() -> dict[str, object]:
    return {
        "run_id": 3,
        "episode_id": 1,
        "episode_label": 0,
        "episode_kind": "benign",
        "label": 0,
        "label_name": "benign",
        "session_id": "ops_b1-0003",
        "txn_id": "3-000001-ops_b1",
        "send_id": "send-0003",
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
    }


def make_canonical_row() -> dict[str, object]:
    return canonicalize_legacy_fprime_transaction(
        make_legacy_transaction(),
        mission_context={"window_class": "startup_window"},
        recent_behavior={
            "command_rate_1m": 2.0,
            "error_rate_1m": 0.25,
            "repeat_command_count_10m": 1,
            "same_target_command_rate_1m": 1.0,
        },
    )


class FeaturePolicyContractTests(unittest.TestCase):
    def test_poster_allowlist_is_disjoint_from_denylist_rules(self) -> None:
        policy = load_blue_feature_policy(BLUE_FEATURE_POLICY_POSTER_DEFAULT)
        forbidden_profile = load_blue_forbidden_feature_policies()["profiles"][BLUE_FEATURE_POLICY_POSTER_DEFAULT]
        exact_forbidden = set(forbidden_profile.get("forbidden_features", {}))
        prefix_forbidden = tuple(forbidden_profile.get("forbidden_prefixes", {}))

        conflicts = [
            name
            for name in policy["allowed_features"]
            if name in exact_forbidden or any(name.startswith(prefix) for prefix in prefix_forbidden)
        ]
        self.assertEqual(conflicts, [])

    def test_poster_allowlist_stays_within_canonical_request_time_namespaces(self) -> None:
        policy = load_blue_feature_policy(BLUE_FEATURE_POLICY_POSTER_DEFAULT)
        allowed_prefixes = (
            "actor_context.",
            "mission_context.",
            "command_semantics.",
            "argument_profile.",
            "normalized_state.",
            "recent_behavior.",
        )
        for name in policy["allowed_features"]:
            if name == "platform_family":
                continue
            self.assertTrue(name.startswith(allowed_prefixes), msg=name)

    def test_poster_policy_rejects_representative_exact_and_prefix_shortcuts(self) -> None:
        report = check_blue_feature_names(
            [
                "platform_family",
                "command_semantics.canonical_command_family",
                "protocol_family",
                "audit_context.raw_command_name",
            ],
            BLUE_FEATURE_POLICY_POSTER_DEFAULT,
        )

        self.assertFalse(report["passed"])
        self.assertEqual(
            {entry["rule_type"] for entry in report["forbidden_violations"]},
            {"exact", "prefix"},
        )
        self.assertEqual(
            {entry["feature_name"] for entry in report["forbidden_violations"]},
            {"protocol_family", "audit_context.raw_command_name"},
        )

    def test_poster_allowlist_is_realizable_on_canonical_rows(self) -> None:
        features = extract_blue_model_features(
            make_canonical_row(),
            policy_name=BLUE_FEATURE_POLICY_POSTER_DEFAULT,
            require_all_allowed=True,
        )

        report = validate_blue_feature_names(features.keys(), BLUE_FEATURE_POLICY_POSTER_DEFAULT)
        self.assertTrue(report["passed"])
        self.assertEqual(set(features), set(load_blue_feature_policy(BLUE_FEATURE_POLICY_POSTER_DEFAULT)["allowed_features"]))
        self.assertEqual(features["platform_family"], "spacecraft")
        self.assertEqual(features["actor_context.trust_class"], "high")
        self.assertEqual(features["normalized_state.target_compute_load_ratio"], 0.11)

    def test_poster_policy_raises_for_forbidden_native_fields(self) -> None:
        with self.assertRaises(BlueFeaturePolicyError) as exc:
            validate_blue_feature_names(
                [
                    "platform_family",
                    "command_semantics.canonical_command_family",
                    "native_fields.legacy_record.command",
                ],
                BLUE_FEATURE_POLICY_POSTER_DEFAULT,
            )

        self.assertIn("native_fields.legacy_record.command", str(exc.exception))


if __name__ == "__main__":
    unittest.main()
