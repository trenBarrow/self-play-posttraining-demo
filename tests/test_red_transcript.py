from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.shared.canonical_records import canonicalize_legacy_fprime_transaction
from tools.shared.schema import adapt_legacy_fprime_transaction
from tools.train.red_transcript import (
    DEFAULT_RED_CONTEXT_BUDGET_PATH,
    RED_TRANSCRIPT_SCHEMA_VERSION,
    RedTranscriptBuildError,
    build_red_transcript,
    load_red_context_budget,
)


def make_legacy_transaction(
    index: int,
    *,
    actor: str = "red_primary",
    actor_role: str = "external",
    actor_trust: float = 0.12,
    command: str = "fileManager.RemoveDirectory",
    service: str = "fileManager",
    phase: str = "downlink",
    result: str = "success",
) -> dict[str, object]:
    request_ts_ms = 1000.0 + float(index * 100)
    transaction = {
        "run_id": 5,
        "episode_id": 2,
        "episode_label": 1,
        "episode_kind": "cyber",
        "label": 1,
        "label_name": "cyber",
        "session_id": f"{actor}-0005",
        "txn_id": f"txn-{index:04d}",
        "send_id": f"send-{index:04d}",
        "target_stream_id": "fprime_a:50050",
        "target_stream_index": float(index),
        "attack_family": "intrusion",
        "phase": phase,
        "actor": actor,
        "actor_role": actor_role,
        "actor_trust": actor_trust,
        "command": command,
        "service": service,
        "args": {"depth": (index % 4) + 1, "arm": True, "mode": "AUTO"},
        "target_service": "fprime_a",
        "target_node_id": 1.0,
        "request_ts_ms": request_ts_ms,
        "packet_gap_ms": 20.0,
        "req_bytes": 48.0,
        "resp_bytes": 16.0,
        "gds_accept": 1.0,
        "sat_success": 1.0,
        "timeout": 0.0,
        "response_code": 0.0,
        "reason": "completed",
        "uplink_ts_ms": request_ts_ms + 10.0,
        "sat_response_ts_ms": request_ts_ms + 25.0,
        "final_ts_ms": request_ts_ms + 40.0,
        "request_to_uplink_ms": 10.0,
        "uplink_to_sat_response_ms": 15.0,
        "sat_response_to_final_ms": 15.0,
        "response_direction_seen": 1.0,
        "final_observed_on_wire": 1.0,
        "txn_warning_events": 0.0,
        "txn_error_events": 0.0,
        "target_cpu_total_pct": 18.0,
        "target_cpu_00_pct": 9.0,
        "target_cpu_01_pct": 9.0,
        "peer_cpu_total_pct": 8.0,
        "peer_cpu_00_pct": 4.0,
        "peer_cpu_01_pct": 4.0,
        "target_blockdrv_cycles_1m": 10.0,
        "peer_blockdrv_cycles_1m": 3.0,
        "target_cmd_errors_1m": 0.0,
        "peer_cmd_errors_1m": 0.0,
        "target_cmds_dispatched_1m": 4.0,
        "peer_cmds_dispatched_1m": 1.0,
        "target_filemanager_errors_1m": 0.0,
        "target_filedownlink_warnings_1m": 0.0,
        "peer_filemanager_errors_1m": 0.0,
        "peer_filedownlink_warnings_1m": 0.0,
        "target_hibuffs_total": 0.0,
        "peer_hibuffs_total": 0.0,
        "target_rg1_max_time_ms": 5.0,
        "target_rg2_max_time_ms": 4.0,
        "peer_rg1_max_time_ms": 3.0,
        "peer_rg2_max_time_ms": 3.0,
        "target_telemetry_age_ms": 180.0,
        "peer_telemetry_age_ms": 200.0,
    }
    if result == "warning":
        transaction["txn_warning_events"] = 2.0
        transaction["response_code"] = 1.0
        transaction["reason"] = "warning_event"
    elif result == "failed":
        transaction["sat_success"] = 0.0
        transaction["txn_error_events"] = 2.0
        transaction["response_code"] = 2.0
        transaction["reason"] = "execution_failed"
    elif result == "timeout":
        transaction["sat_success"] = 0.0
        transaction["timeout"] = 1.0
        transaction["response_code"] = 3.0
        transaction["reason"] = "timeout"
        transaction["response_direction_seen"] = 0.0
        transaction["final_observed_on_wire"] = 0.0
    elif result == "rejected":
        transaction["gds_accept"] = 0.0
        transaction["sat_success"] = 0.0
        transaction["response_code"] = 4.0
        transaction["reason"] = "rejected"
    elif result == "invalid":
        transaction["sat_success"] = 0.0
        transaction["response_code"] = 5.0
        transaction["reason"] = "serialize_error"
    return transaction


def make_wrapped_history_item(
    index: int,
    *,
    actor: str = "red_primary",
    actor_role: str = "external",
    command: str = "fileManager.RemoveDirectory",
    service: str = "fileManager",
    phase: str = "downlink",
    window_class: str = "handoff",
    result: str = "success",
) -> dict[str, object]:
    legacy_transaction = make_legacy_transaction(
        index,
        actor=actor,
        actor_role=actor_role,
        command=command,
        service=service,
        phase=phase,
        result=result,
    )
    return {
        "raw_transaction": adapt_legacy_fprime_transaction(legacy_transaction),
        "canonical_row": canonicalize_legacy_fprime_transaction(
            legacy_transaction,
            mission_context={"window_class": window_class},
        ),
    }


class RedTranscriptTests(unittest.TestCase):
    def test_load_red_context_budget_exposes_explicit_compact_defaults(self) -> None:
        budget = load_red_context_budget()

        self.assertEqual(budget["config_path"], str(DEFAULT_RED_CONTEXT_BUDGET_PATH.resolve()))
        self.assertEqual(budget["schema_version"], "red_context_budget.v1")
        self.assertEqual(budget["transcript_schema_version"], RED_TRANSCRIPT_SCHEMA_VERSION)
        self.assertEqual(budget["limits"]["max_history_entries"], 8)
        self.assertEqual(budget["limits"]["truncation_policy"], "keep_most_recent_preserve_order")
        self.assertEqual(
            budget["transcript_format"]["per_event_field_order"],
            [
                "canonical_command_family",
                "actor_identity_bucket",
                "actor_role",
                "trust_class",
                "mission_phase",
                "window_class",
                "coarse_result_class",
                "mutation_scope",
                "persistence_class",
                "safety_criticality",
                "authority_level",
                "target_scope",
                "argument_size_bucket",
            ],
        )

    def test_build_red_transcript_uses_default_budget_and_keeps_recent_tail(self) -> None:
        history = [
            make_wrapped_history_item(
                index,
                command="fileManager.RemoveDirectory" if index % 2 == 0 else "fileDownlink.SendPartial",
                service="fileManager" if index % 2 == 0 else "fileDownlink",
                window_class="handoff" if index % 2 == 0 else "quiet",
                result="warning" if index % 3 == 0 else "success",
            )
            for index in range(12)
        ]

        transcript = build_red_transcript(history, actor_id="red_primary")

        self.assertEqual(transcript["included_history_count"], 8)
        self.assertEqual(transcript["truncated_event_count"], 4)
        self.assertEqual(
            [event["transaction_id"] for event in transcript["events"]],
            [f"txn-{index:04d}" for index in range(4, 12)],
        )
        self.assertTrue(transcript["serialized_lines"][0].startswith("cf:"))
        self.assertEqual(
            len(transcript["events"][0]["token_strings"]),
            len(load_red_context_budget()["transcript_format"]["per_event_field_order"]),
        )

    def test_build_red_transcript_filters_to_self_and_alias_history(self) -> None:
        history = [
            make_wrapped_history_item(0, actor="red_primary", command="cmdSeq.CS_VALIDATE", service="cmdSeq", result="success"),
            make_wrapped_history_item(1, actor="ops_b1", actor_role="ops_primary", command="cmdDisp.CMD_NO_OP", service="cmdDisp", phase="startup", window_class="maintenance", result="success"),
            make_wrapped_history_item(2, actor="red_secondary", command="fileDownlink.SendPartial", service="fileDownlink", result="warning"),
            make_wrapped_history_item(3, actor="red_primary", command="fileManager.RemoveDirectory", service="fileManager", result="failed"),
            make_wrapped_history_item(4, actor="ops_b1", actor_role="ops_primary", command="cmdDisp.CMD_NO_OP", service="cmdDisp", phase="startup", window_class="maintenance", result="success"),
            make_wrapped_history_item(5, actor="red_secondary", command="fileDownlink.SendPartial", service="fileDownlink", result="timeout"),
            make_wrapped_history_item(6, actor="red_primary", command="cmdSeq.CS_VALIDATE", service="cmdSeq", result="invalid"),
        ]

        transcript = build_red_transcript(
            history,
            actor_id="red_primary",
            actor_aliases=["red_secondary"],
            max_history_entries=4,
        )

        self.assertEqual(transcript["original_history_count"], 7)
        self.assertEqual(transcript["eligible_history_count"], 5)
        self.assertEqual(transcript["included_history_count"], 4)
        self.assertEqual(transcript["truncated_event_count"], 1)
        self.assertEqual(
            [event["transaction_id"] for event in transcript["events"]],
            ["txn-0002", "txn-0003", "txn-0005", "txn-0006"],
        )
        self.assertEqual(transcript["header_fields"]["protocol_family"], "fprime")
        self.assertEqual(transcript["header_fields"]["platform_family"], "spacecraft")
        self.assertEqual(transcript["events"][0]["field_values"]["actor_identity_bucket"], "ally")
        self.assertEqual(transcript["events"][1]["field_values"]["actor_identity_bucket"], "self")
        self.assertEqual(transcript["events"][0]["field_values"]["coarse_result_class"], "warning")
        self.assertEqual(transcript["events"][2]["field_values"]["coarse_result_class"], "timeout")
        self.assertIn("<ctx>", transcript["flattened_token_strings"])
        self.assertIn("<cmd>", transcript["flattened_token_strings"])

    def test_build_red_transcript_rejects_history_budget_above_configured_limit(self) -> None:
        history = [make_wrapped_history_item(index) for index in range(2)]
        configured_limit = int(load_red_context_budget()["limits"]["max_history_entries"])

        with self.assertRaises(RedTranscriptBuildError) as exc:
            build_red_transcript(
                history,
                actor_id="red_primary",
                max_history_entries=configured_limit + 1,
            )

        self.assertIn("exceeds configured transcript budget", str(exc.exception))

    def test_build_red_transcript_derives_coarse_result_classes_from_raw_history(self) -> None:
        result_order = ["success", "warning", "failed", "timeout", "rejected", "invalid"]
        raw_history = [
            adapt_legacy_fprime_transaction(
                make_legacy_transaction(
                    index,
                    result=result,
                    command="fileManager.RemoveDirectory",
                    service="fileManager",
                )
            )
            for index, result in enumerate(result_order)
        ]

        transcript = build_red_transcript(raw_history)

        self.assertEqual(
            [event["field_values"]["coarse_result_class"] for event in transcript["events"]],
            result_order,
        )
        self.assertTrue(all(token.startswith(("cf:", "ib:", "ar:", "tc:", "mp:", "wc:", "rc:", "mu:", "pe:", "sc:", "au:", "ts:", "ab:")) for token in transcript["events"][0]["token_strings"]))

    def test_build_red_transcript_serialization_is_stable_across_runs(self) -> None:
        history = [
            make_wrapped_history_item(0, result="success"),
            make_wrapped_history_item(1, result="warning"),
            make_wrapped_history_item(2, result="failed"),
        ]

        first = build_red_transcript(history, actor_id="red_primary")
        second = build_red_transcript(history, actor_id="red_primary")

        self.assertEqual(first["serialized_text"], second["serialized_text"])
        self.assertEqual(first["flattened_token_strings"], second["flattened_token_strings"])
        self.assertEqual(first["flattened_token_ids"], second["flattened_token_ids"])

    def test_build_red_transcript_requires_actor_id_for_mixed_actor_history(self) -> None:
        history = [
            adapt_legacy_fprime_transaction(make_legacy_transaction(0, actor="red_primary")),
            adapt_legacy_fprime_transaction(make_legacy_transaction(1, actor="red_secondary")),
        ]

        with self.assertRaises(RedTranscriptBuildError):
            build_red_transcript(history)


if __name__ == "__main__":
    unittest.main()
