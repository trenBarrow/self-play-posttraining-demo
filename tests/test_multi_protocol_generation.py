from __future__ import annotations

import json
import sys
import tempfile
import unittest
from pathlib import Path
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

import main as poster_main
from tools.shared import generate_dataset as shared_generate_dataset
from tools.shared.run_manifest import (
    MULTI_PROTOCOL_RUN_MANIFEST_SCHEMA_VERSION,
    build_class_overlap_report,
    build_multi_protocol_run_manifest,
    protocol_row_targets,
)
from tools.shared.schema import (
    RAW_PACKET_SCHEMA_VERSION,
    RAW_TRANSACTION_SCHEMA_VERSION,
    adapt_legacy_fprime_transaction,
)


def write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row, separators=(",", ":")) + "\n")


def make_legacy_fprime_transaction(run_id: int, episode_id: int) -> dict[str, object]:
    legacy_transaction = {
        "run_id": run_id,
        "episode_id": episode_id,
        "episode_label": 0,
        "episode_kind": "benign",
        "label": 0,
        "label_name": "benign",
        "session_id": f"ops_b1-{run_id:04d}",
        "txn_id": f"{run_id}-000001-ops_b1",
        "send_id": f"send-fprime-{run_id}",
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
        "request_ts_ms": 61000.0 + run_id,
        "packet_gap_ms": 20.0,
        "req_bytes": 40.0,
        "resp_bytes": 12.0,
        "gds_accept": 1.0,
        "sat_success": 1.0,
        "timeout": 0.0,
        "response_code": 0.0,
        "reason": "completed",
        "uplink_ts_ms": 61020.0 + run_id,
        "sat_response_ts_ms": 61045.0 + run_id,
        "final_ts_ms": 61070.0 + run_id,
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
    return adapt_legacy_fprime_transaction(
        legacy_transaction,
        source_artifact_paths=[f"logs/fprime/run_{run_id:04d}.jsonl"],
    )


def make_mavlink_raw_transaction(run_id: int, episode_id: int) -> dict[str, object]:
    return {
        "schema_version": RAW_TRANSACTION_SCHEMA_VERSION,
        "record_kind": "raw_transaction",
        "protocol_family": "mavlink",
        "protocol_version": "2.0",
        "platform_family": "multirotor",
        "sender": {
            "logical_id": "gcs_primary",
            "role": "ops_primary",
            "trust_score": 0.9,
            "network_endpoint": None,
        },
        "target": {
            "logical_id": "vehicle_1",
            "role": "airframe",
            "stream_id": "udp:14550",
            "stream_index": 0,
            "network_endpoint": None,
        },
        "command": {
            "raw_name": "MAV_CMD_NAV_TAKEOFF",
            "raw_identifier": {
                "service_name": "command_long",
                "native_service_id": 76,
                "native_command_id": 22,
            },
            "raw_arguments": [10.0, 1.5, True, "AUTO"],
            "raw_argument_representation": [10.0, 1.5, True, "AUTO"],
        },
        "timing": {
            "submitted_at_ms": 1000.0 + run_id,
            "request_forwarded_at_ms": 1001.0 + run_id,
            "protocol_response_at_ms": None,
            "finalized_at_ms": None,
            "latency_ms": None,
        },
        "transport": {
            "transport_family": "udp",
            "request_bytes_on_wire": 33.0,
            "response_bytes_on_wire": None,
        },
        "provenance": {
            "observed_on_wire": True,
            "capture_backend": "pcap",
            "capture_interface": "tap0",
            "timestamp_source": "pcap",
            "bytes_source": "pcap",
            "source_artifact_paths": [f"logs/mavlink/run_{run_id:04d}.jsonl"],
        },
        "outcome": None,
        "correlation": {
            "run_id": run_id,
            "episode_id": episode_id,
            "session_id": f"gcs_primary-{run_id:04d}",
            "transaction_id": f"txn-{run_id:04d}",
            "send_id": f"send-mavlink-{run_id}",
            "stream_id": "udp:14550",
            "stream_index": 0,
        },
        "evaluation_context": {
            "label": 0,
            "label_name": "benign",
            "attack_family": "none",
            "phase": "takeoff",
            "actor_id": "gcs_primary",
            "actor_role": "ops_primary",
            "actor_trust": 0.9,
        },
        "evidence": {
            "related_packet_count": 1,
            "observed_message_families": ["request"],
            "observed_message_stages": ["request"],
            "packet_timestamp_sources": ["pcap"],
            "packet_byte_sources": ["pcap"],
            "request_wire_observed": True,
            "response_wire_observed": False,
            "log_correlation_mode": "session_txn_id",
            "source_artifact_paths": [f"logs/mavlink/run_{run_id:04d}.jsonl"],
        },
        "native_state_snapshot": None,
        "native_fields": {
            "raw_frame_type": "COMMAND_LONG",
        },
    }


def write_protocol_bundle(
    *,
    root: Path,
    protocol_family: str,
    raw_transaction: dict[str, object],
    local_run_id: int,
    local_episode_id: int,
    seed: int,
    schedule_path: str,
    class_episode_counts: dict[str, int] | None = None,
    class_rows_per_episode_summary: dict[str, dict[str, object]] | None = None,
    episode_policy: dict[str, object] | None = None,
) -> None:
    packet = {
        "schema_version": RAW_PACKET_SCHEMA_VERSION,
        "record_kind": "raw_packet",
        "protocol_family": protocol_family,
        "packet_kind": "request",
        "run_id": local_run_id,
        "episode_id": local_episode_id,
    }
    transaction = {
        "run_id": local_run_id,
        "episode_id": local_episode_id,
        "txn_id": f"{protocol_family}-{local_run_id}-txn",
        "command": "cmdDisp.CMD_NO_OP" if protocol_family == "fprime" else "MAV_CMD_NAV_TAKEOFF",
        "service": "cmdDisp" if protocol_family == "fprime" else "command_long",
        "label": 0,
        "episode_kind": "benign",
    }
    write_jsonl(root / "data" / "packets.jsonl", [transaction])
    write_jsonl(root / "data" / "transactions.jsonl", [transaction])
    write_jsonl(root / "data" / "raw_packets.jsonl", [packet])
    write_jsonl(root / "data" / "raw_transactions.jsonl", [raw_transaction])
    write_json(
        root / "reports" / "generation_summary.json",
        {
            "seed": seed,
            "protocol_families": {protocol_family: 1},
            "independent_runs": {
                "class_episode_counts": dict(class_episode_counts or {}),
                "class_rows_per_episode_summary": dict(class_rows_per_episode_summary or {}),
                "episode_policy": dict(episode_policy or {}),
            },
            "per_run_reports": [
                {
                    "run_id": local_run_id,
                    "run_order": local_run_id,
                    "episode_id": local_episode_id,
                    "class_name": "benign",
                    "label": 0,
                    "rows": 1,
                    "schedule_path": schedule_path,
                }
            ],
        },
    )
    write_json(
        root / "reports" / "run_manifest.json",
        {
            "runtime_reset_policy": "fresh_runtime_per_episode",
            "run_count": 1,
            "class_episode_counts": dict(class_episode_counts or {}),
            "class_rows_per_episode_summary": dict(class_rows_per_episode_summary or {}),
            "episode_policy": dict(episode_policy or {}),
        },
    )


class MultiProtocolGenerationTests(unittest.TestCase):
    def test_build_class_overlap_report_tracks_nested_actor_context_values(self) -> None:
        rows = [
            {
                "actor_context": {"role": "ops_primary", "trust_class": "high"},
                "audit_context": {"label_name": "benign"},
            },
            {
                "actor_context": {"role": "ops_primary", "trust_class": "high"},
                "audit_context": {"label_name": "cyber"},
            },
            {
                "actor_context": {"role": "external", "trust_class": "low"},
                "audit_context": {"label_name": "cyber"},
            },
            {
                "actor_context": {"role": "external", "trust_class": "low"},
                "audit_context": {"label_name": "fault"},
            },
        ]

        role_report = build_class_overlap_report(
            rows,
            value_path="actor_context.role",
            family_key="canonical_actor_role",
            item_key="role",
            items_key="roles",
        )
        trust_report = build_class_overlap_report(
            rows,
            value_path="actor_context.trust_class",
            family_key="canonical_actor_trust_class",
            item_key="trust_class",
            items_key="trust_classes",
        )

        self.assertEqual(role_report["summary"]["values_shared_by_at_least_two_classes"], 2)
        self.assertEqual(role_report["summary"]["exclusive_values"], 0)
        self.assertEqual(trust_report["summary"]["values_shared_by_at_least_two_classes"], 2)
        self.assertEqual(trust_report["summary"]["max_dominant_class_share"], 0.5)

    def test_protocol_row_targets_mixed_assigns_both_protocols(self) -> None:
        targets = protocol_row_targets(protocol_mode="mixed", rows=11, mixed_fprime_ratio=0.6)
        self.assertEqual(targets["fprime"] + targets["mavlink"], 11)
        self.assertGreater(targets["fprime"], 0)
        self.assertGreater(targets["mavlink"], 0)

    def test_build_multi_protocol_run_manifest_records_protocol_mix(self) -> None:
        manifest = build_multi_protocol_run_manifest(
            protocol_mode="mixed",
            requested_rows=12,
            nominal_ratio=0.55,
            seed=7,
            mixed_fprime_ratio=0.5,
            history_group_key="run_id",
            history_reset_policy="fresh_runtime_per_episode",
            protocol_execution_order=["fprime", "mavlink"],
            protocol_row_targets_by_family={"fprime": 6, "mavlink": 6},
            runs=[
                {
                    "run_id": 0,
                    "run_order": 0,
                    "protocol_family": "fprime",
                    "class_name": "benign",
                    "label": 0,
                    "episode_id": 0,
                    "rows": 3,
                    "runtime_reset_policy": "fresh_runtime_per_episode",
                    "seed": 7,
                    "schedule_path": "/tmp/fprime_schedule.csv",
                },
                {
                    "run_id": 1,
                    "run_order": 1,
                    "protocol_family": "mavlink",
                    "class_name": "fault",
                    "label": 2,
                    "episode_id": 1,
                    "rows": 4,
                    "runtime_reset_policy": "fresh_runtime_per_episode",
                    "seed": 1007,
                    "schedule_path": "/tmp/mavlink_schedule.csv",
                },
            ],
        )

        self.assertEqual(manifest["schema_version"], MULTI_PROTOCOL_RUN_MANIFEST_SCHEMA_VERSION)
        self.assertEqual(manifest["protocol_mode"], "mixed")
        self.assertEqual(manifest["protocol_row_targets"], {"fprime": 6, "mavlink": 6})
        self.assertEqual(manifest["protocol_run_counts"], {"fprime": 1, "mavlink": 1})
        self.assertEqual(manifest["runtime_reset_policies"], ["fresh_runtime_per_episode"])
        self.assertEqual(manifest["runs"][1]["protocol_family"], "mavlink")
        self.assertEqual(manifest["runs"][1]["schedule_path"], "/tmp/mavlink_schedule.csv")

    def test_merge_protocol_bundles_rebuilds_mixed_dataset_with_unique_run_ids(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            fprime_root = tmp_path / "protocol_runs" / "fprime"
            mavlink_root = tmp_path / "protocol_runs" / "mavlink"
            output_dir = tmp_path / "mixed_output"

            write_protocol_bundle(
                root=fprime_root,
                protocol_family="fprime",
                raw_transaction=make_legacy_fprime_transaction(run_id=0, episode_id=0),
                local_run_id=0,
                local_episode_id=0,
                seed=7,
                schedule_path="/tmp/fprime_schedule.csv",
                class_episode_counts={"benign": 6, "cyber": 6, "fault": 5},
                class_rows_per_episode_summary={
                    "benign": {"episodes": 6, "avg_rows": 11.0},
                    "cyber": {"episodes": 6, "avg_rows": 5.83},
                    "fault": {"episodes": 5, "avg_rows": 3.8},
                },
                episode_policy={"development_run_episode_rows_by_class": {"benign": 13, "cyber": 6, "fault": 4}},
            )
            write_protocol_bundle(
                root=mavlink_root,
                protocol_family="mavlink",
                raw_transaction=make_mavlink_raw_transaction(run_id=0, episode_id=0),
                local_run_id=0,
                local_episode_id=0,
                seed=1007,
                schedule_path="/tmp/mavlink_schedule.csv",
                class_episode_counts={"benign": 4, "cyber": 4, "fault": 4},
                class_rows_per_episode_summary={
                    "benign": {"episodes": 4, "avg_rows": 6.0},
                    "cyber": {"episodes": 4, "avg_rows": 6.0},
                    "fault": {"episodes": 4, "avg_rows": 6.0},
                },
                episode_policy={"development_run_episode_rows_by_class": {"benign": 6, "cyber": 6, "fault": 6}},
            )

            bundles = [
                shared_generate_dataset.load_protocol_bundle(fprime_root, "fprime"),
                shared_generate_dataset.load_protocol_bundle(mavlink_root, "mavlink"),
            ]
            dataset_path = shared_generate_dataset.merge_protocol_bundles(
                bundles=bundles,
                protocol_mode="mixed",
                requested_rows=2,
                nominal_ratio=0.55,
                seed=7,
                mixed_fprime_ratio=0.5,
                protocol_targets={"fprime": 1, "mavlink": 1},
                output_dir=output_dir,
            )

            self.assertTrue(dataset_path.exists())
            canonical_rows = shared_generate_dataset.read_jsonl(output_dir / "data" / "canonical_command_rows.jsonl")
            dataset_rows = shared_generate_dataset.read_jsonl(output_dir / "data" / "dataset.jsonl")
            raw_transactions = shared_generate_dataset.read_jsonl(output_dir / "data" / "raw_transactions.jsonl")
            run_manifest = shared_generate_dataset.read_json(output_dir / "reports" / "run_manifest.json")
            generation_summary = shared_generate_dataset.read_json(output_dir / "reports" / "generation_summary.json")

            self.assertEqual(len(canonical_rows), 2)
            self.assertEqual(len(dataset_rows), 2)
            self.assertEqual({row["protocol_family"] for row in canonical_rows}, {"fprime", "mavlink"})
            self.assertEqual({int(row["run_id"]) for row in canonical_rows}, {0, 1})
            self.assertEqual(
                {int(dict(row["correlation"])["run_id"]) for row in raw_transactions},
                {0, 1},
            )
            self.assertTrue(
                all(float(dict(row["recent_behavior"])["command_rate_1m"]) == 1.0 for row in canonical_rows)
            )
            self.assertEqual(run_manifest["protocol_mode"], "mixed")
            self.assertEqual(run_manifest["protocol_run_counts"], {"fprime": 1, "mavlink": 1})
            self.assertEqual(generation_summary["protocol_mode"], "mixed")
            self.assertEqual(
                generation_summary["protocol_artifact_roots"],
                {
                    "fprime": str(fprime_root.resolve()),
                    "mavlink": str(mavlink_root.resolve()),
                },
            )
            self.assertEqual(
                generation_summary["per_protocol_episode_support"]["fprime"]["class_episode_counts"],
                {"benign": 6, "cyber": 6, "fault": 5},
            )
            self.assertEqual(
                generation_summary["per_protocol_episode_support"]["mavlink"]["class_episode_counts"],
                {"benign": 4, "cyber": 4, "fault": 4},
            )
            self.assertEqual(
                generation_summary["per_protocol_episode_support"]["fprime"]["class_rows_per_episode_summary"]["fault"]["episodes"],
                5,
            )

    def test_run_generate_routes_through_shared_generator(self) -> None:
        completed = mock.Mock()
        completed.returncode = 0
        with mock.patch("main.subprocess.run", return_value=completed) as run_mock:
            dataset_path = poster_main.run_generate(
                Path("/tmp/poster-artifacts"),
                24,
                0.55,
                7,
                protocol_mode="mixed",
                mixed_fprime_ratio=0.25,
            )

        self.assertEqual(dataset_path, Path("/tmp/poster-artifacts") / "data" / "dataset.jsonl")
        invoked = run_mock.call_args.args[0]
        self.assertIn(str(REPO_ROOT / "tools" / "shared" / "generate_dataset.py"), invoked)
        self.assertIn("--protocol-mode", invoked)
        self.assertIn("mixed", invoked)
        self.assertIn("--mixed-fprime-ratio", invoked)
        self.assertIn("0.25", invoked)

    def test_legacy_training_path_rejects_mixed_dataset_reports(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            dataset_path = tmp_path / "data" / "dataset.jsonl"
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            dataset_path.write_text("", encoding="utf-8")
            write_json(
                tmp_path / "reports" / "generation_summary.json",
                {
                    "protocol_mode": "mixed",
                    "protocol_families": {"fprime": 1, "mavlink": 1},
                },
            )

            with self.assertRaises(SystemExit) as exc:
                poster_main.assert_legacy_training_dataset_supported(dataset_path)
        self.assertIn("F´-only datasets", str(exc.exception))

    def test_legacy_generation_path_rejects_non_fprime_protocol_modes(self) -> None:
        with self.assertRaises(SystemExit) as exc:
            poster_main.assert_legacy_generation_protocol_mode("mixed")
        self.assertIn("--protocol-mode fprime", str(exc.exception))


if __name__ == "__main__":
    unittest.main()
