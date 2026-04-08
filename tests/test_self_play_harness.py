from __future__ import annotations

import json
import shutil
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest import mock

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.fprime_real import schedule_profiles as fprime_schedule_profiles
from tools.train import run_self_play
from tools.train.red_policy_model import schedule_row_to_canonical_row
from tools.train.poster_default import canonical_rows_to_training_rows


def _parse_hms(value: str) -> float:
    hours, minutes, seconds = [int(part) for part in value.split(":")]
    return float((hours * 3600 + minutes * 60 + seconds) * 1000)


def _make_raw_transaction(row: dict[str, object], *, index: int) -> dict[str, object]:
    meta = dict(row.get("meta", {}) or {})
    source_service = str(row["source_service"])
    target_service = str(row["target_service"])
    command = str(row["command"])
    service_name = command.split(".", 1)[0] if "." in command else command
    raw_arguments = row.get("arguments")
    session_id = f"fixture-ep-{int(meta.get('episode_id', 0)):04d}"
    transaction_id = f"fixture-{index:04d}-{source_service}"
    send_id = f"fixture-send-{index:04d}"
    submitted_at_ms = _parse_hms(str(row["time_of_day"]))
    label_value = int(meta.get("class_label", 0))
    label_name = str(meta.get("class_name", "benign"))
    is_benign = label_value == 0
    return {
        "schema_version": "raw_transaction.v1",
        "record_kind": "raw_transaction",
        "protocol_family": "fprime",
        "protocol_version": None,
        "platform_family": "spacecraft",
        "sender": {
            "logical_id": source_service,
            "role": meta.get("actor_role"),
            "trust_score": float(meta.get("actor_trust", 0.6)),
            "network_endpoint": None,
        },
        "target": {
            "logical_id": target_service,
            "role": None,
            "stream_id": f"{target_service}:fixture",
            "stream_index": 0,
            "network_endpoint": None,
        },
        "command": {
            "raw_name": command,
            "raw_identifier": {
                "service_name": service_name,
                "native_service_id": None,
                "native_command_id": None,
            },
            "raw_arguments": raw_arguments,
            "raw_argument_representation": raw_arguments,
        },
        "timing": {
            "submitted_at_ms": submitted_at_ms,
            "request_forwarded_at_ms": None,
            "protocol_response_at_ms": None,
            "finalized_at_ms": submitted_at_ms + 120.0,
            "latency_ms": 120.0,
        },
        "transport": {
            "transport_family": "tcp",
            "request_bytes_on_wire": 24.0,
            "response_bytes_on_wire": 48.0,
        },
        "provenance": {
            "observed_on_wire": True,
            "capture_backend": "unit_test",
            "capture_interface": "loopback",
            "timestamp_source": "synthetic_fixture",
            "bytes_source": "synthetic_fixture",
            "source_artifact_paths": [f"/tmp/fixture_schedule_{index:04d}.csv"],
        },
        "outcome": {
            "accepted": True,
            "executed_successfully": True,
            "timed_out": False,
            "raw_code": 0.0,
            "raw_reason": "completed" if is_benign else "completed_unsafe",
            "warning_count": 0.0,
            "error_count": 0.0,
            "response_direction_seen": True,
            "terminal_observed_on_wire": True,
            "raw_event_name": None,
        },
        "correlation": {
            "run_id": 0,
            "episode_id": int(meta.get("episode_id", 0)),
            "session_id": session_id,
            "transaction_id": transaction_id,
            "send_id": send_id,
            "stream_id": f"{target_service}:fixture",
            "stream_index": 0,
        },
        "evaluation_context": {
            "label": label_value,
            "label_name": label_name,
            "attack_family": meta.get("attack_family"),
            "phase": meta.get("phase"),
            "actor_id": source_service,
            "actor_role": meta.get("actor_role"),
            "actor_trust": float(meta.get("actor_trust", 0.6)),
        },
        "evidence": {
            "related_packet_count": 0,
            "observed_message_families": ["request", "response"],
            "observed_message_stages": ["request", "terminal_response"],
            "packet_timestamp_sources": ["synthetic_fixture"],
            "packet_byte_sources": ["synthetic_fixture"],
            "request_wire_observed": True,
            "response_wire_observed": True,
            "log_correlation_mode": "session_txn_id",
            "source_artifact_paths": [f"/tmp/fixture_schedule_{index:04d}.csv"],
        },
        "native_state_snapshot": None,
        "native_fields": {"fixture": True},
    }


def _write_jsonl(path: Path, rows: list[dict[str, object]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as handle:
        for row in rows:
            handle.write(json.dumps(row) + "\n")


def _build_fixture_dataset(
    root: Path,
    *,
    benign_target_rows: int = 2,
    cyber_target_rows: int = 4,
    episode_span: int = 2,
    include_run_ids: bool = False,
) -> Path:
    benign_rows = fprime_schedule_profiles.build_benign_rows(
        target_rows=benign_target_rows,
        seed=7,
        episode_span=episode_span,
    )
    cyber_rows = fprime_schedule_profiles.build_cyber_rows(
        target_rows=cyber_target_rows,
        seed=7,
        episode_offset=1,
        episode_span=episode_span,
    )
    schedule_rows = benign_rows + cyber_rows

    canonical_rows: list[dict[str, object]] = []
    raw_transactions: list[dict[str, object]] = []
    for index, row in enumerate(schedule_rows):
        raw_transaction = _make_raw_transaction(row, index=index)
        episode_id = int((row.get("meta") or {}).get("episode_id", 0))
        run_id = episode_id // 2 if include_run_ids else 0
        raw_transaction["correlation"]["run_id"] = run_id
        canonical_row = schedule_row_to_canonical_row(
            row,
            protocol_family="fprime",
            transaction_id=str(raw_transaction["correlation"]["transaction_id"]),
            send_id=str(raw_transaction["correlation"]["send_id"]),
            window_class=str(dict(row.get("meta", {}) or {}).get("phase", "unspecified")),
        )
        if include_run_ids:
            canonical_row["run_id"] = run_id
        canonical_rows.append(canonical_row)
        raw_transactions.append(raw_transaction)

    dataset_rows = canonical_rows_to_training_rows(canonical_rows)
    dataset_path = root / "data" / "dataset.jsonl"
    _write_jsonl(dataset_path, dataset_rows)
    _write_jsonl(root / "data" / "canonical_command_rows.jsonl", canonical_rows)
    _write_jsonl(root / "data" / "raw_transactions.jsonl", raw_transactions)
    (root / "reports").mkdir(parents=True, exist_ok=True)
    (root / "reports" / "generation_summary.json").write_text(
        json.dumps(
            {
                "protocol_mode": "fprime",
                "rows": len(dataset_rows),
                "canonical_rows": len(canonical_rows),
                "run_count": len(sorted({row.get("run_id", 0) for row in canonical_rows})) if include_run_ids else 1,
            },
            indent=2,
        ),
        encoding="utf-8",
    )
    return dataset_path


def _fake_blue_report() -> dict[str, object]:
    return {
        "deployment_ready": True,
        "deployment_blocked_reason": None,
        "metrics": {
            "model_only": {
                "neural_net": {
                    "multiclass_metrics": {"macro_f1": 0.61, "accuracy": 0.66},
                    "cyber_binary_metrics": {"f1": 0.58},
                    "anomaly_binary_metrics": {"f1": 0.64},
                }
            }
        },
    }


class SelfPlayHarnessTests(unittest.TestCase):
    def _fake_run_training(self, dataset_path: Path, output_dir: Path, seed: int, make_plots: bool, blue_feature_policy_name=None):
        del dataset_path, seed, make_plots, blue_feature_policy_name
        report = _fake_blue_report()
        report_dir = output_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        model_dst = output_dir / "models"
        shutil.copytree(REPO_ROOT / "deployments" / "DetectorRB3" / "config", model_dst)
        return report

    def _fake_run_training_blocked(self, dataset_path: Path, output_dir: Path, seed: int, make_plots: bool, blue_feature_policy_name=None):
        del dataset_path, seed, make_plots, blue_feature_policy_name
        report = _fake_blue_report()
        report["deployment_ready"] = False
        report["deployment_blocked_reason"] = "generalization_metrics_below_threshold"
        research_model_dir = output_dir / "research_models"
        shutil.copytree(REPO_ROOT / "deployments" / "DetectorRB3" / "config", research_model_dir)
        report["analysis_runtime_bundle"] = {
            "artifact_dir": str(research_model_dir.resolve()),
            "manifest_path": str((research_model_dir / "bundle_manifest.json").resolve()),
            "model_path": str((research_model_dir / "blue_model.json").resolve()),
            "deployment_ready": False,
            "deployment_blocked_reason": "generalization_metrics_below_threshold",
        }
        report_dir = output_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        return report

    def test_self_play_smoke_round_writes_round_logs_and_checkpoints(self) -> None:
        with TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "fixture_dataset"
            dataset_path = _build_fixture_dataset(dataset_root)
            output_dir = Path(tmpdir) / "self_play"
            with mock.patch("tools.train.run_self_play.run_training", side_effect=self._fake_run_training):
                report = run_self_play.run_self_play(
                    output_dir=output_dir,
                    rounds=1,
                    seed=7,
                    rows=24,
                    nominal_ratio=0.55,
                    protocol_mode="fprime",
                    mixed_fprime_ratio=0.5,
                    dataset_sources=[dataset_path],
                    initial_blue_model_dir=REPO_ROOT / "deployments" / "DetectorRB3" / "config",
                    red_warmstart_rows_per_protocol=8,
                    red_replay_buffer_limit=32,
                    max_history_entries=4,
                    make_plots=False,
                )

            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["rounds_completed"], 1)
            self.assertTrue((output_dir / "self_play_state.json").exists())
            self.assertTrue((output_dir / "self_play_report.json").exists())
            self.assertTrue((output_dir / "rounds" / "round_0001" / "round_summary.json").exists())
            self.assertTrue((output_dir / "rounds" / "round_0001" / "reward" / "reward_summary.json").exists())
            self.assertTrue((output_dir / "checkpoints" / "blue" / "round_0001" / "artifacts" / "bundle_manifest.json").exists())
            self.assertTrue((output_dir / "checkpoints" / "red" / "round_0001" / "artifacts" / "red_policy_model.json").exists())

            round_summary = json.loads((output_dir / "rounds" / "round_0001" / "round_summary.json").read_text(encoding="utf-8"))
            self.assertEqual(round_summary["mode"], run_self_play.SELF_PLAY_MODE)
            self.assertEqual(round_summary["reward_summary"]["example_count"], 6)
            self.assertEqual(
                round_summary["replay_examples"]["raw_transaction_match_modes"],
                {"transaction_send": 6},
            )
            self.assertEqual(round_summary["reward_summary"]["sandbox_violation_counts"], {})
            self.assertIn("macro_f1", round_summary["blue_update"]["summary"])

    def test_self_play_resumes_from_existing_state(self) -> None:
        with TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "fixture_dataset"
            dataset_path = _build_fixture_dataset(dataset_root)
            output_dir = Path(tmpdir) / "self_play"
            with mock.patch("tools.train.run_self_play.run_training", side_effect=self._fake_run_training):
                first_report = run_self_play.run_self_play(
                    output_dir=output_dir,
                    rounds=1,
                    seed=7,
                    rows=24,
                    nominal_ratio=0.55,
                    protocol_mode="fprime",
                    mixed_fprime_ratio=0.5,
                    dataset_sources=[dataset_path],
                    initial_blue_model_dir=REPO_ROOT / "deployments" / "DetectorRB3" / "config",
                    red_warmstart_rows_per_protocol=8,
                    red_replay_buffer_limit=32,
                    max_history_entries=4,
                    make_plots=False,
                )
                second_report = run_self_play.run_self_play(
                    output_dir=output_dir,
                    rounds=2,
                    seed=7,
                    rows=24,
                    nominal_ratio=0.55,
                    protocol_mode="fprime",
                    mixed_fprime_ratio=0.5,
                    dataset_sources=[dataset_path],
                    initial_blue_model_dir=REPO_ROOT / "deployments" / "DetectorRB3" / "config",
                    red_warmstart_rows_per_protocol=8,
                    red_replay_buffer_limit=32,
                    max_history_entries=4,
                    make_plots=False,
                )

            self.assertEqual(first_report["rounds_completed"], 1)
            self.assertEqual(second_report["rounds_completed"], 2)
            self.assertTrue((output_dir / "rounds" / "round_0002" / "round_summary.json").exists())
            state = json.loads((output_dir / "self_play_state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["rounds_completed"], 2)
            self.assertEqual(state["latest_blue_checkpoint"]["checkpoint_id"], "round_0002")
            self.assertEqual(state["latest_red_checkpoint"]["checkpoint_id"], "round_0002")

    def test_self_play_uses_analysis_runtime_bundle_when_blue_update_is_non_deployable(self) -> None:
        with TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "fixture_dataset"
            dataset_path = _build_fixture_dataset(dataset_root)
            output_dir = Path(tmpdir) / "self_play"
            with mock.patch("tools.train.run_self_play.run_training", side_effect=self._fake_run_training_blocked):
                report = run_self_play.run_self_play(
                    output_dir=output_dir,
                    rounds=1,
                    seed=7,
                    rows=24,
                    nominal_ratio=0.55,
                    protocol_mode="fprime",
                    mixed_fprime_ratio=0.5,
                    dataset_sources=[dataset_path],
                    initial_blue_model_dir=REPO_ROOT / "deployments" / "DetectorRB3" / "config",
                    red_warmstart_rows_per_protocol=8,
                    red_replay_buffer_limit=32,
                    max_history_entries=4,
                    make_plots=False,
                )

            self.assertEqual(report["rounds_completed"], 1)
            checkpoint_dir = output_dir / "checkpoints" / "blue" / "round_0001" / "artifacts"
            self.assertTrue((checkpoint_dir / "bundle_manifest.json").exists())
            self.assertTrue((checkpoint_dir / "blue_model.json").exists())
            state = json.loads((output_dir / "self_play_state.json").read_text(encoding="utf-8"))
            self.assertEqual(state["latest_blue_checkpoint"]["metadata"]["runtime_bundle_source"], "analysis_runtime_bundle")
            self.assertFalse(state["rounds"][0]["blue_update"]["summary"]["deployment_ready"])

    def test_build_replay_examples_supports_non_bootstrap_history_when_replay_keys_are_fallback_aligned(self) -> None:
        with TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "fixture_dataset"
            dataset_path = _build_fixture_dataset(
                dataset_root,
                benign_target_rows=24,
                cyber_target_rows=24,
                episode_span=6,
                include_run_ids=True,
            )

            examples, summary = run_self_play.build_replay_examples_from_dataset(
                dataset_path,
                action_space=run_self_play.load_red_action_space(),
                transcript_budget=run_self_play.load_red_context_budget(),
                max_history_entries=4,
            )

            self.assertGreater(len(examples), 30)
            self.assertEqual(summary["raw_transaction_match_rate"], 1.0)
            self.assertTrue(any(example["action"]["timing_bucket"] != "bootstrap" for example in examples))
            self.assertTrue(
                all(
                    example["raw_transaction"]["correlation"]["session_id"]
                    == example["canonical_row"]["audit_context"]["session_id"]
                    for example in examples
                )
            )


if __name__ == "__main__":
    unittest.main()
