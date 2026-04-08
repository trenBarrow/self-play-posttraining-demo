from __future__ import annotations

import contextlib
import io
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
from tools.train.poster_default import canonical_rows_to_training_rows
from tools.train.red_policy_model import schedule_row_to_canonical_row
from tools.train.red_transcript import RedTranscriptBuildError, load_red_context_budget


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


def _build_fixture_dataset(root: Path) -> Path:
    benign_rows = fprime_schedule_profiles.build_benign_rows(
        target_rows=2,
        seed=7,
        episode_span=2,
    )
    cyber_rows = fprime_schedule_profiles.build_cyber_rows(
        target_rows=4,
        seed=7,
        episode_offset=1,
        episode_span=2,
    )
    schedule_rows = benign_rows + cyber_rows

    canonical_rows: list[dict[str, object]] = []
    raw_transactions: list[dict[str, object]] = []
    for index, row in enumerate(schedule_rows):
        raw_transaction = _make_raw_transaction(row, index=index)
        canonical_row = schedule_row_to_canonical_row(
            row,
            protocol_family="fprime",
            transaction_id=str(raw_transaction["correlation"]["transaction_id"]),
            send_id=str(raw_transaction["correlation"]["send_id"]),
            window_class=str(dict(row.get("meta", {}) or {}).get("phase", "unspecified")),
        )
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
                "run_count": 1,
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


class SelfPlaySmokeTests(unittest.TestCase):
    def _fake_run_training(self, dataset_path: Path, output_dir: Path, seed: int, make_plots: bool, blue_feature_policy_name=None):
        del dataset_path, seed, make_plots, blue_feature_policy_name
        report = _fake_blue_report()
        report_dir = output_dir / "reports"
        report_dir.mkdir(parents=True, exist_ok=True)
        (report_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
        model_dst = output_dir / "models"
        shutil.copytree(REPO_ROOT / "deployments" / "DetectorRB3" / "config", model_dst)
        return report

    def test_self_play_cli_smoke_writes_round_outputs_and_bounded_transcripts(self) -> None:
        with TemporaryDirectory() as tmpdir:
            dataset_path = _build_fixture_dataset(Path(tmpdir) / "fixture_dataset")
            output_dir = Path(tmpdir) / "self_play"
            argv = [
                "run_self_play.py",
                "--dataset",
                str(dataset_path),
                "--rounds",
                "1",
                "--seed",
                "7",
                "--max-history-entries",
                "4",
                "--output-dir",
                str(output_dir),
            ]

            with mock.patch("tools.train.run_self_play.run_training", side_effect=self._fake_run_training):
                with mock.patch.object(sys, "argv", argv):
                    with contextlib.redirect_stdout(io.StringIO()):
                        run_self_play.main()

            report = json.loads((output_dir / "self_play_report.json").read_text(encoding="utf-8"))
            rewarded_examples_path = output_dir / "rounds" / "round_0001" / "reward" / "rewarded_examples.jsonl"
            with rewarded_examples_path.open(encoding="utf-8") as handle:
                rewarded_examples = [json.loads(line) for line in handle if line.strip()]

            self.assertEqual(report["status"], "completed")
            self.assertEqual(report["rounds_completed"], 1)
            self.assertTrue((output_dir / "rounds" / "round_0001" / "round_summary.json").exists())
            self.assertTrue((output_dir / "checkpoints" / "blue" / "round_0001" / "artifacts" / "bundle_manifest.json").exists())
            self.assertTrue((output_dir / "checkpoints" / "red" / "round_0001" / "artifacts" / "red_policy_model.json").exists())
            self.assertTrue(rewarded_examples)
            self.assertTrue(all(example["transcript"]["budget"]["max_history_entries"] == 4 for example in rewarded_examples))

    def test_self_play_rejects_history_budget_above_configured_limit(self) -> None:
        with TemporaryDirectory() as tmpdir:
            dataset_path = _build_fixture_dataset(Path(tmpdir) / "fixture_dataset")
            configured_limit = int(load_red_context_budget()["limits"]["max_history_entries"])

            with mock.patch("tools.train.run_self_play.run_training", side_effect=self._fake_run_training):
                with self.assertRaises(RedTranscriptBuildError) as exc:
                    run_self_play.run_self_play(
                        output_dir=Path(tmpdir) / "self_play",
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
                        max_history_entries=configured_limit + 1,
                        make_plots=False,
                    )

        self.assertIn("exceeds configured transcript budget", str(exc.exception))


if __name__ == "__main__":
    unittest.main()
