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

from tools.train import run_self_play
from tools.train.evaluate_red_vs_blue import (
    LEARNED_ADVERSARY_KIND,
    STATIC_ADVERSARY_KIND,
    build_breakdown_rows,
    evaluate_learned_red_policy_retrieval,
    evaluate_red_vs_blue,
    summarize_evaluation_attempts,
)
from tools.train.red_policy_model import RED_POLICY_MODEL_SCHEMA_VERSION, load_red_action_space
from tools.train.red_transcript import RED_TRANSCRIPT_SCHEMA_VERSION, load_red_context_budget


def _transcript(*, protocol_family: str, platform_family: str, mission_phase: str, window_class: str, included_history_count: int) -> dict[str, object]:
    budget = load_red_context_budget()
    configured_max_history_entries = int(budget["limits"]["max_history_entries"])
    return {
        "schema_version": RED_TRANSCRIPT_SCHEMA_VERSION,
        "record_kind": "red_command_transcript",
        "budget": {
            "max_history_entries": int(
                included_history_count if included_history_count > 0 else configured_max_history_entries
            ),
            "token_id_modulo": 65535,
        },
        "included_history_count": int(included_history_count),
        "flattened_token_ids": [],
        "events": [],
        "header": {
            "protocol_family": protocol_family,
            "platform_family": platform_family,
            "mission_phase": mission_phase,
            "window_class": window_class,
        },
    }


def _rewarded_record(
    *,
    protocol_family: str,
    platform_family: str,
    mission_phase: str,
    window_class: str,
    command_family: str,
    timing_bucket: str,
    identity_bucket: str,
    reward_case: str,
    reward: float,
    unsafe_label: bool,
    unsafe_detected: bool,
    actor_id: str = "ops_a2",
    target_id: str = "fprime_a",
    sequence_index: int = 0,
    group_key: str = "group",
) -> dict[str, object]:
    return {
        "schema_version": "red_self_play_rewarded_example.v1",
        "record_kind": "red_self_play_rewarded_example",
        "transcript": _transcript(
            protocol_family=protocol_family,
            platform_family=platform_family,
            mission_phase=mission_phase,
            window_class=window_class,
            included_history_count=1,
        ),
        "current_context": {
            "protocol_family": protocol_family,
            "platform_family": platform_family,
            "mission_phase": mission_phase,
            "window_class": window_class,
        },
        "action": {
            "command_family": command_family,
            "timing_bucket": timing_bucket,
            "identity_bucket": identity_bucket,
        },
        "metadata": {
            "protocol_family": protocol_family,
            "platform_family": platform_family,
            "actor_id": actor_id,
            "target_id": target_id,
            "group_key": group_key,
            "sequence_index": sequence_index,
            "raw_command_name": f"{command_family}.fixture",
        },
        "blue_feedback": {
            "unsafe_risk": 0.9 if unsafe_detected else 0.1,
            "predicted_class": "cyber" if unsafe_detected else "benign",
            "detector_reason": "unsafe_threshold" if unsafe_detected else "benign_baseline",
        },
        "reward_result": {
            "schema_version": "red_reward_result.v1",
            "record_kind": "red_reward_result",
            "reward": reward,
            "reward_case": reward_case,
            "sandbox": {
                "allowed": True,
                "violation_reasons": [],
            },
            "environment_feedback": {
                "unsafe_label": unsafe_label,
                "accepted": True,
                "executed_successfully": True,
            },
            "blue_feedback": {
                "unsafe_detected": unsafe_detected,
            },
        },
    }


def _attempt(
    *,
    current_context: dict[str, object],
    action: dict[str, object],
    matched_rewarded_record: dict[str, object] | None,
    actor_id: str,
    target_id: str,
    sequence_index: int,
) -> dict[str, object]:
    return {
        "current_context": dict(current_context),
        "action": dict(action),
        "metadata": {
            "actor_id": actor_id,
            "target_id": target_id,
            "group_key": f"{actor_id}:{sequence_index}",
            "sequence_index": sequence_index,
        },
        "matched": matched_rewarded_record is not None,
        "matched_rewarded_record": None if matched_rewarded_record is None else dict(matched_rewarded_record),
        "retrieval": {
            "match_strategy": "unit_test",
            "candidate_count": 0 if matched_rewarded_record is None else 1,
        },
    }


def _constant_red_model_payload(
    *,
    command_family: str,
    timing_bucket: str,
    identity_bucket: str,
) -> dict[str, object]:
    action_space = load_red_action_space()
    transcript_budget = load_red_context_budget()

    def head_payload(head_name: str, allowed_values: list[str], selected_value: str) -> dict[str, object]:
        selected_index = allowed_values.index(selected_value)
        return {
            "model_type": "constant",
            "model_name": "constant_action_head",
            "feature_tier": "bounded_red_policy_context",
            "feature_names": [],
            "class_labels": [selected_index],
            "selected_label": selected_index,
            "action_head": head_name,
            "allowed_values": allowed_values,
        }

    return {
        "schema_version": RED_POLICY_MODEL_SCHEMA_VERSION,
        "record_kind": "red_policy_model",
        "model_family": "bounded_red_policy_mlp_v1",
        "feature_tier": "bounded_red_policy_context",
        "feature_names": [],
        "architecture": {},
        "training_config": {},
        "evaluation": {},
        "warm_start": {},
        "action_space": action_space,
        "transcript_budget": transcript_budget,
        "heads": {
            "command_family": head_payload(
                "command_family",
                list(action_space["command_family_values"]),
                command_family,
            ),
            "timing_bucket": head_payload(
                "timing_bucket",
                list(action_space["timing_bucket_values"]),
                timing_bucket,
            ),
            "identity_bucket": head_payload(
                "identity_bucket",
                list(action_space["identity_bucket_values"]),
                identity_bucket,
            ),
        },
    }


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


class EvaluateRedVsBlueTests(unittest.TestCase):
    def test_summarize_evaluation_attempts_and_breakdowns_capture_coverage(self) -> None:
        matched_success = _rewarded_record(
            protocol_family="fprime",
            platform_family="spacecraft",
            mission_phase="uplink",
            window_class="uplink",
            command_family="persistent_configuration",
            timing_bucket="steady",
            identity_bucket="shared_identity",
            reward_case="unsafe_executed_undetected",
            reward=1.0,
            unsafe_label=True,
            unsafe_detected=False,
            actor_id="ops_a2",
            sequence_index=0,
            group_key="fprime:0:ops_a2",
        )
        matched_detected = _rewarded_record(
            protocol_family="mavlink",
            platform_family="multirotor",
            mission_phase="maintenance",
            window_class="maintenance",
            command_family="transient_control",
            timing_bucket="steady",
            identity_bucket="shared_identity",
            reward_case="unsafe_executed_detected",
            reward=0.2,
            unsafe_label=True,
            unsafe_detected=True,
            actor_id="ops_secondary",
            target_id="vehicle_1",
            sequence_index=1,
            group_key="mavlink:0:ops_secondary",
        )
        attempts = [
            _attempt(
                current_context=dict(matched_success["current_context"]),
                action=dict(matched_success["action"]),
                matched_rewarded_record=matched_success,
                actor_id="ops_a2",
                target_id="fprime_a",
                sequence_index=0,
            ),
            _attempt(
                current_context=dict(matched_detected["current_context"]),
                action=dict(matched_detected["action"]),
                matched_rewarded_record=matched_detected,
                actor_id="ops_secondary",
                target_id="vehicle_1",
                sequence_index=1,
            ),
            _attempt(
                current_context={
                    "protocol_family": "mavlink",
                    "platform_family": "multirotor",
                    "mission_phase": "downlink",
                    "window_class": "downlink",
                },
                action={
                    "command_family": "communications_control",
                    "timing_bucket": "steady",
                    "identity_bucket": "shared_identity",
                },
                matched_rewarded_record=None,
                actor_id="ops_secondary",
                target_id="vehicle_1",
                sequence_index=2,
            ),
        ]

        summary = summarize_evaluation_attempts(attempts)
        self.assertEqual(summary["source_example_count"], 3)
        self.assertEqual(summary["matched_example_count"], 2)
        self.assertEqual(summary["unmatched_prediction_count"], 1)
        self.assertAlmostEqual(summary["retrieval_coverage_rate"], 2.0 / 3.0)
        self.assertEqual(summary["adversary_success_count"], 1)
        self.assertAlmostEqual(summary["adversary_success_rate"], 0.5)
        self.assertAlmostEqual(summary["coverage_adjusted_adversary_success_rate"], 1.0 / 3.0)
        self.assertAlmostEqual(float(summary["blue_recall_under_attack"]), 0.5)
        self.assertAlmostEqual(float(summary["blue_precision_under_attack"]), 1.0)

        breakdown_rows = build_breakdown_rows(
            evaluation_id="evaluation_0001",
            dataset_entry={"dataset_id": "dataset_01", "dataset_label": "fixture", "dataset_path": "/tmp/dataset.jsonl"},
            blue_entry={"blue_id": "blue_01", "blue_label": "blue", "blue_path": "/tmp/blue"},
            red_entry=None,
            adversary_kind=STATIC_ADVERSARY_KIND,
            transcript_length=4,
            attempts=attempts,
        )
        protocol_values = {
            row["breakdown_value"]
            for row in breakdown_rows
            if row["breakdown_field"] == "protocol_family"
        }
        command_values = {
            row["breakdown_value"]
            for row in breakdown_rows
            if row["breakdown_field"] == "command_family"
        }
        window_values = {
            row["breakdown_value"]
            for row in breakdown_rows
            if row["breakdown_field"] == "window_class"
        }
        self.assertEqual(protocol_values, {"fprime", "mavlink"})
        self.assertEqual(command_values, {"communications_control", "persistent_configuration", "transient_control"})
        self.assertEqual(window_values, {"downlink", "maintenance", "uplink"})

    def test_evaluate_learned_red_policy_retrieval_reports_alignment_and_coverage(self) -> None:
        source_examples = [
            {
                "transcript": _transcript(
                    protocol_family="fprime",
                    platform_family="spacecraft",
                    mission_phase="uplink",
                    window_class="uplink",
                    included_history_count=0,
                ),
                "current_context": {
                    "protocol_family": "fprime",
                    "platform_family": "spacecraft",
                    "mission_phase": "uplink",
                    "window_class": "uplink",
                },
                "action": {
                    "command_family": "persistent_configuration",
                    "timing_bucket": "steady",
                    "identity_bucket": "shared_identity",
                },
                "metadata": {
                    "actor_id": "ops_a2",
                    "target_id": "fprime_a",
                    "group_key": "fprime:0:ops_a2",
                    "sequence_index": 0,
                },
            },
            {
                "transcript": _transcript(
                    protocol_family="fprime",
                    platform_family="spacecraft",
                    mission_phase="uplink",
                    window_class="uplink",
                    included_history_count=1,
                ),
                "current_context": {
                    "protocol_family": "fprime",
                    "platform_family": "spacecraft",
                    "mission_phase": "uplink",
                    "window_class": "uplink",
                },
                "action": {
                    "command_family": "transient_control",
                    "timing_bucket": "steady",
                    "identity_bucket": "shared_identity",
                },
                "metadata": {
                    "actor_id": "ops_a2",
                    "target_id": "fprime_a",
                    "group_key": "fprime:0:ops_a2",
                    "sequence_index": 1,
                },
            },
            {
                "transcript": _transcript(
                    protocol_family="mavlink",
                    platform_family="multirotor",
                    mission_phase="downlink",
                    window_class="downlink",
                    included_history_count=0,
                ),
                "current_context": {
                    "protocol_family": "mavlink",
                    "platform_family": "multirotor",
                    "mission_phase": "downlink",
                    "window_class": "downlink",
                },
                "action": {
                    "command_family": "persistent_configuration",
                    "timing_bucket": "steady",
                    "identity_bucket": "shared_identity",
                },
                "metadata": {
                    "actor_id": "ops_secondary",
                    "target_id": "vehicle_1",
                    "group_key": "mavlink:0:ops_secondary",
                    "sequence_index": 0,
                },
            },
        ]
        rewarded_library = [
            _rewarded_record(
                protocol_family="fprime",
                platform_family="spacecraft",
                mission_phase="uplink",
                window_class="uplink",
                command_family="persistent_configuration",
                timing_bucket="steady",
                identity_bucket="shared_identity",
                reward_case="unsafe_executed_undetected",
                reward=1.0,
                unsafe_label=True,
                unsafe_detected=False,
                actor_id="ops_a2",
                sequence_index=2,
                group_key="fprime:0:ops_a2",
            ),
            _rewarded_record(
                protocol_family="fprime",
                platform_family="spacecraft",
                mission_phase="uplink",
                window_class="uplink",
                command_family="persistent_configuration",
                timing_bucket="steady",
                identity_bucket="shared_identity",
                reward_case="unsafe_executed_detected",
                reward=0.2,
                unsafe_label=True,
                unsafe_detected=True,
                actor_id="ops_a2",
                sequence_index=3,
                group_key="fprime:1:ops_a2",
            ),
        ]

        with TemporaryDirectory() as tmpdir:
            model_dir = Path(tmpdir) / "red_constant"
            _write_json(
                model_dir / "red_policy_model.json",
                _constant_red_model_payload(
                    command_family="persistent_configuration",
                    timing_bucket="steady",
                    identity_bucket="shared_identity",
                ),
            )
            attempts, retrieval_summary = evaluate_learned_red_policy_retrieval(
                model_dir / "red_policy_model.json",
                source_examples,
                rewarded_library,
            )

        self.assertEqual(len(attempts), 3)
        self.assertEqual(retrieval_summary["matched_example_count"], 2)
        self.assertEqual(retrieval_summary["unmatched_prediction_count"], 1)
        self.assertAlmostEqual(retrieval_summary["retrieval_coverage_rate"], 2.0 / 3.0)
        alignment = retrieval_summary["alignment"]
        self.assertAlmostEqual(alignment["joint_exact_match_accuracy"], 2.0 / 3.0)
        self.assertAlmostEqual(alignment["head_accuracy"]["command_family"], 2.0 / 3.0)
        self.assertAlmostEqual(alignment["head_accuracy"]["timing_bucket"], 1.0)
        self.assertAlmostEqual(alignment["head_accuracy"]["identity_bucket"], 1.0)

        summary = summarize_evaluation_attempts(attempts)
        self.assertEqual(summary["adversary_success_count"], 1)
        self.assertAlmostEqual(summary["coverage_adjusted_adversary_success_rate"], 1.0 / 3.0)

    def test_evaluate_red_vs_blue_discovers_self_play_outputs_and_writes_reports(self) -> None:
        from tests.test_self_play_harness import _build_fixture_dataset

        with TemporaryDirectory() as tmpdir:
            dataset_root = Path(tmpdir) / "fixture_dataset"
            dataset_path = _build_fixture_dataset(dataset_root)
            self_play_dir = Path(tmpdir) / "self_play_fixture"
            evaluation_dir = Path(tmpdir) / "red_blue_eval"

            def fake_run_training(dataset_path: Path, output_dir: Path, seed: int, make_plots: bool, blue_feature_policy_name=None):
                del dataset_path, seed, make_plots, blue_feature_policy_name
                report = {
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
                report_dir = output_dir / "reports"
                report_dir.mkdir(parents=True, exist_ok=True)
                (report_dir / "metrics.json").write_text(json.dumps(report, indent=2), encoding="utf-8")
                shutil.copytree(REPO_ROOT / "deployments" / "DetectorRB3" / "config", output_dir / "models")
                return report

            with mock.patch("tools.train.run_self_play.run_training", side_effect=fake_run_training):
                run_self_play.run_self_play(
                    output_dir=self_play_dir,
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

            report = evaluate_red_vs_blue(
                dataset_paths=[],
                output_dir=evaluation_dir,
                self_play_output_dirs=[self_play_dir],
                transcript_lengths=[4],
            )

            report_path = evaluation_dir / "reports" / "red_blue_evaluation.json"
            summary_path = evaluation_dir / "reports" / "red_blue_evaluation_summary.txt"
            self.assertTrue(report_path.exists())
            self.assertTrue(summary_path.exists())
            self.assertGreaterEqual(len(report["datasets"]), 1)
            self.assertGreaterEqual(len(report["blue_variants"]), 2)
            self.assertGreaterEqual(len(report["red_variants"]), 2)
            self.assertTrue(any(row["adversary_kind"] == STATIC_ADVERSARY_KIND for row in report["summary_rows"]))
            self.assertTrue(any(row["adversary_kind"] == LEARNED_ADVERSARY_KIND for row in report["summary_rows"]))
            self.assertGreater(len(report["comparison_rows"]), 0)


if __name__ == "__main__":
    unittest.main()
