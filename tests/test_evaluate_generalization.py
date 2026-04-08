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
    MODEL_ONLY_NAMESPACE,
    POSTER_DEFAULT_TRAINING_PATH_NAME,
    SELECTION_METRIC_CLASS_MACRO_F1,
    SELECTION_METRIC_MIN_PER_CLASS_RECALL,
    SELECTION_METRIC_MODEL_CYBER_F1,
    aggregate_evaluation_runs,
)
from tools.train.evaluate_generalization import (
    build_evaluation_matrix,
    evaluate_generalization,
    render_evaluation_matrix_summary,
)


def baseline_stub(*, best_metric_value: float, near_perfect: bool, not_applicable: bool = False, reason: str | None = None) -> dict[str, object]:
    return {
        "feature_names": ["protocol_family", "raw_command_name"],
        "best_metric_path": "model_only.multiclass_metrics.macro_f1",
        "best_metric_value": float(best_metric_value),
        "near_perfect": bool(near_perfect),
        "near_perfect_threshold": 0.95,
        "not_applicable": bool(not_applicable),
        **({"reason": str(reason)} if reason else {}),
    }


def model_metrics(
    *,
    accuracy: float,
    macro_f1: float,
    benign_recall: float,
    cyber_recall: float,
    fault_recall: float,
    cyber_f1: float,
    anomaly_f1: float,
) -> dict[str, object]:
    return {
        MODEL_ONLY_NAMESPACE: {
            "multiclass_metrics": {
                "accuracy": float(accuracy),
                "macro_f1": float(macro_f1),
                "per_class": {
                    "benign": {"recall": float(benign_recall), "support": 3},
                    "cyber": {"recall": float(cyber_recall), "support": 3},
                    "fault": {"recall": float(fault_recall), "support": 3},
                },
            },
            "cyber_binary_metrics": {
                "f1": float(cyber_f1),
            },
            "anomaly_binary_metrics": {
                "f1": float(anomaly_f1),
            },
        }
    }


def evaluation_run(
    *,
    macro_f1: float,
    cyber_f1: float,
    anomaly_f1: float,
    heldout_value: str | None = None,
) -> dict[str, object]:
    payload: dict[str, object] = {
        "models": {
            "neural_net": model_metrics(
                accuracy=macro_f1 + 0.05,
                macro_f1=macro_f1,
                benign_recall=max(0.0, macro_f1 - 0.04),
                cyber_recall=max(0.0, macro_f1 - 0.02),
                fault_recall=max(0.0, macro_f1 - 0.01),
                cyber_f1=cyber_f1,
                anomaly_f1=anomaly_f1,
            )
        },
        "thresholds": {
            "neural_net": {
                MODEL_ONLY_NAMESPACE: {
                    "cyber": 0.5,
                    "anomaly": 0.5,
                }
            }
        },
        "simple_command_baseline": baseline_stub(best_metric_value=0.56, near_perfect=False),
        "request_only_baseline": baseline_stub(best_metric_value=0.58, near_perfect=False),
        "outcome_only_baseline": baseline_stub(best_metric_value=0.0, near_perfect=False, not_applicable=True, reason="poster_path"),
        "protocol_only_baseline": baseline_stub(best_metric_value=0.62, near_perfect=False),
        "raw_protocol_shortcuts_baseline": baseline_stub(best_metric_value=0.64, near_perfect=False),
    }
    if heldout_value is not None:
        payload["heldout_value"] = heldout_value
        payload["family_field"] = "protocol_family"
    return payload


def grouped_report() -> dict[str, object]:
    runs = [
        {"seed": 7, "fold_index": 0, **evaluation_run(macro_f1=0.71, cyber_f1=0.69, anomaly_f1=0.73)},
        {"seed": 8, "fold_index": 0, **evaluation_run(macro_f1=0.75, cyber_f1=0.72, anomaly_f1=0.77)},
    ]
    return {
        "group_key": "run_id",
        "seeds": [7, 8],
        "fold_count": 1,
        "total_runs": len(runs),
        "runs": runs,
        "aggregate": aggregate_evaluation_runs(runs),
    }


def holdout_report(name: str, family_field: str, values: dict[str, list[tuple[float, float, float]]]) -> dict[str, object]:
    runs: list[dict[str, object]] = []
    seeds = [7, 8]
    for seed_index, seed in enumerate(seeds):
        for heldout_value, metrics_by_seed in values.items():
            macro_f1, cyber_f1, anomaly_f1 = metrics_by_seed[seed_index]
            runs.append(
                {
                    "seed": seed,
                    "heldout_value": heldout_value,
                    "family_field": family_field,
                    "split_summary": {},
                    **evaluation_run(
                        macro_f1=macro_f1,
                        cyber_f1=cyber_f1,
                        anomaly_f1=anomaly_f1,
                        heldout_value=heldout_value,
                    ),
                }
            )
    return {
        "name": name,
        "family_field": family_field,
        "group_key": "run_id",
        "seeds": seeds,
        "feasible": True,
        "evaluated_values": sorted(values),
        "total_runs": len(runs),
        "runs": runs,
        "skipped": [],
        "aggregate": aggregate_evaluation_runs(runs),
    }


def synthetic_training_report() -> dict[str, object]:
    scenario = holdout_report(
        "scenario_family_holdout",
        "attack_family",
        {
            "routine_alpha": [(0.68, 0.65, 0.70), (0.70, 0.67, 0.72)],
            "intrusion_beta": [(0.61, 0.60, 0.63), (0.63, 0.62, 0.65)],
        },
    )
    command = holdout_report(
        "command_family_holdout",
        "command",
        {
            "health_ping": [(0.69, 0.66, 0.71), (0.70, 0.67, 0.72)],
            "storage_mutation": [(0.62, 0.61, 0.64), (0.64, 0.63, 0.66)],
        },
    )
    protocol = holdout_report(
        "protocol_family_holdout",
        "protocol_family",
        {
            "fprime": [(0.66, 0.64, 0.69), (0.67, 0.65, 0.70)],
            "mavlink": [(0.64, 0.62, 0.66), (0.65, 0.63, 0.67)],
        },
    )
    return {
        "dataset": "artifacts/mixed_eval/data/dataset.jsonl",
        "rows": 144,
        "class_counts": {"benign": 48, "cyber": 48, "fault": 48},
        "training_path": {
            "name": POSTER_DEFAULT_TRAINING_PATH_NAME,
            "label": "poster default canonical path",
            "comparison_only": False,
        },
        "comparison_only": False,
        "deployment_ready": False,
        "deployment_blocked_reason": "generalization_metrics_below_threshold",
        "deployment_winner": {},
        "dataset_sanity": {
            "passed": True,
            "blocking_issues": [],
        },
        "protocol_only_baseline": baseline_stub(best_metric_value=0.62, near_perfect=False),
        "raw_protocol_shortcuts_baseline": baseline_stub(best_metric_value=0.64, near_perfect=False),
        "ranking_metric_order": [
            SELECTION_METRIC_CLASS_MACRO_F1,
            SELECTION_METRIC_MIN_PER_CLASS_RECALL,
            "model_only.anomaly_f1",
            SELECTION_METRIC_MODEL_CYBER_F1,
        ],
        "evaluation": {
            "repeated_grouped_cv": grouped_report(),
            "scenario_family_holdout": scenario,
            "command_family_holdout": command,
            "protocol_family_holdout": protocol,
            "deployment_gate": {
                "passed": False,
                "eligible_models": [],
                "thresholds": {"protocol_family_holdout": {"class_macro_f1": 0.5}},
                "models": {
                    "neural_net": {
                        "eligible_for_deployment": False,
                        "checks": [
                            {
                                "evaluation": "protocol_family_holdout",
                                "metric": "class_macro_f1",
                                "value": 0.645,
                                "threshold": 0.70,
                                "passed": False,
                            }
                        ],
                    }
                },
            },
        },
    }


class EvaluateGeneralizationTests(unittest.TestCase):
    def test_build_evaluation_matrix_includes_cross_protocol_rows(self) -> None:
        report = synthetic_training_report()

        matrix = build_evaluation_matrix(
            report,
            metrics_report_path=Path("/tmp/mixed_eval/reports/metrics.json"),
            training_invocation_blocked=True,
            training_invocation_message="reason=generalization_metrics_below_threshold",
        )

        self.assertEqual(matrix["record_kind"], "generalization_matrix")
        self.assertEqual(matrix["schema_version"], "generalization_matrix.v1")
        self.assertTrue(matrix["cross_protocol_generalization"]["feasible"])
        self.assertEqual(matrix["cross_protocol_generalization"]["evaluated_values"], ["fprime", "mavlink"])
        grouped = matrix["experiments"]["grouped_run_splits"]
        self.assertEqual(grouped["total_runs"], 2)
        self.assertIn("neural_net", grouped["aggregate_models"])
        protocol_holdout = matrix["experiments"]["protocol_family_holdout"]
        self.assertEqual(sorted(protocol_holdout["heldout_values"]), ["fprime", "mavlink"])
        protocol_rows = [
            row for row in matrix["matrix_rows"]
            if row["evaluation_name"] == "protocol_family_holdout" and row["scope"] == "heldout_value"
        ]
        heldout_values = sorted(row["heldout_value"] for row in protocol_rows)
        self.assertEqual(heldout_values, ["fprime", "mavlink"])
        mavlink_row = next(row for row in protocol_rows if row["heldout_value"] == "mavlink")
        self.assertAlmostEqual(mavlink_row["metrics"]["multiclass_macro_f1"], 0.645)
        self.assertFalse(matrix["deployment"]["ready"])
        self.assertTrue(matrix["training_invocation"]["blocked"])

    def test_render_summary_mentions_protocol_holdout_and_shortcuts(self) -> None:
        report = synthetic_training_report()
        matrix = build_evaluation_matrix(
            report,
            metrics_report_path=Path("/tmp/mixed_eval/reports/metrics.json"),
        )

        summary = render_evaluation_matrix_summary(matrix)

        self.assertIn("Generalization evaluation matrix", summary)
        self.assertIn("raw protocol shortcuts baseline", summary)
        self.assertIn("protocol_family_holdout", summary)
        self.assertIn("held out fprime neural_net", summary)
        self.assertIn("deployment ready: no", summary)

    def test_evaluate_generalization_recovers_saved_metrics_after_training_block(self) -> None:
        report = synthetic_training_report()
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            dataset_path = root / "data" / "dataset.jsonl"
            output_dir = root / "artifacts"
            metrics_path = output_dir / "reports" / "metrics.json"
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            dataset_path.write_text("{}\n", encoding="utf-8")
            metrics_path.parent.mkdir(parents=True, exist_ok=True)
            metrics_path.write_text(json.dumps(report), encoding="utf-8")

            with mock.patch(
                "tools.train.evaluate_generalization.run_training",
                autospec=True,
                side_effect=SystemExit("reason=generalization_metrics_below_threshold"),
            ) as run_training_mock:
                matrix = evaluate_generalization(dataset_path, output_dir, seed=7)

            run_training_mock.assert_called_once()
            self.assertTrue(matrix["training_invocation"]["blocked"])
            self.assertEqual(
                matrix["deployment"]["blocked_reason"],
                "generalization_metrics_below_threshold",
            )
            saved_matrix = json.loads((output_dir / "reports" / "evaluation_matrix.json").read_text(encoding="utf-8"))
            summary_text = (output_dir / "reports" / "evaluation_matrix_summary.txt").read_text(encoding="utf-8")
            self.assertEqual(saved_matrix["cross_protocol_generalization"]["evaluated_values"], ["fprime", "mavlink"])
            self.assertIn("recovered saved report after block", summary_text)


if __name__ == "__main__":
    unittest.main()
