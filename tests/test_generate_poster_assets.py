from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.figures.generate_poster_assets import generate_poster_assets


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _evaluation_matrix_fixture() -> dict[str, object]:
    return {
        "schema_version": "generalization_matrix.v1",
        "record_kind": "generalization_matrix",
        "dataset": "artifacts/mixed_latest/data/dataset.jsonl",
        "rows": 144,
        "class_counts": {"benign": 48, "cyber": 48, "fault": 48},
        "training_path": {"name": "poster_default_canonical", "comparison_only": False},
        "cross_protocol_generalization": {
            "evaluation_name": "protocol_family_holdout",
            "display_name": "protocol_family_holdout",
            "feasible": True,
            "evaluated_values": ["fprime", "mavlink"],
        },
        "shortcut_baselines": {
            "protocol_only": {"best_metric_value": 0.8, "best_metric_path": "anomaly_binary_metrics.f1"},
            "raw_protocol_shortcuts": {"best_metric_value": 0.8, "best_metric_path": "anomaly_binary_metrics.f1"},
        },
        "matrix_rows": [
            {
                "evaluation_name": "repeated_grouped_cv",
                "scope": "aggregate",
                "model_name": "neural_net",
                "metrics": {"multiclass_macro_f1": 0.62, "anomaly_f1": 0.80},
            },
            {
                "evaluation_name": "scenario_family_holdout",
                "scope": "aggregate",
                "model_name": "neural_net",
                "metrics": {"multiclass_macro_f1": 0.54, "anomaly_f1": 0.72},
            },
            {
                "evaluation_name": "command_family_holdout",
                "scope": "aggregate",
                "model_name": "neural_net",
                "metrics": {"multiclass_macro_f1": 0.51, "anomaly_f1": 0.69},
            },
            {
                "evaluation_name": "protocol_family_holdout",
                "scope": "aggregate",
                "model_name": "neural_net",
                "metrics": {"multiclass_macro_f1": 0.42, "anomaly_f1": 0.64},
            },
            {
                "evaluation_name": "protocol_family_holdout",
                "scope": "heldout_value",
                "heldout_value": "fprime",
                "model_name": "neural_net",
                "metrics": {"multiclass_macro_f1": 0.45, "anomaly_f1": 0.71},
            },
            {
                "evaluation_name": "protocol_family_holdout",
                "scope": "heldout_value",
                "heldout_value": "mavlink",
                "model_name": "neural_net",
                "metrics": {"multiclass_macro_f1": 0.39, "anomaly_f1": 0.57},
            },
        ],
    }


def _red_blue_fixture() -> dict[str, object]:
    return {
        "schema_version": "red_blue_evaluation.v1",
        "record_kind": "red_blue_evaluation",
        "summary_rows": [
            {
                "row_id": "summary_0001",
                "scope": "overall",
                "dataset_id": "dataset_01",
                "blue_id": "blue_01",
                "red_id": None,
                "adversary_kind": "static_schedule_replay",
                "transcript_length": 4,
                "coverage_adjusted_reward_mean": 0.02,
                "coverage_adjusted_adversary_success_rate": 0.01,
                "blue_precision_under_attack": 0.67,
                "blue_recall_under_attack": 1.0,
                "retrieval_coverage_rate": 1.0,
                "alignment_joint_exact_match_accuracy": None,
            },
            {
                "row_id": "summary_0002",
                "scope": "overall",
                "dataset_id": "dataset_01",
                "blue_id": "blue_01",
                "red_id": "red_01",
                "adversary_kind": "learned_red_policy_retrieval",
                "transcript_length": 4,
                "coverage_adjusted_reward_mean": 0.06,
                "coverage_adjusted_adversary_success_rate": 0.03,
                "blue_precision_under_attack": 0.91,
                "blue_recall_under_attack": 0.95,
                "retrieval_coverage_rate": 0.31,
                "alignment_joint_exact_match_accuracy": 0.12,
            },
            {
                "row_id": "summary_0003",
                "scope": "overall",
                "dataset_id": "dataset_01",
                "blue_id": "blue_01",
                "red_id": "red_02",
                "adversary_kind": "learned_red_policy_retrieval",
                "transcript_length": 8,
                "coverage_adjusted_reward_mean": 0.09,
                "coverage_adjusted_adversary_success_rate": 0.05,
                "blue_precision_under_attack": 0.88,
                "blue_recall_under_attack": 0.93,
                "retrieval_coverage_rate": 0.46,
                "alignment_joint_exact_match_accuracy": 0.19,
            },
        ],
        "comparison_rows": [
            {
                "comparison_id": "comparison_0001",
                "scope": "overall",
                "dataset_id": "dataset_01",
                "blue_id": "blue_01",
                "red_id": "red_01",
                "transcript_length": 4,
                "delta_blue_precision_under_attack": 0.24,
                "delta_coverage_adjusted_reward_mean": 0.04,
            },
            {
                "comparison_id": "comparison_0002",
                "scope": "overall",
                "dataset_id": "dataset_01",
                "blue_id": "blue_01",
                "red_id": "red_02",
                "transcript_length": 8,
                "delta_blue_precision_under_attack": 0.21,
                "delta_coverage_adjusted_reward_mean": 0.07,
            },
        ],
    }


def _single_protocol_evaluation_matrix_fixture() -> dict[str, object]:
    return {
        "schema_version": "generalization_matrix.v1",
        "record_kind": "generalization_matrix",
        "dataset": "artifacts/fprime_latest/data/dataset.jsonl",
        "rows": 120,
        "class_counts": {"benign": 40, "cyber": 40, "fault": 40},
        "training_path": {"name": "poster_default_canonical", "comparison_only": False},
        "cross_protocol_generalization": {
            "evaluation_name": "protocol_family_holdout",
            "display_name": "protocol_family_holdout",
            "feasible": False,
            "evaluated_values": [],
            "aggregate_models": {},
            "heldout_values": {},
        },
        "shortcut_baselines": {
            "protocol_only": {"best_metric_value": 0.0, "best_metric_path": None},
            "raw_protocol_shortcuts": {"best_metric_value": 0.0, "best_metric_path": None},
        },
        "matrix_rows": [
            {
                "evaluation_name": "repeated_grouped_cv",
                "scope": "aggregate",
                "model_name": "neural_net",
                "metrics": {"multiclass_macro_f1": 0.37, "anomaly_f1": 0.58},
            }
        ],
    }


class GeneratePosterAssetsTests(unittest.TestCase):
    def test_generate_poster_assets_writes_manifest_captions_and_expected_figures(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            evaluation_matrix_path = root / "reports" / "evaluation_matrix.json"
            red_blue_path = root / "reports" / "red_blue_evaluation.json"
            output_dir = root / "poster_assets"
            _write_json(evaluation_matrix_path, _evaluation_matrix_fixture())
            _write_json(red_blue_path, _red_blue_fixture())

            manifest = generate_poster_assets(
                evaluation_matrix_path=evaluation_matrix_path,
                red_blue_evaluation_path=red_blue_path,
                output_dir=output_dir,
            )

            self.assertEqual(manifest["record_kind"], "poster_asset_manifest")
            self.assertEqual(manifest["schema_version"], "poster_asset_manifest.v1")
            self.assertEqual(len(manifest["assets"]), 6)
            self.assertTrue((output_dir / "asset_manifest.json").exists())
            self.assertTrue((output_dir / "captions.md").exists())

            captions_text = (output_dir / "captions.md").read_text(encoding="utf-8")
            self.assertIn("Cross-Protocol Generalization", captions_text)
            self.assertIn("Adversary-Vs-Blue Curves", captions_text)

            expected_ids = {
                "fig01_architecture_overview",
                "fig02_raw_to_canonical_pipeline",
                "fig03_blue_red_loop",
                "fig04_cross_protocol_generalization",
                "fig05_adversary_vs_blue_curves",
                "fig06_blue_feature_families",
            }
            self.assertEqual({asset["asset_id"] for asset in manifest["assets"]}, expected_ids)
            for asset in manifest["assets"]:
                self.assertTrue(Path(asset["svg_path"]).exists())
                self.assertTrue(Path(asset["png_path"]).exists())

    def test_generate_poster_assets_handles_infeasible_protocol_holdout(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            evaluation_matrix_path = root / "reports" / "evaluation_matrix.json"
            red_blue_path = root / "reports" / "red_blue_evaluation.json"
            output_dir = root / "poster_assets"
            _write_json(evaluation_matrix_path, _single_protocol_evaluation_matrix_fixture())
            _write_json(red_blue_path, _red_blue_fixture())

            manifest = generate_poster_assets(
                evaluation_matrix_path=evaluation_matrix_path,
                red_blue_evaluation_path=red_blue_path,
                output_dir=output_dir,
            )

            self.assertEqual(manifest["record_kind"], "poster_asset_manifest")
            captions_text = (output_dir / "captions.md").read_text(encoding="utf-8")
            self.assertIn("Cross-Protocol Generalization", captions_text)
            self.assertTrue((output_dir / "figures" / "fig04_cross_protocol_generalization.svg").exists())
            self.assertTrue((output_dir / "figures" / "fig04_cross_protocol_generalization.png").exists())


if __name__ == "__main__":
    unittest.main()
