from __future__ import annotations

import json
import subprocess
import sys
import unittest
import zipfile
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


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
                "evaluation_name": "protocol_family_holdout",
                "scope": "aggregate",
                "model_name": "neural_net",
                "metrics": {"multiclass_macro_f1": 0.42, "anomaly_f1": 0.64},
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
            }
        ],
    }


class PosterWorkflowScriptTests(unittest.TestCase):
    def test_shell_scripts_parse(self) -> None:
        for path in (
            REPO_ROOT / "tools" / "scripts" / "package_detector.sh",
            REPO_ROOT / "scripts" / "poster_demo.sh",
            REPO_ROOT / "scripts" / "poster_generate_assets.sh",
        ):
            proc = subprocess.run(
                ["bash", "-n", str(path)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr + proc.stdout)

    def test_package_detector_bundles_runtime_contract_files_and_run_reports(self) -> None:
        script_path = REPO_ROOT / "tools" / "scripts" / "package_detector.sh"
        with TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            model_dir = run_dir / "models"
            metrics_path = run_dir / "reports" / "metrics.json"
            summary_path = run_dir / "reports" / "summary.txt"
            scored_summary_path = run_dir / "scored" / "summary.json"

            model_dir.mkdir(parents=True, exist_ok=True)
            for name in ("blue_model.json", "bundle_manifest.json"):
                (model_dir / name).write_bytes((REPO_ROOT / "deployments" / "DetectorRB3" / "config" / name).read_bytes())
            _write_json(metrics_path, {"deployment_ready": True, "deployment_blocked_reason": None})
            summary_path.write_text("deployed_model=neural_net\n", encoding="utf-8")
            _write_json(scored_summary_path, {"rows": 24, "runtime_kind": "poster_blue_single_model_v1"})

            proc = subprocess.run(
                ["bash", str(script_path), "--run-dir", str(run_dir)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr + proc.stdout)
            zip_path = Path(proc.stdout.strip().splitlines()[-1])
            self.assertTrue(zip_path.exists(), msg=str(zip_path))

            with zipfile.ZipFile(zip_path) as archive:
                names = set(archive.namelist())

            self.assertIn("deployments/DetectorRB3/config/blue_model.json", names)
            self.assertIn("deployments/DetectorRB3/config/bundle_manifest.json", names)
            self.assertIn("runtime.py", names)
            self.assertIn("bg_pcyber.py", names)
            self.assertIn("README.md", names)
            self.assertIn("AGENTS.md", names)
            self.assertIn("TODO.md", names)
            self.assertIn("docs/blue_runtime_bundle.md", names)
            self.assertIn("docs/blue_feature_contract.md", names)
            self.assertIn("docs/poster_contract.md", names)
            self.assertIn("configs/feature_policies/blue_allowed_features.yaml", names)
            self.assertIn("schemas/canonical_command_row.schema.json", names)
            self.assertIn("run/reports/metrics.json", names)
            self.assertIn("run/reports/package_bundle_summary.txt", names)
            self.assertIn("run/scored/summary.json", names)

    def test_package_detector_accepts_checked_in_bundle_as_model_dir(self) -> None:
        script_path = REPO_ROOT / "tools" / "scripts" / "package_detector.sh"
        with TemporaryDirectory() as tmpdir:
            run_dir = Path(tmpdir) / "run"
            (run_dir / "reports").mkdir(parents=True, exist_ok=True)

            proc = subprocess.run(
                [
                    "bash",
                    str(script_path),
                    "--run-dir",
                    str(run_dir),
                    "--model-dir",
                    str(REPO_ROOT / "deployments" / "DetectorRB3" / "config"),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr + proc.stdout)
            zip_path = Path(proc.stdout.strip().splitlines()[-1])
            self.assertTrue(zip_path.exists(), msg=str(zip_path))

    def test_poster_demo_script_dry_run_prints_end_to_end_workflow(self) -> None:
        script_path = REPO_ROOT / "scripts" / "poster_demo.sh"
        with TemporaryDirectory() as tmpdir:
            proc = subprocess.run(
                [
                    "bash",
                    str(script_path),
                    "--dry-run",
                    "--output-dir",
                    str(Path(tmpdir) / "poster_demo"),
                    "--rows",
                    "24",
                    "--seed",
                    "7",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr + proc.stdout)
            self.assertIn("main.py generate", proc.stdout)
            self.assertIn("main.py train", proc.stdout)
            self.assertIn("score-packets", proc.stdout)
            self.assertIn("package_detector.sh", proc.stdout)
            self.assertIn("checked-in deployment fallback", proc.stdout)
            self.assertIn("minimum_training_rows: 100", proc.stdout)
            self.assertIn("fresh training will be skipped", proc.stdout)

    def test_poster_demo_small_dataset_skips_fresh_training_honestly(self) -> None:
        script_path = REPO_ROOT / "scripts" / "poster_demo.sh"
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "poster_demo"
            dataset_path = output_dir / "data" / "dataset.jsonl"
            packets_path = output_dir / "data" / "packets.jsonl"
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            dataset_path.write_text(
                "".join(json.dumps({"row_id": index}) + "\n" for index in range(24)),
                encoding="utf-8",
            )
            packets_path.write_text('{"packet_id": "pkt-0001"}\n', encoding="utf-8")

            proc = subprocess.run(
                [
                    "bash",
                    str(script_path),
                    "--skip-generate",
                    "--skip-score",
                    "--skip-package",
                    "--output-dir",
                    str(output_dir),
                    "--rows",
                    "24",
                    "--seed",
                    "7",
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr + proc.stdout)
            summary_text = (output_dir / "reports" / "poster_demo_summary.txt").read_text(encoding="utf-8")
            manifest = json.loads((output_dir / "reports" / "poster_demo_manifest.json").read_text(encoding="utf-8"))

            self.assertIn("training_status=skipped_insufficient_rows_for_fresh_training", summary_text)
            self.assertIn("training_skip_reason=minimum_training_rows=100; actual_rows=24", summary_text)
            self.assertEqual(manifest["training_status"], "skipped_insufficient_rows_for_fresh_training")
            self.assertEqual(manifest["dataset_row_count"], 24)
            self.assertEqual(manifest["minimum_training_rows"], 100)
            self.assertEqual(manifest["training_skip_reason"], "minimum_training_rows=100; actual_rows=24")
            self.assertIsNone(manifest["resolved_model_dir"])
            self.assertEqual(manifest["model_source_kind"], "not_needed")

    def test_poster_asset_script_wraps_existing_reports(self) -> None:
        script_path = REPO_ROOT / "scripts" / "poster_generate_assets.sh"
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            evaluation_matrix_path = root / "inputs" / "evaluation_matrix.json"
            red_blue_path = root / "inputs" / "red_blue_evaluation.json"
            output_dir = root / "asset_workflow"
            _write_json(evaluation_matrix_path, _evaluation_matrix_fixture())
            _write_json(red_blue_path, _red_blue_fixture())

            proc = subprocess.run(
                [
                    "bash",
                    str(script_path),
                    "--evaluation-matrix",
                    str(evaluation_matrix_path),
                    "--red-blue-evaluation",
                    str(red_blue_path),
                    "--output-dir",
                    str(output_dir),
                ],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr + proc.stdout)
            self.assertTrue((output_dir / "poster_assets" / "asset_manifest.json").exists())
            self.assertTrue((output_dir / "poster_assets" / "captions.md").exists())
            self.assertTrue((output_dir / "reports" / "poster_asset_workflow_summary.txt").exists())


if __name__ == "__main__":
    unittest.main()
