from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.figures.generate_protocol_progress_plots import generate_protocol_progress_plots


def _write_json(path: Path, payload: dict[str, object]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _metrics_fixture(*, macro_f1: float, cyber_f1: float, anomaly_f1: float) -> dict[str, object]:
    return {
        "schema_version": "training_metrics.v1",
        "record_kind": "training_metrics",
        "deployment_ready": False,
        "deployment_blocked_reason": "generalization_metrics_below_threshold",
        "blue_model": {
            "training_summary": {
                "best_validation_macro_f1": macro_f1 - 0.03,
                "final_validation_macro_f1": macro_f1 - 0.01,
            },
            "training_history": [
                {
                    "epoch": 1,
                    "train_loss": 1.08,
                    "validation_loss": 1.16,
                    "train_macro_f1": 0.31,
                    "validation_macro_f1": 0.28,
                    "epoch_rows": 64,
                },
                {
                    "epoch": 2,
                    "train_loss": 0.81,
                    "validation_loss": 0.92,
                    "train_macro_f1": 0.47,
                    "validation_macro_f1": 0.41,
                    "epoch_rows": 64,
                },
                {
                    "epoch": 3,
                    "train_loss": 0.56,
                    "validation_loss": 0.71,
                    "train_macro_f1": 0.63,
                    "validation_macro_f1": 0.57,
                    "epoch_rows": 64,
                },
            ],
        },
        "metrics": {
            "model_only": {
                "neural_net": {
                    "multiclass_metrics": {"accuracy": macro_f1 + 0.05, "macro_f1": macro_f1},
                    "cyber_binary_metrics": {"f1": cyber_f1},
                    "anomaly_binary_metrics": {"f1": anomaly_f1},
                }
            }
        },
    }


def _blue_update_report_fixture(*, best_val_loss: float, final_val_loss: float, final_train_loss: float) -> dict[str, object]:
    return {
        "blue_model": {
            "training_summary": {
                "best_epoch": 3,
                "epochs_completed": 5,
                "best_validation_cross_entropy": best_val_loss,
                "final_validation_cross_entropy": final_val_loss,
            },
            "training_history": [
                {"epoch": 1, "train_loss": final_train_loss + 0.3, "validation_loss": final_val_loss + 0.2},
                {"epoch": 5, "train_loss": final_train_loss, "validation_loss": final_val_loss},
            ],
        }
    }


def _self_play_state_fixture(
    *,
    final_macro_f1: float,
    final_cyber_f1: float,
    final_anomaly_f1: float,
    round_one_report_path: str,
    round_two_report_path: str,
) -> dict[str, object]:
    return {
        "schema_version": "self_play_state.v1",
        "record_kind": "self_play_state",
        "status": "completed",
        "rounds_completed": 2,
        "rounds": [
            {
                "schema_version": "self_play_round.v1",
                "record_kind": "self_play_round",
                "round_index": 1,
                "reward_summary": {"reward_mean": 0.04, "reward_min": -0.1, "reward_max": 0.18},
                "candidate_red_policy_alignment": {"joint_exact_match_accuracy": 0.22},
                "blue_update": {
                    "report_path": round_one_report_path,
                    "summary": {
                        "macro_f1": final_macro_f1 - 0.06,
                        "accuracy": final_macro_f1,
                        "cyber_f1": final_cyber_f1 - 0.04,
                        "anomaly_f1": final_anomaly_f1 - 0.05,
                        "deployment_ready": False,
                        "deployment_blocked_reason": "generalization_metrics_below_threshold",
                    }
                },
            },
            {
                "schema_version": "self_play_round.v1",
                "record_kind": "self_play_round",
                "round_index": 2,
                "reward_summary": {"reward_mean": 0.07, "reward_min": -0.08, "reward_max": 0.2},
                "candidate_red_policy_alignment": {"joint_exact_match_accuracy": 0.29},
                "blue_update": {
                    "report_path": round_two_report_path,
                    "summary": {
                        "macro_f1": final_macro_f1,
                        "accuracy": final_macro_f1 + 0.03,
                        "cyber_f1": final_cyber_f1,
                        "anomaly_f1": final_anomaly_f1,
                        "deployment_ready": False,
                        "deployment_blocked_reason": "generalization_metrics_below_threshold",
                    }
                },
            },
        ],
    }


class GenerateProtocolProgressPlotsTests(unittest.TestCase):
    def test_generate_protocol_progress_plots_writes_manifest_captions_and_expected_figures(self) -> None:
        with TemporaryDirectory() as tmpdir:
            root = Path(tmpdir)
            fprime_training_dir = root / "fprime_initial"
            mavlink_training_dir = root / "mavlink_initial"
            fprime_self_play_dir = root / "fprime_self_play"
            mavlink_self_play_dir = root / "mavlink_self_play"
            output_dir = root / "protocol_progress"

            _write_json(
                fprime_training_dir / "reports" / "metrics.json",
                _metrics_fixture(macro_f1=0.62, cyber_f1=0.71, anomaly_f1=0.68),
            )
            _write_json(
                mavlink_training_dir / "reports" / "metrics.json",
                _metrics_fixture(macro_f1=0.58, cyber_f1=0.64, anomaly_f1=0.61),
            )
            _write_json(
                fprime_self_play_dir / "rounds" / "round_0001" / "blue_update" / "reports" / "metrics.json",
                _blue_update_report_fixture(best_val_loss=0.82, final_val_loss=0.84, final_train_loss=0.62),
            )
            _write_json(
                fprime_self_play_dir / "rounds" / "round_0002" / "blue_update" / "reports" / "metrics.json",
                _blue_update_report_fixture(best_val_loss=0.73, final_val_loss=0.76, final_train_loss=0.54),
            )
            _write_json(
                mavlink_self_play_dir / "rounds" / "round_0001" / "blue_update" / "reports" / "metrics.json",
                _blue_update_report_fixture(best_val_loss=0.61, final_val_loss=0.63, final_train_loss=0.47),
            )
            _write_json(
                mavlink_self_play_dir / "rounds" / "round_0002" / "blue_update" / "reports" / "metrics.json",
                _blue_update_report_fixture(best_val_loss=0.55, final_val_loss=0.58, final_train_loss=0.41),
            )
            _write_json(
                fprime_self_play_dir / "self_play_state.json",
                _self_play_state_fixture(
                    final_macro_f1=0.67,
                    final_cyber_f1=0.75,
                    final_anomaly_f1=0.72,
                    round_one_report_path=str((fprime_self_play_dir / "rounds" / "round_0001" / "blue_update" / "reports" / "metrics.json").resolve()),
                    round_two_report_path=str((fprime_self_play_dir / "rounds" / "round_0002" / "blue_update" / "reports" / "metrics.json").resolve()),
                ),
            )
            _write_json(
                mavlink_self_play_dir / "self_play_state.json",
                _self_play_state_fixture(
                    final_macro_f1=0.61,
                    final_cyber_f1=0.68,
                    final_anomaly_f1=0.64,
                    round_one_report_path=str((mavlink_self_play_dir / "rounds" / "round_0001" / "blue_update" / "reports" / "metrics.json").resolve()),
                    round_two_report_path=str((mavlink_self_play_dir / "rounds" / "round_0002" / "blue_update" / "reports" / "metrics.json").resolve()),
                ),
            )

            manifest = generate_protocol_progress_plots(
                fprime_training_dir=fprime_training_dir,
                mavlink_training_dir=mavlink_training_dir,
                fprime_self_play_dir=fprime_self_play_dir,
                mavlink_self_play_dir=mavlink_self_play_dir,
                output_dir=output_dir,
            )

            self.assertEqual(manifest["record_kind"], "protocol_progress_manifest")
            self.assertEqual(manifest["schema_version"], "protocol_progress_manifest.v1")
            self.assertEqual(len(manifest["assets"]), 9)
            self.assertTrue((output_dir / "asset_manifest.json").exists())
            self.assertTrue((output_dir / "captions.md").exists())

            captions_text = (output_dir / "captions.md").read_text(encoding="utf-8")
            self.assertIn("Unified Initial Training Loss", captions_text)
            self.assertIn("Unified Performance Summary", captions_text)

            expected_ids = {
                "fig01_unified_initial_training_loss",
                "fig02_fprime_initial_training_loss",
                "fig03_mavlink_initial_training_loss",
                "fig04_unified_autoresearch_progress",
                "fig05_fprime_autoresearch_progress",
                "fig06_mavlink_autoresearch_progress",
                "fig07_unified_protocol_performance_summary",
                "fig08_fprime_protocol_performance_summary",
                "fig09_mavlink_protocol_performance_summary",
            }
            self.assertEqual({asset["asset_id"] for asset in manifest["assets"]}, expected_ids)
            for asset in manifest["assets"]:
                self.assertTrue(Path(asset["svg_path"]).exists())
                self.assertTrue(Path(asset["png_path"]).exists())


if __name__ == "__main__":
    unittest.main()
