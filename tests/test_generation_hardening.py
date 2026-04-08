from __future__ import annotations

import json
import sys
import unittest
from pathlib import Path
from tempfile import TemporaryDirectory
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.fprime_real.generate_dataset import build_independent_run_plans, resolve_active_capture_interface, snapshot_runtime_artifacts, summarize_behavior_rows, validate_nominal_runs
from tools.shared.run_manifest import class_row_targets


def make_run_row(
    *,
    class_label: int,
    class_name: str,
    intent_context: str,
    command: str,
    reason: str,
    gds_accept: str = "1",
    sat_success: str = "1",
    timeout: str = "0",
) -> dict[str, str]:
    return {
        "source_service": "ops_b1" if class_name == "benign" else "red_a1",
        "target_service": "fprime_a",
        "target_tts_port": "50050",
        "command": command,
        "meta_json": json.dumps(
            {
                "class_label": class_label,
                "class_name": class_name,
                "intent_context": intent_context,
                "attack_family": "none" if class_name == "benign" else ("downlink_abuse" if class_name == "cyber" else "ops_fault"),
                "episode_id": class_label,
                "phase": "science",
            }
        ),
        "gds_accept": gds_accept,
        "sat_success": sat_success,
        "timeout": timeout,
        "reason": reason,
        "event_names_json": "[]",
        "send_exception": "",
    }


class GenerationHardeningTests(unittest.TestCase):
    def test_resolve_active_capture_interface_refreshes_dynamic_bridge_without_override(self) -> None:
        with patch("tools.fprime_real.generate_dataset.env_capture_interface", return_value=""):
            with patch("tools.fprime_real.generate_dataset.preferred_capture_interface", return_value="br-newbridge"):
                self.assertEqual(
                    resolve_active_capture_interface("rdctl_vm_bridge", "br-oldbridge"),
                    "br-newbridge",
                )

    def test_resolve_active_capture_interface_preserves_explicit_override(self) -> None:
        with patch("tools.fprime_real.generate_dataset.env_capture_interface", return_value="bridge100"):
            with patch("tools.fprime_real.generate_dataset.preferred_capture_interface", return_value="br-newbridge"):
                self.assertEqual(
                    resolve_active_capture_interface("rdctl_vm_bridge", "br-oldbridge"),
                    "bridge100",
                )

    def test_build_independent_run_plans_assigns_run_ids_and_shuffles_execution_order(self) -> None:
        plans, manifest = build_independent_run_plans(
            nominal_target=96,
            cyber_target=64,
            fault_target=32,
            seed=7,
        )

        self.assertTrue(plans)
        self.assertEqual(manifest["group_key"], "run_id")
        self.assertEqual(manifest["runtime_reset_policy"], "fresh_runtime_per_episode")
        self.assertEqual(sorted(plan.run_id for plan in plans), list(range(len(plans))))

        fixed_order = (
            ["benign"] * int(manifest["class_episode_counts"]["benign"])
            + ["cyber"] * int(manifest["class_episode_counts"]["cyber"])
            + ["fault"] * int(manifest["class_episode_counts"]["fault"])
        )
        self.assertNotEqual(manifest["class_order"], fixed_order)

        for plan in plans:
            self.assertEqual({int(dict(row["meta"])["run_id"]) for row in plan.schedule_rows}, {plan.run_id})
            self.assertEqual({str(dict(row["meta"])["runtime_reset_policy"]) for row in plan.schedule_rows}, {"fresh_runtime_per_episode"})

    def test_build_independent_run_plans_large_targets_raise_episode_support(self) -> None:
        row_targets = class_row_targets(rows=1000, nominal_ratio=0.55)
        plans, manifest = build_independent_run_plans(
            nominal_target=int(row_targets["benign"]),
            cyber_target=int(row_targets["cyber"]),
            fault_target=int(row_targets["fault"]),
            seed=7,
        )

        self.assertEqual(manifest["class_row_targets"], row_targets)
        self.assertGreaterEqual(int(manifest["class_episode_counts"]["benign"]), 30)
        self.assertGreaterEqual(int(manifest["class_episode_counts"]["cyber"]), 30)
        self.assertGreaterEqual(int(manifest["class_episode_counts"]["fault"]), 20)

        episode_summary = manifest["class_rows_per_episode_summary"]
        self.assertGreaterEqual(int(episode_summary["benign"]["episode_target"]), 30)
        self.assertGreaterEqual(int(episode_summary["cyber"]["episode_target"]), 30)
        self.assertGreaterEqual(int(episode_summary["fault"]["episode_target"]), 20)
        self.assertLessEqual(float(episode_summary["benign"]["avg_rows"]), 18.5)
        self.assertLessEqual(float(episode_summary["cyber"]["avg_rows"]), 10.5)
        self.assertLessEqual(float(episode_summary["fault"]["avg_rows"]), 8.5)
        self.assertEqual(len(plans), int(manifest["run_count"]))

    def test_build_independent_run_plans_development_targets_raise_episode_support(self) -> None:
        row_targets = class_row_targets(rows=120, nominal_ratio=0.55)
        plans, manifest = build_independent_run_plans(
            nominal_target=int(row_targets["benign"]),
            cyber_target=int(row_targets["cyber"]),
            fault_target=int(row_targets["fault"]),
            seed=7,
        )

        self.assertEqual(manifest["class_row_targets"], row_targets)
        self.assertGreaterEqual(int(manifest["class_episode_counts"]["benign"]), 6)
        self.assertGreaterEqual(int(manifest["class_episode_counts"]["cyber"]), 6)
        self.assertGreaterEqual(int(manifest["class_episode_counts"]["fault"]), 5)

        episode_summary = manifest["class_rows_per_episode_summary"]
        self.assertGreaterEqual(int(episode_summary["benign"]["episode_target"]), 6)
        self.assertGreaterEqual(int(episode_summary["cyber"]["episode_target"]), 6)
        self.assertGreaterEqual(int(episode_summary["fault"]["episode_target"]), 5)
        self.assertLessEqual(float(episode_summary["benign"]["avg_rows"]), 13.5)
        self.assertLessEqual(float(episode_summary["cyber"]["avg_rows"]), 6.5)
        self.assertLessEqual(float(episode_summary["fault"]["avg_rows"]), 4.5)
        self.assertEqual(len(plans), int(manifest["run_count"]))

    def test_snapshot_runtime_artifacts_copies_runtime_tree(self) -> None:
        with TemporaryDirectory() as source_dir, TemporaryDirectory() as destination_dir:
            source_root = Path(source_dir)
            destination_root = Path(destination_dir) / "per_run"
            (source_root / "cli_logs").mkdir(parents=True, exist_ok=True)
            (source_root / "logs").mkdir(parents=True, exist_ok=True)
            (source_root / "node_a" / "logs").mkdir(parents=True, exist_ok=True)
            (source_root / "node_b" / "logs").mkdir(parents=True, exist_ok=True)
            (source_root / "cli_logs" / "command.log").write_text("command-entry\n", encoding="utf-8")
            (source_root / "logs" / "send_log.jsonl").write_text('{"send_id":"abc"}\n', encoding="utf-8")
            (source_root / "node_a" / "runtime_env.txt").write_text("node=a\n", encoding="utf-8")
            (source_root / "node_a" / "logs" / "event.log").write_text("event-a\n", encoding="utf-8")
            (source_root / "node_b" / "logs" / "downlink_records.jsonl").write_text('{"channel":"x"}\n', encoding="utf-8")

            snapshot_runtime_artifacts(source_root, destination_root)

            self.assertEqual(
                (destination_root / "cli_logs" / "command.log").read_text(encoding="utf-8"),
                "command-entry\n",
            )
            self.assertEqual(
                (destination_root / "logs" / "send_log.jsonl").read_text(encoding="utf-8"),
                '{"send_id":"abc"}\n',
            )
            self.assertEqual(
                (destination_root / "node_a" / "runtime_env.txt").read_text(encoding="utf-8"),
                "node=a\n",
            )
            self.assertEqual(
                (destination_root / "node_a" / "logs" / "event.log").read_text(encoding="utf-8"),
                "event-a\n",
            )
            self.assertEqual(
                (destination_root / "node_b" / "logs" / "downlink_records.jsonl").read_text(encoding="utf-8"),
                '{"channel":"x"}\n',
            )

    def test_validate_nominal_runs_allows_bounded_benign_noise(self) -> None:
        rows = [
            *[
                make_run_row(
                    class_label=0,
                    class_name="benign",
                    intent_context="benign_clean",
                    command="cmdDisp.CMD_NO_OP",
                    reason="completed",
                )
                for _ in range(6)
            ],
            make_run_row(
                class_label=0,
                class_name="benign",
                intent_context="benign_noisy",
                command="fileDownlink.Cancel",
                reason="warning_event",
                sat_success="0",
            ),
            make_run_row(
                class_label=0,
                class_name="benign",
                intent_context="benign_noisy",
                command="fileManager.FileSize",
                reason="missing_artifact",
                sat_success="0",
            ),
        ]

        report = validate_nominal_runs(rows)
        self.assertEqual(report["summary"]["clean_success_rows"], 6)
        self.assertEqual(report["summary"]["benign_noisy_rows"], 2)
        self.assertEqual(report["summary"]["unexpected_rows"], 0)

    def test_validate_nominal_runs_allows_filesize_format_error_for_benign_noise(self) -> None:
        rows = [
            *[
                make_run_row(
                    class_label=0,
                    class_name="benign",
                    intent_context="benign_clean",
                    command="cmdDisp.CMD_NO_OP",
                    reason="completed",
                )
                for _ in range(5)
            ],
            make_run_row(
                class_label=0,
                class_name="benign",
                intent_context="benign_noisy",
                command="fileManager.FileSize",
                reason="format_error",
                sat_success="0",
            ),
        ]

        report = validate_nominal_runs(rows)
        self.assertEqual(report["summary"]["benign_noisy_rows"], 1)
        self.assertEqual(report["summary"]["unexpected_rows"], 0)

    def test_validate_nominal_runs_rejects_clean_intent_failures(self) -> None:
        rows = [
            make_run_row(
                class_label=0,
                class_name="benign",
                intent_context="benign_clean",
                command="cmdDisp.CMD_NO_OP",
                reason="completed",
            ),
            make_run_row(
                class_label=0,
                class_name="benign",
                intent_context="benign_clean",
                command="fileManager.FileSize",
                reason="missing_artifact",
                sat_success="0",
            ),
        ]

        with self.assertRaises(SystemExit) as exc:
            validate_nominal_runs(rows)
        self.assertIn("bounded benign-noise policy", str(exc.exception))

    def test_validate_nominal_runs_rejects_excessive_benign_noise(self) -> None:
        rows = [
            *[
                make_run_row(
                    class_label=0,
                    class_name="benign",
                    intent_context="benign_clean",
                    command="cmdDisp.CMD_NO_OP",
                    reason="completed",
                )
                for _ in range(8)
            ],
            *[
                make_run_row(
                    class_label=0,
                    class_name="benign",
                    intent_context="benign_noisy",
                    command="fileDownlink.Cancel",
                    reason="warning_event",
                    sat_success="0",
                )
                for _ in range(4)
            ],
        ]

        with self.assertRaises(SystemExit) as exc:
            validate_nominal_runs(rows)
        self.assertIn("bounded benign-noise policy", str(exc.exception))

    def test_behavior_summary_distinguishes_benign_noisy_malicious_and_fault(self) -> None:
        rows = [
            make_run_row(
                class_label=0,
                class_name="benign",
                intent_context="benign_clean",
                command="cmdDisp.CMD_NO_OP",
                reason="completed",
            ),
            make_run_row(
                class_label=0,
                class_name="benign",
                intent_context="benign_noisy",
                command="fileDownlink.Cancel",
                reason="warning_event",
                sat_success="0",
            ),
            make_run_row(
                class_label=1,
                class_name="cyber",
                intent_context="malicious",
                command="fileDownlink.SendPartial",
                reason="execution_error",
                sat_success="0",
            ),
            make_run_row(
                class_label=2,
                class_name="fault",
                intent_context="fault",
                command="fileManager.RemoveDirectory",
                reason="missing_artifact",
                sat_success="0",
            ),
        ]

        summary = summarize_behavior_rows(rows)
        self.assertEqual(summary["intent_context_rows"]["benign_clean"], 1)
        self.assertEqual(summary["intent_context_rows"]["benign_noisy"], 1)
        self.assertEqual(summary["intent_context_rows"]["malicious"], 1)
        self.assertEqual(summary["intent_context_rows"]["fault"], 1)
        self.assertEqual(summary["status_by_intent_context"]["benign_clean"]["success"], 1)
        self.assertEqual(summary["status_by_intent_context"]["benign_noisy"]["anomalous"], 1)
        self.assertEqual(summary["class_intent_rows"]["cyber"]["malicious"], 1)
        self.assertEqual(summary["class_intent_rows"]["fault"]["fault"], 1)


if __name__ == "__main__":
    unittest.main()
