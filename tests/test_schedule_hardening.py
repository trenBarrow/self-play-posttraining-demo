from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from main import (
    COMMAND_ONLY_BASELINE_NEAR_PERFECT_THRESHOLD,
    TRAINING_IMPORT_ERROR,
    evaluate_command_only_baseline,
    evaluate_request_only_baseline,
    grouped_row_split,
)
from runtime import COMMAND_IDS, SERVICE_IDS, inspect_args, normalize_command_args
from tools.fprime_real.benign_catalog import load_benign_command_samples
from tools.fprime_real.schedule_profiles import (
    BENIGN_TEMP_ROOT,
    INTENT_CONTEXT_BENIGN_CLEAN,
    INTENT_CONTEXT_BENIGN_NOISY,
    GOOD_SOURCES,
    EpisodePlan,
    _new_episode_state,
    assert_diverse_episode_signatures,
    build_benign_rows,
    build_command_family_overlap_report,
    build_cyber_rows,
    build_episode_signature_report,
    build_fault_rows,
    has_structural_signature_signal,
)
from tools.shared.run_manifest import build_class_overlap_report


def trust_class_for_score(value: float) -> str:
    if value >= 0.9:
        return "high"
    if value >= 0.6:
        return "medium"
    return "low"


def canonical_overlap_row_from_schedule(row: dict[str, object]) -> dict[str, object]:
    meta = dict(row["meta"])
    trust_score = float(meta["actor_trust"])
    return {
        "actor_context": {
            "role": str(meta["actor_role"]),
            "trust_class": trust_class_for_score(trust_score),
        },
        "audit_context": {
            "label_name": str(meta["class_name"]),
        },
    }


def build_schedule_rows() -> tuple[list[dict[str, object]], list[dict[str, object]], list[dict[str, object]], list[dict[str, object]]]:
    benign = build_benign_rows(target_rows=96, seed=7, episode_offset=0, episode_span=24)
    cyber = build_cyber_rows(target_rows=96, seed=7, episode_offset=4, episode_span=24)
    fault = build_fault_rows(target_rows=96, seed=7, episode_offset=8, episode_span=24)
    return benign, cyber, fault, [*benign, *cyber, *fault]


def baseline_row_from_schedule(row: dict[str, object]) -> dict[str, object]:
    command = str(row["command"])
    meta = dict(row["meta"])
    args = normalize_command_args(command, list(row["arguments"]))
    arg_count, arg_norm, arg_out_of_range, _, _ = inspect_args(command, args)
    service = command.split(".", 1)[0] if "." in command else ""
    return {
        "episode_id": int(meta["episode_id"]),
        "episode_kind": str(meta["class_name"]),
        "episode_label": int(meta["class_label"]),
        "label": int(meta["class_label"]),
        "label_name": str(meta["class_name"]),
        "command": command,
        "service": service,
        "service_id": float(SERVICE_IDS.get(service, 0)),
        "command_id": float(COMMAND_IDS.get(command, 0)),
        "arg_count": float(arg_count),
        "arg_norm": float(arg_norm),
        "arg_out_of_range": float(arg_out_of_range),
    }


def request_baseline_row_from_schedule(row: dict[str, object]) -> dict[str, object]:
    meta = dict(row["meta"])
    command = str(row["command"])
    command_family = command.split(".", 1)[0] if "." in command else command
    return {
        "episode_id": int(meta["episode_id"]),
        "run_id": int(meta["episode_id"]),
        "label": int(meta["class_label"]),
        "label_name": str(meta["class_name"]),
        "command_semantics.canonical_command_family": command_family,
        "mission_context.mission_phase": str(meta.get("phase") or ""),
        "argument_profile.argument_leaf_count": float(len(list(row["arguments"]))),
        "actor_context.role": str(meta["actor_role"]),
        "actor_context.trust_class": trust_class_for_score(float(meta["actor_trust"])),
    }


class ScheduleHardeningTests(unittest.TestCase):
    def test_benign_episode_state_does_not_seed_impossible_carryover_dirs(self) -> None:
        plan = EpisodePlan(
            profile_name="handover_backlog:focus:steady:fprime_a",
            precondition_profile="handover_backlog",
            source_strategy="focus",
            gap_profile="steady",
            target_focus="fprime_a",
            phase_sequence=("startup", "science", "standby"),
            source_sequence=(GOOD_SOURCES[0], GOOD_SOURCES[1], GOOD_SOURCES[0]),
            preferred_variants={},
            value_scale=1.0,
            burst_steps=frozenset(),
        )

        state = _new_episode_state("benign", 3, plan)

        self.assertEqual(state.created_dirs["fprime_a"], [])
        self.assertEqual(state.created_dirs["fprime_b"], [])
        self.assertEqual(state.fallback_remove_dirs["fprime_a"], [f"{BENIGN_TEMP_ROOT}/remove_ready_dir"])
        self.assertEqual(state.fallback_remove_dirs["fprime_b"], [f"{BENIGN_TEMP_ROOT}/remove_ready_dir"])

    def test_episode_signature_report_normalizes_episode_specific_tokens(self) -> None:
        rows = [
            {
                "time_of_day": "00:00:01",
                "source_service": "ops_a1",
                "target_service": "fprime_b",
                "command": "fileDownlink.SendPartial",
                "arguments": ["/workspace/README.md", "ops_partial_0001_03.bin", "128", "512"],
                "meta": {"class_label": 0, "class_name": "benign", "attack_family": "downlink_window", "episode_id": 1, "phase": "downlink"},
            },
            {
                "time_of_day": "00:00:01",
                "source_service": "ops_a1",
                "target_service": "fprime_b",
                "command": "fileDownlink.SendPartial",
                "arguments": ["/workspace/README.md", "ops_partial_0002_03.bin", "256", "768"],
                "meta": {"class_label": 0, "class_name": "benign", "attack_family": "downlink_window", "episode_id": 2, "phase": "downlink"},
            },
        ]

        report = build_episode_signature_report(rows)

        self.assertEqual(report["summary"]["episodes"], 2)
        self.assertEqual(report["summary"]["unique_signatures"], 1)
        self.assertEqual(report["summary"]["max_duplicate_group_count"], 2)

    def test_overlap_report_shows_core_commands_shared_across_classes(self) -> None:
        _, _, _, rows = build_schedule_rows()
        report = build_command_family_overlap_report(rows)
        by_command = {entry["command"]: entry for entry in report["commands"]}

        for command in ("cmdDisp.CMD_TEST_CMD_1", "fileDownlink.SendPartial", "fileManager.RemoveDirectory"):
            self.assertEqual(set(by_command[command]["shared_classes"]), {"benign", "cyber", "fault"})

        summary = report["summary"]
        self.assertGreaterEqual(summary["commands_shared_by_all_classes"], 10)
        self.assertGreaterEqual(summary["overlap_ratio"], 0.75)
        self.assertGreaterEqual(summary["shared_row_fraction"], 0.75)

    def test_benign_schedule_commands_are_all_manifest_known(self) -> None:
        benign, _, _, _ = build_schedule_rows()
        manifest_commands = {sample.command for sample in load_benign_command_samples()}
        benign_commands = {str(row["command"]) for row in benign}
        self.assertTrue(benign_commands)
        self.assertTrue(benign_commands.issubset(manifest_commands))

    def test_benign_schedule_marks_bounded_noisy_intent_rows(self) -> None:
        benign, _, _, _ = build_schedule_rows()
        intent_contexts = [str(dict(row["meta"]).get("intent_context", "")) for row in benign]

        noisy_count = sum(1 for value in intent_contexts if value == INTENT_CONTEXT_BENIGN_NOISY)
        clean_count = sum(1 for value in intent_contexts if value == INTENT_CONTEXT_BENIGN_CLEAN)

        self.assertGreater(noisy_count, 0)
        self.assertGreater(clean_count, 0)
        self.assertLess(noisy_count, len(benign) // 3)

    def test_benign_remove_directory_paths_stay_runtime_realistic(self) -> None:
        benign = build_benign_rows(target_rows=192, seed=7, episode_offset=0, episode_span=24)
        clean_remove_paths = [
            str(list(row["arguments"])[0])
            for row in benign
            if row["command"] == "fileManager.RemoveDirectory"
            and str(dict(row["meta"]).get("intent_context", "")) == INTENT_CONTEXT_BENIGN_CLEAN
        ]
        noisy_remove_paths = [
            str(list(row["arguments"])[0])
            for row in benign
            if row["command"] == "fileManager.RemoveDirectory"
            and str(dict(row["meta"]).get("intent_context", "")) == INTENT_CONTEXT_BENIGN_NOISY
        ]

        self.assertTrue(clean_remove_paths)
        self.assertTrue(noisy_remove_paths)

        for path in clean_remove_paths:
            self.assertNotIn("carryover_", path)
            self.assertNotIn("handover_", path)
            self.assertNotIn("already_removed_", path)
            self.assertTrue(path.endswith("/remove_ready_dir") or "/ep_" in path)

        for path in noisy_remove_paths:
            self.assertNotIn("already_removed_", path)
            self.assertNotIn("handover_", path)
            self.assertIn("carryover_", path)

    def test_episode_signature_report_marks_cyber_and_fault_episodes_unique(self) -> None:
        _, _, _, rows = build_schedule_rows()

        report = build_episode_signature_report(rows)
        assert_diverse_episode_signatures(report)

        self.assertEqual(report["per_class"]["cyber"]["unique_signatures"], report["per_class"]["cyber"]["episodes"])
        self.assertEqual(report["per_class"]["fault"]["unique_signatures"], report["per_class"]["fault"]["episodes"])
        self.assertEqual(report["summary"]["max_duplicate_group_count"], 1)

    def test_grouped_split_heldout_schedule_signatures_do_not_overlap_training(self) -> None:
        _, _, _, rows = build_schedule_rows()
        feature_rows = [baseline_row_from_schedule(row) for row in rows]

        base_rows, calib_rows, test_rows, _ = grouped_row_split(feature_rows, seed=7)
        split_assignments: dict[int, str] = {}
        for split_name, split_rows in (("base", base_rows), ("calibration", calib_rows), ("test", test_rows)):
            for row in split_rows:
                split_assignments[int(row["episode_id"])] = split_name

        report = build_episode_signature_report(rows, split_assignments=split_assignments)

        self.assertTrue(has_structural_signature_signal(report))
        self.assertEqual(report["split_overlap"]["train_like_to_test_shared_signature_count"], 0)
        self.assertEqual(report["split_overlap"]["train_like_to_test_shared_episode_count"], 0)

    def test_large_schedule_mixes_actor_roles_across_classes(self) -> None:
        benign = build_benign_rows(target_rows=240, seed=7, episode_offset=0, episode_span=24)
        cyber = build_cyber_rows(target_rows=240, seed=7, episode_offset=10, episode_span=24)
        fault = build_fault_rows(target_rows=240, seed=7, episode_offset=20, episode_span=24)
        canonical_rows = [canonical_overlap_row_from_schedule(row) for row in [*benign, *cyber, *fault]]

        role_report = build_class_overlap_report(
            canonical_rows,
            value_path="actor_context.role",
            family_key="canonical_actor_role",
            item_key="role",
            items_key="roles",
        )

        self.assertEqual(role_report["summary"]["values_with_dominant_class_share_ge_0_95"], 0)
        by_role = {entry["role"]: entry for entry in role_report["roles"]}
        self.assertGreaterEqual(by_role["ops_primary"]["shared_class_count"], 2)
        self.assertGreaterEqual(by_role["ops_backup"]["shared_class_count"], 2)
        self.assertGreaterEqual(by_role["shared_identity"]["shared_class_count"], 2)
        self.assertGreaterEqual(by_role["external"]["shared_class_count"], 2)

    def test_large_schedule_mixes_actor_trust_classes_across_classes(self) -> None:
        benign = build_benign_rows(target_rows=240, seed=11, episode_offset=0, episode_span=24)
        cyber = build_cyber_rows(target_rows=240, seed=11, episode_offset=10, episode_span=24)
        fault = build_fault_rows(target_rows=240, seed=11, episode_offset=20, episode_span=24)
        canonical_rows = [canonical_overlap_row_from_schedule(row) for row in [*benign, *cyber, *fault]]

        trust_report = build_class_overlap_report(
            canonical_rows,
            value_path="actor_context.trust_class",
            family_key="canonical_actor_trust_class",
            item_key="trust_class",
            items_key="trust_classes",
        )

        self.assertEqual(trust_report["summary"]["values_with_dominant_class_share_ge_0_95"], 0)
        by_trust = {entry["trust_class"]: entry for entry in trust_report["trust_classes"]}
        self.assertGreaterEqual(by_trust["high"]["shared_class_count"], 2)
        self.assertGreaterEqual(by_trust["medium"]["shared_class_count"], 2)
        self.assertGreaterEqual(by_trust["low"]["shared_class_count"], 2)

    @unittest.skipIf(TRAINING_IMPORT_ERROR is not None, f"training dependencies unavailable: {TRAINING_IMPORT_ERROR}")
    def test_request_baseline_without_role_or_trust_remains_meaningful(self) -> None:
        benign = build_benign_rows(target_rows=240, seed=7, episode_offset=0, episode_span=24)
        cyber = build_cyber_rows(target_rows=240, seed=7, episode_offset=10, episode_span=24)
        fault = build_fault_rows(target_rows=240, seed=7, episode_offset=20, episode_span=24)
        feature_rows = [request_baseline_row_from_schedule(row) for row in [*benign, *cyber, *fault]]

        base_rows, _, test_rows, _ = grouped_row_split(feature_rows, seed=7)
        with_identity = evaluate_request_only_baseline(
            base_rows,
            test_rows,
            seed=7,
            feature_names=[
                "command_semantics.canonical_command_family",
                "mission_context.mission_phase",
                "argument_profile.argument_leaf_count",
                "actor_context.role",
                "actor_context.trust_class",
            ],
        )
        without_identity = evaluate_request_only_baseline(
            base_rows,
            test_rows,
            seed=7,
            feature_names=[
                "command_semantics.canonical_command_family",
                "mission_context.mission_phase",
                "argument_profile.argument_leaf_count",
            ],
        )

        self.assertFalse(with_identity["near_perfect"])
        self.assertFalse(without_identity["near_perfect"])
        self.assertGreaterEqual(float(with_identity["class_metrics"]["macro_f1"]), 0.5)
        self.assertGreaterEqual(float(without_identity["class_metrics"]["macro_f1"]), 0.35)
        self.assertLess(
            float(with_identity["class_metrics"]["macro_f1"]) - float(without_identity["class_metrics"]["macro_f1"]),
            0.2,
        )
        self.assertGreaterEqual(float(without_identity["anomaly_binary_metrics"]["f1"]), 0.75)

    @unittest.skipIf(TRAINING_IMPORT_ERROR is not None, f"training dependencies unavailable: {TRAINING_IMPORT_ERROR}")
    def test_command_only_baseline_is_not_near_perfect(self) -> None:
        _, _, _, rows = build_schedule_rows()
        feature_rows = [baseline_row_from_schedule(row) for row in rows]

        base_rows, _, test_rows, _ = grouped_row_split(feature_rows, seed=7)
        baseline = evaluate_command_only_baseline(base_rows, test_rows, seed=7)

        self.assertFalse(baseline["near_perfect"])
        self.assertEqual(baseline["feature_names"], ["service_id", "command_id", "arg_count", "arg_norm", "arg_out_of_range"])
        self.assertLess(baseline["class_metrics"]["accuracy"], COMMAND_ONLY_BASELINE_NEAR_PERFECT_THRESHOLD)
        self.assertLess(baseline["class_metrics"]["macro_f1"], COMMAND_ONLY_BASELINE_NEAR_PERFECT_THRESHOLD)


if __name__ == "__main__":
    unittest.main()
