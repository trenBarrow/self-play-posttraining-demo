from __future__ import annotations

import copy
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.fprime_real import schedule_profiles as fprime_schedule_profiles
from tools.train.red_policy_model import (
    action_from_schedule_row,
    build_red_policy_warmstart_examples,
    current_context_from_schedule_row,
    load_red_action_space,
    schedule_row_to_canonical_row,
)
from tools.train.red_reward import (
    RED_REWARD_RESULT_SCHEMA_VERSION,
    RED_REWARD_SPEC_SCHEMA_VERSION,
    compute_red_reward,
    inspect_red_action_sandbox,
    load_red_reward_spec,
)
from tools.train.red_transcript import build_red_transcript, load_red_context_budget


class RedRewardTests(unittest.TestCase):
    def setUp(self) -> None:
        self.action_space = load_red_action_space()
        self.transcript_budget = load_red_context_budget()
        self.reward_spec = load_red_reward_spec()

    def _first_fprime_example(self) -> dict[str, object]:
        return build_red_policy_warmstart_examples(
            protocol_mode="fprime",
            rows_per_protocol=8,
            seed=7,
            action_space=self.action_space,
            transcript_budget=self.transcript_budget,
        )[0]

    def _wrong_source_service(self, protocol_family: str, identity_bucket: str) -> str:
        services = self.action_space["action_heads"]["identity_bucket"]["protocol_services"][protocol_family]
        opposite_bucket = "shared_identity" if identity_bucket == "external_low" else "external_low"
        return str(services[opposite_bucket][0])

    def test_load_red_reward_spec_exposes_explicit_reward_and_penalty_tables(self) -> None:
        self.assertEqual(self.reward_spec["schema_version"], RED_REWARD_SPEC_SCHEMA_VERSION)
        self.assertEqual(self.reward_spec["record_kind"], "red_reward_spec")
        self.assertEqual(
            self.reward_spec["outcome_rewards"]["unsafe_executed_undetected"],
            1.00,
        )
        self.assertEqual(
            self.reward_spec["outcome_rewards"]["unsafe_executed_detected"],
            0.20,
        )
        self.assertEqual(self.reward_spec["penalties"]["invalid_action"], -1.00)
        self.assertEqual(
            self.reward_spec["sandbox_policy"]["required_current_context_fields"],
            ["protocol_family", "platform_family", "mission_phase", "window_class"],
        )

    def test_undetected_unsafe_execution_receives_full_positive_reward(self) -> None:
        example = self._first_fprime_example()
        reward = compute_red_reward(
            action=example["action"],
            current_context=example["current_context"],
            transcript=example["transcript"],
            source_service=example["metadata"]["actor_id"],
            environment_feedback={
                "label_name": "cyber",
                "accepted": True,
                "executed_successfully": True,
            },
            blue_feedback={
                "predicted_class": "benign",
                "unsafe_risk": 0.18,
                "unsafe_threshold": 0.50,
                "detector_reason": "benign_baseline",
            },
            action_space=self.action_space,
            transcript_budget=self.transcript_budget,
            reward_spec=self.reward_spec,
        )

        self.assertEqual(reward["schema_version"], RED_REWARD_RESULT_SCHEMA_VERSION)
        self.assertTrue(reward["sandbox"]["allowed"])
        self.assertEqual(reward["reward_case"], "unsafe_executed_undetected")
        self.assertAlmostEqual(float(reward["reward"]), 1.00, places=6)

    def test_detected_unsafe_execution_receives_smaller_reward(self) -> None:
        example = self._first_fprime_example()
        reward = compute_red_reward(
            action=example["action"],
            current_context=example["current_context"],
            transcript=example["transcript"],
            source_service=example["metadata"]["actor_id"],
            environment_feedback={
                "label_name": "cyber",
                "accepted": True,
                "executed_successfully": True,
            },
            blue_feedback={
                "predicted_class": "cyber",
                "unsafe_risk": 0.92,
                "unsafe_threshold": 0.50,
                "detector_reason": "model_cyber",
            },
            action_space=self.action_space,
            transcript_budget=self.transcript_budget,
            reward_spec=self.reward_spec,
        )

        self.assertTrue(reward["sandbox"]["allowed"])
        self.assertEqual(reward["reward_case"], "unsafe_executed_detected")
        self.assertAlmostEqual(float(reward["reward"]), 0.20, places=6)

    def test_invalid_action_is_blocked_without_needing_blue_or_environment_feedback(self) -> None:
        example = self._first_fprime_example()
        invalid_action = dict(example["action"])
        invalid_action["command_family"] = "other_or_unknown"

        reward = compute_red_reward(
            action=invalid_action,
            current_context=example["current_context"],
            transcript=example["transcript"],
            source_service=example["metadata"]["actor_id"],
            action_space=self.action_space,
            transcript_budget=self.transcript_budget,
            reward_spec=self.reward_spec,
        )

        self.assertFalse(reward["sandbox"]["allowed"])
        self.assertIn("invalid_action", reward["sandbox"]["violation_reasons"])
        self.assertEqual(reward["reward_case"], "sandbox_blocked")
        self.assertAlmostEqual(float(reward["reward"]), -1.00, places=6)

    def test_benign_rapid_repeat_spam_is_penalized(self) -> None:
        row = fprime_schedule_profiles.build_cyber_rows(target_rows=1, seed=7)[0]
        current_context = current_context_from_schedule_row(
            row,
            protocol_family="fprime",
            action_space=self.action_space,
        )
        action = action_from_schedule_row(
            row,
            protocol_family="fprime",
            previous_self_row=row,
            action_space=self.action_space,
        )
        history = []
        for index in range(2):
            history.append(
                {
                    "canonical_row": schedule_row_to_canonical_row(
                        row,
                        protocol_family="fprime",
                        transaction_id=f"spam:txn:{index}",
                        send_id=f"spam:send:{index}",
                        window_class=current_context["window_class"],
                    ),
                    "coarse_result_class": "success",
                }
            )
        transcript = build_red_transcript(
            history,
            actor_id=str(row["source_service"]),
            budget=self.transcript_budget,
        )

        reward = compute_red_reward(
            action=action,
            current_context=current_context,
            transcript=transcript,
            source_service=str(row["source_service"]),
            environment_feedback={
                "label_name": "benign",
                "accepted": True,
                "executed_successfully": True,
            },
            blue_feedback={
                "predicted_class": "benign",
                "unsafe_risk": 0.05,
                "unsafe_threshold": 0.50,
                "detector_reason": "benign_baseline",
            },
            action_space=self.action_space,
            transcript_budget=self.transcript_budget,
            reward_spec=self.reward_spec,
        )

        self.assertTrue(reward["sandbox"]["allowed"])
        self.assertEqual(reward["reward_case"], "benign_executed")
        self.assertTrue(reward["pattern_analysis"]["rapid_repeat_spam"])
        self.assertFalse(reward["pattern_analysis"]["repeated_ineffective"])
        self.assertAlmostEqual(float(reward["reward"]), -0.60, places=6)

    def test_source_service_outside_identity_bucket_is_blocked_by_sandbox(self) -> None:
        example = self._first_fprime_example()
        wrong_source_service = self._wrong_source_service(
            protocol_family="fprime",
            identity_bucket=str(example["action"]["identity_bucket"]),
        )

        sandbox = inspect_red_action_sandbox(
            action=example["action"],
            current_context=example["current_context"],
            transcript=example["transcript"],
            source_service=wrong_source_service,
            action_space=self.action_space,
            transcript_budget=self.transcript_budget,
            reward_spec=self.reward_spec,
        )

        self.assertFalse(sandbox["allowed"])
        self.assertIn("source_service_out_of_bucket", sandbox["violation_reasons"])

    def test_transcript_budget_abuse_is_penalized(self) -> None:
        example = self._first_fprime_example()
        abused_transcript = copy.deepcopy(example["transcript"])
        abused_transcript["budget"]["max_history_entries"] = (
            int(self.transcript_budget["limits"]["max_history_entries"]) + 1
        )

        reward = compute_red_reward(
            action=example["action"],
            current_context=example["current_context"],
            transcript=abused_transcript,
            source_service=example["metadata"]["actor_id"],
            action_space=self.action_space,
            transcript_budget=self.transcript_budget,
            reward_spec=self.reward_spec,
        )

        self.assertFalse(reward["sandbox"]["allowed"])
        self.assertIn("transcript_budget_abuse", reward["sandbox"]["violation_reasons"])
        self.assertAlmostEqual(float(reward["reward"]), -0.75, places=6)


if __name__ == "__main__":
    unittest.main()
