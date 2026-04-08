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

from tools.train import train_red_policy
from tools.train.red_policy_model import (
    RED_POLICY_EXAMPLES_ARTIFACT_NAME,
    RED_POLICY_MODEL_ARTIFACT_NAME,
    RED_POLICY_REPORT_ARTIFACT_NAME,
    LoadedRedPolicyModel,
    build_red_policy_warmstart_examples,
    export_red_policy_model_payload,
    fit_red_policy_model,
    infer_red_identity_bucket,
    load_red_action_space,
)
from tools.train.red_transcript import load_red_context_budget


class RedPolicyModelTests(unittest.TestCase):
    def test_load_red_action_space_exposes_explicit_constrained_heads(self) -> None:
        action_space = load_red_action_space()

        self.assertEqual(action_space["schema_version"], "red_action_space.v1")
        self.assertEqual(action_space["record_kind"], "red_action_space")
        self.assertEqual(
            action_space["current_context_fields"],
            ["protocol_family", "platform_family", "mission_phase", "window_class"],
        )
        self.assertEqual(
            action_space["command_family_values"],
            [
                "read_only_inspection",
                "transient_control",
                "mission_sequence_control",
                "persistent_configuration",
                "file_payload_management",
                "communications_control",
                "safety_critical_control",
                "maintenance_admin",
            ],
        )
        self.assertEqual(action_space["timing_bucket_values"], ["bootstrap", "rapid", "steady", "delayed"])
        self.assertEqual(action_space["identity_bucket_values"], ["external_low", "shared_identity"])
        self.assertNotIn("other_or_unknown", action_space["command_family_values"])
        protocol_services = action_space["action_heads"]["identity_bucket"]["protocol_services"]
        self.assertIn("red_a1", protocol_services["fprime"]["external_low"])
        self.assertIn("ops_a2", protocol_services["fprime"]["shared_identity"])
        self.assertIn("ops_secondary", protocol_services["mavlink"]["shared_identity"])

    def test_build_red_policy_warmstart_examples_uses_mixed_real_schedule_generators(self) -> None:
        action_space = load_red_action_space()
        examples = build_red_policy_warmstart_examples(protocol_mode="mixed", rows_per_protocol=24, seed=7)

        self.assertEqual(len(examples), 48)
        protocols = {example["metadata"]["protocol_family"] for example in examples}
        self.assertEqual(protocols, {"fprime", "mavlink"})
        self.assertEqual({example["record_kind"] for example in examples}, {"red_policy_example"})
        self.assertEqual({example["schema_version"] for example in examples}, {"red_policy_example.v1"})
        self.assertTrue(any(example["action"]["timing_bucket"] == "bootstrap" for example in examples))
        self.assertTrue(any(example["action"]["identity_bucket"] == "shared_identity" for example in examples))
        self.assertTrue(any(example["action"]["identity_bucket"] == "external_low" for example in examples))
        self.assertTrue(all(example["action"]["command_family"] != "other_or_unknown" for example in examples))
        for example in examples:
            self.assertEqual(example["transcript"]["record_kind"], "red_command_transcript")
            self.assertLessEqual(
                int(example["transcript"]["included_history_count"]),
                int(example["transcript"]["budget"]["max_history_entries"]),
            )
            for head_name, allowed_values in (
                ("command_family", action_space["command_family_values"]),
                ("timing_bucket", action_space["timing_bucket_values"]),
                ("identity_bucket", action_space["identity_bucket_values"]),
            ):
                self.assertIn(example["action"][head_name], allowed_values)

    def test_infer_red_identity_bucket_accepts_trusted_ops_roles_and_external_roles(self) -> None:
        action_space = load_red_action_space()

        self.assertEqual(
            infer_red_identity_bucket(
                protocol_family="fprime",
                source_service="ops_a2",
                actor_role="ops_backup",
                action_space=action_space,
            ),
            "shared_identity",
        )
        self.assertEqual(
            infer_red_identity_bucket(
                protocol_family="mavlink",
                source_service="ops_secondary",
                actor_role="ops_backup",
                action_space=action_space,
            ),
            "shared_identity",
        )
        self.assertEqual(
            infer_red_identity_bucket(
                protocol_family="fprime",
                source_service="red_b2",
                actor_role="external",
                action_space=action_space,
            ),
            "external_low",
        )
        self.assertIsNone(
            infer_red_identity_bucket(
                protocol_family="fprime",
                source_service="unknown_service",
                actor_role="observer",
                action_space=action_space,
            )
        )

    def test_fit_export_reload_red_policy_model_from_warmstart_examples(self) -> None:
        action_space = load_red_action_space()
        transcript_budget = load_red_context_budget()
        examples = build_red_policy_warmstart_examples(
            protocol_mode="mixed",
            rows_per_protocol=72,
            seed=7,
            action_space=action_space,
            transcript_budget=transcript_budget,
        )

        trained = fit_red_policy_model(
            examples,
            seed=7,
            action_space=action_space,
            transcript_budget=transcript_budget,
        )
        evaluation = trained["evaluation"]
        self.assertGreaterEqual(float(evaluation["joint_exact_match_accuracy"]), 0.20)
        self.assertGreaterEqual(float(evaluation["heads"]["command_family"]["accuracy"]), 0.30)
        self.assertGreaterEqual(float(evaluation["heads"]["timing_bucket"]["accuracy"]), 0.60)
        self.assertGreaterEqual(float(evaluation["heads"]["identity_bucket"]["macro_f1"]), 0.30)

        payload = export_red_policy_model_payload(trained, action_space=action_space)
        loaded = LoadedRedPolicyModel.from_payload(payload)
        prediction = loaded.predict_action(examples[0]["transcript"], examples[0]["current_context"])

        self.assertEqual(set(prediction["action"]), {"command_family", "timing_bucket", "identity_bucket"})
        self.assertIn(prediction["action"]["command_family"], action_space["command_family_values"])
        self.assertIn(prediction["action"]["timing_bucket"], action_space["timing_bucket_values"])
        self.assertIn(prediction["action"]["identity_bucket"], action_space["identity_bucket_values"])
        self.assertGreater(len(prediction["simulation_constraints"]["allowed_source_services"]), 0)

    def test_train_red_policy_script_writes_model_report_and_examples(self) -> None:
        with TemporaryDirectory() as tmpdir:
            output_dir = Path(tmpdir) / "red_policy"
            argv = [
                "train_red_policy.py",
                "--protocol-mode",
                "mixed",
                "--rows-per-protocol",
                "24",
                "--output-dir",
                str(output_dir),
            ]
            with mock.patch.object(sys, "argv", argv):
                with mock.patch("builtins.print"):
                    train_red_policy.main()

            model_path = output_dir / RED_POLICY_MODEL_ARTIFACT_NAME
            report_path = output_dir / RED_POLICY_REPORT_ARTIFACT_NAME
            examples_path = output_dir / RED_POLICY_EXAMPLES_ARTIFACT_NAME
            self.assertTrue(model_path.exists())
            self.assertTrue(report_path.exists())
            self.assertTrue(examples_path.exists())

            report = json.loads(report_path.read_text(encoding="utf-8"))
            self.assertEqual(report["protocol_mode"], "mixed")
            self.assertEqual(report["record_kind"], "red_policy_training_report")
            self.assertIn("evaluation", report)


if __name__ == "__main__":
    unittest.main()
