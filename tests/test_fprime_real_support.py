from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.fprime_real.benign_catalog import BenignCommandSample, ObservabilityRequirement
from tools.fprime_real.send_classification import classify_event_history, classify_send_exception
from tools.fprime_real.support_probe import assert_actual_run_observability, assert_nominal_support, build_actual_run_observability_report, is_supported_result, is_supported_row


class SupportClassificationTests(unittest.TestCase):
    def test_supported_row_predicate(self) -> None:
        row = {
            "gds_accept": 1,
            "sat_success": 1,
            "timeout": 0,
            "reason": "completed",
        }
        self.assertTrue(is_supported_row(row))

    def test_unknown_command_exception(self) -> None:
        outcome = classify_send_exception("CommandSendException: command wasn't in the dictionary")
        self.assertEqual(outcome, (0, 0, 0, 1, "unknown_command"))

    def test_argument_reject_exception(self) -> None:
        outcome = classify_send_exception("CommandArgumentsException: value out of range")
        self.assertEqual(outcome, (0, 0, 0, 2, "arg_reject"))

    def test_timeout_without_terminal_event(self) -> None:
        outcome = classify_event_history([])
        self.assertEqual(outcome, (1, 0, 1, 3, "timeout"))

    def test_error_completion_event(self) -> None:
        events = [
            {
                "name": "cmdDisp.OpCodeError",
                "severity": "EventSeverity.COMMAND",
                "display_text": "Opcode 0x806 completed with error EXECUTION_ERROR",
            }
        ]
        outcome = classify_event_history(events)
        self.assertEqual(outcome, (1, 0, 0, 2, "execution_error"))

    def test_supported_result_requires_observability(self) -> None:
        row = {
            "gds_accept": 1,
            "sat_success": 1,
            "timeout": 0,
            "reason": "completed",
        }
        observation = {
            "request_wire_seen": True,
            "op_dispatched_seen": True,
            "terminal_event_seen": True,
            "telemetry_recent": False,
        }
        sample = BenignCommandSample(
            command="cmdDisp.CMD_NO_OP",
            arguments=(),
            required_observability=ObservabilityRequirement(
                request_wire=True,
                op_dispatched=True,
                terminal_event=True,
                telemetry_recent=True,
            ),
        )
        self.assertFalse(is_supported_result(row, observation, sample))

    def test_nominal_support_assertion_raises_on_unsupported_command(self) -> None:
        matrix = {
            "nodes": {
                "fprime_a": {
                    "target_service": "fprime_a",
                    "source_service": "ops_b1",
                    "target_tts_port": 50050,
                    "results": [
                        {
                            "command": "cmdDisp.CMD_NO_OP",
                            "supported": False,
                            "reason": "timeout",
                            "request_wire_seen": False,
                            "op_dispatched_seen": False,
                            "terminal_event_seen": False,
                            "telemetry_recent": False,
                            "send_exception": "",
                        }
                    ],
                }
            }
        }
        with self.assertRaises(SystemExit) as exc:
            assert_nominal_support(matrix)
        self.assertIn("Nominal support preflight failed", str(exc.exception))

    def test_actual_run_observability_report_requires_manifest_and_flags(self) -> None:
        run_rows = [
            {
                "send_id": "send-001",
                "source_service": "ops_b1",
                "target_service": "fprime_a",
                "target_tts_port": "50050",
                "command": "cmdDisp.CMD_NO_OP",
                "meta_json": "{\"class_label\":0,\"intent_context\":\"benign_noisy\"}",
                "gds_accept": "1",
                "sat_success": "0",
                "timeout": "0",
                "reason": "warning_event",
            },
            {
                "send_id": "send-002",
                "source_service": "ops_a1",
                "target_service": "fprime_b",
                "target_tts_port": "50050",
                "command": "cmdDisp.CMD_NO_OP",
                "meta_json": "{\"class_label\":0,\"intent_context\":\"benign_clean\"}",
                "gds_accept": "1",
                "sat_success": "1",
                "timeout": "0",
                "reason": "completed",
            },
        ]
        observations = [
            {
                "row_index": 0,
                "request_wire_seen": True,
                "op_dispatched_seen": True,
                "terminal_event_seen": True,
                "telemetry_recent": True,
                "response_direction_seen": True,
            },
            {
                "row_index": 1,
                "request_wire_seen": True,
                "op_dispatched_seen": False,
                "terminal_event_seen": True,
                "telemetry_recent": False,
                "response_direction_seen": True,
            },
        ]
        report = build_actual_run_observability_report(run_rows, observations)
        self.assertEqual(report["summary"]["benign_rows"], 2)
        self.assertEqual(report["summary"]["observability_failed_rows"], 1)
        self.assertEqual(report["summary"]["nonclean_rows"], 1)
        failing = next(item for item in report["rows"] if item["send_id"] == "send-002")
        self.assertEqual(failing["missing_observability"], ["op_dispatched_seen", "telemetry_recent"])
        with self.assertRaises(SystemExit) as exc:
            assert_actual_run_observability(report)
        self.assertIn("Captured benign run is missing required observability or manifest coverage", str(exc.exception))

    def test_actual_run_observability_allows_nonclean_benign_noisy_row(self) -> None:
        run_rows = [
            {
                "send_id": "send-003",
                "source_service": "ops_b1",
                "target_service": "fprime_a",
                "target_tts_port": "50050",
                "command": "cmdDisp.CMD_NO_OP",
                "meta_json": "{\"class_label\":0,\"intent_context\":\"benign_noisy\"}",
                "gds_accept": "1",
                "sat_success": "0",
                "timeout": "0",
                "reason": "warning_event",
            }
        ]
        observations = [
            {
                "row_index": 0,
                "request_wire_seen": True,
                "op_dispatched_seen": True,
                "terminal_event_seen": True,
                "telemetry_recent": True,
                "response_direction_seen": True,
            }
        ]

        report = build_actual_run_observability_report(run_rows, observations)
        self.assertEqual(report["summary"]["observability_failed_rows"], 0)
        self.assertEqual(report["summary"]["nonclean_rows"], 1)
        self.assertEqual(report["summary"]["intent_context_rows"]["benign_noisy"], 1)
        assert_actual_run_observability(report)


if __name__ == "__main__":
    unittest.main()
