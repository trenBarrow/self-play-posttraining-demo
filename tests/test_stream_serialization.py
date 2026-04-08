from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.fprime_real.run_fprime_schedule import validate_serialized_run_rows


class StreamSerializationTests(unittest.TestCase):
    def test_monotonic_same_target_stream_passes(self) -> None:
        rows = [
            {
                "target_service": "fprime_a",
                "target_stream_id": "fprime_a:50050",
                "target_stream_index": "0",
                "command": "cmdDisp.CMD_NO_OP",
                "send_start_ms": "1000",
                "send_end_ms": "1050",
            },
            {
                "target_service": "fprime_a",
                "target_stream_id": "fprime_a:50050",
                "target_stream_index": "1",
                "command": "cmdDisp.CMD_NO_OP",
                "send_start_ms": "1100",
                "send_end_ms": "1160",
            },
        ]
        validate_serialized_run_rows(rows)

    def test_overlapping_same_target_stream_fails(self) -> None:
        rows = [
            {
                "target_service": "fprime_a",
                "target_stream_id": "fprime_a:50050",
                "target_stream_index": "0",
                "command": "cmdDisp.CMD_NO_OP",
                "send_start_ms": "1000",
                "send_end_ms": "1100",
            },
            {
                "target_service": "fprime_a",
                "target_stream_id": "fprime_a:50050",
                "target_stream_index": "1",
                "command": "cmdDisp.CMD_NO_OP",
                "send_start_ms": "1050",
                "send_end_ms": "1150",
            },
        ]
        with self.assertRaises(SystemExit) as exc:
            validate_serialized_run_rows(rows)
        self.assertIn("serialized-per-target invariant", str(exc.exception))

    def test_target_stream_index_must_increase(self) -> None:
        rows = [
            {
                "target_service": "fprime_b",
                "target_stream_id": "fprime_b:50050",
                "target_stream_index": "0",
                "command": "cmdDisp.CMD_NO_OP",
                "send_start_ms": "1000",
                "send_end_ms": "1100",
            },
            {
                "target_service": "fprime_b",
                "target_stream_id": "fprime_b:50050",
                "target_stream_index": "0",
                "command": "cmdDisp.CMD_NO_OP",
                "send_start_ms": "1200",
                "send_end_ms": "1250",
            },
        ]
        with self.assertRaises(SystemExit) as exc:
            validate_serialized_run_rows(rows)
        self.assertIn("non_monotonic_stream_index", str(exc.exception))


if __name__ == "__main__":
    unittest.main()
