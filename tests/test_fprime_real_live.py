from __future__ import annotations

import json
import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.shared.canonical_records import CANONICAL_COMMAND_ROW_SCHEMA_VERSION
from tools.shared.schema import RAW_PACKET_SCHEMA_VERSION, RAW_TRANSACTION_SCHEMA_VERSION


RUN_LIVE = os.environ.get("RUN_LIVE_FPRIME_TESTS") == "1"


@unittest.skipUnless(RUN_LIVE, "set RUN_LIVE_FPRIME_TESTS=1 to run live F' integration tests")
class LiveFprimeIntegrationTests(unittest.TestCase):
    def test_generate_fails_on_unsupported_benign_manifest(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            manifest_path = tmp_path / "bad_manifest.json"
            manifest_path.write_text(
                json.dumps(
                    [
                        {
                            "command": "cmdDisp.CMD_UNKNOWN_PROBE",
                            "arguments": [],
                            "required_observability": {
                                "request_wire": True,
                                "op_dispatched": True,
                                "terminal_event": True,
                                "telemetry_recent": True,
                            },
                        }
                    ]
                ),
                encoding="utf-8",
            )
            output_dir = tmp_path / "out"
            env = os.environ.copy()
            env["FPRIME_BENIGN_MANIFEST"] = str(manifest_path)
            proc = subprocess.run(
                ["python3", "tools/fprime_real/generate_dataset.py", "--rows", "24", "--output-dir", str(output_dir)],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertNotEqual(proc.returncode, 0)
            self.assertTrue((output_dir / "reports" / "support_matrix.json").exists())
            self.assertIn("Nominal support preflight failed", proc.stderr + proc.stdout)

    def test_generate_twice_uses_distinct_runtime_roots(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            first = tmp_path / "first"
            second = tmp_path / "second"
            for output_dir in (first, second):
                proc = subprocess.run(
                    ["python3", "main.py", "generate", "--rows", "24", "--output-dir", str(output_dir)],
                    cwd=REPO_ROOT,
                    capture_output=True,
                    text=True,
                    check=False,
                )
                self.assertEqual(proc.returncode, 0, msg=proc.stderr + proc.stdout)
                self.assertTrue((output_dir / "data" / "dataset.jsonl").exists())
                self.assertTrue((output_dir / "data" / "packets.jsonl").exists())
                self.assertTrue((output_dir / "data" / "raw_packets.jsonl").exists())
                self.assertTrue((output_dir / "data" / "raw_transactions.jsonl").exists())
                self.assertTrue((output_dir / "data" / "canonical_command_rows.jsonl").exists())
                self.assertTrue((output_dir / "fprime_real" / "cli_logs" / "command.log").exists())
                self.assertTrue((output_dir / "fprime_real" / "logs" / "send_log.jsonl").exists())
                self.assertTrue((output_dir / "reports" / "actual_run_observability.json").exists())
                provenance = json.loads((output_dir / "reports" / "provenance_summary.json").read_text(encoding="utf-8"))
                self.assertTrue(provenance.get("capture_backend"))
                self.assertTrue(provenance.get("capture_interface"))
                self.assertFalse(str(provenance.get("capture_interface", "")).lower().startswith("lo"))
                self.assertEqual(provenance.get("pcap_identity_mode"), "bridge_ip_5tuple")
                self.assertEqual(int(provenance.get("serialization_violations", 1)), 0)
                with (output_dir / "data" / "packets.jsonl").open(encoding="utf-8") as handle:
                    packets = [json.loads(line) for line in handle if line.strip()]
                observed_requests = [packet for packet in packets if packet.get("packet_kind") == "request" and int(packet.get("observed_on_wire", 0)) == 1]
                self.assertTrue(observed_requests)
                self.assertTrue(any(packet.get("src_ip") for packet in observed_requests))
                self.assertTrue(any(packet.get("target_stream_id") for packet in observed_requests))
                with (output_dir / "data" / "raw_packets.jsonl").open(encoding="utf-8") as handle:
                    raw_packet = json.loads(next(line for line in handle if line.strip()))
                with (output_dir / "data" / "raw_transactions.jsonl").open(encoding="utf-8") as handle:
                    raw_transaction = json.loads(next(line for line in handle if line.strip()))
                with (output_dir / "data" / "canonical_command_rows.jsonl").open(encoding="utf-8") as handle:
                    canonical_row = json.loads(next(line for line in handle if line.strip()))
                self.assertEqual(raw_packet.get("schema_version"), RAW_PACKET_SCHEMA_VERSION)
                self.assertEqual(raw_transaction.get("schema_version"), RAW_TRANSACTION_SCHEMA_VERSION)
                self.assertEqual(canonical_row.get("schema_version"), CANONICAL_COMMAND_ROW_SCHEMA_VERSION)
                self.assertEqual(canonical_row.get("record_kind"), "canonical_command_row")
            self.assertNotEqual((first / "fprime_real").resolve(), (second / "fprime_real").resolve())
            with (first / "fprime_real" / "logs" / "send_log.jsonl").open(encoding="utf-8") as handle:
                first_ids = {json.loads(line)["send_id"] for line in handle if line.strip()}
            with (second / "fprime_real" / "logs" / "send_log.jsonl").open(encoding="utf-8") as handle:
                second_ids = {json.loads(line)["send_id"] for line in handle if line.strip()}
            self.assertTrue(first_ids)
            self.assertTrue(second_ids)
            self.assertTrue(first_ids.isdisjoint(second_ids))


if __name__ == "__main__":
    unittest.main()
