from __future__ import annotations

import os
import subprocess
import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


RUN_LIVE = os.environ.get("RUN_LIVE_MAVLINK_TESTS") == "1"


@unittest.skipUnless(RUN_LIVE, "set RUN_LIVE_MAVLINK_TESTS=1 to run live MAVLink integration tests")
class LiveMavlinkIntegrationTests(unittest.TestCase):
    def test_provenance_smoke_script_reconstructs_shared_artifacts(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            smoke_root = Path(tmpdir) / "mavlink_live_smoke"
            env = os.environ.copy()
            env["MAVLINK_PROVENANCE_SMOKE_DIR"] = str(smoke_root)
            proc = subprocess.run(
                ["bash", "scripts/mavlink_real/provenance_smoke_test.sh"],
                cwd=REPO_ROOT,
                env=env,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=proc.stderr + proc.stdout)
            output_dir = smoke_root / "reconstructed"
            for required in (
                output_dir / "data" / "packets.jsonl",
                output_dir / "data" / "transactions.jsonl",
                output_dir / "data" / "raw_packets.jsonl",
                output_dir / "data" / "raw_transactions.jsonl",
                output_dir / "reports" / "provenance_summary.json",
                output_dir / "reports" / "actual_run_observability.json",
            ):
                self.assertTrue(required.exists(), msg=str(required))


if __name__ == "__main__":
    unittest.main()
