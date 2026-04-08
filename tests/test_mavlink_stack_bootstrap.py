from __future__ import annotations

import shutil
import subprocess
import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))


class MavlinkStackBootstrapTests(unittest.TestCase):
    def test_shell_scripts_parse(self) -> None:
        for script in (
            REPO_ROOT / "scripts" / "mavlink_real" / "bootstrap_stack.sh",
            REPO_ROOT / "scripts" / "mavlink_real" / "up.sh",
            REPO_ROOT / "scripts" / "mavlink_real" / "down.sh",
            REPO_ROOT / "scripts" / "mavlink_real" / "run_vehicle.sh",
            REPO_ROOT / "scripts" / "mavlink_real" / "run_gcs.sh",
            REPO_ROOT / "scripts" / "mavlink_real" / "smoke_test.sh",
            REPO_ROOT / "scripts" / "mavlink_real" / "schedule_smoke_test.sh",
            REPO_ROOT / "scripts" / "mavlink_real" / "provenance_smoke_test.sh",
        ):
            proc = subprocess.run(
                ["bash", "-n", str(script)],
                cwd=REPO_ROOT,
                capture_output=True,
                text=True,
                check=False,
            )
            self.assertEqual(proc.returncode, 0, msg=f"{script} failed bash -n: {proc.stderr}{proc.stdout}")

    def test_compose_file_renders(self) -> None:
        docker_bin = shutil.which("docker")
        if docker_bin is None:
            self.skipTest("docker is required to validate compose rendering")

        compose_file = REPO_ROOT / "orchestration" / "docker-compose.mavlink-real.yml"
        proc = subprocess.run(
            [docker_bin, "compose", "-f", str(compose_file), "config"],
            cwd=REPO_ROOT,
            capture_output=True,
            text=True,
            check=False,
        )
        self.assertEqual(proc.returncode, 0, msg=proc.stderr + proc.stdout)
        rendered = proc.stdout
        for token in (
            "mavlink_vehicle:",
            "mavlink_gcs:",
            "ops_primary:",
            "ops_secondary:",
            "red_primary:",
            "red_secondary:",
            "mavlink_real_net:",
            "linux/amd64",
            "/workspace/scripts/mavlink_real/run_vehicle.sh",
            "/workspace/scripts/mavlink_real/run_gcs.sh",
        ):
            self.assertIn(token, rendered)

    def test_bootstrap_and_runtime_scripts_pin_headless_contract(self) -> None:
        bootstrap = (REPO_ROOT / "scripts" / "mavlink_real" / "bootstrap_stack.sh").read_text(encoding="utf-8")
        run_vehicle = (REPO_ROOT / "scripts" / "mavlink_real" / "run_vehicle.sh").read_text(encoding="utf-8")
        run_gcs = (REPO_ROOT / "scripts" / "mavlink_real" / "run_gcs.sh").read_text(encoding="utf-8")
        smoke = (REPO_ROOT / "scripts" / "mavlink_real" / "smoke_test.sh").read_text(encoding="utf-8")
        schedule_smoke = (REPO_ROOT / "scripts" / "mavlink_real" / "schedule_smoke_test.sh").read_text(encoding="utf-8")
        provenance_smoke = (REPO_ROOT / "scripts" / "mavlink_real" / "provenance_smoke_test.sh").read_text(encoding="utf-8")

        self.assertIn("Copter-4.6.3", bootstrap)
        self.assertIn("./waf configure --board sitl", bootstrap)
        self.assertIn("./waf copter", bootstrap)
        self.assertIn("com.poster.mavlink_bootstrap_hash", bootstrap)
        dockerfile = (REPO_ROOT / "orchestration" / "mavlink-real" / "base.Dockerfile").read_text(encoding="utf-8")
        self.assertIn("ARG MAVPROXY_VERSION=1.8.74", dockerfile)
        self.assertIn("ARG PYMAVLINK_VERSION=2.4.49", dockerfile)
        self.assertIn("PyYAML==6.0.2", dockerfile)
        self.assertIn("--serial0 \"udpclient:${GCS_HOST}:${GCS_UDP_PORT}\"", run_vehicle)
        self.assertIn("--serial1 \"tcp:0.0.0.0:${IDENTITY_TCP_PORT}\"", run_vehicle)
        self.assertIn("--master=\"$MASTER_ENDPOINT\"", run_gcs)
        self.assertIn("--non-interactive", run_gcs)
        self.assertIn("--state-basedir=\"$STATE_DIR\"", run_gcs)
        self.assertIn("bash scripts/mavlink_real/up.sh", (REPO_ROOT / "docs" / "mavlink_runtime_bootstrap.md").read_text(encoding="utf-8"))
        self.assertIn("docker compose -f \"$COMPOSE_FILE\" ps --services --status running", smoke)
        self.assertIn("make_good_schedule.py", schedule_smoke)
        self.assertIn("run_mavlink_schedule.py", schedule_smoke)
        self.assertIn("resolve_identity_capture_target", provenance_smoke)
        self.assertIn("write_artifact_bundle", provenance_smoke)


if __name__ == "__main__":
    unittest.main()
