from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.mavlink_real.runtime_layout import (
    CONTAINER_AUTOPILOT_ROOT,
    CONTAINER_RUNTIME_ROOT,
    GCS_SERVICE,
    IDENTITY_SERVICES,
    MAVLINK_NETWORK_NAME,
    SERVICE_IP_BY_NAME,
    VEHICLE_SERVICE,
    default_host_runtime_root,
    ensure_runtime_tree,
    host_bootstrap_metadata_path,
    host_capture_pcap_path,
    host_gcs_stdout_log_path,
    host_identity_logs_dir,
    host_identity_send_log_path,
    host_schedule_run_logs_dir,
    host_startup_metadata_path,
    host_vehicle_stdout_log_path,
    runtime_root_for_output,
)


class MavlinkRuntimeLayoutTests(unittest.TestCase):
    def test_default_runtime_roots_match_repo_contract(self) -> None:
        self.assertEqual(default_host_runtime_root(REPO_ROOT), REPO_ROOT / "gds" / "mavlink_runtime")
        self.assertEqual(runtime_root_for_output(Path("/tmp/run")), Path("/tmp/run") / "mavlink_real")

    def test_runtime_tree_creation_is_deterministic(self) -> None:
        with tempfile.TemporaryDirectory() as tmpdir:
            runtime_root = Path(tmpdir) / "runtime"
            ensure_runtime_tree(runtime_root)

            self.assertTrue(host_vehicle_stdout_log_path(runtime_root).parent.is_dir())
            self.assertTrue(host_gcs_stdout_log_path(runtime_root).parent.is_dir())
            self.assertTrue(host_bootstrap_metadata_path(runtime_root).parent.is_dir())
            self.assertTrue(host_startup_metadata_path(runtime_root).parent.is_dir())
            self.assertTrue(host_schedule_run_logs_dir(runtime_root).is_dir())
            self.assertTrue(host_capture_pcap_path(runtime_root).parent.is_dir())
            for identity_service in IDENTITY_SERVICES:
                self.assertTrue(host_identity_logs_dir(runtime_root, identity_service).is_dir())
                self.assertTrue(host_identity_send_log_path(runtime_root, identity_service).parent.is_dir())

    def test_service_and_container_constants_stay_stable(self) -> None:
        self.assertEqual(CONTAINER_RUNTIME_ROOT, Path("/runtime_root"))
        self.assertEqual(CONTAINER_AUTOPILOT_ROOT, Path("/ardupilot"))
        self.assertEqual(MAVLINK_NETWORK_NAME, "mavlink_real_net")
        self.assertEqual(VEHICLE_SERVICE, "mavlink_vehicle")
        self.assertEqual(GCS_SERVICE, "mavlink_gcs")
        self.assertEqual(
            IDENTITY_SERVICES,
            ("ops_primary", "ops_secondary", "red_primary", "red_secondary"),
        )
        self.assertEqual(SERVICE_IP_BY_NAME[VEHICLE_SERVICE], "192.168.164.2")
        self.assertEqual(SERVICE_IP_BY_NAME[GCS_SERVICE], "192.168.164.3")
        self.assertEqual(SERVICE_IP_BY_NAME["ops_primary"], "192.168.164.12")
        self.assertEqual(SERVICE_IP_BY_NAME["red_primary"], "192.168.164.22")


if __name__ == "__main__":
    unittest.main()
