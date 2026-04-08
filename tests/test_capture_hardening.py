from __future__ import annotations

import json
import os
import sys
import tempfile
import unittest
from pathlib import Path
from unittest.mock import patch

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.fprime_real.pcap_capture import CAPTURE_BACKEND_LOCAL_DOCKER_BRIDGE, CAPTURE_BACKEND_RDCTL_VM_BRIDGE, resolve_capture_backend
from tools.fprime_real.packet_fidelity import PacketBuildResult, PcapParseResult, row_run_id
from tools.fprime_real.support_probe import resolve_identity_capture_target, wait_for_capture_drain


class FakeClock:
    def __init__(self, sleep_increment: float = 0.2):
        self.now = 0.0
        self.sleep_increment = sleep_increment

    def time(self) -> float:
        return self.now

    def sleep(self, seconds: float) -> None:
        self.now += max(seconds, self.sleep_increment)


def packet_result_for(send_id: str, *, request_wire_seen: bool, response_direction_seen: bool, terminal_event_seen: bool) -> PacketBuildResult:
    return PacketBuildResult(
        packets=[],
        observations=[
            {
                "row_index": 0,
                "send_id": send_id,
                "request_wire_seen": request_wire_seen,
                "response_direction_seen": response_direction_seen,
                "terminal_event_seen": terminal_event_seen,
            }
        ],
        channel_inventory={},
        provenance_summary={},
    )


class FakePcapReader:
    def __init__(self, results: list[PcapParseResult]):
        self.results = list(results)
        self.index = 0
        self.current = self.results[0] if self.results else PcapParseResult(packet_count=0, linktype=1, packets=[], connections=[])

    def update(self, path: Path) -> None:
        del path
        if self.index < len(self.results):
            self.current = self.results[self.index]
            self.index += 1

    def parse_result(self) -> PcapParseResult:
        return self.current


class CaptureHardeningTests(unittest.TestCase):
    def test_row_run_id_prefers_meta_run_id_when_top_level_sentinel_is_negative(self) -> None:
        row = {
            "run_id": -1,
            "meta_json": json.dumps({"run_id": 0}),
        }
        self.assertEqual(row_run_id(row), 0)

    def test_auto_backend_prefers_rancher_when_rdctl_is_present(self) -> None:
        with patch("tools.fprime_real.pcap_capture.has_rdctl_vm", return_value=True):
            self.assertEqual(resolve_capture_backend("auto"), CAPTURE_BACKEND_RDCTL_VM_BRIDGE)

    def test_auto_backend_uses_local_docker_bridge_when_rdctl_is_absent(self) -> None:
        with patch("tools.fprime_real.pcap_capture.has_rdctl_vm", return_value=False):
            with patch("tools.fprime_real.pcap_capture.local_docker_bridge_available", return_value=(True, "br-test")):
                self.assertEqual(resolve_capture_backend("auto"), CAPTURE_BACKEND_LOCAL_DOCKER_BRIDGE)

    def test_backend_override_wins_deterministically(self) -> None:
        with patch.dict(os.environ, {"FPRIME_CAPTURE_BACKEND": CAPTURE_BACKEND_LOCAL_DOCKER_BRIDGE}, clear=False):
            with patch("tools.fprime_real.pcap_capture.local_docker_bridge_available", return_value=(True, "br-test")):
                self.assertEqual(resolve_capture_backend(), CAPTURE_BACKEND_LOCAL_DOCKER_BRIDGE)

    def test_unsupported_host_fails_fast(self) -> None:
        with patch("tools.fprime_real.pcap_capture.has_rdctl_vm", return_value=False):
            with patch("tools.fprime_real.pcap_capture.local_docker_bridge_available", return_value=(False, "missing bridge")):
                with patch("tools.fprime_real.pcap_capture.sys.platform", "darwin"):
                    with self.assertRaises(SystemExit) as exc:
                        resolve_capture_backend("auto")
        self.assertIn("unsupported", str(exc.exception).lower())

    def test_auto_capture_interface_prefers_bridge_candidates(self) -> None:
        attempted: list[str] = []

        def fake_validate(**kwargs):
            attempted.append(str(kwargs["interface"]))
            interface = str(kwargs["interface"])
            return interface == "bridge100", {"interface": interface, "matched_identity_request": interface == "bridge100"}

        with patch("tools.fprime_real.support_probe.resolve_capture_backend", return_value=CAPTURE_BACKEND_RDCTL_VM_BRIDGE):
            with patch("tools.fprime_real.support_probe.list_capture_interfaces", return_value=["lo0", "vmenet0", "bridge100", "en0"]):
                with patch("tools.fprime_real.support_probe.preferred_capture_interface", return_value=None):
                    with patch("tools.fprime_real.support_probe.env_capture_interface", return_value=""):
                        with patch("tools.fprime_real.support_probe.validate_capture_interface", side_effect=fake_validate):
                            chosen = resolve_identity_capture_target(
                                REPO_ROOT,
                                REPO_ROOT / "orchestration" / "docker-compose.fprime-real.yml",
                                timeout_seconds=5.0,
                                runtime_root=Path("/tmp/runtime"),
                                dictionary="/workspace/dict.xml",
                            )
        self.assertEqual(chosen.interface, "bridge100")
        self.assertEqual(chosen.backend, CAPTURE_BACKEND_RDCTL_VM_BRIDGE)
        self.assertEqual(attempted[0], "bridge100")

    def test_explicit_loopback_override_fails_validation(self) -> None:
        with patch("tools.fprime_real.support_probe.resolve_capture_backend", return_value=CAPTURE_BACKEND_RDCTL_VM_BRIDGE):
            with patch("tools.fprime_real.support_probe.env_capture_interface", return_value=""):
                with patch("tools.fprime_real.support_probe.list_capture_interfaces", return_value=["lo0", "bridge100"]):
                    with patch("tools.fprime_real.support_probe.preferred_capture_interface", return_value=None):
                        with patch("tools.fprime_real.support_probe.env_capture_interface", return_value="lo0"):
                            with patch(
                                "tools.fprime_real.support_probe.validate_capture_interface",
                                return_value=(False, {"interface": "lo0", "reason": "loopback collapsed sender identity"}),
                            ):
                                with self.assertRaises(SystemExit) as exc:
                                    resolve_identity_capture_target(
                                        REPO_ROOT,
                                        REPO_ROOT / "orchestration" / "docker-compose.fprime-real.yml",
                                        timeout_seconds=5.0,
                                        runtime_root=Path("/tmp/runtime"),
                                        dictionary="/workspace/dict.xml",
                                    )
        self.assertIn("lo0", str(exc.exception))

    def test_wait_for_capture_drain_returns_after_quiet_ready_state(self) -> None:
        expected_rows = [
            {
                "send_id": "send-001",
                "command": "cmdDisp.CMD_NO_OP",
                "source_service": "ops_b1",
                "target_service": "fprime_a",
                "gds_accept": 1,
                "timeout": 0,
            }
        ]
        results = [
            packet_result_for("send-001", request_wire_seen=False, response_direction_seen=False, terminal_event_seen=False),
            packet_result_for("send-001", request_wire_seen=True, response_direction_seen=True, terminal_event_seen=False),
            packet_result_for("send-001", request_wire_seen=True, response_direction_seen=True, terminal_event_seen=False),
        ]
        fake_clock = FakeClock(sleep_increment=0.2)
        fake_reader = FakePcapReader(
            [
                PcapParseResult(packet_count=0, linktype=1, packets=[], connections=[]),
                PcapParseResult(packet_count=3, linktype=1, packets=[], connections=[]),
                PcapParseResult(packet_count=3, linktype=1, packets=[], connections=[]),
            ]
        )

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            send_log_path = tmp_path / "send_log.jsonl"
            send_log_path.write_text(json.dumps(expected_rows[0]) + "\n", encoding="utf-8")
            event_log_paths = {
                "fprime_a": tmp_path / "node_a_event.log",
                "fprime_b": tmp_path / "node_b_event.log",
            }
            for path in event_log_paths.values():
                path.write_text("", encoding="utf-8")

            with patch("tools.fprime_real.support_probe.PcapIncrementalReader", return_value=fake_reader):
                with patch("tools.fprime_real.support_probe.build_packets_from_parsed_sources", side_effect=results):
                    with patch("tools.fprime_real.support_probe.time.time", side_effect=fake_clock.time):
                        with patch("tools.fprime_real.support_probe.time.sleep", side_effect=fake_clock.sleep):
                            wait_for_capture_drain(
                                expected_rows,
                                event_log_paths=event_log_paths,
                                send_log_path=send_log_path,
                                pcap_path=tmp_path / "traffic.pcap",
                                capture_interface="bridge100",
                                timeout_seconds=1.0,
                                quiet_period_seconds=0.2,
                            )

    def test_wait_for_capture_drain_times_out_with_pending_preview(self) -> None:
        expected_rows = [
            {
                "send_id": "send-002",
                "command": "cmdDisp.CMD_NO_OP",
                "source_service": "ops_b1",
                "target_service": "fprime_a",
                "gds_accept": 1,
                "timeout": 0,
            }
        ]
        fake_clock = FakeClock(sleep_increment=5.5)

        with tempfile.TemporaryDirectory() as tmpdir:
            tmp_path = Path(tmpdir)
            send_log_path = tmp_path / "send_log.jsonl"
            send_log_path.write_text(json.dumps(expected_rows[0]) + "\n", encoding="utf-8")
            event_log_paths = {
                "fprime_a": tmp_path / "node_a_event.log",
                "fprime_b": tmp_path / "node_b_event.log",
            }
            for path in event_log_paths.values():
                path.write_text("", encoding="utf-8")

            with patch(
                "tools.fprime_real.support_probe.PcapIncrementalReader",
                return_value=FakePcapReader([PcapParseResult(packet_count=0, linktype=1, packets=[], connections=[])]),
            ):
                with patch(
                    "tools.fprime_real.support_probe.build_packets_from_parsed_sources",
                    return_value=packet_result_for("send-002", request_wire_seen=False, response_direction_seen=False, terminal_event_seen=False),
                ):
                    with patch("tools.fprime_real.support_probe.time.time", side_effect=fake_clock.time):
                        with patch("tools.fprime_real.support_probe.time.sleep", side_effect=fake_clock.sleep):
                            with self.assertRaises(SystemExit) as exc:
                                wait_for_capture_drain(
                                    expected_rows,
                                    event_log_paths=event_log_paths,
                                    send_log_path=send_log_path,
                                    pcap_path=tmp_path / "traffic.pcap",
                                    capture_interface="bridge100",
                                    timeout_seconds=1.0,
                                    quiet_period_seconds=0.2,
                                )
        self.assertIn("Timed out draining capture before shutdown", str(exc.exception))


if __name__ == "__main__":
    unittest.main()
