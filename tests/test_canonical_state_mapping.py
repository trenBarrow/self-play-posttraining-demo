from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.shared.canonical_state import CANONICAL_STATE_FIELDS, summarize_normalized_state


class CanonicalStateMappingTests(unittest.TestCase):
    def test_fprime_state_snapshot_maps_into_protocol_neutral_ratios(self) -> None:
        raw_transaction = {
            "protocol_family": "fprime",
            "native_state_snapshot": {
                "target_fields": {
                    "cpu_total_pct": 42.0,
                    "cpu_00_pct": 18.0,
                    "cpu_01_pct": 24.0,
                    "blockdrv_cycles_1m": 16.0,
                    "cmds_dispatched_1m": 6.0,
                    "cmd_errors_1m": 2.0,
                    "filemanager_errors_1m": 2.0,
                    "filedownlink_warnings_1m": 1.0,
                    "hibuffs_total": 2.0,
                    "rg1_max_time_ms": 8.0,
                    "rg2_max_time_ms": 7.5,
                    "telemetry_age_ms": 1000.0,
                },
                "peer_fields": {
                    "cpu_total_pct": 21.0,
                    "cpu_00_pct": 10.0,
                    "cpu_01_pct": 11.0,
                    "blockdrv_cycles_1m": 8.0,
                    "cmds_dispatched_1m": 3.0,
                    "cmd_errors_1m": 1.0,
                    "filemanager_errors_1m": 0.0,
                    "filedownlink_warnings_1m": 1.0,
                    "hibuffs_total": 1.0,
                    "rg1_max_time_ms": 4.0,
                    "rg2_max_time_ms": 5.0,
                    "telemetry_age_ms": 500.0,
                },
            },
        }

        state = summarize_normalized_state(raw_transaction)

        self.assertEqual(set(state), set(CANONICAL_STATE_FIELDS))
        self.assertTrue(state["state_available"])
        self.assertTrue(state["target_state_present"])
        self.assertTrue(state["peer_state_present"])
        self.assertAlmostEqual(state["target_compute_load_ratio"], 0.42)
        self.assertAlmostEqual(state["target_compute_peak_load_ratio"], 0.24)
        self.assertAlmostEqual(state["target_compute_imbalance_ratio"], 0.06)
        self.assertAlmostEqual(state["target_storage_io_pressure_ratio"], 0.5)
        self.assertAlmostEqual(state["target_command_activity_ratio"], 0.5)
        self.assertAlmostEqual(state["target_command_error_ratio"], 0.5)
        self.assertAlmostEqual(state["target_service_issue_ratio"], 0.75)
        self.assertAlmostEqual(state["target_queue_pressure_ratio"], 0.5)
        self.assertAlmostEqual(state["target_scheduler_pressure_ratio"], 0.8)
        self.assertIsNone(state["target_link_pressure_ratio"])
        self.assertIsNone(state["target_power_pressure_ratio"])
        self.assertIsNone(state["target_control_instability_ratio"])
        self.assertIsNone(state["target_navigation_uncertainty_ratio"])
        self.assertAlmostEqual(state["target_telemetry_staleness_ratio"], 0.2)
        self.assertAlmostEqual(state["peer_compute_load_ratio"], 0.21)
        self.assertAlmostEqual(state["peer_storage_io_pressure_ratio"], 0.25)
        self.assertAlmostEqual(state["peer_command_error_ratio"], 0.25)
        self.assertAlmostEqual(state["peer_service_issue_ratio"], 0.25)
        self.assertAlmostEqual(state["peer_scheduler_pressure_ratio"], 0.5)
        self.assertAlmostEqual(state["peer_telemetry_staleness_ratio"], 0.1)

    def test_mavlink_state_snapshot_maps_into_same_canonical_field_names(self) -> None:
        state = summarize_normalized_state(
            {
                "protocol_family": "mavlink",
                "timing": {"submitted_at_ms": 1000.0},
                "native_state_snapshot": {
                    "snapshot_observed_at_ms": 900.0,
                    "target_fields": {
                        "sys_load_fraction": 0.42,
                        "drop_rate_comm_fraction": 0.12,
                        "battery_remaining_pct": 40.0,
                        "battery_status_remaining_pct": 35.0,
                        "power_vcc_v": 4.6,
                        "power_servo_v": 4.9,
                        "heartbeat_system_status": 5,
                        "heartbeat_base_mode": 128,
                        "gps_fix_type": 2,
                        "gps_satellites_visible": 5,
                        "onboard_control_sensors_enabled": 35,
                        "onboard_control_sensors_health": 3,
                    },
                    "peer_fields": {},
                },
            }
        )

        self.assertEqual(set(state), set(CANONICAL_STATE_FIELDS))
        self.assertTrue(state["state_available"])
        self.assertTrue(state["target_state_present"])
        self.assertFalse(state["peer_state_present"])
        self.assertAlmostEqual(state["target_compute_load_ratio"], 0.42)
        self.assertAlmostEqual(state["target_compute_peak_load_ratio"], 0.42)
        self.assertAlmostEqual(state["target_link_pressure_ratio"], 0.12)
        self.assertAlmostEqual(state["target_power_pressure_ratio"], 0.65)
        self.assertAlmostEqual(state["target_control_instability_ratio"], 0.8)
        self.assertAlmostEqual(state["target_navigation_uncertainty_ratio"], 5.0 / 6.0)
        self.assertAlmostEqual(state["target_telemetry_staleness_ratio"], 0.02)
        self.assertIsNone(state["target_storage_io_pressure_ratio"])
        self.assertIsNone(state["target_scheduler_pressure_ratio"])
        self.assertIsNone(state["peer_compute_load_ratio"])
        self.assertIsNone(state["peer_link_pressure_ratio"])

    def test_unknown_protocol_does_not_guess_state_mapping(self) -> None:
        state = summarize_normalized_state(
            {
                "protocol_family": "custombus",
                "native_state_snapshot": {
                    "target_fields": {"cpu_total_pct": 70.0},
                    "peer_fields": {},
                },
            }
        )

        self.assertFalse(state["state_available"])
        self.assertFalse(state["target_state_present"])
        self.assertFalse(state["peer_state_present"])
        self.assertIsNone(state["target_compute_load_ratio"])
        self.assertIsNone(state["peer_compute_load_ratio"])


if __name__ == "__main__":
    unittest.main()
