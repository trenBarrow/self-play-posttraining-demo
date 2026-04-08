from __future__ import annotations

import sys
import tempfile
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.mavlink_real.schedule_profiles import (
    INTENT_CONTEXT_BENIGN_CLEAN,
    INTENT_CONTEXT_BENIGN_NOISY,
    SCHEDULE_COLUMNS,
    build_benign_rows,
    build_cyber_rows,
    build_fault_rows,
    load_schedule_csv,
    schedule_digest,
    supported_command_rows,
    supported_source_rows,
    validate_schedule_rows,
    write_schedule_csv,
)


class MavlinkScheduleProfilesTests(unittest.TestCase):
    def test_supported_catalogs_cover_expected_surface(self) -> None:
        commands = {row["command"]: row for row in supported_command_rows()}
        self.assertIn("REQUEST_AUTOPILOT_CAPABILITIES", commands)
        self.assertIn("PARAM_SET", commands)
        self.assertIn("MISSION_CLEAR_ALL", commands)
        self.assertEqual(commands["PARAM_SET"]["command_family"], "persistent_configuration")

        sources = {row["source_service"]: row for row in supported_source_rows()}
        self.assertEqual(sources["ops_primary"]["source_system_id"], 21)
        self.assertEqual(sources["red_secondary"]["source_component_id"], 191)

    def test_benign_schedule_is_seed_reproducible(self) -> None:
        rows_a = build_benign_rows(target_rows=36, seed=7)
        rows_b = build_benign_rows(target_rows=36, seed=7)
        self.assertEqual(schedule_digest(rows_a), schedule_digest(rows_b))

    def test_benign_schedule_marks_clean_and_noisy_rows(self) -> None:
        rows = build_benign_rows(target_rows=54, seed=7)
        intent_contexts = [str(dict(row["meta"]).get("intent_context", "")) for row in rows]
        self.assertIn(INTENT_CONTEXT_BENIGN_CLEAN, intent_contexts)
        self.assertIn(INTENT_CONTEXT_BENIGN_NOISY, intent_contexts)

    def test_cyber_schedule_uses_external_and_shared_identity_roles(self) -> None:
        rows = build_cyber_rows(target_rows=72, seed=7)
        roles = {str(dict(row["meta"]).get("actor_role", "")) for row in rows}
        commands = {str(row["command"]) for row in rows}
        self.assertIn("external", roles)
        self.assertIn("shared_identity", roles)
        self.assertIn("PARAM_SET", commands)
        self.assertIn("MISSION_CLEAR_ALL", commands)

    def test_fault_schedule_contains_recovery_commands(self) -> None:
        rows = build_fault_rows(target_rows=48, seed=7)
        commands = {str(row["command"]) for row in rows}
        self.assertIn("MAV_CMD_NAV_RETURN_TO_LAUNCH", commands)
        self.assertIn("MAV_CMD_COMPONENT_ARM_DISARM", commands)
        self.assertIn("MISSION_CLEAR_ALL", commands)

    def test_schedule_csv_round_trip_preserves_rows(self) -> None:
        rows = build_benign_rows(target_rows=24, seed=9)
        validate_schedule_rows(rows)
        with tempfile.TemporaryDirectory() as tmpdir:
            path = Path(tmpdir) / "schedule.csv"
            write_schedule_csv(path, rows)
            loaded = load_schedule_csv(path)
        self.assertEqual(schedule_digest(rows), schedule_digest(loaded))

    def test_schedule_csv_header_stays_stable(self) -> None:
        self.assertEqual(
            SCHEDULE_COLUMNS,
            [
                "time_of_day",
                "source_service",
                "target_service",
                "target_endpoint",
                "command",
                "command_family",
                "arguments_json",
                "meta_json",
            ],
        )


if __name__ == "__main__":
    unittest.main()
