from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from runtime import COMMAND_NAMES
from tools.shared.taxonomy import (
    CommandTaxonomyError,
    load_canonical_command_families,
    load_fprime_command_semantics,
    load_mavlink_command_semantics,
    resolve_command_semantics,
    validate_command_taxonomy,
)


class CommandTaxonomyTests(unittest.TestCase):
    def test_canonical_command_families_stay_protocol_neutral(self) -> None:
        names = {name.lower() for name in load_canonical_command_families()["families"]}
        self.assertFalse(any("fprime" in name for name in names))
        self.assertFalse(any("mavlink" in name for name in names))

    def test_validate_command_taxonomy_covers_runtime_fprime_commands(self) -> None:
        summary = validate_command_taxonomy()
        self.assertEqual(summary["protocols"]["fprime"]["mapped_command_count"], len(COMMAND_NAMES))
        self.assertEqual(summary["protocols"]["fprime"]["missing_runtime_commands"], [])

    def test_fprime_taxonomy_entries_match_runtime_command_inventory(self) -> None:
        mapped = set(load_fprime_command_semantics()["commands"])
        self.assertEqual(mapped, set(COMMAND_NAMES))

    def test_mavlink_taxonomy_contains_planned_real_command_surface(self) -> None:
        mapped = set(load_mavlink_command_semantics()["commands"])
        self.assertIn("MAV_CMD_COMPONENT_ARM_DISARM", mapped)
        self.assertIn("MAV_CMD_NAV_TAKEOFF", mapped)
        self.assertIn("PARAM_SET", mapped)
        self.assertIn("FILE_TRANSFER_PROTOCOL", mapped)

    def test_resolve_command_semantics_returns_family_defaults_and_overrides(self) -> None:
        fprime_semantics = resolve_command_semantics("fprime", "fileManager.RemoveFile", allow_unknown=False)
        mavlink_semantics = resolve_command_semantics("mavlink", "MAV_CMD_NAV_TAKEOFF", allow_unknown=False)

        self.assertEqual(fprime_semantics["canonical_command_family"], "file_payload_management")
        self.assertEqual(fprime_semantics["safety_criticality"], "high")
        self.assertEqual(mavlink_semantics["canonical_command_name"], "takeoff")
        self.assertEqual(mavlink_semantics["canonical_command_family"], "mission_sequence_control")
        self.assertEqual(mavlink_semantics["safety_criticality"], "critical")

    def test_unknown_commands_are_explicitly_handled(self) -> None:
        unknown = resolve_command_semantics("mavlink", "MAV_CMD_UNKNOWN_EXAMPLE", allow_unknown=True)
        self.assertEqual(unknown["canonical_command_family"], "other_or_unknown")
        self.assertEqual(unknown["mutation_scope"], "unknown")
        with self.assertRaises(CommandTaxonomyError):
            resolve_command_semantics("mavlink", "MAV_CMD_UNKNOWN_EXAMPLE", allow_unknown=False)


if __name__ == "__main__":
    unittest.main()
