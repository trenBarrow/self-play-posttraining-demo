from __future__ import annotations

import sys
import unittest
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.fprime_real.telemetry_catalog import (
    MODELED_CANONICAL_STATE_DIMENSIONS,
    MODELED_TELEMETRY_ENTRIES,
    MODELED_TELEMETRY_SEMANTIC_CATEGORIES,
    TELEMETRY_BY_CHANNEL,
    catalog_bucket,
    convert_numeric_entry,
)


class TelemetryCatalogTests(unittest.TestCase):
    def test_modeled_numeric_channel_maps_and_converts(self) -> None:
        entry = TELEMETRY_BY_CHANNEL["rateGroup1.RgMaxTime"]
        self.assertEqual(catalog_bucket("rateGroup1.RgMaxTime"), "modeled")
        self.assertEqual(convert_numeric_entry(entry, "4200 us"), 4.2)

    def test_inventory_only_channel_is_catalogued_but_not_modeled(self) -> None:
        entry = TELEMETRY_BY_CHANNEL["systemResources.FRAMEWORK_VERSION"]
        self.assertEqual(entry.kind, "inventory_only")
        self.assertFalse(entry.enabled_for_model)
        self.assertEqual(catalog_bucket("systemResources.FRAMEWORK_VERSION"), "inventory_only")
        self.assertIsNone(convert_numeric_entry(entry, "v3.2.0"))
        self.assertEqual(entry.semantic_category, "inventory_only")
        self.assertEqual(entry.canonical_dimensions, ())

    def test_unknown_channel_stays_unknown(self) -> None:
        self.assertEqual(catalog_bucket("unknown.Channel"), "unknown")

    def test_modeled_channels_are_classified_into_semantic_categories(self) -> None:
        entry = TELEMETRY_BY_CHANNEL["cmdDisp.CommandErrors"]
        self.assertEqual(entry.semantic_category, "command_error_pressure")
        self.assertEqual(entry.canonical_dimensions, ("command_error_ratio",))
        self.assertIn("compute_load", MODELED_TELEMETRY_SEMANTIC_CATEGORIES)
        self.assertIn("scheduler_pressure", MODELED_TELEMETRY_SEMANTIC_CATEGORIES)
        self.assertIn("scheduler_pressure_ratio", MODELED_CANONICAL_STATE_DIMENSIONS)
        self.assertTrue(all(entry.semantic_category != "inventory_only" for entry in MODELED_TELEMETRY_ENTRIES))
        self.assertTrue(all(entry.canonical_dimensions for entry in MODELED_TELEMETRY_ENTRIES))


if __name__ == "__main__":
    unittest.main()
