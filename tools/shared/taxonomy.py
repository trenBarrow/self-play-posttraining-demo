from __future__ import annotations

from functools import lru_cache
from pathlib import Path
from typing import Any

import yaml

from runtime import COMMAND_NAMES

REPO_ROOT = Path(__file__).resolve().parents[2]
TAXONOMY_DIR = REPO_ROOT / "configs" / "semantic_taxonomy"
CANONICAL_COMMAND_FAMILIES_PATH = TAXONOMY_DIR / "canonical_command_families.yaml"
FPRIME_COMMAND_SEMANTICS_PATH = TAXONOMY_DIR / "fprime_command_semantics.yaml"
MAVLINK_COMMAND_SEMANTICS_PATH = TAXONOMY_DIR / "mavlink_command_semantics.yaml"


class CommandTaxonomyError(ValueError):
    """Raised when command taxonomy configuration or lookup is invalid."""


def _load_yaml(path: Path) -> dict[str, Any]:
    return yaml.safe_load(path.read_text(encoding="utf-8"))


@lru_cache(maxsize=None)
def load_canonical_command_families() -> dict[str, Any]:
    return _load_yaml(CANONICAL_COMMAND_FAMILIES_PATH)


@lru_cache(maxsize=None)
def load_fprime_command_semantics() -> dict[str, Any]:
    return _load_yaml(FPRIME_COMMAND_SEMANTICS_PATH)


@lru_cache(maxsize=None)
def load_mavlink_command_semantics() -> dict[str, Any]:
    return _load_yaml(MAVLINK_COMMAND_SEMANTICS_PATH)


def _mapping_for_protocol(protocol_family: str) -> dict[str, Any]:
    normalized = str(protocol_family).strip().lower()
    if normalized == "fprime":
        return load_fprime_command_semantics()
    if normalized == "mavlink":
        return load_mavlink_command_semantics()
    raise CommandTaxonomyError(f"Unsupported taxonomy protocol family: {protocol_family!r}")


def _family_defaults(family_name: str) -> dict[str, Any]:
    families = load_canonical_command_families().get("families", {})
    family = families.get(family_name)
    if not isinstance(family, dict):
        raise CommandTaxonomyError(f"Unknown canonical command family: {family_name!r}")
    return {
        "canonical_command_family": family_name,
        "mutation_scope": str(family.get("default_mutation_scope", "unknown")),
        "persistence_class": str(family.get("default_persistence_class", "unknown")),
        "safety_criticality": str(family.get("default_safety_criticality", "unknown")),
        "authority_level": str(family.get("default_authority_level", "unknown")),
        "target_scope": str(family.get("default_target_scope", "unspecified")),
    }


def validate_command_taxonomy() -> dict[str, Any]:
    families = load_canonical_command_families().get("families", {})
    if not isinstance(families, dict) or not families:
        raise CommandTaxonomyError("Canonical command family taxonomy is missing families")

    missing_required_family_fields: dict[str, list[str]] = {}
    for family_name, payload in families.items():
        if not isinstance(payload, dict):
            raise CommandTaxonomyError(f"Family {family_name!r} must map to an object")
        missing = [
            key
            for key in (
                "description",
                "default_mutation_scope",
                "default_persistence_class",
                "default_safety_criticality",
                "default_authority_level",
                "default_target_scope",
            )
            if key not in payload
        ]
        if missing:
            missing_required_family_fields[family_name] = missing
    if missing_required_family_fields:
        raise CommandTaxonomyError(
            f"Canonical command families missing required fields: {missing_required_family_fields}"
        )

    protocol_summaries: dict[str, Any] = {}
    for protocol_family, loader in (
        ("fprime", load_fprime_command_semantics),
        ("mavlink", load_mavlink_command_semantics),
    ):
        payload = loader()
        commands = payload.get("commands", {})
        if not isinstance(commands, dict) or not commands:
            raise CommandTaxonomyError(f"{protocol_family} taxonomy is missing command mappings")
        invalid_families: dict[str, str] = {}
        missing_required_fields: dict[str, list[str]] = {}
        for raw_name, entry in commands.items():
            if not isinstance(entry, dict):
                raise CommandTaxonomyError(f"{protocol_family} command {raw_name!r} must map to an object")
            missing = [key for key in ("canonical_command_name", "canonical_command_family", "summary") if key not in entry]
            if missing:
                missing_required_fields[raw_name] = missing
            family_name = entry.get("canonical_command_family")
            if family_name not in families:
                invalid_families[raw_name] = str(family_name)
        if missing_required_fields:
            raise CommandTaxonomyError(
                f"{protocol_family} command mappings missing required fields: {missing_required_fields}"
            )
        if invalid_families:
            raise CommandTaxonomyError(
                f"{protocol_family} command mappings reference unknown families: {invalid_families}"
            )

        if protocol_family == "fprime":
            missing_runtime_commands = sorted(set(COMMAND_NAMES) - set(commands))
            if missing_runtime_commands:
                raise CommandTaxonomyError(
                    f"Fprime command taxonomy is missing runtime-supported commands: {missing_runtime_commands}"
                )
            protocol_summaries[protocol_family] = {
                "mapped_command_count": len(commands),
                "missing_runtime_commands": [],
            }
        else:
            protocol_summaries[protocol_family] = {
                "mapped_command_count": len(commands),
            }

    return {
        "family_count": len(families),
        "protocols": protocol_summaries,
    }


def resolve_command_semantics(
    protocol_family: str,
    raw_command_name: str | None,
    *,
    allow_unknown: bool = True,
) -> dict[str, Any]:
    command_name = str(raw_command_name).strip() if raw_command_name is not None else ""
    if not command_name:
        if allow_unknown:
            return _family_defaults("other_or_unknown") | {"canonical_command_name": None}
        raise CommandTaxonomyError("Command taxonomy lookup requires a non-empty raw command name")

    mapping = _mapping_for_protocol(protocol_family)
    commands = mapping.get("commands", {})
    entry = commands.get(command_name)
    if not isinstance(entry, dict):
        if allow_unknown:
            return _family_defaults("other_or_unknown") | {"canonical_command_name": None}
        raise CommandTaxonomyError(
            f"Unsupported {protocol_family} command {command_name!r}; add an explicit taxonomy entry"
        )

    family_name = str(entry["canonical_command_family"])
    merged = _family_defaults(family_name)
    merged.update(
        {
            "canonical_command_name": entry.get("canonical_command_name"),
            "canonical_command_family": family_name,
            "mutation_scope": entry.get("mutation_scope", merged["mutation_scope"]),
            "persistence_class": entry.get("persistence_class", merged["persistence_class"]),
            "safety_criticality": entry.get("safety_criticality", merged["safety_criticality"]),
            "authority_level": entry.get("authority_level", merged["authority_level"]),
            "target_scope": entry.get("target_scope", merged["target_scope"]),
        }
    )
    return merged
