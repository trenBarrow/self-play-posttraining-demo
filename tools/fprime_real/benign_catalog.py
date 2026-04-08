#!/usr/bin/env python3
"""Shared benign support manifest loader for nominal scheduling and support probing."""

from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path

DEFAULT_MANIFEST_PATH = Path(__file__).with_name("benign_support_manifest.json")


@dataclass(frozen=True)
class ObservabilityRequirement:
    request_wire: bool
    op_dispatched: bool
    terminal_event: bool
    telemetry_recent: bool


@dataclass(frozen=True)
class BenignCommandSample:
    command: str
    arguments: tuple[str, ...]
    required_observability: ObservabilityRequirement


def load_benign_command_samples(manifest_path: Path | None = None) -> tuple[BenignCommandSample, ...]:
    source = manifest_path or DEFAULT_MANIFEST_PATH
    with source.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, list):
        raise ValueError(f"Expected manifest list in {source}")

    samples: list[BenignCommandSample] = []
    for item in payload:
        if not isinstance(item, dict):
            raise ValueError(f"Invalid benign manifest item in {source}: {item!r}")
        observability = item.get("required_observability", {})
        if not isinstance(observability, dict):
            raise ValueError(f"Invalid required_observability in {source}: {observability!r}")
        samples.append(
            BenignCommandSample(
                command=str(item["command"]),
                arguments=tuple(str(value) for value in item.get("arguments", [])),
                required_observability=ObservabilityRequirement(
                    request_wire=bool(observability.get("request_wire", True)),
                    op_dispatched=bool(observability.get("op_dispatched", True)),
                    terminal_event=bool(observability.get("terminal_event", True)),
                    telemetry_recent=bool(observability.get("telemetry_recent", True)),
                ),
            )
        )
    return tuple(samples)


BENIGN_COMMAND_SAMPLES = load_benign_command_samples()


def benign_command_for(index: int, manifest_path: Path | None = None) -> tuple[str, list[str]]:
    samples = BENIGN_COMMAND_SAMPLES if manifest_path is None else load_benign_command_samples(manifest_path)
    sample = samples[index % len(samples)]
    return sample.command, list(sample.arguments)
