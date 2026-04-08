#!/usr/bin/env python3
from __future__ import annotations

import json
import shutil
from pathlib import Path
from typing import Any, Mapping

SELF_PLAY_STATE_SCHEMA_VERSION = "self_play_state.v1"
SELF_PLAY_CHECKPOINT_SCHEMA_VERSION = "self_play_checkpoint.v1"
SELF_PLAY_REPORT_SCHEMA_VERSION = "self_play_report.v1"


class SelfPlayCheckpointError(ValueError):
    """Raised when self-play state or checkpoints are invalid."""


def _clone_json(value: Any) -> Any:
    return json.loads(json.dumps(value))


def _as_mapping(value: Any) -> dict[str, Any]:
    return value if isinstance(value, dict) else {}


def _text(value: Any) -> str | None:
    if value is None:
        return None
    text = str(value).strip()
    return text or None


def _save_json(path: Path, payload: Any) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(payload, indent=2), encoding="utf-8")


def _load_json(path: Path) -> dict[str, Any]:
    payload = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(payload, dict):
        raise SelfPlayCheckpointError(f"{path} must contain a JSON object")
    return payload


def checkpoint_id_for_round(round_index: int) -> str:
    return f"round_{int(round_index):04d}"


def round_slug(round_index: int) -> str:
    return checkpoint_id_for_round(round_index)


def round_dir(output_dir: Path, round_index: int) -> Path:
    return output_dir.resolve() / "rounds" / round_slug(round_index)


def state_path(output_dir: Path) -> Path:
    return output_dir.resolve() / "self_play_state.json"


def report_path(output_dir: Path) -> Path:
    return output_dir.resolve() / "self_play_report.json"


def checkpoint_root(output_dir: Path, side: str) -> Path:
    normalized_side = _text(side)
    if normalized_side not in {"blue", "red"}:
        raise SelfPlayCheckpointError(f"Unsupported checkpoint side {side!r}")
    return output_dir.resolve() / "checkpoints" / normalized_side


def checkpoint_dir(output_dir: Path, side: str, checkpoint_id: str) -> Path:
    normalized_checkpoint_id = _text(checkpoint_id)
    if normalized_checkpoint_id is None:
        raise SelfPlayCheckpointError("checkpoint_id must be non-empty")
    return checkpoint_root(output_dir, side) / normalized_checkpoint_id


def checkpoint_artifact_dir(output_dir: Path, side: str, checkpoint_id: str) -> Path:
    return checkpoint_dir(output_dir, side, checkpoint_id) / "artifacts"


def reset_directory(path: Path) -> None:
    if path.exists():
        shutil.rmtree(path)
    path.mkdir(parents=True, exist_ok=True)


def copy_tree(source_dir: Path, destination_dir: Path) -> None:
    resolved_source = source_dir.resolve()
    if not resolved_source.exists() or not resolved_source.is_dir():
        raise SelfPlayCheckpointError(f"Missing checkpoint source directory: {resolved_source}")
    resolved_destination = destination_dir.resolve()
    if resolved_destination.exists():
        shutil.rmtree(resolved_destination)
    shutil.copytree(resolved_source, resolved_destination)


def create_directory_checkpoint(
    *,
    output_dir: Path,
    side: str,
    checkpoint_id: str,
    source_dir: Path,
    metadata: Mapping[str, Any] | None = None,
) -> dict[str, Any]:
    resolved_source = source_dir.resolve()
    destination_root = checkpoint_dir(output_dir, side, checkpoint_id)
    artifact_dir = destination_root / "artifacts"
    if destination_root.exists():
        shutil.rmtree(destination_root)
    destination_root.mkdir(parents=True, exist_ok=True)
    copy_tree(resolved_source, artifact_dir)
    payload = {
        "schema_version": SELF_PLAY_CHECKPOINT_SCHEMA_VERSION,
        "record_kind": "self_play_checkpoint",
        "side": side,
        "checkpoint_id": checkpoint_id,
        "artifact_dir": str(artifact_dir.resolve()),
        "metadata": _clone_json(dict(metadata or {})),
    }
    _save_json(destination_root / "checkpoint.json", payload)
    return payload


def load_checkpoint_metadata(output_dir: Path, side: str, checkpoint_id: str) -> dict[str, Any]:
    return _load_json(checkpoint_dir(output_dir, side, checkpoint_id) / "checkpoint.json")


def initialize_self_play_state(
    *,
    output_dir: Path,
    config: Mapping[str, Any],
    blue_checkpoint: Mapping[str, Any],
    red_checkpoint: Mapping[str, Any],
) -> dict[str, Any]:
    state = {
        "schema_version": SELF_PLAY_STATE_SCHEMA_VERSION,
        "record_kind": "self_play_state",
        "status": "initialized",
        "output_dir": str(output_dir.resolve()),
        "config": _clone_json(dict(config)),
        "rounds_completed": 0,
        "latest_blue_checkpoint": _clone_json(dict(blue_checkpoint)),
        "latest_red_checkpoint": _clone_json(dict(red_checkpoint)),
        "rounds": [],
    }
    write_self_play_state(output_dir, state)
    return state


def load_self_play_state(output_dir: Path) -> dict[str, Any] | None:
    path = state_path(output_dir)
    if not path.exists():
        return None
    payload = _load_json(path)
    if payload.get("schema_version") != SELF_PLAY_STATE_SCHEMA_VERSION:
        raise SelfPlayCheckpointError(
            f"{path} must contain {SELF_PLAY_STATE_SCHEMA_VERSION!r}; got {payload.get('schema_version')!r}"
        )
    if payload.get("record_kind") != "self_play_state":
        raise SelfPlayCheckpointError(f"{path} must contain record_kind='self_play_state'")
    return payload


def write_self_play_state(output_dir: Path, state: Mapping[str, Any]) -> Path:
    payload = _clone_json(dict(state))
    payload["schema_version"] = SELF_PLAY_STATE_SCHEMA_VERSION
    payload["record_kind"] = "self_play_state"
    output_path = state_path(output_dir)
    _save_json(output_path, payload)
    return output_path


def update_self_play_state(
    state: Mapping[str, Any],
    *,
    round_summary: Mapping[str, Any],
    latest_blue_checkpoint: Mapping[str, Any],
    latest_red_checkpoint: Mapping[str, Any],
    status: str,
) -> dict[str, Any]:
    updated = _clone_json(dict(state))
    rounds = list(updated.get("rounds", []))
    rounds.append(_clone_json(dict(round_summary)))
    updated["rounds"] = rounds
    updated["rounds_completed"] = len(rounds)
    updated["latest_blue_checkpoint"] = _clone_json(dict(latest_blue_checkpoint))
    updated["latest_red_checkpoint"] = _clone_json(dict(latest_red_checkpoint))
    updated["status"] = status
    return updated


def write_self_play_report(output_dir: Path, payload: Mapping[str, Any]) -> Path:
    output_path = report_path(output_dir)
    report = {
        "schema_version": SELF_PLAY_REPORT_SCHEMA_VERSION,
        "record_kind": "self_play_report",
        **_clone_json(dict(payload)),
    }
    _save_json(output_path, report)
    return output_path
