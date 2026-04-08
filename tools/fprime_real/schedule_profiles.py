#!/usr/bin/env python3
"""Shared scenario catalogs and reporting helpers for real F' schedules."""

from __future__ import annotations

import csv
import hashlib
import json
import math
import random
import re
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

CLASS_NAMES = ("benign", "cyber", "fault")
PHASES = ("startup", "science", "downlink", "standby")
INTENT_CONTEXT_BENIGN_CLEAN = "benign_clean"
INTENT_CONTEXT_BENIGN_NOISY = "benign_noisy"
INTENT_CONTEXT_MALICIOUS = "malicious"
INTENT_CONTEXT_FAULT = "fault"
SCHEDULE_COLUMNS = [
    "time_of_day",
    "source_service",
    "target_service",
    "target_tts_port",
    "command",
    "arguments_json",
    "meta_json",
]

REFERENCE_FILE_PATH = "/workspace/README.md"
BENIGN_TEMP_ROOT = "/tmp/fprime_benign"
CYBER_TEMP_ROOT = "/tmp/fprime_redteam"
FAULT_TEMP_ROOT = "/tmp/fprime_fault"


@dataclass(frozen=True)
class SourceProfile:
    source_service: str
    target_service: str
    target_tts_port: int
    actor_role: str
    actor_trust: float


@dataclass(frozen=True)
class ScheduleCommand:
    attack_family: str
    command: str
    arguments: list[str]
    intent_context: str = ""


@dataclass
class EpisodeState:
    created_dirs: dict[str, list[str]] = field(
        default_factory=lambda: {"fprime_a": [], "fprime_b": []}
    )
    fallback_remove_dirs: dict[str, list[str]] = field(
        default_factory=lambda: {"fprime_a": [], "fprime_b": []}
    )
    value_scale: float = 1.0
    target_focus: str = ""
    precondition_profile: str = ""
    prefer_missing_reference: bool = False
    prefer_remove_stale_suffix: bool = False
    prefer_cleanup_retry: bool = False


GOOD_SOURCES = [
    SourceProfile("ops_a1", "fprime_b", 50050, "ops_primary", 0.98),
    SourceProfile("ops_a2", "fprime_b", 50050, "ops_backup", 0.91),
    SourceProfile("ops_b1", "fprime_a", 50050, "ops_primary", 0.97),
    SourceProfile("ops_b2", "fprime_a", 50050, "ops_backup", 0.90),
    # Benign traffic occasionally traverses a shared operator identity window.
    SourceProfile("ops_a1", "fprime_b", 50050, "shared_identity", 0.63),
]

BAD_SOURCES = [
    SourceProfile("red_a1", "fprime_b", 50050, "external", 0.22),
    SourceProfile("red_b1", "fprime_a", 50050, "external", 0.20),
    # Some cyber episodes deliberately blend into the shared-identity surface.
    SourceProfile("ops_a1", "fprime_b", 50050, "shared_identity", 0.58),
    SourceProfile("ops_b1", "fprime_a", 50050, "shared_identity", 0.61),
    # Some cyber episodes explicitly exercise stolen-credential / masquerade paths.
    SourceProfile("ops_a2", "fprime_b", 50050, "ops_backup", 0.91),
    SourceProfile("ops_b2", "fprime_a", 50050, "ops_primary", 0.96),
]

FAULT_SOURCES = [
    SourceProfile("ops_a1", "fprime_b", 50050, "ops_primary", 0.97),
    SourceProfile("ops_a2", "fprime_b", 50050, "ops_backup", 0.89),
    SourceProfile("ops_b1", "fprime_a", 50050, "ops_primary", 0.96),
    SourceProfile("ops_b2", "fprime_a", 50050, "ops_backup", 0.88),
    # Fault episodes include ambiguous shared-identity handoff cases.
    SourceProfile("ops_b1", "fprime_a", 50050, "shared_identity", 0.66),
    # Fault episodes also include degraded external-support relay behavior.
    SourceProfile("red_b1", "fprime_a", 50050, "external", 0.38),
]

VariantBuilder = Callable[[random.Random, EpisodeState, str, int, int], ScheduleCommand]


@dataclass(frozen=True)
class VariantSpec:
    name: str
    builder: VariantBuilder


@dataclass
class EpisodePlan:
    profile_name: str
    precondition_profile: str
    source_strategy: str
    gap_profile: str
    target_focus: str
    phase_sequence: tuple[str, ...]
    source_sequence: tuple[SourceProfile, ...]
    preferred_variants: dict[str, tuple[str, ...]]
    value_scale: float
    burst_steps: frozenset[int]


def fmt_hms(seconds: int) -> str:
    seconds %= 86400
    hh = seconds // 3600
    mm = (seconds % 3600) // 60
    ss = seconds % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def phase_for_step(step: int, episode_span: int) -> str:
    if episode_span <= 1:
        return PHASES[0]
    bucket = min(len(PHASES) - 1, (step * len(PHASES)) // max(1, episode_span))
    return PHASES[bucket]


def _created_dir(state: EpisodeState, target_service: str) -> str | None:
    queue = state.created_dirs.setdefault(target_service, [])
    if not queue:
        return None
    return queue.pop(0)


def _remember_dir(state: EpisodeState, target_service: str, path: str) -> None:
    state.created_dirs.setdefault(target_service, []).append(path)


def _remember_fallback_remove_dir(state: EpisodeState, target_service: str, path: str) -> None:
    state.fallback_remove_dirs.setdefault(target_service, []).append(path)


def _fallback_remove_dir(state: EpisodeState, target_service: str) -> str | None:
    queue = state.fallback_remove_dirs.setdefault(target_service, [])
    if not queue:
        return None
    return queue.pop(0)


def _variant(name: str, builder: VariantBuilder) -> VariantSpec:
    return VariantSpec(name=name, builder=builder)


def _no_op() -> ScheduleCommand:
    return ScheduleCommand("healthcheck", "cmdDisp.CMD_NO_OP", [])


def _no_op_string(prefix: str, episode_id: int, step: int) -> ScheduleCommand:
    return ScheduleCommand(
        "healthcheck",
        "cmdDisp.CMD_NO_OP_STRING",
        [f"{prefix}_{episode_id:04d}_{step:02d}"],
    )


def _scaled_numeric_choice(
    rng: random.Random,
    values: list[int] | list[float],
    state: EpisodeState,
    *,
    minimum: float | None = None,
    maximum: float | None = None,
    digits: int | None = None,
) -> int | float:
    chosen = float(rng.choice(values))
    scaled = chosen * state.value_scale
    if minimum is not None:
        scaled = max(minimum, scaled)
    if maximum is not None:
        scaled = min(maximum, scaled)
    if digits is None:
        return int(round(scaled))
    return round(scaled, digits)


def _test_args(
    rng: random.Random,
    state: EpisodeState,
    values_i32: list[int],
    values_f32: list[float],
    values_u8: list[int],
) -> list[str]:
    return [
        str(_scaled_numeric_choice(rng, values_i32, state, digits=None)),
        f"{_scaled_numeric_choice(rng, values_f32, state, digits=2):.2f}",
        str(_scaled_numeric_choice(rng, values_u8, state, minimum=0.0, maximum=16.0, digits=None)),
    ]


def _benign_test_cmd(rng: random.Random, state: EpisodeState, target_service: str, episode_id: int, step: int) -> ScheduleCommand:
    del target_service
    return ScheduleCommand(
        "dispatch_tuning",
        "cmdDisp.CMD_TEST_CMD_1",
        _test_args(
            rng,
            state,
            [-120, -96, -72, -48, -24, 0, 24, 48, 72, 96],
            [-3.50, -2.25, -1.25, -0.25, 0.50, 1.75, 3.00],
            [1, 2, 4, 6, 8, 12],
        ),
    )


def _cyber_test_cmd(rng: random.Random, state: EpisodeState, target_service: str, episode_id: int, step: int) -> ScheduleCommand:
    del target_service
    return ScheduleCommand(
        "masquerade_abuse",
        "cmdDisp.CMD_TEST_CMD_1",
        _test_args(
            rng,
            state,
            [-128, -96, -64, -32, 0, 32, 64, 96, 128],
            [-4.00, -2.50, -1.00, 0.25, 1.50, 2.75, 4.00],
            [1, 3, 5, 7, 9, 12, 16],
        ),
    )


def _fault_test_cmd(rng: random.Random, state: EpisodeState, target_service: str, episode_id: int, step: int) -> ScheduleCommand:
    del target_service
    return ScheduleCommand(
        "operator_recheck",
        "cmdDisp.CMD_TEST_CMD_1",
        _test_args(
            rng,
            state,
            [-140, -90, -54, -18, 18, 54, 90, 126],
            [-4.50, -2.00, -0.75, 0.00, 1.25, 2.50, 4.25],
            [0, 1, 3, 5, 7, 9, 11],
        ),
    )


def _version_cmd() -> ScheduleCommand:
    return ScheduleCommand("inventory_audit", "systemResources.VERSION", [])


def _dump_filter_cmd() -> ScheduleCommand:
    return ScheduleCommand("inventory_audit", "eventLogger.DUMP_FILTER_STATE", [])


def _sequence_mode_cmd(rng: random.Random, family: str) -> ScheduleCommand:
    return ScheduleCommand(family, rng.choice(["cmdSeq.CS_AUTO", "cmdSeq.CS_MANUAL"]), [])


def _clear_tracking_cmd(family: str) -> ScheduleCommand:
    return ScheduleCommand(family, "cmdDisp.CMD_CLEAR_TRACKING", [])


def _prm_save_cmd(family: str) -> ScheduleCommand:
    return ScheduleCommand(family, "prmDb.PRM_SAVE_FILE", [])


def _file_size_cmd(family: str, path: str = REFERENCE_FILE_PATH) -> ScheduleCommand:
    return ScheduleCommand(family, "fileManager.FileSize", [path])


def _send_partial_cmd(family: str, dest_name: str, offset: int, length: int, source_path: str = REFERENCE_FILE_PATH) -> ScheduleCommand:
    return ScheduleCommand(
        family,
        "fileDownlink.SendPartial",
        [source_path, dest_name, str(offset), str(length)],
    )


def _send_file_cmd(family: str, dest_name: str, source_path: str = REFERENCE_FILE_PATH) -> ScheduleCommand:
    return ScheduleCommand(family, "fileDownlink.SendFile", [source_path, dest_name])


def _cancel_cmd(family: str) -> ScheduleCommand:
    return ScheduleCommand(family, "fileDownlink.Cancel", [])


def _with_intent(command: ScheduleCommand, intent_context: str) -> ScheduleCommand:
    return ScheduleCommand(
        attack_family=command.attack_family,
        command=command.command,
        arguments=list(command.arguments),
        intent_context=intent_context,
    )


def _create_dir_cmd(root: str, family: str, state: EpisodeState, target_service: str, episode_id: int, step: int) -> ScheduleCommand:
    path = f"{root}/ep_{episode_id:04d}_{target_service}_{step:02d}"
    _remember_dir(state, target_service, path)
    return ScheduleCommand(family, "fileManager.CreateDirectory", [path])


def _remove_dir_cmd(root: str, fallback_name: str, family: str, state: EpisodeState, target_service: str, episode_id: int, step: int) -> ScheduleCommand:
    del episode_id, step
    path = _created_dir(state, target_service)
    if path is None:
        path = f"{root}/{fallback_name}"
    return ScheduleCommand(family, "fileManager.RemoveDirectory", [path])


def _benign_remove_dir_available(state: EpisodeState, target_service: str) -> bool:
    return bool(state.created_dirs.get(target_service) or state.fallback_remove_dirs.get(target_service))


def _benign_remove_dir_cmd(rng: random.Random, state: EpisodeState, target_service: str, episode_id: int, step: int) -> ScheduleCommand:
    del rng, episode_id, step
    path = _created_dir(state, target_service)
    if path is None:
        path = _fallback_remove_dir(state, target_service)
    return ScheduleCommand("ops_cleanup", "fileManager.RemoveDirectory", [path or f"{BENIGN_TEMP_ROOT}/remove_ready_dir"])


def _fault_remove_dir_cmd(rng: random.Random, state: EpisodeState, target_service: str, episode_id: int, step: int) -> ScheduleCommand:
    path = _created_dir(state, target_service)
    if path is not None and (state.prefer_remove_stale_suffix or rng.random() < 0.35):
        return ScheduleCommand("cleanup_mistype", "fileManager.RemoveDirectory", [f"{path}_stale"])
    return ScheduleCommand(
        "cleanup_mistype",
        "fileManager.RemoveDirectory",
        [f"{FAULT_TEMP_ROOT}/missing_ep_{episode_id:04d}_{target_service}_{step:02d}"],
    )


def _cyber_unknown_cmd(rng: random.Random, state: EpisodeState, target_service: str, episode_id: int, step: int) -> ScheduleCommand:
    del state, target_service, episode_id, step
    return ScheduleCommand("opcode_bruteforce", "cmdDisp.CMD_UNKNOWN_PROBE", [])


def _benign_downlink_partial(rng: random.Random, state: EpisodeState, target_service: str, episode_id: int, step: int) -> ScheduleCommand:
    del target_service
    return _send_partial_cmd(
        "downlink_window",
        f"ops_partial_{episode_id:04d}_{step:02d}.bin",
        int(_scaled_numeric_choice(rng, [0, 128, 256, 384], state, minimum=0.0, maximum=512.0, digits=None)),
        int(_scaled_numeric_choice(rng, [256, 512, 768], state, minimum=128.0, maximum=1024.0, digits=None)),
    )


def _cyber_downlink_partial(rng: random.Random, state: EpisodeState, target_service: str, episode_id: int, step: int) -> ScheduleCommand:
    del target_service
    return _send_partial_cmd(
        "downlink_abuse",
        f"collect_{episode_id:04d}_{step:02d}.bin",
        int(_scaled_numeric_choice(rng, [0, 128, 256, 512], state, minimum=0.0, maximum=1024.0, digits=None)),
        int(_scaled_numeric_choice(rng, [256, 512, 768, 1024], state, minimum=128.0, maximum=2048.0, digits=None)),
    )


def _fault_downlink_partial(rng: random.Random, state: EpisodeState, target_service: str, episode_id: int, step: int) -> ScheduleCommand:
    del target_service
    if state.prefer_missing_reference or rng.random() < 0.45:
        return _send_partial_cmd(
            "stale_reference",
            f"fault_missing_{episode_id:04d}_{step:02d}.bin",
            0,
            int(_scaled_numeric_choice(rng, [256, 512], state, minimum=128.0, maximum=768.0, digits=None)),
            source_path=f"{FAULT_TEMP_ROOT}/missing_ep_{episode_id:04d}_{step:02d}.bin",
        )
    return _send_partial_cmd(
        "untimely_downlink",
        f"fault_partial_{episode_id:04d}_{step:02d}.bin",
        int(_scaled_numeric_choice(rng, [0, 128, 256], state, minimum=0.0, maximum=384.0, digits=None)),
        int(_scaled_numeric_choice(rng, [256, 512, 768], state, minimum=128.0, maximum=1024.0, digits=None)),
    )


def _benign_downlink_full(rng: random.Random, state: EpisodeState, target_service: str, episode_id: int, step: int) -> ScheduleCommand:
    del rng, state, target_service
    return _send_file_cmd("downlink_window", f"ops_full_{episode_id:04d}_{step:02d}.bin")


def _benign_file_size_recheck(rng: random.Random, state: EpisodeState, target_service: str, episode_id: int, step: int) -> ScheduleCommand:
    del rng, state, target_service
    return _with_intent(
        _file_size_cmd(
            "inventory_recheck",
            path=f"{BENIGN_TEMP_ROOT}/stale_inventory_ep_{episode_id:04d}_{step:02d}.txt",
        ),
        INTENT_CONTEXT_BENIGN_NOISY,
    )


def _benign_cancel_recheck(rng: random.Random, state: EpisodeState, target_service: str, episode_id: int, step: int) -> ScheduleCommand:
    del rng, state, target_service, episode_id, step
    return _with_intent(_cancel_cmd("downlink_recheck"), INTENT_CONTEXT_BENIGN_NOISY)


def _benign_sequence_reissue(rng: random.Random, state: EpisodeState, target_service: str, episode_id: int, step: int) -> ScheduleCommand:
    del state, target_service, episode_id, step
    return _with_intent(_sequence_mode_cmd(rng, "sequence_reissue"), INTENT_CONTEXT_BENIGN_NOISY)


def _benign_remove_dir_retry(rng: random.Random, state: EpisodeState, target_service: str, episode_id: int, step: int) -> ScheduleCommand:
    del rng, state, target_service
    retry_path = f"{BENIGN_TEMP_ROOT}/carryover_ep_{episode_id:04d}_{step:02d}"
    return _with_intent(
        ScheduleCommand(
            "cleanup_retry",
            "fileManager.RemoveDirectory",
            [retry_path],
        ),
        INTENT_CONTEXT_BENIGN_NOISY,
    )


def _cyber_downlink_full(rng: random.Random, state: EpisodeState, target_service: str, episode_id: int, step: int) -> ScheduleCommand:
    del rng, state, target_service
    return _send_file_cmd("downlink_abuse", f"bulk_sync_{episode_id:04d}_{step:02d}.bin")


def _fault_downlink_full(rng: random.Random, state: EpisodeState, target_service: str, episode_id: int, step: int) -> ScheduleCommand:
    del target_service
    if state.prefer_missing_reference or rng.random() < 0.50:
        return _send_file_cmd(
            "stale_reference",
            f"fault_full_{episode_id:04d}_{step:02d}.bin",
            source_path=f"{FAULT_TEMP_ROOT}/missing_full_{episode_id:04d}_{step:02d}.bin",
        )
    return _send_file_cmd("untimely_downlink", f"fault_retry_{episode_id:04d}_{step:02d}.bin")


BENIGN_VARIANTS: dict[str, list[VariantSpec]] = {
    "startup": [
        _variant("noop", lambda rng, state, target_service, episode_id, step: _no_op()),
        _variant("noop_string", lambda rng, state, target_service, episode_id, step: _no_op_string("OPS", episode_id, step)),
        _variant("version", lambda rng, state, target_service, episode_id, step: _version_cmd()),
        _variant("dump_filter", lambda rng, state, target_service, episode_id, step: _dump_filter_cmd()),
        _variant("test_cmd", _benign_test_cmd),
        _variant("sequence_mode", lambda rng, state, target_service, episode_id, step: _sequence_mode_cmd(rng, "sequence_control")),
        _variant("sequence_reissue", _benign_sequence_reissue),
    ],
    "science": [
        _variant("test_cmd", _benign_test_cmd),
        _variant("prm_save", lambda rng, state, target_service, episode_id, step: _prm_save_cmd("config_save")),
        _variant("file_size", lambda rng, state, target_service, episode_id, step: _file_size_cmd("inventory_audit")),
        _variant(
            "create_dir",
            lambda rng, state, target_service, episode_id, step: _create_dir_cmd(BENIGN_TEMP_ROOT, "ops_cleanup", state, target_service, episode_id, step),
        ),
        _variant("sequence_mode", lambda rng, state, target_service, episode_id, step: _sequence_mode_cmd(rng, "sequence_control")),
        _variant("file_size_recheck", _benign_file_size_recheck),
    ],
    "downlink": [
        _variant("downlink_partial", _benign_downlink_partial),
        _variant("downlink_full", _benign_downlink_full),
        _variant("cancel_recheck", _benign_cancel_recheck),
        _variant("file_size", lambda rng, state, target_service, episode_id, step: _file_size_cmd("inventory_audit")),
    ],
    "standby": [
        _variant(
            "remove_dir", _benign_remove_dir_cmd,
        ),
        _variant("clear_tracking", lambda rng, state, target_service, episode_id, step: _clear_tracking_cmd("sequence_control")),
        _variant("sequence_mode", lambda rng, state, target_service, episode_id, step: _sequence_mode_cmd(rng, "sequence_control")),
        _variant("test_cmd", _benign_test_cmd),
        _variant("remove_dir_retry", _benign_remove_dir_retry),
    ],
}

CYBER_VARIANTS: dict[str, list[VariantSpec]] = {
    "startup": [
        _variant("unknown_probe", _cyber_unknown_cmd),
        _variant("noop_string", lambda rng, state, target_service, episode_id, step: _no_op_string("OPS_SYNC", episode_id, step)),
        _variant("version", lambda rng, state, target_service, episode_id, step: _version_cmd()),
        _variant("test_cmd", _cyber_test_cmd),
        _variant("sequence_mode", lambda rng, state, target_service, episode_id, step: _sequence_mode_cmd(rng, "masquerade_abuse")),
    ],
    "science": [
        _variant("test_cmd", _cyber_test_cmd),
        _variant("prm_save", lambda rng, state, target_service, episode_id, step: _prm_save_cmd("persistence_abuse")),
        _variant("file_size", lambda rng, state, target_service, episode_id, step: _file_size_cmd("recon_probe")),
        _variant(
            "create_dir",
            lambda rng, state, target_service, episode_id, step: _create_dir_cmd(CYBER_TEMP_ROOT, "state_tamper", state, target_service, episode_id, step),
        ),
        _variant("dump_filter", lambda rng, state, target_service, episode_id, step: _dump_filter_cmd()),
    ],
    "downlink": [
        _variant("downlink_partial", _cyber_downlink_partial),
        _variant("downlink_full", _cyber_downlink_full),
        _variant("cancel", lambda rng, state, target_service, episode_id, step: _cancel_cmd("downlink_abuse")),
        _variant("file_size", lambda rng, state, target_service, episode_id, step: _file_size_cmd("recon_probe")),
    ],
    "standby": [
        _variant(
            "remove_dir",
            lambda rng, state, target_service, episode_id, step: _remove_dir_cmd(CYBER_TEMP_ROOT, "staging", "state_tamper", state, target_service, episode_id, step),
        ),
        _variant("clear_tracking", lambda rng, state, target_service, episode_id, step: _clear_tracking_cmd("masquerade_abuse")),
        _variant("test_cmd", _cyber_test_cmd),
        _variant("unknown_probe", _cyber_unknown_cmd),
    ],
}

FAULT_VARIANTS: dict[str, list[VariantSpec]] = {
    "startup": [
        _variant("version", lambda rng, state, target_service, episode_id, step: _version_cmd()),
        _variant("dump_filter", lambda rng, state, target_service, episode_id, step: _dump_filter_cmd()),
        _variant("test_cmd", _fault_test_cmd),
        _variant("sequence_mode", lambda rng, state, target_service, episode_id, step: _sequence_mode_cmd(rng, "sequence_misconfig")),
        _variant("noop_string", lambda rng, state, target_service, episode_id, step: _no_op_string("OPS_RETRY", episode_id, step)),
    ],
    "science": [
        _variant("test_cmd", _fault_test_cmd),
        _variant("prm_save", lambda rng, state, target_service, episode_id, step: _prm_save_cmd("config_race")),
        _variant(
            "file_size",
            lambda rng, state, target_service, episode_id, step: _file_size_cmd(
                "stale_reference",
                path=f"{FAULT_TEMP_ROOT}/missing_ep_{episode_id:04d}_{step:02d}.txt"
                if state.prefer_missing_reference or rng.random() < 0.45
                else REFERENCE_FILE_PATH,
            ),
        ),
        _variant(
            "create_dir",
            lambda rng, state, target_service, episode_id, step: _create_dir_cmd(FAULT_TEMP_ROOT, "cleanup_mistype", state, target_service, episode_id, step),
        ),
        _variant("sequence_mode", lambda rng, state, target_service, episode_id, step: _sequence_mode_cmd(rng, "sequence_misconfig")),
    ],
    "downlink": [
        _variant("downlink_partial", _fault_downlink_partial),
        _variant("downlink_full", _fault_downlink_full),
        _variant("cancel", lambda rng, state, target_service, episode_id, step: _cancel_cmd("untimely_downlink")),
        _variant("file_size", lambda rng, state, target_service, episode_id, step: _file_size_cmd("stale_reference")),
    ],
    "standby": [
        _variant("remove_dir", _fault_remove_dir_cmd),
        _variant("clear_tracking", lambda rng, state, target_service, episode_id, step: _clear_tracking_cmd("operator_recheck")),
        _variant("test_cmd", _fault_test_cmd),
        _variant("sequence_mode", lambda rng, state, target_service, episode_id, step: _sequence_mode_cmd(rng, "sequence_misconfig")),
    ],
}


PHASE_TEMPLATES_BY_CLASS = {
    "benign": (
        ("startup", "science", "science", "downlink", "standby"),
        ("startup", "science", "downlink", "science", "standby"),
        ("startup", "science", "downlink", "downlink", "standby"),
        ("startup", "science", "standby", "downlink", "standby"),
    ),
    "cyber": (
        ("startup", "science", "downlink", "science", "standby"),
        ("startup", "science", "downlink", "standby", "downlink"),
        ("startup", "science", "startup", "downlink", "standby"),
        ("startup", "science", "downlink", "science", "downlink", "standby"),
    ),
    "fault": (
        ("startup", "science", "standby", "downlink", "standby"),
        ("startup", "science", "downlink", "science", "standby"),
        ("startup", "science", "downlink", "standby", "science"),
        ("startup", "standby", "science", "downlink", "standby"),
    ),
}

PRECONDITION_PROFILES_BY_CLASS = {
    "benign": ("steady_nominal", "cleanup_followup", "inventory_recheck", "handover_backlog"),
    "cyber": ("staged_drop", "shared_identity_window", "dual_target_probe", "downlink_surge"),
    "fault": ("stale_reference", "cleanup_drift", "sequence_desync", "retry_backlog"),
}

SOURCE_STRATEGIES_BY_CLASS = {
    "benign": ("round_robin", "handoff", "focus"),
    "cyber": ("handoff", "surge", "focus"),
    "fault": ("round_robin", "focus", "surge"),
}

GAP_PROFILES_BY_CLASS = {
    "benign": ("steady", "handoff", "staggered"),
    "cyber": ("surge", "handoff", "staggered"),
    "fault": ("steady", "drift", "staggered"),
}

PROFILE_VARIANT_PREFERENCES = {
    "benign": {
        "steady_nominal": {
            "startup": ("noop", "version", "test_cmd"),
            "science": ("test_cmd", "prm_save", "sequence_mode"),
            "downlink": ("downlink_partial", "downlink_full"),
            "standby": ("remove_dir", "clear_tracking"),
        },
        "cleanup_followup": {
            "science": ("create_dir", "file_size_recheck"),
            "standby": ("remove_dir", "remove_dir_retry"),
        },
        "inventory_recheck": {
            "startup": ("version", "dump_filter"),
            "science": ("file_size", "file_size_recheck"),
            "downlink": ("file_size", "cancel_recheck"),
        },
        "handover_backlog": {
            "startup": ("noop_string", "sequence_reissue"),
            "science": ("create_dir", "sequence_mode"),
            "downlink": ("downlink_partial", "cancel_recheck"),
            "standby": ("remove_dir", "sequence_mode"),
        },
    },
    "cyber": {
        "staged_drop": {
            "science": ("create_dir", "prm_save"),
            "downlink": ("downlink_partial", "downlink_full"),
            "standby": ("remove_dir", "clear_tracking"),
        },
        "shared_identity_window": {
            "startup": ("noop_string", "test_cmd", "sequence_mode"),
            "science": ("test_cmd", "dump_filter"),
            "standby": ("test_cmd", "unknown_probe"),
        },
        "dual_target_probe": {
            "startup": ("version", "unknown_probe"),
            "science": ("file_size", "dump_filter", "test_cmd"),
            "downlink": ("file_size", "downlink_partial"),
        },
        "downlink_surge": {
            "science": ("create_dir", "file_size"),
            "downlink": ("downlink_partial", "downlink_full", "cancel"),
            "standby": ("remove_dir", "unknown_probe"),
        },
    },
    "fault": {
        "stale_reference": {
            "science": ("file_size", "prm_save"),
            "downlink": ("downlink_partial", "downlink_full", "file_size"),
        },
        "cleanup_drift": {
            "science": ("create_dir", "test_cmd"),
            "standby": ("remove_dir", "clear_tracking"),
        },
        "sequence_desync": {
            "startup": ("sequence_mode", "noop_string"),
            "science": ("sequence_mode", "test_cmd"),
            "standby": ("sequence_mode", "clear_tracking"),
        },
        "retry_backlog": {
            "downlink": ("cancel", "downlink_partial", "file_size"),
            "standby": ("remove_dir", "test_cmd"),
        },
    },
}

VALUE_SCALE_RANGE_BY_CLASS = {
    "benign": (0.88, 1.18),
    "cyber": (0.74, 1.36),
    "fault": (0.80, 1.28),
}


def _base_source_weights(pool: list[SourceProfile], *, weighted_random: bool) -> list[float]:
    if not weighted_random:
        return [1.0 for _ in pool]
    default_weights = [0.16, 0.16, 0.16, 0.16, 0.18, 0.18]
    return [default_weights[index] if index < len(default_weights) else 1.0 for index in range(len(pool))]


def _pick_source_index(
    pool: list[SourceProfile],
    base_weights: list[float],
    rng: random.Random,
    *,
    target_focus: str,
    current_index: int | None,
    keep_current_bias: float,
) -> int:
    weights: list[float] = []
    for index, source in enumerate(pool):
        weight = base_weights[index]
        if target_focus and source.target_service == target_focus:
            weight *= 1.8
        if current_index is not None and index == current_index:
            weight *= keep_current_bias
        weights.append(weight)
    return int(rng.choices(range(len(pool)), weights=weights, k=1)[0])


def _build_phase_sequence(class_name: str, episode_span: int, rng: random.Random) -> tuple[str, ...]:
    span = max(1, episode_span)
    template = list(rng.choice(PHASE_TEMPLATES_BY_CLASS[class_name]))
    if len(template) > span:
        template = template[:span]
    lengths = [1 for _ in template]
    for _ in range(max(0, span - len(template))):
        lengths[rng.randrange(len(lengths))] += 1
    sequence: list[str] = []
    for phase, length in zip(template, lengths):
        sequence.extend([phase] * length)
    return tuple(sequence[:span])


def _build_source_sequence(
    pool: list[SourceProfile],
    episode_span: int,
    rng: random.Random,
    *,
    weighted_random: bool,
    target_focus: str,
    source_strategy: str,
) -> tuple[SourceProfile, ...]:
    base_weights = _base_source_weights(pool, weighted_random=weighted_random)
    shuffled = list(range(len(pool)))
    rng.shuffle(shuffled)
    current_index: int | None = None
    streak_remaining = 0
    sequence: list[SourceProfile] = []
    for step in range(max(1, episode_span)):
        if source_strategy == "round_robin":
            if target_focus and rng.random() < 0.45:
                focused = [index for index in shuffled if pool[index].target_service == target_focus]
                source_index = focused[step % len(focused)] if focused else shuffled[step % len(shuffled)]
            else:
                source_index = shuffled[step % len(shuffled)]
        elif source_strategy == "handoff":
            if current_index is None or streak_remaining <= 0:
                current_index = _pick_source_index(
                    pool,
                    base_weights,
                    rng,
                    target_focus=target_focus,
                    current_index=current_index if step > 0 and rng.random() < 0.25 else None,
                    keep_current_bias=0.75,
                )
                streak_remaining = rng.randint(1, 3)
            source_index = current_index
            streak_remaining -= 1
        elif source_strategy == "surge":
            if current_index is None or rng.random() < 0.24:
                current_index = _pick_source_index(
                    pool,
                    base_weights,
                    rng,
                    target_focus=target_focus,
                    current_index=current_index,
                    keep_current_bias=2.6,
                )
            source_index = current_index
        else:
            current_index = _pick_source_index(
                pool,
                base_weights,
                rng,
                target_focus=target_focus,
                current_index=current_index,
                keep_current_bias=1.4,
            )
            source_index = current_index
        sequence.append(pool[source_index])
    return tuple(sequence)


def _preferred_variant_names(
    class_name: str,
    precondition_profile: str,
    phase: str,
    options: list[VariantSpec],
    rng: random.Random,
) -> tuple[str, ...]:
    preferred = list(PROFILE_VARIANT_PREFERENCES.get(class_name, {}).get(precondition_profile, {}).get(phase, ()))
    remaining = [spec.name for spec in options if spec.name not in preferred]
    if remaining:
        rng.shuffle(remaining)
        extra_max = min(len(remaining), max(1, math.ceil(len(options) / 2)))
        preferred.extend(remaining[: rng.randint(1, extra_max)])
    if not preferred:
        preferred.append(rng.choice(options).name)
    return tuple(dict.fromkeys(preferred))


def _build_episode_plan(
    *,
    class_name: str,
    episode_span: int,
    source_pool: list[SourceProfile],
    weighted_sources: bool,
    variants: dict[str, list[VariantSpec]],
    rng: random.Random,
) -> EpisodePlan:
    target_focus = rng.choice(sorted({source.target_service for source in source_pool}))
    precondition_profile = rng.choice(PRECONDITION_PROFILES_BY_CLASS[class_name])
    source_strategy = rng.choice(SOURCE_STRATEGIES_BY_CLASS[class_name])
    gap_profile = rng.choice(GAP_PROFILES_BY_CLASS[class_name])
    scale_low, scale_high = VALUE_SCALE_RANGE_BY_CLASS[class_name]
    phase_sequence = _build_phase_sequence(class_name, episode_span, rng)
    source_sequence = _build_source_sequence(
        source_pool,
        episode_span,
        rng,
        weighted_random=weighted_sources,
        target_focus=target_focus,
        source_strategy=source_strategy,
    )
    preferred_variants = {
        phase: _preferred_variant_names(class_name, precondition_profile, phase, phase_options, rng)
        for phase, phase_options in variants.items()
    }
    burst_population = list(range(1, max(1, episode_span)))
    burst_count = 0 if not burst_population else rng.randint(1, min(len(burst_population), max(1, episode_span // 5)))
    burst_steps = frozenset(rng.sample(burst_population, k=burst_count)) if burst_count else frozenset()
    return EpisodePlan(
        profile_name=f"{precondition_profile}:{source_strategy}:{gap_profile}:{target_focus}",
        precondition_profile=precondition_profile,
        source_strategy=source_strategy,
        gap_profile=gap_profile,
        target_focus=target_focus,
        phase_sequence=phase_sequence,
        source_sequence=source_sequence,
        preferred_variants=preferred_variants,
        value_scale=round(rng.uniform(scale_low, scale_high), 3),
        burst_steps=burst_steps,
    )


def _peer_target_service(target_service: str) -> str:
    return "fprime_b" if target_service == "fprime_a" else "fprime_a"


def _seed_created_dirs(
    state: EpisodeState,
    *,
    root: str,
    prefix: str,
    episode_id: int,
    target_services: list[str],
) -> None:
    for target_service in target_services:
        for index in range(2):
            _remember_dir(
                state,
                target_service,
                f"{root}/{prefix}_ep_{episode_id:04d}_{target_service}_{index:02d}",
            )


def _new_episode_state(class_name: str, episode_id: int, plan: EpisodePlan) -> EpisodeState:
    state = EpisodeState(
        value_scale=plan.value_scale,
        target_focus=plan.target_focus,
        precondition_profile=plan.precondition_profile,
        prefer_missing_reference=plan.precondition_profile in {"stale_reference", "retry_backlog", "inventory_recheck"},
        prefer_remove_stale_suffix=plan.precondition_profile in {"cleanup_drift", "staged_drop"},
        prefer_cleanup_retry=plan.precondition_profile in {"cleanup_followup", "handover_backlog"},
    )
    peer_target = _peer_target_service(plan.target_focus)
    if class_name == "benign":
        for target_service in ("fprime_a", "fprime_b"):
            _remember_fallback_remove_dir(state, target_service, f"{BENIGN_TEMP_ROOT}/remove_ready_dir")
    elif class_name == "cyber":
        if plan.precondition_profile == "staged_drop":
            _seed_created_dirs(
                state,
                root=CYBER_TEMP_ROOT,
                prefix="staged",
                episode_id=episode_id,
                target_services=[plan.target_focus],
            )
        elif plan.precondition_profile == "dual_target_probe":
            _seed_created_dirs(
                state,
                root=CYBER_TEMP_ROOT,
                prefix="dual",
                episode_id=episode_id,
                target_services=[plan.target_focus, peer_target],
            )
    else:
        if plan.precondition_profile == "cleanup_drift":
            _seed_created_dirs(
                state,
                root=FAULT_TEMP_ROOT,
                prefix="drift",
                episode_id=episode_id,
                target_services=[plan.target_focus, peer_target],
            )
        elif plan.precondition_profile == "retry_backlog":
            _seed_created_dirs(
                state,
                root=FAULT_TEMP_ROOT,
                prefix="retry",
                episode_id=episode_id,
                target_services=[plan.target_focus],
            )
    return state


def _pick_variant(
    variants: dict[str, list[VariantSpec]],
    phase: str,
    rng: random.Random,
    plan: EpisodePlan,
    *,
    class_name: str,
    state: EpisodeState,
    target_service: str,
) -> VariantBuilder:
    options = variants[phase]
    if class_name == "benign" and phase == "standby" and not _benign_remove_dir_available(state, target_service):
        options = [spec for spec in options if spec.name != "remove_dir"]
    preferred = set(plan.preferred_variants.get(phase, ()))
    preferred_options = [spec for spec in options if spec.name in preferred]
    candidate_pool = preferred_options if preferred_options and rng.random() < 0.72 else options
    return rng.choice(candidate_pool).builder


def _gap_seconds(
    schedule_kind: str,
    phase: str,
    family: str,
    rng: random.Random,
    plan: EpisodePlan,
    step: int,
    source: SourceProfile,
    previous_source: SourceProfile | None,
) -> int:
    del family
    if schedule_kind == "benign":
        gap = rng.randint(18, 54) if phase == "downlink" else rng.randint(32, 90)
    elif schedule_kind == "cyber":
        gap = rng.randint(6, 22) if phase == "downlink" else rng.randint(8, 28)
    else:
        gap = rng.randint(14, 40) if phase == "downlink" else rng.randint(18, 55)

    if plan.gap_profile == "surge":
        gap *= 0.55 if phase == "downlink" else 0.72
    elif plan.gap_profile == "handoff":
        gap *= 1.25 if previous_source is not None and previous_source.source_service != source.source_service else 0.82
    elif plan.gap_profile == "staggered":
        gap *= 1.20 if step % 4 == 0 else 0.78
    elif plan.gap_profile == "drift":
        gap *= 1.0 + 0.35 * (step / max(1, len(plan.phase_sequence) - 1))

    if step in plan.burst_steps:
        gap *= 0.45 if schedule_kind == "cyber" else 0.62
    return max(3, int(round(gap)))


def _build_rows(
    *,
    target_rows: int,
    seed: int,
    label: int,
    class_name: str,
    episode_offset: int,
    episode_span: int,
    attack_family_override: str | None,
    start_seconds: int,
    source_pool: list[SourceProfile],
    weighted_sources: bool,
    variants: dict[str, list[VariantSpec]],
) -> list[dict[str, Any]]:
    rng = random.Random(seed)
    current_seconds = start_seconds
    plans: dict[int, EpisodePlan] = {}
    states: dict[int, EpisodeState] = {}
    rows: list[dict[str, Any]] = []
    safe_episode_span = max(1, episode_span)
    default_intent_context = INTENT_CONTEXT_BENIGN_CLEAN
    if class_name == "cyber":
        default_intent_context = INTENT_CONTEXT_MALICIOUS
    elif class_name == "fault":
        default_intent_context = INTENT_CONTEXT_FAULT

    for index in range(target_rows):
        episode_id = episode_offset + (index // safe_episode_span)
        episode_step = index % safe_episode_span
        if episode_id not in plans:
            plan = _build_episode_plan(
                class_name=class_name,
                episode_span=safe_episode_span,
                source_pool=source_pool,
                weighted_sources=weighted_sources,
                variants=variants,
                rng=rng,
            )
            plans[episode_id] = plan
            states[episode_id] = _new_episode_state(class_name, episode_id, plan)
        plan = plans[episode_id]
        phase = plan.phase_sequence[min(episode_step, len(plan.phase_sequence) - 1)]
        state = states[episode_id]
        source = plan.source_sequence[min(episode_step, len(plan.source_sequence) - 1)]
        previous_source = None if episode_step <= 0 else plan.source_sequence[episode_step - 1]
        variant = _pick_variant(
            variants,
            phase,
            rng,
            plan,
            class_name=class_name,
            state=state,
            target_service=source.target_service,
        )
        scheduled = variant(rng, state, source.target_service, episode_id, episode_step)
        attack_family = attack_family_override if attack_family_override not in (None, "", "none") else scheduled.attack_family
        intent_context = str(scheduled.intent_context or default_intent_context)
        rows.append(
            {
                "time_of_day": fmt_hms(current_seconds),
                "source_service": source.source_service,
                "target_service": source.target_service,
                "target_tts_port": source.target_tts_port,
                "command": scheduled.command,
                "arguments": list(scheduled.arguments),
                "meta": {
                    "class_label": label,
                    "class_name": class_name,
                    "attack_family": attack_family,
                    "intent_context": intent_context,
                    "actor_role": source.actor_role,
                    "actor_trust": source.actor_trust,
                    "episode_id": episode_id,
                    "phase": phase,
                    "episode_profile": plan.profile_name,
                    "precondition_profile": plan.precondition_profile,
                    "source_strategy": plan.source_strategy,
                    "gap_profile": plan.gap_profile,
                    "target_focus": plan.target_focus,
                },
            }
        )
        current_seconds += _gap_seconds(class_name, phase, attack_family, rng, plan, episode_step, source, previous_source)
    return rows


def build_benign_rows(
    *,
    target_rows: int,
    seed: int,
    label: int = 0,
    class_name: str = "benign",
    attack_family: str | None = None,
    episode_offset: int = 0,
    episode_span: int = 24,
) -> list[dict[str, Any]]:
    return _build_rows(
        target_rows=target_rows,
        seed=seed,
        label=label,
        class_name=class_name,
        episode_offset=episode_offset,
        episode_span=episode_span,
        attack_family_override=attack_family,
        start_seconds=20,
        source_pool=GOOD_SOURCES,
        weighted_sources=False,
        variants=BENIGN_VARIANTS,
    )


def build_cyber_rows(
    *,
    target_rows: int,
    seed: int,
    label: int = 1,
    class_name: str = "cyber",
    episode_offset: int = 0,
    episode_span: int = 24,
) -> list[dict[str, Any]]:
    return _build_rows(
        target_rows=target_rows,
        seed=seed + 5000,
        label=label,
        class_name=class_name,
        episode_offset=episode_offset,
        episode_span=episode_span,
        attack_family_override=None,
        start_seconds=35,
        source_pool=BAD_SOURCES,
        weighted_sources=True,
        variants=CYBER_VARIANTS,
    )


def build_fault_rows(
    *,
    target_rows: int,
    seed: int,
    label: int = 2,
    class_name: str = "fault",
    episode_offset: int = 0,
    episode_span: int = 24,
) -> list[dict[str, Any]]:
    return _build_rows(
        target_rows=target_rows,
        seed=seed + 9000,
        label=label,
        class_name=class_name,
        episode_offset=episode_offset,
        episode_span=episode_span,
        attack_family_override=None,
        start_seconds=50,
        source_pool=FAULT_SOURCES,
        weighted_sources=False,
        variants=FAULT_VARIANTS,
    )


def write_schedule_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="") as handle:
        writer = csv.writer(handle)
        writer.writerow(SCHEDULE_COLUMNS)
        for row in rows:
            writer.writerow(
                [
                    row["time_of_day"],
                    row["source_service"],
                    row["target_service"],
                    row["target_tts_port"],
                    row["command"],
                    json.dumps(row["arguments"], separators=(",", ":")),
                    json.dumps(row["meta"], separators=(",", ":")),
                ]
            )


_SIGNATURE_NUMBER_RE = re.compile(r"\d+")


def _parse_json_payload(value: Any) -> Any:
    if isinstance(value, str) and value.strip():
        try:
            return json.loads(value)
        except (TypeError, ValueError, json.JSONDecodeError):
            return None
    return None


def _row_meta(row: dict[str, Any]) -> dict[str, Any]:
    meta = row.get("meta")
    if isinstance(meta, dict):
        return meta
    parsed = _parse_json_payload(row.get("meta_json"))
    return parsed if isinstance(parsed, dict) else {}


def _row_group_id(row: dict[str, Any], meta: dict[str, Any], group_key: str) -> int | None:
    value = row.get(group_key, meta.get(group_key))
    try:
        return int(value)
    except (TypeError, ValueError):
        return None


def _parse_hms(value: str) -> int | None:
    parts = value.strip().split(":")
    if len(parts) not in (2, 3):
        return None
    try:
        hour = int(parts[0])
        minute = int(parts[1])
        second = int(parts[2]) if len(parts) == 3 else 0
    except ValueError:
        return None
    if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
        return None
    return hour * 3600 + minute * 60 + second


def _row_time_seconds(row: dict[str, Any]) -> float | None:
    for field_name in ("request_ts_ms", "final_ts_ms", "real_ms", "send_start_ms", "send_end_ms"):
        value = row.get(field_name)
        if value not in (None, ""):
            try:
                return float(value) / 1000.0
            except (TypeError, ValueError):
                continue
    for field_name in ("virtual_seconds",):
        value = row.get(field_name)
        if value not in (None, ""):
            try:
                return float(value)
            except (TypeError, ValueError):
                continue
    if "time_of_day" in row:
        parsed = _parse_hms(str(row.get("time_of_day", "")))
        if parsed is not None:
            return float(parsed)
    return None


def _row_arguments(row: dict[str, Any]) -> list[str]:
    arguments = row.get("arguments")
    if isinstance(arguments, list):
        return [str(item) for item in arguments]
    parsed = _parse_json_payload(row.get("arguments_json"))
    if isinstance(parsed, list):
        return [str(item) for item in parsed]
    args = row.get("args")
    if isinstance(args, dict):
        return [f"{key}={args[key]}" for key in sorted(args)]
    return []


def _normalize_signature_argument(argument: str) -> str:
    text = str(argument).strip().lower()
    if not text:
        return ""
    try:
        value = float(text)
    except ValueError:
        normalized = _SIGNATURE_NUMBER_RE.sub("<n>", text)
        normalized = normalized.replace("fprime_a", "node_a").replace("fprime_b", "node_b")
        return normalized
    if not math.isfinite(value):
        return "non_finite"
    if value == 0.0:
        sign = "zero"
    elif value < 0.0:
        sign = "neg"
    else:
        sign = "pos"
    magnitude = abs(value)
    if magnitude < 1.0:
        bucket = "tiny"
    elif magnitude < 10.0:
        bucket = "small"
    elif magnitude < 100.0:
        bucket = "medium"
    elif magnitude < 1000.0:
        bucket = "large"
    else:
        bucket = "xlarge"
    return f"{sign}_{bucket}"


def _gap_bucket(previous_time_s: float | None, current_time_s: float | None) -> str:
    if current_time_s is None:
        return "unknown"
    if previous_time_s is None:
        return "start"
    delta = max(0.0, current_time_s - previous_time_s)
    if delta < 10.0:
        return "burst"
    if delta < 30.0:
        return "short"
    if delta < 75.0:
        return "medium"
    return "long"


def _class_name_for_row(row: dict[str, Any]) -> str:
    explicit = str(row.get("class_name", row.get("label_name", row.get("episode_kind", "")))).strip()
    if explicit:
        return explicit
    meta = _row_meta(row)
    name = str(meta.get("class_name", "")).strip()
    if name:
        return name
    label_value = meta.get("class_label")
    if label_value in (0, 1, 2, "0", "1", "2"):
        return CLASS_NAMES[int(label_value)]
    label_value = row.get("label", row.get("episode_label"))
    if label_value in (0, 1, 2, "0", "1", "2"):
        return CLASS_NAMES[int(label_value)]
    return "unknown"


def _split_overlap_summary(episodes: list[dict[str, Any]]) -> dict[str, Any]:
    if not any(str(episode.get("split", "")).strip() for episode in episodes):
        return {}
    train_like = [
        episode
        for episode in episodes
        if str(episode.get("split", "")).strip() in {"base", "calibration", "calib"}
    ]
    test = [episode for episode in episodes if str(episode.get("split", "")).strip() == "test"]
    train_like_hashes = {str(episode["signature_hash"]) for episode in train_like}
    test_hashes = {str(episode["signature_hash"]) for episode in test}
    shared_hashes = sorted(train_like_hashes & test_hashes)
    return {
        "train_like_episode_count": len(train_like),
        "test_episode_count": len(test),
        "train_like_to_test_shared_signature_count": len(shared_hashes),
        "train_like_to_test_shared_episode_count": sum(1 for episode in test if episode["signature_hash"] in shared_hashes),
        "shared_signature_hashes": shared_hashes,
    }


def build_episode_signature_report(
    rows: list[dict[str, Any]],
    *,
    group_key: str = "episode_id",
    split_assignments: dict[int, str] | None = None,
) -> dict[str, Any]:
    grouped: dict[int, list[tuple[int, dict[str, Any], dict[str, Any]]]] = defaultdict(list)
    for index, row in enumerate(rows):
        meta = _row_meta(row)
        group_id = _row_group_id(row, meta, group_key)
        if group_id is None:
            continue
        grouped[group_id].append((index, row, meta))

    episodes: list[dict[str, Any]] = []
    signature_counts = Counter()
    per_class_signature_counts: dict[str, Counter[str]] = defaultdict(Counter)
    for group_id in sorted(grouped):
        ordered_rows = sorted(
            grouped[group_id],
            key=lambda item: (
                float(_row_time_seconds(item[1])) if _row_time_seconds(item[1]) is not None else float(item[0]),
                item[0],
            ),
        )
        first_row = ordered_rows[0][1]
        first_meta = ordered_rows[0][2]
        previous_time_s: float | None = None
        tokens: list[dict[str, Any]] = []
        for _, row, meta in ordered_rows:
            current_time_s = _row_time_seconds(row)
            target_service = str(row.get("target_service", "")).strip()
            if not target_service:
                target_stream_id = str(row.get("target_stream_id", "")).strip()
                target_service = target_stream_id.split(":", 1)[0] if ":" in target_stream_id else target_stream_id
            tokens.append(
                {
                    "phase": str(row.get("phase", meta.get("phase", ""))).strip() or "unknown",
                    "source_service": str(row.get("source_service", row.get("actor", ""))).strip(),
                    "target_service": target_service,
                    "actor_role": str(row.get("actor_role", meta.get("actor_role", ""))).strip(),
                    "command": str(row.get("command", "")).strip(),
                    "attack_family": str(row.get("attack_family", meta.get("attack_family", "none"))).strip(),
                    "intent_context": str(row.get("intent_context", meta.get("intent_context", ""))).strip(),
                    "gap_bucket": _gap_bucket(previous_time_s, current_time_s),
                    "arguments": [_normalize_signature_argument(argument) for argument in _row_arguments(row)],
                }
            )
            previous_time_s = current_time_s if current_time_s is not None else previous_time_s
        signature_payload = {
            "precondition_profile": str(first_meta.get("precondition_profile", "")).strip(),
            "target_focus": str(first_meta.get("target_focus", "")).strip(),
            "rows": tokens,
        }
        signature_hash = hashlib.sha1(
            json.dumps(signature_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
        ).hexdigest()[:12]
        class_name = _class_name_for_row(first_row)
        episode_summary = {
            "episode_id": int(group_id),
            "class_name": class_name,
            "rows": len(ordered_rows),
            "signature_hash": signature_hash,
            "episode_profile": str(first_meta.get("episode_profile", "")).strip(),
            "precondition_profile": str(first_meta.get("precondition_profile", "")).strip(),
            "source_strategy": str(first_meta.get("source_strategy", "")).strip(),
            "gap_profile": str(first_meta.get("gap_profile", "")).strip(),
            "target_focus": str(first_meta.get("target_focus", "")).strip(),
            "phase_counts": dict(Counter(token["phase"] for token in tokens)),
            "command_counts": dict(Counter(token["command"] for token in tokens)),
            "attack_families": dict(Counter(token["attack_family"] for token in tokens if token["attack_family"])),
        }
        if split_assignments is not None:
            episode_summary["split"] = str(split_assignments.get(int(group_id), "")).strip()
        episodes.append(episode_summary)
        signature_counts[signature_hash] += 1
        per_class_signature_counts[class_name][signature_hash] += 1

    per_class: dict[str, dict[str, Any]] = {}
    for class_name in CLASS_NAMES:
        episode_count = sum(1 for episode in episodes if episode["class_name"] == class_name)
        class_signatures = per_class_signature_counts.get(class_name, Counter())
        unique_count = len(class_signatures)
        per_class[class_name] = {
            "episodes": episode_count,
            "unique_signatures": unique_count,
            "unique_ratio": 0.0 if episode_count == 0 else round(unique_count / episode_count, 4),
            "max_duplicate_group_count": max(class_signatures.values(), default=0),
        }

    total_episodes = len(episodes)
    summary = {
        "episodes": total_episodes,
        "unique_signatures": len(signature_counts),
        "unique_ratio": 0.0 if total_episodes == 0 else round(len(signature_counts) / total_episodes, 4),
        "max_duplicate_group_count": max(signature_counts.values(), default=0),
        "duplicate_signatures": sum(1 for count in signature_counts.values() if count > 1),
    }
    return {
        "group_key": group_key,
        "signature_version": "episode_shape_v1",
        "summary": summary,
        "per_class": per_class,
        "split_overlap": _split_overlap_summary(episodes),
        "episodes": episodes,
    }


def assert_diverse_episode_signatures(report: dict[str, Any]) -> None:
    violations: list[str] = []
    per_class = report.get("per_class", {})
    for class_name in ("cyber", "fault"):
        class_summary = per_class.get(class_name, {})
        episode_count = int(class_summary.get("episodes", 0))
        unique_count = int(class_summary.get("unique_signatures", 0))
        unique_ratio = float(class_summary.get("unique_ratio", 0.0))
        max_duplicate = int(class_summary.get("max_duplicate_group_count", 0))
        if episode_count > 1 and unique_count <= 1:
            violations.append(f"{class_name} collapsed to one signature across {episode_count} episodes")
        elif episode_count >= 4 and unique_ratio < 0.75:
            violations.append(f"{class_name} unique_ratio={unique_ratio:.2f} across {episode_count} episodes")
        if episode_count >= 4 and max_duplicate > max(1, math.ceil(episode_count * 0.25)):
            violations.append(f"{class_name} max_duplicate_group_count={max_duplicate}")
    if violations:
        raise SystemExit(
            "Episode signature diversity check failed. "
            + "; ".join(violations)
        )


def assert_split_episode_separation(report: dict[str, Any]) -> None:
    overlap = report.get("split_overlap", {})
    shared_signature_count = int(overlap.get("train_like_to_test_shared_signature_count", 0))
    shared_episode_count = int(overlap.get("train_like_to_test_shared_episode_count", 0))
    if shared_signature_count <= 0 and shared_episode_count <= 0:
        return
    raise SystemExit(
        "Held-out episode signatures overlap training/calibration structure. "
        f"split_overlap={json.dumps(overlap, sort_keys=True, separators=(',', ':'))}"
    )


def has_structural_signature_signal(report: dict[str, Any]) -> bool:
    episodes = report.get("episodes", [])
    if not isinstance(episodes, list) or not episodes:
        return False
    for episode in episodes:
        if str(episode.get("episode_profile", "")).strip() or str(episode.get("precondition_profile", "")).strip():
            return True
        if len(dict(episode.get("command_counts", {}))) > 1:
            return True
        if len(dict(episode.get("phase_counts", {}))) > 1:
            return True
    return False


def build_command_family_overlap_report(rows: list[dict[str, Any]]) -> dict[str, Any]:
    counts: dict[str, dict[str, int]] = {}
    row_total = 0
    for row in rows:
        command = str(row.get("command", "")).strip()
        if not command:
            continue
        class_name = _class_name_for_row(row)
        class_counts = counts.setdefault(command, {name: 0 for name in CLASS_NAMES})
        if class_name in class_counts:
            class_counts[class_name] += 1
        row_total += 1

    shared_rows = 0
    shared_any = 0
    shared_all = 0
    exclusive = 0
    command_rows: list[dict[str, Any]] = []
    for command in sorted(counts):
        class_counts = counts[command]
        shared_classes = [name for name in CLASS_NAMES if int(class_counts.get(name, 0)) > 0]
        active_count = len(shared_classes)
        total = sum(class_counts.values())
        if active_count >= 2:
            shared_any += 1
            shared_rows += total
        if active_count == len(CLASS_NAMES):
            shared_all += 1
        if active_count == 1:
            exclusive += 1
        dominant = max(class_counts.values()) if class_counts else 0
        command_rows.append(
            {
                "command": command,
                "classes": dict(class_counts),
                "shared_classes": shared_classes,
                "shared_class_count": active_count,
                "exclusive": active_count == 1,
                "dominant_class_share": 0.0 if total <= 0 else round(dominant / total, 4),
                "rows": total,
            }
        )

    total_commands = len(counts)
    return {
        "family_key": "exact_command",
        "summary": {
            "rows": row_total,
            "class_rows": {
                class_name: sum(command_counts[class_name] for command_counts in counts.values())
                for class_name in CLASS_NAMES
            },
            "total_commands": total_commands,
            "commands_shared_by_at_least_two_classes": shared_any,
            "commands_shared_by_all_classes": shared_all,
            "exclusive_commands": exclusive,
            "overlap_ratio": 0.0 if total_commands == 0 else round(shared_any / total_commands, 4),
            "shared_row_fraction": 0.0 if row_total == 0 else round(shared_rows / row_total, 4),
        },
        "commands": command_rows,
    }
