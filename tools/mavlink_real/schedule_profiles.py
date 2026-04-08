#!/usr/bin/env python3
"""Seeded schedule generation and validation for real MAVLink research runs."""

from __future__ import annotations

import csv
import hashlib
import json
import random
import sys
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Callable

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.mavlink_real.command_catalog import (
    COMMAND_SPEC_BY_NAME,
    PARAMETER_MUTATIONS,
    READ_ONLY_PARAM_IDS,
    SOURCE_IDENTITIES,
    TARGET_ENDPOINT,
    TARGET_SERVICE,
    CommandSpec,
    ParameterMutation,
    arm_disarm_payload,
    command_family_for_name,
    mission_clear_all_payload,
    mission_request_list_payload,
    mode_change_payload,
    parameter_mutation_payload,
    param_request_list_payload,
    param_request_read_payload,
    request_autopilot_capabilities_payload,
    source_identity_for_service,
)

CLASS_NAMES = ("benign", "cyber", "fault")
PHASES = ("startup", "mission", "recovery", "standby")
INTENT_CONTEXT_BENIGN_CLEAN = "benign_clean"
INTENT_CONTEXT_BENIGN_NOISY = "benign_noisy"
INTENT_CONTEXT_MALICIOUS = "malicious"
INTENT_CONTEXT_FAULT = "fault"
SCHEDULE_COLUMNS = [
    "time_of_day",
    "source_service",
    "target_service",
    "target_endpoint",
    "command",
    "command_family",
    "arguments_json",
    "meta_json",
]


@dataclass(frozen=True)
class SourceProfile:
    source_service: str
    target_service: str
    target_endpoint: str
    actor_role: str
    actor_trust: float
    source_system_id: int
    source_component_id: int


@dataclass(frozen=True)
class ScheduleCommand:
    attack_family: str
    command: str
    arguments: dict[str, Any]
    intent_context: str = ""

    @property
    def command_family(self) -> str:
        return command_family_for_name(self.command)


@dataclass
class EpisodeState:
    last_mode_name: str = "LOITER"
    last_param_values: dict[str, float] = field(default_factory=dict)


@dataclass
class EpisodePlan:
    profile_name: str
    source_strategy: str
    gap_profile: str
    phase_sequence: tuple[str, ...]
    source_sequence: tuple[SourceProfile, ...]


@dataclass(frozen=True)
class CommandTemplate:
    template_name: str
    attack_family: str
    command_name: str
    intent_context: str
    payload_builder: Callable[[random.Random, EpisodeState], dict[str, Any]]

    def build(self, rng: random.Random, state: EpisodeState) -> ScheduleCommand:
        return ScheduleCommand(
            attack_family=self.attack_family,
            command=self.command_name,
            arguments=self.payload_builder(rng, state),
            intent_context=self.intent_context,
        )


GOOD_SOURCES = (
    SourceProfile(
        source_service="ops_primary",
        target_service=TARGET_SERVICE,
        target_endpoint=TARGET_ENDPOINT,
        actor_role="ops_primary",
        actor_trust=0.98,
        source_system_id=21,
        source_component_id=190,
    ),
    SourceProfile(
        source_service="ops_secondary",
        target_service=TARGET_SERVICE,
        target_endpoint=TARGET_ENDPOINT,
        actor_role="ops_backup",
        actor_trust=0.92,
        source_system_id=22,
        source_component_id=190,
    ),
)

BAD_SOURCES = (
    SourceProfile(
        source_service="red_primary",
        target_service=TARGET_SERVICE,
        target_endpoint=TARGET_ENDPOINT,
        actor_role="external",
        actor_trust=0.24,
        source_system_id=31,
        source_component_id=191,
    ),
    SourceProfile(
        source_service="red_secondary",
        target_service=TARGET_SERVICE,
        target_endpoint=TARGET_ENDPOINT,
        actor_role="external",
        actor_trust=0.33,
        source_system_id=32,
        source_component_id=191,
    ),
    SourceProfile(
        source_service="ops_primary",
        target_service=TARGET_SERVICE,
        target_endpoint=TARGET_ENDPOINT,
        actor_role="shared_identity",
        actor_trust=0.57,
        source_system_id=21,
        source_component_id=190,
    ),
    SourceProfile(
        source_service="ops_secondary",
        target_service=TARGET_SERVICE,
        target_endpoint=TARGET_ENDPOINT,
        actor_role="shared_identity",
        actor_trust=0.63,
        source_system_id=22,
        source_component_id=190,
    ),
)

FAULT_SOURCES = GOOD_SOURCES


def fmt_hms(seconds: int) -> str:
    seconds %= 86400
    hh = seconds // 3600
    mm = (seconds % 3600) // 60
    ss = seconds % 60
    return f"{hh:02d}:{mm:02d}:{ss:02d}"


def parse_hms(value: str) -> int:
    parts = value.strip().split(":")
    if len(parts) not in (2, 3):
        raise ValueError(f"Invalid time_of_day: {value!r}")
    hour = int(parts[0])
    minute = int(parts[1])
    second = int(parts[2]) if len(parts) == 3 else 0
    if not (0 <= hour <= 23 and 0 <= minute <= 59 and 0 <= second <= 59):
        raise ValueError(f"Invalid time_of_day: {value!r}")
    return hour * 3600 + minute * 60 + second


def phase_for_step(step: int, episode_span: int) -> str:
    if episode_span <= 1:
        return PHASES[0]
    bucket = min(len(PHASES) - 1, (step * len(PHASES)) // max(1, episode_span))
    return PHASES[bucket]


def _choose_read_param(rng: random.Random) -> str:
    return str(rng.choice(READ_ONLY_PARAM_IDS))


def _choose_mutation(rng: random.Random, state: EpisodeState) -> ParameterMutation:
    ranked = list(PARAMETER_MUTATIONS)
    rng.shuffle(ranked)
    for mutation in ranked:
        previous_value = state.last_param_values.get(mutation.param_id)
        if len(set(mutation.values)) == 1:
            continue
        if previous_value is None or any(abs(previous_value - value) > 1e-9 for value in mutation.values):
            return mutation
    return ranked[0]


def _pick_mutation_value(rng: random.Random, mutation: ParameterMutation, state: EpisodeState) -> float:
    previous_value = state.last_param_values.get(mutation.param_id)
    candidates = [value for value in mutation.values if previous_value is None or abs(previous_value - value) > 1e-9]
    chosen = float(rng.choice(candidates or list(mutation.values)))
    state.last_param_values[mutation.param_id] = chosen
    return chosen


def _payload_capability_query(rng: random.Random, state: EpisodeState) -> dict[str, Any]:
    del rng, state
    return request_autopilot_capabilities_payload()


def _payload_param_request_read(rng: random.Random, state: EpisodeState) -> dict[str, Any]:
    del state
    return param_request_read_payload(_choose_read_param(rng))


def _payload_param_request_list(rng: random.Random, state: EpisodeState) -> dict[str, Any]:
    del rng, state
    return param_request_list_payload()


def _payload_mission_request_list(rng: random.Random, state: EpisodeState) -> dict[str, Any]:
    del rng, state
    return mission_request_list_payload()


def _payload_mission_clear_all(rng: random.Random, state: EpisodeState) -> dict[str, Any]:
    del rng, state
    return mission_clear_all_payload()


def _payload_param_set(rng: random.Random, state: EpisodeState) -> dict[str, Any]:
    mutation = _choose_mutation(rng, state)
    value = _pick_mutation_value(rng, mutation, state)
    return parameter_mutation_payload(mutation.param_id, value, param_type=mutation.param_type)


def _payload_mode_from(options: tuple[str, ...]) -> Callable[[random.Random, EpisodeState], dict[str, Any]]:
    def _builder(rng: random.Random, state: EpisodeState) -> dict[str, Any]:
        candidates = [mode for mode in options if mode != state.last_mode_name] or list(options)
        mode_name = str(rng.choice(candidates))
        state.last_mode_name = mode_name
        return mode_change_payload(mode_name)

    return _builder


def _payload_arm(rng: random.Random, state: EpisodeState) -> dict[str, Any]:
    del rng, state
    return arm_disarm_payload(arm=True, force=False)


def _payload_disarm(rng: random.Random, state: EpisodeState) -> dict[str, Any]:
    del rng, state
    return arm_disarm_payload(arm=False, force=False)


BENIGN_PHASE_TEMPLATES: dict[str, tuple[CommandTemplate, ...]] = {
    "startup": (
        CommandTemplate("capability_probe", "routine_status_check", "REQUEST_AUTOPILOT_CAPABILITIES", INTENT_CONTEXT_BENIGN_CLEAN, _payload_capability_query),
        CommandTemplate("param_probe", "routine_status_check", "PARAM_REQUEST_READ", INTENT_CONTEXT_BENIGN_CLEAN, _payload_param_request_read),
    ),
    "mission": (
        CommandTemplate("param_probe", "routine_status_check", "PARAM_REQUEST_READ", INTENT_CONTEXT_BENIGN_CLEAN, _payload_param_request_read),
        CommandTemplate("param_inventory", "mission_visibility", "PARAM_REQUEST_LIST", INTENT_CONTEXT_BENIGN_NOISY, _payload_param_request_list),
        CommandTemplate("mission_inventory", "mission_visibility", "MISSION_REQUEST_LIST", INTENT_CONTEXT_BENIGN_CLEAN, _payload_mission_request_list),
        CommandTemplate("trusted_mode_check", "mode_verification", "MAV_CMD_DO_SET_MODE", INTENT_CONTEXT_BENIGN_CLEAN, _payload_mode_from(("STABILIZE", "GUIDED", "LOITER"))),
    ),
    "recovery": (
        CommandTemplate("trusted_loiter", "mode_verification", "MAV_CMD_DO_SET_MODE", INTENT_CONTEXT_BENIGN_CLEAN, _payload_mode_from(("LOITER", "STABILIZE"))),
        CommandTemplate("mission_inventory", "mission_visibility", "MISSION_REQUEST_LIST", INTENT_CONTEXT_BENIGN_CLEAN, _payload_mission_request_list),
        CommandTemplate("param_probe", "routine_status_check", "PARAM_REQUEST_READ", INTENT_CONTEXT_BENIGN_CLEAN, _payload_param_request_read),
    ),
    "standby": (
        CommandTemplate("capability_probe", "routine_status_check", "REQUEST_AUTOPILOT_CAPABILITIES", INTENT_CONTEXT_BENIGN_CLEAN, _payload_capability_query),
        CommandTemplate("param_probe", "routine_status_check", "PARAM_REQUEST_READ", INTENT_CONTEXT_BENIGN_CLEAN, _payload_param_request_read),
    ),
}

CYBER_PHASE_TEMPLATES: dict[str, tuple[CommandTemplate, ...]] = {
    "startup": (
        CommandTemplate("recon_param_probe", "reconnaissance", "PARAM_REQUEST_READ", INTENT_CONTEXT_MALICIOUS, _payload_param_request_read),
        CommandTemplate("recon_inventory", "reconnaissance", "PARAM_REQUEST_LIST", INTENT_CONTEXT_MALICIOUS, _payload_param_request_list),
    ),
    "mission": (
        CommandTemplate("config_mutation", "persistent_tamper", "PARAM_SET", INTENT_CONTEXT_MALICIOUS, _payload_param_set),
        CommandTemplate("mission_clear", "mission_disruption", "MISSION_CLEAR_ALL", INTENT_CONTEXT_MALICIOUS, _payload_mission_clear_all),
        CommandTemplate("rtl_shift", "mission_disruption", "MAV_CMD_NAV_RETURN_TO_LAUNCH", INTENT_CONTEXT_MALICIOUS, lambda rng, state: {}),
        CommandTemplate("land_shift", "mission_disruption", "MAV_CMD_NAV_LAND", INTENT_CONTEXT_MALICIOUS, lambda rng, state: {}),
        CommandTemplate("mode_churn", "masquerade_mode_churn", "MAV_CMD_DO_SET_MODE", INTENT_CONTEXT_MALICIOUS, _payload_mode_from(("AUTO", "RTL", "LAND"))),
    ),
    "recovery": (
        CommandTemplate("arm_attempt", "control_escalation", "MAV_CMD_COMPONENT_ARM_DISARM", INTENT_CONTEXT_MALICIOUS, _payload_arm),
        CommandTemplate("disarm_attempt", "control_escalation", "MAV_CMD_COMPONENT_ARM_DISARM", INTENT_CONTEXT_MALICIOUS, _payload_disarm),
        CommandTemplate("mission_clear", "mission_disruption", "MISSION_CLEAR_ALL", INTENT_CONTEXT_MALICIOUS, _payload_mission_clear_all),
    ),
    "standby": (
        CommandTemplate("recon_param_probe", "reconnaissance", "PARAM_REQUEST_READ", INTENT_CONTEXT_MALICIOUS, _payload_param_request_read),
        CommandTemplate("mode_churn", "masquerade_mode_churn", "MAV_CMD_DO_SET_MODE", INTENT_CONTEXT_MALICIOUS, _payload_mode_from(("RTL", "LAND"))),
    ),
}

FAULT_PHASE_TEMPLATES: dict[str, tuple[CommandTemplate, ...]] = {
    "startup": (
        CommandTemplate("fault_probe", "operator_recheck", "REQUEST_AUTOPILOT_CAPABILITIES", INTENT_CONTEXT_FAULT, _payload_capability_query),
        CommandTemplate("fault_param_probe", "operator_recheck", "PARAM_REQUEST_READ", INTENT_CONTEXT_FAULT, _payload_param_request_read),
    ),
    "mission": (
        CommandTemplate("fault_tuning", "fault_recovery_tuning", "PARAM_SET", INTENT_CONTEXT_FAULT, _payload_param_set),
        CommandTemplate("fault_mode_shift", "fault_recovery_tuning", "MAV_CMD_DO_SET_MODE", INTENT_CONTEXT_FAULT, _payload_mode_from(("GUIDED", "LOITER", "RTL"))),
        CommandTemplate("fault_mission_inventory", "operator_recheck", "MISSION_REQUEST_LIST", INTENT_CONTEXT_FAULT, _payload_mission_request_list),
    ),
    "recovery": (
        CommandTemplate("fault_rtl", "fault_recovery_tuning", "MAV_CMD_NAV_RETURN_TO_LAUNCH", INTENT_CONTEXT_FAULT, lambda rng, state: {}),
        CommandTemplate("fault_land", "fault_recovery_tuning", "MAV_CMD_NAV_LAND", INTENT_CONTEXT_FAULT, lambda rng, state: {}),
        CommandTemplate("fault_disarm", "fault_recovery_tuning", "MAV_CMD_COMPONENT_ARM_DISARM", INTENT_CONTEXT_FAULT, _payload_disarm),
        CommandTemplate("fault_mission_clear", "fault_cleanup", "MISSION_CLEAR_ALL", INTENT_CONTEXT_FAULT, _payload_mission_clear_all),
    ),
    "standby": (
        CommandTemplate("fault_inventory", "operator_recheck", "PARAM_REQUEST_LIST", INTENT_CONTEXT_FAULT, _payload_param_request_list),
        CommandTemplate("fault_param_probe", "operator_recheck", "PARAM_REQUEST_READ", INTENT_CONTEXT_FAULT, _payload_param_request_read),
    ),
}

TEMPLATES_BY_CLASS = {
    "benign": BENIGN_PHASE_TEMPLATES,
    "cyber": CYBER_PHASE_TEMPLATES,
    "fault": FAULT_PHASE_TEMPLATES,
}


def _source_sequence(source_pool: tuple[SourceProfile, ...], episode_span: int, rng: random.Random, strategy: str) -> tuple[SourceProfile, ...]:
    if strategy == "focus_primary":
        primary = source_pool[0]
        secondary = source_pool[min(1, len(source_pool) - 1)]
        sequence = [primary if index % 5 else secondary for index in range(episode_span)]
        return tuple(sequence)
    if strategy == "paired_handoff":
        return tuple(source_pool[index % len(source_pool)] for index in range(episode_span))
    if strategy == "weighted_red":
        weights = [4 if source.source_service.startswith("red_") else 2 for source in source_pool]
        return tuple(rng.choices(source_pool, weights=weights, k=episode_span))
    return tuple(rng.choice(source_pool) for _ in range(episode_span))


def _build_episode_plan(class_name: str, episode_span: int, source_pool: tuple[SourceProfile, ...], rng: random.Random) -> EpisodePlan:
    if class_name == "benign":
        strategy = str(rng.choice(("focus_primary", "paired_handoff")))
        gap_profile = str(rng.choice(("steady", "handoff")))
    elif class_name == "cyber":
        strategy = str(rng.choice(("weighted_red", "paired_handoff")))
        gap_profile = str(rng.choice(("burst", "staggered")))
    else:
        strategy = str(rng.choice(("focus_primary", "paired_handoff")))
        gap_profile = str(rng.choice(("drift", "steady")))
    return EpisodePlan(
        profile_name=f"{class_name}:{strategy}:{gap_profile}",
        source_strategy=strategy,
        gap_profile=gap_profile,
        phase_sequence=tuple(phase_for_step(step, episode_span) for step in range(episode_span)),
        source_sequence=_source_sequence(source_pool, episode_span, rng, strategy),
    )


def _gap_seconds(class_name: str, phase: str, gap_profile: str, step: int, rng: random.Random) -> int:
    if phase == "startup":
        gap = rng.randint(10, 22)
    elif phase == "mission":
        gap = rng.randint(8, 20)
    elif phase == "recovery":
        gap = rng.randint(12, 28)
    else:
        gap = rng.randint(18, 36)

    if class_name == "cyber":
        gap = max(3, int(round(gap * 0.65)))
    elif class_name == "fault":
        gap = max(4, int(round(gap * 0.85)))

    if gap_profile == "handoff":
        gap = int(round(gap * (1.25 if step % 3 == 0 else 0.85)))
    elif gap_profile == "burst":
        gap = int(round(gap * (0.50 if step % 4 else 1.20)))
    elif gap_profile == "staggered":
        gap = int(round(gap * (1.15 if step % 2 else 0.72)))
    elif gap_profile == "drift":
        gap = int(round(gap * (1.0 + (0.35 * step / max(1, 12)))))
    return max(3, gap)


def _default_intent_context(class_name: str) -> str:
    if class_name == "cyber":
        return INTENT_CONTEXT_MALICIOUS
    if class_name == "fault":
        return INTENT_CONTEXT_FAULT
    return INTENT_CONTEXT_BENIGN_CLEAN


def _templates_for(class_name: str, phase: str) -> tuple[CommandTemplate, ...]:
    templates = TEMPLATES_BY_CLASS[class_name][phase]
    if not templates:
        raise ValueError(f"Missing MAVLink templates for class={class_name!r} phase={phase!r}")
    return templates


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
    source_pool: tuple[SourceProfile, ...],
) -> list[dict[str, Any]]:
    if target_rows <= 0:
        raise ValueError("target_rows must be > 0")
    if episode_span <= 0:
        raise ValueError("episode_span must be > 0")

    rng = random.Random(seed)
    current_seconds = start_seconds
    plans: dict[int, EpisodePlan] = {}
    states: dict[int, EpisodeState] = {}
    rows: list[dict[str, Any]] = []

    for index in range(target_rows):
        episode_id = episode_offset + (index // episode_span)
        episode_step = index % episode_span
        if episode_id not in plans:
            plans[episode_id] = _build_episode_plan(class_name, episode_span, source_pool, rng)
            states[episode_id] = EpisodeState()
        plan = plans[episode_id]
        state = states[episode_id]
        phase = plan.phase_sequence[min(episode_step, len(plan.phase_sequence) - 1)]
        source = plan.source_sequence[min(episode_step, len(plan.source_sequence) - 1)]
        template = rng.choice(_templates_for(class_name, phase))
        scheduled = template.build(rng, state)
        attack_family = attack_family_override if attack_family_override not in (None, "", "none") else scheduled.attack_family
        intent_context = scheduled.intent_context or _default_intent_context(class_name)
        rows.append(
            {
                "time_of_day": fmt_hms(current_seconds),
                "source_service": source.source_service,
                "target_service": source.target_service,
                "target_endpoint": source.target_endpoint,
                "command": scheduled.command,
                "command_family": scheduled.command_family,
                "arguments": dict(scheduled.arguments),
                "meta": {
                    "protocol_family": "mavlink",
                    "class_label": label,
                    "class_name": class_name,
                    "attack_family": attack_family,
                    "intent_context": intent_context,
                    "actor_role": source.actor_role,
                    "actor_trust": source.actor_trust,
                    "episode_id": episode_id,
                    "phase": phase,
                    "episode_profile": plan.profile_name,
                    "source_strategy": plan.source_strategy,
                    "gap_profile": plan.gap_profile,
                    "target_focus": source.target_service,
                    "source_system_id": source.source_system_id,
                    "source_component_id": source.source_component_id,
                    "command_template": template.template_name,
                    "command_semantics": command_name_summary(scheduled.command),
                },
            }
        )
        current_seconds += _gap_seconds(class_name, phase, plan.gap_profile, episode_step, rng)

    validate_schedule_rows(rows)
    return rows


def command_name_summary(command_name: str) -> dict[str, Any]:
    spec = COMMAND_SPEC_BY_NAME[command_name]
    return {
        "raw_command_name": spec.raw_command_name,
        "canonical_command_name": spec.canonical_command_name,
        "canonical_command_family": spec.canonical_command_family,
        "send_kind": spec.send_kind,
        "response_name": spec.response_name,
    }


def build_benign_rows(
    *,
    target_rows: int,
    seed: int,
    label: int = 0,
    class_name: str = "benign",
    attack_family: str | None = None,
    episode_offset: int = 0,
    episode_span: int = 18,
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
    )


def build_cyber_rows(
    *,
    target_rows: int,
    seed: int,
    label: int = 1,
    class_name: str = "cyber",
    episode_offset: int = 0,
    episode_span: int = 18,
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
    )


def build_fault_rows(
    *,
    target_rows: int,
    seed: int,
    label: int = 2,
    class_name: str = "fault",
    episode_offset: int = 0,
    episode_span: int = 18,
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
    )


def validate_schedule_rows(rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise ValueError("No MAVLink schedule rows provided")

    previous_seconds: int | None = None
    for row in rows:
        time_of_day = str(row.get("time_of_day", "")).strip()
        current_seconds = parse_hms(time_of_day)
        if previous_seconds is not None and current_seconds < previous_seconds:
            raise ValueError("MAVLink schedule rows must be sorted by time_of_day")
        previous_seconds = current_seconds

        source_service = str(row.get("source_service", "")).strip()
        target_service = str(row.get("target_service", "")).strip()
        target_endpoint = str(row.get("target_endpoint", "")).strip()
        command_name = str(row.get("command", "")).strip()
        command_family = str(row.get("command_family", "")).strip()
        arguments = row.get("arguments")
        meta = row.get("meta")

        source_identity_for_service(source_service)
        if target_service != TARGET_SERVICE:
            raise ValueError(f"Unsupported MAVLink target_service: {target_service!r}")
        if target_endpoint != TARGET_ENDPOINT:
            raise ValueError(f"Unsupported MAVLink target_endpoint: {target_endpoint!r}")
        if command_name not in COMMAND_SPEC_BY_NAME:
            raise ValueError(f"Unsupported MAVLink schedule command: {command_name!r}")
        if command_family != command_family_for_name(command_name):
            raise ValueError(
                f"Command family mismatch for {command_name!r}: got {command_family!r}, expected {command_family_for_name(command_name)!r}"
            )
        if not isinstance(arguments, dict):
            raise ValueError("MAVLink schedule arguments must be a JSON object")
        if not isinstance(meta, dict):
            raise ValueError("MAVLink schedule meta must be a JSON object")


def write_schedule_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    validate_schedule_rows(rows)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(SCHEDULE_COLUMNS)
        for row in rows:
            writer.writerow(
                [
                    row["time_of_day"],
                    row["source_service"],
                    row["target_service"],
                    row["target_endpoint"],
                    row["command"],
                    row["command_family"],
                    json.dumps(row["arguments"], separators=(",", ":"), sort_keys=True),
                    json.dumps(row["meta"], separators=(",", ":"), sort_keys=True),
                ]
            )


def load_schedule_csv(path: Path) -> list[dict[str, Any]]:
    with path.open("r", newline="", encoding="utf-8") as handle:
        reader = csv.DictReader(handle)
        required = set(SCHEDULE_COLUMNS)
        if reader.fieldnames is None or not required.issubset(set(reader.fieldnames)):
            missing = required - set(reader.fieldnames or [])
            raise ValueError(f"Missing MAVLink schedule columns: {sorted(missing)}")
        rows: list[dict[str, Any]] = []
        for raw in reader:
            arguments = json.loads(raw["arguments_json"])
            meta = json.loads(raw["meta_json"])
            row = {
                "time_of_day": str(raw["time_of_day"]).strip(),
                "source_service": str(raw["source_service"]).strip(),
                "target_service": str(raw["target_service"]).strip(),
                "target_endpoint": str(raw["target_endpoint"]).strip(),
                "command": str(raw["command"]).strip(),
                "command_family": str(raw["command_family"]).strip(),
                "arguments": arguments,
                "meta": meta,
            }
            rows.append(row)
    rows.sort(key=lambda row: (parse_hms(str(row["time_of_day"])), str(row["source_service"]), str(row["command"])))
    validate_schedule_rows(rows)
    return rows


def schedule_digest(rows: list[dict[str, Any]]) -> str:
    payload = json.dumps(rows, separators=(",", ":"), sort_keys=True)
    return hashlib.sha256(payload.encode("utf-8")).hexdigest()


def supported_command_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for spec in COMMAND_SPEC_BY_NAME.values():
        rows.append(
            {
                "command": spec.raw_command_name,
                "command_family": spec.canonical_command_family,
                "send_kind": spec.send_kind,
                "response_name": spec.response_name,
            }
        )
    rows.sort(key=lambda row: str(row["command"]))
    return rows


def supported_source_rows() -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for identity in SOURCE_IDENTITIES:
        rows.append(
            {
                "source_service": identity.source_service,
                "source_system_id": identity.source_system_id,
                "source_component_id": identity.source_component_id,
            }
        )
    rows.sort(key=lambda row: str(row["source_service"]))
    return rows
