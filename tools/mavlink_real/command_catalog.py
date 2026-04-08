#!/usr/bin/env python3
"""Research-local MAVLink command and identity catalog for the real ArduPilot path."""

from __future__ import annotations

import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any

REPO_ROOT = Path(__file__).resolve().parents[2]
if str(REPO_ROOT) not in sys.path:
    sys.path.insert(0, str(REPO_ROOT))

from tools.shared.taxonomy import resolve_command_semantics

TARGET_SERVICE = "mavlink_vehicle"
TARGET_ENDPOINT = "tcp:mavlink_vehicle:5760"
TARGET_COMPONENT_AUTOPILOT = 1

MAV_MODE_FLAG_CUSTOM_MODE_ENABLED = 1
MAV_MISSION_TYPE_MISSION = 0
MAV_PARAM_TYPE_INT32 = 6
MAV_PARAM_TYPE_REAL32 = 9

ARDUCOPTER_MODE_NUMBERS = {
    "STABILIZE": 0,
    "AUTO": 3,
    "GUIDED": 4,
    "LOITER": 5,
    "RTL": 6,
    "LAND": 9,
}


@dataclass(frozen=True)
class SourceIdentity:
    source_service: str
    actor_role: str
    actor_trust: float
    source_system_id: int
    source_component_id: int
    target_service: str = TARGET_SERVICE
    target_endpoint: str = TARGET_ENDPOINT


@dataclass(frozen=True)
class ParameterMutation:
    param_id: str
    values: tuple[float, ...]
    param_type: int = MAV_PARAM_TYPE_REAL32


@dataclass(frozen=True)
class CommandSpec:
    raw_command_name: str
    send_kind: str
    response_name: str
    default_timeout_seconds: float
    summary: str

    @property
    def semantics(self) -> dict[str, Any]:
        return resolve_command_semantics("mavlink", self.raw_command_name, allow_unknown=False)

    @property
    def canonical_command_family(self) -> str:
        return str(self.semantics["canonical_command_family"])

    @property
    def canonical_command_name(self) -> str | None:
        value = self.semantics.get("canonical_command_name")
        return str(value) if value not in (None, "") else None


SOURCE_IDENTITIES = (
    SourceIdentity(
        source_service="ops_primary",
        actor_role="ops_primary",
        actor_trust=0.98,
        source_system_id=21,
        source_component_id=190,
    ),
    SourceIdentity(
        source_service="ops_secondary",
        actor_role="ops_backup",
        actor_trust=0.92,
        source_system_id=22,
        source_component_id=190,
    ),
    SourceIdentity(
        source_service="red_primary",
        actor_role="external",
        actor_trust=0.24,
        source_system_id=31,
        source_component_id=191,
    ),
    SourceIdentity(
        source_service="red_secondary",
        actor_role="external",
        actor_trust=0.33,
        source_system_id=32,
        source_component_id=191,
    ),
)
SOURCE_IDENTITY_BY_SERVICE = {identity.source_service: identity for identity in SOURCE_IDENTITIES}

READ_ONLY_PARAM_IDS = (
    "FRAME_CLASS",
    "FS_THR_ENABLE",
    "FENCE_RADIUS",
    "RC7_OPTION",
    "FLTMODE5",
    "SIM_BARO_RND",
    "MOT_THST_HOVER",
)

PARAMETER_MUTATIONS = (
    ParameterMutation("FENCE_RADIUS", (120.0, 150.0, 180.0)),
    ParameterMutation("SIM_BARO_RND", (0.0, 0.05, 0.10)),
    ParameterMutation("MOT_THST_HOVER", (0.37, 0.39, 0.42)),
)
PARAMETER_MUTATION_BY_ID = {mutation.param_id: mutation for mutation in PARAMETER_MUTATIONS}

COMMAND_SPECS = (
    CommandSpec(
        raw_command_name="REQUEST_AUTOPILOT_CAPABILITIES",
        send_kind="command_long",
        response_name="AUTOPILOT_VERSION",
        default_timeout_seconds=8.0,
        summary="Query autopilot capabilities and version metadata.",
    ),
    CommandSpec(
        raw_command_name="PARAM_REQUEST_READ",
        send_kind="param_request_read",
        response_name="PARAM_VALUE",
        default_timeout_seconds=6.0,
        summary="Query one named ArduPilot parameter.",
    ),
    CommandSpec(
        raw_command_name="PARAM_REQUEST_LIST",
        send_kind="param_request_list",
        response_name="PARAM_VALUE",
        default_timeout_seconds=8.0,
        summary="Start parameter inventory streaming and capture the first reply.",
    ),
    CommandSpec(
        raw_command_name="PARAM_SET",
        send_kind="param_set",
        response_name="PARAM_VALUE",
        default_timeout_seconds=8.0,
        summary="Write a bounded parameter value and wait for the echoed value.",
    ),
    CommandSpec(
        raw_command_name="MISSION_REQUEST_LIST",
        send_kind="mission_request_list",
        response_name="MISSION_COUNT",
        default_timeout_seconds=6.0,
        summary="Query stored mission inventory.",
    ),
    CommandSpec(
        raw_command_name="MISSION_CLEAR_ALL",
        send_kind="mission_clear_all",
        response_name="MISSION_ACK",
        default_timeout_seconds=6.0,
        summary="Clear stored mission items.",
    ),
    CommandSpec(
        raw_command_name="MAV_CMD_DO_SET_MODE",
        send_kind="command_long",
        response_name="COMMAND_ACK",
        default_timeout_seconds=6.0,
        summary="Switch ArduCopter operating mode using custom-mode values.",
    ),
    CommandSpec(
        raw_command_name="MAV_CMD_COMPONENT_ARM_DISARM",
        send_kind="command_long",
        response_name="COMMAND_ACK",
        default_timeout_seconds=6.0,
        summary="Arm or disarm the vehicle.",
    ),
    CommandSpec(
        raw_command_name="MAV_CMD_NAV_RETURN_TO_LAUNCH",
        send_kind="command_long",
        response_name="COMMAND_ACK",
        default_timeout_seconds=6.0,
        summary="Request RTL behavior.",
    ),
    CommandSpec(
        raw_command_name="MAV_CMD_NAV_LAND",
        send_kind="command_long",
        response_name="COMMAND_ACK",
        default_timeout_seconds=6.0,
        summary="Request LAND behavior.",
    ),
)
COMMAND_SPEC_BY_NAME = {spec.raw_command_name: spec for spec in COMMAND_SPECS}

COMMAND_LONG_NUMERIC_IDS = {
    "REQUEST_AUTOPILOT_CAPABILITIES": 520,
    "MAV_CMD_DO_SET_MODE": 176,
    "MAV_CMD_COMPONENT_ARM_DISARM": 400,
    "MAV_CMD_NAV_RETURN_TO_LAUNCH": 20,
    "MAV_CMD_NAV_LAND": 21,
}


def source_identity_for_service(source_service: str) -> SourceIdentity:
    identity = SOURCE_IDENTITY_BY_SERVICE.get(str(source_service).strip())
    if identity is None:
        raise ValueError(f"Unsupported MAVLink source service: {source_service!r}")
    return identity


def command_spec_for_name(command_name: str) -> CommandSpec:
    spec = COMMAND_SPEC_BY_NAME.get(str(command_name).strip())
    if spec is None:
        raise ValueError(f"Unsupported MAVLink command: {command_name!r}")
    return spec


def command_family_for_name(command_name: str) -> str:
    return command_spec_for_name(command_name).canonical_command_family


def numeric_command_id(command_name: str) -> int:
    value = COMMAND_LONG_NUMERIC_IDS.get(str(command_name).strip())
    if value is None:
        raise ValueError(f"Command {command_name!r} does not use COMMAND_LONG in this runner")
    return value


def mode_change_payload(mode_name: str) -> dict[str, Any]:
    normalized = str(mode_name).strip().upper()
    if normalized not in ARDUCOPTER_MODE_NUMBERS:
        raise ValueError(f"Unsupported ArduCopter mode: {mode_name!r}")
    return {
        "mode_name": normalized,
        "base_mode": MAV_MODE_FLAG_CUSTOM_MODE_ENABLED,
        "custom_mode": ARDUCOPTER_MODE_NUMBERS[normalized],
        "custom_submode": 0,
    }


def arm_disarm_payload(*, arm: bool, force: bool = False) -> dict[str, Any]:
    return {
        "arm": bool(arm),
        "force": bool(force),
        "param1": 1.0 if arm else 0.0,
        "param2": 21196.0 if force else 0.0,
    }


def request_autopilot_capabilities_payload() -> dict[str, Any]:
    return {
        "param1": 1.0,
    }


def param_request_read_payload(param_id: str) -> dict[str, Any]:
    return {
        "param_id": str(param_id).strip(),
        "param_index": -1,
    }


def param_request_list_payload() -> dict[str, Any]:
    return {}


def mission_request_list_payload() -> dict[str, Any]:
    return {
        "mission_type": MAV_MISSION_TYPE_MISSION,
    }


def mission_clear_all_payload() -> dict[str, Any]:
    return {
        "mission_type": MAV_MISSION_TYPE_MISSION,
    }


def parameter_mutation_payload(param_id: str, param_value: float, *, param_type: int | None = None) -> dict[str, Any]:
    mutation = PARAMETER_MUTATION_BY_ID.get(str(param_id).strip())
    if mutation is None:
        raise ValueError(f"Unsupported MAVLink parameter mutation id: {param_id!r}")
    return {
        "param_id": mutation.param_id,
        "param_value": float(param_value),
        "param_type": int(param_type if param_type is not None else mutation.param_type),
    }


def supported_command_summary() -> list[dict[str, Any]]:
    summary: list[dict[str, Any]] = []
    for spec in COMMAND_SPECS:
        summary.append(
            {
                "raw_command_name": spec.raw_command_name,
                "send_kind": spec.send_kind,
                "response_name": spec.response_name,
                "canonical_command_name": spec.canonical_command_name,
                "canonical_command_family": spec.canonical_command_family,
                "default_timeout_seconds": spec.default_timeout_seconds,
            }
        )
    return summary
