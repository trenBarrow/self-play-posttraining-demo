#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
AUTOPILOT_ROOT="${MAVLINK_AUTOPILOT_ROOT:-/ardupilot}"
RUNTIME_ROOT="${MAVLINK_RUNTIME_ROOT:-/runtime_root}"
GCS_HOST="${MAVLINK_GCS_HOST:-mavlink_gcs}"
GCS_UDP_PORT="${MAVLINK_GCS_UDP_PORT:-14550}"
IDENTITY_TCP_PORT="${MAVLINK_IDENTITY_TCP_PORT:-5760}"
SYSID="${MAVLINK_SYSID:-1}"
SIM_SPEEDUP="${MAVLINK_SIM_SPEEDUP:-1}"
HOME_LOCATION="${MAVLINK_HOME_LOCATION:-37.400337,-122.080035,584,0}"

VEHICLE_BIN="$AUTOPILOT_ROOT/build/sitl/bin/arducopter"
DEFAULT_PARAMS="$AUTOPILOT_ROOT/Tools/autotest/default_params/copter.parm"
STATE_DIR="$RUNTIME_ROOT/vehicle/state"
LOG_DIR="$RUNTIME_ROOT/vehicle/logs"
STDOUT_LOG="$LOG_DIR/vehicle.stdout.log"

mkdir -p "$STATE_DIR" "$LOG_DIR"

if [[ ! -x "$VEHICLE_BIN" ]]; then
  echo "[vehicle] Missing SITL binary at $VEHICLE_BIN" >&2
  exit 1
fi

if [[ ! -f "$DEFAULT_PARAMS" ]]; then
  echo "[vehicle] Missing default params file at $DEFAULT_PARAMS" >&2
  exit 1
fi

cd "$STATE_DIR"

exec > >(tee -a "$STDOUT_LOG") 2>&1

echo "[vehicle] Starting ArduCopter SITL"
echo "[vehicle] autopilot_root=$AUTOPILOT_ROOT"
echo "[vehicle] state_dir=$STATE_DIR"
echo "[vehicle] gcs_endpoint=udpclient:${GCS_HOST}:${GCS_UDP_PORT}"
echo "[vehicle] identity_endpoint=tcp:0.0.0.0:${IDENTITY_TCP_PORT}"

exec "$VEHICLE_BIN" \
  -S \
  --model + \
  --speedup "$SIM_SPEEDUP" \
  --defaults "$DEFAULT_PARAMS" \
  --sim-address=0.0.0.0 \
  --serial0 "udpclient:${GCS_HOST}:${GCS_UDP_PORT}" \
  --serial1 "tcp:0.0.0.0:${IDENTITY_TCP_PORT}" \
  --sysid "$SYSID" \
  --home "$HOME_LOCATION" \
  -I0
