#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
RUNTIME_ROOT="${MAVLINK_RUNTIME_ROOT:-/runtime_root}"
MASTER_ENDPOINT="${MAVLINK_GCS_MASTER_ENDPOINT:-udpin:0.0.0.0:14550}"
AIRCRAFT_NAME="${MAVLINK_AIRCRAFT_NAME:-poster_mavlink_vehicle}"
FORWARD_ENDPOINTS="${MAVLINK_FORWARD_ENDPOINTS:-}"

STATE_DIR="$RUNTIME_ROOT/gcs/state"
LOG_DIR="$RUNTIME_ROOT/gcs/logs"
STDOUT_LOG="$LOG_DIR/mavproxy.stdout.log"

mkdir -p "$STATE_DIR" "$LOG_DIR"

exec > >(tee -a "$STDOUT_LOG") 2>&1

echo "[gcs] Starting MAVProxy"
echo "[gcs] master=$MASTER_ENDPOINT"
echo "[gcs] state_dir=$STATE_DIR"

ARGS=(
  --master="$MASTER_ENDPOINT"
  --non-interactive
  --state-basedir="$STATE_DIR"
  --aircraft="$AIRCRAFT_NAME"
)

if [[ -n "$FORWARD_ENDPOINTS" ]]; then
  IFS=',' read -r -a endpoints <<<"$FORWARD_ENDPOINTS"
  for endpoint in "${endpoints[@]}"; do
    endpoint="$(echo "$endpoint" | xargs)"
    if [[ -n "$endpoint" ]]; then
      ARGS+=(--out="$endpoint")
    fi
  done
fi

exec "$PYTHON_BIN" -m MAVProxy.mavproxy "${ARGS[@]}"
