#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
SMOKE_DIR="${MAVLINK_SMOKE_DIR:-$ROOT_DIR/artifacts/mavlink_smoke}"
RUNTIME_ROOT="$SMOKE_DIR/mavlink_real"
COMPOSE_FILE="$ROOT_DIR/orchestration/docker-compose.mavlink-real.yml"
AUTOPILOT_DIR="$ROOT_DIR/gds/mavlink_project/ardupilot"
VEHICLE_LOG="$RUNTIME_ROOT/vehicle/logs/vehicle.stdout.log"
GCS_LOG="$RUNTIME_ROOT/gcs/logs/mavproxy.stdout.log"
BOOTSTRAP_METADATA="$RUNTIME_ROOT/metadata/bootstrap_metadata.json"
STARTUP_METADATA="$RUNTIME_ROOT/metadata/startup_metadata.json"

cleanup() {
  "$ROOT_DIR/scripts/mavlink_real/down.sh" >/dev/null 2>&1 || true
}
trap cleanup EXIT

wait_for_path() {
  local path="$1"
  local timeout_seconds="$2"
  local label="$3"
  local started_at
  started_at="$(date +%s)"
  while true; do
    if [[ -s "$path" ]]; then
      return 0
    fi
    if [[ $(( "$(date +%s)" - started_at )) -ge "$timeout_seconds" ]]; then
      echo "Smoke test failed: timed out waiting for $label at $path" >&2
      exit 1
    fi
    sleep 2
  done
}

"$ROOT_DIR/scripts/mavlink_real/up.sh" --runtime-root "$RUNTIME_ROOT"

wait_for_path "$BOOTSTRAP_METADATA" 900 "bootstrap metadata"
wait_for_path "$STARTUP_METADATA" 60 "startup metadata"
wait_for_path "$VEHICLE_LOG" 120 "vehicle log"
wait_for_path "$GCS_LOG" 120 "gcs log"

RUNNING_SERVICES="$(
  MAVLINK_RUNTIME_HOST_ROOT="$RUNTIME_ROOT" MAVLINK_AUTOPILOT_HOST_ROOT="$AUTOPILOT_DIR" \
    docker compose -f "$COMPOSE_FILE" ps --services --status running
)"

for service in mavlink_vehicle mavlink_gcs ops_primary ops_secondary red_primary red_secondary; do
  if ! grep -qx "$service" <<<"$RUNNING_SERVICES"; then
    echo "Smoke test failed: expected running service $service" >&2
    exit 1
  fi
done

if [[ ! -d "$RUNTIME_ROOT/vehicle/state" ]]; then
  echo "Smoke test failed: vehicle state dir missing at $RUNTIME_ROOT/vehicle/state" >&2
  exit 1
fi

if [[ ! -d "$RUNTIME_ROOT/gcs/state" ]]; then
  echo "Smoke test failed: gcs state dir missing at $RUNTIME_ROOT/gcs/state" >&2
  exit 1
fi

echo "Smoke test passed"
echo "  runtime root: $RUNTIME_ROOT"
echo "  vehicle log: $VEHICLE_LOG"
echo "  gcs log: $GCS_LOG"
echo "  bootstrap metadata: $BOOTSTRAP_METADATA"
echo "  startup metadata: $STARTUP_METADATA"
