#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
COMPOSE_FILE="$ROOT_DIR/orchestration/docker-compose.mavlink-real.yml"
RUNTIME_ROOT="$ROOT_DIR/gds/mavlink_runtime"
AUTOPILOT_HOST_ROOT="$ROOT_DIR/gds/mavlink_project/ardupilot"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runtime-root)
      RUNTIME_ROOT="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

"$PYTHON_BIN" - <<'PY' "$ROOT_DIR" "$RUNTIME_ROOT"
import shutil
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
runtime_root = Path(sys.argv[2])
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from tools.mavlink_real.runtime_layout import (
    ensure_runtime_tree,
    host_gcs_state_dir,
    host_vehicle_state_dir,
)

for path in (host_vehicle_state_dir(runtime_root), host_gcs_state_dir(runtime_root)):
    if path.exists():
        shutil.rmtree(path)

ensure_runtime_tree(runtime_root)
PY

"$ROOT_DIR/scripts/mavlink_real/bootstrap_stack.sh" --runtime-root "$RUNTIME_ROOT"

"$PYTHON_BIN" - <<'PY' "$ROOT_DIR" "$RUNTIME_ROOT" "$COMPOSE_FILE"
import json
import platform
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

repo_root = Path(sys.argv[1])
runtime_root = Path(sys.argv[2])
compose_file = Path(sys.argv[3])
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from tools.mavlink_real.runtime_layout import (
    GCS_SERVICE,
    IDENTITY_SERVICES,
    MAVLINK_NETWORK_NAME,
    VEHICLE_SERVICE,
    host_startup_metadata_path,
)

payload = {
    "started_at_utc": datetime.now(timezone.utc).isoformat(),
    "runtime_root": str(runtime_root),
    "compose_file": str(compose_file),
    "platform": platform.platform(),
    "docker_version": subprocess.check_output(["docker", "--version"], text=True).strip(),
    "compose_version": subprocess.check_output(["docker", "compose", "version"], text=True).strip(),
    "network": MAVLINK_NETWORK_NAME,
    "services": [VEHICLE_SERVICE, GCS_SERVICE, *IDENTITY_SERVICES],
}

path = host_startup_metadata_path(runtime_root)
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

MAVLINK_RUNTIME_HOST_ROOT="$RUNTIME_ROOT" MAVLINK_AUTOPILOT_HOST_ROOT="$AUTOPILOT_HOST_ROOT" \
  docker compose -f "$COMPOSE_FILE" up -d --force-recreate

echo "MAVLink stack is up"
MAVLINK_RUNTIME_HOST_ROOT="$RUNTIME_ROOT" MAVLINK_AUTOPILOT_HOST_ROOT="$AUTOPILOT_HOST_ROOT" \
  docker compose -f "$COMPOSE_FILE" ps

echo "Network ready: mavlink_real_net"
docker network inspect mavlink_real_net --format 'Subnet: {{(index .IPAM.Config 0).Subnet}}'
