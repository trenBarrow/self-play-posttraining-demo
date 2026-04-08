#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
WORK_DIR="$ROOT_DIR/gds/mavlink_project"
AUTOPILOT_DIR="$WORK_DIR/ardupilot"
RUNTIME_ROOT="$ROOT_DIR/gds/mavlink_runtime"
COMPOSE_IMAGE="anomaly-detector-mavlink-base:latest"
BASE_IMAGE="ardupilot/ardupilot-dev-base:latest"
DOCKERFILE="$ROOT_DIR/orchestration/mavlink-real/base.Dockerfile"
DOCKER_CONTEXT="$ROOT_DIR/orchestration/mavlink-real"
ARDUPILOT_REMOTE="https://github.com/ArduPilot/ardupilot.git"
ARDUPILOT_TAG="Copter-4.6.3"
MAVPROXY_VERSION="1.8.74"
PYMAVLINK_VERSION="2.4.49"
FORCE_REBUILD="false"

image_input_hash="$(
  "$PYTHON_BIN" - <<'PY' "$DOCKERFILE" "$MAVPROXY_VERSION" "$PYMAVLINK_VERSION"
from pathlib import Path
import hashlib
import sys

dockerfile = Path(sys.argv[1])
payload = dockerfile.read_bytes() + b"\0" + sys.argv[2].encode("utf-8") + b"\0" + sys.argv[3].encode("utf-8")
print(hashlib.sha256(payload).hexdigest())
PY
)"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runtime-root)
      RUNTIME_ROOT="$2"
      shift 2
      ;;
    --force-rebuild)
      FORCE_REBUILD="true"
      shift
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

for required_cmd in git docker "$PYTHON_BIN"; do
  if ! command -v "$required_cmd" >/dev/null 2>&1; then
    echo "[bootstrap] Missing required command: $required_cmd" >&2
    exit 1
  fi
done

"$PYTHON_BIN" - <<'PY' "$ROOT_DIR" "$RUNTIME_ROOT"
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
runtime_root = Path(sys.argv[2])
if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from tools.mavlink_real.runtime_layout import ensure_runtime_tree

ensure_runtime_tree(runtime_root)
PY

mkdir -p "$WORK_DIR"

if [[ ! -d "$AUTOPILOT_DIR/.git" ]]; then
  echo "[bootstrap] Cloning ArduPilot tag $ARDUPILOT_TAG"
  git clone --depth 1 --branch "$ARDUPILOT_TAG" "$ARDUPILOT_REMOTE" "$AUTOPILOT_DIR"
else
  echo "[bootstrap] Refreshing ArduPilot checkout to $ARDUPILOT_TAG"
  git -C "$AUTOPILOT_DIR" fetch --depth 1 origin "refs/tags/$ARDUPILOT_TAG:refs/tags/$ARDUPILOT_TAG"
  git -C "$AUTOPILOT_DIR" checkout -f "$ARDUPILOT_TAG"
fi

git -C "$AUTOPILOT_DIR" submodule update --init --recursive --depth 1

if [[ "$FORCE_REBUILD" == "true" ]]; then
  echo "[bootstrap] --force-rebuild set; cleaning ArduPilot SITL outputs"
  docker run --rm \
    --platform linux/amd64 \
    -v "$AUTOPILOT_DIR:/ardupilot" \
    -w /ardupilot \
    "$BASE_IMAGE" \
    bash -lc './waf clean || true'
fi

ARDUCOPTER_BIN="$AUTOPILOT_DIR/build/sitl/bin/arducopter"
if [[ "$FORCE_REBUILD" == "true" || ! -x "$ARDUCOPTER_BIN" ]]; then
  echo "[bootstrap] Building ArduCopter SITL at $ARDUPILOT_TAG"
  docker run --rm \
    --platform linux/amd64 \
    -v "$AUTOPILOT_DIR:/ardupilot" \
    -w /ardupilot \
    "$BASE_IMAGE" \
    bash -lc './waf configure --board sitl && ./waf copter -j"$(getconf _NPROCESSORS_ONLN || echo 4)"'
else
  echo "[bootstrap] Existing SITL binary found; skipping rebuild"
fi

if [[ ! -x "$ARDUCOPTER_BIN" ]]; then
  echo "[bootstrap] Missing expected SITL binary at $ARDUCOPTER_BIN" >&2
  exit 1
fi

if [[ "$FORCE_REBUILD" == "true" ]]; then
  rebuild_image="true"
elif docker image inspect "$COMPOSE_IMAGE" >/dev/null 2>&1; then
  existing_image_hash="$(
    docker image inspect "$COMPOSE_IMAGE" \
      --format '{{ index .Config.Labels "com.poster.mavlink_bootstrap_hash" }}' \
      2>/dev/null || true
  )"
  if [[ "$existing_image_hash" == "$image_input_hash" ]]; then
    rebuild_image="false"
  else
    rebuild_image="true"
  fi
else
  rebuild_image="true"
fi

if [[ "$rebuild_image" == "true" ]]; then
  echo "[bootstrap] Building local MAVLink runtime image $COMPOSE_IMAGE"
  docker build \
    --platform linux/amd64 \
    --build-arg MAVPROXY_VERSION="$MAVPROXY_VERSION" \
    --build-arg PYMAVLINK_VERSION="$PYMAVLINK_VERSION" \
    --label "com.poster.mavlink_bootstrap_hash=$image_input_hash" \
    -f "$DOCKERFILE" \
    -t "$COMPOSE_IMAGE" \
    "$DOCKER_CONTEXT"
else
  echo "[bootstrap] Existing local image found; skipping docker build"
fi

"$PYTHON_BIN" - <<'PY' "$ROOT_DIR" "$RUNTIME_ROOT" "$AUTOPILOT_DIR" "$ARDUPILOT_TAG" "$ARDUPILOT_REMOTE" "$COMPOSE_IMAGE" "$BASE_IMAGE" "$MAVPROXY_VERSION" "$PYMAVLINK_VERSION" "$FORCE_REBUILD"
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

repo_root = Path(sys.argv[1])
runtime_root = Path(sys.argv[2])
autopilot_dir = Path(sys.argv[3])
autopilot_tag = sys.argv[4]
autopilot_remote = sys.argv[5]
compose_image = sys.argv[6]
base_image = sys.argv[7]
mavproxy_version = sys.argv[8]
pymavlink_version = sys.argv[9]
force_rebuild = sys.argv[10] == "true"

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from tools.mavlink_real.runtime_layout import host_bootstrap_metadata_path

head = subprocess.check_output(["git", "-C", str(autopilot_dir), "rev-parse", "HEAD"], text=True).strip()
image_id = subprocess.check_output(["docker", "image", "inspect", compose_image, "--format", "{{.Id}}"], text=True).strip()
image_hash = subprocess.check_output(
    ["docker", "image", "inspect", compose_image, "--format", "{{ index .Config.Labels \"com.poster.mavlink_bootstrap_hash\" }}"],
    text=True,
).strip()

payload = {
    "completed_at_utc": datetime.now(timezone.utc).isoformat(),
    "autopilot_remote": autopilot_remote,
    "autopilot_tag": autopilot_tag,
    "autopilot_head": head,
    "autopilot_dir": str(autopilot_dir),
    "vehicle_binary": str(autopilot_dir / "build" / "sitl" / "bin" / "arducopter"),
    "base_image": base_image,
    "compose_image": compose_image,
    "compose_image_id": image_id,
    "compose_image_input_hash": image_hash,
    "mavproxy_version": mavproxy_version,
    "pymavlink_version": pymavlink_version,
    "force_rebuild": force_rebuild,
}

path = host_bootstrap_metadata_path(runtime_root)
path.write_text(json.dumps(payload, indent=2, sort_keys=True) + "\n", encoding="utf-8")
PY

echo "[bootstrap] Ready"
echo "  autopilot dir: $AUTOPILOT_DIR"
echo "  vehicle binary: $ARDUCOPTER_BIN"
echo "  local image: $COMPOSE_IMAGE"
