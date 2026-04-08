#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
WORK_DIR="$ROOT_DIR/gds/fprime_project"
PROJECT_DIR="$WORK_DIR/FlightPair"
DEPLOYMENT_DIR="$PROJECT_DIR/DualLink"
APP_BIN="$DEPLOYMENT_DIR/build-artifacts/Linux/bin/DualLink"
DICT_XML="$DEPLOYMENT_DIR/build-artifacts/Linux/dict/DualLinkTopologyAppDictionary.xml"
IMAGE="nasafprime/fprime-arm:latest"
MAX_ATTEMPTS=3

FORCE_REBUILD="false"
while [[ $# -gt 0 ]]; do
  case "$1" in
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

safe_remove_dir() {
  local target="$1"
  if [[ -d "$target" ]]; then
    python3 - <<'PY' "$target"
import shutil
import sys
from pathlib import Path

path = Path(sys.argv[1])
if path.exists():
    shutil.rmtree(path)
PY
  fi
}

mkdir -p "$WORK_DIR" "$ROOT_DIR/gds/fprime_runtime/logs" "$ROOT_DIR/gds/fprime_runtime/schedule"

if [[ -d "$PROJECT_DIR" && ! -d "$PROJECT_DIR/fprime" ]]; then
  echo "[bootstrap] Found partial project without fprime submodule; resetting project dir"
  safe_remove_dir "$PROJECT_DIR"
fi

if [[ ! -d "$PROJECT_DIR" ]]; then
  attempt=1
  until [[ $attempt -gt $MAX_ATTEMPTS ]]; do
    echo "[bootstrap] Creating F' project FlightPair (tag v3.2.0), attempt ${attempt}/${MAX_ATTEMPTS}"
    if docker run --rm \
      -v "$ROOT_DIR:/workspace" \
      --entrypoint /bin/bash \
      "$IMAGE" \
      -lc 'set -euo pipefail; cd /workspace/gds/fprime_project; printf "FlightPair\nv3.2.0\n2\n\n" | fprime-util new --project --path /workspace/gds/fprime_project'; then
      break
    fi

    if [[ $attempt -eq $MAX_ATTEMPTS ]]; then
      echo "[bootstrap] Failed creating project after ${MAX_ATTEMPTS} attempts" >&2
      exit 1
    fi

    echo "[bootstrap] Project creation failed; cleaning partial output before retry"
    safe_remove_dir "$PROJECT_DIR"
    attempt=$((attempt + 1))
    sleep 2
  done
fi

if [[ ! -d "$DEPLOYMENT_DIR" ]]; then
  echo "[bootstrap] Creating deployment DualLink"
  docker run --rm \
    -v "$ROOT_DIR:/workspace" \
    --entrypoint /bin/bash \
    "$IMAGE" \
    -lc 'set -euo pipefail; cd /workspace/gds/fprime_project/FlightPair; printf "DualLink\n./fprime\n" | fprime-util new --deployment'
fi

if [[ "$FORCE_REBUILD" == "true" ]]; then
  echo "[bootstrap] --force-rebuild set; purging build caches"
  docker run --rm \
    -v "$ROOT_DIR:/workspace" \
    --entrypoint /bin/bash \
    "$IMAGE" \
    -lc 'set -euo pipefail; cd /workspace/gds/fprime_project/FlightPair/DualLink; fprime-util purge || true'
fi

if [[ ! -x "$APP_BIN" || ! -f "$DICT_XML" ]]; then
  echo "[bootstrap] Building DualLink deployment (first build can take several minutes)"
  docker run --rm \
    -v "$ROOT_DIR:/workspace" \
    --entrypoint /bin/bash \
    "$IMAGE" \
    -lc 'set -euo pipefail; cd /workspace/gds/fprime_project/FlightPair/DualLink; fprime-util generate; fprime-util build'
else
  echo "[bootstrap] Existing build artifacts found; skipping compile"
fi

if [[ ! -x "$APP_BIN" || ! -f "$DICT_XML" ]]; then
  echo "[bootstrap] Build did not produce expected artifacts" >&2
  exit 1
fi

echo "[bootstrap] Ready"
echo "  app: $APP_BIN"
echo "  dict: $DICT_XML"
