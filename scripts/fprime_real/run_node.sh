#!/usr/bin/env bash
set -euo pipefail

NODE=""
while [[ $# -gt 0 ]]; do
  case "$1" in
    --node)
      NODE="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

if [[ "$NODE" != "a" && "$NODE" != "b" ]]; then
  echo "Usage: run_node.sh --node <a|b>" >&2
  exit 2
fi

ROOT_DIR="/workspace"
DEPLOY_DIR="$ROOT_DIR/gds/fprime_project/FlightPair/DualLink"
ARTIFACT_ROOT="$DEPLOY_DIR/build-artifacts"
APP_BIN="$DEPLOY_DIR/build-artifacts/Linux/bin/DualLink"
DICT_XML="$DEPLOY_DIR/build-artifacts/Linux/dict/DualLinkTopologyAppDictionary.xml"
RUNTIME_ROOT="${FPRIME_RUNTIME_ROOT:-$ROOT_DIR/gds/fprime_runtime}"
LOG_DIR="$RUNTIME_ROOT/node_${NODE}/logs"
OUT_DIR="$RUNTIME_ROOT/node_${NODE}"

mkdir -p "$LOG_DIR" "$OUT_DIR"
mkdir -p \
  /tmp/fprime_benign \
  /tmp/fprime_benign/remove_ready_dir \
  /tmp/fprime_benign/staging \
  /tmp/fprime_redteam \
  /tmp/fprime_redteam/staging \
  /tmp/fprime_fault
rm -rf /tmp/fprime_benign/support_created_dir
printf 'node=%s\n' "$NODE" > /tmp/fprime_benign/NODE.txt

if [[ ! -x "$APP_BIN" ]]; then
  echo "Missing app binary: $APP_BIN" >&2
  echo "Run scripts/fprime_real/bootstrap_project.sh first." >&2
  exit 1
fi

if [[ ! -f "$DICT_XML" ]]; then
  echo "Missing dictionary: $DICT_XML" >&2
  echo "Run scripts/fprime_real/bootstrap_project.sh first." >&2
  exit 1
fi

cat > "$OUT_DIR/runtime_env.txt" <<RUNTIME
node=$NODE
started_utc=$(date -u +%Y-%m-%dT%H:%M:%SZ)
app=$APP_BIN
dictionary=$DICT_XML
runtime_root=$RUNTIME_ROOT
RUNTIME

exec fprime-gds \
  --root "$ARTIFACT_ROOT" \
  --dictionary "$DICT_XML" \
  --app "$APP_BIN" \
  --ip-address 0.0.0.0 \
  --ip-port 50000 \
  --tts-addr 0.0.0.0 \
  --tts-port 50050 \
  --gui none \
  --logs "$LOG_DIR" \
  --log-directly
