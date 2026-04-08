#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-}"
SNAPSHOT_ROOT="$ROOT_DIR/artifacts/baseline_fprime_snapshot"
SNAPSHOT_ID="linux_fprime_baseline"
ROWS=240
SEED=7
NOMINAL_RATIO=0.55

usage() {
  cat <<'EOF'
Usage: tools/scripts/capture_baseline_fprime_snapshot.sh [options]

Options:
  --snapshot-root PATH     Snapshot root directory (default: artifacts/baseline_fprime_snapshot)
  --snapshot-id NAME       Snapshot run identifier under generated/ (default: linux_fprime_baseline)
  --rows N                 Training pipeline row count (default: 240)
  --seed N                 Training/generation seed (default: 7)
  --nominal-ratio R        Benign ratio for training generation (default: 0.55)
  --python PATH            Python interpreter to use
  -h, --help               Show this help text
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --snapshot-root)
      SNAPSHOT_ROOT="$2"
      shift 2
      ;;
    --snapshot-id)
      SNAPSHOT_ID="$2"
      shift 2
      ;;
    --rows)
      ROWS="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --nominal-ratio)
      NOMINAL_RATIO="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "Unknown arg: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

if [[ -z "$PYTHON_BIN" ]]; then
  if [[ -x "$ROOT_DIR/.venv/bin/python" ]]; then
    PYTHON_BIN="$ROOT_DIR/.venv/bin/python"
  else
    PYTHON_BIN="python3"
  fi
fi

GENERATED_DIR="$SNAPSHOT_ROOT/generated/$SNAPSHOT_ID"
COMMANDS_DIR="$SNAPSHOT_ROOT/commands"
REPORTS_DIR="$SNAPSHOT_ROOT/reports"
TREES_DIR="$SNAPSHOT_ROOT/trees"
METADATA_DIR="$SNAPSHOT_ROOT/metadata"
INHERITED_BUNDLE_DIR="$SNAPSHOT_ROOT/inherited_deployment_bundle"
INHERITED_EXPORT_DIR="$INHERITED_BUNDLE_DIR/export"
SMOKE_DIR="$GENERATED_DIR/smoke"
TRAIN_DIR="$GENERATED_DIR/train"
PACKAGE_PATH="$TRAIN_DIR/detector_bundle.zip"
INHERITED_BUNDLE_ZIP="$INHERITED_BUNDLE_DIR/detector_bundle.zip"

rm -rf "$GENERATED_DIR" "$COMMANDS_DIR" "$REPORTS_DIR" "$TREES_DIR" "$METADATA_DIR" "$INHERITED_BUNDLE_DIR"
mkdir -p "$COMMANDS_DIR" "$REPORTS_DIR" "$TREES_DIR" "$METADATA_DIR" "$INHERITED_EXPORT_DIR/config" "$GENERATED_DIR"

run_and_capture() {
  local name="$1"
  shift
  local log_path="$COMMANDS_DIR/${name}.log"
  local cmd_path="$COMMANDS_DIR/${name}.command.txt"
  printf 'cwd=%s\n' "$ROOT_DIR" >"$cmd_path"
  printf 'command=' >>"$cmd_path"
  printf '%q ' "$@" >>"$cmd_path"
  printf '\n' >>"$cmd_path"
  set +e
  (
    cd "$ROOT_DIR"
    "$@"
  ) 2>&1 | tee "$log_path"
  local status=${PIPESTATUS[0]}
  set -e
  printf '%s\n' "$status" >"$COMMANDS_DIR/${name}.exit_code.txt"
  # Allow the caller to inspect non-zero command results without aborting the
  # wrapper before it can snapshot logs, reports, and fallback artifacts.
  set +e
  return "$status"
}

record_skipped_command() {
  local name="$1"
  local reason="$2"
  printf 'cwd=%s\n' "$ROOT_DIR" >"$COMMANDS_DIR/${name}.command.txt"
  printf 'command=not-run\n' >>"$COMMANDS_DIR/${name}.command.txt"
  printf 'reason=%s\n' "$reason" >>"$COMMANDS_DIR/${name}.command.txt"
  printf 'skipped: %s\n' "$reason" >"$COMMANDS_DIR/${name}.log"
  printf 'skipped\n' >"$COMMANDS_DIR/${name}.exit_code.txt"
}

copy_report() {
  local src="$1"
  local dest="$2"
  if [[ -f "$src" ]]; then
    mkdir -p "$(dirname "$dest")"
    cp "$src" "$dest"
  fi
}

require_command() {
  if ! command -v "$1" >/dev/null 2>&1; then
    echo "required command missing: $1" >&2
    exit 1
  fi
}

require_command "$PYTHON_BIN"
require_command docker
require_command tcpdump
require_command find
require_command shasum
require_command zip

SNAPSHOT_TS_UTC="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
COMMIT_HASH="$(git -C "$ROOT_DIR" rev-parse HEAD 2>/dev/null || echo untracked)"
BRANCH_NAME="$(git -C "$ROOT_DIR" rev-parse --abbrev-ref HEAD 2>/dev/null || echo detached)"
GIT_TAGS_AT_HEAD="$(git -C "$ROOT_DIR" tag --points-at HEAD 2>/dev/null | paste -sd ',' -)"
GIT_STATUS_SHORT="$(git -C "$ROOT_DIR" status --short 2>/dev/null || true)"
GIT_DIRTY="false"
if [[ -n "$GIT_STATUS_SHORT" ]]; then
  GIT_DIRTY="true"
fi

cat >"$METADATA_DIR/environment.txt" <<EOF
snapshot_id=$SNAPSHOT_ID
snapshot_root=$SNAPSHOT_ROOT
snapshot_timestamp_utc=$SNAPSHOT_TS_UTC
repo_root=$ROOT_DIR
git_commit=$COMMIT_HASH
git_branch=$BRANCH_NAME
git_tags_at_head=$GIT_TAGS_AT_HEAD
git_dirty=$GIT_DIRTY
python_bin=$PYTHON_BIN
python_version=$("$PYTHON_BIN" --version 2>&1)
docker_version=$(docker --version 2>&1)
docker_compose_version=$(docker compose version 2>&1 | tr '\n' ';')
tcpdump_version=$(tcpdump --version 2>&1 | head -n 1)
host_platform=$(uname -a)
EOF

printf '%s\n' "$GIT_STATUS_SHORT" >"$METADATA_DIR/git_status.short.txt"

FAST_TEST_CMD=(
  "$PYTHON_BIN" -m unittest
  tests.test_schedule_hardening
  tests.test_runtime_phase2
  tests.test_stream_serialization
  tests.test_telemetry_catalog
)
run_and_capture fast_regression "${FAST_TEST_CMD[@]}"

SMOKE_CMD=(bash "$ROOT_DIR/scripts/fprime_real/smoke_test.sh")
printf 'cwd=%s\n' "$ROOT_DIR" >"$COMMANDS_DIR/smoke_test.command.txt"
printf 'environment=FPRIME_SMOKE_DIR=%s PYTHON_BIN=%s\n' "$SMOKE_DIR" "$PYTHON_BIN" >>"$COMMANDS_DIR/smoke_test.command.txt"
printf 'command=' >>"$COMMANDS_DIR/smoke_test.command.txt"
printf '%q ' "${SMOKE_CMD[@]}" >>"$COMMANDS_DIR/smoke_test.command.txt"
printf '\n' >>"$COMMANDS_DIR/smoke_test.command.txt"
set +e
(
  cd "$ROOT_DIR"
  FPRIME_SMOKE_DIR="$SMOKE_DIR" PYTHON_BIN="$PYTHON_BIN" "${SMOKE_CMD[@]}"
) 2>&1 | tee "$COMMANDS_DIR/smoke_test.log"
SMOKE_STATUS=${PIPESTATUS[0]}
set -e
printf '%s\n' "$SMOKE_STATUS" >"$COMMANDS_DIR/smoke_test.exit_code.txt"
if [[ "$SMOKE_STATUS" -ne 0 ]]; then
  exit "$SMOKE_STATUS"
fi

TRAIN_CMD=(
  bash "$ROOT_DIR/scripts/fprime_real/train_pipeline.sh"
  --rows "$ROWS"
  --nominal-ratio "$NOMINAL_RATIO"
  --seed "$SEED"
  --output-dir "$TRAIN_DIR"
  --python "$PYTHON_BIN"
)
set +e
run_and_capture train_pipeline "${TRAIN_CMD[@]}"
TRAIN_STATUS=$?
set -e

if [[ ! -f "$TRAIN_DIR/reports/metrics.json" || ! -f "$TRAIN_DIR/reports/summary.txt" ]]; then
  echo "Training pipeline did not produce the required baseline reports." >&2
  exit "${TRAIN_STATUS:-1}"
fi

PACKAGE_CMD=(bash "$ROOT_DIR/tools/scripts/package_detector.sh" "$TRAIN_DIR")
if [[ -f "$TRAIN_DIR/models/model.json" && -f "$TRAIN_DIR/models/novelty.cfg" && -f "$TRAIN_DIR/models/calibrator.json" ]]; then
  set +e
  run_and_capture package_detector "${PACKAGE_CMD[@]}"
  PACKAGE_STATUS=$?
  set -e
else
  PACKAGE_STATUS="skipped"
  record_skipped_command package_detector "training run did not export models; see train_pipeline.log for deployment gate details"
fi

copy_report "$ROOT_DIR/deployments/DetectorRB3/config/model.json" "$INHERITED_EXPORT_DIR/config/model.json"
copy_report "$ROOT_DIR/deployments/DetectorRB3/config/novelty.cfg" "$INHERITED_EXPORT_DIR/config/novelty.cfg"
copy_report "$ROOT_DIR/deployments/DetectorRB3/config/calibrator.json" "$INHERITED_EXPORT_DIR/config/calibrator.json"
copy_report "$ROOT_DIR/runtime.py" "$INHERITED_EXPORT_DIR/runtime.py"
copy_report "$ROOT_DIR/bg_pcyber.py" "$INHERITED_EXPORT_DIR/bg_pcyber.py"
copy_report "$ROOT_DIR/README.md" "$INHERITED_EXPORT_DIR/README.md"
copy_report "$ROOT_DIR/AGENTS.md" "$INHERITED_EXPORT_DIR/AGENTS.md"
cat >"$INHERITED_BUNDLE_DIR/README.txt" <<EOF
This directory preserves the repository's checked-in deployed runtime bundle from:
  $ROOT_DIR/deployments/DetectorRB3/config

It is captured separately from the fresh baseline training run because the live rerun
may fail the repository's deployment gate and therefore not emit a new models/ export.
EOF

if [[ -f "$INHERITED_EXPORT_DIR/config/model.json" && -f "$INHERITED_EXPORT_DIR/config/novelty.cfg" && -f "$INHERITED_EXPORT_DIR/config/calibrator.json" ]]; then
  (
    cd "$INHERITED_EXPORT_DIR"
    zip -qr "$INHERITED_BUNDLE_ZIP" \
      config/model.json \
      config/novelty.cfg \
      config/calibrator.json \
      runtime.py \
      bg_pcyber.py \
      README.md \
      AGENTS.md
  )
fi

copy_report "$SMOKE_DIR/reports/support_matrix.json" "$REPORTS_DIR/smoke_support_matrix.json"
copy_report "$SMOKE_DIR/reports/actual_run_observability.json" "$REPORTS_DIR/smoke_actual_run_observability.json"
copy_report "$SMOKE_DIR/reports/schema.json" "$REPORTS_DIR/smoke_schema.json"
copy_report "$SMOKE_DIR/reports/channel_inventory.json" "$REPORTS_DIR/smoke_channel_inventory.json"
copy_report "$SMOKE_DIR/reports/provenance_summary.json" "$REPORTS_DIR/smoke_provenance_summary.json"
copy_report "$SMOKE_DIR/reports/generation_summary.json" "$REPORTS_DIR/smoke_generation_summary.json"
copy_report "$SMOKE_DIR/reports/behavior_summary.json" "$REPORTS_DIR/smoke_behavior_summary.json"
copy_report "$SMOKE_DIR/reports/run_manifest.json" "$REPORTS_DIR/smoke_run_manifest.json"
copy_report "$TRAIN_DIR/reports/metrics.json" "$REPORTS_DIR/train_metrics.json"
copy_report "$TRAIN_DIR/reports/summary.txt" "$REPORTS_DIR/train_summary.txt"
copy_report "$TRAIN_DIR/reports/run_manifest.json" "$REPORTS_DIR/train_run_manifest.json"
copy_report "$TRAIN_DIR/reports/dataset_sanity.json" "$REPORTS_DIR/train_dataset_sanity.json"
copy_report "$TRAIN_DIR/reports/command_family_overlap.json" "$REPORTS_DIR/train_command_family_overlap.json"
copy_report "$TRAIN_DIR/reports/behavior_summary.json" "$REPORTS_DIR/train_behavior_summary.json"
copy_report "$TRAIN_DIR/reports/benign_nominal_policy.json" "$REPORTS_DIR/train_benign_nominal_policy.json"
copy_report "$TRAIN_DIR/reports/actual_run_observability.json" "$REPORTS_DIR/train_actual_run_observability.json"
copy_report "$TRAIN_DIR/reports/provenance_summary.json" "$REPORTS_DIR/train_provenance_summary.json"
copy_report "$TRAIN_DIR/reports/support_matrix.json" "$REPORTS_DIR/train_support_matrix.json"
copy_report "$TRAIN_DIR/reports/generation_summary.json" "$REPORTS_DIR/train_generation_summary.json"
copy_report "$TRAIN_DIR/reports/schema.json" "$REPORTS_DIR/train_schema.json"

find "$SMOKE_DIR" -type f | LC_ALL=C sort >"$TREES_DIR/smoke_files.txt"
find "$TRAIN_DIR" -type f | LC_ALL=C sort >"$TREES_DIR/train_files.txt"
find "$INHERITED_BUNDLE_DIR" -type f | LC_ALL=C sort >"$TREES_DIR/inherited_deployment_bundle_files.txt"
if [[ -f "$PACKAGE_PATH" ]]; then
  printf '%s\n' "$PACKAGE_PATH" >"$TREES_DIR/package_files.txt"
fi

CHECKSUM_INPUTS=(
  "$METADATA_DIR/environment.txt"
  "$METADATA_DIR/git_status.short.txt"
  "$COMMANDS_DIR/fast_regression.log"
  "$COMMANDS_DIR/smoke_test.log"
  "$COMMANDS_DIR/train_pipeline.log"
  "$COMMANDS_DIR/package_detector.log"
  "$INHERITED_BUNDLE_DIR/README.txt"
)
if [[ -f "$REPORTS_DIR/smoke_support_matrix.json" ]]; then
  CHECKSUM_INPUTS+=("$REPORTS_DIR/smoke_support_matrix.json")
fi
if [[ -f "$REPORTS_DIR/train_metrics.json" ]]; then
  CHECKSUM_INPUTS+=("$REPORTS_DIR/train_metrics.json")
fi
if [[ -f "$REPORTS_DIR/train_summary.txt" ]]; then
  CHECKSUM_INPUTS+=("$REPORTS_DIR/train_summary.txt")
fi
if [[ -f "$PACKAGE_PATH" ]]; then
  CHECKSUM_INPUTS+=("$PACKAGE_PATH")
fi
if [[ -f "$INHERITED_BUNDLE_ZIP" ]]; then
  CHECKSUM_INPUTS+=("$INHERITED_BUNDLE_ZIP")
fi
shasum -a 256 "${CHECKSUM_INPUTS[@]}" >"$METADATA_DIR/checksums.sha256"

cat >"$METADATA_DIR/run_summary.txt" <<EOF
snapshot_id: $SNAPSHOT_ID
smoke_dir: $SMOKE_DIR
train_dir: $TRAIN_DIR
package_bundle: $PACKAGE_PATH
inherited_bundle: $INHERITED_BUNDLE_ZIP
seed: $SEED
rows: $ROWS
nominal_ratio: $NOMINAL_RATIO
train_exit_code: $TRAIN_STATUS
package_status: $PACKAGE_STATUS
train_exported_models: $(if [[ -f "$TRAIN_DIR/models/model.json" ]]; then echo true; else echo false; fi)
EOF

echo "baseline snapshot complete"
echo "  snapshot_root: $SNAPSHOT_ROOT"
echo "  generated_run: $GENERATED_DIR"
echo "  smoke_dir: $SMOKE_DIR"
echo "  train_dir: $TRAIN_DIR"
echo "  package_bundle: $PACKAGE_PATH"
