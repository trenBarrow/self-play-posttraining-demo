#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

MANIFEST_NAME="bundle_manifest.json"
POSTER_MODEL_NAME="blue_model.json"
LEGACY_MODEL_NAME="model.json"

usage() {
  cat <<'EOF'
Usage:
  bash tools/scripts/package_detector.sh [RUN_DIR]
  bash tools/scripts/package_detector.sh --run-dir RUN_DIR [--model-dir MODEL_DIR] [--output-zip ZIP] [--deploy-dir DIR] [--dry-run]

Options:
  --run-dir DIR     Artifact root whose reports/scored outputs should be bundled.
  --model-dir DIR   Explicit runtime bundle source. Defaults to RUN_DIR/models.
  --output-zip ZIP  Output archive path. Defaults to RUN_DIR/detector_bundle.zip.
  --deploy-dir DIR  Deployment config directory to refresh before packaging.
  --dry-run         Print resolved packaging inputs and staged files without writing the zip.
EOF
}

resolve_path() {
  "$PYTHON_BIN" - <<'PY' "$1"
from pathlib import Path
import sys

print(Path(sys.argv[1]).expanduser().resolve())
PY
}

same_resolved_path() {
  "$PYTHON_BIN" - <<'PY' "$1" "$2"
from pathlib import Path
import sys

print("true" if Path(sys.argv[1]).expanduser().resolve() == Path(sys.argv[2]).expanduser().resolve() else "false")
PY
}

print_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

stage_copy() {
  local src="$1"
  local rel="$2"
  mkdir -p "$STAGE_DIR/$(dirname "$rel")"
  cp "$src" "$STAGE_DIR/$rel"
}

append_run_artifact_if_exists() {
  local rel_src="$1"
  local rel_dst="$2"
  local abs_src="$RUN_DIR/$rel_src"
  if [[ -e "$abs_src" ]]; then
    stage_copy "$abs_src" "$rel_dst"
  fi
}

is_poster_bundle_dir() {
  local dir="$1"
  [[ -f "$dir/$POSTER_MODEL_NAME" && -f "$dir/$MANIFEST_NAME" ]]
}

is_legacy_bundle_dir() {
  local dir="$1"
  [[ -f "$dir/$LEGACY_MODEL_NAME" && -f "$dir/novelty.cfg" && -f "$dir/calibrator.json" ]]
}

RUN_DIR=""
MODEL_DIR=""
DEPLOY_DIR=""
OUT_ZIP=""
DRY_RUN=0

while [[ $# -gt 0 ]]; do
  case "$1" in
    --run-dir)
      RUN_DIR="$2"
      shift 2
      ;;
    --model-dir)
      MODEL_DIR="$2"
      shift 2
      ;;
    --output-zip)
      OUT_ZIP="$2"
      shift 2
      ;;
    --deploy-dir)
      DEPLOY_DIR="$2"
      shift 2
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    -*)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
    *)
      if [[ -n "$RUN_DIR" ]]; then
        echo "unexpected extra argument: $1" >&2
        usage >&2
        exit 2
      fi
      RUN_DIR="$1"
      shift
      ;;
  esac
done

RUN_DIR="${RUN_DIR:-$ROOT_DIR/artifacts/latest}"
RUN_DIR="$(resolve_path "$RUN_DIR")"
MODEL_DIR="${MODEL_DIR:-$RUN_DIR/models}"
MODEL_DIR="$(resolve_path "$MODEL_DIR")"
DEPLOY_DIR="${DEPLOY_DIR:-$ROOT_DIR/deployments/DetectorRB3/config}"
DEPLOY_DIR="$(resolve_path "$DEPLOY_DIR")"
OUT_ZIP="${OUT_ZIP:-$RUN_DIR/detector_bundle.zip}"
OUT_ZIP="$(resolve_path "$OUT_ZIP")"

STAGE_DIR="$RUN_DIR/.detector_bundle_stage"
SUMMARY_PATH="$RUN_DIR/reports/package_bundle_summary.txt"

mkdir -p "$RUN_DIR" "$DEPLOY_DIR" "$RUN_DIR/reports" "$(dirname "$OUT_ZIP")"
rm -f "$OUT_ZIP"
rm -rf "$STAGE_DIR"
mkdir -p "$STAGE_DIR"

cleanup() {
  rm -rf "$STAGE_DIR"
}
trap cleanup EXIT

RUNTIME_KIND=""
COMPARISON_ONLY="false"
PACKAGE_MODE=""

if is_poster_bundle_dir "$MODEL_DIR"; then
  RUNTIME_KIND="poster_blue_single_model_v1"
  COMPARISON_ONLY="false"
  PACKAGE_MODE="poster_runtime_bundle"
  if [[ "$(same_resolved_path "$MODEL_DIR/$POSTER_MODEL_NAME" "$DEPLOY_DIR/$POSTER_MODEL_NAME")" != "true" ]]; then
    cp "$MODEL_DIR/$POSTER_MODEL_NAME" "$DEPLOY_DIR/$POSTER_MODEL_NAME"
  fi
  if [[ "$(same_resolved_path "$MODEL_DIR/$MANIFEST_NAME" "$DEPLOY_DIR/$MANIFEST_NAME")" != "true" ]]; then
    cp "$MODEL_DIR/$MANIFEST_NAME" "$DEPLOY_DIR/$MANIFEST_NAME"
  fi
  rm -f "$DEPLOY_DIR/$LEGACY_MODEL_NAME" "$DEPLOY_DIR/novelty.cfg" "$DEPLOY_DIR/calibrator.json"
  stage_copy "$DEPLOY_DIR/$POSTER_MODEL_NAME" "deployments/DetectorRB3/config/$POSTER_MODEL_NAME"
  stage_copy "$DEPLOY_DIR/$MANIFEST_NAME" "deployments/DetectorRB3/config/$MANIFEST_NAME"
elif is_legacy_bundle_dir "$MODEL_DIR"; then
  RUNTIME_KIND="legacy_fprime_comparison_bundle"
  COMPARISON_ONLY="true"
  PACKAGE_MODE="legacy_runtime_bundle"
  if [[ "$(same_resolved_path "$MODEL_DIR/$LEGACY_MODEL_NAME" "$DEPLOY_DIR/$LEGACY_MODEL_NAME")" != "true" ]]; then
    cp "$MODEL_DIR/$LEGACY_MODEL_NAME" "$DEPLOY_DIR/$LEGACY_MODEL_NAME"
  fi
  if [[ "$(same_resolved_path "$MODEL_DIR/novelty.cfg" "$DEPLOY_DIR/novelty.cfg")" != "true" ]]; then
    cp "$MODEL_DIR/novelty.cfg" "$DEPLOY_DIR/novelty.cfg"
  fi
  if [[ "$(same_resolved_path "$MODEL_DIR/calibrator.json" "$DEPLOY_DIR/calibrator.json")" != "true" ]]; then
    cp "$MODEL_DIR/calibrator.json" "$DEPLOY_DIR/calibrator.json"
  fi
  if [[ -f "$MODEL_DIR/$MANIFEST_NAME" ]]; then
    if [[ "$(same_resolved_path "$MODEL_DIR/$MANIFEST_NAME" "$DEPLOY_DIR/$MANIFEST_NAME")" != "true" ]]; then
      cp "$MODEL_DIR/$MANIFEST_NAME" "$DEPLOY_DIR/$MANIFEST_NAME"
    fi
  fi
  rm -f "$DEPLOY_DIR/$POSTER_MODEL_NAME"
  stage_copy "$DEPLOY_DIR/$LEGACY_MODEL_NAME" "deployments/DetectorRB3/config/$LEGACY_MODEL_NAME"
  stage_copy "$DEPLOY_DIR/novelty.cfg" "deployments/DetectorRB3/config/novelty.cfg"
  stage_copy "$DEPLOY_DIR/calibrator.json" "deployments/DetectorRB3/config/calibrator.json"
  if [[ -f "$DEPLOY_DIR/$MANIFEST_NAME" ]]; then
    stage_copy "$DEPLOY_DIR/$MANIFEST_NAME" "deployments/DetectorRB3/config/$MANIFEST_NAME"
  fi
else
  echo "missing supported runtime bundle in model dir: $MODEL_DIR" >&2
  exit 1
fi

REPO_CONTRACT_FILES=(
  "runtime.py"
  "bg_pcyber.py"
  "README.md"
  "AGENTS.md"
  "TODO.md"
  "docs/blue_runtime_bundle.md"
  "docs/blue_feature_contract.md"
  "docs/canonical_semantic_schema.md"
  "docs/canonical_state_mapping.md"
  "docs/poster_contract.md"
  "configs/feature_policies/blue_allowed_features.yaml"
  "configs/feature_policies/blue_forbidden_features.yaml"
  "schemas/canonical_command_row.schema.json"
  "schemas/raw_packet.schema.json"
  "schemas/raw_transaction.schema.json"
)

for rel in "${REPO_CONTRACT_FILES[@]}"; do
  stage_copy "$ROOT_DIR/$rel" "$rel"
done

append_run_artifact_if_exists "reports/metrics.json" "run/reports/metrics.json"
append_run_artifact_if_exists "reports/summary.txt" "run/reports/summary.txt"
append_run_artifact_if_exists "reports/schema.json" "run/reports/schema.json"
append_run_artifact_if_exists "reports/evaluation_matrix.json" "run/reports/evaluation_matrix.json"
append_run_artifact_if_exists "reports/evaluation_matrix_summary.txt" "run/reports/evaluation_matrix_summary.txt"
append_run_artifact_if_exists "reports/red_blue_evaluation.json" "run/reports/red_blue_evaluation.json"
append_run_artifact_if_exists "reports/red_blue_evaluation_summary.txt" "run/reports/red_blue_evaluation_summary.txt"
append_run_artifact_if_exists "reports/poster_demo_manifest.json" "run/reports/poster_demo_manifest.json"
append_run_artifact_if_exists "reports/poster_demo_summary.txt" "run/reports/poster_demo_summary.txt"
append_run_artifact_if_exists "scored/summary.json" "run/scored/summary.json"

{
  echo "runtime_kind=$RUNTIME_KIND"
  echo "comparison_only=$COMPARISON_ONLY"
  echo "package_mode=$PACKAGE_MODE"
  echo "source_model_dir=$MODEL_DIR"
  echo "deploy_dir=$DEPLOY_DIR"
  echo "run_dir=$RUN_DIR"
  echo "output_zip=$OUT_ZIP"
} > "$SUMMARY_PATH"
stage_copy "$SUMMARY_PATH" "run/reports/package_bundle_summary.txt"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "run_dir: $RUN_DIR"
  echo "model_dir: $MODEL_DIR"
  echo "runtime_kind: $RUNTIME_KIND"
  echo "comparison_only: $COMPARISON_ONLY"
  echo "output_zip: $OUT_ZIP"
  echo "staged_files:"
  (
    cd "$STAGE_DIR"
    find . -type f | sort
  )
  exit 0
fi

(
  cd "$STAGE_DIR"
  zip -qr "$OUT_ZIP" .
)

echo "$OUT_ZIP"
