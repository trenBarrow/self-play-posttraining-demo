#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
MIN_TRAIN_ROWS=100

ROWS=24
SEED=7
PROTOCOL_MODE="fprime"
MIXED_FPRIME_RATIO="0.5"
OUTPUT_DIR="$ROOT_DIR/artifacts/poster_demo_latest"
MODEL_DIR_OVERRIDE=""
ALLOW_DEPLOYMENT_FALLBACK=1
NO_PLOTS=1
SKIP_GENERATE=0
SKIP_TRAIN=0
SKIP_SCORE=0
SKIP_PACKAGE=0
DRY_RUN=0

resolve_path() {
  "$PYTHON_BIN" - <<'PY' "$1"
from pathlib import Path
import sys

print(Path(sys.argv[1]).expanduser().resolve())
PY
}

print_cmd() {
  printf '%q ' "$@"
  printf '\n'
}

usage() {
  cat <<'EOF'
Usage:
  bash scripts/poster_demo.sh [options]

Options:
  --rows N
  --seed N
  --protocol-mode {fprime,mavlink,mixed}
  --mixed-fprime-ratio R
  --output-dir DIR
  --model-dir DIR
  --allow-deployment-fallback
  --no-deployment-fallback
  --with-plots
  --skip-generate
  --skip-train
  --skip-score
  --skip-package
  --dry-run
EOF
}

is_supported_bundle_dir() {
  local dir="$1"
  [[ -f "$dir/bundle_manifest.json" && -f "$dir/blue_model.json" ]] && return 0
  [[ -f "$dir/model.json" && -f "$dir/novelty.cfg" && -f "$dir/calibrator.json" ]] && return 0
  return 1
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rows)
      ROWS="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --protocol-mode)
      PROTOCOL_MODE="$2"
      shift 2
      ;;
    --mixed-fprime-ratio)
      MIXED_FPRIME_RATIO="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --model-dir)
      MODEL_DIR_OVERRIDE="$2"
      shift 2
      ;;
    --allow-deployment-fallback)
      ALLOW_DEPLOYMENT_FALLBACK=1
      shift
      ;;
    --no-deployment-fallback)
      ALLOW_DEPLOYMENT_FALLBACK=0
      shift
      ;;
    --with-plots)
      NO_PLOTS=0
      shift
      ;;
    --skip-generate)
      SKIP_GENERATE=1
      shift
      ;;
    --skip-train)
      SKIP_TRAIN=1
      shift
      ;;
    --skip-score)
      SKIP_SCORE=1
      shift
      ;;
    --skip-package)
      SKIP_PACKAGE=1
      shift
      ;;
    --dry-run)
      DRY_RUN=1
      shift
      ;;
    -h|--help)
      usage
      exit 0
      ;;
    *)
      echo "unknown option: $1" >&2
      usage >&2
      exit 2
      ;;
  esac
done

OUTPUT_DIR="$(resolve_path "$OUTPUT_DIR")"
DATASET_PATH="$OUTPUT_DIR/data/dataset.jsonl"
PACKETS_PATH="$OUTPUT_DIR/data/packets.jsonl"
REPORT_DIR="$OUTPUT_DIR/reports"
SCORING_SUMMARY_PATH="$OUTPUT_DIR/scored/summary.json"
DEMO_SUMMARY_PATH="$REPORT_DIR/poster_demo_summary.txt"
DEMO_MANIFEST_PATH="$REPORT_DIR/poster_demo_manifest.json"
PACKAGE_PATH="$OUTPUT_DIR/detector_bundle.zip"
DEFAULT_MODEL_DIR="$ROOT_DIR/deployments/DetectorRB3/config"

GENERATE_CMD=(
  "$PYTHON_BIN" "$ROOT_DIR/main.py" generate
  --rows "$ROWS"
  --seed "$SEED"
  --protocol-mode "$PROTOCOL_MODE"
  --mixed-fprime-ratio "$MIXED_FPRIME_RATIO"
  --output-dir "$OUTPUT_DIR"
)
TRAIN_CMD=(
  "$PYTHON_BIN" "$ROOT_DIR/main.py" train
  --dataset "$DATASET_PATH"
  --seed "$SEED"
  --output-dir "$OUTPUT_DIR"
)
if [[ "$NO_PLOTS" -eq 1 ]]; then
  TRAIN_CMD+=(--no-plots)
fi

mkdir -p "$REPORT_DIR"

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "output_dir: $OUTPUT_DIR"
  echo "dataset_path: $DATASET_PATH"
  echo "packets_path: $PACKETS_PATH"
  echo "protocol_mode: $PROTOCOL_MODE"
  echo "seed: $SEED"
  echo "rows: $ROWS"
  echo "minimum_training_rows: $MIN_TRAIN_ROWS"
  echo "allow_deployment_fallback: $ALLOW_DEPLOYMENT_FALLBACK"
  if [[ "$SKIP_GENERATE" -eq 0 ]]; then
    print_cmd "${GENERATE_CMD[@]}"
  fi
  if [[ "$SKIP_TRAIN" -eq 0 ]]; then
    print_cmd "${TRAIN_CMD[@]}"
    if (( ROWS < MIN_TRAIN_ROWS )); then
      echo "fresh training will be skipped after generation because rows=$ROWS is below minimum_training_rows=$MIN_TRAIN_ROWS"
    fi
  fi
  echo "score_model_dir_resolution_order:"
  if [[ -n "$MODEL_DIR_OVERRIDE" ]]; then
    echo "  1. explicit --model-dir $(resolve_path "$MODEL_DIR_OVERRIDE")"
  fi
  echo "  2. fresh training bundle at $OUTPUT_DIR/models"
  if [[ "$ALLOW_DEPLOYMENT_FALLBACK" -eq 1 ]]; then
    echo "  3. checked-in deployment fallback at $DEFAULT_MODEL_DIR"
  fi
  if [[ "$SKIP_SCORE" -eq 0 ]]; then
    echo "python3 $ROOT_DIR/main.py score-packets --packets $PACKETS_PATH --model-dir <resolved-model-dir> --output-dir $OUTPUT_DIR"
  fi
  if [[ "$SKIP_PACKAGE" -eq 0 ]]; then
    echo "bash $ROOT_DIR/tools/scripts/package_detector.sh --run-dir $OUTPUT_DIR --model-dir <resolved-model-dir>"
  fi
  exit 0
fi

if [[ "$SKIP_GENERATE" -eq 0 ]]; then
  "${GENERATE_CMD[@]}"
fi

if [[ ! -f "$DATASET_PATH" ]]; then
  echo "Poster demo failed: missing dataset at $DATASET_PATH" >&2
  exit 1
fi
if [[ ! -f "$PACKETS_PATH" ]]; then
  echo "Poster demo failed: missing packets at $PACKETS_PATH" >&2
  exit 1
fi

training_status="skipped"
training_exit_code=0
training_blocked_reason=""
training_skip_reason=""
DATASET_ROW_COUNT="$(
  "$PYTHON_BIN" - <<'PY' "$DATASET_PATH"
import sys
from pathlib import Path

path = Path(sys.argv[1])
count = 0
with path.open("r", encoding="utf-8") as handle:
    for line in handle:
        if line.strip():
            count += 1
print(count)
PY
)"

if [[ "$SKIP_TRAIN" -eq 0 ]]; then
  if (( DATASET_ROW_COUNT < MIN_TRAIN_ROWS )); then
    training_status="skipped_insufficient_rows_for_fresh_training"
    training_skip_reason="minimum_training_rows=${MIN_TRAIN_ROWS}; actual_rows=${DATASET_ROW_COUNT}"
  else
    if "${TRAIN_CMD[@]}"; then
      training_status="completed"
    else
      training_exit_code=$?
      training_status="failed"
      if [[ -f "$REPORT_DIR/metrics.json" ]]; then
        training_blocked_reason="$(
          "$PYTHON_BIN" - <<'PY' "$REPORT_DIR/metrics.json"
import json
import sys
from pathlib import Path

payload = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
value = payload.get("deployment_blocked_reason")
print("" if value is None else str(value))
PY
        )"
        if [[ -n "$training_blocked_reason" ]]; then
          training_status="analysis_completed_export_blocked"
        else
          echo "Poster demo failed: training exited nonzero without a recorded deployment_blocked_reason" >&2
          exit "$training_exit_code"
        fi
      else
        echo "Poster demo failed: training exited nonzero and no metrics report was written" >&2
        exit "$training_exit_code"
      fi
    fi
  fi
fi

RESOLVED_MODEL_DIR=""
MODEL_SOURCE_KIND=""
if [[ "$SKIP_SCORE" -eq 0 || "$SKIP_PACKAGE" -eq 0 ]]; then
  if [[ -n "$MODEL_DIR_OVERRIDE" ]]; then
    RESOLVED_MODEL_DIR="$(resolve_path "$MODEL_DIR_OVERRIDE")"
    MODEL_SOURCE_KIND="explicit_model_dir"
  elif is_supported_bundle_dir "$OUTPUT_DIR/models"; then
    RESOLVED_MODEL_DIR="$OUTPUT_DIR/models"
    MODEL_SOURCE_KIND="fresh_training_output"
  elif [[ "$ALLOW_DEPLOYMENT_FALLBACK" -eq 1 && -d "$DEFAULT_MODEL_DIR" ]] && is_supported_bundle_dir "$DEFAULT_MODEL_DIR"; then
    RESOLVED_MODEL_DIR="$(resolve_path "$DEFAULT_MODEL_DIR")"
    MODEL_SOURCE_KIND="deployment_fallback"
  else
    echo "Poster demo failed: no supported runtime bundle available for scoring/package" >&2
    exit 1
  fi
else
  MODEL_SOURCE_KIND="not_needed"
fi

if [[ -n "$RESOLVED_MODEL_DIR" ]] && ! is_supported_bundle_dir "$RESOLVED_MODEL_DIR"; then
  echo "Poster demo failed: unsupported runtime bundle at $RESOLVED_MODEL_DIR" >&2
  exit 1
fi

{
  echo "output_dir=$OUTPUT_DIR"
  echo "dataset_path=$DATASET_PATH"
  echo "packets_path=$PACKETS_PATH"
  echo "report_dir=$REPORT_DIR"
  echo "scoring_summary_path=$SCORING_SUMMARY_PATH"
  echo "package_path=$PACKAGE_PATH"
  echo "protocol_mode=$PROTOCOL_MODE"
  echo "rows=$ROWS"
  echo "dataset_row_count=$DATASET_ROW_COUNT"
  echo "minimum_training_rows=$MIN_TRAIN_ROWS"
  echo "seed=$SEED"
  echo "training_status=$training_status"
  if [[ -n "$training_blocked_reason" ]]; then
    echo "training_blocked_reason=$training_blocked_reason"
  fi
  if [[ -n "$training_skip_reason" ]]; then
    echo "training_skip_reason=$training_skip_reason"
  fi
  echo "resolved_model_dir=$RESOLVED_MODEL_DIR"
  echo "model_source_kind=$MODEL_SOURCE_KIND"
} > "$DEMO_SUMMARY_PATH"

"$PYTHON_BIN" - <<'PY' "$DEMO_MANIFEST_PATH" "$OUTPUT_DIR" "$DATASET_PATH" "$PACKETS_PATH" "$REPORT_DIR" "$SCORING_SUMMARY_PATH" "$PACKAGE_PATH" "$PROTOCOL_MODE" "$ROWS" "$DATASET_ROW_COUNT" "$MIN_TRAIN_ROWS" "$SEED" "$training_status" "$training_blocked_reason" "$training_skip_reason" "$RESOLVED_MODEL_DIR" "$MODEL_SOURCE_KIND"
import json
import sys
from pathlib import Path

manifest_path = Path(sys.argv[1])
payload = {
    "schema_version": "poster_demo_manifest.v1",
    "record_kind": "poster_demo_manifest",
    "output_dir": str(Path(sys.argv[2]).resolve()),
    "dataset_path": str(Path(sys.argv[3]).resolve()),
    "packets_path": str(Path(sys.argv[4]).resolve()),
    "report_dir": str(Path(sys.argv[5]).resolve()),
    "scoring_summary_path": str(Path(sys.argv[6]).resolve()),
    "package_path": str(Path(sys.argv[7]).resolve()),
    "protocol_mode": sys.argv[8],
    "rows": int(sys.argv[9]),
    "dataset_row_count": int(sys.argv[10]),
    "minimum_training_rows": int(sys.argv[11]),
    "seed": int(sys.argv[12]),
    "training_status": sys.argv[13],
    "training_blocked_reason": sys.argv[14] or None,
    "training_skip_reason": sys.argv[15] or None,
    "resolved_model_dir": str(Path(sys.argv[16]).resolve()) if sys.argv[16] else None,
    "model_source_kind": sys.argv[17],
}
manifest_path.parent.mkdir(parents=True, exist_ok=True)
manifest_path.write_text(json.dumps(payload, indent=2), encoding="utf-8")
PY

if [[ "$SKIP_SCORE" -eq 0 ]]; then
  "$PYTHON_BIN" "$ROOT_DIR/main.py" score-packets \
    --packets "$PACKETS_PATH" \
    --model-dir "$RESOLVED_MODEL_DIR" \
    --output-dir "$OUTPUT_DIR"
fi

if [[ "$SKIP_PACKAGE" -eq 0 ]]; then
  bash "$ROOT_DIR/tools/scripts/package_detector.sh" \
    --run-dir "$OUTPUT_DIR" \
    --model-dir "$RESOLVED_MODEL_DIR"
fi

echo "Poster demo completed"
echo "  output dir: $OUTPUT_DIR"
echo "  training status: $training_status"
if [[ -n "$training_blocked_reason" ]]; then
  echo "  deployment blocked reason: $training_blocked_reason"
fi
if [[ -n "$training_skip_reason" ]]; then
  echo "  training skip reason: $training_skip_reason"
fi
echo "  model source kind: $MODEL_SOURCE_KIND"
echo "  score summary: $SCORING_SUMMARY_PATH"
if [[ "$SKIP_PACKAGE" -eq 0 ]]; then
  echo "  detector bundle: $PACKAGE_PATH"
fi
