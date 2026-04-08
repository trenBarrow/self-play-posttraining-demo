#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"

DATASET_PATH=""
SELF_PLAY_OUTPUT_DIR=""
EVALUATION_MATRIX_PATH=""
RED_BLUE_EVALUATION_PATH=""
OUTPUT_DIR="$ROOT_DIR/artifacts/poster_asset_workflow_latest"
SEED=7
TRAINING_PATH="poster_default_canonical"
MAX_HISTORY_ENTRIES=4
MAKE_PLOTS=0
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
  bash scripts/poster_generate_assets.sh [options]

Options:
  --dataset PATH                  Dataset used to build evaluation_matrix.json when one is not supplied.
  --self-play-output-dir PATH     Self-play root used to build red_blue_evaluation.json when one is not supplied.
  --evaluation-matrix PATH        Existing evaluation_matrix.json to reuse directly.
  --red-blue-evaluation PATH      Existing red_blue_evaluation.json to reuse directly.
  --output-dir DIR
  --seed N
  --training-path NAME
  --max-history-entries N
  --make-plots
  --dry-run
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --dataset)
      DATASET_PATH="$2"
      shift 2
      ;;
    --self-play-output-dir)
      SELF_PLAY_OUTPUT_DIR="$2"
      shift 2
      ;;
    --evaluation-matrix)
      EVALUATION_MATRIX_PATH="$2"
      shift 2
      ;;
    --red-blue-evaluation)
      RED_BLUE_EVALUATION_PATH="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --training-path)
      TRAINING_PATH="$2"
      shift 2
      ;;
    --max-history-entries)
      MAX_HISTORY_ENTRIES="$2"
      shift 2
      ;;
    --make-plots)
      MAKE_PLOTS=1
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
WORKFLOW_REPORT_DIR="$OUTPUT_DIR/reports"
EVALUATION_DIR="$OUTPUT_DIR/evaluation"
RED_BLUE_DIR="$OUTPUT_DIR/red_blue"
ASSET_DIR="$OUTPUT_DIR/poster_assets"
SUMMARY_PATH="$WORKFLOW_REPORT_DIR/poster_asset_workflow_summary.txt"

mkdir -p "$WORKFLOW_REPORT_DIR"

if [[ -n "$DATASET_PATH" ]]; then
  DATASET_PATH="$(resolve_path "$DATASET_PATH")"
fi
if [[ -n "$SELF_PLAY_OUTPUT_DIR" ]]; then
  SELF_PLAY_OUTPUT_DIR="$(resolve_path "$SELF_PLAY_OUTPUT_DIR")"
fi
if [[ -n "$EVALUATION_MATRIX_PATH" ]]; then
  EVALUATION_MATRIX_PATH="$(resolve_path "$EVALUATION_MATRIX_PATH")"
fi
if [[ -n "$RED_BLUE_EVALUATION_PATH" ]]; then
  RED_BLUE_EVALUATION_PATH="$(resolve_path "$RED_BLUE_EVALUATION_PATH")"
fi

GENERALIZATION_CMD=(
  "$PYTHON_BIN" "$ROOT_DIR/tools/train/evaluate_generalization.py"
  --dataset "$DATASET_PATH"
  --output-dir "$EVALUATION_DIR"
  --seed "$SEED"
  --training-path "$TRAINING_PATH"
)
if [[ "$MAKE_PLOTS" -eq 1 ]]; then
  GENERALIZATION_CMD+=(--make-plots)
fi

RED_BLUE_CMD=(
  "$PYTHON_BIN" "$ROOT_DIR/tools/train/evaluate_red_vs_blue.py"
  --self-play-output-dir "$SELF_PLAY_OUTPUT_DIR"
  --max-history-entries "$MAX_HISTORY_ENTRIES"
  --output-dir "$RED_BLUE_DIR"
)

if [[ -z "$EVALUATION_MATRIX_PATH" && -z "$DATASET_PATH" ]]; then
  echo "Poster asset workflow failed: provide --evaluation-matrix or --dataset" >&2
  exit 1
fi
if [[ -z "$RED_BLUE_EVALUATION_PATH" && -z "$SELF_PLAY_OUTPUT_DIR" ]]; then
  echo "Poster asset workflow failed: provide --red-blue-evaluation or --self-play-output-dir" >&2
  exit 1
fi

if [[ "$DRY_RUN" -eq 1 ]]; then
  echo "output_dir: $OUTPUT_DIR"
  if [[ -n "$EVALUATION_MATRIX_PATH" ]]; then
    echo "using existing evaluation_matrix: $EVALUATION_MATRIX_PATH"
  else
    print_cmd "${GENERALIZATION_CMD[@]}"
  fi
  if [[ -n "$RED_BLUE_EVALUATION_PATH" ]]; then
    echo "using existing red_blue_evaluation: $RED_BLUE_EVALUATION_PATH"
  else
    print_cmd "${RED_BLUE_CMD[@]}"
  fi
  echo "python3 $ROOT_DIR/tools/figures/generate_poster_assets.py --evaluation-matrix <resolved-evaluation-matrix> --red-blue-evaluation <resolved-red-blue-evaluation> --output-dir $ASSET_DIR"
  exit 0
fi

if [[ -z "$EVALUATION_MATRIX_PATH" ]]; then
  "${GENERALIZATION_CMD[@]}"
  EVALUATION_MATRIX_PATH="$EVALUATION_DIR/reports/evaluation_matrix.json"
fi
if [[ ! -f "$EVALUATION_MATRIX_PATH" ]]; then
  echo "Poster asset workflow failed: missing evaluation matrix at $EVALUATION_MATRIX_PATH" >&2
  exit 1
fi

if [[ -z "$RED_BLUE_EVALUATION_PATH" ]]; then
  "${RED_BLUE_CMD[@]}"
  RED_BLUE_EVALUATION_PATH="$RED_BLUE_DIR/reports/red_blue_evaluation.json"
fi
if [[ ! -f "$RED_BLUE_EVALUATION_PATH" ]]; then
  echo "Poster asset workflow failed: missing red-blue evaluation at $RED_BLUE_EVALUATION_PATH" >&2
  exit 1
fi

"$PYTHON_BIN" "$ROOT_DIR/tools/figures/generate_poster_assets.py" \
  --evaluation-matrix "$EVALUATION_MATRIX_PATH" \
  --red-blue-evaluation "$RED_BLUE_EVALUATION_PATH" \
  --output-dir "$ASSET_DIR"

{
  echo "output_dir=$OUTPUT_DIR"
  echo "evaluation_matrix_path=$EVALUATION_MATRIX_PATH"
  echo "red_blue_evaluation_path=$RED_BLUE_EVALUATION_PATH"
  echo "asset_output_dir=$ASSET_DIR"
  echo "seed=$SEED"
  echo "training_path=$TRAINING_PATH"
  echo "max_history_entries=$MAX_HISTORY_ENTRIES"
} > "$SUMMARY_PATH"

echo "Poster asset workflow completed"
echo "  evaluation matrix: $EVALUATION_MATRIX_PATH"
echo "  red-blue evaluation: $RED_BLUE_EVALUATION_PATH"
echo "  asset dir: $ASSET_DIR"
