#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-}"
OUTPUT_DIR="$ROOT_DIR/artifacts/latest"
ROWS=240
NOMINAL_RATIO=0.55
SEED=7
NO_PLOTS="true"
SKIP_GENERATE="false"
KEEP_STACK_UP="false"
LEGACY_PATH="false"
DRY_RUN="false"

usage() {
  cat <<'EOF'
Usage: ./scripts/fprime_real/train_pipeline.sh [options]

Options:
  --rows N                 Total rows for `main.py run` (default: 240)
  --nominal-ratio R        Benign ratio for generation (default: 0.55)
  --seed N                 Training/generation seed (default: 7)
  --output-dir PATH        Output artifact directory (default: artifacts/latest)
  --python PATH            Python interpreter to use
  --with-plots             Generate plots during training
  --legacy                 Use the explicit legacy F´ baseline path (comparison-only) instead of the poster-default canonical path
  --skip-generate          Reuse an existing dataset in --output-dir
  --keep-stack-up          Do not stop the F' stack on exit
  --dry-run                Print the resolved mode and command without running generation/training
  -h, --help               Show this help text
EOF
}

while [[ $# -gt 0 ]]; do
  case "$1" in
    --rows)
      ROWS="$2"
      shift 2
      ;;
    --nominal-ratio)
      NOMINAL_RATIO="$2"
      shift 2
      ;;
    --seed)
      SEED="$2"
      shift 2
      ;;
    --output-dir)
      OUTPUT_DIR="$2"
      shift 2
      ;;
    --python)
      PYTHON_BIN="$2"
      shift 2
      ;;
    --with-plots)
      NO_PLOTS="false"
      shift
      ;;
    --legacy)
      LEGACY_PATH="true"
      shift
      ;;
    --skip-generate)
      SKIP_GENERATE="true"
      shift
      ;;
    --keep-stack-up)
      KEEP_STACK_UP="true"
      shift
      ;;
    --dry-run)
      DRY_RUN="true"
      shift
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

cleanup() {
  if [[ "$DRY_RUN" == "true" ]]; then
    return 0
  fi
  if [[ "$KEEP_STACK_UP" != "true" ]]; then
    "$ROOT_DIR/scripts/fprime_real/down.sh" >/dev/null 2>&1 || true
  fi
}
trap cleanup EXIT

mkdir -p "$OUTPUT_DIR"

RUN_SUBCOMMAND="run"
if [[ "$LEGACY_PATH" == "true" ]]; then
  RUN_SUBCOMMAND="run-legacy"
fi

MODE_LABEL="poster_default_headline"
COMPARISON_ONLY="false"
if [[ "$LEGACY_PATH" == "true" ]]; then
  MODE_LABEL="legacy_comparison_only"
  COMPARISON_ONLY="true"
fi

RUN_CMD=(
  "$PYTHON_BIN"
  "$ROOT_DIR/main.py"
  "$RUN_SUBCOMMAND"
  --rows "$ROWS"
  --nominal-ratio "$NOMINAL_RATIO"
  --seed "$SEED"
  --output-dir "$OUTPUT_DIR"
)

if [[ "$NO_PLOTS" == "true" ]]; then
  RUN_CMD+=(--no-plots)
fi

if [[ "$SKIP_GENERATE" == "true" ]]; then
  RUN_CMD+=(--skip-generate)
fi

echo "training pipeline"
echo "  python: $PYTHON_BIN"
echo "  output: $OUTPUT_DIR"
echo "  rows: $ROWS"
echo "  nominal_ratio: $NOMINAL_RATIO"
echo "  seed: $SEED"
echo "  training_path: $([[ \"$LEGACY_PATH\" == \"true\" ]] && echo legacy_fprime_baseline || echo poster_default_canonical)"
echo "  mode: $MODE_LABEL"
echo "  comparison_only: $COMPARISON_ONLY"
echo "  run_subcommand: $RUN_SUBCOMMAND"
echo "  plots: $([[ \"$NO_PLOTS\" == \"true\" ]] && echo no || echo yes)"
echo "  skip_generate: $SKIP_GENERATE"
echo "  keep_stack_up: $KEEP_STACK_UP"
echo "  dry_run: $DRY_RUN"
printf '  command:'
for token in "${RUN_CMD[@]}"; do
  printf ' %q' "$token"
done
printf '\n'
echo

if [[ "$DRY_RUN" == "true" ]]; then
  exit 0
fi

if ! command -v docker >/dev/null 2>&1; then
  echo "docker is required" >&2
  exit 1
fi

if ! docker compose version >/dev/null 2>&1; then
  echo "docker compose v2 is required by the F' scripts in this repo" >&2
  exit 1
fi

if ! "$PYTHON_BIN" -c "import dpkt, matplotlib, numpy, sklearn" >/dev/null 2>&1; then
  echo "Python deps are missing for $PYTHON_BIN. Install requirements.txt first." >&2
  exit 1
fi

"${RUN_CMD[@]}"

echo
echo "training pipeline complete"
echo "  metrics: $OUTPUT_DIR/reports/metrics.json"
echo "  packets: $OUTPUT_DIR/data/packets.jsonl"
echo "  dataset: $OUTPUT_DIR/data/dataset.jsonl"
