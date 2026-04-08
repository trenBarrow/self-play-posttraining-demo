#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SMOKE_DIR="${MAVLINK_SCHEDULE_SMOKE_DIR:-$ROOT_DIR/artifacts/mavlink_schedule_smoke}"
RUNTIME_ROOT="$SMOKE_DIR/mavlink_real"
SCHEDULE_PATH="$RUNTIME_ROOT/schedules/benign_smoke.csv"
RUN_LOG_PATH="$RUNTIME_ROOT/logs/schedule_runs/benign_smoke_run.csv"

cleanup() {
  "$ROOT_DIR/scripts/mavlink_real/down.sh" >/dev/null 2>&1 || true
}
trap cleanup EXIT

"$ROOT_DIR/scripts/mavlink_real/up.sh" --runtime-root "$RUNTIME_ROOT"

"$PYTHON_BIN" "$ROOT_DIR/tools/mavlink_real/make_good_schedule.py" \
  --target-rows 6 \
  --seed 7 \
  --episode-span 6 \
  --out "$SCHEDULE_PATH"

"$PYTHON_BIN" "$ROOT_DIR/tools/mavlink_real/run_mavlink_schedule.py" \
  --schedule "$SCHEDULE_PATH" \
  --time-scale 14400 \
  --timeout-seconds 8 \
  --output "$RUN_LOG_PATH"

"$PYTHON_BIN" - <<'PY' "$RUN_LOG_PATH"
import csv
import sys
from pathlib import Path

path = Path(sys.argv[1])
if not path.is_file():
    raise SystemExit(f"Missing MAVLink run log at {path}")
with path.open("r", newline="", encoding="utf-8") as handle:
    reader = csv.DictReader(handle)
    rows = list(reader)
if not rows:
    raise SystemExit("MAVLink schedule smoke test produced no run rows")
print(f"validated_rows={len(rows)}")
PY

echo "Schedule smoke test passed"
echo "  schedule: $SCHEDULE_PATH"
echo "  run log: $RUN_LOG_PATH"
