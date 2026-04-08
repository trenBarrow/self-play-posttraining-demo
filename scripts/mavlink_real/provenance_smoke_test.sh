#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SMOKE_DIR="${MAVLINK_PROVENANCE_SMOKE_DIR:-$ROOT_DIR/artifacts/mavlink_provenance_smoke}"
RUNTIME_ROOT="$SMOKE_DIR/mavlink_real"
OUTPUT_DIR="$SMOKE_DIR/reconstructed"
SCHEDULE_PATH="$RUNTIME_ROOT/schedules/benign_provenance_smoke.csv"
RUN_LOG_PATH="$RUNTIME_ROOT/logs/schedule_runs/benign_provenance_smoke_run.csv"
COMPOSE_FILE="$ROOT_DIR/orchestration/docker-compose.mavlink-real.yml"

cleanup() {
  "$ROOT_DIR/scripts/mavlink_real/down.sh" >/dev/null 2>&1 || true
}
trap cleanup EXIT

"$ROOT_DIR/scripts/mavlink_real/up.sh" --runtime-root "$RUNTIME_ROOT"

"$PYTHON_BIN" "$ROOT_DIR/tools/mavlink_real/make_good_schedule.py" \
  --target-rows 6 \
  --seed 9 \
  --episode-span 6 \
  --out "$SCHEDULE_PATH"

"$PYTHON_BIN" - <<'PY' "$ROOT_DIR" "$PYTHON_BIN" "$COMPOSE_FILE" "$RUNTIME_ROOT" "$SCHEDULE_PATH" "$RUN_LOG_PATH" "$OUTPUT_DIR"
import json
import subprocess
import sys
from pathlib import Path

repo_root = Path(sys.argv[1])
python_bin = sys.argv[2]
compose_file = Path(sys.argv[3])
runtime_root = Path(sys.argv[4])
schedule_path = Path(sys.argv[5])
run_log_path = Path(sys.argv[6])
output_dir = Path(sys.argv[7])

if str(repo_root) not in sys.path:
    sys.path.insert(0, str(repo_root))

from tools.mavlink_real.log_ingest import load_runtime_run_rows
from tools.mavlink_real.packet_fidelity import write_artifact_bundle
from tools.mavlink_real.pcap_capture import capture_pcap
from tools.mavlink_real.runtime_layout import host_capture_pcap_path
from tools.mavlink_real.support_probe import resolve_identity_capture_target, wait_for_capture_drain

resolved_capture = resolve_identity_capture_target(
    repo_root,
    compose_file,
    timeout_seconds=8.0,
    runtime_root=runtime_root,
)
pcap_path = host_capture_pcap_path(runtime_root, "provenance_smoke")
with capture_pcap(
    pcap_path,
    interface=resolved_capture.interface,
    backend=resolved_capture.backend,
) as session:
    subprocess.run(
        [
            python_bin,
            str(repo_root / "tools" / "mavlink_real" / "run_mavlink_schedule.py"),
            "--schedule",
            str(schedule_path),
            "--time-scale",
            "14400",
            "--timeout-seconds",
            "8",
            "--output",
            str(run_log_path),
        ],
        cwd=repo_root,
        check=True,
    )
    run_rows, source_artifact_paths = load_runtime_run_rows(run_log_path, runtime_root=runtime_root)
    source_artifact_paths.append(str(pcap_path.resolve()))
    packet_result = wait_for_capture_drain(
        run_rows,
        pcap_path=pcap_path,
        capture_interface=session.interface,
        capture_backend=resolved_capture.backend,
        source_artifact_paths=source_artifact_paths,
        timeout_seconds=8.0,
    )

data_dir, report_dir = write_artifact_bundle(
    output_dir=output_dir,
    packet_result=packet_result,
    run_rows=run_rows,
)

print(
    json.dumps(
        {
            "capture_backend": resolved_capture.backend,
            "capture_interface": resolved_capture.interface,
            "pcap_path": str(pcap_path.resolve()),
            "data_dir": str(data_dir.resolve()),
            "report_dir": str(report_dir.resolve()),
            "packets": len(packet_result.packets),
            "transactions": len(packet_result.transactions),
        },
        separators=(",", ":"),
        sort_keys=True,
    )
)
PY

for required_path in \
  "$OUTPUT_DIR/data/packets.jsonl" \
  "$OUTPUT_DIR/data/transactions.jsonl" \
  "$OUTPUT_DIR/data/raw_packets.jsonl" \
  "$OUTPUT_DIR/data/raw_transactions.jsonl" \
  "$OUTPUT_DIR/reports/provenance_summary.json" \
  "$OUTPUT_DIR/reports/actual_run_observability.json"; do
  if [[ ! -s "$required_path" ]]; then
    echo "Provenance smoke test failed: missing artifact $required_path" >&2
    exit 1
  fi
done

echo "Provenance smoke test passed"
echo "  runtime root: $RUNTIME_ROOT"
echo "  schedule: $SCHEDULE_PATH"
echo "  run log: $RUN_LOG_PATH"
echo "  reconstructed output: $OUTPUT_DIR"
