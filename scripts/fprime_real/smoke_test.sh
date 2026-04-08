#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
SMOKE_DIR="${FPRIME_SMOKE_DIR:-$ROOT_DIR/artifacts/real_smoke}"
SMOKE_PCAP_DIR="$SMOKE_DIR/fprime_real/pcap"
LEGACY_SMOKE_PCAP="$SMOKE_PCAP_DIR/traffic.pcap"
RUN_LOG="$SMOKE_DIR/fprime_real/logs/all_runs.csv"
SEND_LOG="$SMOKE_DIR/fprime_real/logs/send_log.jsonl"
PACKETS_PATH="$SMOKE_DIR/data/packets.jsonl"
DATASET_PATH="$SMOKE_DIR/data/dataset.jsonl"
SUPPORT_MATRIX="$SMOKE_DIR/reports/support_matrix.json"
ACTUAL_RUN_REPORT="$SMOKE_DIR/reports/actual_run_observability.json"
GENERATION_SUMMARY="$SMOKE_DIR/reports/generation_summary.json"
SCHEMA_REPORT="$SMOKE_DIR/reports/schema.json"
CHANNEL_INVENTORY="$SMOKE_DIR/reports/channel_inventory.json"
PROVENANCE_REPORT="$SMOKE_DIR/reports/provenance_summary.json"

cleanup() {
  "$ROOT_DIR/scripts/fprime_real/down.sh" >/dev/null 2>&1 || true
}
trap cleanup EXIT

"$PYTHON_BIN" "$ROOT_DIR/tools/fprime_real/generate_dataset.py" \
  --rows 24 \
  --output-dir "$SMOKE_DIR" \
  --time-scale 7200

if [[ ! -s "$LEGACY_SMOKE_PCAP" ]]; then
  SMOKE_PCAPS=()
  while IFS= read -r pcap_path; do
    SMOKE_PCAPS+=("$pcap_path")
  done < <(find "$SMOKE_PCAP_DIR" -maxdepth 1 -type f -name 'run_*.pcap' -size +0c | sort)
else
  SMOKE_PCAPS=("$LEGACY_SMOKE_PCAP")
fi

if [[ "${#SMOKE_PCAPS[@]}" -lt 1 ]]; then
  echo "Smoke test failed: no non-empty pcaps were found under $SMOKE_PCAP_DIR" >&2
  exit 1
fi

PACKET_COUNT=0
for pcap_path in "${SMOKE_PCAPS[@]}"; do
  pcap_packets="$(tcpdump -n -r "$pcap_path" 2>/dev/null | wc -l | tr -d ' ')"
  PACKET_COUNT=$((PACKET_COUNT + pcap_packets))
done
if [[ "$PACKET_COUNT" -lt 1 ]]; then
  echo "Smoke test failed: pcap set has zero decodable packets under $SMOKE_PCAP_DIR" >&2
  exit 1
fi

if [[ ! -s "$RUN_LOG" ]]; then
  echo "Smoke test failed: merged run log missing at $RUN_LOG" >&2
  exit 1
fi

if [[ ! -s "$SUPPORT_MATRIX" ]]; then
  echo "Smoke test failed: support matrix missing at $SUPPORT_MATRIX" >&2
  exit 1
fi

for required_report in "$ACTUAL_RUN_REPORT" "$GENERATION_SUMMARY" "$SCHEMA_REPORT" "$CHANNEL_INVENTORY" "$PROVENANCE_REPORT"; do
  if [[ ! -s "$required_report" ]]; then
    echo "Smoke test failed: required report missing at $required_report" >&2
    exit 1
  fi
done

if [[ ! -s "$SEND_LOG" ]]; then
  echo "Smoke test failed: sidecar send log missing at $SEND_LOG" >&2
  exit 1
fi

SUCCESSFUL_ROWS="$("$PYTHON_BIN" - <<'PY' "$RUN_LOG"
import csv
import sys

count = 0
with open(sys.argv[1], newline="") as handle:
    for row in csv.DictReader(handle):
        try:
            if int(row.get("gds_accept", "0")) == 1 and int(row.get("sat_success", "0")) == 1 and int(row.get("timeout", "0")) == 0:
                count += 1
        except ValueError:
            pass
print(count)
PY
)"
if [[ "$SUCCESSFUL_ROWS" -lt 21 ]]; then
  echo "Smoke test failed: expected at least 21 completed command rows in $RUN_LOG, found $SUCCESSFUL_ROWS" >&2
  exit 1
fi

TELEMETRY_PACKET_COUNT="$("$PYTHON_BIN" - <<'PY' "$PACKETS_PATH"
import json
import sys

count = 0
with open(sys.argv[1], encoding="utf-8") as handle:
    for line in handle:
        packet = json.loads(line)
        if packet.get("packet_kind") == "telemetry":
            count += 1
print(count)
PY
)"
if [[ "$TELEMETRY_PACKET_COUNT" -lt 1 ]]; then
  echo "Smoke test failed: expected telemetry packets in $PACKETS_PATH" >&2
  exit 1
fi

"$PYTHON_BIN" - <<'PY' "$ACTUAL_RUN_REPORT" "$GENERATION_SUMMARY" "$PROVENANCE_REPORT" "$PACKETS_PATH"
import json
import sys

actual = json.load(open(sys.argv[1], encoding="utf-8"))
generation = json.load(open(sys.argv[2], encoding="utf-8"))
provenance = json.load(open(sys.argv[3], encoding="utf-8"))
packets_path = sys.argv[4]

actual_summary = actual.get("summary", {})
if int(actual_summary.get("observability_failed_rows", 1)) != 0:
    raise SystemExit("Smoke test failed: benign actual-run observability contains failures")
if int(actual_summary.get("benign_rows", 0)) != 13:
    raise SystemExit("Smoke test failed: benign row count changed unexpectedly")
if int(actual_summary.get("clean_success_rows", 0)) < 11:
    raise SystemExit("Smoke test failed: clean benign successes are below expectation")

if int(generation.get("rows", 0)) != 24:
    raise SystemExit("Smoke test failed: generation summary row count changed unexpectedly")
run_status = generation.get("run_status", {})
if int(run_status.get("success", 0)) < 21:
    raise SystemExit("Smoke test failed: generation summary success count is below expectation")

if int(provenance.get("packets_with_send_id", 0)) < 1:
    raise SystemExit("Smoke test failed: packets are missing send_id coverage")

if not str(provenance.get("capture_backend", "")):
    raise SystemExit("Smoke test failed: provenance summary is missing capture_backend")

capture_interface = str(provenance.get("capture_interface", ""))
if not capture_interface or capture_interface.lower().startswith("lo"):
    raise SystemExit("Smoke test failed: capture interface did not preserve sender identity")

if provenance.get("pcap_identity_mode") != "bridge_ip_5tuple":
    raise SystemExit("Smoke test failed: provenance summary is missing strict identity mode")

if int(provenance.get("serialization_violations", 1)) != 0:
    raise SystemExit("Smoke test failed: serialization invariant violations were reported")

request_anchors = provenance.get("request_anchor_sources", {})
if int(request_anchors.get("pcap", 0)) < 1:
    raise SystemExit("Smoke test failed: request anchors did not record pcap coverage")

with open(packets_path, encoding="utf-8") as handle:
    observed_requests = [
        json.loads(line)
        for line in handle
        if line.strip()
        and json.loads(line).get("packet_kind") == "request"
        and int(json.loads(line).get("observed_on_wire", 0)) == 1
    ]

if not observed_requests or not any(packet.get("src_ip") and packet.get("dst_port") and packet.get("target_stream_id") for packet in observed_requests):
    raise SystemExit("Smoke test failed: observed request packets are missing 5-tuple provenance")
PY

if [[ ! -s "$DATASET_PATH" ]]; then
  echo "Smoke test failed: dataset missing at $DATASET_PATH" >&2
  exit 1
fi

DATASET_ROWS="$(wc -l < "$DATASET_PATH" | tr -d ' ')"
if [[ "$DATASET_ROWS" -lt 1 ]]; then
  echo "Smoke test failed: dataset has no rows" >&2
  exit 1
fi

echo "Smoke test passed"
echo "  packets captured: $PACKET_COUNT"
echo "  pcap files: ${#SMOKE_PCAPS[@]}"
echo "  completed command rows: $SUCCESSFUL_ROWS"
echo "  telemetry packets: $TELEMETRY_PACKET_COUNT"
echo "  dataset rows: $DATASET_ROWS"
echo "  send log: $SEND_LOG"
echo "  sample pcap: ${SMOKE_PCAPS[0]}"
echo "  dataset: $DATASET_PATH"
