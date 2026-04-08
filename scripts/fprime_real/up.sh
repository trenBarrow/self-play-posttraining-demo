#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPOSE_FILE="$ROOT_DIR/orchestration/docker-compose.fprime-real.yml"
RUNTIME_ROOT="$ROOT_DIR/gds/fprime_runtime"

while [[ $# -gt 0 ]]; do
  case "$1" in
    --runtime-root)
      RUNTIME_ROOT="$2"
      shift 2
      ;;
    *)
      echo "Unknown arg: $1" >&2
      exit 2
      ;;
  esac
done

mkdir -p "$RUNTIME_ROOT/cli_logs" "$RUNTIME_ROOT/logs" "$RUNTIME_ROOT/node_a/logs" "$RUNTIME_ROOT/node_b/logs"

"$ROOT_DIR/scripts/fprime_real/bootstrap_project.sh"

FPRIME_RUNTIME_HOST_ROOT="$RUNTIME_ROOT" docker compose -f "$COMPOSE_FILE" up -d --force-recreate

echo "F' nodes are up"
FPRIME_RUNTIME_HOST_ROOT="$RUNTIME_ROOT" docker compose -f "$COMPOSE_FILE" ps

echo "Network ready: fprime_real_net"
docker network inspect fprime_real_net --format 'Subnet: {{(index .IPAM.Config 0).Subnet}}'
