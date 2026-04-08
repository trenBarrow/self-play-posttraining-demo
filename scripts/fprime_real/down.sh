#!/usr/bin/env bash
set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
COMPOSE_FILE="$ROOT_DIR/orchestration/docker-compose.fprime-real.yml"

docker compose -f "$COMPOSE_FILE" down --remove-orphans

echo "F' real-data stack stopped"
