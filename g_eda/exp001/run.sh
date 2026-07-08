#!/bin/bash
# g_eda/exp001 entrypoint.
#
# Usage:
#   bash run.sh [config]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"

PYTHON="${PYTHON:-python3}"
CONFIG="${1:-config.yaml}"

if [ -x "$PROJECT_DIR/g_experiments/exp000/download.sh" ]; then
  echo "== preparing data with g_experiments/exp000 =="
  bash "$PROJECT_DIR/g_experiments/exp000/download.sh"
fi

echo "== running image-processing EDA =="
"$PYTHON" run_image_eda.py --config "$CONFIG"
