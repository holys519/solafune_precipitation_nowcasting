#!/bin/bash
# l_eda/exp001 entrypoint.
#
# Usage:
#   bash run.sh [config]

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$PROJECT_DIR"

CONFIG="${1:-config.yaml}"
if [[ "$CONFIG" != /* ]]; then
  CONFIG="$SCRIPT_DIR/$CONFIG"
fi

if [ -x "$PROJECT_DIR/l_experiments/exp000/download.sh" ]; then
  echo "== preparing data with l_experiments/exp000 =="
  bash "$PROJECT_DIR/l_experiments/exp000/download.sh"
fi

echo "== running image-processing EDA with uv =="
uv run python "$SCRIPT_DIR/run_image_eda.py" --config "$CONFIG"
