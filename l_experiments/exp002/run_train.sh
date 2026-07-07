#!/bin/bash
# exp002 local dev: compute normalization stats (if missing) then train one fold.
#
# Usage:
#   bash run_train.sh [config.yaml] [fold]
#
# Examples:
#   bash run_train.sh                    # config.yaml, fold 0
#   bash run_train.sh config.yaml 2      # config.yaml, fold 2

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

PYTHON="../../.venv/bin/python"
CONFIG="${1:-config.yaml}"
FOLD="${2:-0}"

if [ ! -f norm_stats.json ]; then
  echo "== computing normalization stats =="
  "$PYTHON" normalize_stats.py --config "$CONFIG"
fi

echo "== training fold $FOLD (config: $CONFIG) =="
"$PYTHON" train.py --config "$CONFIG" --fold "$FOLD"
