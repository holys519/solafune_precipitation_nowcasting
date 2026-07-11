#!/bin/bash
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
PYTHON="${PYTHON:-python3}"
TARGET="${1:-both}"
OUT="$ROOT/outputs/analysis/exp021"
mkdir -p "$OUT"
exec > >(tee "$OUT/run.log") 2>&1
for exp in exp016 exp017; do
  if [ "$TARGET" = both ] || [ "$TARGET" = "$exp" ]; then
    CONFIG="$OUT/${exp}_config.yaml"
    "$PYTHON" "$SCRIPT_DIR/prepare_config.py" "$exp" "$CONFIG"
    bash "$ROOT/g_experiments/$exp/run.sh" "$CONFIG" fold4_submit
  fi
done
"$PYTHON" "$SCRIPT_DIR/summarize.py"
