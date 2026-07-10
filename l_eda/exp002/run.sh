#!/bin/bash
# l_eda/exp002 entrypoint. Runs the pixel-level EDA suite with uv's venv.
set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PYTHON="${PYTHON:-$SCRIPT_DIR/../../.venv/bin/python}"
"$PYTHON" "$SCRIPT_DIR/run_pixel_eda.py" --config "${1:-$SCRIPT_DIR/config.yaml}"
