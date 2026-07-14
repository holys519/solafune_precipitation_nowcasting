#!/bin/bash
set -euo pipefail
bash "$(cd "$(dirname "$0")/.." && pwd)/run_variant.sh" "$(cd "$(dirname "$0")" && pwd)" "${1:-0}"
