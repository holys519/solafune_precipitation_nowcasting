#!/bin/bash
# exp002 full-scale training entrypoint (no Slurm; run directly on a GPU box).
#
# Usage:
#   bash run.sh [config] [fold]
#   bash run.sh                          # config.yaml (3090x2 profile), fold 0
#   bash run.sh config.yaml all          # folds 0-4 sequentially (ticket G-002)
#   bash run.sh config_a100x2.yaml 3     # A100x2 profile, fold 3
#
# After all folds finish, ensemble at inference:
#   PY=../../.venv/bin/python
#   $PY inference.py --config config.yaml \
#     --checkpoint ../../g_model/exp002/best_model_fold0.pt \
#     --checkpoint ../../g_model/exp002/best_model_fold1.pt \
#     --checkpoint ../../g_model/exp002/best_model_fold2.pt \
#     --checkpoint ../../g_model/exp002/best_model_fold3.pt \
#     --checkpoint ../../g_model/exp002/best_model_fold4.pt
#   $PY make_submission.py

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
cd "$SCRIPT_DIR"

# Override with e.g. PYTHON=python3 when running inside a container that has torch installed.
PYTHON="${PYTHON:-../../.venv/bin/python}"
CONFIG="${1:-config.yaml}"
FOLD="${2:-0}"

if [ ! -f norm_stats.json ]; then
  echo "== computing normalization stats =="
  "$PYTHON" normalize_stats.py --config "$CONFIG"
fi

if [ "$FOLD" = "all" ]; then
  for f in 0 1 2 3 4; do
    echo "== training fold $f (config: $CONFIG) =="
    "$PYTHON" train.py --config "$CONFIG" --fold "$f"
  done
else
  echo "== training fold $FOLD (config: $CONFIG) =="
  "$PYTHON" train.py --config "$CONFIG" --fold "$FOLD"
fi
