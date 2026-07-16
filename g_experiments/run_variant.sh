#!/bin/bash
# Shared exp017-compatible runner for isolated exp028-exp032 ablations.
set -euo pipefail

VARIANT_DIR="$(cd "${1:?variant directory required}" && pwd)"
MODE="${2:-0}"
BASE_DIR="$(cd "$(dirname "$0")/exp017" && pwd)"
PROJECT_DIR="$(cd "$BASE_DIR/../.." && pwd)"
PYTHON="${PYTHON:-$PROJECT_DIR/.venv/bin/python}"
CONFIG="$VARIANT_DIR/config.yaml"

run_fold() { "$PYTHON" "$BASE_DIR/train.py" --config "$CONFIG" --fold "$1"; }

run_inference() {
  args=()
  while IFS= read -r checkpoint; do args+=(--checkpoint "$checkpoint"); done < <(
    find "$PROJECT_DIR/g_model/$(basename "$VARIANT_DIR")" -name 'best_model_fold*.pt' | sort
  )
  [ "${#args[@]}" -gt 0 ] || { echo "No checkpoints found"; exit 1; }
  "$PYTHON" "$BASE_DIR/inference.py" --config "$CONFIG" "${args[@]}"
}

if [ "$MODE" = "all_submit" ]; then
  for fold in 0 1 2 3 4; do run_fold "$fold"; done
  run_inference
  "$PYTHON" "$BASE_DIR/make_submission.py" --config "$CONFIG"
elif [ "$MODE" = "all" ]; then
  for fold in 0 1 2 3 4; do run_fold "$fold"; done
elif [[ "$MODE" =~ ^analyze_fold([0-4])$ ]]; then
  fold="${BASH_REMATCH[1]}"
  checkpoint="$PROJECT_DIR/g_model/$(basename "$VARIANT_DIR")/best_model_fold${fold}.pt"
  [ -f "$checkpoint" ] || { echo "Checkpoint not found: $checkpoint"; exit 1; }
  "$PYTHON" "$BASE_DIR/analyze_oof.py" --config "$CONFIG" --checkpoint "$checkpoint"
elif [ "$MODE" = "analyze" ]; then
  "$PYTHON" "$BASE_DIR/analyze_oof.py" --config "$CONFIG"
elif [ "$MODE" = "infer" ]; then
  run_inference
elif [ "$MODE" = "submit" ]; then
  "$PYTHON" "$BASE_DIR/make_submission.py" --config "$CONFIG"
else
  run_fold "$MODE"
fi
