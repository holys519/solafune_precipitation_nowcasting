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
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"

# Override with e.g. PYTHON=python3 when running inside a container that has torch installed.
PYTHON="${PYTHON:-../../.venv/bin/python}"
CONFIG="${1:-config.yaml}"
FOLD="${2:-0}"
GPU_LOG_INTERVAL_SECONDS="${GPU_LOG_INTERVAL_SECONDS:-300}"
GPU_LOG_DIR="${GPU_LOG_DIR:-$SCRIPT_DIR/logs}"
GPU_LOGGER_PID=""

start_gpu_logger() {
  if ! command -v nvidia-smi >/dev/null 2>&1; then
    echo "nvidia-smi not found; GPU memory logging disabled."
    return 0
  fi

  mkdir -p "$GPU_LOG_DIR"
  local stamp
  stamp="$(date +%Y%m%d_%H%M%S)"
  local job_id="${SLURM_JOB_ID:-manual}"
  local log_file="$GPU_LOG_DIR/gpu_memory_${job_id}_${stamp}.csv"

  echo "GPU memory log: $log_file (interval=${GPU_LOG_INTERVAL_SECONDS}s)"
  nvidia-smi \
    --query-gpu=timestamp,index,name,memory.used,memory.free,memory.total,utilization.gpu \
    --format=csv,nounits \
    --loop="$GPU_LOG_INTERVAL_SECONDS" \
    > "$log_file" 2>&1 &
  GPU_LOGGER_PID="$!"
}

stop_gpu_logger() {
  if [ -n "$GPU_LOGGER_PID" ] && kill -0 "$GPU_LOGGER_PID" >/dev/null 2>&1; then
    kill "$GPU_LOGGER_PID" >/dev/null 2>&1 || true
    wait "$GPU_LOGGER_PID" >/dev/null 2>&1 || true
  fi
}

trap stop_gpu_logger EXIT

start_gpu_logger

if [ -x "$PROJECT_DIR/g_experiments/exp000/download.sh" ]; then
  echo "== preparing data with g_experiments/exp000 =="
  bash "$PROJECT_DIR/g_experiments/exp000/download.sh"
fi

if [ ! -f norm_stats.json ]; then
  echo "== computing normalization stats =="
  "$PYTHON" normalize_stats.py --config "$CONFIG"
fi

run_fold() {
  local fold="$1"
  echo "== training fold $fold (config: $CONFIG) =="
  "$PYTHON" train.py --config "$CONFIG" --fold "$fold"
}

run_inference() {
  local model_dir="$PROJECT_DIR/g_model/exp002"
  shopt -s nullglob
  local checkpoints=("$model_dir"/best_model_fold*.pt)
  shopt -u nullglob

  if [ "${#checkpoints[@]}" -eq 0 ]; then
    echo "ERROR: no checkpoints found under $model_dir"
    exit 1
  fi

  IFS=$'\n' checkpoints=($(printf '%s\n' "${checkpoints[@]}" | sort))
  unset IFS

  local checkpoint_args=()
  local checkpoint
  for checkpoint in "${checkpoints[@]}"; do
    checkpoint_args+=(--checkpoint "$checkpoint")
  done

  echo "== inference with ${#checkpoints[@]} checkpoint(s) =="
  printf '  %s\n' "${checkpoints[@]}"
  "$PYTHON" inference.py --config "$CONFIG" "${checkpoint_args[@]}"
}

run_submission() {
  echo "== creating submission zip =="
  "$PYTHON" make_submission.py
}

if [ "$FOLD" = "all" ]; then
  for f in 0 1 2 3 4; do
    run_fold "$f"
  done
elif [ "$FOLD" = "all_submit" ]; then
  for f in 0 1 2 3 4; do
    run_fold "$f"
  done
  run_inference
  run_submission
elif [ "$FOLD" = "infer" ] || [ "$FOLD" = "inference" ]; then
  run_inference
elif [ "$FOLD" = "submit" ] || [ "$FOLD" = "submission" ]; then
  run_inference
  run_submission
else
  run_fold "$FOLD"
fi
