#!/bin/bash
# exp048 end-to-end entrypoint (no Slurm; run directly on a GPU box).
#
# Usage:
#   bash run.sh [config] [fold]
#   bash run.sh                          # config.yaml, train folds 0-4 -> analyze -> submit
#   bash run.sh config.yaml all          # train folds 0-4 only
#   bash run.sh config.yaml submit       # analyze existing checkpoints -> submit
#   bash run.sh config.yaml fold4_submit # train fold 4 -> analyze -> infer -> submit
#   bash run.sh config_a100x2.yaml 3     # A100x2 profile, fold 3

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"

# Override with e.g. PYTHON=python3 when running inside a container that has torch installed.
PYTHON="${PYTHON:-../../.venv/bin/python}"
CONFIG="${1:-config.yaml}"
FOLD="${2:-all_submit}"
GPU_LOG_INTERVAL_SECONDS="${GPU_LOG_INTERVAL_SECONDS:-300}"
GPU_LOG_DIR="${GPU_LOG_DIR:-$SCRIPT_DIR/logs}"
USE_OOF_CALIBRATION="${USE_OOF_CALIBRATION:-0}"
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

config_path() {
  # Arm configs use distinct model/analysis dirs to avoid checkpoint collisions; read them
  # from the active config instead of hardcoding exp048's default paths.
  "$PYTHON" - "$CONFIG" "$1" <<'PY'
import sys, yaml
from pathlib import Path
config = yaml.safe_load(open(sys.argv[1]))
value = Path(config["paths"][sys.argv[2]])
print(value if value.is_absolute() else (Path(sys.argv[1]).resolve().parent / value).resolve())
PY
}

run_inference() {
  local model_dir
  model_dir="$(config_path model_dir)"
  local calibration_path
  calibration_path="$(config_path analysis_dir)/oof_calibration.json"
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
  if [ "$USE_OOF_CALIBRATION" = "1" ]; then
    checkpoint_args+=(--use-calibration --calibration "$calibration_path")
  fi

  echo "== inference with ${#checkpoints[@]} checkpoint(s) =="
  printf '  %s\n' "${checkpoints[@]}"
  "$PYTHON" inference.py --config "$CONFIG" "${checkpoint_args[@]}"
}

run_analysis() {
  echo "== creating OOF diagnostics =="
  "$PYTHON" analyze_oof.py --config "$CONFIG"
}

run_submission() {
  echo "== creating submission zip =="
  "$PYTHON" make_submission.py --config "$CONFIG"
}

if [ "$FOLD" = "all" ]; then
  for f in 0 1 2 3 4; do
    run_fold "$f"
  done
elif [ "$FOLD" = "all_submit" ]; then
  for f in 0 1 2 3 4; do
    run_fold "$f"
  done
  run_analysis
  run_inference
  run_submission
elif [ "$FOLD" = "fold4_submit" ]; then
  run_fold 4
  run_analysis
  run_inference
  run_submission
elif [ "$FOLD" = "analyze" ] || [ "$FOLD" = "analysis" ]; then
  run_analysis
elif [ "$FOLD" = "infer" ] || [ "$FOLD" = "inference" ]; then
  run_inference
elif [ "$FOLD" = "submit" ] || [ "$FOLD" = "submission" ]; then
  run_analysis
  run_inference
  run_submission
elif [ "$FOLD" = "submit_calibrated" ]; then
  run_analysis
  USE_OOF_CALIBRATION=1
  run_inference
  run_submission
else
  run_fold "$FOLD"
fi
