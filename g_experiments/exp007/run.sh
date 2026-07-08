#!/bin/bash
# exp007 end-to-end entrypoint (no Slurm; run directly on a GPU box).
#
# Usage:
#   bash run.sh [config] [fold]
#   bash run.sh                          # config.yaml, analyze ensemble sources -> submit
#   bash run.sh config.yaml submit       # analyze existing checkpoints -> submit

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
  echo "ERROR: exp007 is an ensemble/postprocess experiment and does not train fold=$fold."
  echo "Train exp003-exp006 first, then run exp007 submit."
  exit 1
}

run_inference() {
  local calibration_path="$PROJECT_DIR/outputs/analysis/exp007/oof_calibration.json"
  local checkpoint_args=()
  if [ "$USE_OOF_CALIBRATION" = "1" ]; then
    checkpoint_args+=(--use-calibration --calibration "$calibration_path")
  fi

  echo "== ensemble inference from configured sources =="
  "$PYTHON" inference.py --config "$CONFIG" "${checkpoint_args[@]}"
}

run_analysis() {
  echo "== summarizing ensemble sources =="
  "$PYTHON" analyze_ensemble.py --config "$CONFIG"
}

run_submission() {
  echo "== creating submission zip =="
  "$PYTHON" make_submission.py
}

if [ "$FOLD" = "all" ]; then
  run_analysis
elif [ "$FOLD" = "all_submit" ]; then
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
