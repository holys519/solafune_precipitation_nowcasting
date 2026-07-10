#!/bin/bash
# exp015 end-to-end entrypoint (no Slurm; run directly on a GPU box).
#
# This experiment trains nothing: it reuses exp009's checkpoints (g_model/exp009, read-only via
# SOURCE_MODEL_DIR / paths.source_model_dir) and only adds an isotonic OOF calibration curve
# (G-027a) on top. All outputs are written under exp015's own outputs/analysis and
# outputs/submissions directories -- exp009's own outputs are never touched.
#
# Usage:
#   bash run.sh                          # config.yaml, analyze exp009 checkpoints -> exp015 submit (isotonic)
#   bash run.sh config.yaml submit       # analyze -> inference -> zip (uses postprocess.use_oof_calibration as-is)
#   bash run.sh config.yaml submit_calibrated   # analyze -> inference with calibration on -> zip
#   bash run.sh config.yaml analyze      # OOF diagnostics + oof_calibration.json only
#   bash run.sh config_a100x2.yaml submit_calibrated

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
cd "$SCRIPT_DIR"

# Override with e.g. PYTHON=python3 when running inside a container that has torch installed.
PYTHON="${PYTHON:-../../.venv/bin/python}"
CONFIG="${1:-config.yaml}"
FOLD="${2:-submit_calibrated}"
SOURCE_MODEL_DIR="${SOURCE_MODEL_DIR:-$PROJECT_DIR/g_model/exp009}"
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

if [ ! -d "$SOURCE_MODEL_DIR" ]; then
  echo "ERROR: exp009 checkpoints not found under $SOURCE_MODEL_DIR"
  echo "exp015 reuses exp009's trained checkpoints and does not train its own -- run exp009's"
  echo "training stages first, or point SOURCE_MODEL_DIR at wherever they live."
  exit 1
fi

run_inference() {
  local model_dir="$SOURCE_MODEL_DIR"
  local calibration_path="$PROJECT_DIR/outputs/analysis/exp015/oof_calibration.json"
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

  echo "== inference with ${#checkpoints[@]} checkpoint(s) (exp009, read-only) =="
  printf '  %s\n' "${checkpoints[@]}"
  "$PYTHON" inference.py --config "$CONFIG" "${checkpoint_args[@]}"
}

run_analysis() {
  echo "== creating OOF diagnostics + isotonic/linear calibration (G-027a) =="
  "$PYTHON" analyze_oof.py --config "$CONFIG"
}

run_submission() {
  echo "== creating submission zip =="
  # make_submission.py always reads ./config.yaml (no --config flag); matches exp008/exp009's
  # own make_submission.py. Non-default configs (e.g. config_a100x2.yaml) only affect
  # analyze_oof.py/inference.py's batch size here, not the zip step.
  "$PYTHON" make_submission.py
}

case "$FOLD" in
  analyze|analysis)
    run_analysis
    ;;
  infer|inference)
    run_inference
    ;;
  submit|submission)
    run_analysis
    run_inference
    run_submission
    ;;
  submit_calibrated)
    run_analysis
    USE_OOF_CALIBRATION=1
    run_inference
    run_submission
    ;;
  *)
    echo "ERROR: exp015 does not train (it reuses exp009's checkpoints). Valid stages:" \
         "analyze | infer | submit | submit_calibrated. Got: $FOLD"
    exit 1
    ;;
esac
