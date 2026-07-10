#!/bin/bash
#SBATCH --partition=shared-a100-v2
#SBATCH --account=project143
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --time=180
#SBATCH --output=slurm-exp015-%j.out
#SBATCH --error=slurm-exp015-%j.err

# exp015 (G-027a: isotonic OOF calibration on exp009's checkpoints) on a Slurm cluster via
# Singularity. This experiment trains nothing -- it only runs analyze_oof.py/inference.py against
# exp009's existing checkpoints, so the default time budget is much shorter than a training job.
#
# Usage:
#   sbatch singularity_run.sh                        # config.yaml, submit_calibrated (isotonic)
#   sbatch singularity_run.sh config.yaml analyze     # OOF diagnostics + oof_calibration.json only
#   sbatch singularity_run.sh config.yaml submit      # submit without forcing calibration on
#   sbatch singularity_run.sh config_a100x2.yaml submit_calibrated

set -euxo pipefail

if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
fi

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/run.sh" ]; then
  SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="${1:-config.yaml}"
FOLD="${2:-submit_calibrated}"

if [ "$CONFIG" = "analyze" ] || [ "$CONFIG" = "analysis" ] || [ "$CONFIG" = "infer" ] || \
   [ "$CONFIG" = "inference" ] || [ "$CONFIG" = "submit" ] || [ "$CONFIG" = "submission" ] || \
   [ "$CONFIG" = "submit_calibrated" ]; then
  FOLD="$CONFIG"
  CONFIG="config.yaml"
fi

CONTAINER_FOLDER="${CONTAINER_FOLDER:-/group/project143/common/containers}"
CONTAINER_NAME="${CONTAINER_NAME:-kaggle-gpu-images-python-v163.sif}"
CONTAINER_PATH="$CONTAINER_FOLDER/$CONTAINER_NAME"

if [ ! -r "$CONTAINER_PATH" ]; then
  echo "ERROR: Container not readable: $CONTAINER_PATH"
  exit 1
fi

module load singularity/3.5.3 || true

echo "=========================================="
echo "g_experiments/exp015 (G-027a)"
echo "=========================================="
echo "Container: $CONTAINER_PATH"
echo "Project:   $PROJECT_DIR"
echo "Config:    $CONFIG"
echo "Stage:     $FOLD"

SINGULARITY_ARGS=(
  --nv
  --home "$PROJECT_DIR"
  --bind "$PROJECT_DIR:$PROJECT_DIR"
)

if [ -n "${SLURM_JOB_ID:-}" ] && [ -d "/local/job/${SLURM_JOB_ID}" ]; then
  SINGULARITY_ARGS+=(--bind "/local/job/${SLURM_JOB_ID}:/local/job/${SLURM_JOB_ID}")
fi

singularity exec \
  "${SINGULARITY_ARGS[@]}" \
  "$CONTAINER_PATH" \
  env PYTHON=python3 bash "$SCRIPT_DIR/run.sh" "$CONFIG" "$FOLD"
