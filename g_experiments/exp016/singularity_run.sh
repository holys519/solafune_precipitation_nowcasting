#!/bin/bash
#SBATCH --partition=shared-a100-v2
#SBATCH --account=project143
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --time=720
#SBATCH --output=slurm-exp016-%j.out
#SBATCH --error=slurm-exp016-%j.err

# exp016 on a Slurm cluster via Singularity.
# Usage:
#   sbatch singularity_run.sh                      # config.yaml, all_submit
#   sbatch singularity_run.sh config_a100x2.yaml 2 # A100x2 profile, fold 2
#   sbatch --ntasks=4 --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=32 singularity_run.sh config_a100x4.yaml all

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
FOLD="${2:-all_submit}"

if [ "$CONFIG" = "all" ] || [ "$CONFIG" = "all_submit" ] || [ "$CONFIG" = "analyze" ] || \
   [ "$CONFIG" = "analysis" ] || [ "$CONFIG" = "submit" ] || [ "$CONFIG" = "submission" ] || \
   [ "$CONFIG" = "submit_calibrated" ] || [ "$CONFIG" = "infer" ] || [ "$CONFIG" = "inference" ] || \
   [[ "$CONFIG" =~ ^[0-9]+$ ]]; then
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
echo "g_experiments/exp016"
echo "=========================================="
echo "Container: $CONTAINER_PATH"
echo "Project:   $PROJECT_DIR"
echo "Config:    $CONFIG"
echo "Fold:      $FOLD"

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
