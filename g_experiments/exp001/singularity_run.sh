#!/bin/bash

#SBATCH --partition=shared-a100-v2
#SBATCH --account=project143
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --time=720
#SBATCH --output=slurm-exp001-%j.out
#SBATCH --error=slurm-exp001-%j.err

set -euxo pipefail

if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
fi

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/python_run.sh" ]; then
  SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
PROJECT_FOLDER="$(cd "$SCRIPT_DIR/../.." && pwd)"
STAGE="${1:-${EXP001_STAGE:-train}}"

CONTAINER_FOLDER="${CONTAINER_FOLDER:-/group/project143/common/containers}"
CONTAINER_NAME="${CONTAINER_NAME:-kaggle-gpu-images-python-v163.sif}"
CONTAINER_PATH="$CONTAINER_FOLDER/$CONTAINER_NAME"

if [ ! -r "$CONTAINER_PATH" ]; then
  echo "ERROR: Container not readable: $CONTAINER_PATH"
  exit 1
fi

module load singularity/3.5.3 || true

echo "=========================================="
echo "g_experiments/exp001: Compact UNet baseline"
echo "=========================================="
echo "Container: $CONTAINER_PATH"
echo "Project:   $PROJECT_FOLDER"
echo "Stage:     $STAGE"

SINGULARITY_ARGS=(
  --nv
  --home "$PROJECT_FOLDER"
  --bind "$PROJECT_FOLDER:$PROJECT_FOLDER"
)

if [ -n "${SLURM_JOB_ID:-}" ] && [ -d "/local/job/${SLURM_JOB_ID}" ]; then
  SINGULARITY_ARGS+=(--bind "/local/job/${SLURM_JOB_ID}:/local/job/${SLURM_JOB_ID}")
fi

singularity exec \
  "${SINGULARITY_ARGS[@]}" \
  "$CONTAINER_PATH" \
  /bin/bash "$SCRIPT_DIR/python_run.sh" "$STAGE"

echo "=========================================="
echo "Job complete!"
echo "=========================================="
