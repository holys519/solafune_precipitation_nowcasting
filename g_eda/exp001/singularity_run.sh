#!/bin/bash
#SBATCH --partition=shared-a100-v2
#SBATCH --account=project143
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=180
#SBATCH --output=slurm-g-eda-exp001-%j.out
#SBATCH --error=slurm-g-eda-exp001-%j.err

# g_eda/exp001 on a Slurm cluster via Singularity.
# Usage:
#   sbatch singularity_run.sh
#   sbatch singularity_run.sh config.yaml

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

CONTAINER_FOLDER="${CONTAINER_FOLDER:-/group/project143/common/containers}"
CONTAINER_NAME="${CONTAINER_NAME:-kaggle-gpu-images-python-v163.sif}"
CONTAINER_PATH="$CONTAINER_FOLDER/$CONTAINER_NAME"

if [ ! -r "$CONTAINER_PATH" ]; then
  echo "ERROR: Container not readable: $CONTAINER_PATH"
  exit 1
fi

module load singularity/3.5.3 || true

echo "=========================================="
echo "g_eda/exp001"
echo "=========================================="
echo "Container: $CONTAINER_PATH"
echo "Project:   $PROJECT_DIR"
echo "Config:    $CONFIG"

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
  env PYTHON=python3 bash "$SCRIPT_DIR/run.sh" "$CONFIG"
