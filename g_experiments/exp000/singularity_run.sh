#!/bin/bash
#SBATCH --partition=shared-a100-v2
#SBATCH --account=project143
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=120
#SBATCH --output=slurm-exp000-%j.out
#SBATCH --error=slurm-exp000-%j.err

# exp000: HPCクラスタ上でデータ準備
# sbatch singularity_run.sh で実行

set -euxo pipefail

if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
fi

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/download.sh" ]; then
  SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
PROJECT_FOLDER="$(cd "$SCRIPT_DIR/../.." && pwd)"

CONTAINER_FOLDER="${CONTAINER_FOLDER:-/group/project143/common/containers}"
CONTAINER_NAME="${CONTAINER_NAME:-kaggle-gpu-images-python-v163.sif}"
CONTAINER_PATH="$CONTAINER_FOLDER/$CONTAINER_NAME"

if [ ! -r "$CONTAINER_PATH" ]; then
  echo "ERROR: Container not readable: $CONTAINER_PATH"
  exit 1
fi

module load singularity/3.5.3 || true

SINGULARITY_ARGS=(
  --home "$PROJECT_FOLDER"
  --bind "$PROJECT_FOLDER:$PROJECT_FOLDER"
)

if [ -n "${SLURM_JOB_ID:-}" ] && [ -d "/local/job/${SLURM_JOB_ID}" ]; then
  SINGULARITY_ARGS+=(--bind "/local/job/${SLURM_JOB_ID}:/local/job/${SLURM_JOB_ID}")
fi

singularity exec \
  "${SINGULARITY_ARGS[@]}" \
  "$CONTAINER_PATH" \
  bash "$SCRIPT_DIR/download.sh"
