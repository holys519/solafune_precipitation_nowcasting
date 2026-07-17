#!/bin/bash
#SBATCH --partition=shared-a100-v2
#SBATCH --account=project143
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=240
#SBATCH --output=slurm-g-eda-exp006-%j.out
#SBATCH --error=slurm-g-eda-exp006-%j.err

# g_eda/exp006: CPU-bound exact factorization and metric audit.
# The partition mandates one GPU, but this analysis intentionally does not initialize CUDA.
# Usage: sbatch singularity_run.sh [additional run_factorization_audit.py arguments]

set -euo pipefail

if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
fi

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/run_factorization_audit.py" ]; then
  SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

CONTAINER_FOLDER="${CONTAINER_FOLDER:-/group/project143/common/containers}"
CONTAINER_NAME="${CONTAINER_NAME:-kaggle-gpu-images-python-v163.sif}"
CONTAINER_PATH="$CONTAINER_FOLDER/$CONTAINER_NAME"
[ -r "$CONTAINER_PATH" ] || { echo "Container not readable: $CONTAINER_PATH"; exit 1; }

module load singularity/3.5.3 || true

echo "g_eda/exp006"
echo "container=$CONTAINER_PATH"
echo "project=$PROJECT_DIR"
echo "arguments=$*"

singularity exec \
  --home "$PROJECT_DIR" \
  --bind "$PROJECT_DIR:$PROJECT_DIR" \
  "$CONTAINER_PATH" \
  python3 "$SCRIPT_DIR/run_factorization_audit.py" "$@"

