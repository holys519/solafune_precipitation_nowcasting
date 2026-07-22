#!/bin/bash
#SBATCH --partition=shared-a100-v2
#SBATCH --account=project143
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=30
#SBATCH --output=slurm-g-eda-exp011-analyze-%j.out
#SBATCH --error=slurm-g-eda-exp011-analyze-%j.err

# g_eda/exp011 phase 2: compute OOF-optimal blend weights from every source's cache.
# CPU-only by design (no --gpus-per-node, no --nv) -- this stage is pure numpy over the cached
# fp16 npz arrays, so it deliberately does not compete for the scarce GPU allocation the way
# exp036's singularity_run.sh does (that one requests a GPU it does not use).
#
# Usage: sbatch singularity_analyze.sh

set -euxo pipefail

if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
fi

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/optimize_blend.py" ]; then
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

singularity exec --home "$PROJECT_DIR" --bind "$PROJECT_DIR:$PROJECT_DIR" "$CONTAINER_PATH" \
  python3 "$SCRIPT_DIR/optimize_blend.py" --analyze
