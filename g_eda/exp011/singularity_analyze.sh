#!/bin/bash
#SBATCH --partition=shared-a100-v2
#SBATCH --account=project143
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=30
#SBATCH --output=slurm-g-eda-exp011-analyze-%j.out
#SBATCH --error=slurm-g-eda-exp011-analyze-%j.err

# g_eda/exp011 phase 2: compute OOF-optimal blend weights from every source's cache.
# The actual work (pure numpy over cached fp16 npz arrays) does not need a GPU, and this script
# never passes --nv to singularity -- but 2026-07-30 testing found this partition's scheduler
# (GAIA) hard-requires --gpus-per-node >= 1 on every job regardless of workload ("Invalid generic
# resource (gres) specification" otherwise; this script had never actually been run before that
# was discovered, per the absence of any slurm-g-eda-exp011-analyze-*.out log until then). The
# --gpus-per-node=1 below is requested-but-unused for exactly that reason, not a real GPU need.
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
