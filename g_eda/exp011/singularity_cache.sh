#!/bin/bash
#SBATCH --partition=shared-a100-v2
#SBATCH --account=project143
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=60
#SBATCH --output=slurm-g-eda-exp011-cache-%j.out
#SBATCH --error=slurm-g-eda-exp011-cache-%j.err

# g_eda/exp011 phase 1: cache one manifest source's OOF predictions (inference from an existing
# checkpoint -- NOT a training job). Mirrors g_eda/exp003/singularity_cache_exp038.sh's precedent
# but reads module_dir/checkpoint_dir out of sources.json instead of taking them as argv.
#
# Usage: sbatch singularity_cache.sh <source_name>
#   e.g. sbatch singularity_cache.sh exp038_sigmafixed
#        sbatch singularity_cache.sh exp040_metric

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

SOURCE_NAME="${1:?usage: sbatch singularity_cache.sh <source_name from sources.json>}"

singularity exec --nv --home "$PROJECT_DIR" --bind "$PROJECT_DIR:$PROJECT_DIR" "$CONTAINER_PATH" \
  python3 "$SCRIPT_DIR/optimize_blend.py" --cache "$SOURCE_NAME"
