#!/bin/bash
#SBATCH --partition=shared-a100-v2
#SBATCH --account=project143
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=60
#SBATCH --output=slurm-g-eda-exp003-cache038-%j.out
#SBATCH --error=slurm-g-eda-exp003-cache038-%j.err

# Cache exp038 (strict green) OOF predictions for the Track G3 green blend.
# Usage: sbatch singularity_cache_exp038.sh [exp_name] [module_dir] [checkpoint_dir]

set -euxo pipefail

if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
fi

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/run_blend_curve.py" ]; then
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

EXP_NAME="${1:-exp038}"
MODULE_DIR="${2:-exp038}"
CHECKPOINT_DIR="${3:-exp038}"

singularity exec --nv --home "$PROJECT_DIR" --bind "$PROJECT_DIR:$PROJECT_DIR" "$CONTAINER_PATH" \
  python3 "$SCRIPT_DIR/run_blend_curve.py" --cache "$EXP_NAME" \
    --module-dir "$MODULE_DIR" --checkpoint-dir "$CHECKPOINT_DIR"
