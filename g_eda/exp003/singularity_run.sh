#!/bin/bash
#SBATCH --partition=shared-a100-v2
#SBATCH --account=project143
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=240
#SBATCH --output=slurm-g-eda-exp003-%j.out
#SBATCH --error=slurm-g-eda-exp003-%j.err

# g_eda/exp003: OOF blend-weight optimization.
# Usage:
#   sbatch singularity_run.sh              # cache exp016/017/018 then analyze
#   sbatch singularity_run.sh --analyze    # analysis only (caches must exist)

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

run_in_container() {
  singularity exec --nv --home "$PROJECT_DIR" --bind "$PROJECT_DIR:$PROJECT_DIR" "$CONTAINER_PATH" \
    python3 "$SCRIPT_DIR/run_blend_curve.py" "$@"
}

if [ "${1:-}" = "--analyze" ]; then
  run_in_container --analyze
else
  # One process per experiment: dataset/model module namespaces collide across exp dirs.
  for exp in exp016 exp017 exp018; do
    if [ ! -f "$PROJECT_DIR/outputs/g_eda/exp003/${exp}_oof_pred.npz" ]; then
      run_in_container --cache "$exp"
    fi
  done
  run_in_container --analyze
fi
