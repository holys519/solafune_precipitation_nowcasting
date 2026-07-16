#!/bin/bash
#SBATCH --partition=shared-a100-v2
#SBATCH --account=project143
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=720
#SBATCH --output=slurm-exp037-%j.out
#SBATCH --error=slurm-exp037-%j.err

# exp037: 8-view TTA (flips + rot90/180/270) re-inference for exp016/017/018, then the
# exp036 combo blend (per-satellite weights + blur 1.0 + threshold 0.2) + overlap patch.
# Usage:
#   sbatch singularity_run.sh              # full pipeline
#   sbatch singularity_run.sh --blend-only # skip inference, rebuild blend from existing preds

set -euxo pipefail

if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
fi

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/config_exp018.yaml" ]; then
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

run_py() {
  singularity exec --nv --home "$PROJECT_DIR" --bind "$PROJECT_DIR:$PROJECT_DIR" "$CONTAINER_PATH" \
    python3 "$@"
}

if [ "${1:-}" != "--blend-only" ]; then
  for exp in exp016 exp017 exp018; do
    EXP_DIR="$PROJECT_DIR/g_experiments/$exp"
    args=()
    for checkpoint in "$PROJECT_DIR/g_model/$exp"/best_model_fold*.pt; do
      args+=(--checkpoint "$checkpoint")
    done
    run_py "$EXP_DIR/inference.py" --config "$SCRIPT_DIR/config_${exp}.yaml" "${args[@]}"
  done
fi

run_py "$PROJECT_DIR/g_experiments/exp036/run.py" \
  --scheme per_satellite --smooth 0.25,0.3,0.45 --blur-sigma 1.0 --value-threshold 0.2 \
  --sources-root "$PROJECT_DIR/outputs/submissions/exp037" \
  --out-prefix exp037 --zip-raw
