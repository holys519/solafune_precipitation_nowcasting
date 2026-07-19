#!/bin/bash
#SBATCH --partition=shared-a100-v2
#SBATCH --account=project143
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --time=180
#SBATCH --output=slurm-exp041-%j.out
#SBATCH --error=slurm-exp041-%j.err

# exp041: isolated tile-RMSE fine-tuning screen. Reuses exp038/train.py (and, for the
# smoke mode, exp038/losses.py + exp038/model.py via PYTHONPATH) as the shared engine --
# exp041 itself holds only configs, checkpoints, and screening scripts.
# Usage:
#   sbatch singularity_run.sh config_control.yaml smoke   # smoke_test.py via PYTHONPATH=exp038
#   sbatch singularity_run.sh config_control.yaml 0       # train fold 0 via exp038/train.py
#   sbatch singularity_run.sh config_metric.yaml 4        # train fold 4 via exp038/train.py

set -euo pipefail

if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
fi

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
BASE_DIR="$PROJECT_DIR/g_experiments/exp038"

CONFIG="${1:-config_control.yaml}"
FOLD="${2:-0}"
CONFIG_PATH="$CONFIG"
[[ "$CONFIG" = /* ]] || CONFIG_PATH="$SCRIPT_DIR/$CONFIG"

CONTAINER_FOLDER="${CONTAINER_FOLDER:-/group/project143/common/containers}"
CONTAINER_NAME="${CONTAINER_NAME:-kaggle-gpu-images-python-v163.sif}"
CONTAINER_PATH="$CONTAINER_FOLDER/$CONTAINER_NAME"
[ -r "$CONTAINER_PATH" ] || { echo "Container not readable: $CONTAINER_PATH"; exit 1; }

module load singularity/3.5.3 || true

if [ "$FOLD" = "smoke" ]; then
  singularity exec --nv --home "$PROJECT_DIR" --bind "$PROJECT_DIR:$PROJECT_DIR" "$CONTAINER_PATH" \
    env PYTHONPATH="$BASE_DIR" PYTHON=python3 python3 "$SCRIPT_DIR/smoke_test.py"
else
  singularity exec --nv --home "$PROJECT_DIR" --bind "$PROJECT_DIR:$PROJECT_DIR" "$CONTAINER_PATH" \
    env PYTHON=python3 python3 "$BASE_DIR/train.py" --config "$CONFIG_PATH" --fold "$FOLD"
fi
