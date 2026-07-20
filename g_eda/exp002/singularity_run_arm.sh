#!/bin/bash
#SBATCH --partition=shared-a100-v2
#SBATCH --account=project143
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=240
#SBATCH --output=slurm-g-eda-exp002-arm-%j.out
#SBATCH --error=slurm-g-eda-exp002-arm-%j.err

# Oracle ladder for a config-arm variant that shares another experiment's dataset.py/
# model.py but has its own g_model/ checkpoints (e.g. exp038_features, which reuses
# exp038's code with config_features.yaml).
# Usage: sbatch singularity_run_arm.sh <module_exp_dir_name> <checkpoint_dir_name>

set -euxo pipefail

if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
fi

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
MODULE_EXP="${1:?module exp dir name required, e.g. exp038}"
CHECKPOINT_EXP="${2:?checkpoint dir name required, e.g. exp038_features}"

CONTAINER_FOLDER="${CONTAINER_FOLDER:-/group/project143/common/containers}"
CONTAINER_NAME="${CONTAINER_NAME:-kaggle-gpu-images-python-v163.sif}"
CONTAINER_PATH="$CONTAINER_FOLDER/$CONTAINER_NAME"
[ -r "$CONTAINER_PATH" ] || { echo "Container not readable: $CONTAINER_PATH"; exit 1; }

module load singularity/3.5.3 || true

singularity exec --nv --home "$PROJECT_DIR" --bind "$PROJECT_DIR:$PROJECT_DIR" "$CONTAINER_PATH" \
  python3 "$SCRIPT_DIR/run_oracle_ladder.py" \
    --exp-dir "$PROJECT_DIR/g_experiments/$MODULE_EXP" \
    --checkpoint-dir "$CHECKPOINT_EXP"
