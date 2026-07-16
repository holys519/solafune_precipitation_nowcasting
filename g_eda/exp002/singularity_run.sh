#!/bin/bash
#SBATCH --partition=shared-a100-v2
#SBATCH --account=project143
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=240
#SBATCH --output=slurm-g-eda-exp002-%j.out
#SBATCH --error=slurm-g-eda-exp002-%j.err

# g_eda/exp002 (E-1 oracle ladder) on Slurm via Singularity.
# Usage:
#   sbatch singularity_run.sh                    # exp016 exp017 exp018
#   sbatch singularity_run.sh exp018             # single experiment

set -euxo pipefail

if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
fi

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/run_oracle_ladder.py" ]; then
  SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
EXPERIMENTS=("$@")
if [ "${#EXPERIMENTS[@]}" -eq 0 ]; then EXPERIMENTS=(exp016 exp017 exp018); fi

CONTAINER_FOLDER="${CONTAINER_FOLDER:-/group/project143/common/containers}"
CONTAINER_NAME="${CONTAINER_NAME:-kaggle-gpu-images-python-v163.sif}"
CONTAINER_PATH="$CONTAINER_FOLDER/$CONTAINER_NAME"
[ -r "$CONTAINER_PATH" ] || { echo "Container not readable: $CONTAINER_PATH"; exit 1; }

module load singularity/3.5.3 || true

# One process per experiment: each experiment dir defines its own dataset/model modules.
for exp in "${EXPERIMENTS[@]}"; do
  singularity exec --nv --home "$PROJECT_DIR" --bind "$PROJECT_DIR:$PROJECT_DIR" "$CONTAINER_PATH" \
    python3 "$SCRIPT_DIR/run_oracle_ladder.py" --exp-dir "$PROJECT_DIR/g_experiments/$exp"
done
