#!/bin/bash
#SBATCH --partition=GPU_PARTITION
#SBATCH --account=PROJECT_ACCOUNT
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --time=720
#SBATCH --output=exp002_%j.log

# exp002 on a Slurm cluster via Singularity.
# Usage:
#   sbatch singularity_run.sh                      # config.yaml, fold 0
#   sbatch singularity_run.sh config_a100x2.yaml 2 # A100x2 profile, fold 2
#   sbatch --gpus-per-node=4 --cpus-per-task=32 singularity_run.sh config_a100x4.yaml all

source /etc/profile.d/modules.sh
module load singularity/3.5.3 || true

CONTAINER="/path/to/container.sif"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="${1:-config.yaml}"
FOLD="${2:-0}"

singularity exec \
  --nv \
  --bind "$PROJECT_DIR:$PROJECT_DIR" \
  --env PYTHON=python3 \
  "$CONTAINER" \
  bash "$SCRIPT_DIR/run.sh" "$CONFIG" "$FOLD"
