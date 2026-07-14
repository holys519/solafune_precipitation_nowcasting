#!/bin/bash
#SBATCH --partition=shared-a100-v2
#SBATCH --account=project143
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --time=720
set -euo pipefail
VARIANT_DIR="$(cd "${1:?variant directory required}" && pwd)"
MODE="${2:-0}"
PROJECT_DIR="$(cd "$VARIANT_DIR/../.." && pwd)"
CONTAINER_PATH="${CONTAINER_FOLDER:-/group/project143/common/containers}/${CONTAINER_NAME:-kaggle-gpu-images-python-v163.sif}"
[ -r "$CONTAINER_PATH" ] || { echo "Container not readable: $CONTAINER_PATH"; exit 1; }
if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh; fi
module load singularity/3.5.3 || true
singularity exec --nv --home "$PROJECT_DIR" --bind "$PROJECT_DIR:$PROJECT_DIR" "$CONTAINER_PATH" \
  env PYTHON=python3 bash "$PROJECT_DIR/g_experiments/run_variant.sh" "$VARIANT_DIR" "$MODE"
