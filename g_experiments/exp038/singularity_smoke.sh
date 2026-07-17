#!/bin/bash
#SBATCH --partition=shared-a100-v2
#SBATCH --account=project143
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=20
#SBATCH --output=slurm-exp038-smoke-%j.out
#SBATCH --error=slurm-exp038-smoke-%j.err

# exp038 pre-flight smoke test (CPU-side checks inside the GPU container).
# Usage: sbatch singularity_smoke.sh

set -euxo pipefail

if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
fi

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/smoke_test.py" ]; then
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

singularity exec --nv --home "$PROJECT_DIR" --bind "$PROJECT_DIR:$PROJECT_DIR" "$CONTAINER_PATH" \
  bash -c "cd '$SCRIPT_DIR' && python3 -m py_compile dataset.py model.py losses.py train.py inference.py analyze_oof.py make_submission.py && python3 smoke_test.py"
