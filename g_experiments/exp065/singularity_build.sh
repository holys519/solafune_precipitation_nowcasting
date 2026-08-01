#!/bin/bash
#SBATCH --partition=shared-a100-v2
#SBATCH --account=project143
#SBATCH --job-name=exp065-build
#SBATCH --output=slurm-exp065-build-%j.out
#SBATCH --error=slurm-exp065-build-%j.err
#SBATCH --time=00:40:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# g_experiments/exp065 submission build: blends the eval-side TIFF predictions of
# exp056 / exp064_effb3 / exp064_effv2s / exp064_swin_lr2e4 using the weights
# g_eda/exp011/nested_blend.py fit (see build_submission.py's docstring). CPU-only -- this stage
# reads existing TIFFs and averages them, no model inference.
#
# Usage: sbatch singularity_build.sh [extra build_submission.py args]
#   e.g. sbatch singularity_build.sh --dry-run
#        sbatch singularity_build.sh --sources exp056 exp064_effb3 exp064_swin_lr2e4 --name three_way

set -euxo pipefail

if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
fi

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

CONTAINER_FOLDER="${CONTAINER_FOLDER:-/group/project143/common/containers}"
CONTAINER_NAME="${CONTAINER_NAME:-kaggle-gpu-images-python-v163.sif}"
CONTAINER_PATH="$CONTAINER_FOLDER/$CONTAINER_NAME"
[ -r "$CONTAINER_PATH" ] || { echo "Container not readable: $CONTAINER_PATH"; exit 1; }

module load singularity/3.5.3 || true

singularity exec --home "$PROJECT_DIR" --bind "$PROJECT_DIR:$PROJECT_DIR" "$CONTAINER_PATH" \
  python3 "$SCRIPT_DIR/build_submission.py" "$@"
