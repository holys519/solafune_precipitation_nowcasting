#!/bin/bash
#SBATCH --partition=shared-a100-v2
#SBATCH --account=project143
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --time=180
#SBATCH --no-requeue
#SBATCH --output=slurm-exp034-%j.out
#SBATCH --error=slurm-exp034-%j.err
# exp034: thresholded inference for exp016/017/018 -> blend ladder -> optional overlap patch.
# Usage: sbatch singularity_run.sh [run.py options]
set -euo pipefail

source /etc/profile.d/modules.sh 2>/dev/null || true
module load singularity/3.5.3 || true

S="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
R="$(cd "$S/../.." && pwd)"
O="$R/outputs/analysis/exp034"
C="${CONTAINER_FOLDER:-/group/project143/common/containers}/${CONTAINER_NAME:-kaggle-gpu-images-python-v163.sif}"
mkdir -p "$O"

singularity exec --nv --home "$R" --bind "$R:$R" "$C" \
  python3 "$S/run.py" "$@" 2>&1 | tee "$O/run.log"
