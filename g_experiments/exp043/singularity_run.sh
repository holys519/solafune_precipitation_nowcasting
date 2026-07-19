#!/bin/bash
#SBATCH --partition=shared-a100-v2
#SBATCH --account=project143
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=180
#SBATCH --no-requeue
#SBATCH --output=slurm-exp043-%j.out
#SBATCH --error=slurm-exp043-%j.err
# exp043: CPU-only blend ladder and optional exp014 overlap patch.
# Usage: sbatch singularity_run.sh [run.py options]
set -euo pipefail

source /etc/profile.d/modules.sh 2>/dev/null || true
module load singularity/3.5.3 || true

S="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
R="$(cd "$S/../.." && pwd)"
O="$R/outputs/analysis/exp043"
C="${CONTAINER_FOLDER:-/group/project143/common/containers}/${CONTAINER_NAME:-kaggle-gpu-images-python-v163.sif}"
mkdir -p "$O"

singularity exec --home "$R" --bind "$R:$R" "$C" \
  python3 "$S/run.py" "$@" 2>&1 | tee "$O/run.log"
