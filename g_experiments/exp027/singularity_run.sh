#!/bin/bash
#SBATCH --partition=shared-a100-v2
#SBATCH --account=project143
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=360
#SBATCH --output=slurm-exp027-%j.out
#SBATCH --error=slurm-exp027-%j.err
# exp027: exp025 seed inference (GPU) -> exp016+exp017+seeds blend -> exp014 overlap patch -> zips.
# Usage: sbatch singularity_run.sh [seeds]      (default 42,123,2026)
set -euo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true; module load singularity/3.5.3 || true
S="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"; R="$(cd "$S/../.." && pwd)"; O="$R/outputs/analysis/exp027"; mkdir -p "$O"
C="${CONTAINER_FOLDER:-/group/project143/common/containers}/${CONTAINER_NAME:-kaggle-gpu-images-python-v163.sif}"
singularity exec --nv --home "$R" --bind "$R:$R" "$C" \
  python3 "$S/run.py" --seeds "${1:-42,123,2026}" 2>&1 | tee "$O/run.log"
