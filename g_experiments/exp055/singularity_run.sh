#!/bin/bash
#SBATCH --partition=shared-a100-v2
#SBATCH --account=project143
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=60
#SBATCH --output=slurm-exp055-%j.out
#SBATCH --error=slurm-exp055-%j.err

# exp055: green-only manifest blend + causal-only smoothing hook, no overlap patch, ever.
# CPU-only by design (no --gpus-per-node, no --nv) -- blending/zipping already-generated eval
# .tif predictions needs no GPU. This follows g_eda/exp010's genuinely-CPU-only convention
# rather than exp036's (which requests a GPU it never actually uses via --nv).
#
# Usage: sbatch singularity_run.sh [build_submission.py options]

set -euo pipefail

if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
fi
module load singularity/3.5.3 || true

S="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
R="$(cd "$S/../.." && pwd)"
O="$R/outputs/analysis/exp055"
C="${CONTAINER_FOLDER:-/group/project143/common/containers}/${CONTAINER_NAME:-kaggle-gpu-images-python-v163.sif}"
mkdir -p "$O"

singularity exec --home "$R" --bind "$R:$R" "$C" \
  python3 "$S/build_submission.py" "$@" 2>&1 | tee "$O/run.log"
