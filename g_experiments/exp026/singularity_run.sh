#!/bin/bash
#SBATCH --partition=shared-a100-v2
#SBATCH --account=project143
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=30
#SBATCH --output=slurm-exp026-%j.out
#SBATCH --error=slurm-exp026-%j.err
# exp026: pure post-processing (CPU, seconds) — patch the exp024 equal_016_017 blend
# with the exp014 tile-overlap GPM copy. Reuses ../exp014/apply_overlap.py unchanged.
set -euo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true; module load singularity/3.5.3 || true
S="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"; R="$(cd "$S/../.." && pwd)"; O="$R/outputs/analysis/exp026"; mkdir -p "$O"
C="${CONTAINER_FOLDER:-/group/project143/common/containers}/${CONTAINER_NAME:-kaggle-gpu-images-python-v163.sif}"
singularity exec --home "$R" --bind "$R:$R" "$C" \
  python3 "$R/g_experiments/exp014/apply_overlap.py" --config "$S/config.yaml" 2>&1 | tee "$O/run.log"
