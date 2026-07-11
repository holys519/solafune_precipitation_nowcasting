#!/bin/bash
#SBATCH --partition=shared-a100-v2
#SBATCH --account=project143
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=60
#SBATCH --output=slurm-exp024-%j.out
#SBATCH --error=slurm-exp024-%j.err
set -euo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true; module load singularity/3.5.3 || true
S="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"; R="$(cd "$S/../.." && pwd)"; O="$R/outputs/analysis/exp024"; mkdir -p "$O"
C="${CONTAINER_FOLDER:-/group/project143/common/containers}/${CONTAINER_NAME:-kaggle-gpu-images-python-v163.sif}"
singularity exec --nv --home "$R" --bind "$R:$R" "$C" python3 "$S/blend.py" 2>&1 | tee "$O/run.log"
