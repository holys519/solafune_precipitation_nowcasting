#!/bin/bash
#SBATCH --partition=shared-a100-v2
#SBATCH --account=project143
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --time=720
#SBATCH --output=slurm-exp021-%j.out
#SBATCH --error=slurm-exp021-%j.err
set -euo pipefail
source /etc/profile.d/modules.sh 2>/dev/null || true; module load singularity/3.5.3 || true
S="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "$0")" && pwd)}"; R="$(cd "$S/../.." && pwd)"
C="${CONTAINER_FOLDER:-/group/project143/common/containers}/${CONTAINER_NAME:-kaggle-gpu-images-python-v163.sif}"
singularity exec --nv --home "$R" --bind "$R:$R" "$C" env PYTHON=python3 bash "$S/run.sh" "${1:-both}"
