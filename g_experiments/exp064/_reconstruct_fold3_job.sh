#!/bin/bash
#SBATCH --partition=shared-a100-v2
#SBATCH --account=project143
#SBATCH --job-name=exp064-reconstruct-fold3
#SBATCH --output=slurm-reconstruct-fold3-%j.out
#SBATCH --error=slurm-reconstruct-fold3-%j.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2

set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh; fi
module load singularity/3.5.3 || true

singularity exec --home "$PROJECT_DIR" --bind "$PROJECT_DIR:$PROJECT_DIR" \
  /group/project143/common/containers/kaggle-gpu-images-python-v163.sif \
  bash -lc "cd '$SCRIPT_DIR' && python3 _reconstruct_metrics_fold3.py"
