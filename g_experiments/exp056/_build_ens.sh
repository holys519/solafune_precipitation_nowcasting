#!/bin/bash
#SBATCH --partition=shared-a100-v2
#SBATCH --account=project143
#SBATCH --job-name=exp056-seed-ens
#SBATCH --output=slurm-exp056-seed-ens-%j.out
#SBATCH --error=slurm-exp056-seed-ens-%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
set -euo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONTAINER="/group/project143/common/containers/kaggle-gpu-images-python-v163.sif"
if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh; fi
module load singularity/3.5.3 || true
singularity exec --home "$PROJECT_ROOT" --bind "$PROJECT_ROOT:$PROJECT_ROOT" "$CONTAINER" bash -lc "
  cd '$SCRIPT_DIR'
  echo '--- 3-seed equal average (42+123+456) ---'
  python3 build_seed_ensemble.py --members exp056 exp056_seed123 exp056_seed456
  echo '--- 2-seed average (42+456, two best LB) ---'
  python3 build_seed_ensemble.py --members exp056 exp056_seed456 --name exp056_seed_ens_42_456
"
