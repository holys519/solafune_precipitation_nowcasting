#!/bin/bash
#SBATCH --partition=shared-a100-v2
#SBATCH --account=project143
#SBATCH --job-name=exp056-multi-ens
#SBATCH --output=slurm-exp056-multi-ens-%j.out
#SBATCH --error=slurm-exp056-multi-ens-%j.err
#SBATCH --time=00:40:00
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
  echo '--- 6-seed equal average ---'
  python3 build_seed_ensemble.py --members exp056 exp056_seed123 exp056_seed456 exp056_seed789 exp056_seed1337 exp056_seed2024 --name exp056_seed_ens_6
  echo '--- 4-seed (drop seed123 bad-LB, seed1337 bad-OOF) ---'
  python3 build_seed_ensemble.py --members exp056 exp056_seed456 exp056_seed789 exp056_seed2024 --name exp056_seed_ens_4best
"
