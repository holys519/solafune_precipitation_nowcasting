#!/bin/bash
#SBATCH --partition=shared-a100-v2
#SBATCH --account=project143
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --time=720
#SBATCH --output=slurm-exp032-%j.out
#SBATCH --error=slurm-exp032-%j.err
set -euo pipefail
bash "$(cd "$(dirname "$0")/.." && pwd)/singularity_variant.sh" "$(cd "$(dirname "$0")" && pwd)" "${1:-0}"
