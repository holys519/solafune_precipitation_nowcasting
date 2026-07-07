#!/bin/bash
#SBATCH --partition=GPU_PARTITION
#SBATCH --account=PROJECT_ACCOUNT
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=10
#SBATCH --output=extract_scores_%j.log

source /etc/profile.d/modules.sh
module load singularity/3.5.3 || true

CONTAINER="/path/to/container.sif"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

singularity exec \
  --nv \
  --bind "$SCRIPT_DIR:$SCRIPT_DIR" \
  "$CONTAINER" \
  python3 "$SCRIPT_DIR/extract_scores.py"
