#!/bin/bash
#SBATCH --partition=GPU_PARTITION
#SBATCH --account=PROJECT_ACCOUNT
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --time=30
#SBATCH --output=slurm-%j.log

# exp000: HPCクラスタ上でデータ準備
# sbatch singularity_run.sh で実行

CONTAINER="/path/to/container.sif"
SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"

singularity exec \
  --bind "$SCRIPT_DIR:$SCRIPT_DIR" \
  "$CONTAINER" \
  bash "$SCRIPT_DIR/download.sh"
