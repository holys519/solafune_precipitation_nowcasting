#!/bin/bash
#SBATCH --partition=shared-a100-v2
#SBATCH --account=project143
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --time=120
#SBATCH --output=slurm-kaggle-upload-%j.out
#SBATCH --error=slurm-kaggle-upload-%j.err

# One-off Kaggle Models upload, run inside the container (network egress + kaggle CLI are
# only available there, not on the interactive/login shell).
# Usage:
#   sbatch singularity_upload.sh check     # verify kaggle CLI + credentials only
#   sbatch singularity_upload.sh create    # kaggle models create (metadata only, once)
#   sbatch singularity_upload.sh instance  # kaggle models instances create (uploads the tars)

set -euo pipefail

if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
fi
module load singularity/3.5.3 || true

SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)}"
REAL_HOME="/home/ba110396"

CONTAINER_FOLDER="${CONTAINER_FOLDER:-/group/project143/common/containers}"
CONTAINER_NAME="${CONTAINER_NAME:-kaggle-gpu-images-python-v163.sif}"
CONTAINER_PATH="$CONTAINER_FOLDER/$CONTAINER_NAME"
[ -r "$CONTAINER_PATH" ] || { echo "Container not readable: $CONTAINER_PATH"; exit 1; }

MODE="${1:-check}"

run_in_container() {
  singularity exec --nv \
    --bind "$SCRIPT_DIR:$SCRIPT_DIR" \
    --bind "$REAL_HOME/.kaggle:$REAL_HOME/.kaggle" \
    "$CONTAINER_PATH" \
    env KAGGLE_CONFIG_DIR="$REAL_HOME/.kaggle" bash -c "$1"
}

case "$MODE" in
  check)
    run_in_container "set -x; which kaggle; echo EXIT_WHICH=\$?; kaggle --version; echo EXIT_VERSION=\$?; kaggle models list -m; echo EXIT_LIST=\$?"
    ;;
  create)
    run_in_container "cd '$SCRIPT_DIR' && kaggle models create -p ."
    ;;
  instance)
    run_in_container "cd '$SCRIPT_DIR' && kaggle models instances create -p . -r skip"
    ;;
  *)
    echo "Unknown mode: $MODE" >&2
    exit 2
    ;;
esac
