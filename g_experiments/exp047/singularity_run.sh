#!/bin/bash
#SBATCH --partition=shared-a100-v2
#SBATCH --account=project143
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=7
#SBATCH --mem=32G
#SBATCH --time=1-00:00:00
#SBATCH --output=slurm-exp047-%j.out
#SBATCH --error=slurm-exp047-%j.err

# exp047 single-fold worker on a Slurm cluster via Singularity.
# Usage:
#   bash submit_folds.sh config.yaml       # folds 0-4 as separate jobs, then submit
#   sbatch singularity_run.sh config.yaml 2 # run only fold 2
#   sbatch singularity_run.sh config.yaml submit

set -euxo pipefail

if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
fi

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/run.sh" ]; then
  SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONFIG="${1:-config.yaml}"
FOLD="${2:-}"

if [ "$CONFIG" = "all" ] || [ "$CONFIG" = "all_submit" ] || [ "$CONFIG" = "analyze" ] || \
   [ "$CONFIG" = "analysis" ] || [ "$CONFIG" = "submit" ] || [ "$CONFIG" = "submission" ] || \
   [ "$CONFIG" = "submit_calibrated" ] || [ "$CONFIG" = "infer" ] || [ "$CONFIG" = "inference" ] || \
   [[ "$CONFIG" =~ ^[0-9]+$ ]]; then
  FOLD="$CONFIG"
  CONFIG="config.yaml"
fi

if [ -z "$FOLD" ]; then
  echo "ERROR: Specify one fold (0-4) or a post-training action."
  echo "For the full pipeline, run: bash $SCRIPT_DIR/submit_folds.sh $CONFIG"
  exit 2
fi

if [ "$FOLD" = "all" ] || [ "$FOLD" = "all_submit" ]; then
  echo "ERROR: $FOLD would train all folds serially in one 24-hour job."
  echo "Run each fold separately with: bash $SCRIPT_DIR/submit_folds.sh $CONFIG"
  exit 2
fi

CONTAINER_FOLDER="${CONTAINER_FOLDER:-/group/project143/common/containers}"
CONTAINER_NAME="${CONTAINER_NAME:-kaggle-gpu-images-python-v163.sif}"
CONTAINER_PATH="$CONTAINER_FOLDER/$CONTAINER_NAME"

if [ ! -r "$CONTAINER_PATH" ]; then
  echo "ERROR: Container not readable: $CONTAINER_PATH"
  exit 1
fi

module load singularity/3.5.3 || true

echo "=========================================="
echo "g_experiments/exp047"
echo "=========================================="
echo "Container: $CONTAINER_PATH"
echo "Project:   $PROJECT_DIR"
echo "Config:    $CONFIG"
echo "Fold:      $FOLD"

SINGULARITY_ARGS=(
  --nv
  --home "$PROJECT_DIR"
  --bind "$PROJECT_DIR:$PROJECT_DIR"
)

if [ -n "${SLURM_JOB_ID:-}" ] && [ -d "/local/job/${SLURM_JOB_ID}" ]; then
  SINGULARITY_ARGS+=(--bind "/local/job/${SLURM_JOB_ID}:/local/job/${SLURM_JOB_ID}")
fi

singularity exec \
  "${SINGULARITY_ARGS[@]}" \
  "$CONTAINER_PATH" \
  env PYTHON=python3 bash "$SCRIPT_DIR/run.sh" "$CONFIG" "$FOLD"
