#!/bin/bash
#SBATCH --partition=shared-a100-v2
#SBATCH --account=project143
#SBATCH --job-name=exp011-nested-gate
#SBATCH --output=slurm-g-eda-exp011-nested-gate-%j.out
#SBATCH --error=slurm-g-eda-exp011-nested-gate-%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G

# g_eda/exp011 phase 2b (2026-07-30 addition): honest outer-cross-fit blend weight search over
# the new architecture-diverse champion ensemble (exp056 / exp064_effb3 / exp064_effv2s /
# exp064_swin_lr2e4), immediately followed by l_eda/exp005's submission gate against the new solo
# champion. CPU-only -- both stages are pure numpy/csv over already-cached arrays.
#
# Must run AFTER all 4 sources' OOF caches exist (singularity_cache.sh <name> for each).
#
# Usage: sbatch singularity_nested_gate.sh [sources...]
#   default sources: exp056 exp064_effb3 exp064_effv2s exp064_swin_lr2e4

set -euxo pipefail

if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
fi

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/nested_blend.py" ]; then
  SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
PROJECT_DIR="$(cd "$SCRIPT_DIR/../.." && pwd)"

CONTAINER_FOLDER="${CONTAINER_FOLDER:-/group/project143/common/containers}"
CONTAINER_NAME="${CONTAINER_NAME:-kaggle-gpu-images-python-v163.sif}"
CONTAINER_PATH="$CONTAINER_FOLDER/$CONTAINER_NAME"
[ -r "$CONTAINER_PATH" ] || { echo "Container not readable: $CONTAINER_PATH"; exit 1; }

module load singularity/3.5.3 || true

SOURCES=("$@")
if [ "${#SOURCES[@]}" -eq 0 ]; then
  SOURCES=(exp056 exp064_effb3 exp064_effv2s exp064_swin_lr2e4)
fi
OUT_NAME="champion_ensemble_nested_blend"

singularity exec --home "$PROJECT_DIR" --bind "$PROJECT_DIR:$PROJECT_DIR" "$CONTAINER_PATH" bash -lc "
  cd '$SCRIPT_DIR'
  python3 nested_blend.py --sources ${SOURCES[*]} --out-name '$OUT_NAME'
  cd '$PROJECT_DIR/l_eda/exp005'
  python3 submission_gate.py --baseline exp064_effb3 --candidate '$OUT_NAME'
"
