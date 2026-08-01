#!/bin/bash
#SBATCH --partition=shared-a100-v2
#SBATCH --account=project143
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=8
#SBATCH --time=60
#SBATCH --output=slurm-g-eda-exp010-%j.out
#SBATCH --error=slurm-g-eda-exp010-%j.err

# g_eda/exp010: causal-only temporal-smoothing OOF re-tune. The actual work (numpy sweep over a
# cached OOF array) does not need a GPU and this script never passes --nv to singularity -- but
# 2026-07-30 testing (see g_eda/exp011/singularity_analyze.sh's identical fix) found this
# partition's scheduler (GAIA) hard-requires --gpus-per-node >= 1 on every job regardless of
# workload. The --gpus-per-node=1 above is requested-but-unused for exactly that reason.
#
# Prerequisite: the exp038_sigmafixed OOF prediction cache must exist at
# outputs/g_eda/exp003/exp038_sigmafixed_oof_pred.npz. That cache itself requires a GPU forward
# pass and is built with g_eda/exp003's existing caching entrypoint (not modified here):
#   cd g_eda/exp003 && sbatch singularity_cache_exp038.sh exp038_sigmafixed exp038 exp038_sigmafixed
# If a 5-fold exp047 cache also exists at outputs/g_eda/exp003/exp047_oof_pred.npz, it is picked
# up automatically as a second source; otherwise it is skipped and noted in CAUSAL_SMOOTHING.md.
#
# Usage: sbatch singularity_run.sh [-- extra run_causal_smoothing_sweep.py args]
#   e.g. sbatch singularity_run.sh -- --sources champion_ensemble_nested_blend \
#          --cache-dir "$PROJECT_DIR/outputs/g_eda/exp011" \
#          --out-json recommended_causal_weights_exp065.json \
#          --report-name CAUSAL_SMOOTHING_exp065.md
# (2026-07-30: added passthrough args support -- see run_causal_smoothing_sweep.py --help)

set -euxo pipefail

if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
fi

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/run_causal_smoothing_sweep.py" ]; then
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

singularity exec --home "$PROJECT_DIR" --bind "$PROJECT_DIR:$PROJECT_DIR" "$CONTAINER_PATH" \
  python3 "$SCRIPT_DIR/run_causal_smoothing_sweep.py" "$@"
