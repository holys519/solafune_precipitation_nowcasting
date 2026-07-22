#!/bin/bash
# Submit folds 0-4 as independent jobs, followed by analysis/inference/submission.
# Usage:
#   bash submit_folds.sh [config]
#   bash submit_folds.sh config.yaml

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SBATCH_SCRIPT="$SCRIPT_DIR/singularity_run.sh"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
CONFIG="${1:-config.yaml}"

if [[ "$CONFIG" = /* ]]; then
  CONFIG_PATH="$CONFIG"
else
  CONFIG_PATH="$SCRIPT_DIR/$CONFIG"
fi

if [ ! -f "$CONFIG_PATH" ]; then
  echo "ERROR: Config not found: $CONFIG_PATH" >&2
  exit 2
fi

fold_job_ids=()
for fold in 0 1 2 3 4; do
  raw_job_id="$($SBATCH_BIN \
    --parsable \
    --job-name="exp053-fold${fold}" \
    --output="$SCRIPT_DIR/slurm-exp053-fold${fold}-%j.out" \
    --error="$SCRIPT_DIR/slurm-exp053-fold${fold}-%j.err" \
    "$SBATCH_SCRIPT" "$CONFIG" "$fold")"
  job_id="${raw_job_id%%;*}"
  fold_job_ids+=("$job_id")
  echo "submitted fold $fold -> job $job_id"
done

dependency="$(IFS=:; echo "${fold_job_ids[*]}")"
raw_submit_job_id="$($SBATCH_BIN \
  --parsable \
  --dependency="afterok:$dependency" \
  --job-name="exp053-submit" \
  --output="$SCRIPT_DIR/slurm-exp053-submit-%j.out" \
  --error="$SCRIPT_DIR/slurm-exp053-submit-%j.err" \
  "$SBATCH_SCRIPT" "$CONFIG" submit)"
submit_job_id="${raw_submit_job_id%%;*}"

echo "submitted analysis/inference/submission -> job $submit_job_id"
echo "dependency: afterok:$dependency"
