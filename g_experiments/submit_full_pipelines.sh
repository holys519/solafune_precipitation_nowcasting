#!/bin/bash
# Submit the full train -> analyze -> inference -> submission-zip pipeline for
# exp016 and exp017 as two independent Slurm jobs (they run in parallel).
# Run this on the cluster login node, NOT via sbatch:
#
#   cd g_experiments
#   bash submit_full_pipelines.sh                              # both use config.yaml
#   bash submit_full_pipelines.sh config.yaml config_engineered_only.yaml
#   bash submit_full_pipelines.sh - config_engineered_only.yaml   # "-" = default
#
# Arg1 = exp016 config, arg2 = exp017 config (default config.yaml). Each job
# trains folds 0-4 sequentially, then builds outputs/submissions/expNNN_submission.zip
# (the all_submit stage of each experiment's run.sh).
#
# NOTE: each singularity_run.sh must be submitted from its own directory
# (it resolves paths via SLURM_SUBMIT_DIR), hence the subshell cd below.

set -euo pipefail
cd "$(dirname "$0")"

CONFIG016="${1:-config.yaml}"
CONFIG017="${2:-config.yaml}"
[ "$CONFIG016" = "-" ] && CONFIG016=config.yaml
[ "$CONFIG017" = "-" ] && CONFIG017=config.yaml

for pair in "exp016:$CONFIG016" "exp017:$CONFIG017"; do
  exp="${pair%%:*}"
  config="${pair#*:}"
  if [ ! -f "$exp/$config" ]; then
    echo "ERROR: $exp/$config not found" >&2
    exit 1
  fi
  (
    cd "$exp"
    job_id="$(sbatch --parsable singularity_run.sh "$config" all_submit)"
    echo "submitted $exp ($config, all_submit) -> job $job_id (log: $exp/slurm-$exp-$job_id.out)"
  )
done

echo "monitor with: squeue -u \$USER"
echo "zips will appear at: ../outputs/submissions/exp016_submission.zip and exp017_submission.zip"
