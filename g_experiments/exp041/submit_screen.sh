#!/bin/bash
# Submit the two-arm fold-0/fold-4 screen. No analysis/inference/submission job is created.
# Optional:
#   DEPENDENCY=job1:job2:...  -> wait for these jobs after any terminal state
#   AFTER_OK=job3:job4:...    -> run only when these jobs succeeded

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
SBATCH_SCRIPT="$SCRIPT_DIR/singularity_run.sh"
SBATCH_BIN="${SBATCH_BIN:-sbatch}"
DEPENDENCY="${DEPENDENCY:-}"
AFTER_OK="${AFTER_OK:-}"

dependency_args=()
dependency_terms=()
if [ -n "$DEPENDENCY" ]; then
  dependency_terms+=("afterany:$DEPENDENCY")
fi
if [ -n "$AFTER_OK" ]; then
  dependency_terms+=("afterok:$AFTER_OK")
fi
if [ "${#dependency_terms[@]}" -gt 0 ]; then
  dependency_value="$(IFS=,; echo "${dependency_terms[*]}")"
  dependency_args+=("--dependency=$dependency_value")
fi

for arm in control metric; do
  config="$SCRIPT_DIR/config_${arm}.yaml"
  for fold in 0 4; do
    raw_job_id="$($SBATCH_BIN \
      --parsable \
      "${dependency_args[@]}" \
      --job-name="exp041-${arm}-f${fold}" \
      --output="$SCRIPT_DIR/slurm-exp041-${arm}-f${fold}-%j.out" \
      --error="$SCRIPT_DIR/slurm-exp041-${arm}-f${fold}-%j.err" \
      "$SBATCH_SCRIPT" "$config" "$fold")"
    job_id="${raw_job_id%%;*}"
    echo "submitted $arm fold $fold -> job $job_id"
  done
done
