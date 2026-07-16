#!/bin/bash
# Submit one fold for exp028-exp032, then analyze each experiment only after
# its corresponding training job finishes successfully.
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
FOLD="${1:-0}"
EXPERIMENTS=(exp028 exp029 exp030 exp031 exp032)

[[ "$FOLD" =~ ^[0-4]$ ]] || { echo "Fold must be one of 0,1,2,3,4: $FOLD" >&2; exit 2; }

printf '%-8s %-12s %-12s\n' experiment train_job analysis_job
for experiment in "${EXPERIMENTS[@]}"; do
  experiment_dir="$SCRIPT_DIR/$experiment"
  train_job="$(cd "$experiment_dir" && sbatch --parsable singularity_run.sh "$FOLD")"
  train_job="${train_job%%;*}"
  analysis_job="$(cd "$experiment_dir" && sbatch --parsable --dependency="afterok:$train_job" singularity_run.sh "analyze_fold$FOLD")"
  analysis_job="${analysis_job%%;*}"
  printf '%-8s %-12s %-12s\n' "$experiment" "$train_job" "$analysis_job"
done
