#!/bin/bash
#SBATCH --partition=shared-a100-v2
#SBATCH --account=project143
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=16
#SBATCH --time=360
#SBATCH --output=slurm-submissions-%j.out
#SBATCH --error=slurm-submissions-%j.err

# Build submission zips from already-trained checkpoints (analyze -> inference ->
# make_submission via each experiment's run.sh "submit" stage). Produces
# outputs/submissions/expNNN_submission.zip for each experiment given.
#
# Usage:
#   sbatch make_submissions_run.sh                        # default: exp016 exp017
#   sbatch make_submissions_run.sh exp016                 # one experiment
#   sbatch make_submissions_run.sh exp016:config_median_serving.yaml
#   sbatch make_submissions_run.sh exp016::submit_calibrated exp017
#
# Each argument is expNNN[:config[:stage]] — config defaults to config.yaml,
# stage defaults to "submit" (see run.sh; e.g. submit_calibrated, inference).
# Experiments run sequentially; a failure (e.g. no checkpoints yet) does not
# stop the remaining ones, but the job exits non-zero at the end.

set -uxo pipefail

if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
fi

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/make_submissions_run.sh" ]; then
  SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"

CONTAINER_FOLDER="${CONTAINER_FOLDER:-/group/project143/common/containers}"
CONTAINER_NAME="${CONTAINER_NAME:-kaggle-gpu-images-python-v163.sif}"
CONTAINER_PATH="$CONTAINER_FOLDER/$CONTAINER_NAME"

if [ ! -r "$CONTAINER_PATH" ]; then
  echo "ERROR: Container not readable: $CONTAINER_PATH"
  exit 1
fi

module load singularity/3.5.3 || true

TARGETS=("$@")
if [ "${#TARGETS[@]}" -eq 0 ]; then
  TARGETS=(exp016 exp017)
fi

SINGULARITY_ARGS=(
  --nv
  --home "$PROJECT_DIR"
  --bind "$PROJECT_DIR:$PROJECT_DIR"
)

if [ -n "${SLURM_JOB_ID:-}" ] && [ -d "/local/job/${SLURM_JOB_ID}" ]; then
  SINGULARITY_ARGS+=(--bind "/local/job/${SLURM_JOB_ID}:/local/job/${SLURM_JOB_ID}")
fi

FAILED=()

for target in "${TARGETS[@]}"; do
  IFS=':' read -r exp config stage <<<"$target"
  config="${config:-config.yaml}"
  stage="${stage:-submit}"
  exp_dir="$SCRIPT_DIR/$exp"

  echo "=========================================="
  echo "submission: $exp (config=$config stage=$stage)"
  echo "=========================================="

  if [ ! -f "$exp_dir/run.sh" ]; then
    echo "ERROR: $exp_dir/run.sh not found; skipping"
    FAILED+=("$exp")
    continue
  fi

  if singularity exec \
    "${SINGULARITY_ARGS[@]}" \
    "$CONTAINER_PATH" \
    env PYTHON=python3 bash "$exp_dir/run.sh" "$config" "$stage"; then
    echo "OK: $exp -> $PROJECT_DIR/outputs/submissions/${exp}_submission.zip"
  else
    echo "FAILED: $exp (stage=$stage)"
    FAILED+=("$exp")
  fi
done

echo "=========================================="
ls -la "$PROJECT_DIR/outputs/submissions/" || true

if [ "${#FAILED[@]}" -gt 0 ]; then
  echo "FAILED experiments: ${FAILED[*]}"
  exit 1
fi
echo "all submissions built."
