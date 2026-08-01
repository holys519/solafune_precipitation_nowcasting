#!/bin/bash
#SBATCH --partition=shared-a100-v2
#SBATCH --account=project143
#SBATCH --job-name=exp066-smoke-temporal
#SBATCH --output=slurm-exp066-smoke-temporal-%j.out
#SBATCH --error=slurm-exp066-smoke-temporal-%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# 2026-07-31: pre-flight smoke test for exp066's new TemporalFusionFactorizedUNet (ConvLSTM +
# attention fusion arms) before any fold-training GPU budget is spent. Covers random-tensor
# forward/backward + loss-ablation toggles (check_arm/check_loss_ablation_toggles) AND a real-data
# batch through the unmodified exp063 dataset pipeline (check_real_batch), which is the part that
# actually exercises the new per-frame split / causal reordering logic against real channel
# layouts, not just a synthetic tensor of the right shape.
set -uo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONTAINER="/group/project143/common/containers/kaggle-gpu-images-python-v163.sif"
if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh; fi
module load singularity/3.5.3 || true
singularity exec --nv --home "$PROJECT_ROOT" --bind "$PROJECT_ROOT:$PROJECT_ROOT" "$CONTAINER" bash -lc "
  cd '$SCRIPT_DIR'
  export PYTHON=python3
  python3 smoke_test.py
"
