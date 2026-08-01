#!/bin/bash
#SBATCH --partition=shared-a100-v2
#SBATCH --account=project143
#SBATCH --job-name=verify-causal-exp066
#SBATCH --output=/group/project143/yamamoto/solafune_precipitation_nowcasting/scripts/slurm-verify-causal-exp066-%j.out
#SBATCH --error=/group/project143/yamamoto/solafune_precipitation_nowcasting/scripts/slurm-verify-causal-exp066-%j.err
#SBATCH --time=00:30:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# CPU-only causal-replay audit (organizers' 2026-07-20 ruling verification) for exp066's two
# temporal-fusion configs, mirroring the required check documented in scripts/verify_causal_replay.py
# and referenced (but never actually run) in exp066's config descriptions.
set -uo pipefail
PROJECT_ROOT="/group/project143/yamamoto/solafune_precipitation_nowcasting"
CONTAINER="/group/project143/common/containers/kaggle-gpu-images-python-v163.sif"
if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh; fi
module load singularity/3.5.3 || true

cd "$PROJECT_ROOT"
for cfg in config_convlstm_cr2.yaml config_attention_cr2.yaml; do
  echo "=== verify_causal_replay: exp066 / $cfg ==="
  singularity exec --home "$PROJECT_ROOT" --bind "$PROJECT_ROOT:$PROJECT_ROOT" "$CONTAINER" \
    python3 scripts/verify_causal_replay.py --exp-dir g_experiments/exp066 --config "$cfg" --num-rows 30
  echo "exit_code=$?"
done
