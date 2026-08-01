#!/bin/bash
#SBATCH --partition=shared-a100-v2
#SBATCH --account=project143
#SBATCH --job-name=exp064-smoke-new-backbones
#SBATCH --output=slurm-exp064-smoke-new-backbones-%j.out
#SBATCH --error=slurm-exp064-smoke-new-backbones-%j.err
#SBATCH --time=00:20:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G

# 2026-07-30: pre-flight smoke test for the two new backbone-exploration configs
# (config_effb4.yaml, config_convnext_small_lr2e4.yaml) before any fold-training GPU budget is
# spent on them -- verifies the timm encoder name actually resolves and downloads pretrained
# weights, channel math is consistent, and a CPU-side forward/backward pass is finite, using the
# exact same check_arm()/check_loss_ablation_toggles() this experiment already trusts for its
# other arms (see smoke_test.py, _smoke_all.sh). Exits non-zero on ANY failure so that
# --dependency=afterok on this job is a real gate, not just documentation.
set -uo pipefail
SCRIPT_DIR="${SLURM_SUBMIT_DIR:-$(pwd)}"
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONTAINER="/group/project143/common/containers/kaggle-gpu-images-python-v163.sif"
if [ -f /etc/profile.d/modules.sh ]; then source /etc/profile.d/modules.sh; fi
module load singularity/3.5.3 || true
singularity exec --nv --home "$PROJECT_ROOT" --bind "$PROJECT_ROOT:$PROJECT_ROOT" "$CONTAINER" bash -lc "
  cd '$SCRIPT_DIR'
  export PYTHON=python3
  python3 - <<PY
import sys
import smoke_test as st

failed = []
for cfg in ['config_effb4.yaml', 'config_convnext_small_lr2e4.yaml']:
    try:
        r = st.check_arm(cfg)
        print('PASS(check_arm)', cfg, r)
        st.check_loss_ablation_toggles(cfg)
        print('PASS(loss_ablation)', cfg)
    except Exception as e:
        import traceback
        traceback.print_exc()
        print('FAIL', cfg, type(e).__name__, str(e)[:300])
        failed.append(cfg)

if failed:
    print('SMOKE TEST FAILED for:', failed)
    sys.exit(1)
print('ALL NEW BACKBONE CONFIGS PASSED SMOKE TEST')
PY
"
