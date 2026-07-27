#!/bin/bash
#SBATCH --partition=shared-a100-v2
#SBATCH --account=project143
#SBATCH --job-name=exp064-smoke-all
#SBATCH --output=slurm-exp064-smoke-all-%j.out
#SBATCH --error=slurm-exp064-smoke-all-%j.err
#SBATCH --time=00:25:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
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
import smoke_test as st
for cfg in ['config_effb3.yaml','config_effv2s.yaml','config_resnet34.yaml','config_convnext.yaml','config_swin.yaml']:
    try:
        r=st.check_arm(cfg)
        print('PASS', cfg, r)
    except Exception as e:
        import traceback; traceback.print_exc(); print('FAIL', cfg, type(e).__name__, str(e)[:160])
PY
"
