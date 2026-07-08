#!/bin/bash
set -euxo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_FOLDER="$(cd "$SCRIPT_DIR/../.." && pwd)"
STAGE="${1:-${EXP001_STAGE:-train}}"

cd "$PROJECT_FOLDER"

echo "=========================================="
echo "g_experiments/exp001: Compact UNet baseline"
echo "=========================================="
echo "Project root: $PROJECT_FOLDER"
echo "Stage: $STAGE"

nvidia-smi --query-gpu=name,memory.total,memory.free --format=csv || true

if [ -n "${SLURM_JOB_ID:-}" ] && [ -d "/local/job/${SLURM_JOB_ID}" ]; then
  export PYTHONUSERBASE="/local/job/${SLURM_JOB_ID}/pyusr"
else
  export PYTHONUSERBASE="$PROJECT_FOLDER/.cache/pyusr-exp001"
fi
mkdir -p "$PYTHONUSERBASE"
export PATH="$PYTHONUSERBASE/bin:$PATH"

python -c "import yaml" || pip install --no-cache-dir --user pyyaml
python -c "import torch; print(f'PyTorch: {torch.__version__}'); print(f'CUDA: {torch.cuda.is_available()}'); print(f'GPUs: {torch.cuda.device_count()}')"

echo "Preparing data with g_experiments/exp000..."
bash "$PROJECT_FOLDER/g_experiments/exp000/download.sh"

cd "$SCRIPT_DIR"
python check_data.py --config config.yaml

case "$STAGE" in
  check)
    echo "Data check complete."
    ;;
  train)
    python train.py
    ;;
  inference)
    python inference.py
    ;;
  submission)
    python make_submission.py
    ;;
  all)
    python train.py
    python inference.py
    python make_submission.py
    ;;
  *)
    echo "ERROR: unknown stage '$STAGE' (expected check, train, inference, submission, all)"
    exit 1
    ;;
esac

echo "=========================================="
echo "exp001 stage complete: $STAGE"
echo "=========================================="
