#!/bin/bash
#SBATCH --partition=shared-a100-v2
#SBATCH --account=project143
#SBATCH --job-name=exp058-smoke
#SBATCH --output=slurm-exp058-smoke-%j.out
#SBATCH --error=slurm-exp058-smoke-%j.err
#SBATCH --time=00:10:00
#SBATCH --ntasks=1
#SBATCH --ntasks-per-node=1
#SBATCH --gpus-per-node=1
#SBATCH --cpus-per-task=2
#SBATCH --mem=8G

set -euo pipefail

if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ -f "$SLURM_SUBMIT_DIR/singularity_smoke.sh" ]; then
  SCRIPT_DIR="$SLURM_SUBMIT_DIR"
else
  SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
PROJECT_ROOT="$(cd "$SCRIPT_DIR/../.." && pwd)"
CONTAINER="${CONTAINER_FOLDER:-/group/project143/common/containers}/${CONTAINER_NAME:-kaggle-gpu-images-python-v163.sif}"

if [ -f /etc/profile.d/modules.sh ]; then
  source /etc/profile.d/modules.sh
fi
module load singularity/3.5.3 || true

singularity exec --nv --home "$PROJECT_ROOT" --bind "$PROJECT_ROOT:$PROJECT_ROOT" "$CONTAINER" bash -lc "
  cd '$PROJECT_ROOT'
  export PYTHONPATH='$PROJECT_ROOT/src'
  python -m compileall -q src scripts tests
  python - <<'PY'
import importlib.util
from pathlib import Path

path = Path('tests/test_research_pipeline.py').resolve()
spec = importlib.util.spec_from_file_location('exp058_smoke_tests', path)
if spec is None or spec.loader is None:
    raise RuntimeError(f'Cannot load smoke tests: {path}')
suite = importlib.util.module_from_spec(spec)
spec.loader.exec_module(suite)
for name in sorted(item for item in dir(suite) if item.startswith('test_')):
    getattr(suite, name)()
    print('PASS', name)
PY
"
