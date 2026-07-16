#!/bin/bash
#SBATCH --partition=shared-a100-v2
#SBATCH --account=project143
#SBATCH --ntasks=2
#SBATCH --ntasks-per-node=2
#SBATCH --gpus-per-node=2
#SBATCH --cpus-per-task=16
#SBATCH --time=720
#SBATCH --output=slurm-exp028-%j.out
#SBATCH --error=slurm-exp028-%j.err
set -euo pipefail
EXPECTED_EXP=exp028
if [ -n "${SLURM_SUBMIT_DIR:-}" ] && [ "$(basename "$SLURM_SUBMIT_DIR")" = "$EXPECTED_EXP" ]; then
  VARIANT_DIR="$SLURM_SUBMIT_DIR"
else
  VARIANT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
fi
[ -f "$VARIANT_DIR/config.yaml" ] || { echo "Cannot resolve $EXPECTED_EXP directory: $VARIANT_DIR" >&2; exit 1; }
bash "$VARIANT_DIR/../singularity_variant.sh" "$VARIANT_DIR" "${1:-0}"
