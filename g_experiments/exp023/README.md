# exp023: exp016 serving and calibration diagnostics

Reuses exp016 checkpoints (no training) and writes isolated mean/median diagnostics under
`outputs/analysis/exp023/`. One GPU is requested because OOF forward inference is required.

```bash
sbatch singularity_run.sh
```
