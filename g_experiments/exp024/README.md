# exp024: exp009/016/017 prediction blend

No training. Blends already generated evaluation TIFFs and creates weight-specific submission
directories plus a manifest in `outputs/analysis/exp024/`.

```bash
sbatch singularity_run.sh
```

Default candidates are equal 009/016/017, equal 016/017, and 009/016/017 = 0.2/0.4/0.4.
