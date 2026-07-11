# exp021: exp016/017 five-fold completion and comparison

Completes fold 4 for exp016 and exp017, regenerates their OOF diagnostics, and writes a
side-by-side comparison to `outputs/analysis/exp021/`.

Training uses at most 50 epochs, ReduceLROnPlateau (factor 0.3, patience 4), then early
stopping after 10 epochs without an improvement greater than 0.001. The best checkpoint is
still selected using every observed improvement.

```bash
sbatch singularity_run.sh both   # default
sbatch singularity_run.sh exp016
sbatch singularity_run.sh exp017
```
