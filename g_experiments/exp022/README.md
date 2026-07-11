# exp022: exp017 feature ablation on hard folds

Runs full/canonical-only/engineered-only arms without overwriting exp017 checkpoints.
Each arm is stored under `g_model/exp022/<arm>` and `outputs/analysis/exp022/<arm>`.

```bash
sbatch singularity_run.sh 1
sbatch singularity_run.sh 3
```
