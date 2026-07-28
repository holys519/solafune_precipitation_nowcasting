# exp058 — High-resolution total × spatial-distribution nowcaster

This is the first experiment backed by the shared `src/precip_nowcast` research
package. It addresses three measured failure modes in the exp038 family:

1. satellite rasters were resized to the 41×41 target grid before the network;
2. wet intensity and rain probability could multiply into a miscalibrated served mean;
3. most loss weight was not applied to the final submitted prediction.

The model consumes 164×164 satellite frames and emits a mathematically identifiable
decomposition:

```text
prediction = tile_total × spatial_distribution
sum(spatial_distribution) = 1
sum(prediction) = tile_total
```

The occurrence head is auxiliary only. It cannot dilute total precipitation at
serving time.

## Local validation

```bash
uv sync --extra dev
uv run pytest
uv run python -m compileall -q src scripts tests
```

## Fold gate

Run folds 0 and 4 first. Advancement requires improvement on both folds, or one
improvement with the other inside the paired location-level uncertainty interval.

```bash
sbatch singularity_smoke.sh
sbatch singularity_run.sh 0
sbatch singularity_run.sh 4
```

Do not compare only the aggregate fold score. Report tile RMSE, pooled RMSE,
mean absolute total error, per-location differences, and satellite-stratified
differences against `exp038_sigmafixed`.
