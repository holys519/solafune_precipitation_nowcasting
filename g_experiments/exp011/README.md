# exp011: Satellite Adapter Two-Head

`exp004` のtwo-head構造と `exp006` の衛星別input adapterを組み合わせます。

## Strategy

- 入力末尾3chのsatellite one-hotで Himawari/GOES/Meteosat 別stemを選択。
- stem後は共有U-Net body。
- 出力は `rain_logits` と `rain_amount` のtwo-head。
- checkpoint選択とOOF分析は公式metric寄りの `tile_rmse` を優先。
- `value_threshold_grid` による微小雨ゼロ化もOOFで記録。

## Run

```bash
cd g_experiments/exp011
sbatch singularity_run.sh
sbatch singularity_run.sh config.yaml all
sbatch singularity_run.sh submit
```

## Outputs

| Path | Description |
| --- | --- |
| `../../g_model/exp011/best_model_fold{k}.pt` | fold別best checkpoint |
| `../../outputs/submissions/exp011_submission.zip` | 提出zip |
| `../../outputs/analysis/exp011/` | OOF、閾値sweep、予測分布 |
