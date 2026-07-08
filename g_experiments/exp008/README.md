# exp008: Official Metric + Drizzle Post-Processing

`exp004` のtwo-head checkpointを再利用し、公式metricに近い tile別RMSE平均と
微小雨ゼロ化の後処理を検証する実験です。新規学習は不要です。

## Strategy

- OOF/validationで `tile_rmse = mean(sqrt(mean(pixel_error^2)))` を保存。
- `value_threshold_grid` で `pred < threshold` を0にする後処理をOOF sweep。
- 推論時は既定で `../../g_model/exp004` のcheckpointを読み、`exp008` の提出物として出力。
- `temporal_smoothing` は実装済みですが、既定ではoffです。

## Run

```bash
cd g_experiments/exp008
sbatch singularity_run.sh          # analyze exp004 checkpoints -> inference -> zip
sbatch singularity_run.sh submit
sbatch singularity_run.sh submit_calibrated
```

`SOURCE_MODEL_DIR=/path/to/models` で読み込むcheckpointディレクトリを差し替えられます。

## Outputs

| Path | Description |
| --- | --- |
| `../../outputs/submissions/exp008_submission.zip` | 提出zip |
| `../../outputs/analysis/exp008/oof_value_threshold_sweep.csv` | 値閾値sweep |
| `../../outputs/analysis/exp008/oof_calibration.json` | OOF由来のscale/bias/threshold |
| `../../outputs/analysis/exp008/evaluation_prediction_summary.csv` | evaluation予測分布 |
