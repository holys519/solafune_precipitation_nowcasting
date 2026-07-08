# exp009: Successor-Row Frames

`exp004` のtwo-head構造を保ちつつ、同一locationの次行に含まれる衛星フレームも入力に加えます。
通常の3観測slotに successor row の3観測slotを連結するため、入力は `105ch` です。

## Strategy

- 現在行: `T-30/T-20/T-10` の最大3フレーム。
- successor row: 多くのケースで `T/T+10/T+20` の最大3フレーム。
- successorが無い、時刻が飛ぶ、衛星が違う場合はmask=0のゼロslot。
- checkpoint選択とOOF分析は公式metric寄りの `tile_rmse` を優先。
- `value_threshold_grid` による微小雨ゼロ化もOOFで記録。

## Run

```bash
cd g_experiments/exp009
sbatch singularity_run.sh                 # config.yaml, all_submit
sbatch singularity_run.sh config.yaml all # train folds 0-4
sbatch singularity_run.sh submit          # analyze existing checkpoints -> inference -> zip
```

## Outputs

| Path | Description |
| --- | --- |
| `../../g_model/exp009/best_model_fold{k}.pt` | fold別best checkpoint |
| `../../outputs/submissions/exp009_submission.zip` | 提出zip |
| `../../outputs/analysis/exp009/` | OOF、閾値sweep、予測分布 |
