# exp010: Data Cleanup Two-Head

`exp004` のtwo-head構造を保ち、discussionで指摘されたデータ品質問題に対応します。

## Strategy

- 2023-01-01 00:00:00 の疑わしい5つのtrain targetを学習/OOFから除外。
- IR系bandのraw 0をnodataとして扱い、正規化後0に置換。
- GOESの `282x282x4` native visible/near-IRファイルを `C01/C02/C03/C05` に配置。
- checkpoint選択とOOF分析は公式metric寄りの `tile_rmse` を優先。
- `value_threshold_grid` による微小雨ゼロ化もOOFで記録。

## Run

```bash
cd g_experiments/exp010
sbatch singularity_run.sh
sbatch singularity_run.sh config.yaml all
sbatch singularity_run.sh submit
```

## Outputs

| Path | Description |
| --- | --- |
| `../../g_model/exp010/best_model_fold{k}.pt` | fold別best checkpoint |
| `../../outputs/submissions/exp010_submission.zip` | 提出zip |
| `../../outputs/analysis/exp010/` | OOF、閾値sweep、予測分布 |
