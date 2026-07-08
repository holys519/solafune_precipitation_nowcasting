# exp007: Multi-Experiment Ensemble and Post-Processing

exp007は新規学習を行わず、`exp003` から `exp006` までの既存checkpointを読み込んで等重みensembleを作ります。
各checkpointに埋め込まれたconfigからモデル構造を復元するため、`compact_unet`、`two_head_compact_unet`、
`temporal_fusion_unet`、`satellite_adapter_unet` が混ざっていても推論できます。

## Strategy

- `config.yaml` の `ensemble.sources` にある `g_model/exp003`〜`g_model/exp006` を走査。
- 各sourceの重みをsource内のfold数で割り、source間は等重みに正規化。
- 予測は各checkpointのfinal precipitation tensorを平均。
- `outputs/analysis/exp007/ensemble_source_summary.csv` と `ensemble_sources.csv` に使用sourceを記録。
- 既存のGeoTIFF書き出しとsubmission zip作成を再利用。

## Run

```bash
cd g_experiments/exp007
sbatch singularity_run.sh          # analyze sources -> inference -> zip
sbatch singularity_run.sh submit   # same as above
sbatch singularity_run.sh infer    # inference only
sbatch singularity_run.sh analyze  # source summary only
```

## Outputs

| Path | Description |
| --- | --- |
| `../../outputs/submissions/exp007/` | 展開済み提出ディレクトリ |
| `../../outputs/submissions/exp007_submission.zip` | 提出zip |
| `../../outputs/analysis/exp007/ensemble_source_summary.csv` | source別checkpoint/metric概要 |
| `../../outputs/analysis/exp007/ensemble_sources.csv` | 推論で使ったsourceと重み |
| `../../outputs/analysis/exp007/evaluation_prediction_summary.csv` | evaluation予測分布 |

## Notes

未学習のsourceはwarningを出してskipします。すべてのsourceにcheckpointがない場合はエラーになります。
