# exp004: Two-Head Rain Detection + Amount Regression

exp003の診断ログ基盤を維持しつつ、降水の有無と降水量を分けて学習します。目的は、ゼロ雨画素が多い中で
false-positive drizzleを減らし、雨画素の回帰を別に最適化できるか確認することです。

## Strategy

- `two_head_compact_unet`: 共有Compact U-Net bodyに `rain_logits` と `rain_amount` headを追加。
- 最終予測は `sigmoid(rain_logits) * softplus(rain_amount_raw)`。
- lossはBCE + rain amount回帰 + final prediction回帰の合成。
- OOF解析で `rain_prob_threshold` をスイープし、`oof_rain_threshold_sweep.csv` と
  `oof_calibration.json` に保存する。
- `submit_calibrated` ではOOFで選んだrain probability thresholdを推論に適用する。

## Run

```bash
cd g_experiments/exp004
sbatch singularity_run.sh                 # config.yaml, all_submit
sbatch singularity_run.sh config.yaml 0   # train fold 0 only
sbatch singularity_run.sh all             # train folds 0-4 only
sbatch singularity_run.sh submit          # analyze existing checkpoints -> inference -> zip
sbatch singularity_run.sh submit_calibrated
```

## Outputs

| Path | Description |
| --- | --- |
| `../../g_model/exp004/best_model_fold{k}.pt` | fold別best checkpoint |
| `../../g_model/exp004/metrics_fold{k}.json` | fold別metrics |
| `../../outputs/submissions/exp004/` | 展開済み提出ディレクトリ |
| `../../outputs/submissions/exp004_submission.zip` | 提出zip |
| `../../outputs/analysis/exp004/training_log_fold{k}.csv` | epochごとの学習ログ |
| `../../outputs/analysis/exp004/oof_sample_metrics.csv` | OOF sample別予測/target集計 |
| `../../outputs/analysis/exp004/oof_group_metrics.csv` | location/satellite/fold別OOF集計 |
| `../../outputs/analysis/exp004/oof_rain_threshold_sweep.csv` | rain probability threshold sweep |
| `../../outputs/analysis/exp004/oof_calibration.json` | OOF由来のscale/bias/threshold候補 |
| `../../outputs/analysis/exp004/evaluation_prediction_summary.csv` | evaluation予測分布 |

## Notes

まずは通常の `submit` と `submit_calibrated` の両方を作り、OOFとpublicの対応を見るのが良いです。
