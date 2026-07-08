# exp003: 5-Fold Ensemble With Diagnostics

exp002の結果はpublicで改善したものの、fold別CVのばらつきが大きく、何が効いたかを後から追いづらい状態でした。
exp003ではモデルを大きく変えず、`pos_weight=2.0` に弱めた保守的なweighted lossで、5-fold学習から
OOF分析、ensemble推論、submission zip作成までを一気通貫にします。

## Strategy

- exp002の前処理を維持: 衛星別/バンド別正規化、anti-aliased resize、augmentation、location group split。
- 雨画素重みを `4.0 -> 2.0` に下げ、雨への寄せ過ぎやdry regionの悪化を確認する。
- fold/epochごとの学習ログをCSV化し、OOFのsample/location/satellite別指標も保存する。
- 既存checkpointからでも `submit` で提出zipを再生成できるようにする。

## Run

Slurmクラスタ:

```bash
cd g_experiments/exp003
sbatch singularity_run.sh                 # config.yaml, all_submit
sbatch singularity_run.sh config.yaml 0   # train fold 0 only
sbatch singularity_run.sh all             # train folds 0-4 only
sbatch singularity_run.sh submit          # analyze existing checkpoints -> inference -> zip
sbatch singularity_run.sh submit_calibrated
```

`submit_calibrated` はOOFから推定したglobal scale/biasを推論に適用します。通常の比較用にはまず `submit`
を使います。

## Outputs

| Path | Description |
| --- | --- |
| `../../g_model/exp003/best_model_fold{k}.pt` | fold別best checkpoint |
| `../../g_model/exp003/metrics_fold{k}.json` | fold別metrics |
| `../../outputs/submissions/exp003/` | 展開済み提出ディレクトリ |
| `../../outputs/submissions/exp003_submission.zip` | 提出zip |
| `../../outputs/analysis/exp003/training_log_fold{k}.csv` | epochごとの学習ログ |
| `../../outputs/analysis/exp003/epoch_history.csv` | 全foldを結合したepoch履歴 |
| `../../outputs/analysis/exp003/fold_summary.csv` | fold別best epoch/score |
| `../../outputs/analysis/exp003/oof_sample_metrics.csv` | OOF sample別予測/target集計 |
| `../../outputs/analysis/exp003/oof_group_metrics.csv` | location/satellite/fold別OOF集計 |
| `../../outputs/analysis/exp003/oof_calibration.json` | OOF由来のscale/bias候補 |
| `../../outputs/analysis/exp003/evaluation_prediction_summary.csv` | evaluation予測分布 |

## Notes

`CONTAINER_FOLDER` と `CONTAINER_NAME` は環境変数で上書きできます。デフォルトは
`/group/project143/common/containers/kaggle-gpu-images-python-v163.sif` です。
GPUメモリは `logs/gpu_memory_${SLURM_JOB_ID}_*.csv` に300秒間隔で記録します。
