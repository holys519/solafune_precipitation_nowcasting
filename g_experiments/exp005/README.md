# exp005: Lightweight Temporal Fusion U-Net

exp003/exp004では3つの観測時刻を単純にチャネル結合していました。exp005では各時刻を同じstemで処理し、
観測maskで重み付け平均してからU-Netへ渡します。ConvLSTMへ進む前の、安く比較しやすい時間融合実験です。

## Strategy

- 入力54chを `3 time slots x 16 bands + 3 observation masks + 3 satellite one-hot` として分解。
- 各時刻は共有 `time_stem` で特徴抽出。
- 観測が存在するslotだけをmask付き平均で融合。
- satellite one-hotは `1x1 conv` で特徴へ加える。
- lossはexp003相当のweighted MSEに戻し、時間融合の効果を単独で見る。

## Run

```bash
cd g_experiments/exp005
sbatch singularity_run.sh
sbatch singularity_run.sh config.yaml 0
sbatch singularity_run.sh all
sbatch singularity_run.sh submit
sbatch singularity_run.sh submit_calibrated
```

## Outputs

| Path | Description |
| --- | --- |
| `../../g_model/exp005/best_model_fold{k}.pt` | fold別best checkpoint |
| `../../outputs/submissions/exp005_submission.zip` | 提出zip |
| `../../outputs/analysis/exp005/` | 学習ログ、OOF、推論分布 |
