# exp041 — isolated tile-RMSE fine-tuning screen (strict green)

## Question

exp040 Arm D は mean×shape architecture と metric loss を同時に変えているため、metric loss
単体の寄与が分かりません。exp041はexp038の保存済みstrict checkpointから、同一条件で
短時間継続する2 armを比較します。

- `control`: exp038 lossのまま継続
- `metric`: control + `0.3 * mean_batch(tile_RMSE)`

両armともoptimizerは同じく新規AdamW、LRは`1e-4`、最大12 epoch、fold 0/4のみです。
これによりoptimizer stateのリセット・追加学習時間の効果をcontrolで差し引けます。

## Integrity checks

1. epoch 0の検証値は元exp038 checkpointと丸め誤差内で一致すること。
2. 両armの初期checkpoint、data split、seed、architecture、augmentation、LR、epoch数は同一。
3. 差分は`loss.metric_weight`と出力先だけ。
4. evaluation target、Public score、amber/red artifactは学習・選択に使わない。

## Screening gate

元exp038は fold 0 `0.2895437043`、fold 4 `0.5960728986`。

- metricがcontrolをfold 0/4の両方で改善すること。
- metricが元exp038も両foldで改善すること。
- fold 0/4のsample-weighted改善がcontrol比で少なくとも`-0.002`であること。

一つでも満たさなければno-go。通過時だけfold 1–3を追加し、5-fold OOFで最終判断します。
screening結果からsubmissionは自動生成しません。

## Submitted jobs (2026-07-18 UTC)

- smoke: `3937343` — completed, passed
- control fold 0 / 4: `3937344` / `3937345`
- metric fold 0 / 4: `3937346` / `3937347`
- training jobs wait for exp040 jobs `3937254`–`3937257` to terminate

## Run

```bash
cd g_experiments/exp041
sbatch singularity_run.sh config_control.yaml smoke
bash submit_screen.sh
python3 summarize_screen.py
```

exp040完了後に開始したい場合:

```bash
AFTER_OK=<smoke_job_id> DEPENDENCY=3937254:3937255:3937256:3937257 bash submit_screen.sh
```
