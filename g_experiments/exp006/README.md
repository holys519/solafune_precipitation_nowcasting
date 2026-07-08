# exp006: Satellite-Specific Adapter U-Net

Himawari/GOES/Meteosatはすべて16chですが、波長定義や値分布は同一ではありません。exp006では入力stemだけを
衛星ごとに分け、その後のU-Net本体は共有します。

## Strategy

- `satellite_adapter_unet`: 3つの衛星別stemを持つ。
- 入力末尾3chのsatellite one-hot mapで、batch内サンプルごとにstem出力を選択。
- 共有decoderへ渡すため、パラメータ増加は入力近傍に限定。
- lossや前処理はexp003相当のweighted MSEを維持し、衛星adapterの効果を見る。

## Run

```bash
cd g_experiments/exp006
sbatch singularity_run.sh
sbatch singularity_run.sh config.yaml 0
sbatch singularity_run.sh all
sbatch singularity_run.sh submit
sbatch singularity_run.sh submit_calibrated
```

## Outputs

| Path | Description |
| --- | --- |
| `../../g_model/exp006/best_model_fold{k}.pt` | fold別best checkpoint |
| `../../outputs/submissions/exp006_submission.zip` | 提出zip |
| `../../outputs/analysis/exp006/` | 学習ログ、OOF、推論分布 |
