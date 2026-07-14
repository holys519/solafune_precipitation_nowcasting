# exp018: Within-Tile Localization (G-032)

exp016のhurdle log-normal推定を保ちつつ、41x41タイル内の雨域配置を直接改善する実験。
targetはリサイズせず、入力特徴のみ128x128で処理してnative 41x41へadaptive average poolingする。

## Components

- 4-level high-resolution U-Net (`highres_hurdle_lognormal_unet`)
- calibrated occurrence headと独立したwet-mask補助head
- 補助maskは `target >= 0.25`、BCEとsoft Diceの混合
- native / 2x pooled / 4x pooled prediction lossによる位置ずれ耐性
- exp016互換のhurdle log-normal servingと推論出力

## Configs

| Config | High-res | Aux mask | Multi-scale |
| --- | --- | --- | --- |
| `config_highres_only.yaml` | yes | no | no |
| `config_highres_mask.yaml` | yes | yes | no |
| `config.yaml` | yes | yes | yes |

最初は同じfoldで3アームを比較し、full 5-foldは勝った構成だけを実行する。

```bash
sbatch singularity_run.sh config_highres_only.yaml 0
sbatch singularity_run.sh config_highres_mask.yaml 0
sbatch singularity_run.sh config.yaml 0
```

full 5-foldは、各foldを2 GPU・24時間の独立ジョブとして投入する。5 foldが全て成功した後、
OOF解析・推論・submission作成ジョブが自動的に開始される。

```bash
bash submit_folds.sh config.yaml
```

batch sizeはglobal 128（2 GPU時は64/GPU）。batch size 16での実行ではA100 40 GBの
使用量が約1.8 GiB/GPUだったため拡大した。OOM時は64へ下げる。

## Acceptance criteria

- exp016の同一foldよりtile RMSEが改善する
- wet-mask IoUと空間相関が改善し、wet area ratioが極端に増えない
- `highres_only -> highres_mask -> full` の比較で改善要因を分離できる
