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
sbatch singularity_run.sh config.yaml all_submit
```

128x128処理はメモリ使用量が大きいため初期batch sizeは16。OOM時は8へ下げる。

## Acceptance criteria

- exp016の同一foldよりtile RMSEが改善する
- wet-mask IoUと空間相関が改善し、wet area ratioが極端に増えない
- `highres_only -> highres_mask -> full` の比較で改善要因を分離できる
