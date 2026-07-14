# exp028: Target-Time-First + Absolute Change

exp017を対照に、successor rowの最新観測を入力先頭へ移し、各context rowへIR-windowの
`|newest-oldest|`を1チャネル追加する時間設計ablation。他のモデル・lossは固定する。

```bash
sbatch singularity_run.sh 0   # まずfold 0
sbatch singularity_run.sh all_submit  # 5-fold学習→推論→submission zip
bash run.sh 0                 # GPU環境で直接
```

exp017の同一foldよりtile RMSEが改善した場合のみ他foldへ進める。
