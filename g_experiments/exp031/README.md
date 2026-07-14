# exp031: Focal + Tversky Rain Auxiliary Loss

exp017の回帰lossを保持し、rain occurrence headへ低ウェイトのFocal lossとTversky lossを
追加する。`focal_weight=tversky_weight=0.05`で、RMSE最適な回帰を壊さない範囲から開始する。

```bash
sbatch singularity_run.sh 0
sbatch singularity_run.sh all_submit  # 5-fold学習→推論→submission zip
```

tile RMSE、positive RMSEを主指標とし、OOF解析時にwet-mask precision/recallも確認する。
