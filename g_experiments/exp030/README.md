# exp030: Dilated Bottleneck

exp017のCompact U-Net bottleneckをdilation 2、4の連続blockへ置換し、解像度をさらに
落とさず受容野を拡大する空間context ablation。追加メモリを考慮してbatch sizeは96。

```bash
sbatch singularity_run.sh 0
sbatch singularity_run.sh all_submit  # 5-fold学習→推論→submission zip
```

同一foldのtile RMSEに加え、局在化改善の兆候としてpositive RMSEを比較する。
