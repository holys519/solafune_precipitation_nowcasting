# exp032: Satellite-Conditional Heads

空間encoder/decoderはexp017と共有し、GOES・Himawari・Meteosatごとにoccurrence headと
amount headだけを分岐する。完全別モデルより過学習を抑えつつ衛星固有biasを吸収する。

```bash
sbatch singularity_run.sh 0
sbatch singularity_run.sh all_submit  # 5-fold学習→推論→submission zip
```

全体tile RMSEだけでなく衛星別OOFを確認し、少数衛星だけの改善でないことを採用条件とする。
