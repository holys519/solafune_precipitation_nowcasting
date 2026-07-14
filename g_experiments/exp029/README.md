# exp029: Satellite-Aware IR Rain Proxy

exp017へ各フレーム1枚のIR-window cold-cloud proxyを追加する入力特徴ablation。センサー別の
raw値anchorで0〜1へ写像し、共通の温度応答を仮定しない。これは経験曲線の第一段階であり、
OOF改善後にのみtrainから学習したbin曲線へ置換する。

```bash
sbatch singularity_run.sh 0
sbatch singularity_run.sh all_submit  # 5-fold学習→推論→submission zip
```

exp017の同一foldよりtile RMSEとpositive RMSEがともに悪化しないことを採用条件とする。
