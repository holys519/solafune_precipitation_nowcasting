# exp034: OOF Rain-Threshold Inference + Blend

新規学習なしで、exp016/017/018のOOF rain-probability thresholdを評価推論へ反映する。
既存submissionは全て`rain_prob_threshold=0.0`、`value_threshold=0.10`だったため、
OOF sweepの最良設定はまだPublic LBで検証していない。

## Fixed post-processing

| Model | rain probability threshold | OOF raw tile RMSE | OOF thresholded tile RMSE |
| --- | ---: | ---: | ---: |
| exp016 | 0.25 | 0.618607 | 0.617839 |
| exp017 | 0.70 | 0.616285 | 0.615041 |
| exp018 | 0.40 | 0.609261 | 0.608227 |

`value_threshold=0.0`とし、OOF sweepと同じ条件にする。linear/isotonic calibrationと
temporal smoothingは使用しない。

## Flow

1. 学習済み5-fold checkpointで3モデルの評価推論を再実行。
2. thresholded exp016/017を50/50でbaseにする。
3. thresholded exp018を0%、25%、50%、100%混ぜる。
4. blend後にexp014 overlap patchを適用する。

## Run

```bash
cd /group/project143/yamamoto/solafune_precipitation_nowcasting/g_experiments/exp034
sbatch singularity_run.sh
```

推論済み出力からblendだけやり直す場合:

```bash
sbatch singularity_run.sh --skip-inference
```

source/checkpoint確認のみ:

```bash
sbatch singularity_run.sh --dry-run
```

## Outputs

- `outputs/submissions/exp034_thr_w018_000_patched.zip`
- `outputs/submissions/exp034_thr_w018_025_patched.zip`
- `outputs/submissions/exp034_thr_w018_050_patched.zip`
- `outputs/submissions/exp034_thr_w018_100_patched.zip`
- manifest: `outputs/analysis/exp034/analysis_summary.json`

## Adaptive submission order

exp026をunthresholded・exp018重み0%のanchorとして、次の順に比較する。

1. `exp034_thr_w018_000_patched.zip` — threshold効果だけ。
2. `exp033_w018_050_patched.zip` — blend効果だけ。

最初の2結果を見て残り2枠を選ぶ。

| threshold-only | blend-only | 残り2枠 |
| --- | --- | --- |
| 改善 | 改善 | `exp034 thr_w018_025`, `thr_w018_050` |
| 改善 | 悪化 | `exp034 thr_w018_025`, `thr_w018_100` |
| 悪化 | 改善 | `exp033 w018_025`, `w018_075` |
| 悪化 | 悪化 | `exp033 w018_100`を診断提出し、最後の1枠は温存またはraw規約安全候補 |

この分岐に備え、exp033は既定の4重みをすべて生成しておく。

## Rule note

overlap patchは公式許可を別途確認する。未patch zipが必要な場合は
`sbatch singularity_run.sh --skip-patch --zip-raw`を使用する。
