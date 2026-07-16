# exp033: exp018 Blend Ladder + Overlap Patch

LB best `exp026`をanchorとしてexp018の混合比を探索する、学習不要の後処理実験。
4枠すべてをこのladderへ使う初期案だったが、OOF rain thresholdを反映する`exp034`を
追加したため、現在の推奨4提出は`exp034/README.md`を参照する。本実験からは原則として
`w018_050`を1候補だけ提出し、thresholdなしblendの効果を分離する。

`exp026`は、patch前の`exp024/equal_016_017`へexp014 overlap patchを適用したものなので、
exp018を混ぜた後に同じpatchを適用する。patchを先に適用してからblendすると既知領域の値が
薄まるため、処理順序は必ず `raw blend -> patch` とする。

## Candidates

| Submission | exp024 equal 016/017 | exp018 | 用途 |
| --- | ---: | ---: | --- |
| `exp033_w018_025_patched.zip` | 0.75 | 0.25 | 保守的な本命 |
| `exp033_w018_050_patched.zip` | 0.50 | 0.50 | 中央点 |
| `exp033_w018_075_patched.zip` | 0.25 | 0.75 | exp018寄り |
| `exp033_w018_100_patched.zip` | 0.00 | 1.00 | exp018単体のpatch効果診断 |

exp018重み0.00のanchorは提出済み`exp026_submission.zip`、Public RMSE
`0.6746506841387548`。上記4点を追加すると、0.00から1.00まで0.25刻みの曲線が得られる。

## Run

GPU推論や再学習は不要。既存の29,090枚の予測をCPUでblendしてzip化する。

```bash
cd /group/project143/yamamoto/solafune_precipitation_nowcasting/g_experiments/exp033
sbatch singularity_run.sh
```

投入前のsource確認だけを行う場合:

```bash
sbatch singularity_run.sh --dry-run
```

overlap patchを使用しないraw submission zipも必要な場合:

```bash
sbatch singularity_run.sh --skip-patch --zip-raw
```

通常実行でも未patch予測は`outputs/submissions/exp033/*_raw/test_files/`へ保持する。
`--zip-raw`を指定した場合だけ、追加でraw zipを作成する。

## Outputs

- primary zips: `outputs/submissions/exp033_w018_{025,050,075,100}_patched.zip`
- raw prediction directories: `outputs/submissions/exp033/w018_*_raw/`
- patched directories: `outputs/submissions/exp033/w018_*_patched/`
- manifest and SHA-256: `outputs/analysis/exp033/analysis_summary.json`
- run log: `outputs/analysis/exp033/run.log`

## Submission use

本日の第一候補は`exp033_w018_050_patched.zip`。残りの3枠はexp034のthresholded候補へ
使用する。exp034が期限までに完成しない場合のみ、`025 -> 075 -> 100`を予備候補として使う。
Public RMSEはREADMEまたは`doc/public_scores.md`へ記録する。

## Rule note

overlap patchは、重複するtrain tileのGPM targetを評価tileの一部へコピーする。
submission画面の`valid`は手法の最終審査を意味しないため、公式に許可されていない場合は
patched zipを最終提出へ使用しない。規約安全トラックでは`--skip-patch --zip-raw`で作成した
raw zipを使用する。
