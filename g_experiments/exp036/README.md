# exp036: OOF最適重みブレンド + Overlap Patch

exp033の固定比ハシゴの後継。`exp033_w018_050_patched` がPublic **0.6719899228** (新ベスト、
exp026比 −0.0027) となったため、残り枠を手探りに使わず `g_eda/exp003` のOOF simplex探索で
求めた最適重みを直接提出する、学習不要の後処理実験。

## 前提

1. `g_eda/exp003` 完了 → `outputs/g_eda/exp003/recommended_weights.json`
2. eval予測: `outputs/submissions/{exp016,exp017,exp018}/test_files/` (生成済み)

## Run

```bash
cd /group/project143/yamamoto/solafune_precipitation_nowcasting/g_experiments/exp036
sbatch singularity_run.sh --dry-run                    # 重みとsource確認
sbatch singularity_run.sh                              # global_best重みで blend→patch→zip
sbatch singularity_run.sh --scheme per_satellite       # 衛星別最適重み
sbatch singularity_run.sh --weights 0.25,0.25,0.50     # 手動重み (w016,w017,w018)
```

規約安全track用のraw zipは `--skip-patch --zip-raw`。

## Outputs

- `outputs/submissions/exp036_<scheme>_patched.zip`
- manifest: `outputs/analysis/exp036/analysis_summary_<scheme>.json`

## 提出判断

- OOFの `global_best` が `ladder w018=0.5` を **>0.004** (E-3ノイズ閾値) 上回る場合のみ
  exp036を提出。差がそれ未満ならexp033の残り (`w018_075`) で曲線右側を確認する方が情報量が多い。
- `per_satellite` は `per_satellite_composed` のOOFが `global_best` を >0.004 上回る場合のみ。
- blur / value threshold の最良設定 (`recommended_weights.json`) が有意なら、別途
  変換を追加した候補を検討 (現状run.pyは重みブレンドのみ。blurが効けば拡張する)。

## Rule note

overlap patchの規約上の扱いはexp033と同じ: 公式許可が確認できない限り最終提出には
raw zipを使う。
