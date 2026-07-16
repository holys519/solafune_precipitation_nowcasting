# g_eda/exp002: オラクル分解ラダー (E-1)

`doc/plan/round5_experiment_plan_2026-07-16.md` のE-1。exp016/017/018のOOF予測を
checkpointから再生成し、残误差を「量の誤り」と「配置の誤り」へ分解する。

各タイルで以下の反実仮想 tile RMSE を計測する:

| 指標 | 意味 |
| --- | --- |
| `actual` | 実際のOOFスコア |
| `flat_pred` | 予測をタイル平均で均す — 我々の「量」情報だけの得点 |
| `flat_truth` | 正解タイル平均 — 「0.677の壁」の再現 |
| `mask_oracle_flat` | 壁 + 完璧なwet/dryマスク (正解のみから計算) |
| `amount_swap` | タイル総量を正解に合わせた我々の空間パターン — **配置品質の直接測定** |
| `mask_swap` | 我々の総量を正解マスク上へフラット配置 — **量品質の直接測定** |
| `blur_pred_s1/s2` | 予測のガウシアンぼかし — 過剰シャープネスの罰の有無 |

`amount_swap` と `mask_swap` の比較が次の投資先 (配置 vs 量) を決める。
`blur_pred_*` が `actual` を下回るなら後処理ぼかしが即効の勝ち筋。
出力はglobal/fold/衛星別 + wet-tile限定の集計。

## Run

```bash
cd /group/project143/yamamoto/solafune_precipitation_nowcasting/g_eda/exp002
sbatch singularity_run.sh                 # exp016 exp017 exp018
sbatch singularity_run.sh exp018          # 単体
```

出力: `outputs/g_eda/exp002/{exp}_oracle_ladder.{csv,json}`

## 判断基準

- `actual − amount_swap` が大 → 量 (calibration/intensity head) を先に直す
- `actual − mask_swap` が大 → 配置 (localization) を先に直す — 壁の下0.083の回収
- Meteosatのwet集計は特に確認 (exp018のwet IoU 0.186と整合するはず)
