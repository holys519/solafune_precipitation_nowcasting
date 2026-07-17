# g_eda/exp002: オラクル分解ラダー (E-1)

`doc/plan/round5_experiment_plan_2026-07-16.md` のE-1。exp016/017/018のOOF予測を
checkpointから再生成し、量・support・shapeに関する反実仮oracleラダーを計測する。

> **解釈上の注意:** この実験は反実仮想oracleラダーであり、`amount_swap`と`mask_swap`は
> 量・配置を独立に直接測定するものではない。scale変更は空間残差の振幅も変え、mask置換は
> supportとshapeを同時に変える。`amount_swap`はmodel内のamount headを置き換えるものでもない。
> 厳密な加法mean–shape分解とfinal-prediction componentのcross-swapは
> [`../exp006/README.md`](../exp006/README.md)を参照すること。これらがtargetを条件として示すのは
> oracle opportunityであり、deployableなcalibratorやmodelの期待効果ではない。

各タイルで以下の反実仮想 tile RMSE を計測する:

| 指標 | 意味 |
| --- | --- |
| `actual` | 実際のOOFスコア |
| `flat_pred` | 予測をそのtile meanで均一にしたcounterfactual |
| `flat_truth` | 正解タイル平均 — 「0.677の壁」の再現 |
| `mask_oracle_flat` | 壁 + 完璧なwet/dryマスク (正解のみから計算) |
| `amount_swap` | タイル総量を正解に合わせた予測空間パターン — scale oracle診断 |
| `mask_swap` | 予測総量を正解mask上へフラット配置 — support置換oracle診断 |
| `blur_pred_s1/s2` | 予測のガウシアンぼかし — 過剰シャープネスの罰の有無 |

`amount_swap` と `mask_swap` の比較は仮説生成に限定し、単独で次の投資先を決めない。
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

- `actual − amount_swap` が大 → scale/calibration仮説をexp006の厳密分解で再検証
- `actual − mask_swap` が大 → support/localization仮説をcross-swapで再検証。pairは
  leave-one-fold-out内側で選択してheld-out foldで評価し、その後outer-locationで確認する。
  全OOF選択後のfold winsは採択根拠にしない
- Meteosatのwet集計は特に確認 (exp018のwet IoU 0.186と整合するはず)
