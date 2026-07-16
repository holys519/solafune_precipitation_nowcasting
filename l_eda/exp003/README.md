# l_eda/exp003: CV → Public LB 回帰校正 (E-3)

`doc/plan/round5_experiment_plan_2026-07-16.md` のE-3。提出済み実験のOOF指標とPublic RMSEを
突き合わせ、どのCV指標が採択判断に使えるかを実測する。stdlibのみで動く。

```bash
python3 l_eda/exp003/run_cv_lb_calibration.py
```

出力: `outputs/l_eda/exp003/{cv_lb_pairs.csv, CV_LB_CALIBRATION.md, calibration_stats.json}`

## 結果 (2026-07-16, 12ペア — exp016/017/018単体スコア追加後)

| 予測子 | Spearman | 残差std |
| --- | ---: | ---: |
| 5-fold OOF tile_rmse | **0.951** (modelのみ 0.964) | **0.0041** |
| 衛星構成重み付きOOF | 0.951 | 0.0040 |
| fold0のみ | 0.790 | 0.0096 |
| fold0+fold4 | 0.909 | 0.0078 |

(9ペア時点: OOF 0.900/0.0033、fold0単独 0.500 — 新3ペアでfold0も持ち直したが序列は不変)

## 結論 (Round 5 プロトコルへ反映)

1. **5-fold OOF tile_rmseが最良の採択指標** (Spearman 0.95)。実例: exp018−exp017は
   OOF −0.0070 → LB −0.0068 とほぼ一致。
2. fold0単独A/Bは方向確認まで。fold0+4の2-foldスクリーニング (0.91) は実用的な中間。
3. 残差stdは0.0041 → **OOF差 <0.005 はLB上で信頼できない**。実例: exp016/017は
   OOF差0.0023でLBが逆転 (0.6978 vs 0.6997)。
4. 衛星構成重み付けは素のOOFと同等。優先度を下げる。

新しい提出のたびに `PUBLIC_SCORES` へ (exp, public_rmse) を追記して再実行する。
