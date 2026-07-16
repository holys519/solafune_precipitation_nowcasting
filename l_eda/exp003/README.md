# l_eda/exp003: CV → Public LB 回帰校正 (E-3)

`doc/plan/round5_experiment_plan_2026-07-16.md` のE-3。提出済み実験のOOF指標とPublic RMSEを
突き合わせ、どのCV指標が採択判断に使えるかを実測する。stdlibのみで動く。

```bash
python3 l_eda/exp003/run_cv_lb_calibration.py
```

出力: `outputs/l_eda/exp003/{cv_lb_pairs.csv, CV_LB_CALIBRATION.md, calibration_stats.json}`

## 結果 (2026-07-16, 9ペア)

| 予測子 | Spearman | 残差std |
| --- | ---: | ---: |
| 5-fold OOF tile_rmse | **0.900** | **0.0033** |
| 衛星構成重み付きOOF | 0.900 | 0.0032 |
| fold0のみ | 0.500 | 0.0106 |
| fold0+fold4 | 0.783 | 0.0079 |

## 結論 (Round 5 プロトコルへ反映)

1. **fold0単独A/Bはほぼ無情報** (Spearman 0.50)。exp028-032のfold0採否判定は弱い証拠であり、
   最終判断は必ず5-fold OOFで行う。
2. fold0+4の2-foldスクリーニングは「学習が回るか+大まかな方向」の確認用途に留める。
3. 5-fold OOFの残差stdは0.0033 → **OOF差 <0.005 はLB上で信頼できない**
   (ディスカッションの経験則と一致)。
4. 衛星構成重み付けは現ペアでは素のOOFと同等。優先度を下げる。

新しい提出のたびに `PUBLIC_SCORES` へ (exp, public_rmse) を追記して再実行する。
