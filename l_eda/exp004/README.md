# l_eda/exp004: fold分散の解剖 (E-4)

`doc/plan/round5_experiment_plan_2026-07-16.md` のE-4。exp018のOOFタイル指標を
regime (タイルtarget_mean) ビンで層別し、fold間の3倍のtile_rmse差の正体を特定する。

```bash
python3 l_eda/exp004/run_fold_anatomy.py
```

出力: `outputs/l_eda/exp004/{FOLD_ANATOMY.md, location_anatomy.csv}`

## 結果 (2026-07-17)

**fold分散はほぼ完全にregime構成で説明できる。**

| fold | 実測tile_rmse | fold0のregime構成で固定した場合 |
| ---: | ---: | ---: |
| 0 | 0.292 | 0.292 |
| 1 | 0.752 | **0.337** |
| 2 | 0.658 | **0.304** |
| 3 | 0.791 | **0.299** |
| 4 | 0.585 | **0.302** |

within-regimeの誤差はfold間でほぼ一定 (例: regime r3 [0.3-1.0 mm/hr] は全fold 1.19-1.30)。
fold1/3が「悪い」のはモデルが劣るからではなく、雨の多いタイル構成だから。

## 帰結

1. **regime-fixed-mix指標をA/B判定の補助に使う**: fold間比較のノイズが大幅に減る
   (0.29-0.79 → 0.29-0.34)。単fold/2-fold screeningの信頼性が上がる。
2. heavy-rain regime特化の改善 (exp016のtail設計等) の優先度は据え置き — どのfoldでも
   r3/r4の誤差は支配的だが、fold固有の問題ではない。
3. OOF→LBギャップ (~0.06) の一部はevalのregime構成差の可能性。evalの構成は予測tile mean
   で代理推定できる (未実施、優先度中)。
