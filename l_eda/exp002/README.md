# l_eda/exp002: Pixel-Level EDA Round 2

`doc/data_characteristics_review.md` のA-trackチケットと過去コンペ/論文の知見
(Weather4cast, MetNet, DGMR, PERSIANN系IR-QPE, PySTEPS検証手法) に基づく
ピクセルレベル分析スイート。各分析は `config.yaml` の `analyses.*` で個別にon/off可能。

## Run

```bash
cd l_eda/exp002
bash run.sh          # 全分析 (~40秒)
```

出力: `outputs/l_eda/exp002/` (CSV群 + figures/ + PIXEL_EDA_REPORT.md)

## Analyses

| 分析 | 答える問い | 主な結果 (2026-07-10) |
| --- | --- | --- |
| train_eval_adjacency | eval/trainタイルは空間重複するか | **3ペアで重複確定** → `doc/tile_overlap_discovery.md` (G-022) |
| ir_rain_lag | 何分先/前のIRが雨と最も相関するか | **+30分がピーク** (goes 0.428 vs 0分 0.419)、後方は急減衰 → successor重視 (G-024) |
| target_autocorr | GPMの時間自己相関と持続性スキル | lag1(30分) corr 0.561 / persistence skill 0.315、lag4以降スキル負 |
| bt_rain_response | E[rain\|IR値] 曲線 | 最冷ビンで mean_rain 5.78 / P(rain) 0.94 の強い単調非線形 |
| band_health | バンド別ゼロ/飽和率の時刻依存 | 可視バンドは現地夜間にstd崩壊 (band5夜間zero率62%)、IRは安定 → solar特徴の根拠 |
| target_quantization | GPM値は量子化されているか | **99.6%が0.01の倍数**、min positive 0.01 → スナップ後処理候補 (G-025) |
| spectrum | 信号はどの空間スケールにあるか | wavenumber 1-2 (20-41px) にpowerの~81%が集中 |
| position_bias | タイル内位置の気候バイアス | center/edge比 1.07 — 無視可能 |
| oof_join | 夜間誤差はセンサー起因か対流起因か | **同一rain regimeで暗タイル誤差 +16%** (mid) → センサー情報損失が主因 (G-023) |

## Notes

- `train_eval_feature_shift.csv` は l_eda/exp001 の出力のコピー (adjacencyのペア候補表)。
- 相互相関はオーバーラップ正規化FFT実装 (`normalized_xcorr_peak`)。合成テストで
  既知オフセットの完全復元を確認済み。分散フロアは重複サイズ比例でスケール
  (小重複の偽ピーク対策)。
- `oof_join` は `outputs/analysis/exp009/oof_sample_metrics.csv` が無い環境では
  自動スキップ。
