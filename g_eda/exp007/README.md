# exp007 — strict CV→Public transfer audit

exp038 の strict/green Public 結果を受け、既存OOFで次を監査するCPU解析です。

- exp038 と比較対象の sample-paired OOF差（fold / satellite / location / target amount）
- satellite内 location-cluster bootstrap
- evaluation の satellite / 6時間帯 / 観測枚数分布へのOOF再重み付け
- evaluation予測のtarget-freeな強度統計
- CVと既提出Publicのpost-hoc比較

Public値は重み学習やモデル選択に使いません。evaluation targetや予測ラスタも読みません。

## Run

```bash
python3 g_eda/exp007/run_strict_transfer_audit.py
```

出力は `outputs/g_eda/exp007/` に保存されます。主結果は
`TRANSFER_AUDIT.md`、詳細値は隣接CSVです。

## Decision rule

- 大差の妥当性確認にはCVを使う。
- 0.003前後の接戦は単一集約値で順位付けせず、fold 0/4の同方向改善をscreening gateにする。
- exp038 Publicを見て求めた振幅係数は追加しない。
