# Competition Rules Notes

## External Data

外部データセットは禁止。使用できるものは、Solafuneが配布するデータと、そのデータから作成した特徴量・モデル入力に限定します。

### Working Assumption

2026-07-09時点の運用方針:

- Solafuneから配布されたtrain/evaluation CSV、衛星GeoTIFF、train target、sample/evaluation
  template、およびそれらから作成した特徴量・モデル入力・後処理特徴は原則として利用可とみなす。
- evaluation内の同一location隣接行、時系列文脈、モデル予測同士の時間平滑化も、配布データと
  自前モデル出力だけで完結する限りは利用可として実験する。
- 明らかな外部データセット、外部気象データ、外部地理座標、商用/利用不可ライセンスの重みは使わない。
- 運営から追加の明示的な禁止・確認回答が出た場合は、その回答を優先して実験方針を更新する。

事前学習済みモデルの扱いは、Solafuneの参加規約とコンペページに記載された許可ライセンスに従う。
利用した場合は出典を明記する。

## Submission Format

```text
submission.zip
├── evaluation_target.csv
└── test_files/
    ├── {location}_GPM_IMERG_{datetime}.tif
    └── ...
```

## Reproducibility

入賞候補者は、コンペ終了後にソースコードと解法記事URLの提出、および本人確認が求められます。実験ごとに以下を残します。

- 実験IDとGit差分
- データ前処理
- CV分割
- 学習設定
- 推論設定
- 後処理
- 提出zipの作成手順
