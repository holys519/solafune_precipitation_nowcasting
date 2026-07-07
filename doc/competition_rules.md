# Competition Rules Notes

## External Data

外部データセットは禁止。使用できるものは、Solafuneが配布するデータと、そのデータから作成した特徴量・モデル入力に限定します。

事前学習済みモデルの扱いは、Solafuneの参加規約とコンペページで確認します。判断が曖昧な場合は、利用前にディスカッションまたは運営回答を確認します。

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
