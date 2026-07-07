# exp000: Data Preparation

手動で配置したSolafune配布zipを展開し、初期のファイル構成を確認します。

## Expected Input

```text
data/raw/
├── train_dataset.zip
├── evaluation_dataset.zip
└── sample_submission.zip
```

## Run

```bash
cd l_experiments/exp000
bash download.sh
```

このスクリプトは認証情報を使いません。zipが未配置の場合は、配置先だけ表示して終了します。
