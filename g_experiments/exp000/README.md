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
cd g_experiments/exp000
bash download.sh
sbatch singularity_run.sh
```

このスクリプトは認証情報を使いません。zipが未配置の場合は、配置先だけ表示して終了します。
`data/raw/*.zip` だけでなく、`data/` 直下に置いたSolafune配布元ファイル名のzipも自動検出します。
再展開したい場合は `FORCE_UNZIP=1 bash download.sh` を使います。
