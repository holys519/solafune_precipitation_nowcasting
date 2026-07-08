# exp001: Cloud Compact UNet Baseline

`l_experiments/exp001` をクラウド/Singularity実行向けに移植した最小GPUベースラインです。
入力は最大3観測 x 16バンド、観測mask 3ch、衛星one-hot 3ch の合計54chです。

## Run

```bash
cd g_experiments/exp001
sbatch singularity_run.sh        # unzip/check -> train
sbatch singularity_run.sh check  # unzip/check only
sbatch singularity_run.sh all    # train -> inference -> submission zip
```

`CONTAINER_FOLDER` と `CONTAINER_NAME` は環境変数で上書きできます。デフォルトは
`/group/project143/common/containers/kaggle-gpu-images-python-v163.sif` です。

## Data

ジョブ開始時に `../exp000/download.sh` を呼び出します。
zipは以下のどちらでも読めます。

- `data/raw/train_dataset.zip`, `data/raw/evaluation_dataset.zip`, `data/raw/sample_submission.zip`
- `data/` 直下のSolafune配布元ファイル名のzip

## Outputs

| Path | Description |
| --- | --- |
| `../../g_model/exp001/best_model.pt` | best validation model |
| `../../g_model/exp001/metrics.json` | validation metrics and training history |
| `../../outputs/submissions/exp001/` | expanded submission directory |
| `../../outputs/submissions/exp001_submission.zip` | submission zip |
