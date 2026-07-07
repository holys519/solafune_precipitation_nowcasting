# l_experiments

ローカル開発・小規模検証用の実験を格納します。データ確認、1 fold学習、推論スクリプトの動作確認をここで先に行います。

## Usage

```bash
cd l_experiments/exp001
python train.py --config config.yaml --fold 0
```

## Conventions

- 実験番号は `g_experiments/` と揃える。
- 小さな解像度や少数サンプルで素早く動作確認する。
- 実験が有望なら同じ番号で `g_experiments/` に反映する。
- 学習済み重みは `../l_model/` に保存し、Gitには含めない。
