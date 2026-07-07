# exp001: GPU Compact UNet Baseline

## Purpose

RTX 3090 x2 で動く、前処理・モデル学習・推論・提出zip作成までの最小GPUベースラインです。モデルは `../../l_model/exp001/best_model.pt` に保存します。

## Method

1. 最大3観測を古い順に3スロットへ配置する。
2. 各衛星GeoTIFFを16ch tensorへそろえる。
   - channel不足は0 padding。
   - channel過多は先頭16chを使用。
   - GOESの `282x282x4` も41x41へresizeして処理する。
3. 各観測を `41x41` へresizeし、`3 * 16 = 48ch` として結合する。
4. 観測mask 3ch と衛星one-hot 3ch を足し、合計54ch入力にする。
5. Compact UNetで `41x41` のGPM-IMERG降水量を回帰する。
6. 推論時は非負clipし、evaluation template TIFFのメタデータを保って書き出す。

## Run

GPUアクセスが必要なので、サンドボックス外で実行する。

```bash
cd l_experiments/exp001
../../.venv/bin/python train.py
../../.venv/bin/python inference.py
../../.venv/bin/python make_submission.py
```

## Outputs

| Path | Description |
| --- | --- |
| `../../l_model/exp001/best_model.pt` | best validation model |
| `../../l_model/exp001/metrics.json` | validation metrics and training history |
| `../../outputs/submissions/exp001/` | expanded submission directory |
| `../../outputs/submissions/exp001_submission.zip` | submission zip |

