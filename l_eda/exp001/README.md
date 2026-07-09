# l_eda exp001: Image-Processing EDA

画像処理ライブラリ(OpenCVがあればOpenCV、無ければNumPy fallback)を使い、モデルの改善仮説を作るための
基礎研究系EDAです。既存の数表EDAでは見えにくい「雨域の形」「雲と雨のずれ」「時系列の動き」
「train/evaluationの画像特徴差」をCSV/PNG/Markdownに落とします。

## Analyses

| Analysis | Purpose | Main output |
| --- | --- | --- |
| target morphology | 雨域の面積、連結成分、細長さ、ピークを調べる | `target_morphology.csv` |
| parallax shift | IR cold-cloud画像とGPM雨域の相関が最大になる固定shiftをlocation別に推定 | `parallax_shift_by_location.csv` |
| temporal motion | 直近3フレームのphase correlationから雲の移動量を見る | `temporal_motion.csv` |
| spectral texture | band平均/分散、Sobel/Laplacianなどの画像特徴と雨統計の相関を見る | `spectral_texture_correlations.csv` |
| train/eval shift | evaluation locationの衛星画像特徴がtrainのどのlocation/satelliteに近いかを見る | `train_eval_feature_shift.csv` |

## Run

```bash
cd l_eda/exp001
bash run.sh
```

プロジェクトルートから直接実行する場合:

```bash
uv run python l_eda/exp001/run_image_eda.py --config l_eda/exp001/config.yaml
```

軽く試す場合は `config.yaml` の `sampling.max_train_rows` / `max_eval_rows` を下げてください。

## Outputs

| Path | Description |
| --- | --- |
| `../../outputs/l_eda/exp001/EDA_IMAGE_REPORT.md` | 要約レポート |
| `../../outputs/l_eda/exp001/*.csv` | 各解析の表 |
| `../../outputs/l_eda/exp001/figures/*.png` | 分布図と代表例 |

## Follow-Up Ideas

- parallax shiftがlocationごとに安定するなら、入力画像または予測をlocation/satellite別に数pixel補正。
- temporal motionが小さいなら、重いConvLSTMよりsuccessor-frame利用や時系列smoothingを優先。
- morphologyでheavy rainの形が支配的なら、連結成分/edge/texture補助lossやpostprocessを検討。
- train/eval shiftが大きいlocationでは、衛星別・地域別のscaleやthresholdをOOFで再評価。
