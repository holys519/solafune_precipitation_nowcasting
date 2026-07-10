# l_eda

ローカル環境で `uv run` 経由で回すEDA・基礎研究系解析を置きます。`eda/` は軽量な全体把握、
`l_eda/` は画像処理・時系列・物理仮説検証など、モデル改善に直結する材料をローカルで確認する用途です。

## Experiments

| Exp | Theme | Status |
| --- | --- | --- |
| exp001 | OpenCV/NumPy image-processing EDA: morphology, parallax, motion, texture, train/eval shift | Ready |
| exp002 | Pixel-level EDA round 2: tile adjacency/overlap (→ `doc/tile_overlap_discovery.md`), IR→rain lag, GPM autocorr, BT-rain response, band health, quantization, spectra, OOF day/night join | Run — 3 eval↔train tile overlaps confirmed |

## Run

```bash
cd l_eda/exp001
bash run.sh
```

またはプロジェクトルートから直接実行できます。

```bash
uv run python l_eda/exp001/run_image_eda.py --config l_eda/exp001/config.yaml
```
