# g_eda

GPU/HPC環境で回す重めのEDA・基礎研究系解析を置きます。`eda/` は軽量な全体把握、
`g_eda/` は画像処理・時系列・物理仮説検証など、時間がかかってもモデル改善に直結する
材料を作る用途です。

## Experiments

| Exp | Theme | Status |
| --- | --- | --- |
| exp001 | OpenCV/NumPy image-processing EDA: morphology, parallax, motion, texture, train/eval shift | Ready |

## Run

```bash
cd g_eda/exp001
sbatch singularity_run.sh
```
