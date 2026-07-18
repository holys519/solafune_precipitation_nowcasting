# g_eda

GPU/HPC環境で回す重めのEDA・基礎研究系解析を置きます。`eda/` は軽量な全体把握、
`g_eda/` は画像処理・時系列・物理仮説検証など、時間がかかってもモデル改善に直結する
材料を作る用途です。

## Experiments

| Exp | Theme | Status |
| --- | --- | --- |
| exp001 | OpenCV/NumPy image-processing EDA: morphology, parallax, motion, texture, train/eval shift | Ready |
| exp002 | OOF oracle ladder: flat/mean-scale/mask/blur diagnostics | Run; historical interpretation requires exp006 |
| exp003 | OOF blend/simplex/blur/threshold optimization | Run |
| exp004 | temporal prediction smoothing and advected-smoothing tests | Run |
| exp005 | IMERG innovation/parallax exploratory diagnostics | Run; external-constant arms are amber |
| exp006 | exact additive/multiplicative factorization, cross-swap, metric stress, availability | Ready |

## Run

```bash
cd g_eda/exp001
sbatch singularity_run.sh
```
