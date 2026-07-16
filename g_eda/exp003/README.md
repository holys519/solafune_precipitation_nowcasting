# g_eda/exp003: OOFブレンド重み最適化

`exp033_w018_050_patched` がPublic 0.6719899228 (exp026比 −0.0027) で新ベストになったことを
受け、残りの提出枠をハシゴの手探りに使わず、**OOFで最適な混合を直接求める**考察実験。

exp016/017/018のOOF予測をcheckpointから再生成してnpzにキャッシュし (再利用可)、以下を計算:

1. exp033ハシゴと同型の2-way曲線: `(1-w)·equal(016,017) + w·exp018`, w=0〜1 (0.05刻み)
2. 3-way simplex全格子 (0.05刻み): 全体最適トリプルと**衛星別最適トリプル**
3. 最適ブレンドへのgaussian blurスイープ (E-1でσ=1が微益)
4. 最適ブレンドへのvalue thresholdスイープ

## Run

```bash
cd /group/project143/yamamoto/solafune_precipitation_nowcasting/g_eda/exp003
sbatch singularity_run.sh              # キャッシュ生成 (GPU) → 解析
sbatch singularity_run.sh --analyze    # 解析のみ再実行
```

出力: `outputs/g_eda/exp003/` — `{exp}_oof_pred.npz` (fp16キャッシュ、~140MB/model)、
`blend_curve.csv`、`simplex_grid.csv`、`blur_sweep.csv`、`threshold_sweep.csv`、
`BLEND_CURVE.md`、`recommended_weights.json` (**exp036が読む**)。

## 使い方

- `recommended_weights.json` の `global_best` / `per_satellite_best` を
  `g_experiments/exp036` に渡してeval側ブレンド+patch+zipを生成する。
- OOF曲線とLB実測 (w=0: 0.67465 / w=0.5: 0.67199) の対応も確認し、OOF→LBの
  単調性がブレンド軸でも成り立つかを`doc/public_scores.md`に記録する。
