# exp040: Tile Mean-Intensity × Normalized-Shape Factorization (survey v3 P0)

`doc/research_survey_v3_2026-07-16.md` §5.1の最優先仮説（"最も高い勝率候補"）。
これまでのhurdle log-normal系ヘッド（P(rain)×exp(μ+σ²/2)）を、以下の決定論的分解へ置き換える:

```text
m       = 非負のタイル平均強度 (global-pooled decoder featuresからMLPで出力、スカラー)
z       = 非負の空間活性化 g(S) (decoder featuresから畳み込みヘッドで出力、空間map)
s       = z / clamp_min(mean_pixels(z), eps)   # mean_pixels(s) ≈ 1
y_hat   = m * s                                 # mean_pixels(y_hat) ≈ m (恒等式)
```

`g_eda/exp006`の厳密分解監査で、true-meanスケーリングがL2スケールoracleの86-88%を回収
（tile −0.065のヘッドルーム）と分かっている。intensity headに面平均強度の責任を完全に
集中させ、shape headをスケールから解放して局在化に専念させる設計。

exp038のstrict green base (54ch, context_rows=1, no successor row) の上に構築。
architectureは`highres_mean_shape_unet` (exp040/model.py)。

## Arms

| Config | 内容 | 
| --- | --- |
| `config.yaml` (Arm C) | mean/shape分離供給のみ (`field_weight=0, metric_weight=0`)。survey表現の"完全分離"版 |
| `config_metric.yaml` (Arm D) | Arm C + `metric_weight=0.3` の metric-aligned loss (`mean_batch(sqrt(mean_pixels(err²)+eps))`)。identified評価器 (per-file平均RMSE, l_eda/exp003) に直接整合するshaping項 |

両アームとも `mean_weight=1.0, shape_weight=1.0, bce_weight=0.1 (診断用auxiliary、predには
非乗算), aux_mask_weight=0.2, multiscale_weight_2/4` は既存exp018/038系と同じ値を流用。

## 恒等式の検証

`smoke_test.py`は通常のforward/backward検証に加え、mean-shape分解の恒等式を確認する:
- `mean_pixels(shape) ≈ 1`
- `mean_pixels(pred) ≈ mean_intensity`

## Run

```bash
cd /group/project143/yamamoto/solafune_precipitation_nowcasting/g_experiments/exp040
sbatch singularity_smoke.sh
sbatch singularity_run.sh config.yaml 0          # Arm C fold0
sbatch singularity_run.sh config.yaml 4          # Arm C fold4
sbatch singularity_run.sh config_metric.yaml 0   # Arm D fold0
sbatch singularity_run.sh config_metric.yaml 4   # Arm D fold4
```

## 判定基準 (survey v3 stop rules準拠)

対照はexp038 strict green (fold0 0.28954 / fold4 0.59607)。両fold同方向に改善しない限り
5-foldへ進めない。Arm CとArm Dのどちらが勝ってもよいが、両方負けた場合はhurdle系の
勝ちとして本仮説をクローズする。

採用条件: fold0+fold4の両方でexp038比改善、かつOOF衛星別でMeteosat/GOESの悪化がないこと。
