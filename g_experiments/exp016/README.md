# exp016: Hurdle Log-Normal Head (G-030)

exp009(successor-row 105ch入力、CompactUNet本体)をベースに、**ヘッドと損失だけ**を
統計的に正しいhurdle(ゼロ過剰対数正規)設計に置き換える実験。変更軸は1つ。

## Motivation

公式ディスカッションの実測分析(`doc/discussion_insights.md` §2)より:

- wet画素はほぼ完全に対数正規: ln(y)|y>0 ~ N(−0.66, 1.63²)。mean/median = exp(σ²/2) ≈ **3.8倍**。
- 我々が観測したheavy-rainの`pred_max/target_max`≈0.25(約4倍過小)はこれと一致 — 82%のゼロに
  引きずられた単一L2ヘッドの「条件付き中央値化」が原因で、アーキテクチャの問題ではない。
- RMSE最適なservingは条件付き平均 E[Y|X] = P(rain|X)·E[Y|rain,X] のみ。quantile servingは
  2チームがLB −0.006〜−0.15を実測。
- tail重み付き損失・class重み付きBCEはともに実測ネットマイナス(exp009の`pos_weight: 2.0`/
  `bce_pos_weight: 3.0`はこの知見に反していた)。

## Method

`hurdle_lognormal_unet`(model.py)+ `hurdle_lognormal`損失(losses.py):

- **occurrenceヘッド**: 重みなしBCE。sigmoid(logits)がcalibratedな確率になること自体が
  積 P·E の不偏性の条件。
- **intensityヘッド(μ, σ)**: **wet画素のみ**で学習(ゼロ質量を一切見せない)。
  `sigma_mode: predicted`はGaussian NLLでper-pixel σを学習、`fixed`はln(y)へのMSEに帰着。
- **serving**: `pred = P(rain) · exp(μ + serving_sigma_scale·σ²/2)`、amount_cap=150でクリップ。
- データ掃除: 入力ゼロの235行(全Meteosat; france 191/andalusia 23/…と実データで確認済み)を
  **train行のみ**からdrop(`data.drop_zero_obs_rows`)。valid/eval行は不変。
- analyze_oof.pyはexp015の改良版(isotonic + `calibration_comparison`)を継承。

## Configs

| Config | 内容 |
| --- | --- |
| `config.yaml` | 本命: sigma予測(NLL)、mean serving |
| `config_fixed_sigma.yaml` | σ固定アーム(NLL不安定時のフォールバック)— 要再学習 |
| `config_median_serving.yaml` | median serving診断アーム(`serving_sigma_scale: 0.0`)— **再学習不要**: パラメータ集合は同一なので、学習済みcheckpointをこのconfigでanalyze/inferするだけでmean vs medianのOOF A/Bができる(3.8x理論の実地確認) |

## Run

```bash
cd g_experiments/exp016
sbatch singularity_run.sh                       # train 5 folds -> analyze -> submit
sbatch singularity_run.sh config.yaml 0         # single fold A/B first
sbatch singularity_run.sh config.yaml analyze   # OOF diagnostics only
```

## Acceptance criteria

- OOF `tile_rmse` < 0.6239(exp009)。
- `oof_sample_metrics.csv`で `pred_max/target_max`(target_max≥10)が現状の0.25から大幅改善。
- `analysis_summary.json["calibration_comparison"]`: hurdleヘッドは設計上すでに平均をserving
  しているので、isotonic補正の上積みがexp015(−0.0057)より小さくなるはず — そうなっていれば
  「ヘッドが正しく直った」証拠でもある。
- median-servingアームがmean servingより明確に悪ければ対数正規理論の実地確認になる。

## Verification (2026-07-10, local 3090)

- ユニットテスト: serving式 `pred == sigmoid(logits)·exp(μ+σ²/2)` の一致、σのclamp、
  all-dryバッチでμ/σヘッドの勾配が厳密に0(wet-onlyマスキング)かつoccurrenceヘッドは学習継続、
  median-servingアームが同一state_dictで動作(serving-time-onlyスイッチ)。
- 実データsmoke(2 epoch/512サンプル): train→analyze_oof(calibration_comparison出力まで)→
  inference(49行)全ステージ成功。zero-obs行drop発動確認(fold0 train: 44行drop)。
- 全235 zero-obs行が全Meteosatであることを実データで確認(ディスカッションFinding 9と一致)。
