# exp015: Isotonic OOF Calibration (G-027a)

`exp009` の two-head checkpoint (`g_model/exp009`) を再学習なしで再利用し、amount-head出力に対する
**isotonic (単調非減少) キャリブレーション曲線**をOOFから当てはめる実験です。exp008が `exp004` の
checkpointを読んで後処理だけ変えたのと同じパターンで、`exp009` のフォルダ・チェックポイント・
`outputs/analysis/exp009` は一切書き換えません(`paths.source_model_dir` で読み込むだけ)。

## Motivation

`doc/data_characteristics_review.md` で、tile_rmse誤差の87%がmid/heavy-rainタイルに集中し、
heavy-rain (`target_max>=10`) では `pred_max/target_max`≈0.25 と大きく過小評価されていることが
分かっている。既存の `oof_calibration.json` の線形 scale/bias は全ピクセルプールでの最小二乗解
なので、この非線形な過小評価をほぼ補正できない(exp009本番提出でも `scale≈0.99` とほぼ恒等写像で、
実際 `use_oof_calibration: false` のまま提出されている)。Isotonic回帰なら「小さい値はほぼ恒等、
大きい値は強く持ち上げる」ような非線形カーブを学習でき、この症状に直接効く。詳細は
`doc/research_survey_v2.md` §2、ticket `G-027a`(`doc/task_tickets.md`)。

## Method

- `analyze_oof.py`(exp009からコピーし isotonic 対応を追加したもの)がOOFフォワードパス中に
  95エッジの固定ヒストグラム(0〜200、中央粗め・裾はlog間隔)へ (pred, target) を集計し、
  ピクセル数で重み付けした (mean_pred, mean_target) 点に `sklearn.isotonic.IsotonicRegression
  (y_min=0, increasing=True, out_of_bounds="clip")` を当てはめる。既存の線形scale/biasもプール
  和ベースの推定なので、同じ粒度感で一貫性がある。
- 結果は `oof_calibration.json["isotonic"]["x"/"y"]` にknot点として保存され、既存の
  `scale`/`bias`/`threshold`/`rain_prob_threshold` フィールドはそのまま残る。
- `inference.py` に `apply_isotonic_curve()`(torch `searchsorted` による単調区分線形補間、
  `sklearn.predict()` と数値一致を確認済み)と `postprocess.calibration_mode: "linear"|"isotonic"`
  を追加。本実験の `config.yaml` は `calibration_mode: isotonic` がデフォルト。
- `paths.source_model_dir: ../../g_model/exp009` から checkpoint を読み込み、
  `paths.model_dir/output_dir/analysis_dir` は exp015 専用ディレクトリ
  (`g_model/exp015`※未使用, `outputs/submissions/exp015`, `outputs/analysis/exp015`)。

## Run

```bash
cd g_experiments/exp015
sbatch singularity_run.sh                    # config.yaml, submit_calibrated(isotonic) まで一括
sbatch singularity_run.sh config.yaml analyze          # OOF診断+oof_calibration.jsonのみ
sbatch singularity_run.sh config.yaml submit           # calibration切り替えずに提出物生成
sbatch singularity_run.sh config_a100x2.yaml submit_calibrated
```

`SOURCE_MODEL_DIR=/path/to/checkpoints` で読み込むcheckpointディレクトリを差し替え可能(既定は
`../../g_model/exp009`)。学習ステージ(train.py)はこの実験の想定ワークフローには含まれない
— 自己完結性のためファイルはコピーしてあるが、`run.sh` の `analyze`/`infer`/`submit`/
`submit_calibrated` ステージからは呼ばれない。

## Verification (2026-07-10, local 3090)

exp009本番checkpointはこのマシンにないため、以下は1エポック/512サンプルの捨て駒モデルで
パイプライン自体の正しさのみ検証したもの(実際のcalibration値・スコアはクラウドでの再実行が必要):

- 合成カーブ回帰テスト: 既知の「4倍過小評価」カーブを合成データから正しく復元。
- `apply_isotonic_curve`(torch)が `sklearn.IsotonicRegression.predict()` と数値一致。
- 実データでの end-to-end: `train.py` → `analyze_oof.py`(実ヒストグラム集計・fit成功、
  52 bins使用、単調性確認済み) → `inference.py --use-calibration`(linear/isotonic 両モードで
  生ログ出力し、`mode=isotonic` 指定時に生の calibrated 値が linear と有意に乖離することを
  直接確認、max diff 0.036)。

## calibration_comparison (追加: 2026-07-10)

初回のクラウド実行では `oof_calibration.json`(線形scale/bias + isotonic knots)は生成されたが、
「実際にtile_rmseが下がるか」は元の `analyze_oof.py` の設計上まだ記録されていなかった(既存の
線形scale/bias自体も、これまで効果を検証せず値だけ保存していたのと同じ)。そこで
`analyze_oof.py` に `calibration_comparison`(生pred / 線形補正後 / isotonic補正後 の3通りの
OOF `tile_rmse` を、モデルへの再フォワードなしでキャッシュ済みタイルから計算)を追加した
(`compare_calibration_effect()`)。合成データで「isotonicがraw/linearよりtile_rmseを改善する」
ケースを正しく検出できることを確認済み。**この情報は再実行しないと得られない**ため、
`sbatch singularity_run.sh config.yaml analyze` の再実行が必要。

## Next steps

1. クラウドで `sbatch singularity_run.sh config.yaml analyze` を再実行し、
   `analysis_summary.json["calibration_comparison"]` で `raw_tile_rmse` / `linear_tile_rmse` /
   `isotonic_tile_rmse` を比較する。
2. isotonicがraw/linearより明確に低ければ `submit_calibrated` で提出物を作る。悪化・横ばいなら
   G-027aは見送り、G-026(quantile head)に進む。
3. 改善するなら、`exp014`(tile-overlap patch, `apply_overlap.py`)をこの提出物の上に重ねて最終
   提出候補にする — overlap patchは実測GPM値で上書きするので、calibrationの精度に関わらず常に
   後段に置く。

## Outputs

| Path | Description |
| --- | --- |
| `../../outputs/submissions/exp015_submission.zip` | 提出zip |
| `../../outputs/analysis/exp015/oof_calibration.json` | isotonic knot点 + 既存の線形scale/bias |
| `../../outputs/analysis/exp015/oof_group_metrics.csv` | fold/location/satellite別のOOF指標 |
| `../../outputs/analysis/exp015/oof_sample_metrics.csv` | タイル単位のOOF指標(pred_max/target_max等) |
