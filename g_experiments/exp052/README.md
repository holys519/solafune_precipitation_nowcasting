# exp052: Train-Time-Only Future-Frame Auxiliary Head (G-033 reinterpreted)

## なぜgreenか (2026-07-20 公式裁定に基づく)

`doc/submission_registry.md` の2026-07-20付運営裁定は明示的にこう述べている:

> 学習時にtrainフォルダの未来フレームを教師信号として使うのは可 (推論時のリークがないため)

同ファイルの「影響のまとめ」項目4:

> 学習時のみ未来フレームを補助教師信号として使うのは可、という新情報がある — 推論時の入力を
> `context_rows: 1`に保ったまま、学習時だけ補助損失で未来情報を使うアーキテクチャは理論上
> green足りうる (未着手、優先度は要検討)。

本実験はこの項目4を実装したものである。`doc/task_tickets.md` の **G-033 (exp019)** チケット
(「t+0-frame-centric input design」)は運営裁定以前に書かれ、successor row (t+0フレーム) を
**推論入力**として使う前提だった — これは2026-07-20裁定で **確定red** (successor row入力は
timestamp >= T のobservationを使うことになるため)。本実験はG-033を上記の新情報に基づいて
**再解釈**し、「未来フレームを推論入力に使う」のではなく「未来フレームを学習時の補助損失
(教師信号)にのみ使う」設計に置き換えた。

## 何をするか

`g_experiments/exp038` (現strict-green champion, context_rows: 1, 54ch入力,
`highres_hurdle_lognormal_unet`, `hurdle_lognormal`損失) の完全なコピーに、以下を追加する:

1. **dataset.py**: `PrecipDataset._future_aux_target` — TRAIN/VALIDのサンプル構築時のみ、
   同じ`name_location`の `datetime + 30*future_aux_horizon_rows`分 のrowを
   (既存の`_row_by_location_time`ルックアップを再利用して) 探し、そのrowの最新観測フレーム
   (同一衛星のみ)からIR窓帯 (`KEY_BANDS[satellite]["win"]`) を抽出・他バンドと同じ方法で
   正規化し、`(1, H, W)`の補助教師テンソルと妥当性フラグ (未来rowが存在しない/衛星が違う場合は
   0) を返す。**推論時の入力チャネルには一切影響しない** (`expected_in_channels`は変更なし)。
2. **model.py**: `HighResHurdleLogNormalUNet`に`future_aux_head`引数を追加。有効時、
   `forward_features`が内部で計算するボトルネック特徴マップ (デコーダのアップサンプリング前、
   最も圧縮された表現)をタップする小さなデコーダ (`2x[Conv3x3+ReLU]+1x1conv`)を追加し、
   ネイティブ解像度にbilinear補間した`(B,1,H,W)`を`output["future_aux"]`として返す。
   `prediction_from_output`は**一切変更していない** (`output["pred"]`のみを読む) — 補助ブランチ
   は配信予測に影響を与えられない。
3. **losses.py**: `HurdleLogNormalLoss`に`future_aux_weight`を追加。`> 0`の場合のみ
   `output["future_aux"]`と`future_target`の間のmasked MSE (妥当性フラグで無効行を除外)を
   `total`に加算する。`future_aux_weight == 0.0`のときは**exp038と数値的に完全に同一**
   (smoke_test.pyの`check_loss_future_aux_noop`で検証)。
4. **train.py**: 上記フラグをconfig経由で配線し、DataLoaderのdefault collateで
   `future_target`/`future_valid`をバッチに含める (features.future_aux_head有効時のみ)。

## ハード遵守要件 (コード上のアサーションで担保、コメントだけではない)

- **has_target/evaluation_dirガード**: `_future_aux_target`は`has_target=False`
  (このコードベースの評価モード信号) で呼ばれた場合、または`data_dir`が評価用ディレクトリを
  指している場合に`RuntimeError`を送出する。`smoke_test.py`の
  `check_future_aux_guard_evaluation_mode`で両方のケースを検証済み。
- **推論時の強制無効化**: `inference.py`は評価データセット構築時に
  `features["future_aux_head"]`を無条件で`False`に上書きし (`force_no_future_aux_features`)、
  サービングモデル構築時にも`model.future_aux_head`を無条件で`False`に上書きする
  (`force_no_future_aux_model_config`) — チェックポイントに`future_aux_decoder.*`の重みが
  含まれていてもそれらは読み込まれずドロップされる (`load_models`)。どちらも上書きが実際に
  発生した場合はログに明記して出力する。`make_submission.py`はモデル/データセットを一切
  構築しない (既存の`*.tif`をzipするだけ)ため、上書きの必要自体がないことをコメントで
  明示している。
- **ビット同一性の証明**: `smoke_test.py`の`check_future_aux_head_no_inference_coupling`は、
  同一の共有重みを持つ`future_aux_head=True`のモデルと`future_aux_head=False`のモデルを
  評価形状のバッチでforwardし、`prediction_from_output(...)`が**ビット同一**であることを
  アサートする — 補助ブランチが配信予測に一切結合していないことの直接証明。

## Arms

| Config | future_aux_horizon_rows | in_ch | model_dir |
| --- | ---: | ---: | --- |
| `config.yaml` (primary) | 1 (30分先) | 54 | `g_model/exp052` |
| `config_horizon2.yaml` (未提出、準備のみ) | 2 (60分先) | 54 | `g_model/exp052_horizon2` |

共通: `context_rows: 1` (successor無効、exp038と完全に同一の推論時入力)、
`drop_zero_obs_rows: true`、`loss.future_aux_weight: 0.1` (二次的な補助信号として控えめに設定)。

## Run

```bash
cd /group/project143/yamamoto/solafune_precipitation_nowcasting/g_experiments/exp052
sbatch singularity_smoke.sh
sbatch --parsable --job-name=exp052-fold0 \
  --output=slurm-exp052-fold0-%j.out --error=slurm-exp052-fold0-%j.err \
  singularity_run.sh config.yaml 0
sbatch --parsable --job-name=exp052-fold4 \
  --output=slurm-exp052-fold4-%j.out --error=slurm-exp052-fold4-%j.err \
  singularity_run.sh config.yaml 4
```

判定: `doc/plan/round5_experiment_plan_2026-07-16.md` §4のプロトコル通り、fold0**と**fold4の
2-fold A/Bで判定する (fold0単独は実装検証にのみ使う)。対照は exp038 strict の
fold0 tile_rmse=0.28954 / fold4 tile_rmse=0.59607
(`doc/domain_knowledge_review_2026-07-20.md` §3)。両fold改善で5-fold本走、
片方改善・片方タイ (noise閾値 ~0.003-0.005) なら正味プラス判断で5-fold検討、
両fold悪化または混在で正味マイナスなら見送り — この判断は本実験の担当者が
fold0/4の実測を見て行う (本実験の実装者はfold0/4の投入のみ行い、5-fold本走は行わない)。

strict-greenレジストリ (`doc/submission_registry.md`) に登録する際は、推論時の入力設計が
exp038と完全に同一であること (successor row・行間平滑化・overlap patchのいずれも使用しない)
を明記すること。
