# exp002: Pipeline Overhaul

## Purpose

exp001(public RMSE 0.753200)の診断結果(`doc/exp001_retrospective.md`)に基づく学習パイプラインの改修です。
アーキテクチャは基本据え置き(CompactUNet)、データ処理と学習設定を修正します。

## Changes from exp001

| # | Change | exp001 | exp002 |
| --- | --- | --- | --- |
| 1 | 正規化 | 全衛星・全バンド一律 `/255` | 衛星別・バンド別 z-score(`normalize_stats.py` → `norm_stats.json`) |
| 2 | 損失 | plain MSE | `WeightedMSELoss`: 降水>0ピクセルを `1 + pos_weight` 倍で重み付け |
| 3 | CV | 単一ランダム地点ホールドアウト | sklearn `GroupKFold(5, shuffle)` over `name_location`、`--fold` で選択 |
| 4 | ダウンサンプリング | bilinear(エイリアシングあり) | `adaptive_avg_pool2d`(アンチエイリアス) |
| 5 | Augmentation | なし | train時 flip(H/V) + rot90 を入力・ターゲット同期適用 |
| 6 | 推論 | 単一チェックポイント | 複数 `--checkpoint` のアンサンブル平均 + 任意でflip TTA |
| 7 | モデル | CompactUNet固定 | `model.architecture: compact_unet / smp_unet`(pretrained efficientnet-b0)を設定で切替 |

このディレクトリの `config.yaml` はローカル検証用に **データ・エポック数を絞った設定**
(`max_train_samples: 8000`)です。フルスケール学習は `g_experiments/exp002/` の設定で行います
(チケット G-001/G-002、`doc/task_tickets.md`)。

## Run

GPUが必要です。

```bash
cd l_experiments/exp002
bash run_train.sh                 # norm_stats.json 生成(初回のみ)→ fold 0 学習
bash run_train.sh config.yaml 1   # fold 1
../../.venv/bin/python inference.py                     # fold 0 チェックポイントで推論
../../.venv/bin/python inference.py \
  --checkpoint ../../l_model/exp002/best_model_fold0.pt \
  --checkpoint ../../l_model/exp002/best_model_fold1.pt  # 複数fold平均
../../.venv/bin/python make_submission.py
```

## Outputs

| Path | Description |
| --- | --- |
| `norm_stats.json` | 衛星別・バンド別 mean/std(再現用にこのディレクトリに置く) |
| `../../l_model/exp002/best_model_fold{k}.pt` | fold別ベストモデル |
| `../../l_model/exp002/metrics_fold{k}.json` | fold別メトリクスと学習履歴 |
| `../../outputs/submissions/exp002/` | 展開済み提出ディレクトリ |
| `../../outputs/submissions/exp002_submission.zip` | 提出zip |
