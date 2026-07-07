# exp002 (g_experiments): Full-Scale Pipeline Overhaul

`l_experiments/exp002` で検証済みのコードを、フルデータ・フルエポックで学習する本番実験です。
変更内容の詳細は `l_experiments/exp002/README.md` と `doc/exp001_retrospective.md` を参照。
チケットは `doc/task_tickets.md` の G-001(単fold)/ G-002(5-fold ensemble)/ G-003・G-004(A100スケール)。

## Hardware Profiles

| Config | Hardware | batch | lr | workers | Notes |
| --- | --- | ---: | ---: | ---: | --- |
| `config.yaml` | RTX 3090 x2(現行環境) | 128 | 0.001 | 12 | デフォルト |
| `config_a100x2.yaml` | A100 x2 | 256 | 0.0014 | 16 | G-003。実機未検証、初回実行時にbatch/workersを調整 |
| `config_a100x4.yaml` | A100 x4 | 512 | 0.002 | 32 | G-004。DataParallelのままなので4GPUはスケール効率低め。DDP移行(G-004)推奨 |

## Run

直接実行(Slurm不要):

```bash
cd g_experiments/exp002
bash run.sh                          # config.yaml, fold 0
bash run.sh config.yaml all          # fold 0-4 連続実行(G-002)
bash run.sh config_a100x2.yaml 3     # A100x2, fold 3
```

Slurmクラスタ(要 `CONTAINER` パス編集):

```bash
sbatch singularity_run.sh config.yaml 0
sbatch --gpus-per-node=4 --cpus-per-task=32 singularity_run.sh config_a100x4.yaml all
```

5-fold終了後、アンサンブル推論と提出zip作成:

```bash
PY=../../.venv/bin/python
$PY inference.py --config config.yaml \
  --checkpoint ../../g_model/exp002/best_model_fold0.pt \
  --checkpoint ../../g_model/exp002/best_model_fold1.pt \
  --checkpoint ../../g_model/exp002/best_model_fold2.pt \
  --checkpoint ../../g_model/exp002/best_model_fold3.pt \
  --checkpoint ../../g_model/exp002/best_model_fold4.pt
$PY make_submission.py
```

`norm_stats.json` はコミット済み(l_experiments/exp002で生成したものと同一)。再生成する場合は
`run.sh` が自動で `normalize_stats.py` を実行します(ファイルが無い場合のみ)。

## Outputs

| Path | Description |
| --- | --- |
| `../../g_model/exp002/best_model_fold{k}.pt` | fold別ベストモデル(`best_rmse` キーを含む) |
| `../../g_model/exp002/metrics_fold{k}.json` | fold別メトリクス |
| `../../outputs/submissions/exp002/` | 展開済み提出ディレクトリ |
| `../../outputs/submissions/exp002_submission.zip` | 提出zip |

## Reporting

fold別 `best_rmse` は `extract_scores.py`(リポジトリルート)で一覧化できます。5-fold mean/std と
public score を `EXPERIMENT_REPORT.md` に記録してください。

## Estimated Runtime (3090x2)

exp001実測(12000行・batch128で約350s/epoch)からの外挿で、フルデータ(fold当たり約29000-35000行)
は **約15-20分/epoch、20 epochで5-6時間/fold** 程度の見込みです。`run.sh config.yaml all` の
5-fold連続はおよそ25-30時間。ボトルネックはTIFF読み込みなので `num_workers` 増で多少短縮できます。
