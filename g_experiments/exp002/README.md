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

Slurmクラスタ:

```bash
sbatch singularity_run.sh config.yaml 0
sbatch singularity_run.sh all
sbatch singularity_run.sh submit
sbatch singularity_run.sh all_submit
sbatch --ntasks=4 --ntasks-per-node=4 --gpus-per-node=4 --cpus-per-task=32 singularity_run.sh config_a100x4.yaml all
```

`CONTAINER_FOLDER` と `CONTAINER_NAME` は環境変数で上書きできます。デフォルトは
`/group/project143/common/containers/kaggle-gpu-images-python-v163.sif` です。
GPUメモリは `logs/gpu_memory_${SLURM_JOB_ID}_*.csv` に300秒間隔で記録します。
間隔を変える場合は `GPU_LOG_INTERVAL_SECONDS=600 sbatch ...` のように指定します。

5-fold終了後、または既存checkpointから推論と提出zip作成:

```bash
sbatch singularity_run.sh submit
```

`submit` は `../../g_model/exp002/best_model_fold*.pt` をすべて使ってensembleします。
fold0しか無い場合は単fold提出zipを作成します。

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
