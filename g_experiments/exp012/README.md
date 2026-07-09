# exp012: Successor-Row Frames × Satellite-Adapter Two-Head (G-016)

公開スコア1位の exp009 (successor-row frames, 0.7153) と2位の exp011
(satellite-adapter two-head, 0.7232) の統合です。両者は独立な軸
(時間文脈の拡張 vs センサー別入力stem) を変更しており、どちらも単独で
exp004アンカー (0.7253) を上回っているため、組み合わせが Round 2 の
最優先チケットです (`doc/task_tickets.md` G-016)。

## Method

- **データ**: exp009 と同一。現在行 + 同地点30分後のsuccessor行の衛星観測を
  最大6スロット (3観測 × 2行) スタックし、`16ch × 6 + mask 6ch + 衛星one-hot 3ch
  = 105ch` 入力。successor行が無い場合はゼロ埋め + mask 0。
- **モデル**: exp011 の `satellite_adapter_two_head_unet` を105ch入力に拡張。
  衛星one-hotは `context_rows` に関わらず**末尾3ch**に置かれるため、adapterの
  stem選択ロジックは無変更で動く。stemは各衛星専用の `ConvBlock(105, c)` × 3。
- **損失**: exp009 と同じ `two_head_rain` (BCE + 重み付きamount MSE + 予測MSE)。
- **選択メトリクス**: `tile_rmse` (公式メトリクス近似)。
- **正規化**: exp009 の `norm_stats.json` をそのままコピー (決定論的サンプルからの
  per-(satellite, band) 統計であり、データは不変のため再計算不要)。

exp009の勝ちアーム (`two_head_compact_unet`) も config の
`model.architecture` で選択可能なまま残してあり、同一foldでのA/B比較ができる。

## Run

```bash
cd g_experiments/exp012
sbatch singularity_run.sh                       # config.yaml, 5 fold学習 -> OOF分析 -> 提出zip
sbatch singularity_run.sh config_a100x2.yaml 2  # A100x2プロファイル, fold 2のみ
bash run.sh config.yaml 0                       # Slurmなし直接実行 (fold 0)
bash run.sh config.yaml submit                  # 既存checkpointの分析 -> 推論 -> zip
```

## Outputs

| Path | Description |
| --- | --- |
| `../../g_model/exp012/best_model_fold{K}.pt` | fold別ベストcheckpoint |
| `../../outputs/analysis/exp012/` | OOF診断 (analysis_summary.json ほか) |
| `../../outputs/submissions/exp012_submission.zip` | 提出zip |

## Acceptance (G-016)

- 5-fold OOF `tile_rmse` を exp009 (0.6239) / exp011 (0.6328) と比較。
- OOF結果に関わらず提出し、`doc/public_scores.md` に記録する。
