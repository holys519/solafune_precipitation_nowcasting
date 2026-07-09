# exp013: Parallax Registration + Wider Temporal Context (G-017 / G-018)

`doc/data_characteristics_review.md` の分析に基づく2つのデータレベル変更を、
config切替で単離できる形で実装した実験です。モデルはexp009の勝ちアーム
(`two_head_compact_unet`) を既定にし、OOF差分がデータ変更に帰属できるようにしています。

## Changes

### G-017: Parallax registration (`registration.*`)

`outputs/l_eda/exp001` で計測した per-(location, satellite) の median (dy, dx) を使い、
各衛星フレームとmaskを41×41ターゲットグリッド上で位置合わせします
(最大補正 ~5px、jakarta/himawari で整列後相関 +0.089)。

- シフト表 (`parallax_shift_by_location.csv`) と eval→最近傍train地点対応表
  (`train_eval_feature_shift.csv`) は実験ディレクトリ内にコピー済み
  (`outputs/` はgitignoreのため、self-contained化)。
- **未知のeval地点**は特徴空間最近傍のtrain地点のシフトを継承し、無ければ
  衛星別median、それも無ければ (0,0)。
- シフトで生じるゼロ境界はmaskにも同時に反映されるので、モデルはパディングと
  実ゼロを区別できます。
- シフト意味論はl_edaの `corr_at_shift` と一致: `out[y,x] = sat[y+dy, x+dx]`。

### G-018: Wider temporal context (`data.context_offsets`)

exp009の `context_rows` (前方のみ) を任意オフセットに一般化しました。
30分単位の行オフセットのリストで、`[-1, 0, 1]` = 前行 + 現在行 + 後続行。
`[0, 1]` はexp009/exp012の完全再現です。欠損行はゼロ埋め + mask 0。

入力チャネル数: `len(offsets) × 3観測 × 16ch + len(offsets) × 3 mask + one-hot 3ch`
(offsets 3個 → 156ch)。

## Configs

| Config | context_offsets | registration | in_ch | 目的 |
| --- | --- | --- | ---: | --- |
| `config.yaml` | [-1, 0, 1] | on | 156 | フル exp013 アーム |
| `config_registration_only.yaml` | [0, 1] | on | 105 | G-017単離 (exp009と直接比較) |
| `config_context_only.yaml` | [-1, 0, 1] | off | 156 | G-018単離 |
| `config_a100x2.yaml` / `config_a100x4.yaml` | [-1, 0, 1] | on | 156 | フルアームのA100プロファイル |

チケットの受け入れ条件どおり、まず**単一fold** (`bash run.sh <config> 0`) で
registration の有無をA/Bし、勝った構成だけ5-foldに進めてください。

## Run

```bash
cd g_experiments/exp013
bash run.sh config_registration_only.yaml 0   # G-017 A/B (vs exp009 fold0)
bash run.sh config_context_only.yaml 0        # G-018 A/B
bash run.sh config.yaml 0                     # フルアーム
sbatch singularity_run.sh                     # 5 fold -> OOF分析 -> 提出zip (config.yaml)
```

## Outputs

| Path | Description |
| --- | --- |
| `../../g_model/exp013/best_model_fold{K}.pt` | fold別ベストcheckpoint |
| `../../outputs/analysis/exp013/` | OOF診断 |
| `../../outputs/submissions/exp013_submission.zip` | 提出zip |

## Comparison anchors

- exp009 OOF `tile_rmse` 0.6239 (fold0 valid tile_rmse 0.3255) / public 0.7153
- exp011 OOF 0.6328 / public 0.7232
- exp012 (未走行) — 同fold同士で比較すること
