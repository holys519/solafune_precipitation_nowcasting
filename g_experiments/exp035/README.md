# exp035: exp018 × exp028 × exp030 統合 (Round 5 本命)

`doc/plan/round5_experiment_plan_2026-07-16.md` の主実験。独立軸で各々単独有効だった3変更を
1モデルへ統合する:

- **exp018** (base): 高解像内部処理 (128×128) + aux wet-mask head + multi-scale MSE。
  5-fold OOF tile_rmse 0.6093 (単体ベスト)。
- **exp028**: exp017物理チャネル + target-time-first入力順 + |Δ|チャネル (165ch)。
  fold0で対exp017 −0.0052。
- **exp030**: dilated bottleneck (d=2,4)。fold0学習ログで対exp017 −0.0065。
  ※このネットのbottleneckは8×8なのでdilation 4は全域をカバーする——効かなければ
  `config_no_dilation.yaml` が切り分ける。

コードはexp018のコピーを基に、dataset.pyのみexp017版 (features機構) +
`drop_zero_observation_rows` を移植。学習レジームはexp018と同一 (epochs 100,
batch 128, スケジューラなし) にして、A/B差分が入力/ボトルネック変更だけになるようにする。

## Arms

| Config | 内容 | in_ch | model_dir |
| --- | --- | ---: | --- |
| `config.yaml` | full (exp028入力 + dilation) | 165 | `g_model/exp035` |
| `config_no_dilation.yaml` | exp028入力のみ | 165 | `g_model/exp035_no_dilation` |
| `config_dilation_only.yaml` | dilationのみ (105ch) | 105 | `g_model/exp035_dilation_only` |
| `config_tilemean.yaml` | full + タイル平均補助loss (`tile_mean_weight: 0.3`) | 165 | `g_model/exp035_tilemean` |

`config_tilemean.yaml` はE-1オラクル分解 (`outputs/g_eda/exp002`) の帰結: 3モデルとも
残差の支配項は配置ではなく**タイル量誤差** (amount_swap 0.545 vs actual 0.609、
mask_swapは悪化) だったため、multi-scaleラダーの最粗段 (タイル平均そのもの) を
直接教師する項を追加した。

## Protocol (fold0単独A/Bは廃止 — Round 5計画 §4)

1. スモークテスト: `sbatch singularity_smoke.sh`
2. fold0 **と fold4** をA/B: `sbatch singularity_run.sh config.yaml 0` /
   `sbatch singularity_run.sh config.yaml 4` (armも同様)
3. 対照はexp018の同fold saved best: fold0 **0.29234** / fold4 **0.58531** (tile_rmse)
4. 両foldで非悪化かつ片方 >0.003 改善のarmのみ5-fold本走:
   `bash submit_folds.sh <config>`

採用条件: 5-fold OOF tile_rmse < 0.606 (exp018比 −0.003)、衛星別OOFで
Meteosat悪化なし、wet_iou_025非悪化。

## Run

```bash
cd /group/project143/yamamoto/solafune_precipitation_nowcasting/g_experiments/exp035
sbatch singularity_smoke.sh                       # 事前チェック
sbatch singularity_run.sh config.yaml 0           # fold 0
sbatch singularity_run.sh config.yaml 4           # fold 4
bash submit_folds.sh config.yaml                  # 5-fold全投入 (A/B通過後)
sbatch singularity_run.sh config.yaml submit      # analyze -> infer -> zip
```
