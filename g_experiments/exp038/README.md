# exp038: Strict-Green Baseline 再構築 (survey v3 Phase 0-4)

`doc/research_survey_v3_2026-07-16.md` §7 Phase 0 の必須項目。現在のPublic best 0.66617は
overlap patch (red) + successor row / 行間平滑化 (amber) を含む探索スコアであり、
strictなチャンピオンはexp011の0.72323しか存在しない。本実験はそのギャップを埋める:
**CSV行に列挙された観測のみ・patchなし・行間平滑化なし**で、実証済み最強アーキ
(exp018のhigh-res hurdle log-normal + aux wet-mask + multi-scale MSE) を再学習する。

## Arms

| Config | 規約区分 | 内容 | in_ch | model_dir |
| --- | --- | --- | ---: | --- |
| `config.yaml` | **green** | 配布バンドそのまま (16ch×3フレーム+mask+sat) | 54 | `g_model/exp038` |
| `config_features.yaml` | amber | +波長整列/物理特徴 (外部仕様由来の対応表を使うため運営確認まで amber) | 84 | `g_model/exp038_features` |

共通: `context_rows: 1` (successor無効)、dilationなし、`drop_zero_obs_rows: true`、
temporal_smoothing無効。OOF calibration / value_threshold / blur は自行内のgreen後処理として利用可。

## 監査メモ

- exp018系configの `encoder_name/encoder_weights: imagenet` は smp_unet 選択時のみ参照される
  死背景設定であり、実際のアーキ (highres_hurdle_lognormal_unet) はscratch学習の自作ConvBlock。
  **ImageNet重みは学習に使われていない** (survey v3 Phase 0-5 の監査記録)。
- 対照(参考値)は strict exp011 = Public 0.72323 / OOF tile 0.6328。amber側の同アーキexp018は
  OOF 0.6093 — successor rowの寄与がこの差の主要因かを本実験が定量化する。

## Run

```bash
cd /group/project143/yamamoto/solafune_precipitation_nowcasting/g_experiments/exp038
sbatch singularity_smoke.sh
sbatch singularity_run.sh config.yaml 0
sbatch singularity_run.sh config.yaml 4
bash submit_folds.sh config.yaml        # fold0+4ゲート通過後
```

判定: fold0+4スクリーニング後に5-fold。strict-greenレジストリ (`doc/submission_registry.md`)
に登録し、amber系とzip/OOF/重みを混ぜない。
