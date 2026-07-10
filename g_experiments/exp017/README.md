# exp017: Wavelength Alignment + Physics Channels (G-031)

exp009(successor-row入力、two-headモデル、`two_head_rain`損失)をベースに、**入力パイプライン
だけ**を拡張する実験。ヘッド/損失はexp009のまま(exp016と軸を分離し、後でexp016の勝ちヘッドと
合流できるようにする)。

## Motivation(全て公式ディスカッションの実測値、`doc/discussion_insights.md` §3)

1. **波長マッピング**: バンドindex Nは衛星ごとに物理波長が異なり、Meteosatは固定indexだと
   主要3バンド中2つが誤り(index 12はHimawariでIR窓10.4µm、Meteosatではオゾン9.7µm)。
   WVチャネル修正だけで相関0.297→0.370(+25%)。
2. **uint8量子化**: 教科書のsplit-window「差」は1-3Kの信号が量子化で消滅(部分相関~0.00)。
   「比」SPL/(W+1)は生き残る(−0.31/−0.20/−0.13)。IR8.5−W差は「隠れた宝石」(#2追加特徴)。
3. **時間方向**: 有効な時間信号は変化の絶対量(|Δ| Spearman +0.69)。W[t2]−W[t0]の
   temporal diffチャネルを推奨(光学フローは無価値)。
4. **昼夜**: VISバンドは夜間(データの52-73%)ノイズ。UTCではなくVIS平均輝度<5で夜判定。

## Method

`dataset.py`の`features:` config(train.pyがin_channelsの整合を起動時に検証):

- `canonical_bands: true` — フレームごとに6チャネル追加: VIS赤/中赤外3.9/WV7.3/IR8.5/
  IR窓10.5/IR split12.3を衛星別の検証済みindex表(`KEY_BANDS`)で選択。生16chブロックは不変の
  まま、衛星間で物理的に整列したビューを共有重みに与える。
- `engineered: true` — フレームごとに3チャネル(**生uint8値上で計算してから**固定スケーリング):
  split-window比・IR8.5−W・WV7.3−W。さらに行ごとに2チャネル: IR窓バンドのtemporal diff
  (最新−最古フレーム)と昼夜フラグ(最新フレームのVIS平均<5で夜)。
- 正規化は既存のper-(sat,band) mean/stdのまま(median/IQR化は別チケットで — 変更軸を増やさない)。

## Configs

| Config | features | in_channels |
| --- | --- | --- |
| `config.yaml` | canonical + engineered | 163 |
| `config_engineered_only.yaml` | engineeredのみ | 127 |
| `config_canonical_only.yaml` | canonicalのみ | 141 |

チャネル数式: `6フレーム×(16+6c+3e) + 6マスク + 2行×2e + 3 one-hot`(dataset.pyの
`expected_in_channels`が正、configと不一致なら学習起動時に即エラー)。

## Run

```bash
cd g_experiments/exp017
sbatch singularity_run.sh config.yaml 0                    # single-fold A/B first
sbatch singularity_run.sh config_engineered_only.yaml 0    # feature-set ablation
sbatch singularity_run.sh config_canonical_only.yaml 0
sbatch singularity_run.sh                                  # full 5-fold + submit
```

## Acceptance criteria

- single-fold(fold0)OOF `tile_rmse`がexp009同fold比で改善、特に**Meteosat行のサブグループ**
  (`oof_group_metrics.csv`のsatellite=meteosat)で改善が大きいこと(波長修正の主対象)。
- 3構成の比較でどの特徴セットが効くかを分離。勝ち構成をexp016の勝ちヘッドと組み合わせるのが
  次のステップ(それがexp018のベースになる)。

## Verification (2026-07-10, local 3090)

- 実データ3衛星でフレームテンソルを手計算と照合: canonicalブロックが`KEY_BANDS`のindexと
  完全一致、engineered 3チャネルが式どおり、値域O(1)(ratio ±2.4、diff ±2.3)。
- 163chの`__getitem__`レイアウト検証(マスク6・行特徴4・one-hot 3の位置と値域)。
- GPU smoke(1 epoch/512サンプル): train→analyze_oof(calibration_comparison出力)→
  inference(49行)全ステージ成功。
