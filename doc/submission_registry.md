# Submission Registry — 規約リスク区分 (green / amber / red)

*作成: 2026-07-17。基準: `doc/research_survey_v3_2026-07-16.md` §3。*
*派生物は upstream の最も厳しい区分を継承する。「valid表示」は手法の承認ではない。*

## 区分の定義 (要約)

- **green**: CSV行に列挙された観測のみで学習・推論。配布trainからfitした後処理。最終提出可。
- **amber**: successor row入力、evaluation複数行を読む後処理 (行間平滑化など)、外部仕様由来の
  band対応表。運営の書面回答が得られるまで探索専用。
- **red**: overlap patch (train targetのevalへのコピー)、外部データ。最終提出に使わない。

## 提出済みzipの区分

| Submission | Public RMSE | 区分 | 理由 |
| --- | ---: | --- | --- |
| exp001-exp008, exp010 各zip | 0.7250-0.7938 | green | 行内入力のみ、後処理はOOF由来 |
| exp011 | 0.7232307883574975 | green | 行内入力のみ。旧strictチャンピオン |
| **exp038** | **0.6891638997287517** | **green (現strictチャンピオン)** | current-row-only 54ch、scratch 5-fold |
| exp009 | 0.7153438899106017 | amber | successor row入力 |
| exp015 | 0.7096658388930687 | amber | exp009 checkpointsを継承 |
| exp016 / exp017 / exp018 | 0.6978 / 0.6997 / 0.6929 | amber | successor row入力 (105/163ch系) |
| exp024 各blend | 0.6919-0.6961 | amber | amber sourcesのblend |
| exp014 | 0.6968727727408199 | red | overlap patch |
| exp026 / exp027 patched / exp033 patched | 0.6747-0.6849 | red | overlap patch (+amber sources) |
| exp036_per_satellite_blur1_thr0p2_patched | 0.6706858062196032 | red | patch (+amber sources) |
| exp036_per_satellite_sm0p25_blur1_thr0p2_patched | **0.6661746681900441** | red | patch + 行間平滑化 (amber) + amber sources |
| exp037 (TTA) patched | 0.666259584999578 | red | 同上 |
| exp036_per_satellite_blur0p5_joint_patched | 0.6652621793536686 | red | patch + 行間平滑化 + amber sources |
| exp036_per_satellite_blur0p5_joint_raw | 0.6824222826340521 | amber | patchなし、行間平滑化 + successor由来sources |
| exp039_4src_joint_patched | **0.6619116739607654** | red | overlap patch + amber sources |
| exp039_4src_joint_raw | 0.6789588628265085 | amber | patchなし、successor由来sources |

## 未提出アーティファクトの区分

| Artifact | 区分 | 備考 |
| --- | --- | --- |
| exp033/034/036/037 の `*_raw.zip` 群 | amber | patchなしだがsuccessor由来sources (+平滑化) |
| exp035系 checkpoints | amber | successor row入力 (context_rows: 2) |
| exp038 (config_features.yaml) | amber | 波長整列表が外部仕様由来 |
| g_eda/exp003-005 OOFキャッシュ・スイープ | amber | exp016/017/018由来。green判断には流用しない (スクリーニング専用) |
| exp034 threshold zips | amber | exp016/017/018 sources |
| exp043_zero_baseline (all-zero, no model) | green | 診断専用。regime-shift仮説の検証用、スコア改善目的ではない |

## 運用ルール

1. green系の学習・OOF・blend・calibrationにamber/red artifactを混入させない
   (blend重み・平滑化係数・calibration曲線の「値」の流用も不可 — green OOFで再fitする)。
2. 運営回答で successor / 行間平滑化 が許可されたらamber→greenへ昇格し、本表を更新する。
3. overlap patchはいかなる回答でも最終提出へ戻さない (survey v3 §8)。
4. 最終提出の第一候補はgreenチャンピオン、第二候補は (許可された場合の) amberチャンピオン。
