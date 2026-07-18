# Competition Overview

## Task

静止気象衛星のマルチスペクトル観測画像から、GPM-IMERGの較正済み降水量を推定する広域降水ナウキャスティングです。各サンプルには、同一地点・時刻に対する直近30分以内の衛星観測ファイルと、目的変数となるGPM-IMERGファイルが紐づきます。

## Objective

衛星データのみを使い、地上レーダーが乏しい地域にも適用できる降水量推定モデルを構築します。地域を越えた汎化性能が重要です。

## Metric

RMSE。

## Input Satellites

| Satellite | Bands |
| --- | --- |
| Himawari 8/9 | `B01`-`B16` |
| GOES | `C01`-`C16` |
| Meteosat | `vis_04`, `vis_05`, `vis_06`, `vis_08`, `vis_09`, `nir_13`, `nir_16`, `nir_22`, `ir_38`, `wv_63`, `wv_73`, `ir_87`, `ir_97`, `ir_105`, `ir_123`, `ir_133` |

## Target

GPM-IMERGの較正済み降水量バンド。提出時も同じファイル命名規則のGeoTIFFを生成します。

## Rules To Track

- 外部データセットは禁止。
- 配布データと、そのデータから作成した特徴量のみ使用する。
- 入賞候補では再現可能なソースコード提出と本人確認が必要。

## Research Roadmap

現在の実験方針、規約監査、論文化仮説は
[研究調査 Round 3](research_survey_v3_2026-07-16.md) にまとめています。
画像・降水コンペの手法を本課題向けの controlled A/B に翻訳した詳細 inventory は
[画像コンペ・降水コンペから抽出する安全な解法パターン](image_competition_solution_survey_2026-07-17.md)
を参照してください。初期調査の履歴は [research_survey.md](research_survey.md) に残しています。

## Timeline

| Event | Date |
| --- | --- |
| Last updated locally | 2026-07-17 |
| Final submission | Check Solafune competition page |
