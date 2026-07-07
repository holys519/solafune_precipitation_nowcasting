# Solafune Precipitation Nowcasting

> Competition: 宇宙からの降水ナウキャスト - 衛星データを用いた広域降水ナウキャスティング  
> Platform: Solafune  
> Competition ID: `f87811b8-1964-4f4b-84b3-6fddd67ec4b1`  
> Last updated: 2026-07-07

## Overview

静止気象衛星のマルチスペクトル観測から、GPM-IMERGの較正済み降水量を推定する時系列回帰タスクです。入力は直近30分以内の衛星観測画像、目的変数は同一地点・時刻に対応する `GPM_IMERG` 降水量GeoTIFFです。

評価指標はRMSEです。地域をまたいだ汎化性能が重要です。

このコンペでは外部データセットの利用は禁止されています。学習・検証・特徴量作成は配布データ内で完結させます。

## Data

配布予定のzipはGit管理しません。ダウンロードまたは手動配置後、以下のように置いてください。

```text
data/
├── raw/
│   ├── train_dataset.zip
│   ├── evaluation_dataset.zip
│   └── sample_submission.zip
├── train_dataset/
├── evaluation_dataset/
└── sample_submission/
```

`exp000` の準備スクリプトは `data/raw/` にあるzipを展開するだけです。認証情報や個人環境のパスは不要です。

```bash
cd l_experiments/exp000
bash download.sh
```

## Expected Files

| File | Size | Description |
| --- | ---: | --- |
| `train_dataset.zip` | 18 GB | 学習用CSVと複合GeoTIFFデータ |
| `evaluation_dataset.zip` | 13 GB | 評価用CSVと複合GeoTIFFデータ |
| `sample_submission.zip` | 179 MB | 提出形式のサンプル |

主要CSVカラム:

| Column | Description |
| --- | --- |
| `data_id` | 各サンプルの一意ID |
| `name_location` | 観測地点名 |
| `satellite_target` | 衛星名 |
| `datetime` | 観測時刻 |
| `last_30_minutes_observation_filename` | 最大3つの衛星観測ファイル |
| `gpm_imerg_filename` | 目的変数ファイル |

## Submission

提出物はzip形式です。`evaluation_target.csv` と、予測したGPM-IMERG GeoTIFFを `test_files/` 以下に格納します。

```text
submission.zip
├── evaluation_target.csv
└── test_files/
    ├── {location}_GPM_IMERG_{datetime}.tif
    └── ...
```

## Project Layout

```text
.
├── configs/                 # 共通設定
├── data/                    # 配布データと展開先。Git管理対象外
├── doc/                     # コンペ概要、データ仕様、ルール
├── eda/                     # 探索的分析
├── experiments/             # 初期実験
├── g_experiments/           # GPU/HPC向け実験
├── l_experiments/           # ローカル開発向け実験
├── g_model/                 # 学習済み重み。Git管理対象外
├── l_model/                 # ローカル学習済み重み。Git管理対象外
├── outputs/                 # 推論結果・提出zip。Git管理対象外
├── scripts/                 # 汎用スクリプト
└── src/                     # 共通実装
```

## Initial Modeling Notes

- 入力候補: `satellite_target` ごとに16バンド、最大3時刻分を時系列またはチャネル結合で扱う。
- 出力: 1バンドの降水量ラスタ。
- 初期ベースライン: U-Net系の画像回帰、MSE/RMSE最適化。
- 検証: 地点または時系列でリークを避ける分割を優先する。
- 後処理: 非負クリップ、サンプル提出のGeoTIFFメタデータ維持、提出zipのファイル名検証。

## Setup

```bash
uv sync
```

秘密情報、認証ファイル、ローカル絶対パス、大容量データ、モデル重みはGitに含めません。
