# Dataset Specification

Last updated: 2026-07-07

## Local Data Status

データは展開済みです。実験コードでは、基本的に以下の安定パスを参照します。

```text
data/
├── train_dataset/
│   ├── train_dataset.csv
│   ├── goes/
│   ├── gpm_imerg/
│   ├── himawari/
│   └── meteosat/
├── evaluation_dataset/
│   ├── evaluation_target.csv
│   ├── goes/
│   ├── himawari/
│   ├── meteosat/
│   └── test_files/
├── sample_submission/
│   ├── evaluation_target.csv
│   └── test_files/
└── raw/
    ├── train_dataset.zip -> ../train_dataset_b1c74968f2f24eaeb2852b47b80a581e.zip
    ├── evaluation_dataset.zip -> ../evaluation_dataset_ba14cc1598034cc689eaf39b4f80c09d.zip
    └── sample_submission.zip -> ../sample_submission_95c3b1e094034f5fbba421f5e5310f8a.zip
```

`data/` はGit管理対象外です。zip本体、展開済みGeoTIFF、提出物はコミットしません。

## Source Archives

| Role | Original filename | Standard link | Size |
| --- | --- | --- | ---: |
| Train | `train_dataset_b1c74968f2f24eaeb2852b47b80a581e.zip` | `data/raw/train_dataset.zip` | 18 GB |
| Evaluation | `evaluation_dataset_ba14cc1598034cc689eaf39b4f80c09d.zip` | `data/raw/evaluation_dataset.zip` | 13 GB |
| Sample submission | `sample_submission_95c3b1e094034f5fbba421f5e5310f8a.zip` | `data/raw/sample_submission.zip` | 179 MB |

再展開する場合:

```bash
cd l_experiments/exp000
bash download.sh
```

## Extracted File Counts

| Directory | Files | Size | Notes |
| --- | ---: | ---: | --- |
| `data/train_dataset/` | 161377 | 19 GB | train CSV, satellite images, target GPM-IMERG |
| `data/evaluation_dataset/` | 115691 | 13 GB | evaluation CSV, satellite images, sample target files |
| `data/sample_submission/` | 29091 | 237 MB | evaluation CSV and `test_files/` template |

### Train Breakdown

| Path | Files |
| --- | ---: |
| `data/train_dataset/train_dataset.csv` | 1 |
| `data/train_dataset/goes/` | 30788 |
| `data/train_dataset/gpm_imerg/` | 40686 |
| `data/train_dataset/himawari/` | 38971 |
| `data/train_dataset/meteosat/` | 50931 |

### Evaluation Breakdown

| Path | Files |
| --- | ---: |
| `data/evaluation_dataset/evaluation_target.csv` | 1 |
| `data/evaluation_dataset/goes/` | 21717 |
| `data/evaluation_dataset/himawari/` | 34722 |
| `data/evaluation_dataset/meteosat/` | 30161 |
| `data/evaluation_dataset/test_files/` | 29090 |

### Sample Submission Breakdown

| Path | Files |
| --- | ---: |
| `data/sample_submission/evaluation_target.csv` | 1 |
| `data/sample_submission/test_files/` | 29090 |

## CSV Files

| CSV | Rows including header | Samples |
| --- | ---: | ---: |
| `data/train_dataset/train_dataset.csv` | 40687 | 40686 |
| `data/evaluation_dataset/evaluation_target.csv` | 29091 | 29090 |
| `data/sample_submission/evaluation_target.csv` | 29091 | 29090 |

Actual columns:

| Column | Description |
| --- | --- |
| `unique_id` | 各サンプルの一意ID |
| `name_location` | 観測地点名 |
| `satellite_target` | `himawari`, `goes`, `meteosat` |
| `datetime` | 予測対象時刻 |
| `last_30_minutes_observation_filename` | 直近30分以内の最大3つの衛星観測ファイル名リスト |
| `gpm_imerg_filename` | 目的変数または提出対象のGPM-IMERGファイル名 |

Example train row:

```csv
unique_id,name_location,satellite_target,datetime,last_30_minutes_observation_filename,gpm_imerg_filename
c901-207d,aceh,himawari,2023-01-01 00:00:00,"['train_aceh_Himawari_20221231_2330.tif', 'train_aceh_Himawari_20221231_2340.tif', 'train_aceh_Himawari_20221231_2350.tif']",train_aceh_GPM_IMERG_2023-01-01_00-00-00.tif
```

## File Resolution Rules

`satellite_target` に応じて、衛星観測ファイルを以下から読む想定です。

| `satellite_target` | Train image dir | Evaluation image dir |
| --- | --- | --- |
| `goes` | `data/train_dataset/goes/` | `data/evaluation_dataset/goes/` |
| `himawari` | `data/train_dataset/himawari/` | `data/evaluation_dataset/himawari/` |
| `meteosat` | `data/train_dataset/meteosat/` | `data/evaluation_dataset/meteosat/` |

Train target:

```text
data/train_dataset/gpm_imerg/{gpm_imerg_filename}
```

Evaluation/sample target template:

```text
data/evaluation_dataset/test_files/{gpm_imerg_filename}
data/sample_submission/test_files/{gpm_imerg_filename}
```

提出zip作成時は、`evaluation_target.csv` と予測GeoTIFFを `test_files/` 以下に配置します。

## Satellite Bands

Himawari 8/9:

```text
B01 B02 B03 B04 B05 B06 B07 B08 B09 B10 B11 B12 B13 B14 B15 B16
```

GOES:

```text
C01 C02 C03 C04 C05 C06 C07 C08 C09 C10 C11 C12 C13 C14 C15 C16
```

Meteosat:

```text
vis_04 vis_05 vis_06 vis_08 vis_09 nir_13 nir_16 nir_22
ir_38 wv_63 wv_73 ir_87 ir_97 ir_105 ir_123 ir_133
```

GPM-IMERG target:

```text
precipitation
```

## Checks For Modeling

- `last_30_minutes_observation_filename` は文字列化されたPythonリストなので、`ast.literal_eval` で読む。
- 実データの一意ID列は説明文の `data_id` ではなく `unique_id`。
- CSVのファイル名と実ファイルの存在を最初に検証する。
- 各GeoTIFFのshape, dtype, nodata, CRS, transform, band orderを確認する。
- GOESには期待値の `141x141x16` 以外に、CSVから参照される `141x141x15/14/13/12` と `282x282x4` のファイルが少数ある。固定16ch入力へ詰める処理、missing-band mask、または衛星別adapterが必要。
- 観測ファイル数が3未満の行がtrain/evaluation双方にある。空リストも存在するため、Datasetは空観測を扱える必要がある。
- targetの欠損値、負値、極端値、降水ゼロ比率を確認する。
- CVは `name_location` と時系列のリークを避ける分割を優先する。
- 提出GeoTIFFはsample submissionまたはevaluation templateのメタデータを維持して書き出す。
