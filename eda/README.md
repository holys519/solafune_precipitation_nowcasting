# EDA

## Scripts

| File | Purpose |
| --- | --- |
| `01_data_overview.py` | CSV全件の概要、ファイル存在確認、GeoTIFFサンプル統計、基本プロットを生成 |
| `02_deep_dive.py` | target全件分布、観測数異常、train/evaluation地点重複、衛星メタデータ全件を追加確認 |
| `03_satellite_anomalies.py` | 期待shape/channel数から外れる衛星ファイルをCSV参照行と紐づけて調査 |

## Run

```bash
python3 eda/01_data_overview.py
python3 eda/02_deep_dive.py
python3 eda/03_satellite_anomalies.py
```

`pandas`, `rasterio`, `tifffile` なしでも動くよう、標準ライブラリ、NumPy、Matplotlibだけに寄せています。衛星GeoTIFFは16チャンネルTIFFなので、スクリプト内の軽量TIFFリーダでサンプル読み込みします。

## Outputs

主な出力:

| File | Description |
| --- | --- |
| `outputs/EDA_REPORT.md` | EDA結果の要約 |
| `outputs/EDA_DEEP_DIVE.md` | 追加EDA結果の要約 |
| `outputs/SATELLITE_ANOMALIES.md` | 衛星ファイルshape/channel異常の調査結果 |
| `outputs/eda_summary.json` | 機械可読な集計結果 |
| `outputs/eda_deep_dive_summary.json` | 追加EDAの機械可読な集計結果 |
| `outputs/*_satellite_counts.csv` | split別の衛星件数 |
| `outputs/*_top_locations.csv` | split別の地点件数 |
| `outputs/satellite_sample_stats.csv` | 衛星GeoTIFFサンプル統計 |
| `outputs/target_sample_stats.csv` | GPM-IMERG targetサンプル統計 |
| `outputs/target_full_file_stats.csv` | train target全件のファイル別統計 |
| `outputs/target_stats_by_*.csv` | target全件の集約統計 |
| `outputs/observation_anomalies_*.csv` | 観測数が3未満の行 |
| `outputs/satellite_metadata_full_counts.csv` | 衛星GeoTIFF全件のメタデータ集計 |
| `outputs/satellite_anomaly_files.csv` | 期待shape/channel数から外れる衛星ファイル一覧 |
| `outputs/geotiff_metadata_samples.csv` | GeoTIFFメタデータサンプル |
| `outputs/*.png` | 件数・分布・バンド平均のプロット |
