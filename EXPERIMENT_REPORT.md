# Experiment Report

> Competition: 宇宙からの降水ナウキャスト  
> Task: satellite-to-precipitation raster regression  
> Metric: RMSE  
> Last updated: 2026-07-07

## Competition Summary

| Item | Detail |
| --- | --- |
| Input | Last 30 minutes of multispectral satellite observations |
| Target | Calibrated GPM-IMERG precipitation GeoTIFF |
| Satellites | Himawari 8/9, GOES, Meteosat |
| External data | Prohibited |
| Submission | `evaluation_target.csv` and predicted `test_files/*.tif` in zip |

## Submissions

| Rank | Submission | CV RMSE | Public RMSE | Gap | Notes |
| ---: | --- | ---: | ---: | ---: | --- |
| 1 | `exp001_submission.zip` | 0.810831 | 0.753200 | -0.057631 | GPU Compact UNet baseline, single random location holdout CV |

## CV Results

| Exp | Method | CV Mean | CV Std | Folds | Status | Notes |
| --- | --- | ---: | ---: | --- | --- | --- |
| exp001 | 3-time satellite tensor + Compact UNet | 0.810831 |  | 1 location holdout | completed | train rows 12000, valid rows 3000, valid locations: bihar/borno_state/gaza_province/kinshasa; zero baseline RMSE 0.962228 |

## Failed Runs

| Exp | Stage | Error | Action |
| --- | --- | --- | --- |

## Findings

| Date | Finding | Evidence | Next action |
| --- | --- | --- | --- |
| 2026-07-07 | GPU baseline beats zero baseline on held-out locations | exp001 valid RMSE 0.810831 vs zero RMSE 0.962228 | Submit, then improve CV and train on more rows/epochs |
| 2026-07-07 | GOES has variable channel/shape anomalies | EDA found 55 referenced anomaly files | Keep loader padding/resizing logic in all future experiments |
| 2026-07-07 | Evaluation locations do not overlap train locations | EDA location overlap 0 | Avoid random row CV |
| 2026-07-07 | exp001 public RMSE (0.753) beat its own CV RMSE (0.811); single random 4-location holdout is high-variance | `l_model/exp001/metrics.json`, only one split evaluated | Move to 5-fold GroupKFold in exp002/exp003 before trusting CV vs public gaps |
| 2026-07-07 | exp001 only used 12000/3000 of 40686 train rows for 3 epochs; train_rmse had not converged | `l_model/exp001/metrics.json` history (train_rmse 1.223→1.174→1.151) | exp002: full data, more epochs |
| 2026-07-07 | positive-pixel RMSE (~2.52) is ~3x overall RMSE; target is 82% exact zero | `eda/outputs/EDA_DEEP_DIVE.md` full target distribution | exp002: rain-weighted loss |
| 2026-07-07 | Target statistics differ sharply by satellite (meteosat mean 0.134/positive 10.4% vs goes mean 0.385/positive 25.3%) | `eda/outputs/EDA_DEEP_DIVE.md` "Target By Satellite" | exp002: per-satellite/per-band normalization instead of flat /255 |

See `doc/exp001_retrospective.md` for the full diagnosis and `doc/task_tickets.md` for the
prioritized follow-up ticket list (exp002 pipeline overhaul, architecture ablation, GPU-scale
tickets for A100x2/x4 under `g_experiments/`).

## Final Candidates

| Candidate | CV RMSE | Public RMSE | Reason |
| --- | ---: | ---: | --- |
| exp001 | 0.810831 | 0.753200 | First valid end-to-end GPU baseline and submission zip |
