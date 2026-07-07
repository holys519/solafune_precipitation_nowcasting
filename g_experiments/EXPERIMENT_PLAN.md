# Experiment Plan

*Last updated: 2026-07-07*

## Competition

| Item | Detail |
| --- | --- |
| Name | 宇宙からの降水ナウキャスト |
| Platform | Solafune |
| Task | Multispectral satellite imagery to precipitation raster regression |
| Metric | RMSE |
| External data | Prohibited |

## Strategy

1. Validate file layout, GeoTIFF metadata, band order, missing values, and target distribution.
2. Build a simple U-Net regression baseline using max 3 observations x 16 bands.
3. Establish leak-resistant CV by location and/or time.
4. Improve temporal handling with sequence models, 3D/ConvLSTM blocks, or per-time encoders.
5. Tune loss and post-processing for heavy rainfall and non-negative predictions.

## Planned Experiments

| Exp | Theme | Purpose | Status |
| --- | --- | --- | --- |
| exp000 | Data setup | Unzip and inspect distributed archives | Done |
| exp001 | Baseline | 54-channel Compact U-Net regression baseline | Done — public RMSE 0.753200 |
| exp002 | Pipeline overhaul | Per-satellite/band normalization, rain-weighted loss, GroupKFold 5-fold, flip/rot90 augmentation, anti-aliased resize, config-selectable pretrained-encoder architecture | Code ready (`l_experiments/exp002`, `g_experiments/exp002`); full-scale training not yet run — see `doc/task_tickets.md` G-001/G-002 |
| exp003 | Architecture + full ensemble | Promote whichever of CompactUNet/`smp_unet` wins the exp002 ablation, run full 5-fold + ensemble | Blocked on exp002 ablation (ticket L-002) |
| exp004 | Temporal encoder | Encode each observation time separately before fusion | Planned (ticket L-003) |
| exp005 | Post-processing | Calibration, clip thresholds, fold-averaging vs best-fold | Planned (ticket L-004) |

See `doc/task_tickets.md` for the full ticket list, including which items are `g_experiments`-only
(A100x2/x4 scaled configs, DDP migration — these are hardware-scaling concerns and are never
`l_experiments` tasks).

## Experiment Log

| Date | Exp | Result | Notes |
| --- | --- | --- | --- |
| 2026-07-07 | exp001 | CV RMSE 0.810831, public RMSE 0.753200 | Beats zero baseline (0.962 CV / 1.432 full-train); see `doc/exp001_retrospective.md` for diagnosis driving exp002 |
