# Experiment Plan

*Last updated: 2026-07-08*

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
| exp003 | Conservative weighted baseline | Full 5-fold CompactUNet with weaker positive-pixel weighting and OOF diagnostics | Done |
| exp004 | Two-head rain model | Rain/no-rain detection plus amount regression | Done — current best public among local submissions |
| exp005 | Temporal fusion | Shared per-time stem before U-Net fusion | Done |
| exp006 | Satellite adapter | Satellite-specific input stems before shared U-Net | Done |
| exp007 | Multi-exp ensemble | Equal-weight ensemble of exp003-exp006 checkpoints | Done |
| exp008 | Official metric + postprocess | Reuse exp004 checkpoints, add tile-RMSE diagnostics and value-threshold drizzle cleanup | Implemented |
| exp009 | Successor-row inputs | Add successor row frames as extra satellite observations (`105ch`) | Implemented |
| exp010 | Data cleanup | Suspicious-label drop, IR-zero nodata handling, GOES 4ch remap | Implemented |
| exp011 | Adapter two-head | Combine satellite-specific input stems with two-head rain model | Implemented |

See `doc/task_tickets.md` for the full ticket list, including which items are `g_experiments`-only
(A100x2/x4 scaled configs, DDP migration — these are hardware-scaling concerns and are never
`l_experiments` tasks).

## Experiment Log

| Date | Exp | Result | Notes |
| --- | --- | --- | --- |
| 2026-07-07 | exp001 | CV RMSE 0.810831, public RMSE 0.753200 | Beats zero baseline (0.962 CV / 1.432 full-train); see `doc/exp001_retrospective.md` for diagnosis driving exp002 |
| 2026-07-08 | exp004 | public RMSE 0.725253 | Two-head model became the new anchor; subsequent work prioritizes drizzle suppression and official tile metric |
| 2026-07-08 | exp008-exp011 | code implemented | Postprocess, successor frames, data cleanup, and adapter-two-head variants are ready to run |
