# Experiment Plan

*Last updated: 2026-07-10*

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
| exp012 | Successor-rows × adapter two-head | Combine exp009 successor-row frames (105ch) with exp011 satellite-adapter two-head (ticket G-016) | Implemented — smoke-tested on 3090; full 5-fold run pending |
| exp013 | Parallax registration + wider context | Per-(location, satellite) parallax shift correction (G-017) and predecessor+successor context rows, 156ch (G-018); each isolatable via config variants | Implemented — smoke-tested on 3090; single-fold A/B runs pending |
| exp014 | Tile-overlap GPM copy patch | Pure post-processing (no training/inference): overwrite eval prediction pixels with same-time train GPM truth in 3 confirmed spatial-overlap regions (G-022, `doc/tile_overlap_discovery.md`) | Done — public RMSE 0.6968727727408199 (new best, -0.0184711171697818 vs exp009 base) |
| exp015 | Isotonic OOF calibration (G-027a) | No new training: reuses `g_model/exp009` checkpoints read-only (`paths.source_model_dir`, same pattern as exp008 reusing exp004) and fits an isotonic pred->target curve from a binned OOF pixel histogram, replacing/augmenting the linear scale/bias in `oof_calibration.json`. Own `outputs/analysis/exp015`/`outputs/submissions/exp015` so exp009's real outputs stay untouched | Implemented, `sbatch singularity_run.sh` ready; pipeline (`analyze`/`submit_calibrated` stages) verified end-to-end on 3090 against a throwaway checkpoint; needs a real cloud run against exp009's actual checkpoints for a real curve + public score |
| exp016+ (planned) | Amplitude accuracy: quantile head + day/night | Multi-quantile amount head (G-026), per-satellite IR->rain empirical curve as input feature (G-027b), solar/visible-brightness regime feature (G-023) — targets the same 87%-of-error-budget mid/heavy-rain tail as exp015 but requires actual retraining; see `doc/research_survey_v2.md` | Not started |

See `doc/task_tickets.md` for the full ticket list, including which items are `g_experiments`-only
(A100x2/x4 scaled configs, DDP migration — these are hardware-scaling concerns and are never
`l_experiments` tasks).

## Experiment Log

| Date | Exp | Result | Notes |
| --- | --- | --- | --- |
| 2026-07-07 | exp001 | CV RMSE 0.810831, public RMSE 0.753200 | Beats zero baseline (0.962 CV / 1.432 full-train); see `doc/exp001_retrospective.md` for diagnosis driving exp002 |
| 2026-07-08 | exp004 | public RMSE 0.725253 | Two-head model became the new anchor; subsequent work prioritizes drizzle suppression and official tile metric |
| 2026-07-08 | exp008-exp011 | code implemented | Postprocess, successor frames, data cleanup, and adapter-two-head variants are ready to run |
| 2026-07-10 | exp014 | public RMSE 0.6968727727408199 | New best public score. Tile-overlap GPM copy patch applied on top of exp009 submission confirms the leak found in `doc/tile_overlap_discovery.md` is real and effective, not just correct on synthetic data |
