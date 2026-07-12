# Experiment Plan

*Last updated: 2026-07-12*

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
| exp015 | Isotonic OOF calibration (G-027a) | No new training: reuses `g_model/exp009` checkpoints read-only (`paths.source_model_dir`, same pattern as exp008 reusing exp004) and fits an isotonic pred->target curve from a binned OOF pixel histogram, replacing/augmenting the linear scale/bias in `oof_calibration.json`. Own `outputs/analysis/exp015`/`outputs/submissions/exp015` so exp009's real outputs stay untouched | Done — public RMSE 0.7096658388930687 (-0.0056780510175330 vs exp009 base); not yet stacked with exp014's tile-overlap patch |
| exp016 | Hurdle log-normal head (G-030) | Statistically correct head per `doc/discussion_insights.md` §2: unweighted-BCE calibrated occurrence head × wet-only ln(y) intensity head, serve E[Y\|X]=p·exp(μ+σ²/2). Removes pos_weight/bce_pos_weight (measured net-negative by other teams). Drops the 235 zero-input Meteosat rows. Base = exp009 inputs (105ch). Directly targets the measured 4x wet underestimation (=exp(σ²/2)≈3.8). Supersedes G-026's quantile head (serving quantiles measured −0.006 to −0.15 LB by two teams) | Implemented — unit-tested (serving formula, wet-only gradient masking, median-serving arm shares state_dict) + full-stage smoke on 3090; cloud runs pending. Arms: `config.yaml` (NLL σ), `config_fixed_sigma.yaml`, `config_median_serving.yaml` (no retrain needed) |
| exp017 | Physics channels + wavelength alignment (G-031) | Append 6 wavelength-aligned canonical bands per frame (Meteosat is wrong on 2/3 key bands with fixed indices; verified `KEY_BANDS` table) + uint8-domain engineered channels: split-window ratio SPL/(W+1), IR8.5−W, WV−W, temporal-diff W[newest]−W[oldest], day/night flag from VIS mean. Head/loss stay exp009's (axis isolation). in_channels validated at startup via `expected_in_channels` | Implemented — feature math verified against hand computation on real data (3 satellites) + full-stage smoke on 3090; cloud runs pending. Arms: 163ch full / 127ch engineered-only / 141ch canonical-only |
| exp018 | Within-tile localization (G-032) | Break the 0.677 tile-mean oracle wall (`doc/discussion_insights.md` §1): internal high-res processing (upsample input to 128×128, 4-level encoder, native 41×41 output), auxiliary wet-mask segmentation head, multi-scale (2x/4x pooled) MSE for displacement tolerance. OOF diagnostics include per-tile spatial correlation, wet-mask IoU, and rain centroid distance | Implemented — three ablation configs; CPU forward/backward and syntax verified; cloud single-fold A/B pending |
| exp019 | t+0-frame-centric temporal design (G-033) | Make the successor-row t+0 frame (available eval-side 99.4%, information peak is exactly at t) the primary input frame instead of one of 6 equal channels-stacked frames; add |Δ| change-magnitude channels around t (measured Spearman +0.69 vs rain; optical flow measured useless); optional train-time-only future-frame auxiliary prediction head for the 10-min advection gap | Not started |
| exp020 | Harvest: seeds + 6-way TTA + stacked post-processing (G-034) | Best config from exp016-019 × 3 seeds × 5 folds, 6-way TTA (add rot90/180/270 to current flips), OOF-weighted ensemble, isotonic calibration (exp015 machinery), drizzle cut, then exp014 overlap patch applied last. CV metric weighted by test satellite mix (him 39%/met 39%/goes 22%) | Not started |
| exp021 | exp016/017 completion | Resume fold4, regenerate five-fold OOF/submissions, and save comparison outputs | Implemented; cloud run pending |
| exp022 | exp017 feature ablation | Isolated full/engineered/canonical arms without checkpoint collisions | Implemented; cloud run pending |
| exp023 | exp016 serving diagnostics | Compare mean/median serving and calibration without training | Implemented; cloud run pending |
| exp024 | Prediction blend | Build weighted exp009/016/017 blends without training | Implemented; cloud run pending |
| exp025 | Multi-seed variance | Configurable source, seeds, folds, and isolated checkpoints | Implemented; run after winner selection |
| exp026 | Overlap patch on exp024 blend | Apply exp014's tile-overlap GPM copy to `exp024/equal_016_017` (LB best 0.69193 shipped WITHOUT the patch; patch was worth −0.0185 on exp009 base). Pure post-processing, reuses `exp014/apply_overlap.py` with exp026's config | Implemented + smoke-tested (patch machinery verified end-to-end on synthetic sources); cloud run pending |
| exp027 | Seed-ensemble blend + patch | Inference for exp025's exp017×seed{42,123,2026} checkpoints, blend with exp016/exp017 (`half016_half017family` keeps the LB-winning 50/50 type balance; `equal_all`), then overlap-patch every scheme. Outputs `exp027_<scheme>_patched.zip` | Implemented + smoke-tested (blend weights exact, patch region bit-equals train GPM); cloud run pending |

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
| 2026-07-10 | exp015 | public RMSE 0.7096658388930687 | Isotonic OOF calibration (G-027a) improves over exp009 base by 0.0056780510175330. An offline tile-peak proxy check suggested almost no correction of heavy-rain amplitude underestimation, yet the real score improved anyway -- likely from damping over-confident mid/high false-alarm pixels. Not yet combined with exp014's tile-overlap patch |
| 2026-07-12 | exp016/017 | fold0-3 interim comparison | exp016 mean tile_rmse 0.6320 vs exp017 0.6284; exp021-exp025 added for completion, ablation, diagnostics, blending, and seed variance. |
