# Experiment Plan

*Last updated: 2026-07-16*

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
| exp028 | Target-time-first + \|Δ\| (165ch) | exp017 features + successor-row newest frame moved to channel 0 + absolute temporal-change channel; exp017-code variant via `run_variant.sh` | fold0 done — tile 0.30120 vs exp017 0.30642 (−0.0052, positive_rmse ±0) → **adopt candidate**, merged into exp035 |
| exp029 | Satellite-aware IR rain proxy | Per-satellite anchored cold-cloud [0,1] channel per frame | fold0 done — tile 0.30482 (−0.0016) but positive_rmse 2.0073 vs 1.9685 (+0.039) fails its own acceptance rule → **closed** |
| exp030 | Dilated bottleneck (d=2,4) | Receptive-field expansion without extra downsampling on exp017's two-head net | fold0 done — tile 0.29994 (−0.0065), positive_rmse 1.9188 (−0.050) → **adopt candidate**, merged into exp035. NOTE: `g_model/exp030` folds 1-4 checkpoints are stale 1-2-epoch leftovers; the 5-fold OOF 0.6534 in `outputs/analysis/exp030` mixes them and is NOT a valid five-fold result |
| exp031 | Focal+Tversky rain auxiliary | Low-weight (0.05) focal+Tversky terms on the occurrence head | fold0 diverged — valid metrics NaN from epoch 1; needs loss-stability fix (eps/clamp/fp32) before rerun. Low priority |
| exp032 | Satellite-conditional heads | Shared body, per-satellite occurrence/amount heads | fold0 done — tile 0.30882 (+0.0024) → **closed** (positive_rmse improved 1.9467; informs exp036 arm B loss-weight design instead) |
| exp033 | exp018 blend ladder + patch | No-training post-processing: blend exp024 `equal_016_017` with exp018 at 25/50/75/100%, then exp014 overlap patch; anchor exp026 = 0.6746506841387548 | Run complete (job 3934292) — 4 patched zips + raw dirs built, submissions pending per adaptive order in README |
| exp034 | OOF rain-threshold inference + blend | Apply each model's OOF-optimal rain_prob_threshold (exp016 0.25 / exp017 0.70 / exp018 0.40) at inference, blend, patch | Implemented; adaptive submission plan shared with exp033 |
| exp035 | exp018 × exp028 × exp030 integration | Round 5 primary (`doc/plan/round5_experiment_plan_2026-07-16.md`): exp018 high-res localization base + exp028 input design (165ch) + exp030 dilated bottleneck; arms `config_no_dilation` / `config_dilation_only` / `config_tilemean` isolate each axis; collision-free per-arm model dirs | Implemented — smoke test passed on cluster (4 arms). Full-arm fold0/fold4 A/B running vs exp018 (0.29234/0.58531) |
| exp036 | OOF-weighted blend + patch | No-training: serve `g_eda/exp003`'s OOF-optimal exp016/017/018 weights (global or per-satellite triples) on eval predictions, then overlap patch. Manual `--weights` override supported. Submission gate: OOF gain over ladder w018=0.5 must exceed the E-3 noise threshold 0.004 | Implemented; runs after g_eda/exp003 (job 3935470) produces `recommended_weights.json` |

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
| 2026-07-14 | exp021 | five-fold OOF standings | exp009 0.6239 / exp016 0.6186 / exp017 0.6163; exp018 completed separately at **0.6093** (best single model) |
| 2026-07-15 | exp026 | public RMSE 0.6746506841387548 | New best public: exp024 `equal_016_017` blend + exp014 overlap patch (recorded as exp033's anchor) |
| 2026-07-16 | exp028-032 | fold0 ablation verdicts | Adopt exp028 (−0.0052) and exp030 (−0.0065); close exp029 (positive_rmse regression) and exp032 (+0.0024); exp031 NaN-diverged, needs loss fix |
| 2026-07-16 | l_eda/exp003 (E-3) | CV→LB calibration on 9 submitted pairs | 5-fold OOF tile_rmse predicts LB at Spearman 0.90 (residual std 0.0033); **fold0-only is near-uninformative (Spearman 0.50)** — single-fold A/B verdicts above are weak evidence; final accepts must use 5-fold OOF. Fold0+4 screening Spearman 0.78 |
| 2026-07-16 | exp035 | created + cluster smoke passed | Integration of exp018×exp028×exp030 per Round 5 plan; full-arm fold0 (3934680) and fold4 (3934681) running |
| 2026-07-16 | g_eda/exp002 (E-1) | oracle ladder on exp016/017/018 OOF | **Dominant residual is per-tile AMOUNT error, not placement**: amount_swap 0.545 vs actual 0.609, while mask_swap (placement oracle) is worse than actual. Added `exp035/config_tilemean.yaml` (tile-mean MSE aux loss); deprioritized exp036 Meteosat-localization. Blur σ=1 is a consistent free −0.002 (below LB noise; piggyback only). All three models already beat the 0.677 flat-mean wall on train OOF — the LB gap (~0.08) is eval-location generalization |
| 2026-07-16 | exp031 | NaN root cause fixed, fold0 resubmitted (3934743) | Under AMP, fp16 rounds clamp(1e-5, 1-1e-5)'s upper bound to 1.0, so saturated dry pixels hit log(0) in the focal term; FocalTverskyRainLoss now computes in fp32 |
| 2026-07-16 | exp033 | public RMSE 0.671989922822016 (**new best**) | `exp033_w018_050_patched`: 50% exp018 into equal_016_017 before the patch gains −0.00266 over exp026. Instead of walking the rest of the ladder blindly, g_eda/exp003 (job 3935470) computes the OOF-optimal 3-way/per-satellite mixture + blur/threshold sweeps from cached OOF predictions; exp036 serves its recommendation |
| 2026-07-17 | exp035 arms | no_dilation passes the gate (fold0 0.28694 / fold4 0.58538 vs exp018 0.29234/0.58531); folds 1-3 launched (3936214-16). dilation_only loses both folds — the dilated bottleneck is what broke the full arm. tilemean marginal (fold0 −0.0025 / fold4 +0.0007) — parked; revisit as a head/loss design on the no_dilation base per g_eda/exp006's factorization evidence |
| 2026-07-17 | g_eda/exp004 stage 3 | advected smoothing rejected (+0.00012 vs static) — H2 closed for post-processing |
| 2026-07-17 | g_eda/exp006 + l_eda/exp003 | **Hidden evaluator identified as per-file-averaged RMSE**: oof_tile_rmse predicts LB at Spearman 0.961 vs pooled-pixel 0.681 — resolves exp006's aggregation-ambiguity risk (its tile/global scale reversal is real, but the tile branch is the operative one). exp006 also shows true-mean scaling recovers 86-88% of the L2-scale oracle (−0.065 tile headroom) → per-tile amount/scale prediction is the top training lever; component cross-swap fails LOFO 0/5 |
| 2026-07-16 | full submission list | public scores for exp016-018/024/026/027 recorded | exp026 0.67465 best; exp018 solo 0.69295 = best single model (OOF→LB delta matches exp017's within 0.0002); exp016/017 LB-invert their OOF order (Δ0.002 = noise); **exp027 seed-family blends are worse than exp026** — equal-type weighting over weak seed checkpoints dilutes the blend, so exp039 harvest must OOF-weight and prune members. E-3 recalibrated on 12 pairs: 5-fold OOF Spearman 0.951, noise threshold ~0.004-0.005 |
