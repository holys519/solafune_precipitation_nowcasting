# Experiment score history

Generated 2026-07-30 from local JSON/CSV artifacts and `doc/public_scores.md`.

## CV readiness verdict

The project can use full five-fold, location-held-out OOF `tile_rmse` for Public-LB-free model selection, especially for changes larger than the historical calibration residual scale. It is not yet strong enough to treat very small deltas as conclusive.

- E-3 matched pure-model pairs: 11; five-fold OOF/Public Spearman = 0.973.
- In-sample pure-model calibration residual std = 0.0041; differences below roughly this scale need paired fold/location or outer-CV evidence.
- The split is leakage-resistant by `name_location`, but only about 20 train locations / 28 location-month blocks are effectively independent, so fold climate variance is large.
- The hidden metric aggregation remains ambiguous: official material suggests pooled RMSE, while historical Public LB is much better explained by mean per-tile RMSE.
- Ensemble/postprocess experiments without their own cross-fitted OOF cannot be judged from CV alone; reusing an upstream model's OOF is not a matched evaluation.

## exp038 strict-green result

- Full five-fold OOF tile RMSE: **0.613131**; global pooled RMSE: 1.081002; fold-best mean/std: 0.620469 ± 0.179449.
- OOF-selected rain threshold reaches 0.612644 (-0.000487); this delta is too small to treat as robust by the historical 0.0041 scale.
- Versus strict-green exp011 (0.632798): **-0.019666** OOF improvement.
- Submitted Public RMSE: **0.686638** (valid; current strict/green champion).
- The pre-submission historical E-3 mapping predicted **0.6962**; actual was 0.6866 (-0.0095). This single residual is post-hoc and is not a new calibration target.

## Best full-OOF runs

| Run | Coverage | OOF tile RMSE | Global RMSE | Warning |
| --- | --- | ---: | ---: | --- |
| exp064_effb3 | full_oof | 0.597577 | 1.065774 |  |
| exp064_swin_lr2e4 | full_oof | 0.599194 | 1.064538 |  |
| exp056 | full_oof | 0.602519 | 1.071343 |  |
| exp064_effv2s | full_oof | 0.604435 | 1.085556 |  |
| exp064_effb4 | full_oof | 0.604843 | 1.081117 |  |
| exp056_seed789 | full_oof | 0.605965 | 1.070761 |  |
| exp056_seed2024 | full_oof | 0.606103 | 1.069374 | training_folds=4 but checkpoint_count=5 |
| exp050_sigmafixed | full_oof | 0.606427 | 1.077993 |  |
| exp035_no_dilation | full_oof | 0.606612 | 1.068685 |  |
| exp040_metric | full_oof | 0.606816 | 1.078881 |  |
| exp038_features | full_oof | 0.607246 | 1.070149 |  |
| exp047_sigmafixed | full_oof | 0.607468 | 1.089332 |  |
| exp038_canonical_only | full_oof | 0.607732 | 1.074214 |  |
| exp056_seed123 | full_oof | 0.608259 | 1.071312 |  |
| exp038_sigmafixed | full_oof | 0.608280 | 1.079768 |  |
| exp047 | full_oof | 0.608347 | 1.081633 |  |
| exp050 | full_oof | 0.608959 | 1.074583 |  |
| exp018 | full_oof | 0.609261 | 1.070117 |  |
| exp054_midband | full_oof | 0.609662 | 1.073883 |  |
| exp051 | full_oof | 0.610760 | 1.074621 |  |
| exp052 | full_oof | 0.611385 | 1.078893 |  |
| exp038 | full_oof | 0.613131 | 1.081002 |  |
| exp056_seed456 | full_oof | 0.616212 | 1.090431 |  |
| exp017 | full_oof | 0.616285 | 1.076002 |  |
| exp016 | full_oof | 0.618607 | 1.088377 |  |
| exp012 | full_oof | 0.621613 | 1.083154 |  |
| exp009 | full_oof | 0.623941 | 1.083466 |  |
| exp015 | full_oof | 0.623941 | 1.083466 |  |
| exp023/mean | full_oof | 0.627344 | 1.101927 |  |
| exp013 | full_oof | 0.628461 | 1.095505 |  |
| exp023/median | full_oof | 0.628852 | 1.150526 |  |
| exp004 | full_oof | 0.630651 | 1.084985 |  |
| exp008 | full_oof | 0.630651 | 1.084985 |  |
| exp056_seed1337 | full_oof | 0.630906 | 1.107495 |  |
| exp011 | full_oof | 0.632798 | 1.104284 |  |
| exp010 | full_oof | 0.642914 | 1.109473 |  |
| exp006 | full_oof | 0.646806 | 1.095473 |  |
| exp030 | full_oof | 0.653422 | 1.129282 | training_folds=1 but checkpoint_count=5 |
| exp005 | full_oof | 0.655853 | 1.098966 |  |
| exp003 | full_oof | 0.657891 | 1.094293 |  |

## Coverage

- Normalized OOF rows: 43 (40 full-size tile-RMSE rows).
- Fold metric rows: 269.
- Public records: 55 across 34 experiments.
- Experiments with no score artifact: exp000, exp021, exp034, exp043, exp044, exp057, exp058.

## Files

- `cv_oof_by_experiment.svg`: comparable full-size OOF tile RMSE.
- `fold_variability.svg`: per-fold held-out-location variability.
- `public_best_by_experiment.svg`: best recorded Public score per experiment.
- `public_lb_timeline.svg`: all timestamped submissions and best-so-far.
- `cv_vs_public.svg`: audited E-3 matched CV/Public pairs.
- CSV files are the normalized source tables behind the charts.
- Regenerate with `python3 scripts/plot_experiment_scores.py` (standard library only).

## Source caveats

- `outputs/` and model artifacts are git-ignored; this directory is a durable snapshot, but rerunning requires the local artifacts to still exist.
- `doc/public_scores.md` is hand-maintained. The E-3-only exp035_no_dilation score is included in the per-experiment chart without inventing a timestamp.
- exp003-exp006 OOF tile RMSE values are backfilled from the E-3 recomputation because their original summary schema stored only pooled RMSE.
- `artifact_warning` flags schema/provenance inconsistencies such as exp030 reporting one training metric file but five OOF checkpoints.
