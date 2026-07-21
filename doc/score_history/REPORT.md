# Experiment score history

Generated 2026-07-18 from local JSON/CSV artifacts and `doc/public_scores.md`.

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
- Submitted Public RMSE: **0.689164** (valid; current strict/green champion).
- The pre-submission historical E-3 mapping predicted **0.6974**; actual was 0.6892 (-0.0082). This single residual is post-hoc and is not a new calibration target.

## Best full-OOF runs

| Run | Coverage | OOF tile RMSE | Global RMSE | Warning |
| --- | --- | ---: | ---: | --- |
| exp035_no_dilation | full_oof | 0.606612 | 1.068685 |  |
| exp018 | full_oof | 0.609261 | 1.070117 |  |
| exp038 | full_oof | 0.613131 | 1.081002 |  |
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
| exp011 | full_oof | 0.632798 | 1.104284 |  |
| exp010 | full_oof | 0.642914 | 1.109473 |  |
| exp006 | full_oof | 0.646806 | 1.095473 |  |
| exp030 | full_oof | 0.653422 | 1.129282 | training_folds=1 but checkpoint_count=5 |
| exp005 | full_oof | 0.655853 | 1.098966 |  |
| exp003 | full_oof | 0.657891 | 1.094293 |  |

## Coverage

- Normalized OOF rows: 22 (19 full-size tile-RMSE rows).
- Fold metric rows: 110.
- Public records: 34 across 25 experiments.
- Experiments with no score artifact: exp000, exp021, exp034, exp040, exp041.

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
