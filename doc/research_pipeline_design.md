# Research pipeline design

## Purpose

The legacy `g_experiments/expNNN` directories remain immutable reproduction
artifacts. New modeling code lives in `src/precip_nowcast`; an experiment consists
of a validated configuration, launch scripts, and a short hypothesis document.
This prevents copied implementations from silently diverging while preserving the
ability to reproduce historical submissions.

The organization follows the useful properties of the public
`Moyasii/Kaggle-2024-RSNA-Pub` solution: source code is separated from orchestration,
configuration is explicit, and preparation/training/inference entry points are
independently runnable. It does not copy task-specific medical-imaging code.

## Scientific invariants

Every model or dataset change must preserve and test the following:

1. A location belongs to exactly one fold.
2. Evaluation uses no observation later than the target time.
3. Satellite rasters are not reduced to target resolution before feature extraction.
4. Submitted predictions are non-negative and preserve GeoTIFF metadata.
5. The amount-distribution model obeys
   `prediction.sum() == predicted_tile_total`.
6. Both pooled RMSE and mean per-tile RMSE are reported explicitly.
7. Model selection never learns post-processing or blend parameters on the same
   samples used to report their gain.

## Package boundaries

- `config.py`: typed configuration and fail-fast validation.
- `data.py`: TIFF loading, missing-frame semantics, normalization, group folds.
- `model.py`: pure neural architecture; no filesystem or experiment knowledge.
- `losses.py`: loss decomposition returned for audit logging.
- `metrics.py`: streaming metrics with explicit aggregation.
- `utils.py`: deterministic seeding and atomic artifact writes.
- `scripts/train_research.py`: one-fold training orchestration.
- `scripts/predict_research.py`: fold ensemble and metadata-preserving inference.

## exp058 hypothesis

The existing high-resolution model receives 41×41 inputs interpolated to 128×128.
exp058 instead loads satellite data at 164×164 and extracts features before reducing
to the native 41×41 target. Its output is:

```text
tile_total = softplus(global_head(bottleneck))
spatial_distribution = softmax(spatial_logits)
prediction = tile_total × spatial_distribution
```

This removes the non-identifiability of multiplying independently trained
occurrence, intensity, and shape branches. The auxiliary occurrence loss remains
useful for representation learning but cannot change total rainfall at inference.

## Evaluation protocol

The first gate is folds 0 and 4 against `exp038_sigmafixed` using the same location
membership where possible. A candidate is promoted only after:

- paired per-location deltas are computed;
- satellite and target-intensity strata are checked;
- fold-specific failures are inspected;
- the total-amount error improves without material spatial-correlation regression.

Blend and calibration experiments require outer cross-fitting. Full-OOF fitted
weights may be produced diagnostically but cannot be used as evidence of gain.

## Publication checklist

- configuration and checkpoint contain the exact fold/location manifest;
- random seeds and dependency versions are recorded;
- tests cover dry tiles, missing observations, unusual channel counts, and GeoTIFF
  round trips;
- experiment reports include negative results;
- no external dataset-derived statistics or prohibited future observations enter
  training or inference.
