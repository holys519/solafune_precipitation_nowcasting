# exp060 — Exact total × distribution factorization (exp058 intent, rebuilt on exp056)

Recreates the design goal of the parallel-track `exp058`/`TotalShapeNowcaster` — a
**mathematically identifiable** `prediction = tile_total × spatial_distribution` where
`sum(distribution) = 1` so `sum(prediction) = tile_total` exactly, with occurrence as an
auxiliary head only (no `rain_prob` gate diluting the served amount) — but built on
`exp056`'s **stable, champion** encoder/decoder and, critically, `exp056`'s **log-space
training discipline**.

## Why exp058 NaN'd and why exp060 does not

`exp058` (both folds) diverged to NaN by ~epoch 12: its total head was an unbounded softplus
and the dominant loss weight sat on a **raw-mm-space prediction MSE**. Heavy-rain tiles
(≤150 mm/pixel, tile totals in the hundreds–thousands) produced exploding squared errors and
gradients. Its best pre-NaN folds (0.301 / 0.616) were already worse than `exp056`
(0.2816 / 0.5850).

exp060 keeps the exact-sum decomposition but removes the blow-up:
- **Total in log space**: the MLP emits `log_total` (clamped only for fp safety); `total =
  expm1(log_total)`; the loss is `MSE(log_total, log1p(true_tile_total))`. Bounded gradients —
  exactly how `exp056`/`exp038` train their intensity heads (this project's repeatedly-confirmed
  "log space wins" rule).
- **Distribution is a softmax** (bounded, sums to 1), supervised by MSE against the true
  normalized field `target / target.sum()` on wet tiles only (all-dry tiles zero-weighted).
- The served prediction never carries the dominant loss weight in raw mm space; the only raw-mm
  terms are the small `multiscale_weight_2/4` carried over from `exp056`, on a now-bounded pred.

## Clean single-axis change vs the champion

Same input pipeline, CV, `internal_size: 128`, and encoder/decoder as `exp056`. The ONLY change
is the output formulation: `exp056`'s (wet-conditional mean × mean-1-normalized shape × rain_prob
gate) → exp060's (exact softmax distribution × log-space total, occurrence auxiliary). This tests
whether the exact-sum identifiable decomposition beats `exp056`'s soft-normalized one. A
higher-resolution variant (exp058's other axis) is a deliberate separate follow-up so the two
changes are never conflated.

## Gate

`bash singularity_smoke.sh` then fold 0 and 4 vs `exp056` champion (fold0 0.2816 / fold4 0.5850).
Advance to 5-fold only if both improve (or one improves and the other is within noise). The smoke
test explicitly asserts `sum(pred)==tile_total`, `sum(distribution)==1`, and finiteness under a
**heavy-rain** target (the exp058 NaN regression guard).

Note: `analyze_oof.py` still carries exp056's mean_intensity/shape oracle-ladder diagnostic; it
must be adapted to exp060's `tile_total`/`distribution` keys before the submit/OOF stage (not
needed for the fold0/4 gate, which reads train.py's metrics).
