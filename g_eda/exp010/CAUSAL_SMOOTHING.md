# Causal-only temporal smoothing OOF sweep (g_eda/exp010)

Re-tunes exp046's causal-only temporal smoothing (which shipped with untuned
weights copied from the old bidirectional design) on OOF predictions.
Stack: raw prediction -> causal smoothing (T, T-30, T-60 only, next_weight=0)
-> blur sigma -> per-satellite value_threshold. All post-processing is fit on
OOF and next_weight is asserted 0 by causal_smoothing.py's causal_only guard.

## Source: exp038_sigmafixed (n=40686 OOF tiles)

- (a) no smoothing:                        0.60852
- (b) exp046 shipped (untuned, 0.85/0.15/0.0): 0.60800 (delta vs a: -0.00052)
- (c) 2-tap OOF-tuned (center=0.87, prev=0.13): 0.60799 (delta vs b: -0.00001)
- (d) 3-tap OOF-tuned (center=0.85, prev=0.00, prev2=0.15): 0.60736 (delta vs c: -0.00063) -> ADOPTED
- (e) + joint blur/threshold re-opt (sigma=1.0, thresholds={'goes': 0.0, 'himawari': 0.2, 'meteosat': 0.08}): 0.60613 (delta vs d: -0.00123)

Per-satellite (final stack e): goes=0.81238, himawari=0.75276, meteosat=0.37081

**Total delta (a -> e): -0.00239**
**Delta vs exp046 shipped (b -> e): -0.00187**

## Skipped sources

- exp047: no OOF cache at outputs/g_eda/exp003/exp047_oof_pred.npz yet (follow-up once its 5-fold checkpoints/cache are available).

## Recommended config (primary source: exp038_sigmafixed)

See `recommended_causal_weights.json` (schema documented in its own `schema_version` field / this report) for the exact machine-readable recommendation consumed downstream (e.g. by a future exp055 harvest build).

```json
{
  "schema_version": 1,
  "source_experiment": "exp038_sigmafixed",
  "generated_by": "g_eda/exp010/run_causal_smoothing_sweep.py",
  "compliance": "causal_only (2026-07-20 ruling): next_weight is always 0 in this recommendation",
  "temporal_smoothing": {
    "enabled": true,
    "causal_only": true,
    "center_weight": 0.85,
    "prev_weight": 0.0,
    "prev2_weight": 0.15,
    "next_weight": 0.0,
    "max_gap_minutes": 30
  },
  "blur_sigma": 1.0,
  "per_satellite_value_threshold": {
    "goes": 0.0,
    "himawari": 0.2,
    "meteosat": 0.08
  },
  "oof_scores": {
    "no_smoothing": 0.6085249185562134,
    "exp046_shipped_baseline": 0.6080018877983093,
    "tuned_2tap": 0.6079942584037781,
    "tuned_3tap": 0.6073646545410156,
    "final_with_joint_postprocess": 0.6061346530914307
  },
  "used_3tap": true
}
```
