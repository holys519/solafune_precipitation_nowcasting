# Causal-only temporal smoothing OOF sweep (g_eda/exp010)

Re-tunes exp046's causal-only temporal smoothing (which shipped with untuned
weights copied from the old bidirectional design) on OOF predictions.
Stack: raw prediction -> causal smoothing (T, T-30, T-60 only, next_weight=0)
-> blur sigma -> per-satellite value_threshold. All post-processing is fit on
OOF and next_weight is asserted 0 by causal_smoothing.py's causal_only guard.

## Source: champion_ensemble_nested_blend (n=40686 OOF tiles)

- (a) no smoothing:                        0.58341
- (b) exp046 shipped (untuned, 0.85/0.15/0.0): 0.58388 (delta vs a: +0.00047)
- (c) 2-tap OOF-tuned (center=1.00, prev=0.00): 0.58341 (delta vs b: -0.00047)
- (d) 3-tap OOF-tuned (center=1.00, prev=0.00, prev2=0.00): 0.58341 (delta vs c: +0.00000) -> not adopted (no improvement over 2-tap)
- (e) + joint blur/threshold re-opt (sigma=0.0, thresholds={'goes': 0.0, 'himawari': 0.12, 'meteosat': 0.1}): 0.58279 (delta vs c: -0.00062)

Per-satellite (final stack e): goes=0.78556, himawari=0.72140, meteosat=0.35568

**Total delta (a -> e): -0.00062**
**Delta vs exp046 shipped (b -> e): -0.00109**

## Recommended config (primary source: champion_ensemble_nested_blend)

See `recommended_causal_weights.json` (schema documented in its own `schema_version` field / this report) for the exact machine-readable recommendation consumed downstream (e.g. by a future exp055 harvest build).

```json
{
  "schema_version": 1,
  "source_experiment": "champion_ensemble_nested_blend",
  "generated_by": "g_eda/exp010/run_causal_smoothing_sweep.py",
  "compliance": "causal_only (2026-07-20 ruling): next_weight is always 0 in this recommendation",
  "temporal_smoothing": {
    "enabled": true,
    "causal_only": true,
    "center_weight": 1.0,
    "prev_weight": 0.0,
    "prev2_weight": 0.0,
    "next_weight": 0.0,
    "max_gap_minutes": 30
  },
  "blur_sigma": 0.0,
  "per_satellite_value_threshold": {
    "goes": 0.0,
    "himawari": 0.12,
    "meteosat": 0.1
  },
  "oof_scores": {
    "no_smoothing": 0.583407461643219,
    "exp046_shipped_baseline": 0.5838821530342102,
    "tuned_2tap": 0.583407461643219,
    "tuned_3tap": 0.583407461643219,
    "final_with_joint_postprocess": 0.5827921032905579
  },
  "used_3tap": false
}
```
