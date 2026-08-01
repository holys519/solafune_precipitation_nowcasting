# Causal-only temporal smoothing OOF sweep (g_eda/exp010)

Re-tunes exp046's causal-only temporal smoothing (which shipped with untuned
weights copied from the old bidirectional design) on OOF predictions.
Stack: raw prediction -> causal smoothing (T, T-30, T-60 only, next_weight=0)
-> blur sigma -> per-satellite value_threshold. All post-processing is fit on
OOF and next_weight is asserted 0 by causal_smoothing.py's causal_only guard.

## Source: champion_ensemble_nested_blend (n=40686 OOF tiles)

- (a) no smoothing:                        0.58813
- (b) exp046 shipped (untuned, 0.85/0.15/0.0): 0.58849 (delta vs a: +0.00036)
- (c) 2-tap OOF-tuned (center=0.98, prev=0.02): 0.58812 (delta vs b: -0.00037)
- (d) 3-tap OOF-tuned (center=0.95, prev=0.00, prev2=0.05): 0.58803 (delta vs c: -0.00009) -> ADOPTED
- (e) + joint blur/threshold re-opt (sigma=0.5, thresholds={'goes': 0.0, 'himawari': 0.2, 'meteosat': 0.1}): 0.58715 (delta vs d: -0.00088)

Per-satellite (final stack e): goes=0.79474, himawari=0.72583, meteosat=0.35711

**Total delta (a -> e): -0.00098**
**Delta vs exp046 shipped (b -> e): -0.00134**

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
    "center_weight": 0.95,
    "prev_weight": 0.0,
    "prev2_weight": 0.05,
    "next_weight": 0.0,
    "max_gap_minutes": 30
  },
  "blur_sigma": 0.5,
  "per_satellite_value_threshold": {
    "goes": 0.0,
    "himawari": 0.2,
    "meteosat": 0.1
  },
  "oof_scores": {
    "no_smoothing": 0.5881340503692627,
    "exp046_shipped_baseline": 0.5884896516799927,
    "tuned_2tap": 0.5881204605102539,
    "tuned_3tap": 0.5880303382873535,
    "final_with_joint_postprocess": 0.5871509909629822
  },
  "used_3tap": true
}
```
