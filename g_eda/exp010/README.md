# g_eda/exp010 — Causal-only temporal smoothing re-tune

## Motivation

The organizers ruled on 2026-07-20 that prediction-level temporal smoothing is only legal if
strictly **causal**: every target timestamp mixed into T's prediction must be `<= T`
(`doc/submission_registry.md`). exp046 shipped a causal-only rebuild of the old bidirectional
smoothing (`center_weight=0.85, prev_weight=0.15, next_weight=0`) and it beat exp038 solo on the
public LB (-0.00025), but those weights were never OOF-tuned for the causal-only case — they are
exp036/037's old bidirectional weights with `next_weight`'s share folded into `prev_weight` by
hand. This experiment properly re-tunes causal-only smoothing on OOF predictions for the current
green champion, `exp038_sigmafixed`, following the two-stage sweep pattern established by
`g_eda/exp004` (`run_temporal_smoothing.py` + `run_joint_postprocess.py`), adapted to be
causal-only and to add a second causal tap (T-60min).

## Files

- `causal_smoothing.py` — exp010-local copy of `apply_temporal_smoothing` from
  `g_experiments/exp038/inference.py`, extended with a `prev2_weight` tap (T-60min) and a hard
  `causal_only: true` guard that raises `CausalSmoothingConfigError` if `next_weight != 0`. Does
  not modify exp038/exp046's shipped files — a downstream harvest experiment should port the
  winning config into its own serving code.
- `run_causal_smoothing_sweep.py` — CPU-only sweep: loads the `g_eda/exp003`-convention OOF cache
  (`outputs/g_eda/exp003/exp038_sigmafixed_oof_pred.npz`, and `exp047_oof_pred.npz` if present),
  sweeps 2-tap and 3-tap causal weights, then jointly re-sweeps blur sigma + per-satellite
  value_threshold on the winner. Writes `recommended_causal_weights.json`, `CAUSAL_SMOOTHING.md`,
  and `outputs/g_eda/exp010/causal_smoothing_sweep.csv`.
- `singularity_run.sh` — CPU-only sbatch entrypoint (no `--gpus-per-node`, no `--nv`), mirroring
  `g_eda/exp004/singularity_run.sh`'s CPU-only convention.

## Prerequisite: OOF prediction cache

This sweep reads cached OOF prediction arrays, not raw checkpoints. If
`outputs/g_eda/exp003/exp038_sigmafixed_oof_pred.npz` does not exist yet, build it once (GPU,
~lightweight — this reuses `g_eda/exp003`'s existing caching entrypoint unmodified):

```bash
cd g_eda/exp003
sbatch singularity_cache_exp038.sh exp038_sigmafixed exp038 exp038_sigmafixed
```

`exp047` is included automatically as a second source once its 5-fold checkpoints
(`g_model/exp047/best_model_fold{0..4}.pt`) and matching OOF cache exist; until then it is
skipped and noted in `CAUSAL_SMOOTHING.md`.

## Run

```bash
cd g_eda/exp010
sbatch singularity_run.sh     # CPU-only, no GPU needed once the cache above exists
```

## Output schema: `recommended_causal_weights.json`

```jsonc
{
  "schema_version": 1,
  "source_experiment": "exp038_sigmafixed",
  "generated_by": "g_eda/exp010/run_causal_smoothing_sweep.py",
  "compliance": "causal_only (2026-07-20 ruling): next_weight is always 0 in this recommendation",
  "temporal_smoothing": {
    "enabled": true,
    "causal_only": true,
    "center_weight": 0.0,     // winning weight, T
    "prev_weight": 0.0,       // winning weight, T-30min
    "prev2_weight": 0.0,      // winning weight, T-60min (0.0 if 2-tap won)
    "next_weight": 0.0,       // always 0 -- causal guard enforces this
    "max_gap_minutes": 30
  },
  "blur_sigma": 0.0,                              // stage-2 joint re-opt winner
  "per_satellite_value_threshold": {"goes": 0.0, "himawari": 0.0, "meteosat": 0.0},
  "oof_scores": {
    "no_smoothing": 0.0,
    "exp046_shipped_baseline": 0.0,
    "tuned_2tap": 0.0,
    "tuned_3tap": 0.0,
    "final_with_joint_postprocess": 0.0
  },
  "used_3tap": false
}
```

`temporal_smoothing` is drop-in compatible with `causal_smoothing.apply_temporal_smoothing`
(and, modulo the `prev2_weight`/`causal_only` keys, with `g_experiments/exp038/inference.py`'s
`apply_temporal_smoothing`) — a downstream harvest build can load this JSON directly as the
`postprocess.temporal_smoothing` config block, then apply `blur_sigma` and
`per_satellite_value_threshold` in that order after smoothing, before writing the submission.

Note: `g_experiments/exp038/inference.py`'s current `postprocess.value_threshold` is a single
global scalar, not per-satellite, and it has no blur step at all (blur only exists in the
red multi-source blend stacks' post-processing, e.g. `g_experiments/exp036`). A downstream
harvest build that wants to consume `blur_sigma` / `per_satellite_value_threshold` needs to add
that branching logic; it does not exist in exp038/exp046 today. If `blur_sigma` comes back 0.0
and the per-satellite thresholds collapse to one shared value, that additional logic is unneeded
and the existing global-scalar code path is sufficient.
