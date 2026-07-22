# g_eda/exp011: manifest-driven green-only OOF blend optimizer

Direct successor to `g_eda/exp003`'s OOF simplex search, rebuilt for the post-2026-07-20 rules
world. `g_eda/exp003`'s caches/recommendations are **red-derived** (exp016/017/018-family
checkpoints, all successor-row / `context_rows: 2`) per `doc/submission_registry.md` and this
experiment does not read anything from `outputs/g_eda/exp003/` — it starts a clean, green-only
OOF cache tree under `outputs/g_eda/exp011/`.

## Why a manifest instead of a script per source

`g_eda/exp003` added a new Python file for every additional blend member
(`run_4source_blend.py`, `run_5source_blend.py`, `run_4way_simplex.py`, ...). That doesn't scale
to the next 1-2 weeks, where `exp047`/`exp050`/`exp051`/`exp052`/`exp053`/`exp054` are all
candidate green sources that may or may not pass their fold gates. Instead:

- **`sources.json`** is the single list of currently-included green sources.
- **`optimize_blend.py`** reads that list and picks its search strategy automatically from the
  source count (see below) — no code changes needed when a source is added or removed.
- Every source is checked against `doc/submission_registry.md` (via `registry_guard.py`) before
  its cache is touched, so an accidental red/amber addition to the manifest fails loudly instead
  of silently entering a "green" blend.

### Adding a new source once it passes its fold gate

Two edits, no code changes:

1. Add one entry to **`green_allowlist.json`** (`"expNNN": {"status": "green", "registry_note":
   "..."}`) *after actually reading its status in `doc/submission_registry.md` yourself* — this
   file is a maintained cross-check, not a rubber stamp.
2. Add one object to the `"sources"` list in **`sources.json`**:

   ```json
   {
     "name": "exp047",
     "module_dir": "exp047",
     "checkpoint_dir": "exp047",
     "eval_pred_dir": "outputs/submissions/exp047/test_files",
     "oof_cache": "outputs/g_eda/exp011/exp047_oof_pred.npz",
     "public_rmse_solo": null,
     "notes": "local solar time / hemisphere / day-of-year features"
   }
   ```

Then run phase 1 for just the new source and re-run phase 2 for everyone:

```bash
cd g_eda/exp011
sbatch singularity_cache.sh exp047     # phase 1: OOF cache for the new source only
sbatch singularity_analyze.sh          # phase 2: re-optimize with all manifest sources
```

Nothing else in the pipeline (`g_experiments/exp055`) needs to change — it reads whichever
sources are in `recommended_weights.json` off of `sources.json` at build time.

## Files

- `sources.json` — the manifest (see above).
- `green_allowlist.json` — hand-maintained green/red cross-check, read by `registry_guard.py`.
- `registry_guard.py` — `assert_green(name)` / `assert_all_green(names)`. Two independent checks
  (allowlist status + a live text scan of `doc/submission_registry.md` for red markers) must both
  pass; fails closed (unknown source names, missing registry mentions, or any red/amber signal all
  raise `RegistryComplianceError`). Also used by `g_experiments/exp055` at build time.
- `optimize_blend.py` — phase 1 (`--cache NAME`, GPU or CPU inference from an existing
  checkpoint, not training) and phase 2 (`--analyze`, CPU-only numpy). See its docstring for the
  N-aware search strategy (1 source: trivial; 2: full 1-D ladder; 3: full simplex grid; >3:
  greedy forward blend-in, matching `g_eda/exp003`'s measured-safe approximation for that regime).
- `singularity_cache.sh <source_name>` — GPU sbatch wrapper for phase 1 (inference on an existing
  checkpoint; this is the only GPU-touching step in this experiment, and it is inference, not
  training).
- `singularity_analyze.sh` — CPU-only sbatch wrapper for phase 2 (no `--gpus-per-node`, no
  `--nv` — this stage is pure numpy over the cached arrays).
- `recommended_weights.json` / `BLEND_CURVE.md` — phase 2 output, consumed by
  `g_experiments/exp055`.

## Causal-only smoothing hook (`g_eda/exp010`)

`g_eda/exp010` is being built in parallel to recommend causal-only (never non-causal) temporal
smoothing coefficients, to be layered on top of the blend. `optimize_blend.py` checks for
`g_eda/exp010/recommended_causal_weights.json` at analyze time:

- **If present**: loaded into `recommended_weights.json["causal_smoothing_hook"]`, with a hard
  assertion that `temporal_smoothing.next_weight == 0` (refuses to wire in anything non-causal,
  per the 2026-07-20 ruling) and `causal_only == true`.
- **If absent** (the case as of this run): recorded as `"available": false` with a note that the
  hook is wired but inactive. `g_experiments/exp055` applies the actual smoothing at submission
  build time (once the file exists), not this script — this script only surfaces whether it's
  ready.

## Run

```bash
cd g_eda/exp011
sbatch singularity_cache.sh exp038_sigmafixed
sbatch singularity_cache.sh exp040_metric
sbatch singularity_analyze.sh              # after both caches land
```

Outputs land in `outputs/g_eda/exp011/`: `{source}_oof_pred.npz` (fp16 cache, reusable),
`blend_curve.csv` / `simplex_grid.csv` / `greedy_path.csv` (whichever the source count produced),
`recommended_weights.json`, `BLEND_CURVE.md`.

## 2026-07-21/22 result (2-source manifest: exp038_sigmafixed + exp040_metric)

Ran end-to-end today (both OOF caches built via `singularity_cache.sh`, ~2min each; analyze via
`optimize_blend.py --analyze`, 2-way ladder branch, 40686 tiles):

| | OOF tile_rmse |
| --- | ---: |
| exp038_sigmafixed solo | 0.60852 |
| exp040_metric solo | 0.60772 |
| **global blend (48/52 exp038_sigmafixed/exp040_metric)** | **0.59982** |
| per-satellite composed blend | 0.59982 (goes 0.80299 / himawari 0.74521 / meteosat 0.36726) |

Blend delta vs exp038_sigmafixed solo: **-0.00870**; vs best solo (exp040_metric): -0.00790. This
is well outside this project's ~0.002-0.004 OOF noise band (see `doc/public_scores.md`'s E-3
audit references) — real diversity gain from a weaker-solo-but-distinct architecture, as expected
per the registry's rationale for keeping exp040_metric as a Track G3 blend candidate.

`g_eda/exp010/recommended_causal_weights.json` landed mid-run and was picked up automatically
(see `BLEND_CURVE.md`'s causal-hook section) — but note its coefficients were OOF-tuned against
**exp038_sigmafixed solo**, not this blend; layering them on the blend as-is measured as a wash
(see `g_experiments/exp055`'s README for the exact number). Re-tuning smoothing/blur/threshold
specifically against the blend's OOF predictions is a natural next step, not done today.

Full grids: `outputs/g_eda/exp011/blend_curve.csv` (101-step global ladder).
