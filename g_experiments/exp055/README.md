# exp055: green-only manifest blend submission

Consumes `g_eda/exp011/recommended_weights.json` the same way `g_experiments/exp036` consumed
`g_eda/exp003`'s recommendation — but generalized to N named manifest sources (not a hardcoded
exp016/017/018 triple), and with the overlap-patch step removed entirely rather than merely
defaulted off.

## Rule compliance, enforced not just documented

1. **Green-only, checked twice.** `g_eda/exp011` already refuses to compute weights over a
   red/amber source. `build_submission.py` re-checks every manifest source against
   `doc/submission_registry.md` again at build time via `registry_guard.assert_all_green` —
   independently, so a stale/edited `recommended_weights.json` cannot smuggle a red source in.
   This raises `RegistryComplianceError` and aborts before any prediction file is opened.
2. **No overlap patch, structurally.** Unlike `exp036` (which applies `exp014`'s overlap patch by
   default and needs a `--skip-patch` flag to avoid it), `build_submission.py` never imports
   `apply_overlap.py` at all. There is no flag that turns overlap-patching on, because the ruling
   from 2026-07-20 makes it permanently disqualifying (reverse engineering).
3. **Causal-only smoothing, never non-causal.** If `g_eda/exp010/recommended_causal_weights.json`
   exists, its `temporal_smoothing` config is applied via `g_eda/exp010/causal_smoothing.py`'s
   `apply_temporal_smoothing` — with a hard assertion here (in addition to that module's own
   internal guard) that `next_weight == 0` and `causal_only == true`. If the file doesn't exist
   yet, smoothing is skipped and the submission is the raw blend only (recorded in the analysis
   summary JSON either way, so it's traceable whether smoothing was actually applied).

## Prerequisites

1. `g_eda/exp011` has run both phases for every source currently in `g_eda/exp011/sources.json`
   (`recommended_weights.json` exists and its `manifest_sources` matches the manifest exactly —
   `build_submission.py` refuses to run otherwise, forcing you to re-run
   `optimize_blend.py --analyze` after editing the manifest).
2. Each source's evaluation predictions already exist under `outputs/submissions/{name}/test_files/`
   (`eval_pred_dir` in `sources.json`) — for `exp038_sigmafixed` and `exp040_metric` these were
   already generated as part of each experiment's own `make_submission.py` run, so exp055 needs
   **no new inference at all** for today's 2-source build.

## Run

```bash
cd g_experiments/exp055
sbatch singularity_run.sh --dry-run                       # show weights/sources, no I/O
sbatch singularity_run.sh                                  # global weights, blend (+causal if ready), zip
sbatch singularity_run.sh --scheme per_satellite            # per-satellite weights
sbatch singularity_run.sh --skip-causal-smoothing           # raw blend only
```

CPU-only by design (`singularity_run.sh` requests no GPU) — blending already-generated `.tif`
predictions and zipping needs no GPU, unlike `exp036`'s wrapper which requests one it never uses.

## Outputs

- `outputs/submissions/exp055_<scheme>[_causal].zip`
- raw prediction dir: `outputs/submissions/exp055/<scheme>[_causal]_raw/`
- manifest: `outputs/analysis/exp055/analysis_summary_<scheme>[_causal].json` (weights used,
  sources, sha256, whether causal smoothing was actually applied)

## 2026-07-21/22 result

Ran end-to-end today on the 2-source manifest (`exp038_sigmafixed` + `exp040_metric`), global
weights (48/52), producing two candidate zips:

- `outputs/submissions/exp055_global_blend.zip` — raw OOF-optimal blend only
  (`--skip-causal-smoothing`).
- `outputs/submissions/exp055_global_blend_causal.zip` — blend + `g_eda/exp010`'s causal-only
  smoothing/blur/per-satellite-threshold stack (that recommendation landed mid-run today).

Both validated as well-formed zips (29091 entries: 1 CSV + 29090 `.tif`, no corrupt entries).
OOF numbers (computed directly on the cached OOF arrays, matching each zip's post-process exactly):

| Stack | OOF tile_rmse | delta vs exp038_sigmafixed solo |
| --- | ---: | ---: |
| exp038_sigmafixed solo | 0.60852 | — |
| raw blend (48/52) | 0.59982 | -0.00870 |
| blend + exp010 causal/blur/threshold | 0.59984 | -0.00868 |

**The exp010 post-process stack is a wash on top of the blend (+0.00002, inside noise)** — its
coefficients were OOF-tuned against exp038_sigmafixed *solo*, not this blend, so they don't
transfer meaningfully. All of today's real gain is from the blend itself. Re-tuning
smoothing/blur/threshold specifically against the blend's OOF predictions (not attempted today)
is the obvious next step before choosing which of the two zips to actually submit; until then
either is a reasonable candidate since they're statistically indistinguishable.

## Extending

Nothing in this directory needs to change when a new green source lands — add it to
`g_eda/exp011/sources.json` (see that experiment's README), re-run `g_eda/exp011`'s two phases,
then re-run this experiment's `singularity_run.sh`. The set of sources actually blended always
comes from `recommended_weights.json["manifest_sources"]`, never a hardcoded list here.
