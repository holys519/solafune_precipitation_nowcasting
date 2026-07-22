# exp054: Amount-Bin Loss-Reweighting Ablation — Which Rainfall Regime Is Worth Optimizing?

Cheap (fold0/fold4-only, no architecture change) ablation to settle a cross-team disagreement
about which precipitation-amount regime dominates prediction error, raised in
`discussion/round6_findings_2026-07-19_ja.md` section 3. Self-contained duplicate of
`g_experiments/exp038` (strict/green champion: `context_rows: 1`, 54ch input,
`HighResHurdleLogNormalUNet`, `HurdleLogNormalLoss`) — same data pipeline, same architecture,
same training regime. The **only** change is a new per-pixel amount-bin weight on the wet-pixel
intensity term of the loss.

## The disagreement (source: round6_findings_2026-07-19_ja.md §3)

Two other competing teams read their own OOF error breakdowns and drew opposite conclusions
about where to invest modeling effort:

- **shafimiakhil**: heavy rain (>=2mm/pixel) is only ~4.13% of pixels but accounts for **85.7%**
  of total error (84.2%/85.7% reproduced across two architectures) — invest in the heavy tail.
- **Bull**: low-intensity rain is "almost worthless" to optimize for; the bulk of error sits in
  a mid, "broadly wet" band — invest there, NOT in the heavy tail. Also reports (§7) that a
  plain MSE loss in log1p space beat their more elaborate loss designs (an Occam's-razor point
  worth keeping in mind against our own hurdle/multiscale/aux-mask complexity).

The `round6_findings` doc's own read is that these two claims are not necessarily contradictory —
"heavy (>=2mm)" and "mid ('broadly wet', maybe ~0.05-2mm)" may just be different definitions of
the same rough region — but which of OUR bins actually carries the error mass, on OUR model and
OOF, had not been measured. Our own oracle-ladder finding (E-1, `g_eda/exp002`, `amount_swap`
delta -0.06) established that **amount error dominates over placement error** in aggregate, but
never broke that down by amount bin. This experiment does that breakdown directly and tests
both hypotheses at once via loss-reweighting, rather than re-deriving them from someone else's
OOF error tables.

## Mechanism

`HurdleLogNormalLoss` (see `losses.py`, unchanged from exp038 except for this addition) fits a
BCE occurrence term (all pixels) plus a log-normal NLL intensity term fitted **only on wet
pixels** (`ln(y)` under `N(mu, sigma^2)`, or MSE-on-`ln(y)` under `sigma_mode: fixed`). exp054
adds a per-pixel weight to that intensity term, keyed by which bin of the **target** (ground
truth) amount the pixel falls into, applied *before* the mean reduction:

```python
bin_weight = amount_bin_weight_map(target, self.amount_bin_weights)   # per-pixel, same shape
intensity = (nll * bin_weight)[wet].mean()
```

Bin edges are **fixed** (not configurable) at `[0, 0.01, 0.1, 0.3, 1.0, 100]` mm — copied
verbatim from `l_eda/exp004/run_fold_anatomy.py`'s `REGIME_EDGES`, which is the existing
regime-bin convention already used elsewhere in this project's EDA (E-4 fold-anatomy analysis).
Only the per-bin **weights** vary across arms:

| bin | range (mm) | meaning |
| --- | --- | --- |
| b0 | `[0, 0.01)` | trace / near-zero |
| b1 | `[0.01, 0.1)` | drizzle |
| b2 | `[0.1, 0.3)` | light |
| b3 | `[0.3, 1.0)` | moderate — top of the "broadly wet" band |
| b4 | `[1.0, 100]` | heavy |

`config.yaml`'s `loss.amount_bin_weights: [1.0, 1.0, 1.0, 1.0, 1.0]` is an exact no-op:
multiplying a float32 tensor by exactly 1.0 introduces no rounding, so this arm is numerically
identical to exp038's original (unweighted) loss. This is verified explicitly in
`smoke_test.py::check_amount_bin_reweighting`, both against a from-scratch reference
implementation of the pre-exp054 formula (`torch.allclose`, atol=1e-6) and via
`torch.equal` between `HurdleLogNormalLoss()` (all defaults) and
`HurdleLogNormalLoss(amount_bin_weights=(1,1,1,1,1))` (explicit uniform weights) on identical
inputs — plus a bin-index sanity check against 11 hand-picked probe values spanning every edge.

## Test arms

| Config | `amount_bin_weights` | Hypothesis tested |
| --- | --- | --- |
| `config.yaml` | `[1.0, 1.0, 1.0, 1.0, 1.0]` | Control — reproduces exp038 exactly |
| `config_midband.yaml` | `[1.0, 1.0, 1.5, 1.5, 1.0]` | Bull: invest in b2/b3 ("broadly wet"), not the tail |
| `config_heavytail.yaml` | `[1.0, 1.0, 1.0, 1.5, 2.0]` | shafimiakhil: invest in b3/b4 (heavy rain) |

Both arms leave b0/b1 (trace/drizzle) at baseline weight in every case — neither hypothesis
argues those matter, and this keeps the two arms differing only in the higher bins so any effect
is attributable to the intended mechanism.

Everything else (architecture, `in_channels: 54`, `context_rows: 1`, `bottleneck_dilations: []`,
`aux_mask_weight`, `multiscale_weight_2/4`, optimizer, batch size, epochs, TTA) is identical to
exp038's `config.yaml`, copied unchanged. `norm_stats.json` is also copied unchanged from exp038
since the data/feature configuration (`features.*` all false, `context_rows: 1`,
`satellite_channels: 16`) is identical, so recomputing normalization stats would just reproduce
the same numbers.

## Analysis: per-amount-bin OOF breakdown

`analyze_oof.py` (copied from exp038 with additions) reports, in a new
`outputs/analysis/exp054{,_midband,_heavytail}/oof_amount_bin_metrics.csv`, one row per bin with:

- **Tile-level**: `tile_rmse` over tiles whose own `target_mean` falls in the bin (mirrors
  `l_eda/exp004/run_fold_anatomy.py`'s regime-bin grouping, so directly comparable to that EDA's
  tables), plus `tile_count`/`tile_share`.
- **Pixel-level**: `positive_pixel_rmse` and `positive_pixel_mae` over wet pixels (`target > 0`)
  whose own value falls in the bin (independent of which tile they sit in — a mostly-dry tile
  can still contain a few b4 pixels), plus `positive_pixel_count`,
  `positive_pixel_share_of_wet`, and `positive_pixel_share_of_sse` (each bin's share of total
  wet-pixel squared error — the number that actually adjudicates the two hypotheses).

This is also written into `oof_group_metrics.csv` under `group_type="amount_bin_tile_mean"` (tile
breakdown, reusing the existing grouping machinery) and as an `amount_bin_tile_mean` column on
every row of `oof_sample_metrics.csv`, for ad-hoc slicing.

**How to read the results**: for each config (control / midband / heavytail), compare
`positive_pixel_rmse` and `positive_pixel_share_of_sse` per bin against the control. If
`config_midband.yaml` reduces `positive_pixel_rmse` in b2/b3 without regressing b4 (and the
aggregate `tile_rmse` improves), Bull's hypothesis wins. If `config_heavytail.yaml` reduces
`positive_pixel_rmse` in b4 and that improves the aggregate more, shafimiakhil's hypothesis wins.
If neither aggregate improves over control, the reweighting mechanism itself may be net-neutral
on this data (consistent with the discussion's own note that the two hurdle-loss terms were
originally left deliberately unweighted because "tail/class re-weighting was measured
net-negative on this dataset by the discussion authors" — see `losses.py` docstring) — a
legitimate, useful negative result.

## Gating protocol

Fold0/fold4 only (cheap screen), no 5-fold, per the standard pattern used by exp047/exp048/
exp050/exp051 gate experiments in this repo:

```bash
cd g_experiments/exp054
sbatch singularity_smoke.sh

sbatch --job-name="exp054-midband-fold0" \
  --output=slurm-exp054-midband-fold0-%j.out --error=slurm-exp054-midband-fold0-%j.err \
  singularity_run.sh config_midband.yaml 0
sbatch --job-name="exp054-midband-fold4" \
  --output=slurm-exp054-midband-fold4-%j.out --error=slurm-exp054-midband-fold4-%j.err \
  singularity_run.sh config_midband.yaml 4

sbatch --job-name="exp054-heavytail-fold0" \
  --output=slurm-exp054-heavytail-fold0-%j.out --error=slurm-exp054-heavytail-fold0-%j.err \
  singularity_run.sh config_heavytail.yaml 0
sbatch --job-name="exp054-heavytail-fold4" \
  --output=slurm-exp054-heavytail-fold4-%j.out --error=slurm-exp054-heavytail-fold4-%j.err \
  singularity_run.sh config_heavytail.yaml 4
```

After each fold's checkpoint lands, run `python analyze_oof.py --config config_midband.yaml
--checkpoint <path to best_model_fold0.pt> --checkpoint <path to best_model_fold4.pt>` (and same
for heavytail) to produce the per-bin CSVs described above.

**Baseline to beat** (exp038 strict, unmodified): fold0 `tile_rmse=0.28954`, fold4
`tile_rmse=0.59607`. Neither test arm is expected to necessarily beat this on the aggregate — the
point of this experiment is the per-bin breakdown, not a leaderboard attempt. If one arm *does*
improve the aggregate fold0/4 `tile_rmse` over exp038 strict, that is itself notable and worth a
5-fold follow-up under its own experiment number.

## Files

Standard exp038-derived file set: `config.yaml` / `config_midband.yaml` /
`config_heavytail.yaml`, `dataset.py`, `model.py`, `losses.py` (amount-bin mechanism added),
`train.py`, `inference.py`, `analyze_oof.py` (per-bin breakdown added), `make_submission.py`,
`normalize_stats.py`, `norm_stats.json` (copied, not recomputed), `smoke_test.py` (amount-bin
unit check added, `ARMS` covers all three configs), `run.sh`, `singularity_run.sh`,
`singularity_smoke.sh`, `submit_folds.sh` (kept for completeness/future 5-fold follow-up, not
used in this gate).
