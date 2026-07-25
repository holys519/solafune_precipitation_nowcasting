# exp056: Mean-Intensity x Normalized-Shape Factorized Architecture

Round 5+ architectural swing (not a feature ablation). Pure input/CV pipeline is copied
byte-for-byte from `exp038` (context_rows: 1, 54ch strict-green input, GroupKFold-by-location split,
same training regime) — the only thing that changes is the model architecture and its loss. This
keeps the comparison to `exp038_sigmafixed` (current green champion, Public 0.68664) a clean
architecture-only ablation.

## Motivation: the oracle-ladder finding

`g_eda/exp002`'s oracle-decomposition ladder (`outputs/g_eda/exp002/*_oracle_ladder.json`,
summarized in `doc/plan/round5_experiment_plan_2026-07-16.md` §8 "E-1の結果" and
`doc/public_scores.md`) recomputes counterfactual tile_rmse on our own OOF predictions:

| Counterfactual | exp018 | exp038 | Reading |
| --- | ---: | ---: | --- |
| `actual` | 0.6093 | 0.6137 | what we actually score |
| `amount_swap` (true tile mean, OUR placement) | **0.5446** | **0.5534** | **+0.06 headroom — dominant term** |
| `mask_swap` (OUR total, true placement) | 0.7111 | 0.7265 | *worse* than actual — our spatial pattern already beats a flat/oracle-mask baseline |

Swapping in the true tile mean intensity while keeping our own spatial placement recovers most of
the remaining gap to the "wall" oracle; swapping in the true spatial mask while keeping our own
amount makes things *worse*. In other words: **for this architecture family, AMOUNT/INTENSITY
error dominates the residual, not spatial placement** — our placement is already better than a
flat baseline, so further encoder/decoder capacity spent on localization has diminishing returns,
while the amount side is still just a per-pixel byproduct of the same shared decoder.

`doc/research_survey_v3_2026-07-16.md` §10 ("最終判断") names this exact factorization —
"mean-intensity/shape factorization" — as the top recommended next step, ahead of quantization-aware
spectral regimes and an OOF blend controller. Recent feature-engineering ablations on the existing
joint architecture (exp047/exp050/exp051, all statistically tied with `exp038_sigmafixed`) motivate
trying an architecturally distinct approach instead of another feature axis.

**Not the same as exp035's `config_tilemean.yaml` arm.** That arm added a `tile_mean_weight` MSE
term directly on top of the existing joint per-pixel `(mu, sigma)` output — a loss-only nudge, no
new head, no explicit shape supervision. exp056 is a structurally different model: two separately
supervised heads (a tile-level scalar and a full-resolution normalized spatial field) that are
multiplied together to *form* the served amount, replacing the joint intensity head entirely.

## Architecture

`FactorizedMeanShapeUNet` in `model.py` (`architecture: factorized_mean_shape`), sharing the exact
same encoder/decoder backbone as exp038's `HighResHurdleLogNormalUNet` (4-level ConvBlock
encoder/decoder, 128x128 internal processing via bilinear resize on the way in, native 41x41 GPM
grid recovered via `adaptive_avg_pool2d` on the way out) so this is a like-for-like comparison, not
confounded by encoder capacity:

- **`rain_head`** — UNCHANGED from exp038. Native-resolution occurrence logits, unweighted BCE.
  Not part of this ablation; the "does it rain here" decision stays exactly as it was.
- **`mean_intensity` head (new)** — global-average-pooled decoder features (the same 128x128xC
  feature map the spatial heads read from, pooled over H,W) → 2-layer MLP (`Linear → SiLU →
  Linear`) → `softplus` → one non-negative scalar per tile, `M`.
- **`shape` head (new)** — a second full high-res decoder head, identical conv-head design to
  `rain_head`, `softplus`'d non-negative at native 41x41, then divided by its own full-tile spatial
  mean (`+ shape_eps`) so `shape.mean(dim=(2,3)) == 1` exactly, for every tile.
- **`aux_mask_head`** — UNCHANGED from exp038 (same `aux_mask_weight`/threshold/Dice design), kept
  so the auxiliary-supervision budget matches exp038's exactly.

Served amount = `M` (broadcast over H,W) `* shape`, capped at `amount_cap` (unchanged default
150.0). Served prediction = `rain_prob * amount` — the same gating convention as
`prediction_from_output` already uses for exp038.

### Design choices made explicitly (not just copied)

1. **"Mean intensity" = full-tile mean, not wet-pixel-only mean.** Confirmed by reading
   `g_eda/exp002/run_oracle_ladder.py`'s `ladder_metrics()`: `amount_swap` rescales the served
   prediction by `truth.mean(dim=(1,2,3)) / pred.mean(dim=(1,2,3))` — a mean over **all** H*W
   pixels, including dry ones, not `truth[wet].mean()`. `mean_intensity` is trained to match this
   exact quantity so the oracle-ladder sanity check below is a fair apples-to-apples comparison.
2. **Shape normalization axis = full tile, not wet-pixel-only.** `shape_target = target /
   (target.mean(dim=(1,2,3)) + eps)`, over the full 41x41 grid (dry pixels included, mostly landing
   near 0). This is consistent with (1): `M * shape` reconstructs a field whose full-tile mean is
   exactly `M`, matching the same convention the mean-intensity head was trained against.
3. **Shape *loss* is wet-pixel-masked, even though the normalization denominator uses the full
   tile.** `FactorizedMeanShapeLoss` computes `(shape_pred - shape_target)^2` only at pixels where
   `target > rain_threshold`, mirroring exp038's own wet-only intensity NLL. All-dry tiles
   naturally contribute zero wet pixels to this mean, so they are automatically zero-weighted with
   no explicit branch or risk of `0/0` NaNs — verified directly in `smoke_test.py`
   (`check_loss_ablation_toggles` runs an all-dry-target batch through the default-weight loss and
   both ablation toggles, asserting finiteness and finite gradients in every case).
4. **`mean_intensity` regression loss is log1p-space MSE** (`MSE(log1p(pred), log1p(true))`),
   matching exp038's own intensity head training in log-space (its `mu` is `E[ln(y)|rain]`) while
   staying finite and well-scaled at `true=0` (dry tiles), unlike a plain `log()`.
5. **Combined amount is still gated by `rain_prob`, per the task's explicit instruction**, even
   though `shape` is *not* itself masked to zero at pixels the shape loss doesn't touch (dry
   pixels are unconstrained by the wet-only shape loss, since the final product is gated by
   `rain_prob` regardless of what `shape` predicts there). This mirrors exp038's own
   `prediction_from_output` convention exactly (`pred = rain_prob * amount`) so the two models are
   comparable end-to-end, but it does mean `rain_prob` and `shape` jointly encode "where," which is
   worth watching in the analysis (see `wet_iou_025` / `spatial_correlation` in
   `oof_sample_metrics.csv`, unchanged from exp038's own diagnostics).

## Loss composition (`losses.py`, `FactorizedMeanShapeLoss`, `loss.name: factorized_mean_shape`)

```
total = bce_weight            * occurrence_BCE                              (unchanged from exp038)
      + mean_intensity_weight * MSE(log1p(M_pred), log1p(tile_mean_true))    (new)
      + shape_weight          * MSE(shape_pred, shape_true)[wet pixels only] (new)
      + aux_mask_weight       * (BCE + Dice on coarse wet/dry mask)          (unchanged from exp038)
      + multiscale_weight_2/4 * pooled MSE on the served pred                (unchanged from exp038)
```

`loss.mean_intensity_weight: 0` and `loss.shape_weight: 0` are independently safe no-ops (the
corresponding head simply gets no gradient from this loss; no crash, no NaN) — both are exercised
in `smoke_test.py::check_loss_ablation_toggles`, including the all-dry-tile degenerate case.

## Oracle-ladder sanity check (new in `analyze_oof.py`)

In addition to the standard `tile_rmse` / official-metric reporting exp038's `analyze_oof.py`
already does (`oof_group_metrics.csv`, `oof_sample_metrics.csv`), this experiment's copy adds
`amount_swap_tile_rmse` (`amount_swap_tile_rmse()` in `analyze_oof.py`) computed identically to
`g_eda/exp002/run_oracle_ladder.py`'s `amount_swap` counterfactual, but applied directly to
**this model's own served predictions** rather than requiring a separate EDA job:

```
pred_mean = pred.mean()      # full-tile mean of THIS model's served prediction
scale = target.mean() / pred_mean       (or fall back to a flat target.mean() field if pred_mean ~ 0)
amount_swap_pred = pred * scale
amount_swap_tile_rmse = rmse(amount_swap_pred, target)
```

This is written per-sample (`oof_sample_metrics.csv`'s new `amount_swap_tile_rmse` column) and
aggregated per group (new `oof_oracle_ladder.csv`, same `fold`/`location`/`satellite`/
`fold_location`/`fold_satellite`/`global` grouping as `oof_group_metrics.csv`, plus
`placement_headroom = actual_tile_rmse - amount_swap_tile_rmse` for each group). The
`analysis_summary.json`'s `oracle_ladder` block also prints a pass/fail `sanity_flag`:

- **How to read it**: if the factorization is wired correctly, `mean_intensity` should already be
  close to the true tile mean most of the time, so rescaling the served prediction to match the
  true mean exactly should land close to the *historical* `amount_swap` rung for this architecture
  family. Two reference points are used depending on how many checkpoints were passed in:
  - **Full 5-fold OOF**: exp018's `amount_swap` was 0.5446 (global), exp038's was 0.5534.
  - **fold0+fold4-only** (this experiment's gating run): fold0/fold4 are individually much easier
    than the OOF-wide average (see `doc/domain_knowledge_review_2026-07-20.md` §3), so the 5-fold
    global numbers above are not a fair comparison. Instead we use exp038's own fold0
    (`amount_swap=0.24878`, n=8590) + fold4 (`amount_swap=0.54929`, n=8737) pooled by sample count
    → **~0.400** as the fold0/4-only reference.
- If `|global_amount_swap_tile_rmse - reference| > 0.08`, the flag reads `FLAG: ... check
  mean_intensity/shape head wiring` — meaning the factorization likely is *not* doing what it's
  supposed to (e.g. `mean_intensity` collapsed to a near-constant value, or `shape` isn't actually
  tracking the true spatial distribution), independent of whatever the raw `tile_rmse` number says.
  This is deliberately a **wiring sanity check**, not a leaderboard metric — a model can pass this
  check and still lose to exp038 on `tile_rmse` (e.g. if the occurrence head or the shape head's
  spatial accuracy regressed even while the amount factorization itself is correctly wired).

## Gating protocol

Fold0/fold4 only (cheap screen), no 5-fold, per the standard pattern used by exp047/exp050/
exp051/exp052/exp054 in this repo. **Do not** run `submit_folds.sh` (all 5 folds) without a human/
orchestrator decision from the fold0/4 results below.

```bash
cd g_experiments/exp056
sbatch singularity_smoke.sh   # CPU-side checks + forward/backward + ablation-toggle sanity, all inside the container

sbatch --parsable --job-name=exp056-fold0 \
  --output=slurm-exp056-fold0-%j.out --error=slurm-exp056-fold0-%j.err \
  singularity_run.sh config.yaml 0
sbatch --parsable --job-name=exp056-fold4 \
  --output=slurm-exp056-fold4-%j.out --error=slurm-exp056-fold4-%j.err \
  singularity_run.sh config.yaml 4
```

After both checkpoints land:

```bash
python analyze_oof.py --config config.yaml \
  --checkpoint ../../g_model/exp056/best_model_fold0.pt \
  --checkpoint ../../g_model/exp056/best_model_fold4.pt
```

produces `outputs/analysis/exp056/oof_group_metrics.csv` (standard tile_rmse per group) and
`outputs/analysis/exp056/oof_oracle_ladder.csv` + the `oracle_ladder` block in
`analysis_summary.json` (the sanity check described above).

**Baseline to beat** (exp038 strict, unmodified): fold0 `tile_rmse=0.28954`, fold4
`tile_rmse=0.59607` (`doc/domain_knowledge_review_2026-07-20.md` §3). Decision rule, per
`doc/plan/round5_experiment_plan_2026-07-16.md` §4/§8 protocol: both folds improve → proceed to
5-fold; one improves and the other ties (noise band ~0.003-0.005) → net-positive, consider 5-fold;
both worse or mixed with no net gain → do not advance. This experiment's implementer submits
fold0/4 only and reports the results; the 5-fold call is the orchestrator's / a human's, not made
here.

## Files

Copied unmodified from `exp038` (same strict-green input pipeline, same CV strategy — this stays a
pure architecture ablation): `dataset.py`, `inference.py`, `make_submission.py`,
`normalize_stats.py`, `norm_stats.json`, `tiff_utils.py`, `amp_utils.py`, `train.py` (only the
default `analysis_dir`/experiment-name fallback strings changed), `run.sh`, `singularity_run.sh`,
`singularity_smoke.sh`, `submit_folds.sh` (job-name/log-path strings only).

New/modified for this experiment: `model.py` (adds `FactorizedMeanShapeUNet`, registers
`architecture: factorized_mean_shape` in `build_model`; all exp038 architectures kept as reference
arms), `losses.py` (adds `FactorizedMeanShapeLoss`, registers `loss.name: factorized_mean_shape`;
`HurdleLogNormalLoss` and friends kept unchanged for reference), `config.yaml` (new architecture/
loss keys; input/split/train sections identical to exp038's `config.yaml`), `smoke_test.py`
(extended with head-shape/normalization assertions and independent loss-ablation-toggle checks),
`analyze_oof.py` (adds the oracle-ladder sanity check described above).
