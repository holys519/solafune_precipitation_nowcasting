# Task Tickets

Last updated: 2026-07-10

Tracks concrete follow-up work after `exp001` (public RMSE 0.7531995875751526). See
`doc/exp001_retrospective.md` for the analysis behind the exp002 tickets, and
`doc/research_survey.md` for the research-driven roadmap after exp003. See
`doc/public_scores.md` for the current leaderboard table,
`doc/experiment_tracking_design.md` for the registry/graphing design behind the `I-XXX` tickets
below, `doc/research_survey_v2.md` for the round-2 literature survey behind the `G-026`-`G-029`
tickets, and `doc/discussion_insights.md` for the official-discussion analysis behind the
`G-030`-`G-034` (Round 4 / exp016-exp020) tickets.

## Track convention

- **L-XXX tickets** run under `l_experiments/` — local dev, correctness checks, small samples,
  fast iteration on the current 3090x2 workstation. Nothing here needs multi-day compute budget.
- **G-XXX tickets** run under `g_experiments/` — full-scale/production training, anything that
  needs a full epoch budget over the full dataset, multi-fold ensembles, or scaled-up hardware.
- **GPU scaling work (A100 x2 / x4) is G-track only.** It is a production/throughput concern, not
  a correctness concern, so it never belongs in `l_experiments/`.
- **I-XXX tickets** are infra/tooling — experiment-result tracking, registries, plotting — not a
  modeling change, no GPU needed.

## Open Tickets

| ID | Track | Exp | Title | Priority | Status |
| --- | --- | --- | --- | --- | --- |
| L-001 | l_experiments | exp002 | Pipeline overhaul: normalization, weighted loss, augmentation, GroupKFold, anti-aliased resize | P0 | Done (this session) |
| G-001 | g_experiments | exp002 | Full-dataset exp002 training on current 3090x2 | P0 | Ready to run |
| G-002 | g_experiments | exp002 | 5-fold GroupKFold full training + fold ensemble at inference | P0 | Ready to run (loop `run.sh` over fold 0-4) |
| G-003 | g_experiments | exp002 | A100x2 scaled config (larger batch/workers) | P1 | Config drafted, needs a real A100 run to tune batch size/LR |
| G-004 | g_experiments | exp002 | A100x4 scaled config + DDP migration | P1 | Config drafted; DDP migration not implemented (see notes) |
| L-002 | l_experiments | exp002 | Architecture ablation: CompactUNet vs `smp_unet` (pretrained efficientnet-b0) on a fixed fold | P1 | Not started |
| G-005 | g_experiments | exp003 | Promote winning exp002 architecture, full 5-fold + ensemble | P1 | Blocked on L-002 |
| G-008 | g_experiments | exp005 | Per-observation-time encoder before fusion (currently: naive channel concat of 3 times) | P2 | Minimal implementation added |
| G-009 | g_experiments | exp007 | Post-processing sweep: clip thresholds, calibration, blending fold predictions vs simple average | P2 | Minimal ensemble implementation added |
| G-006 | g_experiments | — | Harmonize checkpoint schema so `extract_scores.py` also scans `.pt` files under `l_model`/`g_model` and reads `best_rmse` in addition to `best_score`/`.pth` | P2 | Done (this session, backward compatible) |
| G-010 | g_experiments | exp004 | Two-head rain/no-rain detection + rain amount regression with OOF threshold sweep | P1 | Minimal implementation added |
| G-011 | g_experiments | exp006 | Satellite-specific input stems/adapters for Himawari, GOES, and Meteosat | P2 | Minimal implementation added |
| G-007 | g_experiments | exp007 | OOF-weighted ensemble and post-processing sweep across exp003+ checkpoints | P1 | Minimal implementation added |
| G-012 | g_experiments | exp008 | Official tile-RMSE diagnostics and value-threshold drizzle post-processing on exp004 checkpoints | P0 | Implemented |
| G-013 | g_experiments | exp009 | Successor-row satellite frames (`T/T+10/T+20`) as additional inputs | P0 | Implemented |
| G-014 | g_experiments | exp010 | Dataset cleanup: suspicious labels, IR zero nodata, GOES 4ch native remap | P1 | Implemented |
| G-015 | g_experiments | exp011 | Combine satellite-specific adapter with two-head rain model | P1 | Implemented |

## Ticket Details

### L-001 — exp002 pipeline overhaul (done)

Implemented in `l_experiments/exp002/`:
- `normalize_stats.py`: computes per-`(satellite, band)` mean/std from a deterministic sample of
  train rows, writes `norm_stats.json`. Run once (CPU-only) before training.
- `dataset.py`: applies the normalization stats instead of `/255`; downsizes satellite tensors with
  `adaptive_avg_pool2d` (anti-aliased) instead of bilinear interpolation; adds synchronized
  flip/rot90 augmentation for train split; `make_group_kfold_split` using `sklearn.GroupKFold` over
  `name_location`, selects one of 5 folds by index.
- `losses.py`: `WeightedMSELoss` — per-pixel weight `1 + pos_weight * (target > 0)`.
- `model.py`: `CompactUNet` (unchanged from exp001) plus `SMPUNet` (segmentation_models_pytorch
  U-Net, `encoder_name=efficientnet-b0`, `encoder_weights=imagenet`, `in_channels` set to match our
  input), selected via `model.architecture` in config.
- `train.py` / `inference.py`: fold-aware, use full data by default (`max_train_samples: null`),
  more epochs, optional flip-TTA and multi-checkpoint ensembling at inference.

Local run uses a **reduced epoch/data budget by default** relative to the g_experiments version so
it finishes quickly for correctness checks — see `l_experiments/exp002/README.md`. The full-scale
run is G-001/G-002, not this ticket.

### G-001 / G-002 — full-scale training + 5-fold ensemble

Run on the current 3090x2 box via `g_experiments/exp002/run.sh <config> <fold>` (plain bash, no
Slurm needed) or `sbatch singularity_run.sh` on a Slurm cluster. Loop folds 0-4 for G-002, then
average the 5 checkpoints' predictions at inference (`inference.py` already supports multiple
`--checkpoint` args).

Acceptance: report 5-fold mean/std CV RMSE in `EXPERIMENT_REPORT.md`, plus the resulting public
score after submission.

### G-003 / G-004 — A100 scaling

`g_experiments/exp002/config_a100x2.yaml` and `config_a100x4.yaml` raise `batch_size` and
`num_workers` to use the larger VRAM/CPU headroom (drafted this session, not yet run — we do not
have A100 hardware in this sandbox to validate against). `train.py` still uses
`nn.DataParallel` for multi-GPU, which is simple but scales sub-linearly past 2 GPUs.

**G-004 follow-up (not implemented yet):** migrate to `DistributedDataParallel` with a `torchrun`
launcher before running on 4 GPUs for real — DataParallel's single-process gather/scatter becomes
the bottleneck at that GPU count. Until then, G-004's 4-GPU config is usable but expect a Y then
not-quite-4x throughput.

### L-002 / G-005 — architecture ablation

`model.architecture: compact_unet | smp_unet` is implemented and config-selectable in exp002 code
right now. The ablation itself (train both on the same fold, same data budget, compare CV RMSE) is
not run yet — do it locally on a single fold before committing a production 5-fold run to whichever
architecture wins, since a pretrained encoder roughly doubles+ parameter count and per-epoch cost.

### G-008 — temporal encoder (exp005)

Currently the 3 observation times are just concatenated as extra channels with no explicit temporal
inductive bias (order is implicit in channel position). Worth trying a shared per-time-step
encoder (weight-shared across the 3 slots) whose outputs are fused (concat or small attention) before
the U-Net decoder, so the model isn't forced to learn 3 separate representations for the same
sensor at slightly different times from scratch.

### G-009 — post-processing (exp007)

Non-negative clipping is already done. Still open: calibration curve (is predicted vs actual biased
at high rain rates?), and whether averaging multiple fold predictions beats picking a single best
fold.

### G-010 — two-head rain detection + amount regression (exp004)

Research survey priority 1. Split the task into a rain/no-rain head and a precipitation amount head
to address the large zero-rain imbalance. Add OOF threshold sweep for the rain probability threshold
and log detection metrics (precision, recall, CSI, false alarm ratio) in addition to RMSE.

Acceptance: `exp004` can produce a full submission zip, plus OOF CSVs that show whether gains/losses
come from false-positive rain reduction, missed light rain, or amount regression.

### G-011 — satellite-specific adapters (exp006)

Research survey priority 3. Add small satellite-specific input stems for Himawari, GOES, and
Meteosat before the shared backbone. Evaluate per-satellite OOF RMSE to confirm the adapter improves
domain handling rather than overfitting one sensor.

### G-007 — OOF ensemble and post-processing sweep (exp007)

Research survey priority 4. Use OOF predictions from exp003+ to tune model weights, per-satellite
scale/bias, rain-probability thresholds, non-negative clipping, and optional upper clipping. Avoid
any leaderboard-driven tuning.

### G-006 — checkpoint/extract_scores harmonization (done)

`extract_scores.py` previously only scanned `g_model*/*.pth` for a `best_score` key.
`l_experiments`/`g_experiments` exp001/exp002 checkpoints are `.pt` with a `best_rmse` key. Updated
`extract_scores.py` to scan both `.pth` and `.pt`, and read `best_score` or `best_rmse` (lower is
better for RMSE, so scores are sorted/reported as-is — no inversion needed, just noted the metric
name is RMSE not an accuracy-style "score").

### G-012 — official metric and drizzle post-processing (exp008)

Uses `exp004` checkpoints by default through `paths.source_model_dir`, then writes separate exp008
analysis/submission outputs. Adds `tile_rmse` as the primary selection metric and writes
`oof_value_threshold_sweep.csv` for `pred < threshold -> 0` post-processing. Temporal smoothing is
implemented but disabled by default in config.

### G-013 — successor-row frames (exp009)

Extends the input from 3 observation slots to 6 slots: current row plus the same-location successor
row when it exists. This makes `model.in_channels=105`. Missing successor rows are zero-filled with
mask channels set to 0.

### G-014 — dataset cleanup (exp010)

Drops the five suspicious first-frame train targets identified in discussion, treats raw zeros in IR
bands as nodata by setting normalized values to 0, and remaps GOES `282x282x4` native files into
`C01/C02/C03/C05` slots.

### G-015 — satellite-adapter two-head model (exp011)

Adds `satellite_adapter_two_head_unet`: satellite-specific input stems selected from the one-hot
satellite maps, shared U-Net body, and two heads for rain probability and amount.

## Insights From l_eda/exp001 and outputs/analysis (2026-07-09)

Source: `outputs/l_eda/exp001/EDA_IMAGE_REPORT.md` (+ its CSVs) and
`outputs/analysis/exp003..exp011/analysis_summary.json`, cross-referenced with
`doc/public_scores.md`.

1. **Successor-row frames (exp009) is the single biggest lever found so far** — public RMSE
   0.7153, ~0.01 better than the exp004 anchor and better than any architecture-only change
   (adapters, two-head, temporal fusion each individually). Feeding more real temporal context
   beats making the network cleverer about the 3 frames it already has.
2. **`l_eda` temporal-motion EDA explains why**: `temporal_motion.csv`/`EDA_IMAGE_REPORT.md` show
   phase displacement between consecutive frames is small (sub-pixel to a few pixels; "tiny phase
   displacement supports simple frame stacking... over heavy recurrent models"). This matches the
   leaderboard: exp005's shared-encoder temporal fusion (0.7446) underperformed the simpler exp004
   baseline (0.7253), while naive successor-frame concatenation (exp009) won. Recurrent/optical-flow
   temporal modeling is not where the signal is — more simply-stacked frames is.
3. **exp009 (successor rows, 105ch, no adapter) and exp011 (adapter+two-head, 54ch, no successor
   rows) have never been combined** — confirmed via `g_experiments/exp011/config.yaml`
   (`in_channels: 54`, no `context_rows`). Since both individually beat the exp004 anchor and
   target different axes (temporal context vs. sensor-specific stems), combining them is the
   highest-expected-value untried experiment (see G-016).
4. **Parallax/registration offsets are real, stable, and location+satellite specific** —
   `parallax_shift_by_location.csv` shows median shifts up to ~5px (e.g. jakarta/himawari
   `dy=1, dx=-5`, correlation gain +0.089 after alignment). No experiment so far corrects for this
   before stacking frames; it is a plausible source of blur/misalignment in the 41x41 target (see
   G-017).
5. **Spectral/texture band means correlate only weakly with rain** (band01/band02 mean vs.
   target_positive_ratio/mean, `pearson_corr` ≈ 0.14–0.16, from `spectral_texture_correlations.csv`)
   — weaker than expected. Low priority for hand-crafted global scalar features; the spatial field
   already used by the U-Net is doing the real work, so effort is better spent on temporal/spatial
   preprocessing (points 2-4) than new scalar band-statistic inputs.
6. **Train/eval feature-distance gaps are moderate, not extreme**
   (`train_eval_feature_shift.csv`: nearest_distance 0.01–0.05 vs. median_train_distance 0.04–0.12)
   — no catastrophically out-of-distribution eval location. Supports keeping global/per-satellite
   calibration (already done in exp003+) rather than investing in per-location domain adaptation.
7. **exp010 (dataset cleanup) underperforms exp004/008/009/011 on public score** (0.7349) despite a
   competitive OOF `tile_rmse` (0.6498, close to exp009's 0.6239). This mismatch between OOF and
   public score — bigger than any other experiment's OOF/LB gap — suggests one of the three cleanup
   changes (suspicious-label drop, IR-zero-as-nodata, GOES 4ch remap) doesn't generalize; it should
   not be folded into a future default until isolated (see G-020).
8. **exp007's ensemble is stale**: it only combines exp003–006 checkpoints
   (`outputs/analysis/exp007/analysis_summary.json`), all of which are now worse on public score
   than exp008/009/011. An OOF-weighted ensemble has not been tried over the current best set (see
   G-019).
9. **Rain-detection quality plateaus around CSI≈0.44–0.46** across exp008/009/010/011
   (`best_rain_threshold.csi` in each `analysis_summary.json`), with only the probability threshold
   tuned so far, not the loss shaping the rain head itself (see G-021).
10. **`analysis_summary.json` schema drifted mid-project**: exp003–006 use
    `training.best_rmse_mean/std`; exp008–011 rename this to `training.best_metric_mean/std` and add
    `selection_metric`/`oof_official_metric`/`tile_rmse`; exp007 (ensemble) has a different schema
    with no `oof_global` at all. This blocks writing one parser across all experiments (see
    `doc/experiment_tracking_design.md`).
11. **All of `outputs/` — including every `analysis_summary.json`, OOF CSV, and l_eda figure — is
    git-ignored.** The only durable, version-controlled score record today is
    `doc/public_scores.md` (hand-maintained) plus each experiment's `config.yaml`/`README.md`. The
    richer CV/OOF/calibration history exists only as scattered files across the HPC group directory
    (`/group/project143/yamamoto/...`, visible inside the JSON paths) and this NAS mount — with no
    accumulation strategy for graphing trends across dozens of future cloud runs. See
    `doc/experiment_tracking_design.md` for the proposed fix (I-001).

## Round 2 Open Tickets (2026-07-09, post exp008-011)

| ID | Track | Exp | Title | Priority | Status |
| --- | --- | --- | --- | --- | --- |
| G-016 | g_experiments | exp012 | Combine successor-row frames (exp009) with satellite-adapter two-head (exp011) | P0 | Implemented (`g_experiments/exp012`, smoke-tested on 3090; full 5-fold run pending) |
| G-017 | g_experiments | exp013 | Per-(location, satellite) parallax registration before frame stacking | P1 | Implemented (`g_experiments/exp013`, `config_registration_only.yaml` isolates it; single-fold A/B pending) |
| G-018 | g_experiments | exp013 | Extend context beyond 1 successor row (predecessor + successor, or wider window) | P1 | Implemented (`data.context_offsets: [-1, 0, 1]`, `config_context_only.yaml` isolates it) |
| G-019 | g_experiments | exp007 | Refresh OOF-weighted ensemble over exp008/009/011 (+ exp010 pending G-020) | P0 | Not started |
| G-020 | g_experiments | exp010 | Ablate the 3 cleanup changes individually to find the one hurting public score | P1 | Not started |
| G-021 | g_experiments | exp009/011 | Dice/Focal loss on the rain-probability head to push CSI past ~0.46 | P2 | Not started |
| I-001 | infra | — | Experiment registry + trend-graphing pipeline (see `doc/experiment_tracking_design.md`) | P0 | Design documented, not implemented |
| G-022 | g_experiments | exp014 | Tile-overlap GPM copy: overwrite eval overlap regions with same-time train GPM truth (see `doc/tile_overlap_discovery.md`) | P0 | Done — public RMSE 0.6968727727408199 (2026/07/10 12:44:13), new best, -0.0184711171697818 vs exp009 base. See `doc/public_scores.md` |
| G-023 | g_experiments | exp015 | Solar/visible-brightness input feature (day/night conditioning) | P1 | Justified by l_eda/exp002: same-rain-regime night error +16%, visible bands collapse at night |
| G-024 | g_experiments | exp013 | Successor-weighted context variant `context_offsets: [0, 1, 2]` | P2 | Config-only variant; IR->rain lag curve peaks at +30min and decays fast backwards |
| G-025 | g_experiments | — | Post-processing: snap predictions to the 0.01 value grid (targets are 99.6% multiples of 0.01, min positive 0.01) | P3 | Not started; expect tiny but safe gain, sweep on OOF first |

## Round 3 Open Tickets (2026-07-10, from `doc/research_survey_v2.md`)

Round 2's OOF/public analysis (`doc/data_characteristics_review.md` §1.3, §1.5) found mid+heavy-rain
tiles are 41.3% of tiles but 87% of the `tile_rmse` error budget, and heavy-rain amplitude is
underestimated ~4x (`pred_max/target_max` ≈ 0.25 at `target_max >= 10`). Round 3 targets that
specifically, via a literature survey filtered through our own measured constraints (small tile,
~20 independent episodes, small motion, `tile_rmse` metric) — see `doc/research_survey_v2.md` for
the full reasoning and reference list. Both new citations (arxiv 2605.12762, arxiv 2511.11197) were
independently re-fetched and verified to match the survey's claims before these tickets were added.

| ID | Track | Exp | Title | Priority | Status |
| --- | --- | --- | --- | --- | --- |
| G-026 | g_experiments | — | Multi-quantile amount head (pinball loss, τ∈{0.5,0.9,0.99} + monotonicity), OOF-swept for `tile_rmse` | — | **Superseded by G-030** (2026-07-10): two teams' measured evidence in the official discussion (`doc/discussion_insights.md` §2) shows serving quantiles costs 0.006–0.15 LB because the RMSE-optimal serving is E[Y\|X] and wet pixels are log-normal with mean/median≈3.8x. The hurdle log-normal head (G-030/exp016) targets the same 4x underestimation with the mathematically correct estimator |
| G-027a | g_experiments | exp015 | Post-processing: OOF-fit isotonic calibration of amount-head output (replaces linear scale/bias in `oof_calibration.json`) | P0 | Done — public RMSE 0.7096658388930687 (2026/07/10 08:47:38), -0.0056780510175330 vs exp009 base. Not yet stacked with exp014's tile-overlap patch |
| G-027b | g_experiments | — | Input feature: per-satellite IR->rain empirical curve (`bt_rain_response.csv`) evaluated at coldest IR band, as auxiliary input channel | P1 | Not started |
| G-028 | g_experiments | — | Architecture: dilated-conv bottleneck (dilation 2,4) instead of extra downsampling in `enc2`/`enc3` | P2 | Not started; exploratory, motivated by `radial_power_spectrum.csv` (81% of target power at wavenumber 1-2) |
| G-029 | g_experiments | — | Loss: radial-power-spectrum auxiliary term comparing predicted vs. target spectra (reuses `l_eda/exp002` FFT code) | P3 | Not started; run only after G-026/G-027 to keep comparisons readable |

G-023 (solar/visible-brightness feature) is updated, not replaced: `doc/research_survey_v2.md` §5
adds a precedent for day/twilight/night **regime gating** (e.g. a 3-bucket one-hot or FiLM-style
conditioning) rather than assuming a single continuous solar-zenith-angle channel captures the effect
linearly enough for a from-scratch ~2M-param CNN trained on ~20 episodes. Compare both variants via
OOF on `outputs/l_eda/exp002/oof_daynight_conditional.csv`'s existing split when implementing G-023.

### G-026 — exp015: multi-quantile amount head

Add quantile output heads (τ ∈ {0.5, 0.9, 0.99}) to the amount branch of `two_head_compact_unet` /
`satellite_adapter_two_head_unet`, trained with pinball loss plus a monotonicity constraint across
heads (see `doc/research_survey_v2.md` §1 for the `Q-SRDRN`/`IncrementBound` reference this is
modeled on), alongside the existing rain-probability head unchanged. This is the literature's largest
measured effect size (18x extreme-event detection-rate gain in the source paper) and directly targets
the 87%-of-error-budget mid/heavy-rain tail. Not a drop-in win: our metric is `tile_rmse`, which
rewards the conditional mean, not a well-calibrated tail, so which quantile (or blend, e.g.
`0.5*median + 0.5*p90`) actually minimizes `tile_rmse` is an open empirical question — OOF-sweep it,
separately for the whole distribution and for the mid/heavy-rain subset
(`doc/data_characteristics_review.md` §1.3 bins). Start from whichever of exp012/exp013 is winning by
the time this is implemented; single-fold A/B before any 5-fold commitment.

### G-027 — IR-to-rain calibration curve (a: post-processing, b: input feature)

(a) Fit an isotonic (or piecewise-linear) OOF calibration curve mapping raw amount-head output to a
calibrated value — same mechanism `oof_calibration.json` already uses for linear scale/bias in
exp008+, just swap the fit method. Try on the current best submission before touching training;
essentially free. (b) If (a) doesn't close the gap, add the per-satellite `bt_rain_response.csv`
curve (already computed in `outputs/l_eda/exp002`) evaluated at each pixel's coldest-band IR value, as
one extra input channel — mirrors the Weather4cast 2025 2nd-place solution's explicit
BT->rainfall-rate transformation stage (`doc/research_survey_v2.md` §2). Both derived purely from
distributed data.

**(a) implementation (2026-07-10)**: `analyze_oof.py` (exp009/011/012/013, and now `exp015`) accumulates
a fixed 95-bin pixel-value histogram of (pred, target) during the existing OOF forward pass (~52-76
bins actually populated in practice), collapses each bin to (mean_pred, mean_target) weighted by
pixel count, and fits `sklearn.isotonic.IsotonicRegression(y_min=0, increasing=True,
out_of_bounds="clip")` on those points — cheap regardless of OOF pixel count, and consistent with
the existing linear scale/bias fit also being a pooled-sum (not per-tile-weighted) estimate. The
fitted knots serialize into `oof_calibration.json["isotonic"]["x"/"y"]` alongside the untouched
linear fields. `inference.py` gained `apply_isotonic_curve()` (torch `searchsorted`-based monotonic
piecewise-linear interpolation, verified bit-close to `sklearn`'s own `.predict()`) and a
`postprocess.calibration_mode: "linear"|"isotonic"` config switch (default `"linear"` in
exp009/011/012/013 so no existing run's behavior changes silently; default `"isotonic"` in exp015,
since that's the whole point of that experiment); requesting `"isotonic"` without a fitted curve
present falls back to linear with a warning.

**Where it actually runs**: rather than re-running `analyze_oof.py`/`inference.py` inside exp009's
own folder (which would overwrite exp009's real `outputs/analysis/exp009/oof_calibration.json` and
mix a diagnostic-only run into its history), the isotonic-calibration *run* lives in its own
`g_experiments/exp015` — same "reuse another experiment's checkpoints read-only" pattern exp008
already established for exp004 (`paths.source_model_dir: ../../g_model/exp009`, own
`output_dir`/`analysis_dir`/`submission_zip` under `.../exp015`). `exp015` trains nothing; its
`run.sh`/`singularity_run.sh` support `analyze` / `infer` / `submit` / `submit_calibrated` stages,
submittable via `sbatch singularity_run.sh` per the standard HPC convention.

Verified: synthetic-data unit test (recovers a known 4x heavy-rain-underestimation curve, `torch`
output matches `sklearn` exactly), and a real end-to-end run of `exp015`'s actual shipped
`run.sh analyze` / `run.sh submit_calibrated` stages on the local 3090 against a throwaway
1-epoch/512-sample checkpoint (standing in for exp009's real one via `SOURCE_MODEL_DIR` override) —
produced a valid monotonic `oof_calibration.json` and an `inference.py` run with
`calibration_mode: isotonic` whose raw calibrated values genuinely diverge from linear mode (max
diff 0.036 on this undertrained checkpoint).

**Result (2026-07-10, real exp009 checkpoints on cloud)**: `exp015_submission.zip` scored
0.7096658388930687 on the public leaderboard, -0.0056780510175330 vs. exp009's raw 0.7153438899106017
(`doc/public_scores.md`). Confirmed real: `oof_calibration.json`'s linear `scale`/`bias` matched
exp009's own historical values exactly. Notably, an offline approximate check (applying the
isotonic curve to `oof_sample_metrics.csv`'s per-tile `pred_max`) showed almost no correction of the
tile-peak underestimation the ticket originally targeted (mean `pred_max` 2.13 -> 2.10 vs. a target
mean of 7.90; the isotonic curve's top bins are sparse and get pooled near-flat by PAV, e.g.
pred>=13.24 all map to y~13.10) — yet the real public score still improved. Most likely explanation:
the win comes from damping over-confident mid/high false-alarm pixels (reducing their contribution to
squared error) rather than fixing peak amplitude, which the tile-peak proxy wasn't measuring. Also
added `analyze_oof.py`'s `calibration_comparison` (raw/linear/isotonic OOF `tile_rmse` computed from
cached OOF tiles, no extra GPU pass) for a more direct before/after read next time, though the public
score already answers the go/no-go question for this round. **Not yet stacked with exp014's
tile-overlap patch** — pointing `g_experiments/exp014/apply_overlap.py` at exp015's raw predictions
instead of exp009's (both effects are independent: calibration changes non-overlap pixels, the patch
overwrites overlap-region pixels with real GPM truth) is the next likely best-single-submission
candidate, expected around 0.6968727727408199 - 0.0056780510175330 ≈ 0.691 if the two effects are
roughly additive.

### G-028 — dilated-conv bottleneck (exploratory)

`outputs/l_eda/exp002`'s `radial_power_spectrum.csv` shows ~81% of target spatial-frequency power at
wavenumber 1-2 (20-41px wavelength) — the network's job is mostly large-scale patterns, not fine
texture, which argues against more downsampling and mildly favors growing receptive field without
discarding resolution. Single-fold ablation: replace `enc2`/`enc3`'s `avg_pool2d` downsampling with
dilated convolutions (dilation 2, 4) at fixed resolution, same channel widths, same
data/loss/architecture as the current fold-0 champion. Compare OOF `tile_rmse` overall and on the
mid/heavy-rain-tile subset specifically. No measured effect size from directly analogous work exists
yet (unlike G-026/G-027) — this is a plausible-but-unproven architectural bet.

### G-029 — radial-power-spectrum auxiliary loss (exploratory)

Add a small auxiliary loss term comparing the radial power spectrum of prediction vs. target on the
amount head's output, reusing the existing `radial_power_spectrum` FFT code from `l_eda/exp002`
(motivated by the "double penalty" MSE-blur diagnosis in `doc/research_survey_v2.md` §4). Run only
after G-026/G-027 are evaluated — it targets the same blur/tail symptom via a different mechanism, and
stacking three simultaneous untested changes would make the comparison unreadable. Our tile is small
enough that a full wavelet-decomposition approach (WADEPre) likely has too little scale range to pay
off, so this stays scoped to the simpler radial-power-spectrum term, not a wavelet loss.

## Round 4 Open Tickets (2026-07-10, from `doc/discussion_insights.md`)

The official discussion (5 posts, analyzed in `doc/discussion_insights.md`) reframes the whole
problem: a perfect flat tile-mean predictor scores 0.677 on the official metric ("the wall"), the LB
is literally cut at rank 16/17 by that value, and the perfect-tile-mean + perfect-wet-mask oracle
scores 0.594 — meaning **within-tile localization** carries 0.083 of headroom, 2.5x the entire
#1-to-pack gap. Our exp009 lineage (including exp015's calibration) optimizes tile-level amount and
sits exactly at the wall. Round 4 shifts effort to (a) the mathematically correct estimator (hurdle
log-normal, replacing both our miscalibrated weighted-BCE two-head and the superseded quantile plan),
(b) the discussion's measured input-feature wins, and (c) localization. Priorities assume the cloud
cluster can run many jobs in parallel — exp016/017/019 are independent axes that can run concurrently;
exp018 builds on whichever head wins; exp020 harvests.

| ID | Track | Exp | Title | Priority | Status |
| --- | --- | --- | --- | --- | --- |
| G-030 | g_experiments | exp016 | Hurdle log-normal head: unweighted-BCE occurrence × wet-only ln(y) intensity, serve p·exp(μ+σ²/2) | P0 | Implemented + smoke-tested on 3090 (unit tests: serving formula, wet-only gradients, median arm shares state_dict; 235 zero-obs rows confirmed all-Meteosat); cloud single-fold A/B pending |
| G-031 | g_experiments | exp017 | Wavelength-aligned canonical bands + uint8-domain physics channels (split-window ratio, IR8.5−W, WV−W, temporal diff, day/night flag) | P0 | Implemented + smoke-tested on 3090 (feature math verified vs hand computation on all 3 satellites); median/IQR norm deferred to keep one change axis; cloud 3-arm A/B pending |
| G-032 | g_experiments | exp018 | Within-tile localization: high-res internal processing, aux wet-mask head, multi-scale displacement-tolerant loss, new spatial OOF diagnostics | P0 | Not started; the strategic bet — depends on G-030's head |
| G-033 | g_experiments | exp019 | t+0-frame-centric input design + \|Δ\| change-magnitude channels (+ optional train-time future-frame aux head) | P1 | Not started |
| G-034 | g_experiments | exp020 | Harvest: 3 seeds × 5 folds of the best config, 6-way TTA, OOF-weighted ensemble, isotonic calibration, drizzle cut, overlap patch last; satellite-mix-weighted CV | P1 | Not started; run after exp016-019 A/Bs settle |

### G-030 — exp016: hurdle log-normal head

Replace `two_head_rain` with the estimator the target's own statistics dictate
(`doc/discussion_insights.md` §2; wet pixels are log-normal ln(y)~N(−0.66,1.63²)):

- Occurrence head: plain **unweighted** BCE on rain>0 (drop `bce_pos_weight: 3.0` — weighted BCE
  deliberately mis-calibrates P(rain), and calibration is what makes the product unbiased).
- Intensity head: trained **only on wet pixels** (masked loss — never sees the 82% zero mass, so it
  can't be dragged to zero), target ln(y), either fixed-σ MSE or Gaussian NLL predicting per-pixel
  (μ,σ). Drop `pos_weight: 2.0` (tail re-weighting measured net-negative by the discussion authors).
- Serving: `pred = P(rain) * exp(μ + σ²/2)` — the conditional mean, never a median/quantile. If NLL's
  σ is unstable on ~20 episodes, fall back to a per-satellite constant σ estimated from train wet
  pixels (measured global σ=1.63).
- Data cleanup from Finding 9: drop the 235 zero-input (all-Meteosat) rows; pad 2-frame rows by
  repeating the last frame (verify what our current padding does first).
- Base: exp009's 105ch successor-row inputs and CompactUNet body, so the head change is the only
  variable. Acceptance: OOF `tile_rmse` < 0.6239 AND `pred_max/target_max` at `target_max>=10`
  substantially above the current 0.25 (this head targets exactly the exp(σ²/2)≈3.8 mean/median gap
  we measured as ~4x).

### G-031 — exp017: wavelength alignment + physics channels

Two stacked A/B stages, both from measured discussion findings (`doc/discussion_insights.md` §3):

1. **Alignment only**: reorder each satellite's 16 bands into a canonical physical-wavelength order
   using the discussion's index table (Meteosat's fixed-index channels are wrong for 2 of 3 key bands;
   the WV fix alone was measured +25% correlation). This makes shared conv weights see the same
   physics at the same channel position across satellites — cheaper and more principled than exp011's
   learned per-satellite stems.
2. **+Features**: engineered channels computed on raw uint8 **before** normalization: split-window
   ratio `SPL/(W+1)` (partial corr −0.31/−0.20/−0.13; the classic difference is quantized to death),
   `IR8.5−W` ("hidden gem" #2 additive feature), `WV−W` (deep-convection proxy), temporal diff
   `W[t2]−W[t0]`, day/night flag from mean-VIS threshold (covers G-023's intent). Switch normalization
   to per-(sat,band) median/IQR with clamp(±5).

### G-032 — exp018: within-tile localization (the strategic bet)

The only path below the 0.677 wall is knowing where the rain sits inside the 41×41 tile
(`doc/discussion_insights.md` §1: wet-mask rung = 0.083 headroom; σ=2px-blurred truth scores 0.38 —
displacement tolerance beats sharpness). Three mechanisms, ablatable via config:

- **High-res internal processing**: bilinear-upsample input to 128×128, 4-level encoder (the public
  0.69 solution's layout), decode and adaptive-pool to native 41×41 output (Finding 8: predict at
  native 41×41; tiles are event-centered so resampling the *target* is harmful, resampling the
  *input* is not).
- **Aux wet-mask head**: a segmentation head (Dice or soft-IoU) on rain>0, separate from the
  calibrated occurrence head so calibration is not polluted; encourages explicit spatial structure.
- **Multi-scale MSE**: MSE at 41×41 plus 2×2- and 4×4-avg-pooled scales — a cheap
  displacement-tolerant term that rewards correctly-placed mass even when fine placement is off.
- **New OOF diagnostics** in analyze_oof: per-tile pred-target spatial correlation (on wet tiles) and
  wet-mask IoU at the OOF-optimal threshold — these measure localization directly, which `tile_rmse`
  alone hides. Also report the oracle-wall numbers (flat tile-mean of *our* predictions vs of truth)
  per run so we can see when we've genuinely broken below flat-mean information.

### G-033 — exp019: t+0-centric temporal design

Information peaks exactly at target time t, and the t+0 frame is available for 99.4% of eval rows via
the successor row's observation list — which exp009 already ingests, but only as 3 of 6
equally-stacked frames (`doc/discussion_insights.md` §4). Restructure the input so t+0 is the primary
frame: order channels so the model's first conv sees t+0 first; add |Δ| change-magnitude channels
(measured Spearman +0.69 with rain — the useful temporal signal; do NOT add optical flow, measured
ρ≈−0.05, confirming our own exp005/l_eda findings). For rows lacking a successor (~0.6%), fall back to
the newest causal frame with the existing mask-channel convention. Optional second stage: a
train-time-only auxiliary head predicting the t+0 IR-window frame from causal frames (advection
learning); low priority since we already have t+0 at eval time.

### G-034 — exp020: harvest

Once exp016-019 A/Bs pick winners: 3 seeds × 5 folds of the winning config; extend TTA from 3-view
flips to 6-way (flips + rot90/180/270 — the metric-relevant augmentation group our training already
uses); OOF-weighted ensemble (updates G-019); isotonic calibration refit on the ensemble's OOF
(exp015 machinery); drizzle value-threshold; exp014 overlap patch applied last (it overwrites with
GPM truth, so it always stacks on top). Selection decisions throughout use CV weighted by the test
satellite mix (himawari 39%, meteosat 39%, goes 22%) per Finding 10, and treat LB deltas < ~0.005 as
noise (single-split geography luck spans 1.15–1.71 per the discussion's measurement).

### G-016 — exp012: successor-row + satellite-adapter two-head

Merge `g_experiments/exp009` (dataset.py: successor-row loading, `context_rows`, 105ch input) with
`g_experiments/exp011`'s `satellite_adapter_two_head_unet` model. The adapter's per-satellite input
stems need to accept 105ch (2x the slots of exp011) instead of 54ch — extend the adapter to key off
satellite one-hot per slot the same way exp009's mask channels already do. Reuse exp009's loss
(`two_head_rain`), normalization stats, and GroupKFold split unchanged. This is the top-priority
ticket: both parent experiments individually beat the exp004 anchor on public score, and they change
independent axes (temporal context vs. sensor handling), so the combination is the best-expected-
value untried change.

Acceptance: 5-fold OOF `tile_rmse` reported in `outputs/analysis/exp012/`, compared against exp009's
0.6239 and exp011's 0.6328; submit and record in `doc/public_scores.md` regardless of OOF result.

### G-017 — exp013a: parallax registration preprocessing

`outputs/l_eda/exp001/parallax_shift_by_location.csv` gives a per-(location, satellite) median
`(dy, dx)` pixel shift with a measured correlation gain after alignment (up to +0.089). Add an
optional preprocessing step in `dataset.py` that shifts each satellite frame by its
location/satellite median offset (nearest-neighbor or sub-pixel via `cv2.warpAffine`) before
resizing/stacking, gated by a config flag so it can be A/B tested against the unregistered baseline
on a single fold before committing to a 5-fold run.

Acceptance: single-fold OOF RMSE with vs. without registration on the same architecture (start from
exp009 or exp012, whichever is the current best); only proceed to full 5-fold if registration wins.

### G-018 — exp013b: wider temporal context

exp009 adds one successor row (`context_rows: 2` counting the current row). Given `l_eda` shows
motion is small (point 2 above), try adding the predecessor row too (i.e. up to 9 observation slots:
predecessor + current + successor, each up to 3 observations) or a wider successor window, and
measure whether OOF `tile_rmse` keeps improving or plateaus. Track `in_channels` growth vs. compute
cost — this is a channel-count/compute tradeoff, not a new architecture.

### G-019 — exp007 refresh: ensemble the current best models

Re-point `g_experiments/exp007`'s `paths.source_model_dir` list at `exp008`, `exp009`, `exp011` (add
`exp010` only if G-020 clears it), and compute OOF-weighted blend weights instead of the current
equal-weight scheme. Also close the gap noted in insight #10: have the ensemble script compute its
own `oof_global` (rmse/tile_rmse/csi/etc. on the blended prediction, not just per-source summaries)
so this ticket produces a normal `analysis_summary.json` that fits the registry schema in
`doc/experiment_tracking_design.md`.

Acceptance: ensembled OOF `tile_rmse` beats the best individual source (exp009's 0.6239); submit and
record in `doc/public_scores.md`.

### G-020 — exp010 cleanup ablation

`g_experiments/exp010` bundles three changes (suspicious-label drop, IR-zero-as-nodata, GOES 4ch
remap) and underperforms on public score despite a decent OOF `tile_rmse`. Re-run with each change
toggled independently (3 single-fold runs, config-flag gated) to find which one causes the OOF/public
mismatch before any future experiment inherits exp010's cleanup as a default.

### G-021 — rain-head loss upgrade

`best_rain_threshold.csi` sits at 0.44–0.46 across exp008/009/010/011 regardless of architecture,
with only the decision threshold tuned. Try a Dice or Focal loss term on the rain-probability head
(in addition to or instead of `bce_weight`/`bce_pos_weight` in `loss.two_head_rain`) on top of
whichever architecture wins G-016, and compare CSI/precision/recall at the OOF-selected threshold
against the current plateau.
