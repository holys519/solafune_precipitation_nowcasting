# Task Tickets

Last updated: 2026-07-09

Tracks concrete follow-up work after `exp001` (public RMSE 0.7531995875751526). See
`doc/exp001_retrospective.md` for the analysis behind the exp002 tickets, and
`doc/research_survey.md` for the research-driven roadmap after exp003. See
`doc/public_scores.md` for the current leaderboard table and
`doc/experiment_tracking_design.md` for the registry/graphing design behind the `I-XXX` tickets
below.

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
