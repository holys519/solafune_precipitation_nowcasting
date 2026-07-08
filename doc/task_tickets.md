# Task Tickets

Last updated: 2026-07-08

Tracks concrete follow-up work after `exp001` (public RMSE 0.7531995875751526). See
`doc/exp001_retrospective.md` for the analysis behind the exp002 tickets, and
`doc/research_survey.md` for the research-driven roadmap after exp003.

## Track convention

- **L-XXX tickets** run under `l_experiments/` — local dev, correctness checks, small samples,
  fast iteration on the current 3090x2 workstation. Nothing here needs multi-day compute budget.
- **G-XXX tickets** run under `g_experiments/` — full-scale/production training, anything that
  needs a full epoch budget over the full dataset, multi-fold ensembles, or scaled-up hardware.
- **GPU scaling work (A100 x2 / x4) is G-track only.** It is a production/throughput concern, not
  a correctness concern, so it never belongs in `l_experiments/`.

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
