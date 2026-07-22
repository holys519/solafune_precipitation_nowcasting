# exp053: Autoregressive Own-Past-Prediction Input Channel

Status: implemented, fold0/fold4 gate in progress (see "Gate status" below — update this section
once `self_pred_oof.py` has run for both folds).

## Why this is green

The organizers' 2026-07-20 final ruling (`doc/submission_registry.md`, "2026-07-20 追記2") lists,
under **許可 (permitted)**:

> 自分自身の過去 (< T) の予測値の再利用 (自己回帰/recursive手法)

("Reuse of one's own past (< T) predictions, i.e. autoregressive/recursive methods") — explicitly
distinguished from the **禁止 (banned)** item immediately above it in the same ruling:

> timestamp ≥ Tのobservationを予測Tに使用 (successor rowのT, T+10, T+20フレームを含む) — 確定

This avenue had never been tried in this project before exp053 (see registry §5, "新しく開けた道").
The mechanism here is standard real-world nowcasting practice (persistence/AR baselines): every
value fed into the model as the "previous prediction" channel is for a timestamp strictly
`< T` relative to the row being predicted (T−30min), so it is always causal. The two sourcing
modes (below) differ in *where that causal value comes from*, never in *when* it is from.

## Design

Base: `g_experiments/exp038` strict/green champion (`context_rows: 1`, 54ch, current-row-only
inputs, `HighResHurdleLogNormalUNet`, `hurdle_lognormal` loss) — fully duplicated into this
directory following the `g_experiments/exp047` template pattern (see its README/diff for how a
prior experiment cloned exp038 wholesale and layered one new feature on top).

New: `features.autoregressive_prev_pred: true` in `config.yaml`, plus 2 extra input channels
(`model.in_channels: 56` = 54 + 2), appended at the end of the channel stack in
`dataset.py::PrecipDataset._input_tensor`:

1. **value channel**: the precip value at the SAME `name_location`, 30 minutes earlier (T−30min),
   at native (H,W)=(41,41) resolution.
2. **validity/mask channel**: 1.0 everywhere if that earlier row/prediction exists, 0.0
   (zero-filled value channel too) otherwise — the same zero-pad + mask-channel convention
   `_observation_tensors` already uses for missing observation slots.

Implemented in `dataset.py::PrecipDataset._autoregressive_prev_pred_channels`, with two sourcing
modes selected by whether `PrecipDataset.ar_cache` is `None` or a dict:

- **`ar_cache is None` — teacher forcing (train/valid, the default)**: looks up the SAME
  train/valid-split row at `(name_location, T−30min)` via the dataset's own
  `_row_by_location_time` dict (the same lookup pattern `dataset.py` already uses for
  `context_rows` successor-row lookups). If that row exists in the split, its TRUE
  ground-truth GPM target (`gpm_imerg_filename`, read via `read_target_tensor`) is used as the
  "previous prediction" value. This is legal because it is causal ground truth already known at
  training time (not future data) — the ruling's permission for using train-fold future frames
  as a training-time *auxiliary signal* is separate but reinforces the same causality logic.
  If no such row exists in the split, the channel falls back to zero + mask=0.
- **`ar_cache` is a dict — evaluation, and the self-prediction-substituted OOF pass**: the true
  T−30min GPM value does not exist for these rows (it is exactly what would be predicted for
  that earlier row too), so ground truth is **never** read in this mode, even when `has_target`
  is also True (needed so the self-pred-substituted OOF pass can still score the *current* row
  against its true target, just not source its *input* from it). Instead, the model's own
  just-computed prediction is looked up from `ar_cache`, which is populated by the caller's
  chronological, per-`name_location` sequential loop. Missing key ⇒ zero + mask=0 (used for the
  first row of every location's sequence, which has no earlier row).

### Sequential inference restructuring (the hard part)

At evaluation time the true T−30min GPM value is unknowable (that IS the eval target for the
earlier row), so `inference.py`'s prediction loop was restructured from a single flat, order-
independent `DataLoader` pass into `run_autoregressive_inference`:

1. Evaluation rows are grouped by `name_location` and sorted ascending by `datetime`
   (`build_location_sequences`).
2. Processing proceeds in **steps**: step *i* = the *i*-th row of every location's own
   chronological sequence. All locations present at a given step are batched together on the
   GPU (locations are independent of each other), but the per-location time axis is strictly
   sequential — step *i*'s predictions must be written into `ar_cache` before step *i+1* reads
   them.
3. Each row's own final (calibration + rain_prob-threshold + value-threshold + non-negative-clip
   applied) prediction is cached under `(name_location, that row's datetime)` and becomes the AR
   input feature for the same location's next row, 30 minutes later.
4. The first row of every location naturally falls back to zero + mask=0 (no cache entry yet).

`PrecipDataset.input_tensor(row)` is a new public wrapper around the existing private
`_input_tensor` used by this loop (row-driven access, bypassing index-based `__getitem__`, since
processing order here is dependency-driven, not dataset-index-driven — and inference must never
apply the training-time augmentation `__getitem__` would otherwise gate on `self.augment`, which
is already `False` for eval datasets regardless).

`self_pred_oof.py` mirrors the identical sequential/chronological loop over **validation** rows
for the gating metric (see below), the only difference being it also reads each row's true target
to score against, and does not apply TTA/calibration/thresholding (to stay directly comparable to
`train.py`'s plain `evaluate()`).

## Known risk: exposure bias

Teacher forcing (train/valid) feeds the model **true** GPM values as its own "past prediction."
At real inference time, the model instead sees its **own, imperfect** predictions in that same
slot. This train/inference mismatch — "exposure bias" in the seq2seq/autoregressive-modeling
literature — means the naive teacher-forced OOF number is optimistic and must not be used alone
as the gating criterion.

To account for this, exp053 implements **two** distinct OOF passes:

1. **Naive / teacher-forced OOF**: exactly what `train.py`'s standard `evaluate()` computes
   during normal training (since `ar_cache` stays `None`, ground truth is used for the AR
   channel) — this is `best_tile_rmse` in `g_model/exp053/metrics_fold{0,4}.json`, produced with
   zero extra code.
2. **Self-prediction-substituted OOF** (`self_pred_oof.py`) — the actual gating criterion: for
   each fold's validation rows, mimics real inference exactly (chronological, per-location,
   `ar_cache`-driven self-prediction substitution; see above), then computes the same
   `tile_rmse`/`rmse` formulas `train.py::evaluate()` uses, so it is directly comparable to the
   exp038 strict-baseline gate numbers.

**Gate**: compare (2) — not (1) — against exp038 strict baseline fold0/fold4 `tile_rmse`
(0.28954 / 0.59607, from `doc/submission_registry.md`). Only if (2) clears the gate does exp053
proceed to all 5 folds and a submission.

## Gate status

Smoke test (`slurm-exp053-smoke-3942930.out`) passed 2026-07-22 (config/channel consistency,
CPU forward/backward, AR-channel unit checks, real-data teacher-forcing verification against
true GPM targets, and a sequential ar_cache simulation over real rows — see that log for
detail). Training jobs submitted:

- fold0: job **3942933** (`slurm-exp053-fold0-3942933.out/.err`)
- fold4: job **3942934** (`slurm-exp053-fold4-3942934.out/.err`)

Both were queued (`PD`, reason `Priority`) behind ~14 other pending jobs from parallel
experiments (exp047/exp050/exp051/exp054/exp038-canonical) at submission time. Based on
exp038/exp047's fold-training wall-clock history, expect roughly 5 hours of GPU time per fold
once each job starts running, plus queue wait.

| Fold | exp038 strict gate (tile_rmse) | exp053 teacher-forced OOF (naive, optimistic) | exp053 self-prediction-substituted OOF (actual gate metric) | Verdict |
| ---: | ---: | ---: | ---: | --- |
| 0 | 0.28954 | *pending — see `g_model/exp053/metrics_fold0.json` after job 3942933 finishes* | *pending — run `self_pred_oof.py --fold 0` after job 3942933, see `outputs/analysis/exp053/self_pred_oof_fold0.json`* | pending |
| 4 | 0.59607 | *pending — see `g_model/exp053/metrics_fold4.json` after job 3942934 finishes* | *pending — run `self_pred_oof.py --fold 4` after job 3942934, see `outputs/analysis/exp053/self_pred_oof_fold4.json`* | pending |

Fill in once both fold checkpoints exist and `self_pred_oof.py` has run for both folds (see
"Run" below). Only submit all 5 folds (`submit_folds.sh`) if the self-prediction-substituted
number improves on (or is at least not worse than) the exp038 strict gate on **both** folds.

## Known simplifications / limitations

- **Teacher-forcing lookup uses the (possibly zero-obs-filtered) training row list, not the raw
  CSV.** `train.py` drops ~235 all-Meteosat zero-observation rows from `train_rows` before
  constructing `train_ds` (`drop_zero_obs_rows: true`, unchanged from exp038). Because
  `PrecipDataset._row_by_location_time` is built only from the rows it is constructed with,
  a dropped row can never serve as another row's "previous ground truth" during training, even
  though its own GPM target is otherwise valid as an AR input. This is a rare edge case (~0.6%
  of rows) and was left as-is rather than threading a second, unfiltered row list through the
  constructor — flagged here rather than silently fixed, since it changes training data
  composition and deserves its own ablation if it turns out to matter.
- **`postprocess.temporal_smoothing` is not exercised by the sequential AR loop.** It is
  disabled by default (matches exp038), and if ever enabled, the caller applies it *after* the
  full `run_autoregressive_inference` pass completes — so cached AR values fed to earlier steps
  are always the pre-smoothing prediction, not the smoothed one. This is fine while smoothing is
  off; if exp053 is later combined with `exp046`-style causal smoothing, the interaction needs
  its own audit.
- **Scheduled sampling (stretch goal, not implemented).** Mixing ground truth and self-generated
  predictions during training (gradually annealing from GT to self-predictions) is the standard
  mitigation for exposure bias in autoregressive sequence models, and would likely narrow the
  gap between the two OOF numbers above. This was intentionally left for a follow-up experiment
  (e.g. `exp05x_scheduled_sampling`) — exp053 ships the simpler pure-teacher-forcing version
  first, per the task's own priority ordering.
- **No TTA/calibration in `self_pred_oof.py`**, by design, to stay apples-to-apples with the
  `train.py::evaluate()` gate baseline (which also has neither). The real evaluation pipeline
  (`inference.py`) does support both, applied per-step inside the sequential loop before the
  cache write (see "Sequential inference restructuring" above).
- **Sequential AR inference is slower** than the flat batched loop it replaces: instead of one
  DataLoader pass with `num_workers>0` shuffling through all ~29k eval rows, it is `max(rows per
  location)` (~2928 for `kanto_region`/similar) forward passes with per-step batch size = number
  of locations still active at that step (≤18). This is an accepted, documented cost of
  correctness — a location's step *i+1* genuinely cannot be computed before step *i*'s
  prediction exists.

## Run

```bash
cd /group/project143/yamamoto/solafune_precipitation_nowcasting/g_experiments/exp053
sbatch singularity_smoke.sh
sbatch --parsable --job-name=exp053-fold0 \
  --output=slurm-exp053-fold0-%j.out --error=slurm-exp053-fold0-%j.err \
  singularity_run.sh config.yaml 0
sbatch --parsable --job-name=exp053-fold4 \
  --output=slurm-exp053-fold4-%j.out --error=slurm-exp053-fold4-%j.err \
  singularity_run.sh config.yaml 4
# after both fold0/fold4 checkpoints exist:
sbatch --job-name=exp053-gate singularity_run.sh config.yaml self_pred_oof
# only if the gate clears on BOTH folds:
bash submit_folds.sh config.yaml
```

`self_pred_oof.py` can also be run standalone (still needs GPU/container):

```bash
python self_pred_oof.py --config config.yaml --fold 0
python self_pred_oof.py --config config.yaml --fold 4
```
