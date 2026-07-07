# exp001 Retrospective and exp002 Direction

Last updated: 2026-07-07

## Result

| Metric | Value |
| --- | ---: |
| Public RMSE | 0.7531995875751526 |
| CV RMSE (location holdout: bihar/borno_state/gaza_province/kinshasa) | 0.810831 |
| Zero-predictor RMSE (full train) | 1.432370 |
| Zero-predictor RMSE (exp001 valid split) | 0.962228 |

exp001 clearly beats the zero baseline (~16-25% lower RMSE depending on split), so the overall
approach (concat satellite tensors → compact U-Net → per-pixel precipitation regression) is sound
and worth continuing. But the training run itself was intentionally minimal (a correctness-first
GPU smoke test), so most of the gap to a strong score is still on the table.

The public score being *better* than the CV score (0.753 vs 0.811) is most likely sampling noise:
CV used only 4 held-out locations and 3000 rows, chosen by a single random split, not an average
over folds. It should not be read as "the model generalizes better than CV suggests" until we have
a 5-fold GroupKFold estimate to compare against.

## What exp001 actually did (`l_experiments/exp001/`)

- 3 of 3 observation slots → 16ch each → resized with bilinear interpolation to the native
  GPM-IMERG grid (41×41) → concatenated to 48ch, plus 3 mask channels and 3 satellite one-hot
  channels → 54ch input.
- Pixel values normalized as `raw_uint8 / 255.0`, identical treatment for every satellite/band.
- CompactUNet trained from scratch (3 conv-block encoder/decoder levels, ~48 base channels), plain
  `nn.MSELoss`, no augmentation, AdamW, 3 epochs.
- Split: one random 80/20 split of *locations* (not GroupKFold), capped at 12000 train / 3000 valid
  **rows** out of 40686 available train rows.

## Diagnosis

1. **Severe under-training relative to available data.** Only ~30% of train rows were used, for 3
   epochs, with no augmentation. `train_rmse` was still falling each epoch
   (1.223 → 1.174 → 1.151) and had not converged. This is very likely the single largest lever —
   more epochs and more data should give a large, "free" improvement before touching anything else.

2. **Loss dominated by the wrong pixels.** Target is 82% exact zero
   (`eda/outputs/EDA_DEEP_DIVE.md`), and `positive_rmse` (2.52–2.56) is roughly **3x** the overall
   RMSE (0.81–0.82). Plain MSE mostly optimizes for correctly predicting zero and barely penalizes
   errors on the rain pixels that dominate the actual difficulty (and, per satellite, correlate with
   the heavy right tail up to 96.5mm). A loss that up-weights positive-precipitation pixels should
   move the needle on the component that currently hurts most.

3. **Normalization ignores per-satellite/per-band statistics.** All three satellites are
   fundamentally different sensors (Himawari/GOES bands are similar in spirit, Meteosat's band set
   and value ranges are their own thing per `doc/dataset.md`), and even within one satellite,
   visible vs. IR/water-vapor bands have very different pixel distributions. Dividing everything by
   255 discards that structure and forces the first conv layer to relearn per-channel scale from
   scratch with very little training. Per-satellite, per-band z-score normalization computed from
   the data is cheap and standard.

4. **Downsampling method is inconsistent with an anti-aliasing goal.** Himawari (81px), GOES
   (141px), Meteosat (144px) are all being downsized to the 41×41 IMERG grid — a 2–3.5x reduction —
   using bilinear interpolation, which aliases under significant downsampling. Average pooling (or
   area-interpolation) is more appropriate whenever the source is larger than the target, which is
   always true here.

5. **CV is a single split, not a robust estimate.** `configs/default.yaml` already specifies
   `GroupKFold, n_splits=5`, but exp001's `train.py` implements one fixed random holdout instead.
   With only 20 train locations total, a single 4-location holdout is high-variance — this may
   explain part of the CV/public gap in point 0. A real 5-fold GroupKFold run (and eventually a
   5-fold ensemble at inference) will give a much more trustworthy number to optimize against.

6. **No augmentation, despite being planned.** `configs/default.yaml` lists
   `horizontal_flip/vertical_flip/rotate90`, but exp001's dataset does not apply any. All three
   satellite grids and the target are simple square rasters with no fixed "up" direction implied by
   the model (no absolute geolocation encoding beyond the `x_coord`/`y_coord` features in the
   *separate*, currently-unused `features.py` path), so flips/rot90 are safe, cheap augmentation.

7. **Architecture is a small from-scratch CNN.** `configs/default.yaml` originally intended a
   `segmentation_models_pytorch` U-Net with an ImageNet-pretrained `efficientnet-b0` encoder;
   exp001 used a much smaller custom `CompactUNet` instead (reasonable for a first correctness
   check, since a 54-channel non-RGB input can't use pretrained weights out of the box without
   adapting the stem). This is worth a controlled A/B once the data pipeline fixes above are in,
   rather than changing it in the same run — otherwise we can't tell which change mattered.

## Priority for exp002

Ordered by expected impact per unit of implementation/compute cost:

1. Per-satellite/per-band normalization (item 3) — cheap, should help immediately.
2. Full training data + more epochs (item 1) — free, just needs the wall-clock budget.
3. Rain-weighted loss (item 2) — cheap, directly targets the worst-performing pixel class.
4. GroupKFold 5-fold CV (item 5) — needed to trust any of the above numbers going forward.
5. Anti-aliased downsampling + flip/rot90 augmentation (items 4, 6) — cheap, standard.
6. Pretrained-encoder architecture ablation (item 7) — deferred to its own controlled comparison
   (see `doc/task_tickets.md`), implemented as a config-selectable option in exp002's `model.py` so
   it can be switched on without new code once 1–5 are validated.

Items 1–5 are implemented together in exp002 as a single "fix the training pipeline" experiment,
since they are independent, all cheap, and would otherwise require re-running the full training
loop once per fix. See `l_experiments/exp002/README.md` for the implementation and
`doc/task_tickets.md` for everything not yet done.
