# Research Survey: Satellite Precipitation Nowcasting

Last updated: 2026-07-08

## Purpose

This document summarizes external research, competition solutions, and practical implementation ideas
for the Solafune precipitation nowcasting competition. The goal is not to copy a large external
framework, but to identify changes that are likely to improve our current pipeline:

- Input: up to 3 recent geostationary satellite observations, each with 16 multispectral bands.
- Target: GPM-IMERG calibrated precipitation GeoTIFF.
- Metric: RMSE.
- Constraint: external datasets are prohibited. Any external pretrained weights should be treated as
  rule-sensitive and used only after explicit confirmation from the competition organizer.

Current reference results:

| Experiment | Environment | Public RMSE | Notes |
| --- | --- | ---: | --- |
| exp001 | local | 0.7531995875751526 | Compact U-Net, simple concat, plain MSE |
| exp001 | A100 test | 0.7937729717031525 | Reproduced on cloud, worse than local run |
| exp002 | A100 | 0.7479569114058262 | Normalization, weighted loss, 5-fold ensemble pipeline |

The exp002 gain is real but modest. The next high-value direction is to improve the inductive bias
for precipitation: rain/no-rain imbalance, temporal fusion, sensor differences, and robust
post-processing.

## Local Competition Assumptions

From `doc/overview.md`, `doc/dataset.md`, and `doc/competition_rules.md`:

- The task maps multispectral satellite images to a precipitation raster, so it is an image-to-image
  dense regression problem.
- There are three satellite families: Himawari, GOES, and Meteosat. All expose 16 bands, but the
  actual sensor characteristics and band names differ.
- Each row has up to 3 observations from the previous 30 minutes. Some rows have fewer observations,
  and empty observation lists exist.
- Targets are non-negative precipitation fields, with a large zero-rain majority and rare high-rain
  tails.
- Leakage-resistant CV should respect `name_location` and, ideally, temporal structure.
- Submission must preserve GeoTIFF layout and create `evaluation_target.csv` plus `test_files/`.

These constraints make the problem closer to satellite quantitative precipitation estimation (QPE)
than long-horizon weather forecasting.

## Main Research Takeaways

### 1. Two-stage rain detection plus rain amount estimation

Several satellite precipitation retrieval systems separate "is it raining?" from "how much rain?".
This is attractive here because the target has many zero pixels and RMSE can be harmed by false
positive drizzle everywhere.

Most relevant reference:

- **Oya: Deep Learning for Accurate Global Precipitation Estimation**
  - URL: https://arxiv.org/abs/2511.10562
  - Uses a two-stage approach with one U-Net for precipitation detection and another U-Net for
    quantitative precipitation estimation.
  - The stated motivation is the strong imbalance between rain and no-rain events.

Implementation idea for this repo:

- Add a shared backbone with two heads:
  - `rain_logits`: binary rain/no-rain probability.
  - `rain_amount`: non-negative precipitation regression.
- Train with:
  - `BCEWithLogitsLoss` for `target > threshold`.
  - MSE/RMSE or SmoothL1 for amount.
  - Optional masked amount loss where positive rain pixels receive higher weight.
- Infer with either:
  - `sigmoid(rain_logits) * rain_amount`, or
  - `rain_amount` zeroed where `sigmoid(rain_logits) < threshold`.
- Use OOF predictions to sweep thresholds and calibration parameters.

Expected benefit:

- Lower false-positive light rain.
- More direct optimization of the zero-rain decision.
- Better diagnostics via detection metrics: precision, recall, CSI, false alarm ratio.

Risk:

- If thresholding is too aggressive, it can erase light rain and hurt RMSE.
- Need OOF tuning, not public leaderboard tuning.

Recommended experiment:

- `exp004`: `TwoHeadUNet`, with OOF threshold sweep and intensity-bin RMSE logging.

### 2. Temporal fusion should not be naive channel concatenation

The current baseline concatenates the 3 observation slots along the channel axis. This is simple, but
it forces the first convolution layer to learn separate filters for each time slot and does not encode
motion or temporal consistency.

Relevant references:

- **Convolutional LSTM Network: A Machine Learning Approach for Precipitation Nowcasting**
  - URL: https://arxiv.org/abs/1506.04214
  - Frames precipitation nowcasting as spatiotemporal sequence forecasting and proposes ConvLSTM.
- **Global Precipitation Nowcasting of IMERG: A U-Net Convolutional LSTM Architecture**
  - URL: https://arxiv.org/abs/2307.10843
  - Combines U-Net with ConvLSTM for IMERG precipitation nowcasting.
- **Nowcasting-Nets**
  - URL: https://arxiv.org/abs/2108.06868
  - Studies recurrent and convolutional networks for IMERG nowcasting.

Implementation idea for this repo:

- Reshape input as `(time, bands, height, width)` instead of one flat channel stack.
- Apply a weight-shared per-time encoder to each observation.
- Fuse features using one of:
  - ConvGRU/ConvLSTM at bottleneck.
  - Temporal attention over the 3 encoded feature maps.
  - Simple learned weighted average with missing-observation masks.
- Keep the decoder close to current Compact U-Net at first, so the ablation isolates temporal fusion.

Expected benefit:

- Better use of cloud motion and development/dissipation signals.
- Cleaner handling of missing observations.
- More reusable architecture for future multi-frame experiments.

Risk:

- More code and more moving parts.
- With only 3 frames, a small temporal module may beat a heavy one.

Recommended experiment:

- `exp005`: `TemporalFusionUNet`, starting with shared encoder + ConvGRU/ConvLSTM bottleneck.

### 3. Satellite-specific adapters are likely useful

Himawari, GOES, and Meteosat each have 16 bands, but the spectral definitions and value distributions
are not identical. A single first convolution layer treats all satellite-band channels as if they
were the same feature space.

Implementation idea for this repo:

- Add a small satellite-specific stem:
  - `HimawariStem`: `Conv2d(16, C, kernel_size=3)`
  - `GoesStem`: `Conv2d(16, C, kernel_size=3)`
  - `MeteosatStem`: `Conv2d(16, C, kernel_size=3)`
- Share the deeper U-Net encoder/decoder after the stem.
- Optionally add satellite embeddings through FiLM-like scale/bias in the encoder.
- Preserve the existing per-satellite/per-band normalization from exp002/exp003.

Expected benefit:

- Better sensor-domain adaptation.
- Less pressure on the shared backbone to learn incompatible channel scales.
- Potentially better cross-location generalization.

Risk:

- More parameters for fewer samples per satellite.
- Need per-satellite OOF metrics to avoid improving one sensor while degrading another.

Recommended experiment:

- `exp006`: `SatelliteAdapterUNet`, evaluated with OOF metrics grouped by `satellite_target`.

### 4. Rain intensity-aware losses are more targeted than a single positive-pixel weight

exp002 used positive-pixel weighting. This is a good first step, but precipitation has multiple
regimes: exact zero, light rain, moderate rain, heavy rain, and rare extremes.

Relevant references:

- **Global IMERG U-Net ConvLSTM paper**
  - URL: https://arxiv.org/abs/2307.10843
  - Reports that regression models capture light precipitation well, while classification-style
    objectives can perform better for stronger precipitation thresholds.
- **PAUNet**
  - URL: https://arxiv.org/abs/2311.18306
  - Uses attention U-Net ideas and a focal precipitation loss variant for medium and heavy rain.
- **RainAI / Weather4cast 2023**
  - URL: https://arxiv.org/abs/2311.18398
  - Emphasizes data preparation, importance sampling, and alternative loss design.

Implementation idea for this repo:

- Add intensity bins:
  - `zero`: `target == 0`
  - `light`: `0 < target <= 1`
  - `moderate`: `1 < target <= 4`
  - `heavy`: `4 < target <= 8`
  - `extreme`: `target > 8`
- Log RMSE per bin in OOF and validation CSV.
- Try:
  - weighted MSE by bin,
  - SmoothL1 for stability on outliers,
  - `log1p` target transform plus inverse transform,
  - auxiliary bin-classification head.

Expected benefit:

- More interpretable validation.
- Reduced overfitting to zero pixels.
- Ability to tune for the RMSE-sensitive high-error tail without blindly increasing all positive
  weights.

Risk:

- Public RMSE may prefer conservative predictions if extremes are rare.
- Heavy-rain improvements can trade off against light-rain false positives.

Recommended experiment:

- Add bin metrics immediately to exp003 analysis.
- Use bin-aware training in exp004 or exp005 after baseline diagnostics are available.

### 5. Distribution calibration and OOF-driven post-processing are important

Because the public result can move without clear CV agreement, post-processing must be tuned on OOF,
not on leaderboard feedback.

Implementation idea for this repo:

- Continue writing OOF sample/group metrics as exp003 does.
- Add calibration candidates:
  - global linear scale/bias,
  - per-satellite scale/bias,
  - rain-probability threshold,
  - non-negative clip,
  - upper clipping percentile,
  - quantile mapping or empirical CDF matching.
- Record both overall RMSE and bin/satellite/location-level RMSE.

Relevant implementation reference:

- **pySTEPS**
  - URL: https://pysteps.readthedocs.io/en/stable/
  - Includes optical-flow nowcasting, ensemble verification, probability matching, and FSS-style
    verification. We probably should not depend on it directly inside the competition pipeline unless
    needed, but its verification and probability-matching ideas are useful.

Expected benefit:

- Can correct systematic over/underprediction.
- Makes ensemble blending more principled.
- Helps decide whether calibrated submission should be separate from raw submission.

Risk:

- Per-location calibration can leak if not handled through OOF. Avoid using target distribution from
  held-out evaluation data.

Recommended experiment:

- `exp007`: ensemble and OOF post-processing sweep across exp003/exp004/exp005/exp006 checkpoints.

### 6. Pretraining is rule-sensitive

General CV practice would suggest ImageNet-pretrained encoders or satellite foundation models, but
the competition says external datasets are prohibited. A pretrained model trained on external data is
likely not safe unless Solafune explicitly permits it.

Safer alternatives:

- Train from scratch.
- Self-supervised pretraining only on distributed train and evaluation satellite images, if the rules
  allow use of unlabeled evaluation inputs.
- Masked reconstruction, denoising, or next-frame prediction using the provided satellite images.

Implementation idea:

- Pretrain encoder on distributed satellite images with:
  - masked band/time reconstruction,
  - random crop/rotation reconstruction,
  - frame-order or next-frame auxiliary task.
- Fine-tune on precipitation labels.

Risk:

- More engineering time.
- Must document exactly which files are used.

Recommended experiment:

- Defer until the supervised pipeline is stronger. Treat as `exp008+`, not the next immediate step.

## Broader Architectures and Why They Are Lower Priority

### Transformer and attention models

Relevant references:

- **MetNet**
  - URL: https://arxiv.org/abs/2003.12140
  - Uses axial attention for large-context probabilistic precipitation forecasting.
- **Earthformer**
  - URL: https://arxiv.org/abs/2207.05833
  - Space-time transformer with cuboid attention for earth-system forecasting.
- **Global MetNet**
  - URL: https://arxiv.org/abs/2510.13050
  - Operational global satellite-based nowcasting model with large-scale data and additional inputs.

Why not first:

- These models usually need large training data, longer sequences, larger spatial context, and often
  external NWP/radar/sensor inputs.
- Our current images are small after resizing to the target grid, and the next bottleneck is probably
  task formulation rather than raw model capacity.

Possible later use:

- Add lightweight attention at bottleneck or skip connections.
- Use temporal attention over 3 frames before trying a full transformer.

### Generative/diffusion nowcasting

Relevant references:

- **DGMR**
  - URL: https://arxiv.org/abs/2104.00954
  - Deep generative radar nowcasting, strong for realistic probabilistic forecasts.
- **DYffCast**
  - URL: https://arxiv.org/abs/2412.02723
  - Diffusion-style regional precipitation nowcasting over IMERG.
- **DiffCast**
  - URL: https://arxiv.org/abs/2312.06734
  - Residual diffusion framework for radar echo nowcasting.

Why not first:

- The competition metric is deterministic RMSE, not probabilistic or perceptual quality.
- Generative models can produce sharper fields but are costly and harder to stabilize.
- A U-Net/ConvLSTM/two-head model is simpler and likely gives better return per GPU-hour.

Possible later use:

- Use diffusion/generative ideas only after deterministic OOF saturates.
- Consider probabilistic ensemble mean as RMSE target, not a single generated sample.

### Optical flow and classical nowcasting

Relevant references:

- **pySTEPS**
  - URL: https://pysteps.readthedocs.io/en/stable/
  - Modular framework for precipitation nowcasting, optical flow, STEPS ensemble, verification, and
    probability matching.

Why limited here:

- We do not have previous precipitation target frames at inference, only satellite observation
  frames. Optical flow on cloud bands may still help, but direct use of radar/precipitation
  extrapolation is not available.

Possible use:

- Estimate cloud motion from IR/water-vapor bands and add motion magnitude/direction features.
- Use only distributed satellite images, not external meteorological inputs.

## Kaggle and Remote-Sensing Competition Lessons

Although there may not be a public Kaggle solution for this exact Solafune competition, several
remote-sensing competition patterns are relevant.

### Multispectral segmentation pipelines

Reference:

- **Satellite Imagery Feature Detection using Deep Convolutional Neural Network: A Kaggle
  Competition**
  - URL: https://arxiv.org/abs/1706.06169
  - Discusses fully convolutional networks for multispectral satellite imagery, augmentation, and
    feature engineering.

Applicable lessons:

- Treat band semantics carefully; do not assume RGB-style preprocessing.
- Use stable augmentation and validation discipline.
- Keep ensembles and post-processing reproducible.

### Weather4cast precipitation competitions

References:

- **RainAI**
  - URL: https://arxiv.org/abs/2311.18398
- **PAUNet**
  - URL: https://arxiv.org/abs/2311.18306
- **Efficient Baseline for Quantitative Precipitation Forecasting in Weather4cast 2023**
  - URL: https://arxiv.org/abs/2311.18806

Applicable lessons:

- A well-designed 2D U-Net can outperform a heavier 3D baseline if data preparation and loss design
  are good.
- Importance sampling for rainy examples can matter as much as architecture.
- Attention and focal precipitation losses are reasonable follow-ups, but should be tested against
  a strong simple baseline.

## Solafune-Specific Notes

Sources checked:

- Solafune main site: https://solafune.com/
- Solafune competitions page: https://community.solafune.com/competitions
- Solafune tools GitHub: https://github.com/Solafune-Inc/solafune-tools

Findings:

- The public Solafune site confirms the platform focus on satellite/geospatial competitions and
  geospatial analytics.
- The competition listing page is dynamic and did not expose this competition's full internal
  discussion or solution material through static browsing.
- `solafune-tools` is a general geospatial/competition helper library. It may be useful for future
  utility checks, but the current repo already has custom GeoTIFF writing and submission creation.

Practical implication:

- Do not assume public Solafune competition solutions are available.
- Keep all rule-sensitive decisions documented in this repo.

## Prioritized Experiment Roadmap

| Priority | Experiment | Main idea | Why |
| ---: | --- | --- | --- |
| 1 | exp004 | Two-head rain detection + amount regression | Directly targets zero-rain imbalance |
| 2 | exp005 | Temporal shared encoder + ConvGRU/ConvLSTM fusion | Uses the 3 observations as a sequence |
| 3 | exp006 | Satellite-specific stems/adapters | Handles Himawari/GOES/Meteosat domain differences |
| 4 | exp007 | OOF ensemble + calibration sweep | Turns diagnostics into reliable submission choices |
| 5 | exp008+ | Self-supervised pretraining on provided satellite images | Potentially useful but more engineering and rule review |

## Recommended Metrics to Add

All future experiments should write these to CSV, preferably under `outputs/analysis/{exp_name}/`:

- Epoch-level train/valid RMSE.
- Valid RMSE by `satellite_target`.
- Valid RMSE by `name_location`.
- Valid RMSE by intensity bin.
- Positive-pixel RMSE and zero-pixel RMSE.
- Rain/no-rain classification metrics:
  - precision,
  - recall,
  - CSI,
  - false alarm ratio.
- Prediction distribution:
  - mean,
  - std,
  - min/max,
  - nonzero fraction,
  - p50/p90/p95/p99.
- Target distribution for validation rows.
- Calibration parameters selected from OOF only.

## Implementation Notes for exp004

The next implementation should be narrow and measurable:

1. Start from `g_experiments/exp003`.
2. Add model option `two_head_unet`.
3. Preserve the exp003 data loader, normalization, Slurm/Singularity runner, submission creation,
   and analysis output.
4. Add config fields:
   - `loss.rain_threshold`
   - `loss.bce_weight`
   - `loss.amount_weight`
   - `postprocess.rain_prob_threshold`
5. Add OOF threshold sweep:
   - thresholds from 0.05 to 0.95.
   - choose the best threshold by OOF RMSE.
6. Produce both:
   - raw submission,
   - thresholded/calibrated submission.

Acceptance criteria:

- `sbatch singularity_run.sh` can train all folds and create `exp004_submission.zip`.
- OOF analysis identifies whether the two-head formulation improves zero-rain false positives.
- Logs are sufficient to answer whether the gain/loss came from rain detection, rain amount, or
  post-processing.

## References

### Competition and data

- Solafune: https://solafune.com/
- Solafune competitions: https://community.solafune.com/competitions
- Solafune tools: https://github.com/Solafune-Inc/solafune-tools
- NASA IMERG overview: https://gpm.nasa.gov/data/imerg

### Precipitation nowcasting and satellite QPE

- ConvLSTM: https://arxiv.org/abs/1506.04214
- IMERG U-Net ConvLSTM: https://arxiv.org/abs/2307.10843
- Nowcasting-Nets: https://arxiv.org/abs/2108.06868
- MetNet: https://arxiv.org/abs/2003.12140
- DGMR: https://arxiv.org/abs/2104.00954
- Earthformer: https://arxiv.org/abs/2207.05833
- DYffCast: https://arxiv.org/abs/2412.02723
- TUPANN: https://arxiv.org/abs/2511.05471
- Oya: https://arxiv.org/abs/2511.10562
- Huayu: https://arxiv.org/abs/2512.15222
- PRISMA: https://arxiv.org/abs/2605.14426

### Competition-style and implementation references

- U-Net: https://arxiv.org/abs/1505.04597
- Kaggle DSTL satellite imagery solution: https://arxiv.org/abs/1706.06169
- RainAI / Weather4cast 2023: https://arxiv.org/abs/2311.18398
- PAUNet: https://arxiv.org/abs/2311.18306
- Efficient Weather4cast baseline: https://arxiv.org/abs/2311.18806
- pySTEPS documentation: https://pysteps.readthedocs.io/en/stable/
