# Research Survey Round 2: Satellite Precipitation Nowcasting

Last updated: 2026-07-10

## Purpose

Continuation of `doc/research_survey.md` (round 1, which led to exp004-exp007: two-head rain
detection, temporal fusion, satellite adapters, OOF-driven post-processing). Round 1's roadmap is
done; exp008-exp014 (see `doc/task_tickets.md` Round 2) added successor-row frames, official-metric
post-processing, dataset cleanup, and a verified tile-overlap copy exploit. This round asks: **given
what `doc/data_characteristics_review.md` and `l_eda/exp002` found about our own data, what does
2024-2026 literature say about the specific failure mode we now know dominates the score?**

That failure mode, precisely: `doc/data_characteristics_review.md` §1.3 found mid+heavy-rain tiles
are 41.3% of tiles but **87% of the tile_rmse error budget** (mid 53.0% + heavy 33.7%), and §1.5 found
heavy-rain amplitude is underestimated by ~4x (`pred_max/target_max` ≈ 0.25 for `target_max >= 10`
tiles). Rain-detection quality (CSI) has plateaued at 0.44-0.46 across exp008/009/010/011 regardless
of architecture (`doc/task_tickets.md` insight #9) while nothing has yet directly targeted amplitude
accuracy on the tail. This round of research is deliberately narrow: **find literature that targets
extreme-value amplitude accuracy under an MSE-family metric on a small dense-regression grid**, not a
general architecture survey.

Current reference results (from `doc/public_scores.md`, 2026-07-09):

| Experiment | Public RMSE | Notes |
| --- | ---: | --- |
| exp009 | 0.7153438899106017 | Successor-row frames — current best |
| exp011 | 0.7232307883574975 | Satellite adapter + two-head |
| exp008 | 0.7250185237499447 | Official metric + drizzle post-processing |
| exp004 | 0.7252533726905589 | Two-head rain detection + amount regression (round-1 anchor) |

exp012 (exp009 x exp011 merge, G-016) and exp013 (registration + wider context, G-017/G-018) are
implemented but not yet scored publicly as of this writing; exp014 (tile-overlap copy, G-022) has a
submission built but not yet scored either. None of round 1's or Round 2's changes so far have
directly attacked amplitude accuracy on the mid/heavy-rain tail — that is this document's focus.

## Hard Constraints Used to Filter Every Technique

Every candidate below is checked against these five constraints (from our own measured data
characteristics and competition rules), not against benchmark results on unrelated datasets:

1. **No external data/pretrained weights** without organizer confirmation. Anything requiring
   ImageNet/foundation-model weights or pretraining on other satellite datasets is flagged
   "rule-sensitive," same as round 1's treatment of this issue — not simply recommended.
2. **41x41 output grid** (satellite inputs up to 141x141 before resize). Architectures whose value
   proposition is a large receptive field, deep encoder stack, or long-range attention over
   hundreds of tokens are oversized here; anything proposed must justify itself at this scale.
3. **~20 independent train episodes**, not 40k independent samples (`doc/data_characteristics_review.md`
   §1.1). Techniques whose benefit comes primarily from data volume/diversity (self-supervised
   pretraining requiring many independent scenes, big quantile-ensemble ranges needing many extreme
   examples per class) are suspect until checked against our actual extreme-event count.
4. **Small motion**: sub-pixel to a few pixels over 30min (`outputs/l_eda/exp001`). exp005's
   temporal-fusion architecture already lost to naive frame-stacking empirically — any
   motion-modeling proposal must explain why it would do better than that result, not just cite
   general nowcasting literature where motion is larger (radar nowcasting typically covers
   256x256km+ domains with visible storm translation over the lead time).
5. **Metric is `tile_rmse`**: `mean over samples of sqrt(mean per-tile squared pixel error)`
   (`g_experiments/exp008/README.md`) — plain amplitude accuracy, not CRPS, SSIM, or a perceptual
   score. Probabilistic/generative techniques must be translated into a single point estimate that
   minimizes this specific metric; "looks more realistic" is not itself a reason to adopt one.

## Main Research Takeaways

### 1. Multi-quantile regression heads for the amount head (highest priority)

MSE-trained regressors are conditional-mean estimators, so they structurally cannot reach the upper
tail of a skewed target — this is exactly what we measured (`pred_max/target_max` ≈ 0.25 at
`target_max >= 10`). Recent literature treats this as "deep imbalanced regression," not a novel
weather-specific problem, which is useful: the fixes are architecture-agnostic and cheap to retrofit.

Most relevant reference:

- **Multi-Quantile Regression for Extreme Precipitation Downscaling** (2026)
  - URL: https://arxiv.org/abs/2605.12762
  - Trains a CNN (Q-SRDRN) with **pinball loss at multiple quantile levels**
    (τ ∈ {0.50, 0.95, 0.99, 0.999}) via **separate per-quantile output heads**, plus an
    `IncrementBound` mechanism enforcing monotonicity across quantile heads while keeping gradients
    distinct per head — i.e. a small architectural addition (N extra 1x1 conv heads), not a new
    training pipeline.
  - Reported gains are large: on one region, extreme-event (200mm/day-equivalent) detection rate
    for the p999 head went from 4.2% (baseline) to 75.7% (18x), with 63% lower KL divergence on the
    tail distribution. Other regions show similarly large detection-rate jumps.
- **Balanced MSE for Imbalanced Visual Regression** (CVPR 2022, still the standard baseline cited by
  2025-2026 imbalanced-regression work) and the 2026 review **"Deconstructing deep imbalanced
  regression"** (Artificial Intelligence Review) — taxonomy of retrofit-cheap options: static
  reweighting by label-density bins (what our `WeightedMSELoss.pos_weight` already approximates),
  Focal-R (dynamically rescale by current error magnitude, no extra head needed), Balanced MSE
  (reweights via an estimated label density, needs a label histogram), DistLoss (2025, adds a
  distribution-alignment term between predicted and target batch distributions).
  - URL: https://arxiv.org/abs/2203.16427 ; https://link.springer.com/article/10.1007/s10462-026-11570-1

Why this fits our constraints: pinball-loss quantile heads are a small addition to an existing
`amount_head` (replace or supplement its single-channel output with 3-4 channels + monotonicity
constraint), need no external data, work with our existing ~2M-param CompactUNet body, and directly
target the exact failure mode we measured rather than a generic "more capacity" bet. The open
question specific to us: **our metric is `tile_rmse`, not calibration/detection-rate**, so we cannot
just report the p999 head's detection improvement like the source paper — we need an OOF sweep over
which quantile (or blend, e.g. `0.5*median + 0.5*p90`) minimizes `tile_rmse`, since the metric
rewards the conditional mean, not a well-calibrated tail. This is a genuinely new experiment, not a
drop-in win — propose as a fold-0 A/B before any 5-fold commitment.

**Proposed ticket G-026**: Add quantile output heads (τ ∈ {0.5, 0.9, 0.99}) to the amount branch of
`two_head_compact_unet` / `satellite_adapter_two_head_unet`, trained with pinball loss +
monotonicity constraint, alongside the existing rain-probability head unchanged. OOF-sweep which
quantile (or fixed linear blend) minimizes `tile_rmse` specifically, separately for the whole
distribution and for the mid/heavy-rain subset (`doc/data_characteristics_review.md` §1.3 bins).
Start from whichever of exp012/exp013 is winning by then. Priority: P0.

### 2. IR-to-rain calibration curve as an explicit component, not just an implicit CNN mapping

Round 1 already covered PERSIANN-family IR-QPE and the "two-stage detection + estimation" idea
(→ exp004). What's new: a concrete, recent competition precedent for making the
IR-brightness-to-rain-rate relationship an *explicit* fitted curve rather than leaving it entirely
implicit inside the CNN.

- **A solution for the Weather4cast 2025 challenge (computationally-efficient models)**
  - URL: https://arxiv.org/abs/2511.11197
  - Placed 2nd on the Weather4cast 2025 cumulative-rainfall task using a deliberately minimal
    pipeline: a **single IR channel** (10.8μm), a ConvGRU to forecast future brightness-temperature
    fields, then an **empirically-derived nonlinear transformation** converting predicted BT into
    rainfall rate as a distinct second stage — explicitly *not* end-to-end regression from BT to
    rain in one network.
- We already computed exactly this empirical curve for our own data: `outputs/l_eda/exp002/bt_rain_response.csv`
  gives `E[rain | IR value]` and `P(rain>0 | IR value)` per satellite, e.g. for himawari's coldest
  IR bin, `mean_rain=5.78`, `rain_prob=0.94` (`l_eda/exp002/README.md`). This is a strongly
  non-linear, monotonic curve per satellite.

Why this fits our constraints: this is not "add a bigger model," it's "use data we already computed."
Two ways to use it that stay small: (a) as an **auxiliary input channel** — the per-pixel value of
this curve evaluated at each observed IR pixel, giving the network a monotonic "physical prior" it
doesn't have to relearn from scratch on ~20 episodes; (b) as an **isotonic-regression post-hoc
calibration** on the amount head's output, fit on OOF predictions the same way `oof_calibration.json`
already fits linear scale/bias in exp008+ — isotonic regression instead of linear preserves
monotonicity without adding model capacity at all, so it is essentially free to try. Directly reuses
data already in the repo, no new EDA needed.

**Proposed ticket G-027**: (a) Add an OOF-fit isotonic (or piecewise-linear) calibration curve
mapping raw amount-head output to a calibrated value, evaluated the same way `oof_calibration.json`
currently stores linear scale/bias — cheapest version, try first, on the current best submission
before touching training. (b) If (a) doesn't fully close the gap, add the per-satellite
`bt_rain_response` curve evaluated at the coldest input IR band as one extra input channel (derived
purely from distributed data, no external source). Priority: P0 for (a) — near-zero cost — P1 for (b).

### 3. Small-tile architecture: avoid over-downsampling, prefer dilation over depth

Round 1 didn't examine whether CompactUNet's 2-stage downsample (41→~21→~11) is well-matched to a
grid this small, since it was carried over unchanged from exp001. `outputs/l_eda/exp002`'s
`radial_power_spectrum.csv` gives a direct, repo-specific answer: **~81% of target spatial-frequency
power sits at wavenumber 1-2** (wavelength 20-41px, i.e. features spanning most of the tile), with
power dropping off fast at higher wavenumbers (finer detail). This means the network's job is mostly
about a small number of large-scale spatial patterns, not fine texture — which argues against adding
depth/downsampling for a bigger receptive field (we already cover the tile in 2 stages) and mildly
favors techniques that grow the receptive field *without* discarding resolution.

- General literature on dilated convolutions in U-Nets confirms the mechanism (exponential receptive
  field growth without downsampling, at the cost of some local-feature granularity at high dilation
  rates) but nothing found is specific to grids this small (41x41) or to precipitation — this is a
  weaker, more exploratory lead than 1-2, included for completeness rather than as a strong bet.

Why this fits our constraints: replacing `enc2`/`enc3`'s stride-2 pooling with dilated convolutions
at fixed resolution is a same-parameter-budget architectural swap, not a scale-up — it fits our tiny
model regime and needs no new data. But the expected effect size is unclear without our own ablation,
unlike takeaways 1-2 which have measured effect sizes from directly analogous work.

**Proposed ticket G-028**: Single-fold ablation, `CompactUNet` bottleneck with dilated convs
(dilation 2, 4) replacing the `avg_pool2d` downsampling in `enc2`/`enc3`, same channel widths, same
data/loss as whichever architecture is the current fold-0 champion. Compare OOF `tile_rmse` and,
specifically, the mid/heavy-rain-tile subset RMSE (where the error budget lives). Priority: P2 —
plausible but speculative relative to 1-2.

### 4. Frequency-domain regularization against MSE blur (exploratory, cheap to prototype)

Round 1 covered diffusion/generative approaches (DGMR, DYffCast) as ways to fight "MSE blur," ruled
low-priority because our metric is deterministic RMSE, not perceptual quality. A 2025 result gives a
non-generative alternative worth a brief mention because it's cheap and reuses tooling we already have.

- **Fixing the Double Penalty in Data-Driven Weather Forecasting Through a Modified Spherical
  Harmonic Loss Function** (ICML 2025)
  - URL: https://arxiv.org/abs/2501.19374
  - Diagnoses the "double penalty" effect: pixel-wise MSE punishes a correctly-shaped but slightly
    displaced feature twice (once for the miss, once for the false "blank" where it should have
    been), which is exactly what drives models toward blurry, averaged-out predictions. Their fix
    separates a spectral-amplitude loss term from a positional/decorrelation loss term using
    spherical-harmonic decomposition (global-sphere-specific, not applicable to our flat 41x41 tile
    as-is).
- **WADEPre** (wavelet-decomposition nowcasting, 2026): reformulates the *prediction target* itself
  into multi-scale (wavelet) coefficients so common light-rain and rare extreme structure are
  separated into different frequency bands before the loss is applied — described as lightweight
  relative to DGMR/diffusion baselines. URL: https://arxiv.org/abs/2602.02096

Why this fits our constraints, and why it's ranked low: the *principle* (penalize spectral amplitude
error separately from a smoothed/blurred prediction, rather than one pixel-wise MSE term) is cheap to
prototype on our own data — `l_eda/exp002`'s `radial_power_spectrum` code already computes 2D FFT
radial power, so a training-time auxiliary loss term comparing predicted vs. target radial power
spectra is a same-day addition, not a new subsystem. But WADEPre's core mechanism (multi-scale
wavelet decomposition) needs several usable octaves to pay off, and our target's power is already
concentrated in the lowest 1-2 wavenumbers of a 41px tile (§3 above) — there may not be enough scale
range on this tile size for wavelet decomposition specifically to help, even though the general
frequency-domain framing is sound. Spherical-harmonic methods are not applicable to a flat tile at
all and are cited only for the underlying diagnosis, not as a technique to port.

**Proposed ticket G-029**: Add a small auxiliary loss term (weight tunable, likely small) comparing
the radial power spectrum of prediction vs. target on the amount head's output, using the existing
`radial_power_spectrum` FFT code as a starting point, on top of whichever loss wins from G-026/G-027.
Treat as exploratory; run only after G-026/G-027 are evaluated, since it targets the same blur/tail
symptom via a different mechanism and comparing three simultaneous changes would be unreadable.
Priority: P3.

### 5. Day/night conditioning: architectural precedent for the existing G-023 ticket

`doc/task_tickets.md` G-023 already proposes a solar-zenith/day-night input feature, justified by
`l_eda/exp002`'s finding that same-rain-regime OOF error is 16-38% higher for visually dark tiles.
This round adds one implementation-relevant precedent, not a new ticket:

- Satellite cloud-retrieval literature commonly handles day/night not just as an extra scalar
  feature but as **explicit regime separation** — e.g. training separate models (or model branches)
  for day (`0° ≤ SZA < 80°`), twilight (`80°-90°`), and night (`≥90°`) scenes, because the visible
  bands aren't just "dimmer" at twilight, they become unreliable in a regime-dependent way rather
  than a smoothly scaling one. Illumination-normalization approaches (adaptive normalization keyed
  on solar zenith angle) are also common for the day-to-day sunrise/noon/sunset range within the
  "day" regime itself.

Why this fits our constraints: this doesn't change G-023's priority or cost estimate (still
datetime-derived, no new data, ~1 day), but it argues for evaluating **regime gating** (e.g. a
FiLM-style scale/bias conditioned on a discretized day/twilight/night bucket, or even just adding the
bucket as a one-hot alongside the existing continuous solar-angle feature) rather than assuming a
single continuous SZA input channel captures the effect linearly enough for our from-scratch
~2M-param CNN to learn the regime boundary from ~20 episodes. Recommend G-023's implementation note
be updated to try both a continuous SZA channel and a 3-bucket one-hot, compared via OOF on the
existing day/night conditional split in `outputs/l_eda/exp002/oof_daynight_conditional.csv`.

### 6. Why large transformers and generative nowcasting remain deprioritized (updated, not repeated)

Round 1 already ruled out MetNet/Earthformer/DGMR/DYffCast at a general level; this section only adds
what's new since then.

- **A Space-Time Transformer for Precipitation Nowcasting (SaTformer)**, 1st place, Weather4cast 2025
  cumulative-rainfall task. URL: https://arxiv.org/abs/2511.11090. Uses full space-time attention
  over **11-band SEVIRI imagery at 15-minute cadence**, city/regional-scale tiles far larger than our
  41x41 grid, optimized for a longer forecast horizon than our ≤30min/single-step estimation task.
  The fact that a full-attention transformer wins *at that scale* is not evidence it would win at
  ours — if anything, the 2nd-place solution (§2 above, a single IR channel + ConvGRU + explicit
  calibration curve) is the more analogous result, and it explicitly chose a smaller, more
  interpretable pipeline over a bigger model. This is a useful data point: **even within one
  competition, the lighter, more physically-structured approach was competitive with the heaviest
  one**, which supports staying on our current "small CNN + explicit inductive bias" trajectory
  rather than pivoting to attention-heavy architectures.
- No genuinely new evidence found this round that changes round 1's conclusion on DGMR-style
  diffusion nowcasting for our deterministic-RMSE, small-tile setting.

## Prioritized Experiment Roadmap (Round 2, continuing `doc/task_tickets.md`)

| Priority | Ticket | Experiment | Main idea | Why |
| ---: | --- | --- | --- | --- |
| P0 | G-027a | post-processing | OOF-fit isotonic calibration of amount-head output | Near-zero cost, reuses existing `oof_calibration.json` mechanism, targets the measured amplitude bias directly |
| P0 | G-026 | exp012/exp013 + quantile heads | Pinball-loss multi-quantile amount head, OOF-swept for `tile_rmse` | Largest measured effect size in the literature found (18x extreme-detection gain in the source paper); directly targets the 87%-of-error-budget mid/heavy-rain tail |
| P1 | G-027b | input feature | Per-satellite IR→rain empirical curve as auxiliary input channel | Reuses `bt_rain_response.csv` already computed; matches a real 2nd-place competition technique |
| P1 | G-023 (update) | exp015 | Solar zenith input — add day/twilight/night regime bucket, not just continuous SZA | New architectural precedent from satellite-retrieval literature; same cost as originally scoped |
| P2 | G-028 | architecture | Dilated-conv bottleneck instead of extra downsampling | Matches our own measured spectral-power concentration finding; effect size unproven, needs a fold-0 ablation |
| P3 | G-029 | loss | Radial-power-spectrum auxiliary loss term | Cheap to prototype (reuses existing FFT code) but speculative; run after G-026/G-027 to keep comparisons readable |

## References

### Amplitude/extreme-value accuracy (new this round)

- Multi-Quantile Regression for Extreme Precipitation Downscaling: https://arxiv.org/abs/2605.12762
- Balanced MSE for Imbalanced Visual Regression: https://arxiv.org/abs/2203.16427
- Deconstructing deep imbalanced regression (review): https://link.springer.com/article/10.1007/s10462-026-11570-1
- Delving into Deep Imbalanced Regression: https://arxiv.org/abs/2102.09554
- NowcastingGPT with Extreme Value Loss: https://arxiv.org/abs/2403.03929
- PP-Loss (plotting-position imbalanced regression loss for precipitation nowcasting): https://link.springer.com/article/10.1007/s00704-024-04984-w
  (found via search; full text was paywalled during this survey — worth a follow-up fetch before
  implementing G-026, since it may be more directly precipitation-specific than the general
  imbalanced-regression literature above)

### IR-rain calibration and Weather4cast 2024-2025

- Weather4cast 2025 computationally-efficient solution (2nd place): https://arxiv.org/abs/2511.11197
- SaTformer, Weather4cast 2025 1st place: https://arxiv.org/abs/2511.11090
- Weather4cast 2025 site: https://weather4cast.net/neurips2025/

### Small-scale / frequency-domain

- Double-penalty spherical harmonic loss fix (ICML 2025): https://arxiv.org/abs/2501.19374
- WADEPre (wavelet decomposition, extreme precipitation): https://arxiv.org/abs/2602.02096

### Day/night conditioning

- General finding on solar-zenith-angle regime modeling in satellite cloud retrieval (search-derived,
  no single canonical citation — see search summary in this document's drafting notes; treat as a
  design precedent, not a specific paper to replicate).

### Not pursued further this round (checked, no new actionable material)

- Cloud-motion-vector-replacement solar forecasting paper (Straub et al. 2024,
  https://onlinelibrary.wiley.com/doi/full/10.1002/solr.202400475) — paywalled, could not verify
  claims; the general "deep learning can replace explicit motion vectors at short lead times" theme
  already matches our own exp005 finding, so this is low-priority to chase further.
