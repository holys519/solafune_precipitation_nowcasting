"""Model architectures for exp056.

Carried over UNCHANGED from exp038 (copied file, not imported, per the self-contained-experiment
convention) as control/reference arms:
- "compact_unet", "two_head_compact_unet", "hurdle_lognormal_unet", "smp_unet",
  "highres_hurdle_lognormal_unet" (exp038's own current green champion architecture).

New in exp056: `FactorizedMeanShapeUNet` (`architecture: factorized_mean_shape`) -- the
mean-intensity x normalized-shape factorization recommended by
doc/research_survey_v3_2026-07-16.md's own final-judgment section (Section 10) and motivated by
the oracle-ladder finding in g_eda/exp002 / doc/plan/round5_experiment_plan_2026-07-16.md Section 8
that per-tile AMOUNT error, not spatial placement, is this architecture family's dominant residual
(exp018: actual 0.6093 -> amount_swap 0.5446, a -0.065 rung; mask_swap 0.7111, worse than actual).
Instead of exp038's joint per-pixel (mu, sigma) log-normal intensity field, this model explicitly
factorizes the served amount into:
  1. a single scalar per tile, `mean_intensity` (globally pooled encoder features -> small MLP ->
     softplus), and
  2. a full-resolution `shape` field (reusing the same high-res decoder head design as exp038),
     softplus'd non-negative and then normalized -- decoupling "where" from "how much".
Served amount = mean_intensity (broadcast) * shape; served pred = rain_prob * amount, gated by the
SAME occurrence head design as exp038 (unweighted BCE on rain_logits), exactly mirroring
`prediction_from_output`'s existing convention -- an explicit hurdle gate, mirroring exp038's
`rain_prob * exp(mu + sigma^2/2)`.

2026-07-24 bugfix (fold0/4 gate originally failed both folds by +0.025..+0.033 tile_rmse): the
first version regressed `mean_intensity` against the FULL-TILE mean of target (dry pixels
included, matching g_eda/exp002's `amount_swap` counterfactual exactly) while STILL gating the
served amount by `rain_prob` -- double-diluting wet-pixel predictions by roughly
1/target_positive_ratio (~5.6x too small at this dataset's ~0.18 positive ratio), since
exp038's reference mu/sigma heads are fit on wet pixels ONLY, precisely so that gating by
rain_prob recovers the right unconditional expectation. `mean_intensity` and the `shape` training
target are now both defined relative to the WET-PIXEL-CONDITIONAL mean (`losses.py`'s
`wet_mean = wet_pixel_sum / wet_pixel_count`), matching the reference hurdle design. The model's
own `shape` self-normalization below (computed without ground truth, since it also runs at
inference) uses a `rain_prob`-weighted spatial average as a soft proxy for "the wet-pixel mean",
consistent with what's now supervised. See README.md for the full rationale, the exact loss
composition (in losses.py), and the design choices spelled out explicitly.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, dilation: int = 1) -> None:
        super().__init__()
        groups = 8 if out_channels % 8 == 0 else 1
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=dilation, dilation=dilation, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.block(x)


class CompactUNet(nn.Module):
    def __init__(self, in_channels: int = 54, base_channels: int = 48) -> None:
        super().__init__()
        c = base_channels
        self.enc1 = ConvBlock(in_channels, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.bottleneck = ConvBlock(c * 4, c * 4)
        self.dec2 = ConvBlock(c * 4 + c * 2, c * 2)
        self.dec1 = ConvBlock(c * 2 + c, c)
        self.head = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, 1, kernel_size=1),
        )

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, kernel_size=2, ceil_mode=True))
        e3 = self.enc3(F.avg_pool2d(e2, kernel_size=2, ceil_mode=True))
        b = self.bottleneck(e3)
        d2 = F.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        return self.dec1(torch.cat([d1, e1], dim=1))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.head(self.forward_features(x))


class TwoHeadCompactUNet(CompactUNet):
    """Compact U-Net with explicit rain occurrence and rain amount heads."""

    def __init__(self, in_channels: int = 54, base_channels: int = 48) -> None:
        super().__init__(in_channels=in_channels, base_channels=base_channels)
        c = base_channels
        self.rain_head = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, 1, kernel_size=1),
        )
        self.amount_head = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, 1, kernel_size=1),
        )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.forward_features(x)
        rain_logits = self.rain_head(features)
        rain_prob = torch.sigmoid(rain_logits)
        rain_amount = F.softplus(self.amount_head(features))
        pred = rain_prob * rain_amount
        return {
            "pred": pred,
            "rain_logits": rain_logits,
            "rain_prob": rain_prob,
            "rain_amount": rain_amount,
        }


class HurdleLogNormalUNet(CompactUNet):
    """Compact U-Net with a hurdle (zero-inflated log-normal) head.

    E[Y|X] = P(rain|X) * E[Y|rain,X] with E[Y|rain,X] = exp(mu + sigma^2/2).
    - rain_head outputs logits; must be trained with UNWEIGHTED BCE so sigmoid(logits) is a
      calibrated probability (calibration is what makes the product an unbiased mean).
    - mu_head predicts E[ln(y) | rain, X]; trained on wet pixels only.
    - sigma is either predicted per-pixel (log_sigma_head, Gaussian NLL training) or a fixed
      config constant (mu trained with wet-only MSE on ln(y)).
    - serving_sigma_scale scales the sigma^2/2 mean correction at serving time only:
      1.0 = log-normal mean (default), 0.0 = median serving (ablation arm to confirm the
      measured ~3.8x mean/median gap on our own OOF).
    """

    def __init__(
        self,
        in_channels: int = 105,
        base_channels: int = 48,
        sigma_mode: str = "predicted",
        fixed_sigma: float = 1.0,
        mu_min: float = -6.0,
        mu_max: float = 5.0,
        sigma_min: float = 0.1,
        sigma_max: float = 2.0,
        amount_cap: float = 150.0,
        serving_sigma_scale: float = 1.0,
    ) -> None:
        super().__init__(in_channels=in_channels, base_channels=base_channels)
        if sigma_mode not in ("predicted", "fixed"):
            raise ValueError(f"Unknown sigma_mode: {sigma_mode!r}")
        c = base_channels
        self.sigma_mode = sigma_mode
        self.fixed_sigma = float(fixed_sigma)
        self.mu_min = float(mu_min)
        self.mu_max = float(mu_max)
        self.sigma_min = float(sigma_min)
        self.sigma_max = float(sigma_max)
        self.amount_cap = float(amount_cap)
        self.serving_sigma_scale = float(serving_sigma_scale)
        self.rain_head = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, 1, kernel_size=1),
        )
        self.mu_head = nn.Sequential(
            nn.Conv2d(c, c, kernel_size=3, padding=1),
            nn.SiLU(inplace=True),
            nn.Conv2d(c, 1, kernel_size=1),
        )
        if sigma_mode == "predicted":
            self.log_sigma_head = nn.Sequential(
                nn.Conv2d(c, c, kernel_size=3, padding=1),
                nn.SiLU(inplace=True),
                nn.Conv2d(c, 1, kernel_size=1),
            )

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.forward_features(x)
        rain_logits = self.rain_head(features)
        mu = self.mu_head(features).float().clamp(self.mu_min, self.mu_max)
        if self.sigma_mode == "predicted":
            # softplus keeps the pre-clamp value smooth; clamp bounds it away from collapse (~0)
            # and blow-up (exp(sigma^2/2) at sigma_max=2.0 is x7.4, already generous).
            sigma = F.softplus(self.log_sigma_head(features).float()).clamp(self.sigma_min, self.sigma_max)
        else:
            sigma = torch.full_like(mu, self.fixed_sigma)
        rain_prob = torch.sigmoid(rain_logits.float())
        amount = torch.exp(mu + self.serving_sigma_scale * 0.5 * sigma * sigma)
        amount = amount.clamp(max=self.amount_cap)
        pred = rain_prob * amount
        return {
            "pred": pred,
            "rain_logits": rain_logits,
            "rain_prob": rain_prob,
            "mu": mu,
            "sigma": sigma,
            "rain_amount": amount,
        }


class HighResHurdleLogNormalUNet(nn.Module):
    """Four-level high-resolution U-Net with hurdle and auxiliary spatial-mask heads.

    Only the input/features are resized. Predictions are adaptively pooled back to the native
    target grid so the 41x41 GPM target is never interpolated during training.
    """

    def __init__(
        self,
        in_channels: int = 105,
        base_channels: int = 32,
        internal_size: int = 128,
        output_size: tuple[int, int] = (41, 41),
        aux_mask_enabled: bool = True,
        sigma_mode: str = "predicted",
        fixed_sigma: float = 1.0,
        mu_min: float = -6.0,
        mu_max: float = 5.0,
        sigma_min: float = 0.1,
        sigma_max: float = 2.0,
        amount_cap: float = 150.0,
        serving_sigma_scale: float = 1.0,
        bottleneck_dilations: tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        if sigma_mode not in ("predicted", "fixed"):
            raise ValueError(f"Unknown sigma_mode: {sigma_mode!r}")
        c = base_channels
        self.internal_size = int(internal_size)
        self.output_size = tuple(int(v) for v in output_size)
        self.aux_mask_enabled = bool(aux_mask_enabled)
        self.sigma_mode = sigma_mode
        self.fixed_sigma = float(fixed_sigma)
        self.mu_min, self.mu_max = float(mu_min), float(mu_max)
        self.sigma_min, self.sigma_max = float(sigma_min), float(sigma_max)
        self.amount_cap = float(amount_cap)
        self.serving_sigma_scale = float(serving_sigma_scale)

        self.enc1 = ConvBlock(in_channels, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.enc4 = ConvBlock(c * 4, c * 8)
        if bottleneck_dilations:
            # exp030's receptive-field expansion, ported to the high-res net. The bottleneck
            # here sits at internal_size/16 (8x8 for 128), so dilation 4 spans the whole grid.
            self.bottleneck: nn.Module = nn.Sequential(
                *(ConvBlock(c * 8, c * 8, dilation=int(d)) for d in bottleneck_dilations)
            )
        else:
            self.bottleneck = ConvBlock(c * 8, c * 8)
        self.dec4 = ConvBlock(c * 8 + c * 8, c * 8)
        self.dec3 = ConvBlock(c * 8 + c * 4, c * 4)
        self.dec2 = ConvBlock(c * 4 + c * 2, c * 2)
        self.dec1 = ConvBlock(c * 2 + c, c)

        def head() -> nn.Sequential:
            return nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.SiLU(inplace=True), nn.Conv2d(c, 1, 1))

        self.rain_head = head()
        self.mu_head = head()
        if sigma_mode == "predicted":
            self.log_sigma_head = head()
        if self.aux_mask_enabled:
            self.aux_mask_head = head()

    @staticmethod
    def _up(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(self.internal_size, self.internal_size), mode="bilinear", align_corners=False)
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        e4 = self.enc4(F.avg_pool2d(e3, 2))
        b = self.bottleneck(F.avg_pool2d(e4, 2))
        d4 = self.dec4(torch.cat([self._up(b, e4), e4], dim=1))
        d3 = self.dec3(torch.cat([self._up(d4, e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._up(d3, e2), e2], dim=1))
        return self.dec1(torch.cat([self._up(d2, e1), e1], dim=1))

    def _native(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(x, self.output_size)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.forward_features(x)
        rain_logits = self._native(self.rain_head(features))
        mu = self._native(self.mu_head(features)).float().clamp(self.mu_min, self.mu_max)
        if self.sigma_mode == "predicted":
            sigma = F.softplus(self._native(self.log_sigma_head(features)).float()).clamp(
                self.sigma_min, self.sigma_max
            )
        else:
            sigma = torch.full_like(mu, self.fixed_sigma)
        rain_prob = torch.sigmoid(rain_logits.float())
        amount = torch.exp(mu + self.serving_sigma_scale * 0.5 * sigma.square()).clamp(max=self.amount_cap)
        output = {
            "pred": rain_prob * amount,
            "rain_logits": rain_logits,
            "rain_prob": rain_prob,
            "mu": mu,
            "sigma": sigma,
            "rain_amount": amount,
        }
        if self.aux_mask_enabled:
            output["aux_mask_logits"] = self._native(self.aux_mask_head(features))
        return output


class FactorizedMeanShapeUNet(nn.Module):
    """High-res U-Net sharing `HighResHurdleLogNormalUNet`'s encoder/decoder backbone, but with the
    wet-pixel intensity factorized into a tile-level mean-intensity scalar x a normalized spatial
    shape field, instead of a joint per-pixel (mu, sigma) log-normal field. See module docstring
    and README.md for the motivating oracle-ladder evidence and design rationale.

    Heads:
    - rain_head: UNCHANGED occurrence head design from exp038 (native rain_logits -> sigmoid ->
      rain_prob), trained with the same unweighted BCE. Not part of this ablation.
    - mean_intensity head: global-average-pooled decoder features (the same `features` tensor the
      spatial heads read from, pooled over its 128x128 internal grid) -> 2-layer MLP -> softplus.
      Produces one non-negative scalar per tile, predicting the tile's FULL-TILE mean of the true
      target (matches g_eda/exp002's `amount_swap` oracle target exactly, not a wet-pixel-only
      mean -- see model.py module docstring).
    - shape_head: a second full high-res decoder head (identical conv-head design to rain_head/
      mu_head), softplus'd non-negative, then divided by its own full-tile spatial mean (+eps) so
      shape's mean is exactly 1 per tile -- explicitly decoupled from overall amount.
    - aux_mask_head: UNCHANGED coarse wet/dry auxiliary head from exp038 (same
      `aux_mask_weight`/threshold/dice loss), kept so the auxiliary-supervision budget matches
      exp038's exactly and this stays a pure ablation of the amount-factorization axis.

    Served amount = mean_intensity (broadcast over H,W) * shape, capped at `amount_cap`.
    Served pred = rain_prob * amount -- the same gating convention as
    `HighResHurdleLogNormalUNet.forward`/`prediction_from_output`.
    """

    def __init__(
        self,
        in_channels: int = 54,
        base_channels: int = 32,
        internal_size: int = 128,
        output_size: tuple[int, int] = (41, 41),
        aux_mask_enabled: bool = True,
        mean_intensity_hidden: int = 64,
        shape_eps: float = 1e-6,
        amount_cap: float = 150.0,
        bottleneck_dilations: tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        c = base_channels
        self.internal_size = int(internal_size)
        self.output_size = tuple(int(v) for v in output_size)
        self.aux_mask_enabled = bool(aux_mask_enabled)
        self.shape_eps = float(shape_eps)
        self.amount_cap = float(amount_cap)

        self.enc1 = ConvBlock(in_channels, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.enc4 = ConvBlock(c * 4, c * 8)
        if bottleneck_dilations:
            self.bottleneck: nn.Module = nn.Sequential(
                *(ConvBlock(c * 8, c * 8, dilation=int(d)) for d in bottleneck_dilations)
            )
        else:
            self.bottleneck = ConvBlock(c * 8, c * 8)
        self.dec4 = ConvBlock(c * 8 + c * 8, c * 8)
        self.dec3 = ConvBlock(c * 8 + c * 4, c * 4)
        self.dec2 = ConvBlock(c * 4 + c * 2, c * 2)
        self.dec1 = ConvBlock(c * 2 + c, c)

        def head() -> nn.Sequential:
            return nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.SiLU(inplace=True), nn.Conv2d(c, 1, 1))

        self.rain_head = head()
        self.shape_head = head()
        if self.aux_mask_enabled:
            self.aux_mask_head = head()
        self.mean_intensity_mlp = nn.Sequential(
            nn.Linear(c, mean_intensity_hidden),
            nn.SiLU(inplace=True),
            nn.Linear(mean_intensity_hidden, 1),
        )

    @staticmethod
    def _up(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(self.internal_size, self.internal_size), mode="bilinear", align_corners=False)
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        e4 = self.enc4(F.avg_pool2d(e3, 2))
        b = self.bottleneck(F.avg_pool2d(e4, 2))
        d4 = self.dec4(torch.cat([self._up(b, e4), e4], dim=1))
        d3 = self.dec3(torch.cat([self._up(d4, e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._up(d3, e2), e2], dim=1))
        return self.dec1(torch.cat([self._up(d2, e1), e1], dim=1))

    def _native(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(x, self.output_size)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.forward_features(x)
        rain_logits = self._native(self.rain_head(features))
        rain_prob = torch.sigmoid(rain_logits.float())

        shape_raw = F.softplus(self._native(self.shape_head(features)).float())
        # rain_prob-weighted spatial average as a soft proxy for "the wet-pixel mean of shape_raw"
        # -- ground truth wet mask isn't available at inference, but this stays consistent with
        # losses.py's wet-pixel-conditional shape_target (2026-07-24 bugfix, see module docstring).
        # A plain weighted average is always bounded within [min(shape_raw), max(shape_raw)]
        # regardless of how small rain_prob's weights get, so this is numerically safe even for
        # confidently-all-dry tiles (rain_prob ~ 0 everywhere).
        rain_prob_detached = rain_prob.detach()
        shape_mean = (shape_raw * rain_prob_detached).sum(dim=(2, 3), keepdim=True) / (
            rain_prob_detached.sum(dim=(2, 3), keepdim=True) + self.shape_eps
        )
        shape = shape_raw / (shape_mean + self.shape_eps)

        pooled = features.float().mean(dim=(2, 3))  # global average pool over the 128x128 grid
        # .float() before softplus, matching shape_raw's cast above: under AMP the MLP itself may
        # still run in fp16 (autocast casts fp32 inputs back down for Linear), and softplus(x) can
        # overflow fp16 once x exceeds ~11 (softplus is applied to raw, unbounded MLP output).
        mean_intensity = F.softplus(self.mean_intensity_mlp(pooled).float()).view(-1, 1, 1, 1)

        amount = (mean_intensity * shape).clamp(max=self.amount_cap)
        output = {
            "pred": rain_prob * amount,
            "rain_logits": rain_logits,
            "rain_prob": rain_prob,
            "shape": shape,
            "shape_raw": shape_raw,
            "mean_intensity": mean_intensity.view(-1),
            "rain_amount": amount,
        }
        if self.aux_mask_enabled:
            output["aux_mask_logits"] = self._native(self.aux_mask_head(features))
        return output


class ConvLSTMCell(nn.Module):
    """Single-step ConvLSTM cell: convolutional gates over spatial feature maps."""

    def __init__(self, in_channels: int, hidden_channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        padding = kernel_size // 2
        self.hidden_channels = hidden_channels
        self.gates = nn.Conv2d(in_channels + hidden_channels, 4 * hidden_channels, kernel_size, padding=padding)

    def forward(self, x: torch.Tensor, h: torch.Tensor, c: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        combined = torch.cat([x, h], dim=1)
        gates = self.gates(combined)
        i, f, o, g = gates.chunk(4, dim=1)
        i, f, o, g = torch.sigmoid(i), torch.sigmoid(f), torch.sigmoid(o), torch.tanh(g)
        c_next = f * c + i * g
        h_next = o * torch.tanh(c_next)
        return h_next, c_next


class ConvLSTMTemporalFusion(nn.Module):
    """Fuses a causally-ordered (oldest -> newest) sequence of per-frame feature maps into one
    feature map via a single-layer ConvLSTM, returning the final hidden state -- which has "seen"
    every frame in temporal order, unlike a crude channel-stack where the first conv layer must
    learn arbitrary cross-frame mixing from a flat blob with no explicit ordering prior."""

    def __init__(self, channels: int, kernel_size: int = 3) -> None:
        super().__init__()
        self.cell = ConvLSTMCell(channels, channels, kernel_size)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W), causal order (index 0 = oldest, index T-1 = newest/current)
        b, t, c, h_dim, w_dim = x.shape
        h = x.new_zeros(b, c, h_dim, w_dim)
        cell_state = x.new_zeros(b, c, h_dim, w_dim)
        for step in range(t):
            h, cell_state = self.cell(x[:, step], h, cell_state)
        return h


class AttentionTemporalFusion(nn.Module):
    """Alternative fusion: a learned scalar attention weight per frame (1x1 conv + global pool +
    linear scorer), softmax-normalized over the T frames, then a weighted sum of the per-frame
    feature maps. Permutation-invariant (does not use causal order in the weighting itself, unlike
    ConvLSTMTemporalFusion) -- the second fusion mechanism named in the round8 plan's 'temporal
    attention/ConvLSTM' queued idea, selectable via `model.temporal_fusion: attention`."""

    def __init__(self, channels: int) -> None:
        super().__init__()
        self.score = nn.Sequential(nn.Conv2d(channels, channels, kernel_size=1), nn.SiLU(inplace=True))
        self.score_proj = nn.Linear(channels, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        b, t, c, h_dim, w_dim = x.shape
        flat = x.reshape(b * t, c, h_dim, w_dim)
        scored = self.score(flat).mean(dim=(2, 3))  # (B*T, C) global-avg-pooled
        logits = self.score_proj(scored).view(b, t)
        weights = torch.softmax(logits, dim=1).view(b, t, 1, 1, 1)
        return (x * weights).sum(dim=1)


class TemporalFusionFactorizedUNet(nn.Module):
    """Proper temporal architecture for causal history (round8 plan 'Next generations' item),
    replacing exp063's crude channel-stack of predecessor rows (T-30/T-60/...) -- which failed its
    own fold0/4 gate at every history depth tried (cr2/cr3/cr4) -- with a shared per-frame encoder
    + explicit temporal fusion (ConvLSTM or attention, `model.temporal_fusion`) feeding the SAME
    factorized_mean_shape decoder/heads used by the champion lineage (exp056/exp064). Hypothesis:
    crude concatenation forces the first conv layer to learn arbitrary cross-frame-and-channel
    mixing from a flat (frames*channels) blob with no temporal structure prior; a shared encoder
    (frame-order-invariant weights) + an explicit sequence model should let the network learn "how
    the storm is moving/evolving" instead of just overfitting to incidental cross-frame
    correlations in the train locations.

    Design:
    - Each of the T = context_rows * max_observations frames (16 raw satellite channels + 1
      observed/padded mask channel + the 3-channel satellite one-hot, broadcast identically to
      every frame) is run independently through a SHARED 4-level encoder (enc1..enc4, identical
      ConvBlock design to FactorizedMeanShapeUNet) -- weight sharing across frames is the "shared
      per-frame encoder" this experiment is named for.
    - Temporal fusion happens ONLY at the bottleneck (smallest spatial size, cheapest to run a
      ConvLSTM/attention over): the T per-frame bottleneck-input feature maps are reordered into
      strict causal (oldest -> newest) order -- dataset.py's raw layout is newest-ROW-first (see
      `_context_rows`), so this is a row-group flip, not a per-observation reorder -- then fused
      into one feature map via `temporal_fusion`.
    - Skip connections (e1..e4) are taken from the CURRENT (newest, closest-to-target-time) frame
      only, exactly like the single-frame champion architecture's own skip connections: fine
      spatial detail comes from the most recent observation, while the fused bottleneck carries
      the abstract "trend from history" signal. This keeps the decoder's skip-connection channel
      dims identical to FactorizedMeanShapeUNet's, so dec4..dec1 and every head (rain_head/
      shape_head/mean_intensity_mlp/aux_mask_head) are copied verbatim, unmodified.
    - Input wire format is UNCHANGED from exp063's flat (B, in_channels, H, W) tensor -- dataset.py
      is not modified at all. This model reconstructs the per-frame view internally from the known
      channel layout (`expected_in_channels` in dataset.py), so train.py/inference.py/
      smoke_test.py all work unmodified.
    """

    def __init__(
        self,
        context_rows: int,
        max_observations: int,
        satellite_channels: int,
        base_channels: int = 32,
        internal_size: int = 128,
        output_size: tuple[int, int] = (41, 41),
        aux_mask_enabled: bool = True,
        mean_intensity_hidden: int = 64,
        shape_eps: float = 1e-6,
        amount_cap: float = 150.0,
        temporal_fusion: str = "convlstm",
        num_satellites: int = 3,
        bottleneck_dilations: tuple[int, ...] = (),
    ) -> None:
        super().__init__()
        if temporal_fusion not in ("convlstm", "attention"):
            raise ValueError(f"Unknown temporal_fusion: {temporal_fusion!r}")
        c = base_channels
        self.context_rows = int(context_rows)
        self.max_observations = int(max_observations)
        self.per_frame_channels = int(satellite_channels)
        self.num_satellites = int(num_satellites)
        self.num_frames = self.context_rows * self.max_observations
        self.frame_in_channels = self.per_frame_channels + 1 + self.num_satellites  # raw + mask + sat one-hot
        self.internal_size = int(internal_size)
        self.output_size = tuple(int(v) for v in output_size)
        self.aux_mask_enabled = bool(aux_mask_enabled)
        self.shape_eps = float(shape_eps)
        self.amount_cap = float(amount_cap)

        self.enc1 = ConvBlock(self.frame_in_channels, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.enc4 = ConvBlock(c * 4, c * 8)
        if temporal_fusion == "convlstm":
            self.temporal_fusion: nn.Module = ConvLSTMTemporalFusion(c * 8)
        else:
            self.temporal_fusion = AttentionTemporalFusion(c * 8)
        if bottleneck_dilations:
            self.bottleneck: nn.Module = nn.Sequential(
                *(ConvBlock(c * 8, c * 8, dilation=int(d)) for d in bottleneck_dilations)
            )
        else:
            self.bottleneck = ConvBlock(c * 8, c * 8)
        self.dec4 = ConvBlock(c * 8 + c * 8, c * 8)
        self.dec3 = ConvBlock(c * 8 + c * 4, c * 4)
        self.dec2 = ConvBlock(c * 4 + c * 2, c * 2)
        self.dec1 = ConvBlock(c * 2 + c, c)

        def head() -> nn.Sequential:
            return nn.Sequential(nn.Conv2d(c, c, 3, padding=1), nn.SiLU(inplace=True), nn.Conv2d(c, 1, 1))

        self.rain_head = head()
        self.shape_head = head()
        if self.aux_mask_enabled:
            self.aux_mask_head = head()
        self.mean_intensity_mlp = nn.Sequential(
            nn.Linear(c, mean_intensity_hidden),
            nn.SiLU(inplace=True),
            nn.Linear(mean_intensity_hidden, 1),
        )

    @staticmethod
    def _up(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

    def _split_frames(self, x: torch.Tensor) -> torch.Tensor:
        """Reconstructs the per-frame (B, T, frame_in_channels, H, W) view from dataset.py's flat
        layout: [T frames x per_frame_channels raw][T mask channels][satellite one-hot]. See
        dataset.py `_input_tensor`/`expected_in_channels` -- this architecture requires the
        strict-green feature set (no `channels_per_row` extras), asserted below."""
        b, total_c, h_dim, w_dim = x.shape
        t = self.num_frames
        pf = self.per_frame_channels
        raw_end = t * pf
        mask_end = raw_end + t
        sat_end = mask_end + self.num_satellites
        if sat_end != total_c:
            raise ValueError(
                f"unexpected input channel count {total_c}, expected {sat_end} "
                "(row-feature extras are not supported by this architecture)"
            )
        raw = x[:, :raw_end].reshape(b, t, pf, h_dim, w_dim)
        masks = x[:, raw_end:mask_end].reshape(b, t, 1, h_dim, w_dim)
        sat_onehot = x[:, mask_end:sat_end]  # (B, num_satellites, H, W), identical for every frame
        sat_onehot = sat_onehot.unsqueeze(1).expand(b, t, self.num_satellites, h_dim, w_dim)
        return torch.cat([raw, masks, sat_onehot], dim=2)

    def _causal_order(self, frames: torch.Tensor) -> torch.Tensor:
        """dataset.py lays frames out newest-ROW-first (see `_context_rows`): index groups of
        `max_observations` go [current row's obs][T-30 row's obs][T-60 row's obs]... Flipping the
        row-group axis gives strict oldest -> newest order; each row's own observations are
        already stored oldest -> newest internally, so only the row-group axis needs flipping."""
        b, t, c, h_dim, w_dim = frames.shape
        grouped = frames.view(b, self.context_rows, self.max_observations, c, h_dim, w_dim)
        grouped = grouped.flip(dims=[1])
        return grouped.reshape(b, t, c, h_dim, w_dim)

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        frames = self._split_frames(x)
        b, t, c_in, h_dim, w_dim = frames.shape
        frames = frames.reshape(b * t, c_in, h_dim, w_dim)
        frames = F.interpolate(
            frames, size=(self.internal_size, self.internal_size), mode="bilinear", align_corners=False
        )

        e1 = self.enc1(frames)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        e4 = self.enc4(F.avg_pool2d(e3, 2))
        bottleneck_in = F.avg_pool2d(e4, 2)

        def to_causal_seq(feat: torch.Tensor) -> torch.Tensor:
            _, fc, fh, fw = feat.shape
            return self._causal_order(feat.view(b, t, fc, fh, fw))

        bottleneck_seq = to_causal_seq(bottleneck_in)
        fused = self.temporal_fusion(bottleneck_seq)
        bottleneck = self.bottleneck(fused)

        # current (newest) frame's skip connections == last entry once reordered causally
        e1_cur = to_causal_seq(e1)[:, -1]
        e2_cur = to_causal_seq(e2)[:, -1]
        e3_cur = to_causal_seq(e3)[:, -1]
        e4_cur = to_causal_seq(e4)[:, -1]

        d4 = self.dec4(torch.cat([self._up(bottleneck, e4_cur), e4_cur], dim=1))
        d3 = self.dec3(torch.cat([self._up(d4, e3_cur), e3_cur], dim=1))
        d2 = self.dec2(torch.cat([self._up(d3, e2_cur), e2_cur], dim=1))
        return self.dec1(torch.cat([self._up(d2, e1_cur), e1_cur], dim=1))

    def _native(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(x, self.output_size)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        features = self.forward_features(x)
        rain_logits = self._native(self.rain_head(features))
        rain_prob = torch.sigmoid(rain_logits.float())

        shape_raw = F.softplus(self._native(self.shape_head(features)).float())
        rain_prob_detached = rain_prob.detach()
        shape_mean = (shape_raw * rain_prob_detached).sum(dim=(2, 3), keepdim=True) / (
            rain_prob_detached.sum(dim=(2, 3), keepdim=True) + self.shape_eps
        )
        shape = shape_raw / (shape_mean + self.shape_eps)

        pooled = features.float().mean(dim=(2, 3))
        mean_intensity = F.softplus(self.mean_intensity_mlp(pooled).float()).view(-1, 1, 1, 1)

        amount = (mean_intensity * shape).clamp(max=self.amount_cap)
        output = {
            "pred": rain_prob * amount,
            "rain_logits": rain_logits,
            "rain_prob": rain_prob,
            "shape": shape,
            "shape_raw": shape_raw,
            "mean_intensity": mean_intensity.view(-1),
            "rain_amount": amount,
        }
        if self.aux_mask_enabled:
            output["aux_mask_logits"] = self._native(self.aux_mask_head(features))
        return output


class SMPUNet(nn.Module):
    """Thin wrapper so callers use the same forward()/output shape as CompactUNet."""

    def __init__(
        self,
        in_channels: int = 54,
        encoder_name: str = "efficientnet-b0",
        encoder_weights: str | None = "imagenet",
        encoder_depth: int = 3,
        decoder_channels: tuple[int, ...] = (128, 64, 32),
    ) -> None:
        super().__init__()
        import segmentation_models_pytorch as smp

        self.net = smp.Unet(
            encoder_name=encoder_name,
            encoder_weights=encoder_weights,
            encoder_depth=encoder_depth,
            decoder_channels=decoder_channels,
            in_channels=in_channels,
            classes=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


def build_model(config: dict) -> nn.Module:
    model_cfg = config["model"]
    architecture = model_cfg.get("architecture", "compact_unet")
    in_channels = int(model_cfg["in_channels"])
    if architecture == "compact_unet":
        return CompactUNet(in_channels=in_channels, base_channels=int(model_cfg["base_channels"]))
    if architecture == "two_head_compact_unet":
        return TwoHeadCompactUNet(in_channels=in_channels, base_channels=int(model_cfg["base_channels"]))
    if architecture == "hurdle_lognormal_unet":
        return HurdleLogNormalUNet(
            in_channels=in_channels,
            base_channels=int(model_cfg["base_channels"]),
            sigma_mode=str(model_cfg.get("sigma_mode", "predicted")),
            fixed_sigma=float(model_cfg.get("fixed_sigma", 1.0)),
            mu_min=float(model_cfg.get("mu_min", -6.0)),
            mu_max=float(model_cfg.get("mu_max", 5.0)),
            sigma_min=float(model_cfg.get("sigma_min", 0.1)),
            sigma_max=float(model_cfg.get("sigma_max", 2.0)),
            amount_cap=float(model_cfg.get("amount_cap", 150.0)),
            serving_sigma_scale=float(model_cfg.get("serving_sigma_scale", 1.0)),
        )
    if architecture == "highres_hurdle_lognormal_unet":
        return HighResHurdleLogNormalUNet(
            in_channels=in_channels,
            base_channels=int(model_cfg.get("base_channels", 32)),
            internal_size=int(model_cfg.get("internal_size", 128)),
            output_size=(int(config["data"]["target_height"]), int(config["data"]["target_width"])),
            aux_mask_enabled=bool(model_cfg.get("aux_mask_enabled", True)),
            sigma_mode=str(model_cfg.get("sigma_mode", "predicted")),
            fixed_sigma=float(model_cfg.get("fixed_sigma", 1.0)),
            mu_min=float(model_cfg.get("mu_min", -6.0)),
            mu_max=float(model_cfg.get("mu_max", 5.0)),
            sigma_min=float(model_cfg.get("sigma_min", 0.1)),
            sigma_max=float(model_cfg.get("sigma_max", 2.0)),
            amount_cap=float(model_cfg.get("amount_cap", 150.0)),
            serving_sigma_scale=float(model_cfg.get("serving_sigma_scale", 1.0)),
            bottleneck_dilations=tuple(model_cfg.get("bottleneck_dilations", []) or []),
        )
    if architecture == "factorized_mean_shape":
        return FactorizedMeanShapeUNet(
            in_channels=in_channels,
            base_channels=int(model_cfg.get("base_channels", 32)),
            internal_size=int(model_cfg.get("internal_size", 128)),
            output_size=(int(config["data"]["target_height"]), int(config["data"]["target_width"])),
            aux_mask_enabled=bool(model_cfg.get("aux_mask_enabled", True)),
            mean_intensity_hidden=int(model_cfg.get("mean_intensity_hidden", 64)),
            shape_eps=float(model_cfg.get("shape_eps", 1e-6)),
            amount_cap=float(model_cfg.get("amount_cap", 150.0)),
            bottleneck_dilations=tuple(model_cfg.get("bottleneck_dilations", []) or []),
        )
    if architecture == "temporal_fusion_factorized":
        return TemporalFusionFactorizedUNet(
            context_rows=int(config["data"].get("context_rows", 1)),
            max_observations=int(config["data"]["max_observations"]),
            satellite_channels=int(config["data"]["satellite_channels"]),
            base_channels=int(model_cfg.get("base_channels", 32)),
            internal_size=int(model_cfg.get("internal_size", 128)),
            output_size=(int(config["data"]["target_height"]), int(config["data"]["target_width"])),
            aux_mask_enabled=bool(model_cfg.get("aux_mask_enabled", True)),
            mean_intensity_hidden=int(model_cfg.get("mean_intensity_hidden", 64)),
            shape_eps=float(model_cfg.get("shape_eps", 1e-6)),
            amount_cap=float(model_cfg.get("amount_cap", 150.0)),
            temporal_fusion=str(model_cfg.get("temporal_fusion", "convlstm")),
            bottleneck_dilations=tuple(model_cfg.get("bottleneck_dilations", []) or []),
        )
    if architecture == "smp_unet":
        return SMPUNet(
            in_channels=in_channels,
            encoder_name=model_cfg.get("encoder_name", "efficientnet-b0"),
            encoder_weights=model_cfg.get("encoder_weights", "imagenet"),
            encoder_depth=int(model_cfg.get("encoder_depth", 3)),
            decoder_channels=tuple(model_cfg.get("decoder_channels", [128, 64, 32])),
        )
    raise ValueError(f"Unknown model.architecture: {architecture!r}")


def prediction_from_output(output: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(output, dict):
        return output["pred"]
    return output
