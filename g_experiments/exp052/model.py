"""Model architectures for exp052: exp018's high-resolution hurdle log-normal U-Net plus an
optional dilated bottleneck ported from exp030 (`model.bottleneck_dilations`), plus a NEW
optional train-time-only auxiliary future-frame decoder head (`model.future_aux_head`,
ticket G-033 reinterpreted per the 2026-07-20 organizer ruling -- see README.md).

The future_aux branch taps the same bottleneck feature map `forward_features` already
computes internally (before the decoder's upsampling path) and predicts a (B,1,H,W) IR-window
frame at native target resolution. It is purely an auxiliary output: `prediction_from_output`
below is UNCHANGED and never reads it, so it has zero influence on the served prediction
whether or not the flag is enabled (verified in smoke_test.py).

Carried over from exp016/exp018 unchanged as control arms:
- "compact_unet", "two_head_compact_unet", "hurdle_lognormal_unet", "smp_unet".
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
        future_aux_head: bool = False,
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
        self.future_aux_head = bool(future_aux_head)

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
        if self.future_aux_head:
            # NEW (exp052, G-033 reinterpreted): small decoder tapping the bottleneck feature
            # map `forward_features` computes internally, BEFORE the decoder's upsampling path
            # -- i.e. the coarsest, most compressed representation, at internal_size/16
            # resolution (8x8 for internal_size=128). 2x [Conv3x3+ReLU] + 1x1 conv per the
            # ticket's spec; bilinear-upsampled to native output resolution in forward(). This
            # module is TRAIN-TIME-ONLY: `prediction_from_output` never reads its output, and
            # inference.py/make_submission.py hard-force future_aux_head=False when serving.
            self.future_aux_decoder = nn.Sequential(
                nn.Conv2d(c * 8, c * 8, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(c * 8, c * 8, kernel_size=3, padding=1),
                nn.ReLU(inplace=True),
                nn.Conv2d(c * 8, 1, kernel_size=1),
            )

    @staticmethod
    def _up(x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        return F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)

    def forward_features(
        self, x: torch.Tensor, return_bottleneck: bool = False
    ) -> torch.Tensor | tuple[torch.Tensor, torch.Tensor]:
        x = F.interpolate(x, size=(self.internal_size, self.internal_size), mode="bilinear", align_corners=False)
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, 2))
        e3 = self.enc3(F.avg_pool2d(e2, 2))
        e4 = self.enc4(F.avg_pool2d(e3, 2))
        b = self.bottleneck(F.avg_pool2d(e4, 2))
        d4 = self.dec4(torch.cat([self._up(b, e4), e4], dim=1))
        d3 = self.dec3(torch.cat([self._up(d4, e3), e3], dim=1))
        d2 = self.dec2(torch.cat([self._up(d3, e2), e2], dim=1))
        d1 = self.dec1(torch.cat([self._up(d2, e1), e1], dim=1))
        if return_bottleneck:
            return d1, b
        return d1

    def _native(self, x: torch.Tensor) -> torch.Tensor:
        return F.adaptive_avg_pool2d(x, self.output_size)

    def forward(self, x: torch.Tensor) -> dict[str, torch.Tensor]:
        if self.future_aux_head:
            features, bottleneck = self.forward_features(x, return_bottleneck=True)
        else:
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
        if self.future_aux_head:
            # Bilinear-upsampled straight from the coarse bottleneck resolution to native
            # target resolution (not `_native`'s adaptive_avg_pool2d, which assumes a
            # downsampling source) -- this branch is train-time-only advisory supervision, not
            # the served metric, so the "never interpolate the native grid" invariant that
            # matters for pred/aux_mask_logits does not apply here.
            aux_features = self.future_aux_decoder(bottleneck)
            output["future_aux"] = F.interpolate(
                aux_features, size=self.output_size, mode="bilinear", align_corners=False
            )
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
            future_aux_head=bool(model_cfg.get("future_aux_head", False)),
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
