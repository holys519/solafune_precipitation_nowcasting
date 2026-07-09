"""Model architectures for exp013 (ticket G-016: exp009 successor rows x exp011 adapter).

Options, selected via config `model.architecture`:
- "compact_unet": same from-scratch CNN as exp001, carried over unchanged as the control arm.
- "two_head_compact_unet": shared Compact U-Net body with rain/no-rain and rain-amount heads
  (exp009's winning arm, kept selectable for same-fold A/B against the adapter).
- "satellite_adapter_two_head_unet": exp011's satellite-specific input stems in front of the
  shared body + two heads. The stems take the full input (105ch here vs 54ch in exp011); the
  satellite one-hot maps stay the last 3 channels regardless of context_rows, so the adapter
  selection logic carries over unchanged.
- "smp_unet": segmentation_models_pytorch U-Net with an ImageNet-pretrained encoder, adapted to
  our N-channel (non-RGB) input via smp's in_channels handling. encoder_depth defaults to 3
  (not the usual 5) because inputs here are only 41x41 -- a 5-stage encoder would downsample
  past the point of being useful on a grid this small.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        groups = 8 if out_channels % 8 == 0 else 1
        self.block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, padding=1, bias=False),
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


class SatelliteAdapterTwoHeadUNet(nn.Module):
    """Satellite-specific input stems with shared U-Net body and two precipitation heads.

    Carried over from exp011 unchanged: each stem sees the whole input tensor (including the
    successor-row slots added by exp009's dataset), and the output of the stem matching the
    row's satellite (selected via the one-hot maps in the last 3 channels) is what flows into
    the shared body.
    """

    def __init__(self, in_channels: int = 54, base_channels: int = 48) -> None:
        super().__init__()
        c = base_channels
        self.stems = nn.ModuleList([ConvBlock(in_channels, c) for _ in range(3)])
        self.fuse_refine = ConvBlock(c, c)
        self.enc2 = ConvBlock(c, c * 2)
        self.enc3 = ConvBlock(c * 2, c * 4)
        self.bottleneck = ConvBlock(c * 4, c * 4)
        self.dec2 = ConvBlock(c * 4 + c * 2, c * 2)
        self.dec1 = ConvBlock(c * 2 + c, c)
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

    def forward_features(self, x: torch.Tensor) -> torch.Tensor:
        if x.shape[1] < 3:
            raise ValueError("satellite_adapter_two_head_unet expects satellite one-hot maps in the last 3 channels")
        satellite_maps = x[:, -3:]
        weights = satellite_maps[:, :, :1, :1].unsqueeze(2)
        stem_outputs = torch.stack([stem(x) for stem in self.stems], dim=1)
        e1 = (stem_outputs * weights).sum(dim=1)
        e1 = self.fuse_refine(e1)
        e2 = self.enc2(F.avg_pool2d(e1, kernel_size=2, ceil_mode=True))
        e3 = self.enc3(F.avg_pool2d(e2, kernel_size=2, ceil_mode=True))
        b = self.bottleneck(e3)
        d2 = F.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        return self.dec1(torch.cat([d1, e1], dim=1))

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
    if architecture == "satellite_adapter_two_head_unet":
        return SatelliteAdapterTwoHeadUNet(in_channels=in_channels, base_channels=int(model_cfg["base_channels"]))
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
