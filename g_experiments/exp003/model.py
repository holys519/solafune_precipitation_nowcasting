"""Model architectures for exp003.

Two options, selected via config `model.architecture`:
- "compact_unet": same from-scratch CNN as exp001, carried over unchanged as the control arm.
- "smp_unet": segmentation_models_pytorch U-Net with an ImageNet-pretrained encoder, adapted to
  our N-channel (non-RGB) input via smp's in_channels handling. encoder_depth defaults to 3
  (not the usual 5) because inputs here are only 41x41 -- a 5-stage encoder would downsample
  past the point of being useful on a grid this small.

See doc/task_tickets.md ticket L-002 for the planned A/B between the two.
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

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        e1 = self.enc1(x)
        e2 = self.enc2(F.avg_pool2d(e1, kernel_size=2, ceil_mode=True))
        e3 = self.enc3(F.avg_pool2d(e2, kernel_size=2, ceil_mode=True))
        b = self.bottleneck(e3)
        d2 = F.interpolate(b, size=e2.shape[-2:], mode="bilinear", align_corners=False)
        d2 = self.dec2(torch.cat([d2, e2], dim=1))
        d1 = F.interpolate(d2, size=e1.shape[-2:], mode="bilinear", align_corners=False)
        d1 = self.dec1(torch.cat([d1, e1], dim=1))
        return self.head(d1)


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
    if architecture == "smp_unet":
        return SMPUNet(
            in_channels=in_channels,
            encoder_name=model_cfg.get("encoder_name", "efficientnet-b0"),
            encoder_weights=model_cfg.get("encoder_weights", "imagenet"),
            encoder_depth=int(model_cfg.get("encoder_depth", 3)),
            decoder_channels=tuple(model_cfg.get("decoder_channels", [128, 64, 32])),
        )
    raise ValueError(f"Unknown model.architecture: {architecture!r}")
