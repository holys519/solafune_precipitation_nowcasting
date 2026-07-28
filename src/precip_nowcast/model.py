"""High-resolution, identifiable amount-distribution nowcaster."""

from __future__ import annotations

from dataclasses import dataclass
from math import gcd

import torch
import torch.nn.functional as F
from torch import nn


@dataclass
class PrecipitationOutput:
    """Model output with an identifiable physical decomposition.

    ``spatial_distribution`` sums to one for every tile. Consequently
    ``prediction.sum() == tile_total`` up to floating-point error; there is no
    second rain-probability gate that can silently dilute the predicted amount.
    """

    prediction: torch.Tensor
    tile_total: torch.Tensor
    spatial_distribution: torch.Tensor
    occurrence_logits: torch.Tensor


class ConvNormAct(nn.Sequential):
    def __init__(self, in_channels: int, out_channels: int, stride: int = 1) -> None:
        groups = gcd(8, out_channels)
        super().__init__(
            nn.Conv2d(in_channels, out_channels, 3, stride=stride, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
            nn.Conv2d(out_channels, out_channels, 3, padding=1, bias=False),
            nn.GroupNorm(groups, out_channels),
            nn.SiLU(inplace=True),
        )


class ResidualDown(nn.Module):
    def __init__(self, in_channels: int, out_channels: int) -> None:
        super().__init__()
        self.body = ConvNormAct(in_channels, out_channels, stride=2)
        self.skip = nn.Conv2d(in_channels, out_channels, 1, stride=2, bias=False)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.body(x) + self.skip(x)


class UpBlock(nn.Module):
    def __init__(self, in_channels: int, skip_channels: int, out_channels: int) -> None:
        super().__init__()
        self.block = ConvNormAct(in_channels + skip_channels, out_channels)

    def forward(self, x: torch.Tensor, skip: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=skip.shape[-2:], mode="bilinear", align_corners=False)
        return self.block(torch.cat((x, skip), dim=1))


class TotalShapeNowcaster(nn.Module):
    """Preserve native satellite detail and separate amount from spatial allocation."""

    def __init__(
        self,
        in_channels: int = 54,
        base_channels: int = 32,
        total_hidden: int = 128,
        dropout: float = 0.1,
        output_size: tuple[int, int] = (41, 41),
    ) -> None:
        super().__init__()
        c = base_channels
        self.output_size = output_size
        self.stem = ConvNormAct(in_channels, c)
        self.down1 = ResidualDown(c, 2 * c)
        self.down2 = ResidualDown(2 * c, 4 * c)
        self.down3 = ResidualDown(4 * c, 8 * c)
        self.down4 = ResidualDown(8 * c, 8 * c)
        self.up3 = UpBlock(8 * c, 8 * c, 8 * c)
        self.up2 = UpBlock(8 * c, 4 * c, 4 * c)
        self.up1 = UpBlock(4 * c, 2 * c, 2 * c)
        self.up0 = UpBlock(2 * c, c, c)
        self.spatial_head = nn.Conv2d(c, 1, 1)
        self.occurrence_head = nn.Conv2d(c, 1, 1)
        self.total_head = nn.Sequential(
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(8 * c, total_hidden),
            nn.SiLU(inplace=True),
            nn.Dropout(dropout),
            nn.Linear(total_hidden, 1),
        )

    def forward(self, x: torch.Tensor) -> PrecipitationOutput:
        e0 = self.stem(x)
        e1 = self.down1(e0)
        e2 = self.down2(e1)
        e3 = self.down3(e2)
        bottleneck = self.down4(e3)
        d3 = self.up3(bottleneck, e3)
        d2 = self.up2(d3, e2)
        d1 = self.up1(d2, e1)
        d0 = self.up0(d1, e0)

        spatial_logits = F.adaptive_avg_pool2d(self.spatial_head(d0), self.output_size)
        occurrence_logits = F.adaptive_avg_pool2d(self.occurrence_head(d0), self.output_size)
        batch = spatial_logits.shape[0]
        distribution = torch.softmax(spatial_logits.flatten(1), dim=1).view_as(spatial_logits)
        tile_total = F.softplus(self.total_head(bottleneck).float()).view(batch, 1, 1, 1)
        prediction = tile_total * distribution
        return PrecipitationOutput(
            prediction=prediction,
            tile_total=tile_total.view(batch),
            spatial_distribution=distribution,
            occurrence_logits=occurrence_logits,
        )
