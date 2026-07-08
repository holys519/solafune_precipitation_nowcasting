"""Losses for exp003."""

from __future__ import annotations

import torch
from torch import nn


class WeightedMSELoss(nn.Module):
    """MSE with extra weight on pixels where the target precipitation is positive.

    Plain MSE is dominated by the ~82% exact-zero pixels (see doc/exp001_retrospective.md);
    this up-weights the rain pixels that actually drive the error we care about without
    changing the target's scale (unlike a log1p transform), so the loss stays aligned with
    the RMSE evaluation metric.
    """

    def __init__(self, pos_weight: float = 2.0) -> None:
        super().__init__()
        self.pos_weight = pos_weight

    def forward(self, pred: torch.Tensor, target: torch.Tensor) -> torch.Tensor:
        weight = 1.0 + self.pos_weight * (target > 0).float()
        return torch.mean(weight * torch.square(pred - target))
