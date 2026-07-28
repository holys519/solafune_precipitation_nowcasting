"""Objectives aligned with the served prediction and official metric."""

from __future__ import annotations

import torch
import torch.nn.functional as F
from torch import nn

from .config import LossConfig
from .model import PrecipitationOutput


class CompositePrecipitationLoss(nn.Module):
    def __init__(self, config: LossConfig) -> None:
        super().__init__()
        self.config = config

    def forward(
        self, output: PrecipitationOutput, target: torch.Tensor
    ) -> tuple[torch.Tensor, dict[str, torch.Tensor]]:
        target = target.float()
        pred = output.prediction.float()
        target_total = target.flatten(1).sum(dim=1)
        target_distribution = target / target_total.clamp_min(1e-6).view(-1, 1, 1, 1)
        wet_tiles = target_total > 0

        field = F.huber_loss(pred, target, delta=self.config.huber_delta)
        total = F.huber_loss(
            output.tile_total,
            target_total,
            delta=self.config.huber_delta * target.shape[-1] * target.shape[-2],
        ) / target[0].numel()
        if wet_tiles.any():
            distribution = F.kl_div(
                output.spatial_distribution[wet_tiles].clamp_min(1e-8).log(),
                target_distribution[wet_tiles],
                reduction="batchmean",
            )
        else:
            distribution = pred.new_zeros(())
        occurrence_target = (target > self.config.wet_threshold).float()
        occurrence = F.binary_cross_entropy_with_logits(
            output.occurrence_logits.float(), occurrence_target
        )
        pooled_pred = F.avg_pool2d(pred, 2, 2)
        pooled_target = F.avg_pool2d(target, 2, 2)
        multiscale = F.mse_loss(pooled_pred, pooled_target)

        components = {
            "field_huber": field,
            "total_huber": total,
            "distribution_kl": distribution,
            "occurrence_bce": occurrence,
            "multiscale_mse": multiscale,
        }
        loss = (
            self.config.field_huber * field
            + self.config.total_huber * total
            + self.config.distribution * distribution
            + self.config.occurrence_bce * occurrence
            + self.config.multiscale_mse * multiscale
        )
        return loss, components
