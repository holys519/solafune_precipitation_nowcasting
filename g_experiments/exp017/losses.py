"""Losses for exp017 (copied from exp009; unchanged -- this experiment changes only the input pipeline)."""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def prediction_from_output(output: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(output, dict):
        return output["pred"]
    return output


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

    def forward(self, pred: torch.Tensor | dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        pred = prediction_from_output(pred)
        weight = 1.0 + self.pos_weight * (target > 0).float()
        return torch.mean(weight * torch.square(pred - target))


class TwoHeadRainLoss(nn.Module):
    """Joint loss for rain/no-rain detection and quantitative precipitation estimation."""

    def __init__(
        self,
        rain_threshold: float = 0.0,
        pos_weight: float = 2.0,
        bce_weight: float = 0.1,
        bce_pos_weight: float = 3.0,
        amount_weight: float = 0.5,
        prediction_weight: float = 1.0,
    ) -> None:
        super().__init__()
        self.rain_threshold = rain_threshold
        self.pos_weight = pos_weight
        self.bce_weight = bce_weight
        self.register_buffer("bce_pos_weight", torch.tensor(float(bce_pos_weight)))
        self.amount_weight = amount_weight
        self.prediction_weight = prediction_weight

    def forward(self, output: torch.Tensor | dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        if not isinstance(output, dict):
            weight = 1.0 + self.pos_weight * (target > self.rain_threshold).float()
            return torch.mean(weight * torch.square(output - target))

        rain_target = (target > self.rain_threshold).float()
        bce = F.binary_cross_entropy_with_logits(
            output["rain_logits"],
            rain_target,
            pos_weight=self.bce_pos_weight.to(output["rain_logits"].device),
        )
        weight = 1.0 + self.pos_weight * rain_target
        amount_loss = torch.mean(weight * torch.square(output["rain_amount"] - target))
        prediction_loss = torch.mean(weight * torch.square(output["pred"] - target))
        return self.bce_weight * bce + self.amount_weight * amount_loss + self.prediction_weight * prediction_loss


class FocalTverskyRainLoss(TwoHeadRainLoss):
    """Control regression losses plus calibrated, low-weight spatial occurrence auxiliaries."""

    def __init__(self, focal_weight: float = 0.05, tversky_weight: float = 0.05,
                 focal_gamma: float = 2.0, tversky_alpha: float = 0.3,
                 tversky_beta: float = 0.7, **kwargs) -> None:
        super().__init__(**kwargs)
        self.focal_weight, self.tversky_weight = focal_weight, tversky_weight
        self.focal_gamma, self.tversky_alpha, self.tversky_beta = focal_gamma, tversky_alpha, tversky_beta

    def forward(self, output, target):
        base = super().forward(output, target)
        if not isinstance(output, dict):
            return base
        truth = (target > self.rain_threshold).float()
        prob = output["rain_prob"].clamp(1e-5, 1.0 - 1e-5)
        pt = torch.where(truth > 0, prob, 1.0 - prob)
        focal = (-(1.0 - pt).pow(self.focal_gamma) * pt.log()).mean()
        dims = (1, 2, 3)
        tp = (prob * truth).sum(dims)
        fp = (prob * (1.0 - truth)).sum(dims)
        fn = ((1.0 - prob) * truth).sum(dims)
        tversky = 1.0 - ((tp + 1.0) / (tp + self.tversky_alpha * fp + self.tversky_beta * fn + 1.0)).mean()
        return base + self.focal_weight * focal + self.tversky_weight * tversky


def build_loss(config: dict) -> nn.Module:
    loss_cfg = config.get("loss", {})
    name = loss_cfg.get("name", "weighted_mse")
    if name in {"two_head_rain", "focal_tversky_rain"}:
        kwargs = dict(
            rain_threshold=float(loss_cfg.get("rain_threshold", 0.0)),
            pos_weight=float(loss_cfg.get("pos_weight", 2.0)),
            bce_weight=float(loss_cfg.get("bce_weight", 0.1)),
            bce_pos_weight=float(loss_cfg.get("bce_pos_weight", 3.0)),
            amount_weight=float(loss_cfg.get("amount_weight", 0.5)),
            prediction_weight=float(loss_cfg.get("prediction_weight", 1.0)),
        )
        if name == "focal_tversky_rain":
            return FocalTverskyRainLoss(
                focal_weight=float(loss_cfg.get("focal_weight", 0.05)),
                tversky_weight=float(loss_cfg.get("tversky_weight", 0.05)),
                focal_gamma=float(loss_cfg.get("focal_gamma", 2.0)),
                tversky_alpha=float(loss_cfg.get("tversky_alpha", 0.3)),
                tversky_beta=float(loss_cfg.get("tversky_beta", 0.7)), **kwargs)
        return TwoHeadRainLoss(**kwargs)
    return WeightedMSELoss(pos_weight=float(loss_cfg.get("pos_weight", 2.0)))
