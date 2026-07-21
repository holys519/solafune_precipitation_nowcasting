"""Losses for exp038 (copied from exp018): hurdle likelihood plus spatial auxiliary objectives."""

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


class HurdleLogNormalLoss(nn.Module):
    """Hurdle (zero-inflated log-normal) loss — ticket G-030, doc/discussion_insights.md §2.

    Two terms, both deliberately UNWEIGHTED (tail/class re-weighting was measured net-negative
    on this dataset by the discussion authors):
    - occurrence: plain BCE on rain>threshold. No pos_weight — weighted BCE mis-calibrates
      P(rain), and calibration is exactly what makes P(rain)*E[Y|rain] an unbiased mean.
    - intensity: fitted ONLY on wet pixels, on ln(y). With sigma_mode=predicted this is the
      Gaussian NLL of ln(y) under N(mu, sigma^2); with fixed sigma it reduces to MSE on ln(y).
      Wet-only masking is the point: the branch never sees the ~82% zero mass, so it cannot be
      dragged toward zero (the regression-to-the-mean failure of single-head L2 models here).

    prediction_weight (default 0) optionally adds MSE between the served product and the raw
    target — kept configurable as a hedge, but nonzero values re-introduce the zero-mass drag
    on the intensity branch through the product; the clean hurdle design leaves it at 0.
    """

    def __init__(
        self,
        rain_threshold: float = 0.0,
        bce_weight: float = 1.0,
        intensity_weight: float = 1.0,
        prediction_weight: float = 0.0,
        min_wet_target: float = 0.01,
        aux_mask_weight: float = 0.0,
        aux_mask_threshold: float = 0.25,
        aux_dice_weight: float = 0.5,
        multiscale_weight_2: float = 0.0,
        multiscale_weight_4: float = 0.0,
        tile_mean_weight: float = 0.0,
        metric_weight: float = 0.0,
        metric_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.rain_threshold = float(rain_threshold)
        self.bce_weight = float(bce_weight)
        self.intensity_weight = float(intensity_weight)
        self.prediction_weight = float(prediction_weight)
        self.min_wet_target = float(min_wet_target)
        self.aux_mask_weight = float(aux_mask_weight)
        self.aux_mask_threshold = float(aux_mask_threshold)
        self.aux_dice_weight = float(aux_dice_weight)
        self.multiscale_weight_2 = float(multiscale_weight_2)
        self.multiscale_weight_4 = float(multiscale_weight_4)
        self.tile_mean_weight = float(tile_mean_weight)
        self.metric_weight = float(metric_weight)
        self.metric_eps = float(metric_eps)

    def forward(self, output: dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        if not isinstance(output, dict) or "mu" not in output:
            raise TypeError("hurdle_lognormal loss requires the hurdle_lognormal_unet output dict")
        target = target.float()
        rain_target = (target > self.rain_threshold).float()
        bce = F.binary_cross_entropy_with_logits(output["rain_logits"].float(), rain_target)

        wet = target > self.rain_threshold
        if wet.any():
            ln_y = torch.log(target.clamp_min(self.min_wet_target))
            mu = output["mu"].float()
            sigma = output["sigma"].float()
            z2 = torch.square((ln_y - mu) / sigma)
            nll = 0.5 * z2 + torch.log(sigma)
            intensity = nll[wet].mean()
        else:
            intensity = torch.zeros((), device=target.device)

        total = self.bce_weight * bce + self.intensity_weight * intensity
        if self.prediction_weight > 0:
            total = total + self.prediction_weight * torch.mean(
                torch.square(output["pred"].float() - target)
            )
        if self.aux_mask_weight > 0:
            if "aux_mask_logits" not in output:
                raise KeyError("aux_mask_weight > 0 requires model output['aux_mask_logits']")
            mask_target = (target >= self.aux_mask_threshold).float()
            logits = output["aux_mask_logits"].float()
            bce_mask = F.binary_cross_entropy_with_logits(logits, mask_target)
            prob = torch.sigmoid(logits)
            dims = tuple(range(1, prob.ndim))
            intersection = (prob * mask_target).sum(dim=dims)
            dice = 1.0 - ((2.0 * intersection + 1.0) / (prob.sum(dim=dims) + mask_target.sum(dim=dims) + 1.0)).mean()
            mask_loss = (1.0 - self.aux_dice_weight) * bce_mask + self.aux_dice_weight * dice
            total = total + self.aux_mask_weight * mask_loss
        for factor, weight in ((2, self.multiscale_weight_2), (4, self.multiscale_weight_4)):
            if weight > 0:
                pooled_pred = F.avg_pool2d(output["pred"].float(), factor, factor)
                pooled_target = F.avg_pool2d(target, factor, factor)
                total = total + weight * F.mse_loss(pooled_pred, pooled_target)
        if self.tile_mean_weight > 0:
            # E-1 (g_eda/exp002): the dominant residual is per-tile amount error — amount_swap
            # (perfect tile mean, our placement) scores 0.545 vs actual 0.609 on exp018's OOF.
            # This term supervises the tile mean directly, the coarsest rung of the multi-scale
            # ladder that multiscale_weight_2/4 leave uncovered.
            pred_mean = output["pred"].float().mean(dim=(1, 2, 3))
            target_mean = target.mean(dim=(1, 2, 3))
            total = total + self.tile_mean_weight * F.mse_loss(pred_mean, target_mean)
        if self.metric_weight > 0:
            # The identified evaluator averages per-file RMSE, rather than taking one global
            # pooled RMSE. This smooth epsilon keeps the derivative finite on exact matches.
            diff2 = torch.square(output["pred"].float() - target)
            tile_rmse = torch.sqrt(diff2.mean(dim=(1, 2, 3)) + self.metric_eps)
            total = total + self.metric_weight * tile_rmse.mean()
        return total


def build_loss(config: dict) -> nn.Module:
    loss_cfg = config.get("loss", {})
    name = loss_cfg.get("name", "weighted_mse")
    if name == "hurdle_lognormal":
        return HurdleLogNormalLoss(
            rain_threshold=float(loss_cfg.get("rain_threshold", 0.0)),
            bce_weight=float(loss_cfg.get("bce_weight", 1.0)),
            intensity_weight=float(loss_cfg.get("intensity_weight", 1.0)),
            prediction_weight=float(loss_cfg.get("prediction_weight", 0.0)),
            min_wet_target=float(loss_cfg.get("min_wet_target", 0.01)),
            aux_mask_weight=float(loss_cfg.get("aux_mask_weight", 0.0)),
            aux_mask_threshold=float(loss_cfg.get("aux_mask_threshold", 0.25)),
            aux_dice_weight=float(loss_cfg.get("aux_dice_weight", 0.5)),
            multiscale_weight_2=float(loss_cfg.get("multiscale_weight_2", 0.0)),
            multiscale_weight_4=float(loss_cfg.get("multiscale_weight_4", 0.0)),
            tile_mean_weight=float(loss_cfg.get("tile_mean_weight", 0.0)),
            metric_weight=float(loss_cfg.get("metric_weight", 0.0)),
            metric_eps=float(loss_cfg.get("metric_eps", 1e-6)),
        )
    if name == "two_head_rain":
        return TwoHeadRainLoss(
            rain_threshold=float(loss_cfg.get("rain_threshold", 0.0)),
            pos_weight=float(loss_cfg.get("pos_weight", 2.0)),
            bce_weight=float(loss_cfg.get("bce_weight", 0.1)),
            bce_pos_weight=float(loss_cfg.get("bce_pos_weight", 3.0)),
            amount_weight=float(loss_cfg.get("amount_weight", 0.5)),
            prediction_weight=float(loss_cfg.get("prediction_weight", 1.0)),
        )
    return WeightedMSELoss(pos_weight=float(loss_cfg.get("pos_weight", 2.0)))
