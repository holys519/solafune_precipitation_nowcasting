"""Losses for exp040: tile mean-intensity x normalized-shape factorization
(survey v3 P0, `doc/research_survey_v3_2026-07-16.md` §5.1).

MeanShapeLoss supervises the two components of `highres_mean_shape_unet`'s output
separately, plus optional auxiliary/regularizing terms:

- mean_weight: MSE(m, tile_target_mean) — the UNCONDITIONAL tile mean (all pixels,
  including zeros). This is the direct fix for the dominant OOF residual identified by
  g_eda/exp006 (true-mean rescaling recovers 86-88% of the L2-scale oracle).
- shape_weight: normalized-field MSE between the model's `shape` (mean_pixels==1) and the
  target's own normalized field (mean_pixels==1), computed only on tiles with a nonzero
  target mean (shape is undefined on all-dry tiles, matching survey 5.1's caveat).
- bce_weight: optional UNWEIGHTED wet BCE on an auxiliary rain_logits head (diagnostic /
  regularizer only — not multiplied into the served prediction, unlike the hurdle heads).
- aux_mask_weight/aux_dice_weight: same wet-mask auxiliary segmentation term as exp018/038.
- multiscale_weight_2/4: pooled-scale MSE on the served field (exp018/038 pattern).
- field_weight: optional plain MSE(pred, target) — a hedge; nonzero values reintroduce the
  zero-mass drag the factorization is designed to avoid, so default 0.
- metric_weight: the "candidate metric loss" from survey §4.3/§7 Phase 1 point 2 —
  mean_batch(sqrt(mean_pixels(error^2) + eps)), i.e. the tile-RMSE-shaped loss that matches
  the identified per-file-averaged evaluator (l_eda/exp003) more directly than plain MSE.
  This is Arm D on top of Arm C's mean/shape supervision.
"""

from __future__ import annotations

import torch
from torch import nn
from torch.nn import functional as F


def prediction_from_output(output: torch.Tensor | dict[str, torch.Tensor]) -> torch.Tensor:
    if isinstance(output, dict):
        return output["pred"]
    return output


class MeanShapeLoss(nn.Module):
    def __init__(
        self,
        rain_threshold: float = 0.0,
        mean_weight: float = 1.0,
        shape_weight: float = 1.0,
        min_wet_target: float = 0.01,
        bce_weight: float = 0.0,
        aux_mask_weight: float = 0.0,
        aux_mask_threshold: float = 0.25,
        aux_dice_weight: float = 0.5,
        multiscale_weight_2: float = 0.0,
        multiscale_weight_4: float = 0.0,
        field_weight: float = 0.0,
        metric_weight: float = 0.0,
        metric_eps: float = 1e-6,
    ) -> None:
        super().__init__()
        self.rain_threshold = float(rain_threshold)
        self.mean_weight = float(mean_weight)
        self.shape_weight = float(shape_weight)
        self.min_wet_target = float(min_wet_target)
        self.bce_weight = float(bce_weight)
        self.aux_mask_weight = float(aux_mask_weight)
        self.aux_mask_threshold = float(aux_mask_threshold)
        self.aux_dice_weight = float(aux_dice_weight)
        self.multiscale_weight_2 = float(multiscale_weight_2)
        self.multiscale_weight_4 = float(multiscale_weight_4)
        self.field_weight = float(field_weight)
        self.metric_weight = float(metric_weight)
        self.metric_eps = float(metric_eps)

    def forward(self, output: dict[str, torch.Tensor], target: torch.Tensor) -> torch.Tensor:
        if not isinstance(output, dict) or "shape" not in output or "mean_intensity" not in output:
            raise TypeError("mean_shape loss requires the highres_mean_shape_unet output dict")
        target = target.float()
        target_mean = target.mean(dim=(1, 2, 3))

        total = target.new_zeros(())
        if self.mean_weight > 0:
            total = total + self.mean_weight * F.mse_loss(output["mean_intensity"].float(), target_mean)

        if self.shape_weight > 0:
            wet = target_mean > self.min_wet_target
            if wet.any():
                target_shape = target[wet] / target_mean[wet].clamp_min(self.min_wet_target).view(-1, 1, 1, 1)
                shape_loss = F.mse_loss(output["shape"][wet].float(), target_shape)
                total = total + self.shape_weight * shape_loss

        if self.bce_weight > 0:
            if "rain_logits" not in output:
                raise KeyError("bce_weight > 0 requires model output['rain_logits']")
            rain_target = (target > self.rain_threshold).float()
            bce = F.binary_cross_entropy_with_logits(output["rain_logits"].float(), rain_target)
            total = total + self.bce_weight * bce

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

        pred = output["pred"].float()
        for factor, weight in ((2, self.multiscale_weight_2), (4, self.multiscale_weight_4)):
            if weight > 0:
                pooled_pred = F.avg_pool2d(pred, factor, factor)
                pooled_target = F.avg_pool2d(target, factor, factor)
                total = total + weight * F.mse_loss(pooled_pred, pooled_target)

        if self.field_weight > 0:
            total = total + self.field_weight * F.mse_loss(pred, target)

        if self.metric_weight > 0:
            # Arm D: mean_batch(sqrt(mean_pixels(err^2) + eps)) — matches the identified
            # per-file-averaged evaluator (l_eda/exp003) more directly than plain MSE.
            diff2 = torch.square(pred - target)
            tile_rmse = torch.sqrt(diff2.mean(dim=(1, 2, 3)) + self.metric_eps)
            total = total + self.metric_weight * tile_rmse.mean()

        return total


def build_loss(config: dict) -> nn.Module:
    loss_cfg = config.get("loss", {})
    name = loss_cfg.get("name", "mean_shape")
    if name != "mean_shape":
        raise ValueError(f"exp040 only implements the mean_shape loss, got {name!r}")
    return MeanShapeLoss(
        rain_threshold=float(loss_cfg.get("rain_threshold", 0.0)),
        mean_weight=float(loss_cfg.get("mean_weight", 1.0)),
        shape_weight=float(loss_cfg.get("shape_weight", 1.0)),
        min_wet_target=float(loss_cfg.get("min_wet_target", 0.01)),
        bce_weight=float(loss_cfg.get("bce_weight", 0.0)),
        aux_mask_weight=float(loss_cfg.get("aux_mask_weight", 0.0)),
        aux_mask_threshold=float(loss_cfg.get("aux_mask_threshold", 0.25)),
        aux_dice_weight=float(loss_cfg.get("aux_dice_weight", 0.5)),
        multiscale_weight_2=float(loss_cfg.get("multiscale_weight_2", 0.0)),
        multiscale_weight_4=float(loss_cfg.get("multiscale_weight_4", 0.0)),
        field_weight=float(loss_cfg.get("field_weight", 0.0)),
        metric_weight=float(loss_cfg.get("metric_weight", 0.0)),
        metric_eps=float(loss_cfg.get("metric_eps", 1e-6)),
    )
