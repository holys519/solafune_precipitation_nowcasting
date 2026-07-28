"""Streaming competition metrics with explicit aggregation semantics."""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class MetricAccumulator:
    squared_error: float = 0.0
    pixels: int = 0
    tile_rmse_sum: float = 0.0
    tile_count: int = 0
    absolute_total_error: float = 0.0

    @torch.no_grad()
    def update(self, prediction: torch.Tensor, target: torch.Tensor) -> None:
        diff2 = (prediction.float() - target.float()).square()
        self.squared_error += float(diff2.sum())
        self.pixels += diff2.numel()
        self.tile_rmse_sum += float(diff2.flatten(1).mean(dim=1).sqrt().sum())
        self.tile_count += target.shape[0]
        self.absolute_total_error += float(
            (prediction.flatten(1).sum(1) - target.flatten(1).sum(1)).abs().sum()
        )

    def compute(self) -> dict[str, float]:
        if self.tile_count == 0:
            raise RuntimeError("cannot compute metrics before any update")
        return {
            "pooled_rmse": (self.squared_error / self.pixels) ** 0.5,
            "tile_rmse": self.tile_rmse_sum / self.tile_count,
            "mean_absolute_total_error": self.absolute_total_error / self.tile_count,
            "samples": float(self.tile_count),
        }
