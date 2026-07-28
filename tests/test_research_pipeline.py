from __future__ import annotations

import torch

from precip_nowcast.config import LossConfig
from precip_nowcast.data import make_balanced_group_folds
from precip_nowcast.losses import CompositePrecipitationLoss
from precip_nowcast.metrics import MetricAccumulator
from precip_nowcast.model import TotalShapeNowcaster


def test_model_is_amount_identifiable() -> None:
    model = TotalShapeNowcaster(in_channels=54, base_channels=8, total_hidden=16)
    output = model(torch.randn(2, 54, 64, 64))
    assert output.prediction.shape == (2, 1, 41, 41)
    assert torch.allclose(output.spatial_distribution.flatten(1).sum(1), torch.ones(2))
    assert torch.allclose(
        output.prediction.flatten(1).sum(1), output.tile_total, rtol=1e-5, atol=1e-5
    )


def test_loss_is_finite_for_dry_and_wet_tiles() -> None:
    model = TotalShapeNowcaster(in_channels=54, base_channels=8, total_hidden=16)
    output = model(torch.randn(2, 54, 64, 64))
    target = torch.zeros(2, 1, 41, 41)
    target[1, 0, 10:15, 12:18] = 2.0
    loss, components = CompositePrecipitationLoss(LossConfig())(output, target)
    assert torch.isfinite(loss)
    assert all(torch.isfinite(value) for value in components.values())
    loss.backward()
    assert any(parameter.grad is not None for parameter in model.parameters())


def test_group_folds_never_split_a_location() -> None:
    rows = [
        {"name_location": f"loc_{location}", "satellite_target": ("goes", "himawari", "meteosat")[location % 3]}
        for location in range(10)
        for _ in range(location + 1)
    ]
    assignment = make_balanced_group_folds(rows, n_splits=5, seed=42)
    assert set(assignment) == {f"loc_{index}" for index in range(10)}
    assert set(assignment.values()) == set(range(5))


def test_metric_aggregation_is_explicit() -> None:
    prediction = torch.tensor([[[[0.0, 2.0]]], [[[1.0, 1.0]]]])
    target = torch.zeros_like(prediction)
    metric = MetricAccumulator()
    metric.update(prediction, target)
    values = metric.compute()
    assert values["pooled_rmse"] > 0
    assert values["tile_rmse"] > 0
    assert values["samples"] == 2
