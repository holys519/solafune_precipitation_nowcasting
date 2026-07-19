#!/usr/bin/env python3
"""Fast container smoke test for exp041 configs, loss delta, and initial checkpoints."""

from __future__ import annotations

from copy import deepcopy
from pathlib import Path

import torch
import yaml

from losses import build_loss
from model import build_model


SCRIPT_DIR = Path(__file__).resolve().parent
BASE_DIR = SCRIPT_DIR.parent / "exp038"


def load(name: str) -> dict:
    return yaml.safe_load((SCRIPT_DIR / name).read_text())


def comparable(config: dict) -> dict:
    result = deepcopy(config)
    result["experiment"].pop("name", None)
    result["experiment"].pop("description", None)
    result["loss"].pop("metric_weight", None)
    result["paths"] = {"norm_stats": result["paths"]["norm_stats"]}
    result["postprocess"]["calibration_path"] = "<arm-specific>"
    return result


def synthetic_output() -> tuple[dict[str, torch.Tensor], torch.Tensor]:
    target = torch.rand(2, 1, 41, 41)
    target = target * (torch.rand_like(target) > 0.8)
    return {
        "pred": torch.rand_like(target, requires_grad=True),
        "rain_logits": torch.randn_like(target, requires_grad=True),
        "mu": torch.randn_like(target, requires_grad=True),
        "sigma": torch.rand_like(target, requires_grad=True) + 0.1,
        "aux_mask_logits": torch.randn_like(target, requires_grad=True),
    }, target


def main() -> None:
    control = load("config_control.yaml")
    metric = load("config_metric.yaml")
    assert comparable(control) == comparable(metric), "arms differ outside declared fields"
    assert float(control["loss"]["metric_weight"]) == 0.0
    assert float(metric["loss"]["metric_weight"]) == 0.3

    output, target = synthetic_output()
    control_loss = build_loss(control)(output, target)
    metric_loss = build_loss(metric)(output, target)
    eps = float(metric["loss"]["metric_eps"])
    expected_extra = 0.3 * torch.sqrt(
        torch.square(output["pred"] - target).mean(dim=(1, 2, 3)) + eps
    ).mean()
    assert torch.allclose(metric_loss - control_loss, expected_extra, atol=1e-6)
    metric_loss.backward()
    assert output["pred"].grad is not None
    assert torch.isfinite(output["pred"].grad).all()

    model = build_model(control)
    template = str(control["train"]["init_checkpoint"])
    for fold in (0, 4):
        checkpoint_path = (BASE_DIR / template.format(fold=fold)).resolve()
        assert checkpoint_path.is_file(), checkpoint_path
        checkpoint = torch.load(checkpoint_path, map_location="cpu")
        assert int(checkpoint["fold"]) == fold
        model.load_state_dict(checkpoint["model_state_dict"], strict=True)
        print(
            f"fold={fold} checkpoint={checkpoint_path} "
            f"best_tile_rmse={float(checkpoint['best_tile_rmse']):.10f}"
        )

    print(
        f"loss delta OK: control={float(control_loss):.6f} "
        f"metric={float(metric_loss):.6f} extra={float(expected_extra):.6f}"
    )
    print("exp041 smoke test PASSED")


if __name__ == "__main__":
    main()
