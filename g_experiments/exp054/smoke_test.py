#!/usr/bin/env python3
"""exp054 smoke test: config/channel consistency, CPU forward/backward for every arm, and a
real-data batch through the full-arm dataset pipeline. Run inside the container before
submitting training folds."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml
from torch.nn import functional as F

from dataset import (
    PrecipDataset,
    drop_zero_observation_rows,
    expected_in_channels,
    features_from_config,
    load_norm_stats,
    make_group_kfold_split,
    read_rows,
)
from losses import AMOUNT_BIN_EDGES, HurdleLogNormalLoss, amount_bin_index, build_loss
from model import build_model

SCRIPT_DIR = Path(__file__).resolve().parent
ARMS = ["config.yaml", "config_midband.yaml", "config_heavytail.yaml"]


def check_arm(cfg_file: str) -> dict:
    config = yaml.safe_load((SCRIPT_DIR / cfg_file).read_text())
    features = features_from_config(config)
    expected = expected_in_channels(
        satellite_channels=int(config["data"]["satellite_channels"]),
        max_observations=int(config["data"]["max_observations"]),
        context_rows=int(config["data"].get("context_rows", 1)),
        features=features,
    )
    configured = int(config["model"]["in_channels"])
    assert configured == expected, f"{cfg_file}: in_channels {configured} != expected {expected}"

    model = build_model(config)
    loss_fn = build_loss(config)
    x = torch.randn(2, expected, 41, 41)
    y = torch.rand(2, 1, 41, 41) * (torch.rand(2, 1, 41, 41) > 0.8)
    out = model(x)
    assert out["pred"].shape == (2, 1, 41, 41), out["pred"].shape
    assert (out["pred"] >= 0).all(), "served prediction must be non-negative"
    assert "aux_mask_logits" in out, "aux mask head missing"
    loss = loss_fn(out, y)
    assert torch.isfinite(loss), f"{cfg_file}: non-finite loss {loss}"
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads), "non-finite gradients"
    dilated = any(name.startswith("bottleneck.1") for name, _ in model.named_parameters())
    wants_dilated = bool(config["model"].get("bottleneck_dilations"))
    assert dilated == wants_dilated, f"{cfg_file}: dilated={dilated} but config wants {wants_dilated}"
    return {
        "config": cfg_file,
        "in_channels": expected,
        "loss": float(loss),
        "params": sum(p.numel() for p in model.parameters()),
        "dilated_bottleneck": dilated,
    }


def check_amount_bin_reweighting() -> None:
    """exp054's core correctness requirement (see task/README.md): the amount-bin edges must
    bin values the way l_eda/exp004's regime_of does, all-1.0 weights must reproduce the
    pre-exp054 (exp038) hurdle loss bit-for-bit, and non-uniform weights must actually change
    the loss (otherwise the mechanism would be a no-op regardless of config)."""
    assert AMOUNT_BIN_EDGES == (0.0, 0.01, 0.1, 0.3, 1.0, 100.0), AMOUNT_BIN_EDGES

    probe_values = torch.tensor([0.0, 0.005, 0.01, 0.05, 0.1, 0.2, 0.3, 0.6, 1.0, 5.0, 99.0])
    expected_bins = [0, 0, 1, 1, 2, 2, 3, 3, 4, 4, 4]
    computed_bins = amount_bin_index(probe_values).tolist()
    assert computed_bins == expected_bins, f"bin index mismatch: {computed_bins} != {expected_bins}"

    torch.manual_seed(0)
    batch, height, width = 3, 8, 8
    output = {
        "rain_logits": torch.randn(batch, 1, height, width),
        "mu": torch.randn(batch, 1, height, width),
        "sigma": torch.rand(batch, 1, height, width) * 0.5 + 0.2,
        "pred": torch.rand(batch, 1, height, width),
    }
    target = torch.zeros(batch, 1, height, width)
    flat_target = target.view(-1)
    for i, value in enumerate(probe_values.tolist()):
        flat_target[i] = value
    target = flat_target.view(batch, 1, height, width)

    baseline_loss_fn = HurdleLogNormalLoss()  # every default, including amount_bin_weights
    uniform_loss_fn = HurdleLogNormalLoss(amount_bin_weights=(1.0, 1.0, 1.0, 1.0, 1.0))
    loss_baseline = baseline_loss_fn(output, target)
    loss_uniform = uniform_loss_fn(output, target)
    assert torch.equal(loss_baseline, loss_uniform), (
        f"default amount_bin_weights must reproduce the unweighted loss bit-for-bit: "
        f"{loss_baseline.item()} != {loss_uniform.item()}"
    )

    # Independent reference computation of the pre-exp054 (exp038) hurdle loss, with NO bin
    # weighting logic at all, as the ground truth for the "numerically identical" requirement.
    with torch.no_grad():
        rain_target = (target > 0).float()
        bce_ref = F.binary_cross_entropy_with_logits(output["rain_logits"], rain_target)
        wet_ref = target > 0
        ln_y_ref = torch.log(target.clamp_min(0.01))
        z2_ref = torch.square((ln_y_ref - output["mu"]) / output["sigma"])
        nll_ref = 0.5 * z2_ref + torch.log(output["sigma"])
        intensity_ref = nll_ref[wet_ref].mean()
        loss_ref = bce_ref + intensity_ref
    assert torch.allclose(loss_baseline, loss_ref, atol=1e-6), (
        f"exp054 default loss diverges from the reference exp038 formula: "
        f"{loss_baseline.item()} vs {loss_ref.item()}"
    )

    midband_loss_fn = HurdleLogNormalLoss(amount_bin_weights=(1.0, 1.0, 1.5, 1.5, 1.0))
    heavytail_loss_fn = HurdleLogNormalLoss(amount_bin_weights=(1.0, 1.0, 1.0, 1.5, 2.0))
    loss_midband = midband_loss_fn(output, target)
    loss_heavytail = heavytail_loss_fn(output, target)
    assert loss_midband.item() != loss_baseline.item(), "midband weights must change the loss"
    assert loss_heavytail.item() != loss_baseline.item(), "heavytail weights must change the loss"
    assert loss_midband.item() != loss_heavytail.item(), "the two arms must diverge from each other"

    try:
        HurdleLogNormalLoss(amount_bin_weights=(1.0, 1.0, 1.0, 1.0))
    except ValueError:
        pass
    else:
        raise AssertionError("wrong-length amount_bin_weights must raise ValueError")

    print(
        "amount_bin_reweighting unit check OK: "
        f"baseline={loss_baseline.item():.6f} midband={loss_midband.item():.6f} "
        f"heavytail={loss_heavytail.item():.6f}"
    )


def check_real_batch() -> None:
    config = yaml.safe_load((SCRIPT_DIR / "config.yaml").read_text())
    train_csv = (SCRIPT_DIR / config["data"]["train_csv"]).resolve()
    if not train_csv.exists():
        print(f"real-data check SKIPPED: {train_csv} not found")
        return
    rows = read_rows(train_csv)
    train_rows, valid_rows, _ = make_group_kfold_split(
        rows, n_splits=int(config["split"]["n_splits"]), fold=0, seed=int(config["experiment"]["seed"])
    )
    n_before = len(train_rows)
    train_rows = drop_zero_observation_rows(train_rows)
    print(f"drop_zero_obs_rows: {n_before} -> {len(train_rows)}")
    norm_stats = load_norm_stats(SCRIPT_DIR / config["paths"]["norm_stats"])
    ds = PrecipDataset(
        valid_rows[:6],
        (SCRIPT_DIR / config["data"]["train_dir"]).resolve(),
        max_observations=int(config["data"]["max_observations"]),
        satellite_channels=int(config["data"]["satellite_channels"]),
        target_size=(int(config["data"]["target_height"]), int(config["data"]["target_width"])),
        context_rows=int(config["data"].get("context_rows", 1)),
        has_target=True,
        norm_stats=norm_stats,
        augment=False,
        features=features_from_config(config),
    )
    expected = int(config["model"]["in_channels"])
    for i in range(len(ds)):
        item = ds[i]
        assert item["x"].shape == (expected, 41, 41), item["x"].shape
        assert torch.isfinite(item["x"]).all(), f"non-finite input for row {i}"
        assert item["y"].shape == (1, 41, 41), item["y"].shape
    model = build_model(config)
    x = torch.stack([ds[i]["x"] for i in range(4)])
    out = model(x)
    assert out["pred"].shape == (4, 1, 41, 41)
    print(f"real-data check OK: {len(ds)} rows, input {tuple(x.shape)}")


def main() -> None:
    check_amount_bin_reweighting()
    for arm in ARMS:
        result = check_arm(arm)
        print(result)
    check_real_batch()
    print("exp054 smoke test PASSED")


if __name__ == "__main__":
    main()
