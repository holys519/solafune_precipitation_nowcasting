#!/usr/bin/env python3
"""exp060 smoke test: config/channel consistency, CPU forward/backward (incl. both new loss terms
individually ablated to loss weight 0, to prove that's a sane no-crash configuration), a
mean_intensity/shape wiring sanity check, and a real-data batch through the dataset pipeline. Run
inside the container before submitting training folds."""

from __future__ import annotations

from pathlib import Path

import torch
import yaml

from dataset import (
    PrecipDataset,
    drop_zero_observation_rows,
    expected_in_channels,
    features_from_config,
    load_norm_stats,
    make_group_kfold_split,
    read_rows,
)
from losses import build_loss
from model import build_model

SCRIPT_DIR = Path(__file__).resolve().parent
ARMS = ["config.yaml"]


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
    assert "distribution" in out, "distribution head missing"
    assert "tile_total" in out and "log_total" in out, "total head missing"
    # distribution sums to exactly 1 per tile (softmax over the whole 41x41 grid)
    dist_sum = out["distribution"].flatten(1).sum(1)
    assert torch.allclose(dist_sum, torch.ones_like(dist_sum), atol=1e-4), (
        f"distribution does not sum to 1 per tile: {dist_sum}"
    )
    assert (out["distribution"] >= 0).all(), "distribution must be non-negative"
    # exact identifiability: sum(prediction) == tile_total (no rain_prob double-gate)
    pred_sum = out["pred"].flatten(1).sum(1)
    assert torch.allclose(pred_sum, out["tile_total"], rtol=1e-4, atol=1e-4), (
        f"sum(pred) != tile_total: {pred_sum} vs {out['tile_total']}"
    )
    assert (out["tile_total"] >= 0).all(), "tile_total must be non-negative"
    assert torch.isfinite(out["tile_total"]).all(), "tile_total must be finite (log-clamp safety)"

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


def check_loss_ablation_toggles(cfg_file: str) -> None:
    """loss.total_weight: 0 and loss.distribution_weight: 0 must each run without crashing,
    including the degenerate all-dry-target case where the distribution term has zero wet tiles
    (undefined normalized field). Also exercises a HEAVY-rain target to confirm the log-space
    total loss stays finite where exp058's raw-mm loss diverged."""
    base_config = yaml.safe_load((SCRIPT_DIR / cfg_file).read_text())
    x = torch.randn(3, int(base_config["model"]["in_channels"]), 41, 41)
    y_mixed = torch.rand(3, 1, 41, 41) * (torch.rand(3, 1, 41, 41) > 0.8)
    y_all_dry = torch.zeros(3, 1, 41, 41)
    y_heavy = torch.rand(3, 1, 41, 41) * 150.0 * (torch.rand(3, 1, 41, 41) > 0.5)  # tile totals in the thousands

    for weight_key in ("total_weight", "distribution_weight"):
        for y in (y_mixed, y_all_dry, y_heavy):
            config = yaml.safe_load((SCRIPT_DIR / cfg_file).read_text())
            config["loss"][weight_key] = 0.0
            model = build_model(config)
            loss_fn = build_loss(config)
            out = model(x)
            loss = loss_fn(out, y)
            assert torch.isfinite(loss), f"{weight_key}=0, all_dry={torch.equal(y, y_all_dry)}: non-finite loss"
            loss.backward()
            grads = [p.grad for p in model.parameters() if p.grad is not None]
            assert all(torch.isfinite(g).all() for g in grads), f"{weight_key}=0: non-finite gradients"

    # both primary terms off simultaneously
    config = yaml.safe_load((SCRIPT_DIR / cfg_file).read_text())
    config["loss"]["total_weight"] = 0.0
    config["loss"]["distribution_weight"] = 0.0
    model = build_model(config)
    loss_fn = build_loss(config)
    out = model(x)
    loss = loss_fn(out, y_mixed)
    assert torch.isfinite(loss)
    loss.backward()

    # default weights: all-dry AND heavy-rain must both stay finite (the exp058 NaN regression guard)
    for y, label in ((y_all_dry, "all-dry"), (y_heavy, "heavy-rain")):
        config = yaml.safe_load((SCRIPT_DIR / cfg_file).read_text())
        model = build_model(config)
        loss_fn = build_loss(config)
        out = model(x)
        loss = loss_fn(out, y)
        assert torch.isfinite(loss), f"default weights + {label} target: non-finite loss"
        loss.backward()
        grads = [p.grad for p in model.parameters() if p.grad is not None]
        assert all(torch.isfinite(g).all() for g in grads), f"default + {label}: non-finite gradients"
    print(f"{cfg_file}: loss ablation toggles (total_weight=0 / distribution_weight=0 / both, "
          "incl. all-dry and heavy-rain finiteness guards) all finite, no crash")


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
    for arm in ARMS:
        result = check_arm(arm)
        print(result)
        check_loss_ablation_toggles(arm)
    check_real_batch()
    print("exp060 smoke test PASSED")


if __name__ == "__main__":
    main()
