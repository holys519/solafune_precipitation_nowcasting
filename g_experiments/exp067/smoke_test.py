#!/usr/bin/env python3
"""exp056 smoke test: config/channel consistency, CPU forward/backward (incl. both new loss terms
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
    load_prev_pred_lookup,
    make_group_kfold_split,
    read_rows,
)
from losses import build_loss
from model import build_model

SCRIPT_DIR = Path(__file__).resolve().parent
ARMS = ["config_effb3_ar.yaml"]


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
    assert "mean_intensity" in out, "mean_intensity head missing"
    assert "shape" in out, "shape head missing"
    assert out["mean_intensity"].shape == (2,), out["mean_intensity"].shape
    assert (out["mean_intensity"] >= 0).all(), "mean_intensity must be non-negative"
    assert out["shape"].shape == (2, 1, 41, 41), out["shape"].shape
    assert (out["shape"] >= 0).all(), "shape must be non-negative"
    # shape is normalized by the rain_prob-WEIGHTED spatial mean (2026-07-24 wet-conditional
    # design), so the correct invariant is that the rain_prob-weighted mean is 1, not the plain mean.
    rp = out["rain_prob"].detach()
    weighted_mean = (out["shape"] * rp).sum(dim=(2, 3)) / (rp.sum(dim=(2, 3)) + 1e-6)
    weighted_mean = weighted_mean.view(-1)
    assert torch.allclose(weighted_mean, torch.ones_like(weighted_mean), atol=1e-3), (
        f"shape is not rain_prob-weighted-normalized to mean 1: {weighted_mean}"
    )
    # combined amount reconstructs mean_intensity * shape exactly (before the rain_prob gate)
    amount = out["rain_amount"]
    expected_amount = out["mean_intensity"].view(-1, 1, 1, 1) * out["shape"]
    assert torch.allclose(amount, expected_amount.clamp(max=model.amount_cap), atol=1e-4)

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
    """loss.mean_intensity_weight: 0 and loss.shape_weight: 0 must each run without crashing
    (task requirement: independent ablation toggles that "behave sanely, not crash"), including
    the degenerate all-dry-target case where the shape term has zero wet pixels."""
    base_config = yaml.safe_load((SCRIPT_DIR / cfg_file).read_text())
    x = torch.randn(3, int(base_config["model"]["in_channels"]), 41, 41)
    y_mixed = torch.rand(3, 1, 41, 41) * (torch.rand(3, 1, 41, 41) > 0.8)
    y_all_dry = torch.zeros(3, 1, 41, 41)

    for weight_key in ("mean_intensity_weight", "shape_weight"):
        for y in (y_mixed, y_all_dry):
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

    # both new terms off simultaneously, plus the baseline all-dry case with default weights
    config = yaml.safe_load((SCRIPT_DIR / cfg_file).read_text())
    config["loss"]["mean_intensity_weight"] = 0.0
    config["loss"]["shape_weight"] = 0.0
    model = build_model(config)
    loss_fn = build_loss(config)
    out = model(x)
    loss = loss_fn(out, y_mixed)
    assert torch.isfinite(loss)
    loss.backward()

    config = yaml.safe_load((SCRIPT_DIR / cfg_file).read_text())
    model = build_model(config)
    loss_fn = build_loss(config)
    out = model(x)
    loss = loss_fn(out, y_all_dry)
    assert torch.isfinite(loss), "default weights + all-dry target: non-finite loss"
    loss.backward()
    print(f"{cfg_file}: loss ablation toggles (mean_intensity_weight=0 / shape_weight=0 / both, "
          "incl. all-dry-tile degenerate case) all finite, no crash")


def check_real_batch() -> None:
    config = yaml.safe_load((SCRIPT_DIR / "config_effb3_ar.yaml").read_text())
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

    prev_pred_lookup_path = (SCRIPT_DIR / config["data"]["prev_pred_lookup_npz"]).resolve()
    assert prev_pred_lookup_path.exists(), f"prev_pred_lookup_npz not found: {prev_pred_lookup_path}"
    prev_pred_lookup = load_prev_pred_lookup(prev_pred_lookup_path, rows)
    assert len(prev_pred_lookup) > len(rows) * 0.5, (
        f"prev_pred_lookup only resolved {len(prev_pred_lookup)}/{len(rows)} rows -- join is broken"
    )

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
        prev_pred_lookup=prev_pred_lookup,
    )
    expected = int(config["model"]["in_channels"])
    ar_channel_nonzero_count = 0
    for i in range(len(ds)):
        item = ds[i]
        assert item["x"].shape == (expected, 41, 41), item["x"].shape
        assert torch.isfinite(item["x"]).all(), f"non-finite input for row {i}"
        assert item["y"].shape == (1, 41, 41), item["y"].shape
        ar_mask_channel = item["x"][-1]
        if bool((ar_mask_channel > 0).any()):
            ar_channel_nonzero_count += 1
    print(
        f"AR mask channel non-empty for {ar_channel_nonzero_count}/{len(ds)} sampled valid rows "
        "(some zero is expected -- first row of a location's sequence has no earlier row)"
    )

    # ar_cache branch (eval / self-prediction-substituted mode) must take priority over
    # prev_pred_lookup and never read ground truth.
    sample_row = valid_rows[0]
    fake_pred = __import__("numpy").zeros((41, 41), dtype="float32") + 7.0
    from datetime import datetime, timedelta

    key = (sample_row["name_location"], datetime.fromisoformat(sample_row["datetime"]) - timedelta(minutes=30))
    ds.ar_cache = {key: fake_pred}
    x = ds.input_tensor(sample_row)
    assert float(x[-2].max()) == 7.0 and float(x[-1].max()) == 1.0, (
        f"ar_cache branch did not take priority over prev_pred_lookup: value_max={float(x[-2].max())} "
        f"mask_max={float(x[-1].max())}"
    )
    ds.ar_cache = None

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
    print("exp056 smoke test PASSED")


if __name__ == "__main__":
    main()
