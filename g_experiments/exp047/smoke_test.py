#!/usr/bin/env python3
"""exp047 smoke test: config/channel consistency, CPU forward/backward for every arm, and a
real-data batch through the full-arm dataset pipeline. Run inside the container before
submitting training folds."""

from __future__ import annotations

from datetime import datetime
from pathlib import Path

import torch
import yaml

from dataset import (
    N_SOLAR_CHANNELS,
    PrecipDataset,
    drop_zero_observation_rows,
    expected_in_channels,
    features_from_config,
    geo_table_from_config,
    load_norm_stats,
    make_group_kfold_split,
    read_rows,
)
from losses import build_loss
from model import build_model
from solar_features import solar_position_channels

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
    geo_table = geo_table_from_config(config, lambda p: (SCRIPT_DIR / p).resolve())
    assert geo_table, "geocoded_locations_path configured but table loaded empty"
    missing = {row["name_location"] for row in valid_rows[:50]} - set(geo_table)
    assert not missing, f"locations missing from geocoded table: {missing}"

    ds_kwargs = dict(
        max_observations=int(config["data"]["max_observations"]),
        satellite_channels=int(config["data"]["satellite_channels"]),
        target_size=(int(config["data"]["target_height"]), int(config["data"]["target_width"])),
        context_rows=int(config["data"].get("context_rows", 1)),
        has_target=True,
        norm_stats=norm_stats,
        augment=False,
        features=features_from_config(config),
    )
    train_dir = (SCRIPT_DIR / config["data"]["train_dir"]).resolve()
    ds = PrecipDataset(valid_rows[:6], train_dir, geo_table=geo_table, **ds_kwargs)
    expected = int(config["model"]["in_channels"])
    for i in range(len(ds)):
        item = ds[i]
        assert item["x"].shape == (expected, 41, 41), item["x"].shape
        assert torch.isfinite(item["x"]).all(), f"non-finite input for row {i}"
        assert item["y"].shape == (1, 41, 41), item["y"].shape

        row = valid_rows[i]
        lat, lon = geo_table[row["name_location"]]
        expected_solar = solar_position_channels(lat, lon, datetime.fromisoformat(row["datetime"]))
        actual_solar = tuple(float(v) for v in item["x"][-N_SOLAR_CHANNELS:, 0, 0])
        assert all(abs(a - e) < 1e-5 for a, e in zip(actual_solar, expected_solar)), (
            f"row {i} ({row['name_location']}): solar channels {actual_solar} != "
            f"directly-computed {expected_solar}"
        )
        # constant across the spatial grid, like the existing satellite one-hot channels
        assert torch.equal(item["x"][-N_SOLAR_CHANNELS:, 0, 0], item["x"][-N_SOLAR_CHANNELS:, -1, -1])
    model = build_model(config)
    x = torch.stack([ds[i]["x"] for i in range(4)])
    out = model(x)
    assert out["pred"].shape == (4, 1, 41, 41)
    print(f"real-data check OK: {len(ds)} rows, input {tuple(x.shape)}")


def main() -> None:
    for arm in ARMS:
        result = check_arm(arm)
        print(result)
    check_real_batch()
    print("exp047 smoke test PASSED")


if __name__ == "__main__":
    main()
