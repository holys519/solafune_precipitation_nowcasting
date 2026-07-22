#!/usr/bin/env python3
"""exp053 smoke test: config/channel consistency, CPU forward/backward, and real-data checks
for the new autoregressive_prev_pred channel -- both sourcing modes (teacher forcing with
ground truth, and the injected self-prediction cache used by inference.py/self_pred_oof.py).
Run inside the container before submitting training folds."""

from __future__ import annotations

from datetime import datetime, timedelta
from pathlib import Path

import numpy as np
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
    read_target_tensor,
)
from losses import build_loss
from model import build_model

SCRIPT_DIR = Path(__file__).resolve().parent
ARMS = ["config.yaml"]


def check_arm(cfg_file: str) -> dict:
    config = yaml.safe_load((SCRIPT_DIR / cfg_file).read_text())
    features = features_from_config(config)
    assert features.get("autoregressive_prev_pred"), f"{cfg_file}: expected AR feature enabled"
    expected = expected_in_channels(
        satellite_channels=int(config["data"]["satellite_channels"]),
        max_observations=int(config["data"]["max_observations"]),
        context_rows=int(config["data"].get("context_rows", 1)),
        features=features,
    )
    assert expected == 56, f"{cfg_file}: expected 56 channels (54 base + 2 AR), got {expected}"
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


def check_ar_channel_synthetic() -> None:
    """Unit-level check of `_autoregressive_prev_pred_channels`, independent of real data:
    builds two synthetic rows 30 minutes apart at the same location and checks both sourcing
    modes without touching disk."""
    config = yaml.safe_load((SCRIPT_DIR / "config.yaml").read_text())
    features = features_from_config(config)
    target_size = (int(config["data"]["target_height"]), int(config["data"]["target_width"]))

    row_t0 = {
        "unique_id": "u0",
        "name_location": "synthetic_loc",
        "satellite_target": "himawari",
        "datetime": "2023-01-01 00:00:00",
        "last_30_minutes_observation_filename": "[]",
        "gpm_imerg_filename": "synthetic_t0.tif",
    }
    row_t1 = {
        "unique_id": "u1",
        "name_location": "synthetic_loc",
        "satellite_target": "himawari",
        "datetime": "2023-01-01 00:30:00",
        "last_30_minutes_observation_filename": "[]",
        "gpm_imerg_filename": "synthetic_t1.tif",
    }
    ds = PrecipDataset(
        [row_t0, row_t1],
        Path("/nonexistent"),
        max_observations=int(config["data"]["max_observations"]),
        satellite_channels=int(config["data"]["satellite_channels"]),
        target_size=target_size,
        context_rows=1,
        has_target=False,  # ar_cache path must not require has_target
        features=features,
    )

    # (a) no prior row / no cache entry -> zero value + mask=0
    zero_value, zero_mask = ds._autoregressive_prev_pred_channels(row_t0)
    assert torch.equal(zero_value, torch.zeros_like(zero_value)), "expected zero value with no prior row"
    assert torch.equal(zero_mask, torch.zeros_like(zero_mask)), "expected mask=0 with no prior row"

    # (b) ar_cache mode: inject a fake "own prediction" for t0 and confirm row_t1 picks it up
    fake_pred = np.full(target_size, 3.5, dtype=np.float32)
    ds.ar_cache = {("synthetic_loc", datetime(2023, 1, 1, 0, 0, 0)): fake_pred}
    value, mask = ds._autoregressive_prev_pred_channels(row_t1)
    assert torch.allclose(value, torch.full_like(value, 3.5)), "expected cached self-prediction value"
    assert torch.equal(mask, torch.ones_like(mask)), "expected mask=1 when cache has an entry"

    # (c) ar_cache present but empty for this key -> falls back to zero + mask=0, even though
    # a same-split ground-truth row exists at that timestamp (ar_cache must win, never GT)
    ds.ar_cache = {}
    value2, mask2 = ds._autoregressive_prev_pred_channels(row_t1)
    assert torch.equal(value2, torch.zeros_like(value2)), "ar_cache set but empty must still zero-fill"
    assert torch.equal(mask2, torch.zeros_like(mask2)), "ar_cache set but empty must still mask=0"

    # (d) missing/unparseable datetime -> zero + mask=0, no exception
    bad_row = dict(row_t1)
    bad_row["datetime"] = "not-a-date"
    ds.ar_cache = None
    value3, mask3 = ds._autoregressive_prev_pred_channels(bad_row)
    assert torch.equal(value3, torch.zeros_like(value3))
    assert torch.equal(mask3, torch.zeros_like(mask3))

    print("AR channel synthetic checks OK (zero-fill, self-prediction cache, cache-overrides-GT, bad datetime)")


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
    train_dir = (SCRIPT_DIR / config["data"]["train_dir"]).resolve()
    features = features_from_config(config)

    # Use enough valid rows, sorted by (location, datetime), to have a good chance of hitting a
    # same-location, 30-minutes-apart consecutive pair for the teacher-forcing check below.
    sample_rows = sorted(valid_rows, key=lambda r: (r["name_location"], r["datetime"]))[:200]
    ds = PrecipDataset(
        sample_rows,
        train_dir,
        max_observations=int(config["data"]["max_observations"]),
        satellite_channels=int(config["data"]["satellite_channels"]),
        target_size=(int(config["data"]["target_height"]), int(config["data"]["target_width"])),
        context_rows=int(config["data"].get("context_rows", 1)),
        has_target=True,
        norm_stats=norm_stats,
        augment=False,
        features=features,
    )
    expected = int(config["model"]["in_channels"])
    found_prev_present = False
    found_prev_absent = False
    for i in range(len(ds)):
        item = ds[i]
        assert item["x"].shape == (expected, 41, 41), item["x"].shape
        assert torch.isfinite(item["x"]).all(), f"non-finite input for row {i}"
        assert item["y"].shape == (1, 41, 41), item["y"].shape

        row = sample_rows[i]
        ar_value = item["x"][-2]
        ar_mask = item["x"][-1]
        assert set(torch.unique(ar_mask).tolist()) <= {0.0, 1.0}, "AR mask must be binary"
        location = row["name_location"]
        prev_time = datetime.fromisoformat(row["datetime"]) - timedelta(minutes=30)
        prev_row = ds._row_by_location_time.get((location, prev_time))
        if prev_row is not None:
            found_prev_present = True
            assert float(ar_mask[0, 0]) == 1.0, f"row {i}: prior row exists but mask=0"
            true_target = read_target_tensor(train_dir / "gpm_imerg" / prev_row["gpm_imerg_filename"])
            assert torch.allclose(ar_value, true_target[0], atol=1e-4), (
                f"row {i}: teacher-forced AR value does not match prior row's true GPM target"
            )
        else:
            found_prev_absent = True
            assert float(ar_mask[0, 0]) == 0.0, f"row {i}: no prior row but mask=1"
            assert float(ar_value.abs().max()) == 0.0, f"row {i}: no prior row but AR value nonzero"

    assert found_prev_absent, "expected at least one row with no prior row in this sample"
    print(
        f"real-data check OK: {len(ds)} rows, input {tuple(item['x'].shape)}, "
        f"found_prev_present={found_prev_present}, found_prev_absent={found_prev_absent}"
    )

    model = build_model(config)
    x = torch.stack([ds[i]["x"] for i in range(4)])
    out = model(x)
    assert out["pred"].shape == (4, 1, 41, 41)

    # Sequential inference-mode simulation on 3 real rows from one location: ar_cache-driven,
    # never touching ground truth, mirroring inference.py/self_pred_oof.py.
    by_loc: dict[str, list[dict]] = {}
    for row in sample_rows:
        by_loc.setdefault(row["name_location"], []).append(row)
    loc_with_multi = next((rs for rs in by_loc.values() if len(rs) >= 2), None)
    if loc_with_multi:
        loc_rows = sorted(loc_with_multi, key=lambda r: r["datetime"])[:3]
        ar_cache: dict = {}
        ds.ar_cache = ar_cache
        with torch.no_grad():
            for row in loc_rows:
                xi = ds.input_tensor(row).unsqueeze(0)
                out_i = model(xi)
                pred_arr = out_i["pred"][0, 0].numpy()
                ar_cache[(row["name_location"], datetime.fromisoformat(row["datetime"]))] = pred_arr
        print(f"sequential ar_cache simulation OK over {len(loc_rows)} rows of one location")
        ds.ar_cache = None
    else:
        print("sequential ar_cache simulation SKIPPED: no location with >=2 sampled rows")


def main() -> None:
    for arm in ARMS:
        result = check_arm(arm)
        print(result)
    check_ar_channel_synthetic()
    check_real_batch()
    print("exp053 smoke test PASSED")


if __name__ == "__main__":
    main()
