#!/usr/bin/env python3
"""exp052 smoke test: config/channel consistency, CPU forward/backward for every arm, a real-
data batch through the full-arm dataset pipeline, and the compliance checks required for the
new train-time-only future-frame auxiliary head (ticket G-033, reinterpreted per the
2026-07-20 organizer ruling -- see README.md):

- check_loss_future_aux_noop: future_aux_weight == 0.0 must reproduce unmodified exp038's
  HurdleLogNormalLoss bit-for-bit, regardless of what future_target/future_valid/output are.
- check_future_aux_head_no_inference_coupling: prediction_from_output(...) must be bit-
  identical whether model.future_aux_head is True or False, given identical shared weights --
  proves zero inference-time coupling.
- check_future_aux_guard_evaluation_mode: PrecipDataset._future_aux_target must raise if ever
  called on a dataset built with has_target=False, or pointed at an evaluation data dir.

Run inside the container before submitting training folds.
"""

from __future__ import annotations

import copy
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
from losses import HurdleLogNormalLoss, build_loss
from model import HighResHurdleLogNormalUNet, build_model, prediction_from_output

SCRIPT_DIR = Path(__file__).resolve().parent
ARMS = ["config.yaml", "config_horizon2.yaml"]


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

    future_aux_weight = float(config.get("loss", {}).get("future_aux_weight", 0.0))
    if future_aux_weight > 0:
        assert "future_aux" in out, f"{cfg_file}: model.future_aux_head enabled but output missing 'future_aux'"
        assert out["future_aux"].shape == (2, 1, 41, 41), out["future_aux"].shape
        future_target = torch.rand(2, 1, 41, 41)
        future_valid = torch.tensor([1.0, 0.0])
        loss = loss_fn(out, y, future_target=future_target, future_valid=future_valid)
    else:
        loss = loss_fn(out, y)
    assert torch.isfinite(loss), f"{cfg_file}: non-finite loss {loss}"
    loss.backward()
    grads = [p.grad for p in model.parameters() if p.grad is not None]
    assert grads and all(torch.isfinite(g).all() for g in grads), "non-finite gradients"
    dilated = any(name.startswith("bottleneck.1") for name, _ in model.named_parameters())
    wants_dilated = bool(config["model"].get("bottleneck_dilations"))
    assert dilated == wants_dilated, f"{cfg_file}: dilated={dilated} but config wants {wants_dilated}"
    has_future_aux_decoder = any(name.startswith("future_aux_decoder.") for name, _ in model.named_parameters())
    wants_future_aux_head = bool(config["model"].get("future_aux_head", False))
    assert has_future_aux_decoder == wants_future_aux_head, (
        f"{cfg_file}: future_aux_decoder present={has_future_aux_decoder} but config wants {wants_future_aux_head}"
    )
    return {
        "config": cfg_file,
        "in_channels": expected,
        "loss": float(loss),
        "params": sum(p.numel() for p in model.parameters()),
        "dilated_bottleneck": dilated,
        "future_aux_head": has_future_aux_decoder,
    }


def check_loss_future_aux_noop() -> None:
    """HARD requirement (task 2/losses.py): future_aux_weight == 0.0 must be numerically
    IDENTICAL to unmodified exp038's HurdleLogNormalLoss -- i.e. a strict no-op, regardless of
    whether future_target/future_valid/output["future_aux"] are provided at all."""
    torch.manual_seed(0)
    batch, h, w = 3, 41, 41
    output = {
        "pred": torch.rand(batch, 1, h, w),
        "rain_logits": torch.randn(batch, 1, h, w),
        "mu": torch.randn(batch, 1, h, w),
        "sigma": torch.rand(batch, 1, h, w) + 0.1,
        "aux_mask_logits": torch.randn(batch, 1, h, w),
    }
    target = torch.rand(batch, 1, h, w) * (torch.rand(batch, 1, h, w) > 0.8)

    baseline_loss_fn = HurdleLogNormalLoss(aux_mask_weight=0.2, multiscale_weight_2=0.2, multiscale_weight_4=0.1)
    assert baseline_loss_fn.future_aux_weight == 0.0
    baseline = baseline_loss_fn(output, target)

    # Same weight config (future_aux_weight defaults to 0.0), called with NO future args.
    no_future_args = baseline_loss_fn(output, target, future_target=None, future_valid=None)
    assert torch.equal(baseline, no_future_args), "future_aux_weight=0.0 with no future args must match baseline"

    # Same weight config, but with future_target/future_valid/output["future_aux"] all
    # provided -- must STILL be a no-op because future_aux_weight is 0.0.
    output_with_aux = dict(output)
    output_with_aux["future_aux"] = torch.randn(batch, 1, h, w)
    future_target = torch.rand(batch, 1, h, w)
    future_valid = torch.tensor([1.0, 0.0, 1.0])
    with_future_args = baseline_loss_fn(output_with_aux, target, future_target=future_target, future_valid=future_valid)
    assert torch.equal(baseline, with_future_args), (
        "future_aux_weight=0.0 must be a strict no-op even when future_target/future_valid/"
        "output['future_aux'] are all provided -- this is the exp038-equivalence guarantee"
    )

    # Sanity check the term is NOT a no-op once future_aux_weight > 0, so the guard above is
    # actually testing something (i.e. the term has a real, nonzero effect when enabled).
    active_loss_fn = HurdleLogNormalLoss(aux_mask_weight=0.2, multiscale_weight_2=0.2, multiscale_weight_4=0.1, future_aux_weight=0.5)
    active = active_loss_fn(output_with_aux, target, future_target=future_target, future_valid=future_valid)
    assert not torch.equal(baseline, active), "future_aux_weight > 0 should change the loss value"
    print("check_loss_future_aux_noop OK: future_aux_weight=0.0 is a bit-exact no-op vs exp038")


def check_future_aux_head_no_inference_coupling() -> None:
    """HARD requirement (task 3c): prediction_from_output(...) must be BIT-IDENTICAL whether
    model.future_aux_head is True or False, given identical shared weights -- this proves zero
    inference-time coupling between the auxiliary branch and the served prediction."""
    torch.manual_seed(42)
    kwargs = dict(
        in_channels=54,
        base_channels=8,
        internal_size=32,
        output_size=(41, 41),
        aux_mask_enabled=True,
        sigma_mode="predicted",
    )
    model_with_aux = HighResHurdleLogNormalUNet(future_aux_head=True, **kwargs).eval()
    model_without_aux = HighResHurdleLogNormalUNet(future_aux_head=False, **kwargs).eval()

    # Copy every shared-parameter weight from model_with_aux into model_without_aux so the two
    # differ ONLY in whether the future_aux_decoder module (and its forward-pass branch) exists.
    with_aux_state = model_with_aux.state_dict()
    shared_state = {k: v for k, v in with_aux_state.items() if not k.startswith("future_aux_decoder.")}
    missing, unexpected = model_without_aux.load_state_dict(shared_state, strict=False)
    assert not unexpected, f"unexpected keys copying shared weights: {unexpected}"
    assert not missing, f"missing keys copying shared weights: {missing}"

    # Eval-shaped batch: no target, batch of evaluation-typical size.
    x = torch.randn(4, 54, 41, 41)
    with torch.no_grad():
        out_with_aux = model_with_aux(x)
        out_without_aux = model_without_aux(x)

    assert "future_aux" in out_with_aux, "future_aux_head=True must produce output['future_aux']"
    assert "future_aux" not in out_without_aux, "future_aux_head=False must NOT produce output['future_aux']"

    pred_with_aux = prediction_from_output(out_with_aux)
    pred_without_aux = prediction_from_output(out_without_aux)
    assert torch.equal(pred_with_aux, pred_without_aux), (
        "prediction_from_output(...) differs between future_aux_head=True/False with identical "
        "shared weights -- this would mean the auxiliary branch leaks into the served "
        "prediction, which must never happen"
    )
    print("check_future_aux_head_no_inference_coupling OK: served prediction is bit-identical")


def check_future_aux_guard_evaluation_mode() -> None:
    """HARD requirement (task 3b): the future-target lookup must only ever be reachable from
    train/valid data. Assert PrecipDataset._future_aux_target raises when has_target=False, and
    when the dataset is pointed at a directory whose name signals evaluation data."""
    row = {
        "name_location": "nowhere",
        "satellite_target": "himawari",
        "datetime": "2026-01-01T00:00:00",
    }
    common = dict(
        rows=[row],
        max_observations=3,
        satellite_channels=16,
        target_size=(41, 41),
        features={"future_aux_head": True},
        future_aux_horizon_rows=1,
    )

    eval_mode_ds = PrecipDataset(data_dir=Path("/tmp/does_not_matter"), has_target=False, **common)
    try:
        eval_mode_ds._future_aux_target(row)
    except RuntimeError as exc:
        assert "has_target=False" in str(exc) or "evaluation" in str(exc).lower()
    else:
        raise AssertionError("_future_aux_target must raise when has_target=False")

    eval_dir_ds = PrecipDataset(
        data_dir=Path("/tmp/fake/evaluation_dataset"), has_target=True, **common
    )
    try:
        eval_dir_ds._future_aux_target(row)
    except RuntimeError as exc:
        assert "evaluation" in str(exc).lower()
    else:
        raise AssertionError("_future_aux_target must raise when data_dir looks like an evaluation dir")

    # Sanity: the guard does NOT fire for an ordinary train-mode dataset (it should just return
    # a zero placeholder here since /tmp/does_not_matter/himawari/... won't exist).
    train_mode_ds = PrecipDataset(data_dir=Path("/tmp/does_not_matter_train"), has_target=True, **common)
    target, valid = train_mode_ds._future_aux_target(row)
    assert target.shape == (1, 41, 41)
    assert float(valid) == 0.0  # no future row exists for this synthetic single-row dataset
    print("check_future_aux_guard_evaluation_mode OK: has_target=False / evaluation_dir both raise")


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
    ds = PrecipDataset(
        valid_rows[:12],
        train_dir,
        max_observations=int(config["data"]["max_observations"]),
        satellite_channels=int(config["data"]["satellite_channels"]),
        target_size=(int(config["data"]["target_height"]), int(config["data"]["target_width"])),
        context_rows=int(config["data"].get("context_rows", 1)),
        has_target=True,
        norm_stats=norm_stats,
        augment=False,
        features=features_from_config(config),
        future_aux_horizon_rows=int(config["data"].get("future_aux_horizon_rows", 1)),
    )
    expected = int(config["model"]["in_channels"])
    n_valid_future = 0
    for i in range(len(ds)):
        item = ds[i]
        assert item["x"].shape == (expected, 41, 41), item["x"].shape
        assert torch.isfinite(item["x"]).all(), f"non-finite input for row {i}"
        assert item["y"].shape == (1, 41, 41), item["y"].shape
        assert "future_target" in item and "future_valid" in item, "future_aux_head enabled but keys missing"
        assert item["future_target"].shape == (1, 41, 41), item["future_target"].shape
        assert torch.isfinite(item["future_target"]).all(), f"non-finite future_target for row {i}"
        assert float(item["future_valid"]) in (0.0, 1.0)
        n_valid_future += int(item["future_valid"])
    print(f"real-data check: {n_valid_future}/{len(ds)} rows had a valid future-frame target")

    model = build_model(config)
    x = torch.stack([ds[i]["x"] for i in range(4)])
    out = model(x)
    assert out["pred"].shape == (4, 1, 41, 41)
    assert "future_aux" in out, "config.yaml has model.future_aux_head=true but forward() output is missing it"
    assert out["future_aux"].shape == (4, 1, 41, 41)

    # Also exercise the augmentation path (x, y, future_target must transform together).
    ds_aug = PrecipDataset(
        valid_rows[:4],
        train_dir,
        max_observations=int(config["data"]["max_observations"]),
        satellite_channels=int(config["data"]["satellite_channels"]),
        target_size=(int(config["data"]["target_height"]), int(config["data"]["target_width"])),
        context_rows=int(config["data"].get("context_rows", 1)),
        has_target=True,
        norm_stats=norm_stats,
        augment=True,
        features=features_from_config(config),
        future_aux_horizon_rows=int(config["data"].get("future_aux_horizon_rows", 1)),
    )
    for i in range(len(ds_aug)):
        item = ds_aug[i]
        assert item["x"].shape == (expected, 41, 41)
        assert item["future_target"].shape == (1, 41, 41)
    print(f"real-data check OK: {len(ds)} rows, input {tuple(x.shape)}")


def main() -> None:
    for arm in ARMS:
        result = check_arm(arm)
        print(result)
    check_loss_future_aux_noop()
    check_future_aux_head_no_inference_coupling()
    check_future_aux_guard_evaluation_mode()
    check_real_batch()
    print("exp052 smoke test PASSED")


if __name__ == "__main__":
    main()
