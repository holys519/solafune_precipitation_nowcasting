#!/usr/bin/env python3
"""Train exp045: exp018's high-res localization net + exp028 temporal input design + exp030
dilated bottleneck, each gated by config so single axes can be switched off."""

from __future__ import annotations

import argparse
import csv
import json
import random
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from amp_utils import cuda_autocast, make_grad_scaler
from dataset import (
    PrecipDataset,
    drop_zero_observation_rows,
    expected_in_channels,
    features_from_config,
    load_norm_stats,
    make_group_kfold_split,
    read_rows,
    registration_shifts_from_config,
    sample_rows,
)
from losses import build_loss
from model import build_model, prediction_from_output

SCRIPT_DIR = Path(__file__).resolve().parent


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def resolve_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (SCRIPT_DIR / p).resolve()


def seed_everything(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def worker_init_fn(worker_id: int) -> None:
    torch.set_num_threads(1)
    seed = torch.initial_seed() % 2**32
    random.seed(seed + worker_id)
    np.random.seed(seed + worker_id)


def unwrap_model(model: nn.Module) -> nn.Module:
    return model.module if isinstance(model, nn.DataParallel) else model


def metric_key(config: dict[str, Any]) -> str:
    return str(config.get("metric", {}).get("selection", "tile_rmse"))


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    clip_min: float,
) -> dict[str, float]:
    model.eval()
    sse = 0.0
    zero_sse = 0.0
    positive_sse = 0.0
    pixels = 0
    positive_pixels = 0
    tile_rmse_sum = 0.0
    samples = 0
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        pred = prediction_from_output(model(x)).clamp_min(clip_min)
        diff = pred - y
        sse += float(torch.square(diff).sum().item())
        zero_sse += float(torch.square(y).sum().item())
        pixels += int(y.numel())
        tile_mse = torch.square(diff).flatten(1).mean(dim=1)
        tile_rmse_sum += float(torch.sqrt(tile_mse).sum().item())
        samples += int(y.shape[0])
        mask = y > 0
        if mask.any():
            positive_sse += float(torch.square(diff[mask]).sum().item())
            positive_pixels += int(mask.sum().item())
    return {
        "rmse": float(np.sqrt(sse / pixels)),
        "tile_rmse": float(tile_rmse_sum / max(samples, 1)),
        "zero_rmse": float(np.sqrt(zero_sse / pixels)),
        "positive_rmse": float(np.sqrt(positive_sse / positive_pixels)) if positive_pixels else float("nan"),
        "pixels": float(pixels),
        "positive_pixels": float(positive_pixels),
        "samples": float(samples),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"))
    parser.add_argument("--fold", type=int, default=None, help="Override split.fold from config")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    fold = args.fold if args.fold is not None else int(config["split"]["fold"])
    seed = int(config["experiment"]["seed"])
    seed_everything(seed)
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run this script outside the sandbox with GPU access.")

    train_csv = resolve_path(config["data"]["train_csv"])
    train_dir = resolve_path(config["data"]["train_dir"])
    model_dir = resolve_path(config["paths"]["model_dir"])
    model_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir = resolve_path(config["paths"].get("analysis_dir", "../../outputs/analysis/exp045"))
    analysis_dir.mkdir(parents=True, exist_ok=True)
    norm_stats_path = resolve_path(config["paths"]["norm_stats"])
    if not norm_stats_path.exists():
        raise FileNotFoundError(
            f"{norm_stats_path} not found. Run `python normalize_stats.py --config {args.config}` first."
        )
    norm_stats = load_norm_stats(norm_stats_path)
    registration_shifts = registration_shifts_from_config(config, resolve_path)

    rows = read_rows(train_csv)
    n_splits = int(config["split"]["n_splits"])
    train_rows, valid_rows, valid_locations = make_group_kfold_split(
        rows, n_splits=n_splits, fold=fold, seed=seed
    )
    if bool(config["data"].get("drop_zero_obs_rows", False)):
        train_rows = drop_zero_observation_rows(train_rows)
    train_rows = sample_rows(train_rows, config["split"].get("max_train_samples"), seed)
    valid_rows = sample_rows(valid_rows, config["split"].get("max_valid_samples"), seed + 1)

    target_size = (int(config["data"]["target_height"]), int(config["data"]["target_width"]))
    train_ds = PrecipDataset(
        train_rows,
        train_dir,
        max_observations=int(config["data"]["max_observations"]),
        satellite_channels=int(config["data"]["satellite_channels"]),
        target_size=target_size,
        context_rows=int(config["data"].get("context_rows", 1)),
        has_target=True,
        norm_stats=norm_stats,
        augment=bool(config["train"].get("augment", True)),
        features=features_from_config(config),
        registration_shifts=registration_shifts,
    )
    valid_ds = PrecipDataset(
        valid_rows,
        train_dir,
        max_observations=int(config["data"]["max_observations"]),
        satellite_channels=int(config["data"]["satellite_channels"]),
        target_size=target_size,
        context_rows=int(config["data"].get("context_rows", 1)),
        has_target=True,
        norm_stats=norm_stats,
        augment=False,
        features=features_from_config(config),
        registration_shifts=registration_shifts,
    )

    expected = expected_in_channels(
        satellite_channels=int(config["data"]["satellite_channels"]),
        max_observations=int(config["data"]["max_observations"]),
        context_rows=int(config["data"].get("context_rows", 1)),
        features=features_from_config(config),
    )
    configured = int(config["model"]["in_channels"])
    if configured != expected:
        raise ValueError(
            f"model.in_channels={configured} but the data/features config produces {expected} "
            "channels — fix model.in_channels (see expected_in_channels in dataset.py)"
        )

    batch_size = int(config["train"]["batch_size"])
    num_workers = int(config["train"]["num_workers"])
    train_loader = DataLoader(
        train_ds,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_init_fn,
    )
    valid_loader = DataLoader(
        valid_ds,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers,
        pin_memory=True,
        persistent_workers=num_workers > 0,
        worker_init_fn=worker_init_fn,
    )

    device = torch.device("cuda")
    model = build_model(config)
    initialized_from: dict[str, Any] | None = None
    init_checkpoint_template = config["train"].get("init_checkpoint")
    if init_checkpoint_template:
        init_checkpoint_path = resolve_path(str(init_checkpoint_template).format(fold=fold))
        if not init_checkpoint_path.is_file():
            raise FileNotFoundError(f"initial checkpoint not found: {init_checkpoint_path}")
        source_checkpoint = torch.load(init_checkpoint_path, map_location="cpu")
        source_fold = int(source_checkpoint.get("fold", fold))
        if source_fold != fold:
            raise ValueError(
                f"initial checkpoint fold={source_fold} does not match requested fold={fold}: "
                f"{init_checkpoint_path}"
            )
        state_dict = source_checkpoint.get("model_state_dict", source_checkpoint)
        model.load_state_dict(state_dict, strict=True)
        initialized_from = {
            "path": str(init_checkpoint_path),
            "fold": source_fold,
            "best_metric": source_checkpoint.get("best_metric"),
            "selection_metric": source_checkpoint.get("selection_metric"),
        }
        del source_checkpoint
    model = model.to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    scaler = make_grad_scaler(enabled=bool(config["train"]["amp"]))
    loss_fn = build_loss(config)
    epochs = int(config["train"]["epochs"])
    scheduler_cfg = config.get("scheduler", {})
    scheduler = None
    if scheduler_cfg.get("name", "none") == "reduce_on_plateau":
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            optimizer,
            mode="min",
            factor=float(scheduler_cfg.get("factor", 0.3)),
            patience=int(scheduler_cfg.get("patience", 4)),
            min_lr=float(scheduler_cfg.get("min_lr", 1e-5)),
        )
    early_patience = int(config["train"].get("early_stopping_patience", 0))
    early_min_delta = float(config["train"].get("early_stopping_min_delta", 0.0))
    early_best = float("inf")
    epochs_without_improvement = 0
    stopped_early = False
    clip_min = float(config["model"]["clip_min"])
    selection_metric = metric_key(config)
    best_metric = float("inf")
    best_rmse = float("inf")
    best_tile_rmse = float("inf")
    best_epoch: int | None = None
    initial_metrics: dict[str, float] | None = None
    history: list[dict[str, Any]] = []

    checkpoint_path = model_dir / f"best_model_fold{fold}.pt"
    metrics_path = model_dir / f"metrics_fold{fold}.json"
    training_log_path = analysis_dir / f"training_log_fold{fold}.csv"
    training_log_fields = [
        "fold",
        "epoch",
        "train_rmse",
        "train_tile_rmse",
        "valid_rmse",
        "valid_tile_rmse",
        "zero_rmse",
        "positive_rmse",
        "samples",
        "pixels",
        "positive_pixels",
        "elapsed_seconds",
        "lr",
        "selection_metric",
        "best_metric_so_far",
    ]
    with training_log_path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=training_log_fields).writeheader()

    experiment_name = str(config.get("experiment", {}).get("name", "exp045"))
    print(
        f"{experiment_name} GPU train fold={fold}/{n_splits} "
        f"architecture={config['model'].get('architecture')} "
        f"train rows={len(train_ds)} valid rows={len(valid_ds)} valid_locations={valid_locations} "
        f"gpus={torch.cuda.device_count()}",
        flush=True,
    )
    if initialized_from is not None:
        initial_metrics = evaluate(model, valid_loader, device, clip_min=clip_min)
        best_metric = float(initial_metrics.get(selection_metric, initial_metrics["rmse"]))
        best_rmse = float(initial_metrics["rmse"])
        best_tile_rmse = float(initial_metrics["tile_rmse"])
        best_epoch = 0
        early_best = best_metric
        checkpoint = {
            "model_state_dict": unwrap_model(model).state_dict(),
            "config": config,
            "fold": fold,
            "valid_locations": valid_locations,
            "history": history,
            "initial_metrics": initial_metrics,
            "initialized_from": initialized_from,
            "best_epoch": best_epoch,
            "best_rmse": best_rmse,
            "best_tile_rmse": best_tile_rmse,
            "best_metric": best_metric,
            "selection_metric": selection_metric,
        }
        torch.save(checkpoint, checkpoint_path)
        print(
            json.dumps(
                {
                    "epoch": 0,
                    "phase": "initialized_checkpoint_baseline",
                    **initial_metrics,
                    "selection_metric": selection_metric,
                    "initialized_from": initialized_from,
                },
                indent=2,
            ),
            flush=True,
        )

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()
        train_loss_sum = 0.0
        train_pixels = 0
        train_tile_rmse_sum = 0.0
        train_samples = 0
        for step, batch in enumerate(train_loader, 1):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with cuda_autocast(enabled=bool(config["train"]["amp"])):
                output = model(x)
                pred = prediction_from_output(output)
                loss = loss_fn(output, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["train"]["grad_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                diff = pred.detach() - y
                mse = torch.square(diff).mean()
                tile_mse = torch.square(diff).flatten(1).mean(dim=1)
            train_loss_sum += float(mse.item()) * y.numel()
            train_pixels += int(y.numel())
            train_tile_rmse_sum += float(torch.sqrt(tile_mse).sum().item())
            train_samples += int(y.shape[0])
            if step % 20 == 0:
                print(
                    f"epoch={epoch} step={step}/{len(train_loader)} "
                    f"train_rmse={np.sqrt(train_loss_sum / train_pixels):.5f} "
                    f"train_tile_rmse={train_tile_rmse_sum / max(train_samples, 1):.5f}",
                    flush=True,
                )

        valid_metrics = evaluate(model, valid_loader, device, clip_min=clip_min)
        train_rmse = float(np.sqrt(train_loss_sum / train_pixels))
        train_tile_rmse = float(train_tile_rmse_sum / max(train_samples, 1))
        record = {
            "epoch": epoch,
            "train_rmse": train_rmse,
            "train_tile_rmse": train_tile_rmse,
            **valid_metrics,
            "elapsed_seconds": time.time() - epoch_start,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        print(json.dumps(record, indent=2), flush=True)

        current_metric = float(valid_metrics.get(selection_metric, valid_metrics["rmse"]))
        if current_metric < best_metric:
            best_metric = current_metric
            best_rmse = float(valid_metrics["rmse"])
            best_tile_rmse = float(valid_metrics["tile_rmse"])
            best_epoch = epoch
            checkpoint = {
                "model_state_dict": unwrap_model(model).state_dict(),
                "config": config,
                "fold": fold,
                "valid_locations": valid_locations,
                "history": history,
                "initial_metrics": initial_metrics,
                "initialized_from": initialized_from,
                "best_epoch": best_epoch,
                "best_rmse": best_rmse,
                "best_tile_rmse": best_tile_rmse,
                "best_metric": best_metric,
                "selection_metric": selection_metric,
            }
            torch.save(checkpoint, checkpoint_path)

        if scheduler is not None:
            scheduler.step(current_metric)
        if current_metric < early_best - early_min_delta:
            early_best = current_metric
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1

        with training_log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=training_log_fields)
            writer.writerow(
                {
                    "fold": fold,
                    "epoch": epoch,
                    "train_rmse": train_rmse,
                    "train_tile_rmse": train_tile_rmse,
                    "valid_rmse": valid_metrics["rmse"],
                    "valid_tile_rmse": valid_metrics["tile_rmse"],
                    "zero_rmse": valid_metrics["zero_rmse"],
                    "positive_rmse": valid_metrics["positive_rmse"],
                    "samples": valid_metrics["samples"],
                    "pixels": valid_metrics["pixels"],
                    "positive_pixels": valid_metrics["positive_pixels"],
                    "elapsed_seconds": record["elapsed_seconds"],
                    "lr": record["lr"],
                    "selection_metric": selection_metric,
                    "best_metric_so_far": best_metric,
                }
            )

        if early_patience > 0 and epochs_without_improvement >= early_patience:
            stopped_early = True
            print(
                f"early stopping at epoch={epoch}: no improvement > {early_min_delta} "
                f"for {early_patience} epochs",
                flush=True,
            )
            break

    metrics_path.write_text(
        json.dumps(
            {
                "fold": fold,
                "valid_locations": valid_locations,
                "history": history,
                "best_rmse": best_rmse,
                "best_tile_rmse": best_tile_rmse,
                "best_metric": best_metric,
                "best_epoch": best_epoch,
                "selection_metric": selection_metric,
                "initial_metrics": initial_metrics,
                "initialized_from": initialized_from,
                "stopped_early": stopped_early,
                "epochs_completed": len(history),
                "train_rows_used": len(train_ds),
                "valid_rows_used": len(valid_ds),
                "torch_version": torch.__version__,
                "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
                "training_log": str(training_log_path),
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(
        f"saved best model to {checkpoint_path} "
        f"selection_metric={selection_metric} best_metric={best_metric:.6f} "
        f"best_rmse={best_rmse:.6f} best_tile_rmse={best_tile_rmse:.6f}",
        flush=True,
    )


if __name__ == "__main__":
    main()
