#!/usr/bin/env python3
"""Train exp003: normalized inputs + weighted loss + diagnostics."""

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
from dataset import PrecipDataset, load_norm_stats, make_group_kfold_split, read_rows, sample_rows
from losses import WeightedMSELoss
from model import build_model

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
    for batch in loader:
        x = batch["x"].to(device, non_blocking=True)
        y = batch["y"].to(device, non_blocking=True)
        pred = model(x).clamp_min(clip_min)
        diff = pred - y
        sse += float(torch.square(diff).sum().item())
        zero_sse += float(torch.square(y).sum().item())
        pixels += int(y.numel())
        mask = y > 0
        if mask.any():
            positive_sse += float(torch.square(diff[mask]).sum().item())
            positive_pixels += int(mask.sum().item())
    return {
        "rmse": float(np.sqrt(sse / pixels)),
        "zero_rmse": float(np.sqrt(zero_sse / pixels)),
        "positive_rmse": float(np.sqrt(positive_sse / positive_pixels)) if positive_pixels else float("nan"),
        "pixels": float(pixels),
        "positive_pixels": float(positive_pixels),
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
    analysis_dir = resolve_path(config["paths"].get("analysis_dir", "../../outputs/analysis/exp003"))
    analysis_dir.mkdir(parents=True, exist_ok=True)
    norm_stats_path = resolve_path(config["paths"]["norm_stats"])
    if not norm_stats_path.exists():
        raise FileNotFoundError(
            f"{norm_stats_path} not found. Run `python normalize_stats.py --config {args.config}` first."
        )
    norm_stats = load_norm_stats(norm_stats_path)

    rows = read_rows(train_csv)
    n_splits = int(config["split"]["n_splits"])
    train_rows, valid_rows, valid_locations = make_group_kfold_split(
        rows, n_splits=n_splits, fold=fold, seed=seed
    )
    train_rows = sample_rows(train_rows, config["split"].get("max_train_samples"), seed)
    valid_rows = sample_rows(valid_rows, config["split"].get("max_valid_samples"), seed + 1)

    target_size = (int(config["data"]["target_height"]), int(config["data"]["target_width"]))
    train_ds = PrecipDataset(
        train_rows,
        train_dir,
        max_observations=int(config["data"]["max_observations"]),
        satellite_channels=int(config["data"]["satellite_channels"]),
        target_size=target_size,
        has_target=True,
        norm_stats=norm_stats,
        augment=bool(config["train"].get("augment", True)),
    )
    valid_ds = PrecipDataset(
        valid_rows,
        train_dir,
        max_observations=int(config["data"]["max_observations"]),
        satellite_channels=int(config["data"]["satellite_channels"]),
        target_size=target_size,
        has_target=True,
        norm_stats=norm_stats,
        augment=False,
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
    model = build_model(config).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    scaler = make_grad_scaler(enabled=bool(config["train"]["amp"]))
    loss_fn = WeightedMSELoss(pos_weight=float(config["loss"]["pos_weight"]))
    epochs = int(config["train"]["epochs"])
    clip_min = float(config["model"]["clip_min"])
    best_rmse = float("inf")
    history: list[dict[str, Any]] = []

    checkpoint_path = model_dir / f"best_model_fold{fold}.pt"
    metrics_path = model_dir / f"metrics_fold{fold}.json"
    training_log_path = analysis_dir / f"training_log_fold{fold}.csv"
    training_log_fields = [
        "fold",
        "epoch",
        "train_rmse",
        "valid_rmse",
        "zero_rmse",
        "positive_rmse",
        "pixels",
        "positive_pixels",
        "elapsed_seconds",
        "lr",
        "best_rmse_so_far",
    ]
    with training_log_path.open("w", newline="", encoding="utf-8") as f:
        csv.DictWriter(f, fieldnames=training_log_fields).writeheader()

    print(
        f"exp003 GPU train fold={fold}/{n_splits} architecture={config['model'].get('architecture')} "
        f"train rows={len(train_ds)} valid rows={len(valid_ds)} valid_locations={valid_locations} "
        f"gpus={torch.cuda.device_count()}",
        flush=True,
    )

    for epoch in range(1, epochs + 1):
        model.train()
        epoch_start = time.time()
        train_loss_sum = 0.0
        train_pixels = 0
        for step, batch in enumerate(train_loader, 1):
            x = batch["x"].to(device, non_blocking=True)
            y = batch["y"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with cuda_autocast(enabled=bool(config["train"]["amp"])):
                pred = model(x)
                loss = loss_fn(pred, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["train"]["grad_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            with torch.no_grad():
                mse = torch.square(pred.detach() - y).mean()
            train_loss_sum += float(mse.item()) * y.numel()
            train_pixels += int(y.numel())
            if step % 20 == 0:
                print(
                    f"epoch={epoch} step={step}/{len(train_loader)} "
                    f"train_rmse={np.sqrt(train_loss_sum / train_pixels):.5f}",
                    flush=True,
                )

        valid_metrics = evaluate(model, valid_loader, device, clip_min=clip_min)
        train_rmse = float(np.sqrt(train_loss_sum / train_pixels))
        record = {
            "epoch": epoch,
            "train_rmse": train_rmse,
            **valid_metrics,
            "elapsed_seconds": time.time() - epoch_start,
            "lr": float(optimizer.param_groups[0]["lr"]),
        }
        history.append(record)
        print(json.dumps(record, indent=2), flush=True)

        if valid_metrics["rmse"] < best_rmse:
            best_rmse = valid_metrics["rmse"]
            checkpoint = {
                "model_state_dict": unwrap_model(model).state_dict(),
                "config": config,
                "fold": fold,
                "valid_locations": valid_locations,
                "history": history,
                "best_rmse": best_rmse,
            }
            torch.save(checkpoint, checkpoint_path)

        with training_log_path.open("a", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=training_log_fields)
            writer.writerow(
                {
                    "fold": fold,
                    "epoch": epoch,
                    "train_rmse": train_rmse,
                    "valid_rmse": valid_metrics["rmse"],
                    "zero_rmse": valid_metrics["zero_rmse"],
                    "positive_rmse": valid_metrics["positive_rmse"],
                    "pixels": valid_metrics["pixels"],
                    "positive_pixels": valid_metrics["positive_pixels"],
                    "elapsed_seconds": record["elapsed_seconds"],
                    "lr": record["lr"],
                    "best_rmse_so_far": best_rmse,
                }
            )

    metrics_path.write_text(
        json.dumps(
            {
                "fold": fold,
                "valid_locations": valid_locations,
                "history": history,
                "best_rmse": best_rmse,
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
    print(f"saved best model to {checkpoint_path} best_rmse={best_rmse:.6f}", flush=True)


if __name__ == "__main__":
    main()
