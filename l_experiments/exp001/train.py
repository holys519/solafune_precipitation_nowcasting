#!/usr/bin/env python3
"""Train exp001 GPU baseline."""

from __future__ import annotations

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

from dataset import PrecipDataset, make_location_split, read_rows, sample_rows
from model import CompactUNet


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
    config = load_config(SCRIPT_DIR / "config.yaml")
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

    rows = read_rows(train_csv)
    train_rows, valid_rows, valid_locations = make_location_split(
        rows,
        valid_fraction=float(config["split"]["valid_fraction_locations"]),
        seed=seed,
    )
    train_rows = sample_rows(train_rows, int(config["split"]["max_train_samples"]), seed)
    valid_rows = sample_rows(valid_rows, int(config["split"]["max_valid_samples"]), seed + 1)

    target_size = (int(config["data"]["target_height"]), int(config["data"]["target_width"]))
    train_ds = PrecipDataset(
        train_rows,
        train_dir,
        max_observations=int(config["data"]["max_observations"]),
        satellite_channels=int(config["data"]["satellite_channels"]),
        target_size=target_size,
        has_target=True,
    )
    valid_ds = PrecipDataset(
        valid_rows,
        train_dir,
        max_observations=int(config["data"]["max_observations"]),
        satellite_channels=int(config["data"]["satellite_channels"]),
        target_size=target_size,
        has_target=True,
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
    model = CompactUNet(
        in_channels=int(config["model"]["in_channels"]),
        base_channels=int(config["model"]["base_channels"]),
    ).to(device)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    optimizer = torch.optim.AdamW(
        model.parameters(),
        lr=float(config["train"]["lr"]),
        weight_decay=float(config["train"]["weight_decay"]),
    )
    scaler = torch.amp.GradScaler("cuda", enabled=bool(config["train"]["amp"]))
    loss_fn = nn.MSELoss()
    epochs = int(config["train"]["epochs"])
    clip_min = float(config["model"]["clip_min"])
    best_rmse = float("inf")
    history: list[dict[str, Any]] = []

    print(
        f"exp001 GPU train rows={len(train_ds)} valid rows={len(valid_ds)} "
        f"valid_locations={valid_locations} gpus={torch.cuda.device_count()}",
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
            with torch.amp.autocast("cuda", enabled=bool(config["train"]["amp"])):
                pred = model(x)
                loss = loss_fn(pred, y)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), float(config["train"]["grad_clip_norm"]))
            scaler.step(optimizer)
            scaler.update()
            train_loss_sum += float(loss.item()) * y.numel()
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
        }
        history.append(record)
        print(json.dumps(record, indent=2), flush=True)

        if valid_metrics["rmse"] < best_rmse:
            best_rmse = valid_metrics["rmse"]
            checkpoint = {
                "model_state_dict": unwrap_model(model).state_dict(),
                "config": config,
                "valid_locations": valid_locations,
                "history": history,
                "best_rmse": best_rmse,
            }
            torch.save(checkpoint, model_dir / "best_model.pt")

    (model_dir / "metrics.json").write_text(
        json.dumps(
            {
                "valid_locations": valid_locations,
                "history": history,
                "best_rmse": best_rmse,
                "train_rows_used": len(train_ds),
                "valid_rows_used": len(valid_ds),
                "torch_version": torch.__version__,
                "cuda_devices": [torch.cuda.get_device_name(i) for i in range(torch.cuda.device_count())],
            },
            indent=2,
            ensure_ascii=False,
        ),
        encoding="utf-8",
    )
    print(f"saved best model to {model_dir / 'best_model.pt'} best_rmse={best_rmse:.6f}", flush=True)


if __name__ == "__main__":
    main()

