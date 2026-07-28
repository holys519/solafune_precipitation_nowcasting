#!/usr/bin/env python3
"""Train one fold of the shared precipitation-nowcasting pipeline."""

from __future__ import annotations

import argparse
import time
from dataclasses import asdict
from pathlib import Path

import torch
from torch.utils.data import DataLoader

from precip_nowcast.config import load_config
from precip_nowcast.data import (
    NormalizationStats,
    PrecipitationDataset,
    has_observation,
    make_balanced_group_folds,
    read_rows,
)
from precip_nowcast.losses import CompositePrecipitationLoss
from precip_nowcast.metrics import MetricAccumulator
from precip_nowcast.model import TotalShapeNowcaster
from precip_nowcast.utils import atomic_torch_save, seed_everything, worker_init_fn, write_json


@torch.inference_mode()
def evaluate(
    model: TotalShapeNowcaster, loader: DataLoader, device: torch.device
) -> dict[str, float]:
    model.eval()
    metrics = MetricAccumulator()
    for batch in loader:
        target = batch["y"].to(device, non_blocking=True)
        output = model(batch["x"].to(device, non_blocking=True))
        metrics.update(output.prediction.clamp_min(0), target)
    return metrics.compute()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--fold", type=int)
    args = parser.parse_args()
    cfg = load_config(args.config)
    fold = cfg.split.fold if args.fold is None else args.fold
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required for full-resolution training")
    seed_everything(cfg.split.seed)
    device = torch.device("cuda")

    rows = read_rows(cfg.data.train_csv)
    if cfg.data.drop_zero_observation_rows:
        rows = [row for row in rows if has_observation(row)]
    folds = make_balanced_group_folds(rows, cfg.split.n_splits, cfg.split.seed)
    train_rows = [row for row in rows if folds[row["name_location"]] != fold]
    valid_rows = [row for row in rows if folds[row["name_location"]] == fold]
    if not train_rows or not valid_rows:
        raise RuntimeError(f"empty split: train={len(train_rows)}, valid={len(valid_rows)}")

    stats = NormalizationStats.load(cfg.data.norm_stats)
    common = dict(
        root=cfg.data.train_dir,
        stats=stats,
        input_size=cfg.data.input_size,
        output_size=cfg.data.output_size,
        max_observations=cfg.data.max_observations,
        satellite_channels=cfg.data.satellite_channels,
        has_target=True,
    )
    train_ds = PrecipitationDataset(train_rows, augment=True, **common)
    valid_ds = PrecipitationDataset(valid_rows, augment=False, **common)
    loader_options = dict(
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        persistent_workers=cfg.train.num_workers > 0,
    )
    generator = torch.Generator().manual_seed(cfg.split.seed)
    train_loader = DataLoader(train_ds, shuffle=True, generator=generator, **loader_options)
    valid_loader = DataLoader(valid_ds, shuffle=False, **loader_options)

    model = TotalShapeNowcaster(
        in_channels=train_ds.in_channels,
        base_channels=cfg.model.base_channels,
        total_hidden=cfg.model.total_hidden,
        dropout=cfg.model.dropout,
        output_size=cfg.model.output_size,
    ).to(device)
    criterion = CompositePrecipitationLoss(cfg.loss)
    optimizer = torch.optim.AdamW(
        model.parameters(), lr=cfg.train.lr, weight_decay=cfg.train.weight_decay
    )
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg.train.epochs)
    scaler = torch.amp.GradScaler("cuda", enabled=cfg.train.amp)
    checkpoint_path = cfg.output_dir / f"fold{fold}" / "best.pt"
    history: list[dict[str, float]] = []
    best = float("inf")
    stale_epochs = 0

    for epoch in range(1, cfg.train.epochs + 1):
        started = time.time()
        model.train()
        loss_sum = 0.0
        sample_count = 0
        for batch in train_loader:
            x = batch["x"].to(device, non_blocking=True)
            target = batch["y"].to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.autocast("cuda", enabled=cfg.train.amp):
                output = model(x)
                loss, _ = criterion(output, target)
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.train.grad_clip_norm)
            scaler.step(optimizer)
            scaler.update()
            loss_sum += float(loss.detach()) * x.shape[0]
            sample_count += x.shape[0]
        scheduler.step()
        valid = evaluate(model, valid_loader, device)
        record = {
            "epoch": float(epoch),
            "train_loss": loss_sum / sample_count,
            **valid,
            "lr": optimizer.param_groups[0]["lr"],
            "seconds": time.time() - started,
        }
        history.append(record)
        write_json(history, cfg.output_dir / f"fold{fold}" / "history.json")
        print(record, flush=True)
        if valid["tile_rmse"] < best:
            best = valid["tile_rmse"]
            stale_epochs = 0
            atomic_torch_save(
                {
                    "model": model.state_dict(),
                    "config": asdict(cfg),
                    "fold": fold,
                    "valid_locations": sorted(
                        location for location, assigned in folds.items() if assigned == fold
                    ),
                    "metrics": valid,
                    "epoch": epoch,
                },
                checkpoint_path,
            )
        else:
            stale_epochs += 1
            if stale_epochs >= cfg.train.patience:
                break


if __name__ == "__main__":
    main()
