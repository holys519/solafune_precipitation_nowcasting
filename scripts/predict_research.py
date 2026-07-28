#!/usr/bin/env python3
"""Fold-ensemble inference with metadata-preserving GeoTIFF output."""

from __future__ import annotations

import argparse
import csv
import shutil
from pathlib import Path

import numpy as np
import rasterio
import torch
from torch.utils.data import DataLoader

from precip_nowcast.config import load_config
from precip_nowcast.data import NormalizationStats, PrecipitationDataset, read_rows
from precip_nowcast.model import TotalShapeNowcaster


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--checkpoints", type=Path, nargs="+", required=True)
    parser.add_argument("--submission-dir", type=Path, required=True)
    args = parser.parse_args()
    cfg = load_config(args.config)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rows = read_rows(cfg.data.evaluation_csv)
    dataset = PrecipitationDataset(
        rows,
        root=cfg.data.evaluation_dir,
        stats=NormalizationStats.load(cfg.data.norm_stats),
        input_size=cfg.data.input_size,
        output_size=cfg.data.output_size,
        max_observations=cfg.data.max_observations,
        satellite_channels=cfg.data.satellite_channels,
        has_target=False,
        augment=False,
    )
    loader = DataLoader(
        dataset,
        batch_size=cfg.train.batch_size,
        num_workers=cfg.train.num_workers,
        pin_memory=device.type == "cuda",
        shuffle=False,
    )
    models = []
    for path in args.checkpoints:
        model = TotalShapeNowcaster(
            in_channels=dataset.in_channels,
            base_channels=cfg.model.base_channels,
            total_hidden=cfg.model.total_hidden,
            dropout=cfg.model.dropout,
            output_size=cfg.model.output_size,
        )
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint["model"])
        models.append(model.to(device).eval())

    output_files = args.submission_dir / "test_files"
    output_files.mkdir(parents=True, exist_ok=True)
    cursor = 0
    with torch.inference_mode():
        for batch in loader:
            x = batch["x"].to(device)
            prediction = torch.stack([model(x).prediction for model in models]).mean(0)
            arrays = prediction.clamp_min(0).cpu().numpy()[:, 0]
            for array in arrays:
                row = rows[cursor]
                filename = row["gpm_imerg_filename"]
                template = cfg.data.evaluation_dir / "test_files" / filename
                with rasterio.open(template) as source:
                    profile = source.profile.copy()
                profile.update(dtype="float32", count=1)
                with rasterio.open(output_files / filename, "w", **profile) as destination:
                    destination.write(np.asarray(array, dtype=np.float32), 1)
                cursor += 1
    shutil.copy2(cfg.data.evaluation_csv, args.submission_dir / cfg.data.evaluation_csv.name)
    manifest = args.submission_dir / "inference_manifest.csv"
    with manifest.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(("checkpoint",))
        writer.writerows((str(path.resolve()),) for path in args.checkpoints)


if __name__ == "__main__":
    main()
