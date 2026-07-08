#!/usr/bin/env python3
"""Run GPU inference for exp001 and write prediction GeoTIFFs."""

from __future__ import annotations

import csv
import json
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from amp_utils import cuda_autocast
from dataset import PrecipDataset, read_rows
from model import CompactUNet
from tiff_utils import write_float32_like_template


SCRIPT_DIR = Path(__file__).resolve().parent


def load_config(path: Path) -> dict[str, Any]:
    with path.open() as f:
        return yaml.safe_load(f)


def resolve_path(path: str) -> Path:
    p = Path(path)
    if p.is_absolute():
        return p
    return (SCRIPT_DIR / p).resolve()


def worker_init_fn(worker_id: int) -> None:
    torch.set_num_threads(1)


def copy_csv(src: Path, dst: Path) -> None:
    dst.parent.mkdir(parents=True, exist_ok=True)
    dst.write_bytes(src.read_bytes())


@torch.no_grad()
def main() -> None:
    config = load_config(SCRIPT_DIR / "config.yaml")
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run this script outside the sandbox with GPU access.")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    eval_csv = resolve_path(config["data"]["evaluation_csv"])
    eval_dir = resolve_path(config["data"]["evaluation_dir"])
    sample_submission_dir = resolve_path(config["data"]["sample_submission_dir"])
    model_dir = resolve_path(config["paths"]["model_dir"])
    output_dir = resolve_path(config["paths"]["output_dir"])
    prediction_dir = output_dir / "test_files"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    copy_csv(eval_csv, output_dir / "evaluation_target.csv")

    checkpoint = torch.load(model_dir / "best_model.pt", map_location="cpu")
    model = CompactUNet(
        in_channels=int(config["model"]["in_channels"]),
        base_channels=int(config["model"]["base_channels"]),
    )
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.cuda().eval()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)

    rows = read_rows(eval_csv)
    target_size = (int(config["data"]["target_height"]), int(config["data"]["target_width"]))
    ds = PrecipDataset(
        rows,
        eval_dir,
        max_observations=int(config["data"]["max_observations"]),
        satellite_channels=int(config["data"]["satellite_channels"]),
        target_size=target_size,
        has_target=False,
    )
    loader = DataLoader(
        ds,
        batch_size=int(config["train"]["batch_size"]),
        shuffle=False,
        num_workers=int(config["train"]["num_workers"]),
        pin_memory=True,
        persistent_workers=int(config["train"]["num_workers"]) > 0,
        worker_init_fn=worker_init_fn,
    )

    clip_min = float(config["model"]["clip_min"])
    start = time.time()
    written = 0
    for batch in loader:
        x = batch["x"].cuda(non_blocking=True)
        with cuda_autocast(enabled=bool(config["train"]["amp"])):
            pred = model(x).clamp_min(clip_min)
        pred_np = pred.detach().cpu().numpy().astype(np.float32)
        filenames = batch["gpm_imerg_filename"]
        for i, filename in enumerate(filenames):
            name = str(filename)
            template = eval_dir / "test_files" / name
            if not template.exists():
                template = sample_submission_dir / "test_files" / name
            array = np.nan_to_num(pred_np[i, 0], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            write_float32_like_template(template, prediction_dir / name, array)
            written += 1
        if written % 2048 < len(filenames):
            elapsed = time.time() - start
            print(f"inference {written}/{len(ds)} elapsed={elapsed:.1f}s", flush=True)

    summary = {
        "rows": len(ds),
        "prediction_dir": str(prediction_dir),
        "checkpoint_best_rmse": checkpoint.get("best_rmse"),
        "elapsed_seconds": time.time() - start,
    }
    (output_dir / "inference_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote predictions: {prediction_dir} files={written}", flush=True)


if __name__ == "__main__":
    main()
