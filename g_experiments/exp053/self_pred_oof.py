#!/usr/bin/env python3
"""Second OOF pass for exp053's autoregressive channel -- the gating criterion.

train.py's normal `evaluate()` computes tile_rmse under TEACHER FORCING: for each validation
row, the AR input channel is the SAME (name_location, T-30min) row's TRUE ground-truth GPM
value (dataset.py's `ar_cache is None` branch). That number (saved as `best_tile_rmse` in
g_model/exp053/metrics_fold{N}.json) is optimistic -- at real evaluation time no ground truth
for T-30min ever exists (it's exactly what's being predicted for that row too), so the model
must condition on its OWN earlier prediction instead. This creates a train/inference mismatch
("exposure bias": the model never saw its own imperfect predictions as input during training).

This script mimics real inference exactly: validation rows are processed in ascending-datetime
"steps" grouped by name_location (see inference.py's `run_autoregressive_inference`, whose
sequencing logic this mirrors), and the model's own just-computed prediction for T-30min is
substituted for the AR input feature at T (dataset.py's `ar_cache is not None` branch), even
though this dataset also carries has_target=True (needed so we can still read the TRUE target
for the row being scored -- just not for the AR *input* feature of that row).

No TTA / calibration / value-threshold post-processing is applied (single checkpoint, single
forward pass, clamp_min only) so the reported tile_rmse/rmse are directly comparable to
train.py's evaluate() and the doc/submission_registry.md gate baseline (exp038 strict:
fold0 tile_rmse=0.28954, fold4 tile_rmse=0.59607).

Usage:
    python self_pred_oof.py --config config.yaml --fold 0
    python self_pred_oof.py --config config.yaml --fold 4
"""

from __future__ import annotations

import argparse
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml

from amp_utils import cuda_autocast
from dataset import (
    PrecipDataset,
    features_from_config,
    load_norm_stats,
    make_group_kfold_split,
    read_rows,
    read_target_tensor,
)
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


def build_location_sequences(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    sequences: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        sequences.setdefault(row["name_location"], []).append(row)
    for loc_rows in sequences.values():
        loc_rows.sort(key=lambda r: datetime.fromisoformat(r["datetime"]))
    return sequences


@torch.no_grad()
def run_self_prediction_substituted_pass(
    ds: PrecipDataset,
    rows: list[dict[str, str]],
    model: torch.nn.Module,
    device: torch.device,
    clip_min: float,
    amp: bool,
) -> dict[str, float]:
    """Sequential per-location pass that scores each row against its TRUE target while
    sourcing the AR input feature purely from this pass's own earlier predictions (never
    ground truth) -- i.e. the same exposure-bias-affected regime real evaluation will see."""
    sequences = build_location_sequences(rows)
    ar_cache: dict[tuple[str, datetime], np.ndarray] = {}
    ds.ar_cache = ar_cache  # forces dataset.py's self-prediction branch even though has_target=True

    max_len = max((len(seq) for seq in sequences.values()), default=0)
    sse = 0.0
    pixels = 0
    tile_rmse_sum = 0.0
    samples = 0
    start = time.time()
    for step in range(max_len):
        step_rows = [seq[step] for seq in sequences.values() if step < len(seq)]
        if not step_rows:
            continue
        x = torch.stack([ds.input_tensor(row) for row in step_rows]).to(device, non_blocking=True)
        y = torch.stack(
            [
                read_target_tensor(ds.data_dir / "gpm_imerg" / row["gpm_imerg_filename"])
                for row in step_rows
            ]
        ).to(device, non_blocking=True)
        with cuda_autocast(enabled=amp):
            output = model(x)
        pred = prediction_from_output(output).float().clamp_min(clip_min)
        diff = pred - y
        sse += float(torch.square(diff).sum().item())
        pixels += int(y.numel())
        tile_mse = torch.square(diff).flatten(1).mean(dim=1)
        tile_rmse_sum += float(torch.sqrt(tile_mse).sum().item())
        samples += int(y.shape[0])

        pred_np = pred.detach().cpu().numpy().astype(np.float32)
        for i, row in enumerate(step_rows):
            location = row["name_location"]
            row_time = datetime.fromisoformat(row["datetime"])
            ar_cache[(location, row_time)] = pred_np[i, 0].copy()

        if step % 50 == 0 or step == max_len - 1:
            elapsed = time.time() - start
            print(
                f"self-pred OOF step={step}/{max_len - 1} rows_done={samples}/{len(rows)} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )

    return {
        "rmse": float(np.sqrt(sse / pixels)) if pixels else float("nan"),
        "tile_rmse": float(tile_rmse_sum / max(samples, 1)),
        "samples": float(samples),
        "pixels": float(pixels),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"))
    parser.add_argument("--fold", type=int, default=None, help="Override split.fold from config")
    parser.add_argument("--checkpoint", default=None, help="Override checkpoint path")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    fold = args.fold if args.fold is not None else int(config["split"]["fold"])

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run this script outside the sandbox with GPU access.")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    features = features_from_config(config)
    if not features.get("autoregressive_prev_pred"):
        raise ValueError(
            "self_pred_oof.py is only meaningful when features.autoregressive_prev_pred=true "
            "(nothing to substitute otherwise)"
        )

    train_csv = resolve_path(config["data"]["train_csv"])
    train_dir = resolve_path(config["data"]["train_dir"])
    norm_stats_path = resolve_path(config["paths"]["norm_stats"])
    norm_stats = load_norm_stats(norm_stats_path)

    rows = read_rows(train_csv)
    n_splits = int(config["split"]["n_splits"])
    seed = int(config["experiment"]["seed"])
    # NOTE: matches train.py's split exactly. drop_zero_obs_rows is applied only to TRAIN rows
    # there (never to valid_rows), so valid_rows here is identical to what train.py validates
    # against -- this OOF pass and the teacher-forced number in metrics_fold{N}.json score the
    # same row set.
    _, valid_rows, valid_locations = make_group_kfold_split(rows, n_splits=n_splits, fold=fold, seed=seed)

    target_size = (int(config["data"]["target_height"]), int(config["data"]["target_width"]))
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
        features=features,
    )

    model_dir = resolve_path(config["paths"]["model_dir"])
    checkpoint_path = Path(args.checkpoint) if args.checkpoint else model_dir / f"best_model_fold{fold}.pt"
    if not checkpoint_path.is_file():
        raise FileNotFoundError(f"checkpoint not found: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    device = torch.device("cuda")
    model = model.to(device).eval()

    clip_min = float(config["model"]["clip_min"])
    amp = bool(config["train"]["amp"])

    print(
        f"exp053 self-prediction-substituted OOF fold={fold} valid_rows={len(valid_rows)} "
        f"valid_locations={valid_locations} checkpoint={checkpoint_path} "
        f"teacher_forced_best_tile_rmse={checkpoint.get('best_tile_rmse')}",
        flush=True,
    )
    metrics = run_self_prediction_substituted_pass(valid_ds, valid_rows, model, device, clip_min, amp)

    result = {
        "fold": fold,
        "valid_locations": valid_locations,
        "valid_rows": len(valid_rows),
        "checkpoint": str(checkpoint_path),
        "teacher_forced_best_tile_rmse": checkpoint.get("best_tile_rmse"),
        "teacher_forced_best_rmse": checkpoint.get("best_rmse"),
        "self_prediction_substituted": metrics,
    }
    print(json.dumps(result, indent=2), flush=True)

    analysis_dir = resolve_path(config["paths"].get("analysis_dir", "../../outputs/analysis/exp053"))
    analysis_dir.mkdir(parents=True, exist_ok=True)
    out_path = analysis_dir / f"self_pred_oof_fold{fold}.json"
    out_path.write_text(json.dumps(result, indent=2), encoding="utf-8")
    print(f"wrote {out_path}", flush=True)


if __name__ == "__main__":
    main()
