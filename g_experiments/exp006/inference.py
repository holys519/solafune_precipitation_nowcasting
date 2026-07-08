#!/usr/bin/env python3
"""Run GPU inference for exp006 and write prediction GeoTIFFs.

Supports:
- multi-checkpoint ensembling (average raw predictions across --checkpoint args, e.g. one
  per GroupKFold fold), before the single non-negative clip at the end.
- optional flip TTA (config `tta.enabled`/`tta.flip`): averages predictions over identity,
  horizontal-flip, and vertical-flip views of the input (flipped back before averaging).
"""

from __future__ import annotations

import argparse
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
from dataset import PrecipDataset, load_norm_stats, read_rows
from model import build_model, prediction_from_output
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


def load_calibration(path: Path | None) -> dict[str, float] | None:
    if path is None or not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    return {
        "scale": float(raw.get("scale", 1.0)),
        "bias": float(raw.get("bias", 0.0)),
        "threshold": float(raw.get("threshold", 0.0)),
        "rain_prob_threshold": float(raw.get("rain_prob_threshold", 0.0)),
    }


def apply_postprocess(pred: torch.Tensor, clip_min: float, calibration: dict[str, float] | None) -> torch.Tensor:
    if calibration is not None:
        pred = pred * calibration["scale"] + calibration["bias"]
        threshold = calibration.get("threshold", 0.0)
        if threshold > 0:
            pred = torch.where(pred < threshold, torch.zeros_like(pred), pred)
    return pred.clamp_min(clip_min)


def load_models(config: dict[str, Any], checkpoint_paths: list[Path]) -> list[nn.Module]:
    models = []
    for path in checkpoint_paths:
        checkpoint = torch.load(path, map_location="cpu", weights_only=False)
        model = build_model(config)
        model.load_state_dict(checkpoint["model_state_dict"])
        model = model.cuda().eval()
        if torch.cuda.device_count() > 1:
            model = nn.DataParallel(model)
        models.append(model)
        print(f"loaded {path} best_rmse={checkpoint.get('best_rmse')}", flush=True)
    return models


@torch.no_grad()
def predict_ensemble(
    models: list[nn.Module],
    x: torch.Tensor,
    amp: bool,
    use_flip_tta: bool,
) -> dict[str, torch.Tensor]:
    views: list[tuple[torch.Tensor, tuple[int, ...] | None]] = [(x, None)]
    if use_flip_tta:
        views.append((torch.flip(x, dims=(-1,)), (-1,)))
        views.append((torch.flip(x, dims=(-2,)), (-2,)))

    total: dict[str, torch.Tensor] = {}
    count = 0
    for model in models:
        for view, flip_dims in views:
            with cuda_autocast(enabled=amp):
                output = model(view)
            pred = prediction_from_output(output)
            if flip_dims is not None:
                pred = torch.flip(pred, dims=flip_dims)
            pred = pred.float()
            total["pred"] = pred if "pred" not in total else total["pred"] + pred
            if isinstance(output, dict) and "rain_prob" in output:
                rain_prob = output["rain_prob"]
                if flip_dims is not None:
                    rain_prob = torch.flip(rain_prob, dims=flip_dims)
                rain_prob = rain_prob.float()
                total["rain_prob"] = rain_prob if "rain_prob" not in total else total["rain_prob"] + rain_prob
            count += 1
    return {key: value / count for key, value in total.items()}


@torch.no_grad()
def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"))
    parser.add_argument(
        "--checkpoint",
        action="append",
        default=None,
        help="Checkpoint path(s) to ensemble. Repeat for multiple folds. "
        "Default: paths.model_dir/best_model_fold<split.fold>.pt",
    )
    parser.add_argument("--calibration", default=None, help="Optional OOF calibration JSON.")
    parser.add_argument("--use-calibration", action="store_true", help="Apply --calibration or config calibration.")
    args = parser.parse_args()

    config = load_config(Path(args.config))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is not available. Run this script outside the sandbox with GPU access.")
    torch.backends.cudnn.benchmark = True
    torch.set_float32_matmul_precision("high")

    eval_csv = resolve_path(config["data"]["evaluation_csv"])
    eval_dir = resolve_path(config["data"]["evaluation_dir"])
    sample_submission_dir = resolve_path(config["data"]["sample_submission_dir"])
    model_dir = resolve_path(config["paths"]["model_dir"])
    output_dir = resolve_path(config["paths"]["output_dir"])
    analysis_dir = resolve_path(config["paths"].get("analysis_dir", str(output_dir)))
    norm_stats_path = resolve_path(config["paths"]["norm_stats"])
    prediction_dir = output_dir / "test_files"
    prediction_dir.mkdir(parents=True, exist_ok=True)
    analysis_dir.mkdir(parents=True, exist_ok=True)
    copy_csv(eval_csv, output_dir / "evaluation_target.csv")

    norm_stats = load_norm_stats(norm_stats_path)

    if args.checkpoint:
        checkpoint_paths = [Path(p) for p in args.checkpoint]
    else:
        checkpoint_paths = [model_dir / f"best_model_fold{int(config['split']['fold'])}.pt"]
    models = load_models(config, checkpoint_paths)

    rows = read_rows(eval_csv)
    target_size = (int(config["data"]["target_height"]), int(config["data"]["target_width"]))
    ds = PrecipDataset(
        rows,
        eval_dir,
        max_observations=int(config["data"]["max_observations"]),
        satellite_channels=int(config["data"]["satellite_channels"]),
        target_size=target_size,
        has_target=False,
        norm_stats=norm_stats,
        augment=False,
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
    use_flip_tta = bool(config["tta"]["enabled"]) and bool(config["tta"].get("flip", True))
    amp = bool(config["train"]["amp"])
    post_cfg = config.get("postprocess", {})
    calibration_path = None
    if args.calibration:
        calibration_path = Path(args.calibration)
    elif post_cfg.get("calibration_path"):
        calibration_path = resolve_path(post_cfg["calibration_path"])
    use_calibration = bool(args.use_calibration or post_cfg.get("use_oof_calibration", False))
    calibration = load_calibration(calibration_path) if use_calibration else None
    if calibration is not None:
        print(f"using calibration: {calibration_path} {calibration}", flush=True)

    start = time.time()
    written = 0
    prediction_summary_rows: list[dict[str, Any]] = []
    for batch in loader:
        x = batch["x"].cuda(non_blocking=True)
        output = predict_ensemble(models, x, amp=amp, use_flip_tta=use_flip_tta)
        pred = output["pred"]
        if calibration is not None:
            rain_prob_threshold = float(calibration.get("rain_prob_threshold", 0.0))
        else:
            rain_prob_threshold = float(post_cfg.get("rain_prob_threshold", 0.0))
        if rain_prob_threshold > 0 and "rain_prob" in output:
            pred = torch.where(output["rain_prob"] < rain_prob_threshold, torch.zeros_like(pred), pred)
        pred = apply_postprocess(pred, clip_min=clip_min, calibration=calibration)
        pred_np = pred.detach().cpu().numpy().astype(np.float32)
        filenames = batch["gpm_imerg_filename"]
        unique_ids = batch["unique_id"]
        locations = batch.get("name_location", [""] * len(filenames))
        satellites = batch.get("satellite_target", [""] * len(filenames))
        datetimes = batch.get("datetime", [""] * len(filenames))
        for i, filename in enumerate(filenames):
            name = str(filename)
            template = eval_dir / "test_files" / name
            if not template.exists():
                template = sample_submission_dir / "test_files" / name
            array = np.nan_to_num(pred_np[i, 0], nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
            write_float32_like_template(template, prediction_dir / name, array)
            prediction_summary_rows.append(
                {
                    "unique_id": str(unique_ids[i]),
                    "name_location": str(locations[i]),
                    "satellite_target": str(satellites[i]),
                    "datetime": str(datetimes[i]),
                    "gpm_imerg_filename": name,
                    "pred_mean": float(array.mean()),
                    "pred_std": float(array.std()),
                    "pred_min": float(array.min()),
                    "pred_max": float(array.max()),
                    "pred_sum": float(array.sum()),
                    "pred_positive_ratio": float((array > 0).mean()),
                }
            )
            written += 1
        if written % 2048 < len(filenames):
            elapsed = time.time() - start
            print(f"inference {written}/{len(ds)} elapsed={elapsed:.1f}s", flush=True)

    prediction_summary_path = analysis_dir / "evaluation_prediction_summary.csv"
    with prediction_summary_path.open("w", newline="", encoding="utf-8") as f:
        fieldnames = [
            "unique_id",
            "name_location",
            "satellite_target",
            "datetime",
            "gpm_imerg_filename",
            "pred_mean",
            "pred_std",
            "pred_min",
            "pred_max",
            "pred_sum",
            "pred_positive_ratio",
        ]
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(prediction_summary_rows)

    summary = {
        "rows": len(ds),
        "prediction_dir": str(prediction_dir),
        "checkpoints": [str(p) for p in checkpoint_paths],
        "flip_tta": use_flip_tta,
        "calibration": calibration,
        "prediction_summary_csv": str(prediction_summary_path),
        "elapsed_seconds": time.time() - start,
    }
    (output_dir / "inference_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote predictions: {prediction_dir} files={written}", flush=True)


if __name__ == "__main__":
    main()
