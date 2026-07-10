#!/usr/bin/env python3
"""Create fold logs and OOF diagnostics for exp016."""

from __future__ import annotations

import argparse
import csv
import json
import re
import time
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from sklearn.isotonic import IsotonicRegression
from torch import nn
from torch.utils.data import DataLoader

from amp_utils import cuda_autocast
from dataset import PrecipDataset, load_norm_stats, make_group_kfold_split, read_rows
from model import build_model, prediction_from_output


SCRIPT_DIR = Path(__file__).resolve().parent
FOLD_RE = re.compile(r"best_model_fold(\d+)\.pt$")


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


def checkpoint_fold(path: Path) -> int:
    match = FOLD_RE.search(path.name)
    if not match:
        raise ValueError(f"Cannot parse fold from checkpoint name: {path}")
    return int(match.group(1))


def load_model(config: dict[str, Any], path: Path) -> nn.Module:
    checkpoint = torch.load(path, map_location="cpu", weights_only=False)
    model = build_model(config)
    model.load_state_dict(checkpoint["model_state_dict"])
    model = model.cuda().eval()
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    print(f"loaded fold={checkpoint.get('fold')} {path} best_rmse={checkpoint.get('best_rmse')}", flush=True)
    return model


def metric_key(config: dict[str, Any]) -> str:
    return str(config.get("metric", {}).get("selection", "tile_rmse"))


@torch.no_grad()
def predict_with_tta(model: nn.Module, x: torch.Tensor, amp: bool, use_flip_tta: bool) -> dict[str, torch.Tensor]:
    views: list[tuple[torch.Tensor, tuple[int, ...] | None]] = [(x, None)]
    if use_flip_tta:
        views.append((torch.flip(x, dims=(-1,)), (-1,)))
        views.append((torch.flip(x, dims=(-2,)), (-2,)))

    total: dict[str, torch.Tensor] = {}
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
    return {key: value / len(views) for key, value in total.items()}


def empty_stats() -> dict[str, float]:
    return {
        "samples": 0.0,
        "pixels": 0.0,
        "sse": 0.0,
        "tile_rmse_sum": 0.0,
        "sae": 0.0,
        "bias_sum": 0.0,
        "target_sum": 0.0,
        "pred_sum": 0.0,
        "target_positive": 0.0,
        "pred_positive": 0.0,
    }


def empty_detection_stats() -> dict[str, float]:
    return {"tp": 0.0, "fp": 0.0, "fn": 0.0, "tn": 0.0}


def isotonic_bin_edges() -> np.ndarray:
    """Fixed pixel-value bins for the isotonic calibration histogram.

    Fine (0.05) resolution near zero where most mass sits, coarser mid-range, log-spaced tail
    up to 200 to cover the heaviest observed GPM values (target_max up to ~96 in exp009 OOF).
    """
    low = np.linspace(0.0, 2.0, 41)
    mid = np.linspace(2.0, 20.0, 37)[1:]
    high = np.geomspace(20.0, 200.0, 21)[1:]
    return np.concatenate([low, mid, high]).astype(np.float64)


def empty_isotonic_hist(n_bins: int) -> dict[str, np.ndarray]:
    return {
        "count": np.zeros(n_bins, dtype=np.float64),
        "sum_pred": np.zeros(n_bins, dtype=np.float64),
        "sum_target": np.zeros(n_bins, dtype=np.float64),
    }


def update_isotonic_hist(
    hist: dict[str, np.ndarray], edges: np.ndarray, flat_pred: np.ndarray, flat_target: np.ndarray
) -> None:
    n_bins = len(edges) - 1
    bin_idx = np.clip(np.digitize(flat_pred, edges) - 1, 0, n_bins - 1)
    hist["count"] += np.bincount(bin_idx, minlength=n_bins).astype(np.float64)
    hist["sum_pred"] += np.bincount(bin_idx, weights=flat_pred, minlength=n_bins)
    hist["sum_target"] += np.bincount(bin_idx, weights=flat_target, minlength=n_bins)


def tile_rmse_of(pred: np.ndarray, target: np.ndarray) -> float:
    """Mean over samples of sqrt(mean((pred-target)^2)) per (H, W) tile -- the official metric."""
    diff = pred.astype(np.float64) - target.astype(np.float64)
    return float(np.sqrt(np.square(diff).reshape(diff.shape[0], -1).mean(axis=1)).mean())


def compare_calibration_effect(
    all_pred: np.ndarray,
    all_target: np.ndarray,
    scale: float,
    bias: float,
    clip_min: float,
    isotonic_calibration: dict[str, Any] | None,
) -> dict[str, Any]:
    """OOF tile_rmse under no calibration vs. linear vs. isotonic, on the same cached OOF tiles.

    Pure numpy/CPU -- reuses the (pred, target) tiles already produced by the OOF forward pass,
    so this costs no extra GPU time. Answers "does calibration actually help tile_rmse" directly,
    since the histogram-based fit above only minimizes a pixel-pooled objective, not tile_rmse.
    """
    result: dict[str, Any] = {
        "n_samples": int(all_pred.shape[0]),
        "raw_tile_rmse": tile_rmse_of(all_pred, all_target),
    }
    linear_pred = np.clip(all_pred * scale + bias, clip_min, None)
    result["linear_tile_rmse"] = tile_rmse_of(linear_pred, all_target)
    if isotonic_calibration is not None:
        x = np.asarray(isotonic_calibration["x"], dtype=np.float64)
        y = np.asarray(isotonic_calibration["y"], dtype=np.float64)
        isotonic_pred = np.interp(all_pred.astype(np.float64), x, y)
        result["isotonic_tile_rmse"] = tile_rmse_of(isotonic_pred, all_target)
    else:
        result["isotonic_tile_rmse"] = None
    return result


def fit_isotonic_calibration(hist: dict[str, np.ndarray], min_bins: int = 2) -> dict[str, Any] | None:
    """Fit a monotonic (isotonic) pred->target calibration curve from a binned pixel histogram.

    Bins are collapsed to (mean_pred, mean_target) points weighted by pixel count before fitting,
    which keeps this cheap (dozens of points) regardless of how many OOF pixels were seen, and is
    consistent with the existing linear scale/bias fit also being computed from pooled sums rather
    than per-tile-weighted statistics.
    """
    count = hist["count"]
    valid = count > 0
    if int(valid.sum()) < min_bins:
        return None
    bin_pred = hist["sum_pred"][valid] / count[valid]
    bin_target = hist["sum_target"][valid] / count[valid]
    bin_weight = count[valid]

    model = IsotonicRegression(y_min=0.0, increasing=True, out_of_bounds="clip")
    model.fit(bin_pred, bin_target, sample_weight=bin_weight)
    x = np.asarray(model.X_thresholds_, dtype=np.float64)
    y = np.asarray(model.y_thresholds_, dtype=np.float64)
    if x.size < 2:
        return None
    return {
        "x": x.tolist(),
        "y": y.tolist(),
        "n_bins_used": int(valid.sum()),
        "n_pixels": float(count.sum()),
    }


def update_detection_stats(stats: dict[str, float], pred_rain: np.ndarray, target_rain: np.ndarray) -> None:
    stats["tp"] += float(np.logical_and(pred_rain, target_rain).sum())
    stats["fp"] += float(np.logical_and(pred_rain, np.logical_not(target_rain)).sum())
    stats["fn"] += float(np.logical_and(np.logical_not(pred_rain), target_rain).sum())
    stats["tn"] += float(np.logical_and(np.logical_not(pred_rain), np.logical_not(target_rain)).sum())


def detection_row(
    threshold: float,
    stats: dict[str, float],
    sse: float,
    pixels: float,
    tile_rmse_sum: float,
    samples: float,
) -> dict[str, float]:
    tp = stats["tp"]
    fp = stats["fp"]
    fn = stats["fn"]
    tn = stats["tn"]
    precision = tp / (tp + fp) if tp + fp > 0 else 0.0
    recall = tp / (tp + fn) if tp + fn > 0 else 0.0
    csi = tp / (tp + fp + fn) if tp + fp + fn > 0 else 0.0
    false_alarm_ratio = fp / (tp + fp) if tp + fp > 0 else 0.0
    return {
        "rain_prob_threshold": float(threshold),
        "rmse": float(np.sqrt(sse / max(pixels, 1.0))),
        "tile_rmse": float(tile_rmse_sum / max(samples, 1.0)),
        "precision": float(precision),
        "recall": float(recall),
        "csi": float(csi),
        "false_alarm_ratio": float(false_alarm_ratio),
        "tp": float(tp),
        "fp": float(fp),
        "fn": float(fn),
        "tn": float(tn),
    }


def threshold_row(threshold: float, stats: dict[str, float]) -> dict[str, float]:
    row = stats_row("value_threshold", str(threshold), stats)
    return {
        "value_threshold": float(threshold),
        "samples": float(row["samples"]),
        "pixels": float(row["pixels"]),
        "rmse": float(row["rmse"]),
        "tile_rmse": float(row["tile_rmse"]),
        "mae": float(row["mae"]),
        "bias": float(row["bias"]),
        "target_mean": float(row["target_mean"]),
        "pred_mean": float(row["pred_mean"]),
        "target_positive_ratio": float(row["target_positive_ratio"]),
        "pred_positive_ratio": float(row["pred_positive_ratio"]),
    }


def update_batch_stats(stats: dict[str, float], pred: np.ndarray, target: np.ndarray) -> None:
    diff = pred - target
    batch = int(target.shape[0])
    pixels = float(target.size)
    stats["samples"] += float(batch)
    stats["pixels"] += pixels
    stats["sse"] += float(np.square(diff).sum())
    stats["tile_rmse_sum"] += float(np.sqrt(np.square(diff).reshape(batch, -1).mean(axis=1)).sum())
    stats["sae"] += float(np.abs(diff).sum())
    stats["bias_sum"] += float(diff.sum())
    stats["target_sum"] += float(target.sum())
    stats["pred_sum"] += float(pred.sum())
    stats["target_positive"] += float((target > 0).sum())
    stats["pred_positive"] += float((pred > 0).sum())


def update_stats(stats: dict[str, float], pred: np.ndarray, target: np.ndarray) -> None:
    diff = pred - target
    stats["samples"] += 1
    stats["pixels"] += float(target.size)
    stats["sse"] += float(np.square(diff).sum())
    stats["tile_rmse_sum"] += float(np.sqrt(np.square(diff).mean()))
    stats["sae"] += float(np.abs(diff).sum())
    stats["bias_sum"] += float(diff.sum())
    stats["target_sum"] += float(target.sum())
    stats["pred_sum"] += float(pred.sum())
    stats["target_positive"] += float((target > 0).sum())
    stats["pred_positive"] += float((pred > 0).sum())


def stats_row(group_type: str, group_value: str, stats: dict[str, float]) -> dict[str, Any]:
    pixels = max(stats["pixels"], 1.0)
    return {
        "group_type": group_type,
        "group_value": group_value,
        "samples": int(stats["samples"]),
        "pixels": int(stats["pixels"]),
        "rmse": float(np.sqrt(stats["sse"] / pixels)),
        "tile_rmse": float(stats["tile_rmse_sum"] / max(stats["samples"], 1.0)),
        "mae": float(stats["sae"] / pixels),
        "bias": float(stats["bias_sum"] / pixels),
        "target_mean": float(stats["target_sum"] / pixels),
        "pred_mean": float(stats["pred_sum"] / pixels),
        "target_positive_ratio": float(stats["target_positive"] / pixels),
        "pred_positive_ratio": float(stats["pred_positive"] / pixels),
    }


def summarize_training(config: dict[str, Any], analysis_dir: Path) -> dict[str, Any]:
    model_dir = resolve_path(config["paths"].get("source_model_dir", config["paths"]["model_dir"]))
    metric_paths = sorted(model_dir.glob("metrics_fold*.json"))
    epoch_rows: list[dict[str, Any]] = []
    fold_rows: list[dict[str, Any]] = []
    selection_metric = metric_key(config)

    for path in metric_paths:
        data = json.loads(path.read_text(encoding="utf-8"))
        fold = int(data["fold"])
        history = data["history"]
        best_record = min(history, key=lambda row: row.get(selection_metric, row["rmse"]))
        fold_rows.append(
            {
                "fold": fold,
                "best_epoch": best_record["epoch"],
                "best_rmse": data.get("best_rmse", best_record["rmse"]),
                "best_tile_rmse": data.get("best_tile_rmse", best_record.get("tile_rmse")),
                "best_metric": data.get("best_metric", best_record.get(selection_metric, best_record["rmse"])),
                "selection_metric": data.get("selection_metric", selection_metric),
                "best_positive_rmse": best_record["positive_rmse"],
                "best_zero_rmse": best_record["zero_rmse"],
                "train_rows_used": data["train_rows_used"],
                "valid_rows_used": data["valid_rows_used"],
                "valid_locations": "|".join(data["valid_locations"]),
            }
        )
        for row in history:
            epoch_rows.append(
                {
                    "fold": fold,
                    "epoch": row["epoch"],
                    "train_rmse": row["train_rmse"],
                    "train_tile_rmse": row.get("train_tile_rmse"),
                    "valid_rmse": row["rmse"],
                    "valid_tile_rmse": row.get("tile_rmse"),
                    "zero_rmse": row["zero_rmse"],
                    "positive_rmse": row["positive_rmse"],
                    "samples": row.get("samples"),
                    "pixels": row["pixels"],
                    "positive_pixels": row["positive_pixels"],
                    "elapsed_seconds": row["elapsed_seconds"],
                }
            )

    if epoch_rows:
        with (analysis_dir / "epoch_history.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(epoch_rows[0]))
            writer.writeheader()
            writer.writerows(epoch_rows)
    if fold_rows:
        with (analysis_dir / "fold_summary.csv").open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(fold_rows[0]))
            writer.writeheader()
            writer.writerows(sorted(fold_rows, key=lambda row: row["fold"]))

    best_values = [float(row["best_metric"]) for row in fold_rows]
    return {
        "folds": len(fold_rows),
        "selection_metric": selection_metric,
        "best_metric_mean": float(np.mean(best_values)) if best_values else None,
        "best_metric_std": float(np.std(best_values)) if best_values else None,
    }


@torch.no_grad()
def analyze_oof(config: dict[str, Any], checkpoint_paths: list[Path], analysis_dir: Path) -> dict[str, Any]:
    train_csv = resolve_path(config["data"]["train_csv"])
    train_dir = resolve_path(config["data"]["train_dir"])
    norm_stats = load_norm_stats(resolve_path(config["paths"]["norm_stats"]))
    rows = read_rows(train_csv)
    n_splits = int(config["split"]["n_splits"])
    target_size = (int(config["data"]["target_height"]), int(config["data"]["target_width"]))
    batch_size = int(config["train"]["batch_size"])
    num_workers = int(config["train"]["num_workers"])
    clip_min = float(config["model"]["clip_min"])
    amp = bool(config["train"]["amp"])
    use_flip_tta = bool(config["tta"]["enabled"]) and bool(config["tta"].get("flip", True))
    post_cfg = config.get("postprocess", {})
    threshold_grid = [float(x) for x in post_cfg.get("rain_prob_threshold_grid", [])]
    if not threshold_grid:
        threshold_grid = [round(x, 2) for x in np.linspace(0.05, 0.95, 19)]
    value_threshold_grid = [float(x) for x in post_cfg.get("value_threshold_grid", [])]
    if not value_threshold_grid:
        value_threshold_grid = [0.0, 0.03, 0.05, 0.08, 0.10, 0.12, 0.15, 0.20]
    selection_metric = metric_key(config)

    sample_rows: list[dict[str, Any]] = []
    grouped: dict[tuple[str, str], dict[str, float]] = {}
    global_stats = empty_stats()
    threshold_stats = {threshold: empty_detection_stats() for threshold in threshold_grid}
    threshold_sse = {threshold: 0.0 for threshold in threshold_grid}
    threshold_pixels = {threshold: 0.0 for threshold in threshold_grid}
    threshold_tile_rmse_sum = {threshold: 0.0 for threshold in threshold_grid}
    threshold_samples = {threshold: 0.0 for threshold in threshold_grid}
    value_threshold_stats = {threshold: empty_stats() for threshold in value_threshold_grid}
    calibration_sums = {
        "n": 0.0,
        "sum_pred": 0.0,
        "sum_target": 0.0,
        "sum_pred2": 0.0,
        "sum_pred_target": 0.0,
    }
    isotonic_edges = isotonic_bin_edges()
    isotonic_hist = empty_isotonic_hist(len(isotonic_edges) - 1)
    # Cached per-batch (pred, target) tiles so we can score calibration_comparison (raw vs.
    # linear vs. isotonic OOF tile_rmse) in a cheap CPU-only second pass after fitting, without
    # a second GPU forward pass. ~68M pixels * 4 bytes * 2 arrays =~ 550MB for the full OOF
    # set -- fine for a single analysis process.
    cached_pred_batches: list[np.ndarray] = []
    cached_target_batches: list[np.ndarray] = []

    for checkpoint_path in checkpoint_paths:
        fold = checkpoint_fold(checkpoint_path)
        _, valid_rows, valid_locations = make_group_kfold_split(
            rows,
            n_splits=n_splits,
            fold=fold,
            seed=int(config["experiment"]["seed"]),
        )
        print(f"OOF fold={fold} rows={len(valid_rows)} locations={valid_locations}", flush=True)
        ds = PrecipDataset(
            valid_rows,
            train_dir,
            max_observations=int(config["data"]["max_observations"]),
            satellite_channels=int(config["data"]["satellite_channels"]),
            target_size=target_size,
            context_rows=int(config["data"].get("context_rows", 1)),
            has_target=True,
            norm_stats=norm_stats,
            augment=False,
        )
        loader = DataLoader(
            ds,
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=True,
            persistent_workers=num_workers > 0,
            worker_init_fn=worker_init_fn,
        )
        model = load_model(config, checkpoint_path)

        for batch in loader:
            x = batch["x"].cuda(non_blocking=True)
            y = batch["y"].cuda(non_blocking=True)
            output = predict_with_tta(model, x, amp=amp, use_flip_tta=use_flip_tta)
            pred = output["pred"].clamp_min(clip_min)
            pred_np = pred.detach().cpu().numpy().astype(np.float32)
            target_np = y.detach().cpu().numpy().astype(np.float32)

            if "rain_prob" in output:
                rain_prob_np = output["rain_prob"].detach().cpu().numpy().astype(np.float32)
                target_rain_np = target_np > float(config.get("loss", {}).get("rain_threshold", 0.0))
                for threshold in threshold_grid:
                    threshold_pred_np = np.where(rain_prob_np < threshold, 0.0, pred_np)
                    threshold_sse[threshold] += float(np.square(threshold_pred_np - target_np).sum())
                    threshold_pixels[threshold] += float(target_np.size)
                    threshold_tile_rmse_sum[threshold] += float(
                        np.sqrt(
                            np.square(threshold_pred_np - target_np)
                            .reshape(target_np.shape[0], -1)
                            .mean(axis=1)
                        ).sum()
                    )
                    threshold_samples[threshold] += float(target_np.shape[0])
                    update_detection_stats(threshold_stats[threshold], rain_prob_np >= threshold, target_rain_np)

            for threshold in value_threshold_grid:
                threshold_pred_np = np.where(pred_np < threshold, 0.0, pred_np)
                update_batch_stats(value_threshold_stats[threshold], threshold_pred_np, target_np)

            flat_pred = pred_np.reshape(-1)
            flat_target = target_np.reshape(-1)
            calibration_sums["n"] += float(flat_pred.size)
            calibration_sums["sum_pred"] += float(flat_pred.sum())
            calibration_sums["sum_target"] += float(flat_target.sum())
            calibration_sums["sum_pred2"] += float(np.square(flat_pred).sum())
            calibration_sums["sum_pred_target"] += float((flat_pred * flat_target).sum())
            update_isotonic_hist(isotonic_hist, isotonic_edges, flat_pred, flat_target)
            cached_pred_batches.append(pred_np[:, 0].copy())
            cached_target_batches.append(target_np[:, 0].copy())

            for i, unique_id in enumerate(batch["unique_id"]):
                pred_arr = pred_np[i, 0]
                target_arr = target_np[i, 0]
                diff = pred_arr - target_arr
                location = str(batch["name_location"][i])
                satellite = str(batch["satellite_target"][i])
                update_stats(global_stats, pred_arr, target_arr)
                for key in [
                    ("fold", str(fold)),
                    ("location", location),
                    ("satellite", satellite),
                    ("fold_location", f"{fold}:{location}"),
                    ("fold_satellite", f"{fold}:{satellite}"),
                ]:
                    grouped.setdefault(key, empty_stats())
                    update_stats(grouped[key], pred_arr, target_arr)

                sample_rows.append(
                    {
                        "fold": fold,
                        "unique_id": str(unique_id),
                        "name_location": location,
                        "satellite_target": satellite,
                        "datetime": str(batch["datetime"][i]),
                        "gpm_imerg_filename": str(batch["gpm_imerg_filename"][i]),
                        "rmse": float(np.sqrt(np.square(diff).mean())),
                        "tile_rmse": float(np.sqrt(np.square(diff).mean())),
                        "mae": float(np.abs(diff).mean()),
                        "bias": float(diff.mean()),
                        "target_mean": float(target_arr.mean()),
                        "pred_mean": float(pred_arr.mean()),
                        "target_max": float(target_arr.max()),
                        "pred_max": float(pred_arr.max()),
                        "target_positive_ratio": float((target_arr > 0).mean()),
                        "pred_positive_ratio": float((pred_arr > 0).mean()),
                    }
                )

        del model
        torch.cuda.empty_cache()

    sample_path = analysis_dir / "oof_sample_metrics.csv"
    if sample_rows:
        with sample_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(sample_rows[0]))
            writer.writeheader()
            writer.writerows(sample_rows)

    group_rows = [stats_row(group_type, group_value, stats) for (group_type, group_value), stats in grouped.items()]
    group_path = analysis_dir / "oof_group_metrics.csv"
    if group_rows:
        with group_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(group_rows[0]))
            writer.writeheader()
            writer.writerows(sorted(group_rows, key=lambda row: (row["group_type"], row["group_value"])))

    threshold_rows = [
        detection_row(
            threshold,
            threshold_stats[threshold],
            threshold_sse[threshold],
            threshold_pixels[threshold],
            threshold_tile_rmse_sum[threshold],
            threshold_samples[threshold],
        )
        for threshold in threshold_grid
        if threshold_pixels[threshold] > 0
    ]
    threshold_path = analysis_dir / "oof_rain_threshold_sweep.csv"
    best_threshold_row = None
    if threshold_rows:
        best_threshold_row = min(threshold_rows, key=lambda row: row.get(selection_metric, row["rmse"]))
        with threshold_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(threshold_rows[0]))
            writer.writeheader()
            writer.writerows(threshold_rows)

    value_threshold_rows = [
        threshold_row(threshold, value_threshold_stats[threshold])
        for threshold in value_threshold_grid
        if value_threshold_stats[threshold]["samples"] > 0
    ]
    value_threshold_path = analysis_dir / "oof_value_threshold_sweep.csv"
    best_value_threshold_row = None
    if value_threshold_rows:
        best_value_threshold_row = min(value_threshold_rows, key=lambda row: row.get(selection_metric, row["rmse"]))
        with value_threshold_path.open("w", newline="", encoding="utf-8") as f:
            writer = csv.DictWriter(f, fieldnames=list(value_threshold_rows[0]))
            writer.writeheader()
            writer.writerows(value_threshold_rows)

    n = calibration_sums["n"]
    sum_pred = calibration_sums["sum_pred"]
    sum_target = calibration_sums["sum_target"]
    sum_pred2 = calibration_sums["sum_pred2"]
    sum_pred_target = calibration_sums["sum_pred_target"]
    scale = sum_pred_target / sum_pred2 if sum_pred2 > 0 else 1.0
    denom = n * sum_pred2 - sum_pred * sum_pred
    if abs(denom) > 1e-12:
        scale_with_bias = (n * sum_pred_target - sum_pred * sum_target) / denom
        bias = (sum_target - scale_with_bias * sum_pred) / n
    else:
        scale_with_bias = scale
        bias = 0.0
    isotonic_calibration = fit_isotonic_calibration(isotonic_hist)
    calibration = {
        "scale": float(scale_with_bias),
        "bias": float(bias),
        "scale_no_bias": float(scale),
        "threshold": float(best_value_threshold_row["value_threshold"]) if best_value_threshold_row else 0.0,
        "rain_prob_threshold": float(best_threshold_row["rain_prob_threshold"]) if best_threshold_row else 0.0,
        "selection_metric": selection_metric,
        "source": "exp016_oof",
        "isotonic": isotonic_calibration,
    }
    calibration_path = analysis_dir / "oof_calibration.json"
    calibration_path.write_text(json.dumps(calibration, indent=2), encoding="utf-8")

    all_pred = np.concatenate(cached_pred_batches, axis=0) if cached_pred_batches else np.zeros((0,))
    all_target = np.concatenate(cached_target_batches, axis=0) if cached_target_batches else np.zeros((0,))
    calibration_comparison = compare_calibration_effect(
        all_pred, all_target, scale=scale_with_bias, bias=bias, clip_min=clip_min,
        isotonic_calibration=isotonic_calibration,
    )
    print(f"calibration_comparison (OOF tile_rmse, lower is better): {calibration_comparison}", flush=True)

    global_row = stats_row("global", "oof", global_stats)
    summary = {
        "oof_global": global_row,
        "checkpoint_count": len(checkpoint_paths),
        "selection_metric": selection_metric,
        "oof_official_metric": global_row.get("tile_rmse"),
        "sample_metrics_csv": str(sample_path),
        "group_metrics_csv": str(group_path),
        "rain_threshold_sweep_csv": str(threshold_path) if threshold_rows else None,
        "value_threshold_sweep_csv": str(value_threshold_path) if value_threshold_rows else None,
        "calibration_json": str(calibration_path),
        "calibration": calibration,
        "calibration_comparison": calibration_comparison,
        "best_rain_threshold": best_threshold_row,
        "best_value_threshold": best_value_threshold_row,
    }
    return summary


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", default=str(SCRIPT_DIR / "config.yaml"))
    parser.add_argument("--checkpoint", action="append", default=None)
    args = parser.parse_args()

    start = time.time()
    config = load_config(Path(args.config))
    analysis_dir = resolve_path(config["paths"].get("analysis_dir", "../../outputs/analysis/exp016"))
    analysis_dir.mkdir(parents=True, exist_ok=True)
    model_dir = resolve_path(config["paths"].get("source_model_dir", config["paths"]["model_dir"]))
    checkpoint_paths = [Path(p) for p in args.checkpoint] if args.checkpoint else sorted(model_dir.glob("best_model_fold*.pt"))
    checkpoint_paths = sorted(checkpoint_paths, key=checkpoint_fold)
    if not checkpoint_paths:
        raise FileNotFoundError(f"No checkpoints found under {model_dir}")

    training_summary = summarize_training(config, analysis_dir)
    oof_summary = analyze_oof(config, checkpoint_paths, analysis_dir)
    summary = {
        "training": training_summary,
        **oof_summary,
        "elapsed_seconds": time.time() - start,
    }
    summary_path = analysis_dir / "analysis_summary.json"
    summary_path.write_text(json.dumps(summary, indent=2, ensure_ascii=False), encoding="utf-8")
    print(json.dumps(summary, indent=2, ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
