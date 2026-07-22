#!/usr/bin/env python3
"""Run GPU inference for exp053 (exp038 strict baseline + autoregressive own-past-prediction
input channel) and write prediction GeoTIFFs.

Supports:
- multi-checkpoint ensembling (average raw predictions across --checkpoint args, e.g. one
  per GroupKFold fold), before the single non-negative clip at the end.
- optional flip TTA (config `tta.enabled`/`tta.flip`): averages predictions over identity,
  horizontal-flip, and vertical-flip views of the input (flipped back before averaging).

When `features.autoregressive_prev_pred` is enabled (exp053's default), the true T-30min GPM
value used as an input feature during training (teacher forcing, see dataset.py) does NOT
exist at evaluation time -- it is exactly what would be predicted for that earlier row too.
`run_autoregressive_inference` below restructures the prediction loop accordingly: evaluation
rows are grouped by `name_location` and processed in ascending-datetime "steps" (step i = the
i-th row of each location's own chronological sequence), threading each row's own just-computed
prediction forward as the AR input feature for that same location's next row (30 min later).
Different locations are independent, so all locations present at a given step are still
batched together for GPU throughput; only the per-location time axis is strictly sequential.
The very first row of each location has no earlier row, so it naturally falls back to the
zero-value/mask=0 convention (see dataset.py's `_autoregressive_prev_pred_channels`).
"""

from __future__ import annotations

import argparse
import csv
import json
import time
from datetime import datetime
from pathlib import Path
from typing import Any

import numpy as np
import torch
import yaml
from torch import nn
from torch.utils.data import DataLoader

from amp_utils import cuda_autocast
from dataset import PrecipDataset, features_from_config, load_norm_stats, read_rows
from model import build_model, prediction_from_output
from tiff_utils import write_float32_like_template

SCRIPT_DIR = Path(__file__).resolve().parent


def build_location_sequences(rows: list[dict[str, str]]) -> dict[str, list[dict[str, str]]]:
    """Group rows by name_location and sort each group ascending by datetime -- the ordering
    the autoregressive chain depends on (each row's AR input is the previous row's own output,
    30 minutes earlier, at the SAME location)."""
    sequences: dict[str, list[dict[str, str]]] = {}
    for row in rows:
        sequences.setdefault(row["name_location"], []).append(row)
    for loc_rows in sequences.values():
        loc_rows.sort(key=lambda r: datetime.fromisoformat(r["datetime"]))
    return sequences


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


def load_calibration(path: Path | None) -> dict[str, Any] | None:
    if path is None or not path.exists():
        return None
    raw = json.loads(path.read_text(encoding="utf-8"))
    calibration: dict[str, Any] = {
        "scale": float(raw.get("scale", 1.0)),
        "bias": float(raw.get("bias", 0.0)),
        "threshold": float(raw.get("threshold", 0.0)),
        "rain_prob_threshold": float(raw.get("rain_prob_threshold", 0.0)),
    }
    isotonic = raw.get("isotonic")
    if isotonic and isotonic.get("x") and isotonic.get("y"):
        x = torch.tensor([float(v) for v in isotonic["x"]], dtype=torch.float32)
        y = torch.tensor([float(v) for v in isotonic["y"]], dtype=torch.float32)
        if x.numel() >= 2:
            calibration["isotonic_x"] = x
            calibration["isotonic_y"] = y
    return calibration


def apply_isotonic_curve(pred: torch.Tensor, x_knots: torch.Tensor, y_knots: torch.Tensor) -> torch.Tensor:
    """Monotonic piecewise-linear interpolation through (x_knots, y_knots), clipped at the ends.

    x_knots must be sorted ascending. Mirrors sklearn IsotonicRegression(out_of_bounds="clip").
    """
    shape = pred.shape
    flat = pred.reshape(-1)
    x_knots = x_knots.to(device=flat.device, dtype=flat.dtype)
    y_knots = y_knots.to(device=flat.device, dtype=flat.dtype)
    idx = torch.searchsorted(x_knots, flat).clamp(1, x_knots.numel() - 1)
    x0 = x_knots[idx - 1]
    x1 = x_knots[idx]
    y0 = y_knots[idx - 1]
    y1 = y_knots[idx]
    t = ((flat - x0) / (x1 - x0).clamp_min(1e-12)).clamp(0.0, 1.0)
    out = y0 + t * (y1 - y0)
    out = torch.where(flat <= x_knots[0], y_knots[0], out)
    out = torch.where(flat >= x_knots[-1], y_knots[-1], out)
    return out.reshape(shape)


def apply_calibration(
    pred: torch.Tensor,
    clip_min: float,
    calibration: dict[str, Any] | None,
    mode: str = "linear",
) -> torch.Tensor:
    if calibration is not None:
        if mode == "isotonic" and "isotonic_x" in calibration:
            pred = apply_isotonic_curve(pred, calibration["isotonic_x"], calibration["isotonic_y"])
        else:
            pred = pred * calibration["scale"] + calibration["bias"]
    return pred.clamp_min(clip_min)


def postprocess_array(array: np.ndarray, threshold: float) -> np.ndarray:
    array = np.nan_to_num(array, nan=0.0, posinf=0.0, neginf=0.0).astype(np.float32)
    if threshold > 0:
        array = np.where(array < threshold, 0.0, array).astype(np.float32)
    return array


def apply_temporal_smoothing(items: list[dict[str, Any]], post_cfg: dict[str, Any]) -> list[dict[str, Any]]:
    smooth_cfg = post_cfg.get("temporal_smoothing", {})
    if not bool(smooth_cfg.get("enabled", False)):
        return items

    center_weight = float(smooth_cfg.get("center_weight", 0.70))
    prev_weight = float(smooth_cfg.get("prev_weight", 0.15))
    next_weight = float(smooth_cfg.get("next_weight", 0.15))
    max_gap_minutes = int(smooth_cfg.get("max_gap_minutes", 30))
    by_location: dict[str, list[int]] = {}
    for idx, item in enumerate(items):
        by_location.setdefault(str(item["name_location"]), []).append(idx)

    smoothed_arrays: list[np.ndarray | None] = [None] * len(items)
    for indices in by_location.values():
        indices = sorted(indices, key=lambda idx: str(items[idx]["datetime"]))
        datetimes = [np.datetime64(str(items[idx]["datetime"]).replace(" ", "T")) for idx in indices]
        for pos, idx in enumerate(indices):
            weighted = items[idx]["array"] * center_weight
            total_weight = center_weight
            if pos > 0:
                gap = (datetimes[pos] - datetimes[pos - 1]) / np.timedelta64(1, "m")
                if 0 < gap <= max_gap_minutes:
                    weighted = weighted + items[indices[pos - 1]]["array"] * prev_weight
                    total_weight += prev_weight
            if pos + 1 < len(indices):
                gap = (datetimes[pos + 1] - datetimes[pos]) / np.timedelta64(1, "m")
                if 0 < gap <= max_gap_minutes:
                    weighted = weighted + items[indices[pos + 1]]["array"] * next_weight
                    total_weight += next_weight
            smoothed_arrays[idx] = (weighted / max(total_weight, 1e-8)).astype(np.float32)

    for idx, array in enumerate(smoothed_arrays):
        if array is not None:
            items[idx]["array"] = array
    return items


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
def run_autoregressive_inference(
    ds: PrecipDataset,
    rows: list[dict[str, str]],
    models: list[nn.Module],
    amp: bool,
    use_flip_tta: bool,
    clip_min: float,
    calibration: dict[str, Any] | None,
    calibration_mode: str,
    rain_prob_threshold: float,
    value_threshold: float,
) -> list[dict[str, Any]]:
    """Sequential, order-dependent prediction loop for `features.autoregressive_prev_pred`.

    Sets `ds.ar_cache` to a fresh dict so the dataset sources the AR input channel ONLY from
    self-generated predictions (never ground truth -- eval rows have none). Processes rows in
    "steps": step i is the i-th row of every location's own chronological sequence, batched
    together across locations (independent) for GPU throughput; sequential only along each
    location's own time axis, which is what the AR dependency actually requires. The array
    cached for the next step is the SAME fully-postprocessed array written to disk for this
    step (calibration + rain_prob threshold + value threshold + non-negative clip already
    applied) -- i.e., each row's AR input is literally "this location's own last prediction",
    matching what a real deployed system would feed forward. The one exception is cross-row
    temporal smoothing (`postprocess.temporal_smoothing`), which is applied by the caller AFTER
    this loop returns (disabled by default for exp053) and is therefore NOT reflected in the
    cached AR values fed to later steps -- documented in README.md as a known limitation.
    """
    sequences = build_location_sequences(rows)
    ar_cache: dict[tuple[str, datetime], np.ndarray] = {}
    ds.ar_cache = ar_cache

    max_len = max((len(seq) for seq in sequences.values()), default=0)
    results: list[dict[str, Any]] = []
    processed = 0
    start = time.time()
    for step in range(max_len):
        step_rows = [seq[step] for seq in sequences.values() if step < len(seq)]
        if not step_rows:
            continue
        x = torch.stack([ds.input_tensor(row) for row in step_rows]).cuda(non_blocking=True)
        output = predict_ensemble(models, x, amp=amp, use_flip_tta=use_flip_tta)
        pred = output["pred"]
        if rain_prob_threshold > 0 and "rain_prob" in output:
            pred = torch.where(output["rain_prob"] < rain_prob_threshold, torch.zeros_like(pred), pred)
        pred = apply_calibration(pred, clip_min=clip_min, calibration=calibration, mode=calibration_mode)
        pred_np = pred.detach().cpu().numpy().astype(np.float32)
        pred_np = postprocess_array(pred_np, threshold=value_threshold)

        for i, row in enumerate(step_rows):
            arr = pred_np[i, 0].copy()
            location = row["name_location"]
            row_time = datetime.fromisoformat(row["datetime"])
            ar_cache[(location, row_time)] = arr
            results.append(
                {
                    "unique_id": str(row["unique_id"]),
                    "name_location": location,
                    "satellite_target": str(row.get("satellite_target", "")),
                    "datetime": str(row["datetime"]),
                    "gpm_imerg_filename": str(row["gpm_imerg_filename"]),
                    "array": arr,
                }
            )
        processed += len(step_rows)
        if step % 20 == 0 or step == max_len - 1:
            elapsed = time.time() - start
            print(
                f"AR inference step={step}/{max_len - 1} rows_done={processed}/{len(rows)} "
                f"elapsed={elapsed:.1f}s",
                flush=True,
            )
    return results


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
    model_dir = resolve_path(config["paths"].get("source_model_dir", config["paths"]["model_dir"]))
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
        context_rows=int(config["data"].get("context_rows", 1)),
        has_target=False,
        norm_stats=norm_stats,
        augment=False,
        features=features_from_config(config),
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
    calibration_mode = str(post_cfg.get("calibration_mode", "linear"))
    if calibration is not None:
        if calibration_mode == "isotonic" and "isotonic_x" not in calibration:
            print("calibration_mode=isotonic requested but oof_calibration.json has no usable "
                  "isotonic curve; falling back to linear scale/bias", flush=True)
            calibration_mode = "linear"
        print(f"using calibration (mode={calibration_mode}): {calibration_path} "
              f"{ {k: v for k, v in calibration.items() if not k.startswith('isotonic')} }", flush=True)

    if calibration is not None:
        rain_prob_threshold = float(calibration.get("rain_prob_threshold", 0.0))
        value_threshold = float(calibration.get("threshold", 0.0))
    else:
        rain_prob_threshold = float(post_cfg.get("rain_prob_threshold", 0.0))
        value_threshold = float(post_cfg.get("value_threshold", post_cfg.get("threshold", 0.0)))

    start = time.time()
    features = features_from_config(config)
    if features.get("autoregressive_prev_pred"):
        # Exposure-bias-affected path (see module docstring): true T-30min GPM is unavailable
        # at eval time, so rows are grouped by name_location, sorted chronologically, and
        # processed step-by-step with each row's own prediction threaded forward as the AR
        # input feature for that location's next row.
        print(
            f"features.autoregressive_prev_pred=true -> sequential per-location inference "
            f"({len(rows)} rows, {len(build_location_sequences(rows))} locations)",
            flush=True,
        )
        prediction_items = run_autoregressive_inference(
            ds,
            rows,
            models,
            amp=amp,
            use_flip_tta=use_flip_tta,
            clip_min=clip_min,
            calibration=calibration,
            calibration_mode=calibration_mode,
            rain_prob_threshold=rain_prob_threshold,
            value_threshold=value_threshold,
        )
    else:
        collected = 0
        prediction_items = []
        for batch in loader:
            x = batch["x"].cuda(non_blocking=True)
            output = predict_ensemble(models, x, amp=amp, use_flip_tta=use_flip_tta)
            pred = output["pred"]
            if rain_prob_threshold > 0 and "rain_prob" in output:
                pred = torch.where(output["rain_prob"] < rain_prob_threshold, torch.zeros_like(pred), pred)
            pred = apply_calibration(pred, clip_min=clip_min, calibration=calibration, mode=calibration_mode)
            pred_np = pred.detach().cpu().numpy().astype(np.float32)
            filenames = batch["gpm_imerg_filename"]
            unique_ids = batch["unique_id"]
            locations = batch.get("name_location", [""] * len(filenames))
            satellites = batch.get("satellite_target", [""] * len(filenames))
            datetimes = batch.get("datetime", [""] * len(filenames))
            for i, filename in enumerate(filenames):
                prediction_items.append(
                    {
                        "unique_id": str(unique_ids[i]),
                        "name_location": str(locations[i]),
                        "satellite_target": str(satellites[i]),
                        "datetime": str(datetimes[i]),
                        "gpm_imerg_filename": str(filename),
                        "array": pred_np[i, 0].copy(),
                    }
                )
                collected += 1
            if collected % 2048 < len(filenames):
                elapsed = time.time() - start
                print(f"inference {collected}/{len(ds)} elapsed={elapsed:.1f}s", flush=True)

    prediction_items = apply_temporal_smoothing(prediction_items, post_cfg)

    prediction_summary_rows: list[dict[str, Any]] = []
    written = 0
    for item in prediction_items:
        name = str(item["gpm_imerg_filename"])
        template = eval_dir / "test_files" / name
        if not template.exists():
            template = sample_submission_dir / "test_files" / name
        array = postprocess_array(item["array"], threshold=value_threshold)
        write_float32_like_template(template, prediction_dir / name, array)
        prediction_summary_rows.append(
            {
                "unique_id": item["unique_id"],
                "name_location": item["name_location"],
                "satellite_target": item["satellite_target"],
                "datetime": item["datetime"],
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
        if written % 5000 == 0:
            print(f"wrote {written}/{len(prediction_items)}", flush=True)

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
        "autoregressive_prev_pred": bool(features.get("autoregressive_prev_pred")),
        "calibration": {k: v for k, v in calibration.items() if not k.startswith("isotonic")} if calibration else None,
        "calibration_mode": calibration_mode,
        "value_threshold": value_threshold,
        "temporal_smoothing": post_cfg.get("temporal_smoothing", {}),
        "prediction_summary_csv": str(prediction_summary_path),
        "elapsed_seconds": time.time() - start,
    }
    (output_dir / "inference_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(f"wrote predictions: {prediction_dir} files={written}", flush=True)


if __name__ == "__main__":
    main()
