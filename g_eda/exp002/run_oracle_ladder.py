#!/usr/bin/env python3
"""E-1 (Round 5 plan): oracle-decomposition ladder on our own OOF predictions.

For one experiment (its code imported from --exp-dir, checkpoints from g_model/<exp>),
regenerates OOF predictions fold by fold and scores counterfactual predictors per tile:

- actual            rmse(pred, truth)                      — what we actually score
- flat_pred         rmse(mean(pred), truth)                — our amount information only
- flat_truth        rmse(mean(truth), truth)               — the 0.677 "wall" (oracle amount, flat)
- mask_oracle_flat  wet-mean placed on the TRUE wet mask   — wall + perfect mask rung
- amount_swap       pred rescaled to the true tile mean    — perfect amount, OUR placement
- mask_swap         our tile total flat on the TRUE mask   — our amount, perfect placement
- blur_pred_s{1,2}  gaussian-blurred pred                  — is our field over-sharp?

Comparing amount_swap vs mask_swap decomposes the remaining error into placement vs amount.
Run one experiment per process (module names collide across experiment dirs).
"""

from __future__ import annotations

import argparse
import csv
import json
import math
import sys
import time
from pathlib import Path

import torch
from torch.utils.data import DataLoader

PROJECT_DIR = Path(__file__).resolve().parents[2]


def gaussian_kernel(sigma: float, device: torch.device) -> torch.Tensor:
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coords = torch.arange(-radius, radius + 1, dtype=torch.float32, device=device)
    kernel_1d = torch.exp(-0.5 * (coords / sigma) ** 2)
    kernel_1d = kernel_1d / kernel_1d.sum()
    kernel_2d = torch.outer(kernel_1d, kernel_1d)
    return kernel_2d.view(1, 1, *kernel_2d.shape)


def blur(pred: torch.Tensor, kernel: torch.Tensor) -> torch.Tensor:
    pad = kernel.shape[-1] // 2
    return torch.nn.functional.conv2d(
        torch.nn.functional.pad(pred, (pad, pad, pad, pad), mode="replicate"), kernel
    )


def tile_rmse(pred: torch.Tensor, truth: torch.Tensor) -> torch.Tensor:
    return torch.sqrt(torch.square(pred - truth).flatten(1).mean(dim=1))


def ladder_metrics(
    pred: torch.Tensor, truth: torch.Tensor, kernels: dict[str, torch.Tensor]
) -> dict[str, torch.Tensor]:
    b = truth.shape[0]
    flat_dims = (1, 2, 3)
    truth_mean = truth.mean(dim=flat_dims, keepdim=True)
    pred_mean = pred.mean(dim=flat_dims, keepdim=True)
    wet_mask = (truth > 0).float()
    n_wet = wet_mask.sum(dim=flat_dims, keepdim=True)
    n_pixels = float(truth.shape[-1] * truth.shape[-2])

    metrics = {
        "actual": tile_rmse(pred, truth),
        "flat_pred": tile_rmse(pred_mean.expand_as(truth), truth),
        "flat_truth": tile_rmse(truth_mean.expand_as(truth), truth),
    }
    # wall + perfect wet/dry mask: the true wet-pixel mean placed only on the true wet pixels
    wet_mean = torch.where(n_wet > 0, truth.sum(dim=flat_dims, keepdim=True) / n_wet.clamp_min(1.0), torch.zeros_like(n_wet))
    metrics["mask_oracle_flat"] = tile_rmse(wet_mask * wet_mean, truth)
    # perfect tile amount, our spatial pattern (dry-pred tiles fall back to the flat oracle)
    scale = torch.where(pred_mean > 1e-6, truth_mean / pred_mean.clamp_min(1e-6), torch.zeros_like(pred_mean))
    amount_swap = torch.where(pred_mean > 1e-6, pred * scale, truth_mean.expand_as(truth))
    metrics["amount_swap"] = tile_rmse(amount_swap, truth)
    # our tile total, placed flat on the TRUE mask
    our_total = pred.sum(dim=flat_dims, keepdim=True)
    mask_swap = wet_mask * torch.where(n_wet > 0, our_total / n_wet.clamp_min(1.0), torch.zeros_like(n_wet))
    metrics["mask_swap"] = tile_rmse(mask_swap, truth)
    for name, kernel in kernels.items():
        metrics[name] = tile_rmse(blur(pred, kernel), truth)
    metrics["wet_tile"] = (n_wet.flatten(1).squeeze(1) > 0).float()
    assert all(v.shape[0] == b for v in metrics.values())
    return metrics


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--exp-dir", required=True, help="e.g. ../../g_experiments/exp018")
    parser.add_argument("--out-dir", default=str(PROJECT_DIR / "outputs" / "g_eda" / "exp002"))
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=12)
    parser.add_argument("--blur-sigmas", type=float, nargs="*", default=[1.0, 2.0])
    args = parser.parse_args()

    exp_dir = Path(args.exp_dir).resolve()
    exp_name = exp_dir.name
    sys.path.insert(0, str(exp_dir))
    import dataset as dataset_mod  # noqa: E402
    import model as model_mod  # noqa: E402

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_dir = PROJECT_DIR / "g_model" / exp_name
    checkpoints = sorted(model_dir.glob("best_model_fold*.pt"))
    if not checkpoints:
        raise FileNotFoundError(f"no checkpoints under {model_dir}")

    kernels = {f"blur_pred_s{sigma:g}": gaussian_kernel(sigma, device) for sigma in args.blur_sigmas}

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    # accumulators keyed by (group_type, group_value, metric)
    sums: dict[tuple[str, str, str], float] = {}
    counts: dict[tuple[str, str], int] = {}

    started = time.time()
    metric_names: list[str] = []
    for checkpoint_path in checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        fold = int(checkpoint["fold"])
        model = model_mod.build_model(config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        train_csv = (exp_dir / config["data"]["train_csv"]).resolve()
        rows = dataset_mod.read_rows(train_csv)
        _, valid_rows, valid_locations = dataset_mod.make_group_kfold_split(
            rows, n_splits=int(config["split"]["n_splits"]), fold=fold, seed=int(config["experiment"]["seed"])
        )
        norm_stats = dataset_mod.load_norm_stats((exp_dir / config["paths"]["norm_stats"]).resolve())
        ds_kwargs = {}
        if hasattr(dataset_mod, "features_from_config"):
            ds_kwargs["features"] = dataset_mod.features_from_config(config)
        ds = dataset_mod.PrecipDataset(
            valid_rows,
            (exp_dir / config["data"]["train_dir"]).resolve(),
            max_observations=int(config["data"]["max_observations"]),
            satellite_channels=int(config["data"]["satellite_channels"]),
            target_size=(int(config["data"]["target_height"]), int(config["data"]["target_width"])),
            context_rows=int(config["data"].get("context_rows", 1)),
            has_target=True,
            norm_stats=norm_stats,
            augment=False,
            **ds_kwargs,
        )
        loader = DataLoader(ds, batch_size=args.batch_size, shuffle=False,
                            num_workers=args.num_workers, pin_memory=True)
        clip_min = float(config["model"]["clip_min"])
        print(f"{exp_name} fold={fold} rows={len(ds)} locations={valid_locations}", flush=True)

        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(device, non_blocking=True)
                y = batch["y"].to(device, non_blocking=True).float()
                pred = model_mod.prediction_from_output(model(x)).float().clamp_min(clip_min)
                metrics = ladder_metrics(pred, y, kernels)
                wet = metrics.pop("wet_tile")
                if not metric_names:
                    metric_names = list(metrics.keys())
                satellites = batch["satellite_target"]
                for name, values in metrics.items():
                    values = values.detach().cpu()
                    for i in range(values.shape[0]):
                        value = float(values[i])
                        groups = [("global", "all"), ("fold", str(fold)), ("satellite", satellites[i])]
                        if float(wet[i]) > 0:
                            groups.append(("global_wet", "all"))
                            groups.append(("satellite_wet", satellites[i]))
                        for group in groups:
                            sums[(*group, name)] = sums.get((*group, name), 0.0) + value
                for i in range(len(satellites)):
                    groups = [("global", "all"), ("fold", str(fold)), ("satellite", satellites[i])]
                    if float(wet[i]) > 0:
                        groups.append(("global_wet", "all"))
                        groups.append(("satellite_wet", satellites[i]))
                    for group in groups:
                        counts[group] = counts.get(group, 0) + 1

    group_rows = []
    for (group_type, group_value), n in sorted(counts.items()):
        row: dict[str, object] = {"group_type": group_type, "group_value": group_value, "samples": n}
        for name in metric_names:
            row[name] = sums[(group_type, group_value, name)] / n
        group_rows.append(row)

    csv_path = out_dir / f"{exp_name}_oracle_ladder.csv"
    with csv_path.open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["group_type", "group_value", "samples", *metric_names])
        writer.writeheader()
        writer.writerows(group_rows)

    summary = {
        "experiment": exp_name,
        "checkpoints": [str(p) for p in checkpoints],
        "elapsed_seconds": time.time() - started,
        "groups": group_rows,
    }
    json_path = out_dir / f"{exp_name}_oracle_ladder.json"
    json_path.write_text(json.dumps(summary, indent=2), encoding="utf-8")
    print(json.dumps([r for r in group_rows if r["group_type"] in ("global", "global_wet")], indent=2))
    print(f"wrote {csv_path} and {json_path}", flush=True)


if __name__ == "__main__":
    main()
