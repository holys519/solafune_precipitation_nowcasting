#!/usr/bin/env python3
"""OOF blend-weight optimization for exp016/exp017/exp018 (Round 5 follow-up to exp033).

LB evidence (2026-07-16): mixing exp018 at w=0.5 into the equal exp016/017 blend improved
public RMSE 0.6746 -> 0.6720 (both after the overlap patch). This script locates the
OOF-optimal mixture instead of spending submissions walking the ladder blindly:

Phase 1 (GPU, once per experiment): regenerate OOF predictions fold-by-fold from the saved
checkpoints and cache them as fp16 npz under outputs/g_eda/exp003/ (reusable by any future
post-processing study).

Phase 2 (CPU): from the caches compute
- the 2-way curve: (1-w) * mean(exp016, exp017) + w * exp018 for w in 0..1 (step 0.05)
- the full 3-way simplex grid (step 0.05): best global triple and best per-satellite triples
- gaussian blur sweep on the best blend (E-1 measured sigma=1 as a small free win)
- value-threshold sweep on the best blend

Outputs: blend_curve.csv, simplex_grid.csv, blur_sweep.csv, threshold_sweep.csv,
BLEND_CURVE.md, and recommended_weights.json (consumed by g_experiments/exp036).

Run one experiment's phase 1 per process (module namespaces collide across exp dirs):
the sbatch wrapper handles this.
"""

from __future__ import annotations

import argparse
import csv
import itertools
import json
import math
import sys
import time
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
OUT_DIR = PROJECT_DIR / "outputs" / "g_eda" / "exp003"
EXPERIMENTS = ("exp016", "exp017", "exp018")
SATELLITES = ("goes", "himawari", "meteosat")


# ---------------------------------------------------------------- phase 1: cache OOF preds

def cache_path(exp_name: str) -> Path:
    return OUT_DIR / f"{exp_name}_oof_pred.npz"


def build_cache(exp_name: str, batch_size: int, num_workers: int) -> None:
    import torch
    from torch.utils.data import DataLoader

    exp_dir = PROJECT_DIR / "g_experiments" / exp_name
    sys.path.insert(0, str(exp_dir))
    import dataset as dataset_mod  # noqa: E402
    import model as model_mod  # noqa: E402

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    checkpoints = sorted((PROJECT_DIR / "g_model" / exp_name).glob("best_model_fold*.pt"))
    if len(checkpoints) != 5:
        raise FileNotFoundError(f"{exp_name}: expected 5 checkpoints, found {len(checkpoints)}")

    preds, targets, unique_ids, satellites, folds = [], [], [], [], []
    for checkpoint_path in checkpoints:
        checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
        config = checkpoint["config"]
        fold = int(checkpoint["fold"])
        model = model_mod.build_model(config).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])
        model.eval()

        rows = dataset_mod.read_rows((exp_dir / config["data"]["train_csv"]).resolve())
        _, valid_rows, _ = dataset_mod.make_group_kfold_split(
            rows, n_splits=int(config["split"]["n_splits"]), fold=fold, seed=int(config["experiment"]["seed"])
        )
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
            norm_stats=dataset_mod.load_norm_stats((exp_dir / config["paths"]["norm_stats"]).resolve()),
            augment=False,
            **ds_kwargs,
        )
        loader = DataLoader(ds, batch_size=batch_size, shuffle=False,
                            num_workers=num_workers, pin_memory=True)
        clip_min = float(config["model"]["clip_min"])
        print(f"{exp_name} fold={fold} rows={len(ds)}", flush=True)
        with torch.no_grad():
            for batch in loader:
                x = batch["x"].to(device, non_blocking=True)
                y = batch["y"].float()
                pred = model_mod.prediction_from_output(model(x)).float().clamp_min(clip_min)
                preds.append(pred.squeeze(1).cpu().numpy().astype(np.float16))
                targets.append(y.squeeze(1).numpy().astype(np.float16))
                unique_ids.extend(batch["unique_id"])
                satellites.extend(batch["satellite_target"])
                folds.extend([fold] * len(batch["unique_id"]))

    OUT_DIR.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        cache_path(exp_name),
        pred=np.concatenate(preds),
        target=np.concatenate(targets),
        unique_id=np.asarray(unique_ids),
        satellite=np.asarray(satellites),
        fold=np.asarray(folds, dtype=np.int8),
    )
    print(f"cached {cache_path(exp_name)}", flush=True)


# ---------------------------------------------------------------- phase 2: curves

def tile_rmse(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.sqrt(np.square(pred - target).reshape(pred.shape[0], -1).mean(axis=1))


def gaussian_blur(pred: np.ndarray, sigma: float) -> np.ndarray:
    """Separable gaussian blur with edge padding, matching E-1's replicate-pad kernel."""
    radius = max(1, int(math.ceil(3.0 * sigma)))
    coords = np.arange(-radius, radius + 1, dtype=np.float32)
    kernel = np.exp(-0.5 * (coords / sigma) ** 2)
    kernel /= kernel.sum()
    padded = np.pad(pred, ((0, 0), (radius, radius), (0, 0)), mode="edge")
    out = np.zeros_like(pred)
    for i, k in enumerate(kernel):
        out += k * padded[:, i : i + pred.shape[1], :]
    padded = np.pad(out, ((0, 0), (0, 0), (radius, radius)), mode="edge")
    out = np.zeros_like(pred)
    for i, k in enumerate(kernel):
        out += k * padded[:, :, i : i + pred.shape[2]]
    return out


def analyze() -> None:
    caches = {}
    for exp in EXPERIMENTS:
        path = cache_path(exp)
        if not path.exists():
            raise FileNotFoundError(f"{path} missing — run phase 1 for {exp} first")
        caches[exp] = np.load(path, allow_pickle=False)

    ref_ids = caches["exp016"]["unique_id"]
    for exp in EXPERIMENTS[1:]:
        if not np.array_equal(np.sort(caches[exp]["unique_id"]), np.sort(ref_ids)):
            raise ValueError(f"{exp}: unique_id set differs from exp016")
    order = {exp: np.argsort(caches[exp]["unique_id"]) for exp in EXPERIMENTS}

    aligned = {exp: caches[exp]["pred"].astype(np.float32)[order[exp]] for exp in EXPERIMENTS}
    target = caches["exp016"]["target"].astype(np.float32)[order["exp016"]]
    satellite = caches["exp016"]["satellite"][order["exp016"]]
    sat_masks = {sat: satellite == sat for sat in SATELLITES}
    n = target.shape[0]
    print(f"aligned {n} tiles", flush=True)

    def score(pred: np.ndarray) -> dict[str, float]:
        per_tile = tile_rmse(pred, target)
        result = {"overall": float(per_tile.mean())}
        for sat, mask in sat_masks.items():
            result[sat] = float(per_tile[mask].mean())
        return result

    # 2-way ladder curve: base = equal(016, 017) — mirrors the exp033 submission ladder
    base = 0.5 * (aligned["exp016"] + aligned["exp017"])
    curve_rows = []
    for w in np.round(np.arange(0.0, 1.0001, 0.05), 2):
        curve_rows.append({"w018": float(w), **score((1.0 - w) * base + w * aligned["exp018"])})
    best_curve = min(curve_rows, key=lambda r: r["overall"])

    # 3-way simplex grid
    grid_rows = []
    step = 0.05
    steps = int(round(1.0 / step))
    for i, j in itertools.product(range(steps + 1), repeat=2):
        if i + j > steps:
            continue
        w16, w17 = i * step, j * step
        w18 = 1.0 - w16 - w17
        pred = w16 * aligned["exp016"] + w17 * aligned["exp017"] + w18 * aligned["exp018"]
        grid_rows.append({"w016": round(w16, 2), "w017": round(w17, 2), "w018": round(w18, 2),
                          **score(pred)})
    best_global = min(grid_rows, key=lambda r: r["overall"])
    best_per_sat = {sat: min(grid_rows, key=lambda r: r[sat]) for sat in SATELLITES}

    # per-satellite composed blend: each satellite uses its own best triple
    composed = np.zeros_like(target)
    for sat, mask in sat_masks.items():
        w = best_per_sat[sat]
        composed[mask] = (w["w016"] * aligned["exp016"][mask]
                          + w["w017"] * aligned["exp017"][mask]
                          + w["w018"] * aligned["exp018"][mask])
    composed_score = score(composed)

    # blur + value-threshold sweeps on the best global blend
    best_pred = (best_global["w016"] * aligned["exp016"]
                 + best_global["w017"] * aligned["exp017"]
                 + best_global["w018"] * aligned["exp018"])
    blur_rows = [{"sigma": 0.0, **score(best_pred)}]
    for sigma in (0.5, 0.75, 1.0, 1.25):
        blur_rows.append({"sigma": sigma, **score(gaussian_blur(best_pred, sigma))})
    best_blur = min(blur_rows, key=lambda r: r["overall"])
    threshold_rows = []
    for threshold in (0.0, 0.05, 0.10, 0.15, 0.20):
        thresholded = np.where(best_pred < threshold, 0.0, best_pred)
        threshold_rows.append({"value_threshold": threshold, **score(thresholded)})
    best_threshold = min(threshold_rows, key=lambda r: r["overall"])

    for name, rows in (("blend_curve", curve_rows), ("simplex_grid", grid_rows),
                       ("blur_sweep", blur_rows), ("threshold_sweep", threshold_rows)):
        with (OUT_DIR / f"{name}.csv").open("w", newline="") as f:
            writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
            writer.writeheader()
            writer.writerows(rows)

    recommendation = {
        "source": "g_eda/exp003 OOF blend optimization",
        "n_tiles": int(n),
        "equal_016_017_baseline": score(base),
        "ladder_best": best_curve,
        "global_best": best_global,
        "per_satellite_best": best_per_sat,
        "per_satellite_composed": composed_score,
        "blur_best_on_global": best_blur,
        "value_threshold_best_on_global": best_threshold,
    }
    (OUT_DIR / "recommended_weights.json").write_text(json.dumps(recommendation, indent=2),
                                                      encoding="utf-8")

    lines = ["# OOF blend optimization (g_eda/exp003)", "",
             f"- equal(016,017) OOF tile_rmse: {score(base)['overall']:.4f}",
             f"- ladder best: w018={best_curve['w018']:.2f} -> {best_curve['overall']:.4f}",
             f"- global best triple: 016={best_global['w016']:.2f} 017={best_global['w017']:.2f} "
             f"018={best_global['w018']:.2f} -> {best_global['overall']:.4f}",
             f"- per-satellite composed: {composed_score['overall']:.4f} "
             f"(goes {composed_score['goes']:.4f} him {composed_score['himawari']:.4f} "
             f"met {composed_score['meteosat']:.4f})",
             f"- blur on global best: sigma={best_blur['sigma']} -> {best_blur['overall']:.4f}",
             f"- value threshold on global best: {best_threshold['value_threshold']} -> "
             f"{best_threshold['overall']:.4f}", "",
             "Full grids in blend_curve.csv / simplex_grid.csv / blur_sweep.csv / "
             "threshold_sweep.csv; exp036 consumes recommended_weights.json."]
    (OUT_DIR / "BLEND_CURVE.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--cache", metavar="EXP", help="phase 1: cache OOF preds for one experiment")
    parser.add_argument("--analyze", action="store_true", help="phase 2: compute curves from caches")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--num-workers", type=int, default=12)
    args = parser.parse_args()

    started = time.time()
    if args.cache:
        build_cache(args.cache, args.batch_size, args.num_workers)
    if args.analyze:
        analyze()
    if not args.cache and not args.analyze:
        raise SystemExit("specify --cache EXP and/or --analyze")
    print(f"done in {time.time() - started:.1f}s", flush=True)


if __name__ == "__main__":
    main()
