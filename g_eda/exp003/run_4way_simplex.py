#!/usr/bin/env python3
"""Refine the coarse exp039 4-source blend (single no_dilation blend-in parameter) into a
proper 4-way per-satellite simplex search: exp016/017/018/exp035_no_dilation weights that
sum to 1, searched jointly rather than fixing the 3-way ratio and sweeping one extra weight.

The earlier run_4source_blend.py approximation found per-satellite no_dilation weights
(goes .30/him .25/met .50) blended into the FIXED 3-way optimum — this fixes the 3-way ratio,
which is suboptimal once a 4th source is added (the fixed ratio was tuned for 3 sources).
This script does a real 4-simplex grid search per satellite (step 0.1, coarser than the
3-way's 0.05 to keep the O(n^3) grid tractable) and reports the gain over exp039's blend.
"""

from __future__ import annotations

import csv
import itertools
import json
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_DIR / "outputs" / "g_eda" / "exp003"
EXPERIMENTS = ("exp016", "exp017", "exp018", "exp035_no_dilation")
SATELLITES = ("goes", "himawari", "meteosat")
STEP = 0.1


def tile_rmse(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.sqrt(np.square(pred - target).reshape(pred.shape[0], -1).mean(axis=1))


def main() -> None:
    caches = {exp: np.load(CACHE_DIR / f"{exp}_oof_pred.npz", allow_pickle=False) for exp in EXPERIMENTS}
    order = {exp: np.argsort(caches[exp]["unique_id"]) for exp in EXPERIMENTS}
    ref_ids = caches["exp016"]["unique_id"][order["exp016"]]
    for exp in EXPERIMENTS:
        ids = caches[exp]["unique_id"][order[exp]]
        if not np.array_equal(ids, ref_ids):
            raise ValueError(f"{exp}: unique_id order mismatch")
    aligned = {exp: caches[exp]["pred"].astype(np.float32)[order[exp]] for exp in EXPERIMENTS}
    target = caches["exp016"]["target"].astype(np.float32)[order["exp016"]]
    satellite = caches["exp016"]["satellite"][order["exp016"]]
    sat_masks = {sat: satellite == sat for sat in SATELLITES}

    steps = int(round(1.0 / STEP))
    grid = []
    for i, j, k in itertools.product(range(steps + 1), repeat=3):
        if i + j + k > steps:
            continue
        w16, w17, w18 = i * STEP, j * STEP, k * STEP
        wnd = 1.0 - w16 - w17 - w18
        grid.append((round(w16, 2), round(w17, 2), round(w18, 2), round(wnd, 2)))
    print(f"grid size: {len(grid)} quadruples", flush=True)

    best_per_sat: dict[str, dict] = {}
    for sat, mask in sat_masks.items():
        sub_target = target[mask]
        sub_preds = {exp: aligned[exp][mask] for exp in EXPERIMENTS}
        best = None
        for w16, w17, w18, wnd in grid:
            pred = (w16 * sub_preds["exp016"] + w17 * sub_preds["exp017"]
                    + w18 * sub_preds["exp018"] + wnd * sub_preds["exp035_no_dilation"])
            value = float(tile_rmse(pred, sub_target).mean())
            if best is None or value < best[1]:
                best = ((w16, w17, w18, wnd), value)
        best_per_sat[sat] = {"weights": best[0], "tile_rmse": best[1]}
        print(f"{sat}: best={best[0]} tile_rmse={best[1]:.5f}", flush=True)

    combined = np.zeros_like(target)
    for sat, mask in sat_masks.items():
        w16, w17, w18, wnd = best_per_sat[sat]["weights"]
        combined[mask] = (w16 * aligned["exp016"][mask] + w17 * aligned["exp017"][mask]
                          + w18 * aligned["exp018"][mask] + wnd * aligned["exp035_no_dilation"][mask])
    overall = {"overall": float(tile_rmse(combined, target).mean())}
    for sat, mask in sat_masks.items():
        overall[sat] = float(tile_rmse(combined[mask], target[mask]).mean())

    # compare to exp039's approximation
    approx = json.loads((CACHE_DIR / "4source_recommendation.json").read_text())
    prior_weights = json.loads((CACHE_DIR / "recommended_weights.json").read_text())["per_satellite_best"]
    approx_pred = np.zeros_like(target)
    for sat, mask in sat_masks.items():
        base = prior_weights[sat]
        wnd = approx["per_satellite_no_dilation_weight"][sat]
        approx_pred[mask] = ((1 - wnd) * (base["w016"] * aligned["exp016"][mask]
                             + base["w017"] * aligned["exp017"][mask]
                             + base["w018"] * aligned["exp018"][mask])
                             + wnd * aligned["exp035_no_dilation"][mask])
    approx_score = float(tile_rmse(approx_pred, target).mean())

    result = {
        "grid_step": STEP,
        "per_satellite_best": best_per_sat,
        "combined_full_simplex": overall,
        "prior_approx_score": approx_score,
        "delta_vs_prior_approx": overall["overall"] - approx_score,
    }
    (CACHE_DIR / "4way_simplex_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# 4-way per-satellite simplex refinement", "",
             f"- prior approximation (fixed 3-way + no_dilation blend-in): {approx_score:.5f}",
             f"- full 4-way simplex (step {STEP}): {overall['overall']:.5f}",
             f"- delta: {result['delta_vs_prior_approx']:+.5f}", "",
             "## Per-satellite optimal quadruples (w016, w017, w018, w_no_dilation)", ""]
    for sat, info in best_per_sat.items():
        lines.append(f"- {sat}: {info['weights']} -> {info['tile_rmse']:.5f}")
    (CACHE_DIR / "4WAY_SIMPLEX.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
