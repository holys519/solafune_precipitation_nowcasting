#!/usr/bin/env python3
"""4-source blend optimization: exp016/017/018 + exp035_no_dilation.

no_dilation is a decent solo model (LB 0.68601, beats exp018 solo 0.69295 by 0.0069)
but its real value is as a blend ingredient. Reuses the exp016/017/018 caches from
g_eda/exp003 stage 1 and the exp035_no_dilation cache built via run_blend_curve.py
--cache exp035_no_dilation --module-dir exp035 --checkpoint-dir exp035_no_dilation.

Strategy given time pressure: keep the existing per-satellite 3-way optimum
(exp016/017/018) fixed and sweep a single no_dilation blend-in weight per satellite
(coarser 4-way search would take longer than the submission window allows).
"""

from __future__ import annotations

import csv
import json
import math
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_DIR / "outputs" / "g_eda" / "exp003"
OUT_DIR = CACHE_DIR
EXPERIMENTS = ("exp016", "exp017", "exp018")
NEW_SOURCE = "exp035_no_dilation"
SATELLITES = ("goes", "himawari", "meteosat")


def gaussian_blur(pred: np.ndarray, sigma: float) -> np.ndarray:
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


def tile_rmse(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.sqrt(np.square(pred - target).reshape(pred.shape[0], -1).mean(axis=1))


def main() -> None:
    caches = {exp: np.load(CACHE_DIR / f"{exp}_oof_pred.npz", allow_pickle=False)
              for exp in EXPERIMENTS}
    caches[NEW_SOURCE] = np.load(CACHE_DIR / f"{NEW_SOURCE}_oof_pred.npz", allow_pickle=False)
    order = {exp: np.argsort(caches[exp]["unique_id"]) for exp in caches}
    ref_ids = caches["exp016"]["unique_id"][order["exp016"]]
    for exp in caches:
        ids = caches[exp]["unique_id"][order[exp]]
        if not np.array_equal(ids, ref_ids):
            raise ValueError(f"{exp}: unique_id order mismatch with exp016")
    aligned = {exp: caches[exp]["pred"].astype(np.float32)[order[exp]] for exp in caches}
    target = caches["exp016"]["target"].astype(np.float32)[order["exp016"]]
    satellite = caches["exp016"]["satellite"][order["exp016"]]
    sat_masks = {sat: satellite == sat for sat in SATELLITES}
    n = target.shape[0]
    print(f"aligned {n} tiles across 4 sources", flush=True)

    rec = json.loads((CACHE_DIR / "recommended_weights.json").read_text())

    def score(pred: np.ndarray) -> dict[str, float]:
        per_tile = tile_rmse(pred, target)
        result = {"overall": float(per_tile.mean())}
        for sat, mask in sat_masks.items():
            result[sat] = float(per_tile[mask].mean())
        return result

    # baseline: existing per-satellite 3-way optimum (no no_dilation)
    baseline = np.zeros_like(target)
    for sat, mask in sat_masks.items():
        w = rec["per_satellite_best"][sat]
        baseline[mask] = (w["w016"] * aligned["exp016"][mask]
                          + w["w017"] * aligned["exp017"][mask]
                          + w["w018"] * aligned["exp018"][mask])
    baseline_score = score(baseline)

    # sweep a single no_dilation blend-in weight per satellite
    rows = []
    best_w_nd: dict[str, float] = {}
    for sat, mask in sat_masks.items():
        best = None
        for w_nd in np.round(np.arange(0.0, 0.55, 0.05), 2):
            pred_sat = (1.0 - w_nd) * baseline[mask] + w_nd * aligned[NEW_SOURCE][mask]
            per_tile = tile_rmse(pred_sat, target[mask])
            value = float(per_tile.mean())
            rows.append({"satellite": sat, "w_no_dilation": float(w_nd), "tile_rmse": value})
            if best is None or value < best[1]:
                best = (w_nd, value)
        best_w_nd[sat] = best[0]

    combined = np.zeros_like(target)
    for sat, mask in sat_masks.items():
        w_nd = best_w_nd[sat]
        combined[mask] = (1.0 - w_nd) * baseline[mask] + w_nd * aligned[NEW_SOURCE][mask]
    combined_score = score(combined)

    with (OUT_DIR / "4source_sweep.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["satellite", "w_no_dilation", "tile_rmse"])
        writer.writeheader()
        writer.writerows(rows)

    recommendation = {
        "baseline_3way": baseline_score,
        "per_satellite_no_dilation_weight": best_w_nd,
        "combined_4way": combined_score,
        "delta": combined_score["overall"] - baseline_score["overall"],
    }
    (OUT_DIR / "4source_recommendation.json").write_text(json.dumps(recommendation, indent=2),
                                                         encoding="utf-8")
    lines = ["# 4-source blend (+ exp035_no_dilation) sweep", "",
             f"- 3-way baseline (016/017/018 per-satellite): {baseline_score['overall']:.5f}",
             f"- 4-way with no_dilation blended in: {combined_score['overall']:.5f}",
             f"- delta: {recommendation['delta']:+.5f}",
             f"- per-satellite no_dilation weight: {json.dumps(best_w_nd)}", ""]
    (OUT_DIR / "4SOURCE_BLEND.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
