#!/usr/bin/env python3
"""5-source blend optimization: exp016/017/018/exp035_no_dilation (fixed 4-way weights from
4way_simplex_result.json) + exp038_features (current-row + wavelength-aligned physics,
architecturally distinct from the successor-row family).

Same coarse blend-in approach as run_4source_blend.py: fix the existing 4-way per-satellite
optimum and sweep a single exp038_features weight per satellite (a full 5-way simplex would
be a much larger grid; the blend-in approximation was already shown to be within 0.00008 of
the true 4-way optimum, so it is a reliable shortcut here too).
"""

from __future__ import annotations

import csv
import json
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_DIR / "outputs" / "g_eda" / "exp003"
FOUR_WAY = ("exp016", "exp017", "exp018", "exp035_no_dilation")
NEW_SOURCE = "exp038_features"
SATELLITES = ("goes", "himawari", "meteosat")


def tile_rmse(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.sqrt(np.square(pred - target).reshape(pred.shape[0], -1).mean(axis=1))


def main() -> None:
    caches = {exp: np.load(CACHE_DIR / f"{exp}_oof_pred.npz", allow_pickle=False)
              for exp in (*FOUR_WAY, NEW_SOURCE)}
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
    print(f"aligned {n} tiles across 5 sources", flush=True)

    four_way = json.loads((CACHE_DIR / "4way_simplex_result.json").read_text())["per_satellite_best"]

    def score(pred: np.ndarray) -> dict[str, float]:
        per_tile = tile_rmse(pred, target)
        result = {"overall": float(per_tile.mean())}
        for sat, mask in sat_masks.items():
            result[sat] = float(per_tile[mask].mean())
        return result

    baseline = np.zeros_like(target)
    for sat, mask in sat_masks.items():
        w16, w17, w18, wnd = four_way[sat]["weights"]
        baseline[mask] = (w16 * aligned["exp016"][mask] + w17 * aligned["exp017"][mask]
                          + w18 * aligned["exp018"][mask] + wnd * aligned["exp035_no_dilation"][mask])
    baseline_score = score(baseline)

    rows = []
    best_w_new: dict[str, float] = {}
    for sat, mask in sat_masks.items():
        best = None
        for w_new in np.round(np.arange(0.0, 0.65, 0.05), 2):
            pred_sat = (1.0 - w_new) * baseline[mask] + w_new * aligned[NEW_SOURCE][mask]
            value = float(tile_rmse(pred_sat, target[mask]).mean())
            rows.append({"satellite": sat, "w_new": float(w_new), "tile_rmse": value})
            if best is None or value < best[1]:
                best = (w_new, value)
        best_w_new[sat] = best[0]

    combined = np.zeros_like(target)
    for sat, mask in sat_masks.items():
        w_new = best_w_new[sat]
        combined[mask] = (1.0 - w_new) * baseline[mask] + w_new * aligned[NEW_SOURCE][mask]
    combined_score = score(combined)

    with (CACHE_DIR / "5source_sweep.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=["satellite", "w_new", "tile_rmse"])
        writer.writeheader()
        writer.writerows(rows)

    recommendation = {
        "baseline_4way": baseline_score,
        "per_satellite_new_source_weight": best_w_new,
        "combined_5way": combined_score,
        "delta": combined_score["overall"] - baseline_score["overall"],
        "four_way_weights": {sat: four_way[sat]["weights"] for sat in SATELLITES},
    }
    (CACHE_DIR / "5source_recommendation.json").write_text(json.dumps(recommendation, indent=2),
                                                           encoding="utf-8")
    lines = ["# 5-source blend (+ exp038_features) sweep", "",
             f"- 4-way baseline: {baseline_score['overall']:.5f}",
             f"- 5-way with exp038_features blended in: {combined_score['overall']:.5f}",
             f"- delta: {recommendation['delta']:+.5f}",
             f"- per-satellite exp038_features weight: {json.dumps(best_w_new)}", ""]
    (CACHE_DIR / "5SOURCE_BLEND.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
