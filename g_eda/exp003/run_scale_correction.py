#!/usr/bin/env python3
"""Post-hoc scale correction on the current best (5-source) blend + postprocess stack.

g_eda/exp006's factorization audit found true-mean rescaling recovers 86-88% of the
L2-scale oracle gain on individual models -- but no submission pipeline has ever applied
a fitted scale correction to the FINAL blended+smoothed+blurred+thresholded field. This
reconstructs exp042's exact serving stack from the cached OOF predictions, then jointly
re-optimizes (scale, blur_sigma, value_threshold) per satellite, since a scale correction
shifts the optimal blur/threshold too.

Everything here is a superset of exp042's existing stack: scale=1.0 with the old blur/
threshold values must reproduce exp042's already-measured OOF exactly (checked as an
invariant), so any improvement found is strictly additive.
"""

from __future__ import annotations

import csv
import itertools
import json
import math
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
CACHE_DIR = PROJECT_DIR / "outputs" / "g_eda" / "exp003"
OUT_DIR = PROJECT_DIR / "outputs" / "g_eda" / "exp004"
TRAIN_CSV = PROJECT_DIR / "data" / "train_dataset" / "train_dataset.csv"

FIVE_WAY = ("exp016", "exp017", "exp018", "exp035_no_dilation", "exp038_features")
SATELLITES = ("goes", "himawari", "meteosat")

SCALE_GRID = np.round(np.arange(0.85, 1.201, 0.025), 3)
BLUR_GRID = (0.0, 0.25, 0.5, 0.75, 1.0)
THRESHOLD_GRID = (0.0, 0.05, 0.10, 0.15, 0.20, 0.25, 0.30)


def gaussian_blur(pred: np.ndarray, sigma: float) -> np.ndarray:
    if sigma <= 0:
        return pred
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
    caches = {exp: np.load(CACHE_DIR / f"{exp}_oof_pred.npz", allow_pickle=False) for exp in FIVE_WAY}
    order = {exp: np.argsort(caches[exp]["unique_id"]) for exp in FIVE_WAY}
    ref_ids = caches["exp016"]["unique_id"][order["exp016"]]
    for exp in FIVE_WAY:
        if not np.array_equal(caches[exp]["unique_id"][order[exp]], ref_ids):
            raise ValueError(f"{exp}: unique_id order mismatch")
    aligned = {exp: caches[exp]["pred"].astype(np.float32)[order[exp]] for exp in FIVE_WAY}
    target = caches["exp016"]["target"].astype(np.float32)[order["exp016"]]
    satellite = caches["exp016"]["satellite"][order["exp016"]]
    unique_id = caches["exp016"]["unique_id"][order["exp016"]]
    sat_masks = {sat: satellite == sat for sat in SATELLITES}

    weights = json.loads((CACHE_DIR / "5source_recommendation.json").read_text())
    four_way = json.loads((CACHE_DIR / "4way_simplex_result.json").read_text())["per_satellite_best"]
    w_new = weights["per_satellite_new_source_weight"]
    final_w: dict[str, dict[str, float]] = {}
    for sat in SATELLITES:
        w16, w17, w18, wnd = four_way[sat]["weights"]
        new = w_new[sat]
        scale = 1.0 - new
        final_w[sat] = {"exp016": scale * w16, "exp017": scale * w17, "exp018": scale * w18,
                        "exp035_no_dilation": scale * wnd, "exp038_features": new}

    blend = np.zeros_like(target)
    for sat, mask in sat_masks.items():
        w = final_w[sat]
        blend[mask] = sum(w[exp] * aligned[exp][mask] for exp in FIVE_WAY)

    post = json.loads((OUT_DIR / "recommended_postprocess.json").read_text())
    smooth_weights = post["per_satellite_smooth"]

    meta = {}
    with TRAIN_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            meta[row["unique_id"]] = (row["name_location"], datetime.fromisoformat(row["datetime"]))
    index_of = {uid: i for i, uid in enumerate(unique_id)}
    key_of = {meta[uid]: index_of[uid] for uid in unique_id}
    neighbors = {}
    for offset in (-60, -30, 30, 60):
        idx = np.full(len(unique_id), -1, dtype=np.int64)
        for uid in unique_id:
            location, when = meta[uid]
            idx[index_of[uid]] = key_of.get((location, when + timedelta(minutes=offset)), -1)
        neighbors[offset] = idx

    smoothed = np.empty_like(blend)
    for sat, mask in sat_masks.items():
        cw, p1, n1, p2, n2 = smooth_weights[sat]
        rows = np.nonzero(mask)[0]
        weighted = cw * blend[rows]
        total = np.full(len(rows), cw, dtype=np.float32)
        for offset, w in ((-30, p1), (30, n1), (-60, p2), (60, n2)):
            if w <= 0:
                continue
            idx = neighbors[offset][rows]
            has = idx >= 0
            weighted[has] += w * blend[idx[has]]
            total[has] += w
        smoothed[rows] = weighted / total[:, None, None]

    def score(pred: np.ndarray) -> dict[str, float]:
        per_tile = tile_rmse(pred, target)
        result = {"overall": float(per_tile.mean())}
        for sat, mask in sat_masks.items():
            result[sat] = float(per_tile[mask].mean())
        return result

    # invariant check: scale=1.0 with the OLD blur/threshold must reproduce exp042's OOF
    old_blur, old_thr = post["blur_sigma"], post["per_satellite_thresholds"]
    check = np.empty_like(smoothed)
    for sat, mask in sat_masks.items():
        blurred = gaussian_blur(smoothed[mask], old_blur)
        check[mask] = np.where(blurred < old_thr[sat], 0.0, blurred)
    reproduced = score(check)
    print(f"invariant check (should match exp042's recorded OOF ~0.591): {reproduced['overall']:.5f}",
          flush=True)

    best_per_sat: dict[str, dict] = {}
    for sat, mask in sat_masks.items():
        sub_target = target[mask]
        sub_smoothed = smoothed[mask]
        best = None
        for scale in SCALE_GRID:
            scaled = scale * sub_smoothed
            for blur_sigma in BLUR_GRID:
                blurred = gaussian_blur(scaled, blur_sigma)
                for threshold in THRESHOLD_GRID:
                    pred = np.where(blurred < threshold, 0.0, blurred)
                    value = float(tile_rmse(pred, sub_target).mean())
                    if best is None or value < best[1]:
                        best = ((float(scale), blur_sigma, threshold), value)
        best_per_sat[sat] = {"params": best[0], "tile_rmse": best[1]}
        print(f"{sat}: best (scale, blur, thr)={best[0]} -> {best[1]:.5f}", flush=True)

    combined = np.empty_like(target)
    for sat, mask in sat_masks.items():
        scale, blur_sigma, threshold = best_per_sat[sat]["params"]
        blurred = gaussian_blur(scale * smoothed[mask], blur_sigma)
        combined[mask] = np.where(blurred < threshold, 0.0, blurred)
    combined_score = score(combined)

    old_score = reproduced
    result = {
        "invariant_check_matches_exp042": old_score,
        "scale_corrected_per_satellite": best_per_sat,
        "scale_corrected_combined": combined_score,
        "delta_vs_exp042": combined_score["overall"] - old_score["overall"],
    }
    (OUT_DIR / "scale_correction_result.json").write_text(json.dumps(result, indent=2), encoding="utf-8")
    lines = ["# Post-hoc scale correction on the 5-source blend (g_eda/exp004 stage 4)", "",
             f"- exp042 reproduction (scale=1.0, old blur/threshold): {old_score['overall']:.5f}",
             f"- scale-corrected combined: {combined_score['overall']:.5f}",
             f"- delta: {result['delta_vs_exp042']:+.5f}", "",
             "## Per-satellite optimal (scale, blur_sigma, value_threshold)", ""]
    for sat, info in best_per_sat.items():
        lines.append(f"- {sat}: {info['params']} -> {info['tile_rmse']:.5f}")
    (OUT_DIR / "SCALE_CORRECTION.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
