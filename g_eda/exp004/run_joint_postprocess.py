#!/usr/bin/env python3
"""Joint post-processing re-optimization on the cached OOF blend (g_eda/exp004 stage 2).

Temporal smoothing landed −0.0045 on the LB (better than its −0.0038 OOF estimate), so the
serving stack changed and every other knob deserves re-tuning around it:

1. 5-tap temporal window (±60 min): does a second neighbor pair add over the ±30 min taps?
2. per-satellite smoothing weights (meteosat is 88.5% zero — its optimum may differ)
3. blur re-check AFTER smoothing (smoothing already averages; blur may now be redundant)
4. per-satellite value thresholds (currently one global 0.2)
5. GPM 0.01 quantization snap (targets are 99.6% multiples of 0.01 — l_eda/exp002 G-025)

Outputs JOINT_POSTPROCESS.md + recommended_postprocess.json under outputs/g_eda/exp004/.
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
EXPERIMENTS = ("exp016", "exp017", "exp018")
SATELLITES = ("goes", "himawari", "meteosat")

# (center, prev1, next1, prev2, next2); first row = the currently served setting
SMOOTH_GRID = [
    (0.25, 0.30, 0.45, 0.00, 0.00),
    (0.20, 0.30, 0.50, 0.00, 0.00),
    (0.22, 0.26, 0.40, 0.05, 0.07),
    (0.20, 0.24, 0.36, 0.08, 0.12),
    (0.18, 0.22, 0.32, 0.11, 0.17),
    (0.16, 0.20, 0.30, 0.13, 0.21),
    (0.14, 0.18, 0.26, 0.16, 0.26),
]
BLUR_GRID = (0.0, 0.5, 0.75, 1.0, 1.25)
THRESHOLD_GRID = (0.0, 0.10, 0.15, 0.20, 0.25, 0.30)


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
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    caches = {exp: np.load(CACHE_DIR / f"{exp}_oof_pred.npz", allow_pickle=False)
              for exp in EXPERIMENTS}
    order = {exp: np.argsort(caches[exp]["unique_id"]) for exp in EXPERIMENTS}
    aligned = {exp: caches[exp]["pred"].astype(np.float32)[order[exp]] for exp in EXPERIMENTS}
    target = caches["exp016"]["target"].astype(np.float32)[order["exp016"]]
    satellite = caches["exp016"]["satellite"][order["exp016"]]
    unique_id = caches["exp016"]["unique_id"][order["exp016"]]
    sat_masks = {sat: satellite == sat for sat in SATELLITES}

    rec = json.loads((CACHE_DIR / "recommended_weights.json").read_text())
    blend = np.zeros_like(target)
    for sat, mask in sat_masks.items():
        w = rec["per_satellite_best"][sat]
        blend[mask] = (w["w016"] * aligned["exp016"][mask]
                       + w["w017"] * aligned["exp017"][mask]
                       + w["w018"] * aligned["exp018"][mask])

    meta: dict[str, tuple[str, datetime]] = {}
    with TRAIN_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            meta[row["unique_id"]] = (row["name_location"],
                                      datetime.fromisoformat(row["datetime"]))
    index_of = {uid: i for i, uid in enumerate(unique_id)}
    key_of = {(meta[uid][0], meta[uid][1]): index_of[uid] for uid in unique_id}
    neighbor = {}
    for offset in (-60, -30, 30, 60):
        idx = np.full(len(unique_id), -1, dtype=np.int64)
        for uid in unique_id:
            location, when = meta[uid]
            idx[index_of[uid]] = key_of.get((location, when + timedelta(minutes=offset)), -1)
        neighbor[offset] = idx

    def smooth(pred: np.ndarray, weights: tuple[float, ...],
               mask: np.ndarray | None = None) -> np.ndarray:
        cw, p1, n1, p2, n2 = weights
        rows = np.arange(pred.shape[0]) if mask is None else np.nonzero(mask)[0]
        weighted = cw * pred[rows]
        total = np.full(len(rows), cw, dtype=np.float32)
        for offset, w in ((-30, p1), (30, n1), (-60, p2), (60, n2)):
            if w <= 0:
                continue
            idx = neighbor[offset][rows]
            has = idx >= 0
            weighted[has] += w * pred[idx[has]]
            total[has] += w
        out = pred.copy() if mask is not None else np.empty_like(pred)
        out[rows] = weighted / total[:, None, None]
        return out

    def score(pred: np.ndarray) -> dict[str, float]:
        per_tile = tile_rmse(pred, target)
        result = {"overall": float(per_tile.mean())}
        for sat, mask in sat_masks.items():
            result[sat] = float(per_tile[mask].mean())
        return result

    report: list[str] = ["# Joint post-processing re-optimization (g_eda/exp004 stage 2)", ""]

    # --- 1+2: smoothing grid, globally and per satellite (scored pre-blur/threshold)
    smooth_scores = []
    for weights in SMOOTH_GRID:
        smooth_scores.append({"weights": weights, **score(smooth(blend, weights))})
    best_global_smooth = min(smooth_scores, key=lambda r: r["overall"])
    per_sat_smooth = {sat: min(smooth_scores, key=lambda r: r[sat])["weights"]
                      for sat in SATELLITES}
    smoothed = np.empty_like(blend)
    for sat, mask in sat_masks.items():
        smoothed[mask] = smooth(blend, per_sat_smooth[sat], mask=mask)[mask]
    report += ["## Smoothing (pre blur/threshold)",
               f"- served (0.25,0.30,0.45,±30only): {smooth_scores[0]['overall']:.5f}",
               f"- best global 5-tap: {best_global_smooth['weights']} -> "
               f"{best_global_smooth['overall']:.5f}",
               f"- per-satellite 5-tap: {json.dumps({s: list(w) for s, w in per_sat_smooth.items()})} -> "
               f"{score(smoothed)['overall']:.5f}", ""]

    # --- 3+4: blur (global) x per-satellite thresholds on the per-satellite-smoothed field
    best = None
    for sigma in BLUR_GRID:
        blurred = smoothed if sigma == 0.0 else gaussian_blur(smoothed, sigma)
        per_tile_by_thr = {thr: tile_rmse(np.where(blurred < thr, 0.0, blurred), target)
                           for thr in THRESHOLD_GRID}
        sat_thr = {}
        total = 0.0
        for sat, mask in sat_masks.items():
            best_thr = min(THRESHOLD_GRID, key=lambda thr: per_tile_by_thr[thr][mask].mean())
            sat_thr[sat] = best_thr
            total += float(per_tile_by_thr[best_thr][mask].mean()) * mask.mean()
        candidate = {"sigma": sigma, "thresholds": sat_thr, "overall": total}
        if best is None or candidate["overall"] < best["overall"]:
            best = candidate
    report += ["## Blur x per-satellite thresholds (after smoothing)",
               f"- best: sigma={best['sigma']} thresholds={best['thresholds']} -> "
               f"{best['overall']:.5f}", ""]

    # --- 5: quantization snap on the final field
    final = smoothed if best["sigma"] == 0.0 else gaussian_blur(smoothed, best["sigma"])
    for sat, mask in sat_masks.items():
        thr = best["thresholds"][sat]
        if thr > 0:
            final[mask] = np.where(final[mask] < thr, 0.0, final[mask])
    snapped = np.round(final / 0.01) * 0.01
    report += ["## Quantization snap (0.01 grid)",
               f"- before: {score(final)['overall']:.5f}",
               f"- after:  {score(snapped)['overall']:.5f}", ""]

    recommendation = {
        "per_satellite_smooth": {sat: list(w) for sat, w in per_sat_smooth.items()},
        "blur_sigma": best["sigma"],
        "per_satellite_thresholds": best["thresholds"],
        "snap_delta": score(snapped)["overall"] - score(final)["overall"],
        "final_overall": score(final)["overall"],
        "served_baseline": None,
    }
    # served baseline: smoothing (0.25,0.30,0.45) -> blur 1.0 -> thr 0.2 (today's LB 0.66617)
    served = smooth(blend, SMOOTH_GRID[0])
    served = gaussian_blur(served, 1.0)
    served = np.where(served < 0.2, 0.0, served)
    recommendation["served_baseline"] = score(served)["overall"]
    report += ["## Summary",
               f"- served stack OOF: {recommendation['served_baseline']:.5f}",
               f"- re-optimized stack OOF: {recommendation['final_overall']:.5f} "
               f"(delta {recommendation['final_overall'] - recommendation['served_baseline']:+.5f})",
               f"- snap adds: {recommendation['snap_delta']:+.6f}"]

    (OUT_DIR / "recommended_postprocess.json").write_text(json.dumps(recommendation, indent=2),
                                                          encoding="utf-8")
    (OUT_DIR / "JOINT_POSTPROCESS.md").write_text("\n".join(report), encoding="utf-8")
    print("\n".join(report), flush=True)


if __name__ == "__main__":
    main()
