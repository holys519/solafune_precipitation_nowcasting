#!/usr/bin/env python3
"""H2 practical test: advected temporal smoothing of predictions (g_eda/exp004 stage 3).

g_eda/exp005 measured a real alignment gain between consecutive GPM frames (+0.139 corr,
53% of pairs prefer a nonzero shift), but its RMSE comparison was corrupted by np.roll
wrap-around. The question that matters for serving is different anyway: does shifting the
NEIGHBOR PREDICTION to align with the center prediction before blending beat the static
smoothing we ship now? The shift must be estimable without truth — here it comes from the
predictions themselves (pred-to-pred alignment), which works identically on eval.

Stack under test: per-satellite blend -> [static | advected] 3-tap smoothing
-> blur 1.0 -> threshold 0.2 (the shipped 0.66617 stack, smoothing arm swapped).
Shifted neighbors are blended on the valid overlap only; outside it the center
prediction keeps full weight (no wrap-around).
"""

from __future__ import annotations

import csv
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
SMOOTH = (0.25, 0.30, 0.45)
MAX_SHIFT = 3


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


def align_shift(neighbor: np.ndarray, center: np.ndarray) -> tuple[int, int]:
    """Best integer shift of `neighbor` toward `center`, overlap-only correlation."""
    if neighbor.std() < 1e-6 or center.std() < 1e-6:
        return 0, 0
    best = (-2.0, 0, 0)
    n = neighbor.shape[0]
    for dy in range(-MAX_SHIFT, MAX_SHIFT + 1):
        for dx in range(-MAX_SHIFT, MAX_SHIFT + 1):
            a = neighbor[max(0, -dy):n + min(0, -dy), max(0, -dx):n + min(0, -dx)]
            b = center[max(0, dy):n + min(0, dy), max(0, dx):n + min(0, dx)]
            sa, sb = a.std(), b.std()
            if sa < 1e-6 or sb < 1e-6:
                continue
            corr = float(((a - a.mean()) * (b - b.mean())).mean() / (sa * sb))
            if corr > best[0]:
                best = (corr, dy, dx)
    return best[1], best[2]


def apply_shift_overlap(neighbor: np.ndarray, dy: int, dx: int) -> tuple[np.ndarray, np.ndarray]:
    """Shift `neighbor` by (dy, dx); returns (shifted, valid_mask) without wrap-around."""
    n = neighbor.shape[0]
    shifted = np.zeros_like(neighbor)
    valid = np.zeros_like(neighbor, dtype=bool)
    src = neighbor[max(0, -dy):n + min(0, -dy), max(0, -dx):n + min(0, -dx)]
    shifted[max(0, dy):n + min(0, dy), max(0, dx):n + min(0, dx)] = src
    valid[max(0, dy):n + min(0, dy), max(0, dx):n + min(0, dx)] = True
    return shifted, valid


def main() -> None:
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

    meta = {}
    with TRAIN_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            meta[row["unique_id"]] = (row["name_location"],
                                      datetime.fromisoformat(row["datetime"]))
    index_of = {uid: i for i, uid in enumerate(unique_id)}
    key_of = {meta[uid]: index_of[uid] for uid in unique_id}
    prev_idx = np.full(len(unique_id), -1, dtype=np.int64)
    next_idx = np.full(len(unique_id), -1, dtype=np.int64)
    for uid in unique_id:
        i = index_of[uid]
        location, when = meta[uid]
        prev_idx[i] = key_of.get((location, when - timedelta(minutes=30)), -1)
        next_idx[i] = key_of.get((location, when + timedelta(minutes=30)), -1)

    cw, pw, nw = SMOOTH

    def finish(smoothed: np.ndarray) -> np.ndarray:
        out = gaussian_blur(smoothed, 1.0)
        return np.where(out < 0.2, 0.0, out)

    # arm A: static smoothing (the shipped stack)
    static = np.empty_like(blend)
    for i in range(len(unique_id)):
        weighted = cw * blend[i]
        total = cw
        if prev_idx[i] >= 0:
            weighted = weighted + pw * blend[prev_idx[i]]
            total += pw
        if next_idx[i] >= 0:
            weighted = weighted + nw * blend[next_idx[i]]
            total += nw
        static[i] = weighted / total
    static_score = float(tile_rmse(finish(static), target).mean())

    # arm B: advected smoothing — align each neighbor prediction to the center prediction
    advected = np.empty_like(blend)
    nonzero_shifts = 0
    total_neighbors = 0
    for i in range(len(unique_id)):
        weighted = cw * blend[i]
        total = np.full(blend[i].shape, cw, dtype=np.float32)
        for j, w in ((prev_idx[i], pw), (next_idx[i], nw)):
            if j < 0:
                continue
            total_neighbors += 1
            dy, dx = align_shift(blend[j], blend[i])
            if (dy, dx) != (0, 0):
                nonzero_shifts += 1
            shifted, valid = apply_shift_overlap(blend[j], dy, dx)
            weighted = weighted + np.where(valid, w * shifted, 0.0)
            total = total + np.where(valid, w, 0.0)
        advected[i] = weighted / total
        if (i + 1) % 5000 == 0:
            print(f"advected {i + 1}/{len(unique_id)}", flush=True)
    advected_score = float(tile_rmse(finish(advected), target).mean())

    lines = ["# Advected vs static temporal smoothing (g_eda/exp004 stage 3, H2 practical)", "",
             f"- static smoothing (shipped): {static_score:.5f}",
             f"- advected smoothing (pred-to-pred alignment, overlap-only): {advected_score:.5f}",
             f"- delta: {advected_score - static_score:+.5f}",
             f"- neighbors with nonzero estimated shift: {nonzero_shifts}/{total_neighbors} "
             f"({nonzero_shifts / max(total_neighbors, 1):.1%})", "",
             "Verdict: adopt advected smoothing in exp036 only if the delta beats ~-0.001",
             "(it must survive the OOF->LB attenuation seen on previous post-processing steps)."]
    (OUT_DIR / "ADVECTED_SMOOTHING.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
