#!/usr/bin/env python3
"""Temporal-smoothing OOF sweep on the cached blend predictions (g_eda/exp003 caches).

Motivation: GPM's lag-1 (30 min) autocorrelation is 0.561 (l_eda/exp002), and every
inference.py has an `apply_temporal_smoothing` implementation — but it is `enabled: false`
in every config and has NEVER been validated. This sweep measures it on OOF, stacked
exactly like the current best serving (exp036): per-satellite blend -> temporal smoothing
-> blur -> value threshold.

Weight semantics mirror inference.py's apply_temporal_smoothing: neighbors are the same
location at +-30 min; missing neighbors renormalize the remaining weights.

CPU-only; reads outputs/g_eda/exp003/*.npz. Writes smoothing_sweep.csv and
TEMPORAL_SMOOTHING.md under outputs/g_eda/exp004/.
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
RECOMMENDED = CACHE_DIR / "recommended_weights.json"
EXPERIMENTS = ("exp016", "exp017", "exp018")
SATELLITES = ("goes", "himawari", "meteosat")

# (center, prev, next) triples; symmetric ladder plus one-sided probes
WEIGHT_GRID = [
    (1.00, 0.00, 0.00),
    (0.90, 0.05, 0.05),
    (0.80, 0.10, 0.10),
    (0.70, 0.15, 0.15),
    (0.60, 0.20, 0.20),
    (0.50, 0.25, 0.25),
    # first sweep's optimum sat at the 0.50 edge — extend toward stronger smoothing
    (0.45, 0.275, 0.275),
    (0.40, 0.30, 0.30),
    (0.34, 0.33, 0.33),
    (0.30, 0.35, 0.35),
    (0.25, 0.375, 0.375),
    (0.80, 0.20, 0.00),
    (0.80, 0.00, 0.20),
    # one-sided probes showed next >> prev (t+0 information peak) — try asymmetric mixes
    (0.40, 0.20, 0.40),
    (0.35, 0.25, 0.40),
    (0.30, 0.25, 0.45),
    (0.30, 0.30, 0.40),
    (0.25, 0.30, 0.45),
    (0.20, 0.35, 0.45),
    (0.20, 0.30, 0.50),
]
BLUR_SIGMA = 1.0
VALUE_THRESHOLD = 0.2


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

    # per-satellite blend (the exp036 serving weights)
    rec = json.loads(RECOMMENDED.read_text())
    blend = np.zeros_like(target)
    for sat, mask in sat_masks.items():
        w = rec["per_satellite_best"][sat]
        blend[mask] = (w["w016"] * aligned["exp016"][mask]
                       + w["w017"] * aligned["exp017"][mask]
                       + w["w018"] * aligned["exp018"][mask])

    # unique_id -> (location, datetime) from the train CSV, then neighbor indices at +-30 min
    meta: dict[str, tuple[str, datetime]] = {}
    with TRAIN_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            meta[row["unique_id"]] = (row["name_location"],
                                      datetime.fromisoformat(row["datetime"]))
    index_of = {uid: i for i, uid in enumerate(unique_id)}
    key_of = {}
    for uid in unique_id:
        location, when = meta[uid]
        key_of[(location, when)] = index_of[uid]
    prev_idx = np.full(len(unique_id), -1, dtype=np.int64)
    next_idx = np.full(len(unique_id), -1, dtype=np.int64)
    for uid in unique_id:
        i = index_of[uid]
        location, when = meta[uid]
        prev_idx[i] = key_of.get((location, when - timedelta(minutes=30)), -1)
        next_idx[i] = key_of.get((location, when + timedelta(minutes=30)), -1)
    print(f"tiles={len(unique_id)} with prev={int((prev_idx >= 0).sum())} "
          f"next={int((next_idx >= 0).sum())}", flush=True)

    def smooth(pred: np.ndarray, cw: float, pw: float, nw: float) -> np.ndarray:
        has_prev = prev_idx >= 0
        has_next = next_idx >= 0
        weighted = cw * pred
        total = np.full(pred.shape[0], cw, dtype=np.float32)
        weighted[has_prev] += pw * pred[prev_idx[has_prev]]
        total[has_prev] += pw
        weighted[has_next] += nw * pred[next_idx[has_next]]
        total[has_next] += nw
        return weighted / total[:, None, None]

    def score(pred: np.ndarray) -> dict[str, float]:
        per_tile = tile_rmse(pred, target)
        result = {"overall": float(per_tile.mean())}
        for sat, mask in sat_masks.items():
            result[sat] = float(per_tile[mask].mean())
        return result

    rows = []
    for cw, pw, nw in WEIGHT_GRID:
        smoothed = smooth(blend, cw, pw, nw) if (pw or nw) else blend
        combo = gaussian_blur(smoothed, BLUR_SIGMA)
        combo = np.where(combo < VALUE_THRESHOLD, 0.0, combo)
        rows.append({"center": cw, "prev": pw, "next": nw, **score(combo)})
        print(rows[-1], flush=True)
    best = min(rows, key=lambda r: r["overall"])
    baseline = rows[0]

    with (OUT_DIR / "smoothing_sweep.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)
    lines = ["# Temporal smoothing OOF sweep (g_eda/exp004)", "",
             "Stack: per-satellite blend -> smoothing -> blur 1.0 -> threshold 0.2", "",
             f"- baseline (no smoothing): {baseline['overall']:.5f}",
             f"- best: center={best['center']} prev={best['prev']} next={best['next']} -> "
             f"{best['overall']:.5f} (delta {best['overall'] - baseline['overall']:+.5f})", "",
             "Full grid in smoothing_sweep.csv. If the delta beats the E-3 noise band (~0.004),",
             "enable postprocess.temporal_smoothing with these weights in exp036's serving."]
    (OUT_DIR / "TEMPORAL_SMOOTHING.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
