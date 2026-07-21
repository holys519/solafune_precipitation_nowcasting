#!/usr/bin/env python3
"""Causal-only temporal smoothing OOF sweep for exp038 (green, context_rows=1) -- the model
exp046 already applies UNTUNED weights (center=0.85, prev=0.15) to on top of, confirmed on
LB to help (-0.00025 vs exp038 solo, 2026-07-20). This finds better (center, prev) weights
now that next_weight is fixed at 0 (non-causal smoothing banned by the 2026-07-20 ruling).

Adapted from g_eda/exp004/run_temporal_smoothing.py (which swept center/prev/next on the now
-red multi-source blend); this version uses exp038's own single-model OOF cache
(outputs/g_eda/exp003/exp038_oof_pred.npz) and only ever looks at the PREVIOUS row (30 min
back), never next.

CPU-only. Writes causal_smoothing_sweep.csv and CAUSAL_SMOOTHING.md under
outputs/g_eda/exp009/.
"""

from __future__ import annotations

import csv
from datetime import datetime, timedelta
from pathlib import Path

import numpy as np

PROJECT_DIR = Path(__file__).resolve().parents[2]
CACHE_PATH = PROJECT_DIR / "outputs" / "g_eda" / "exp003" / "exp038_oof_pred.npz"
OUT_DIR = PROJECT_DIR / "outputs" / "g_eda" / "exp009"
TRAIN_CSV = PROJECT_DIR / "data" / "train_dataset" / "train_dataset.csv"
SATELLITES = ("goes", "himawari", "meteosat")

# (center, prev) pairs; center+prev always sums to 1.0 here (no missing-neighbor
# renormalization ambiguity needed since we score only the achievable weight, matching
# inference.py's renormalize-when-missing behavior for rows with no prev neighbor).
WEIGHT_GRID = [
    (1.00, 0.00),
    (0.95, 0.05),
    (0.90, 0.10),
    (0.85, 0.15),  # exp046's current (untuned, carried over from the old bidirectional split)
    (0.80, 0.20),
    (0.75, 0.25),
    (0.70, 0.30),
    (0.65, 0.35),
    (0.60, 0.40),
    (0.55, 0.45),
    (0.50, 0.50),
]


def tile_rmse(pred: np.ndarray, target: np.ndarray) -> np.ndarray:
    return np.sqrt(np.square(pred - target).reshape(pred.shape[0], -1).mean(axis=1))


def main() -> None:
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    cache = np.load(CACHE_PATH, allow_pickle=False)
    order = np.argsort(cache["unique_id"])
    pred = cache["pred"].astype(np.float32)[order]
    target = cache["target"].astype(np.float32)[order]
    satellite = cache["satellite"][order]
    unique_id = cache["unique_id"][order]
    sat_masks = {sat: satellite == sat for sat in SATELLITES}

    meta: dict[str, tuple[str, datetime]] = {}
    with TRAIN_CSV.open(newline="") as f:
        for row in csv.DictReader(f):
            meta[row["unique_id"]] = (row["name_location"], datetime.fromisoformat(row["datetime"]))
    index_of = {uid: i for i, uid in enumerate(unique_id)}
    key_of = {}
    for uid in unique_id:
        location, when = meta[uid]
        key_of[(location, when)] = index_of[uid]
    prev_idx = np.full(len(unique_id), -1, dtype=np.int64)
    for uid in unique_id:
        i = index_of[uid]
        location, when = meta[uid]
        prev_idx[i] = key_of.get((location, when - timedelta(minutes=30)), -1)
    has_prev = prev_idx >= 0
    print(f"tiles={len(unique_id)} with causal prev neighbor={int(has_prev.sum())}", flush=True)

    def smooth(cw: float, pw: float) -> np.ndarray:
        weighted = cw * pred
        total = np.full(pred.shape[0], cw, dtype=np.float32)
        weighted[has_prev] += pw * pred[prev_idx[has_prev]]
        total[has_prev] += pw
        return weighted / total[:, None, None]

    def score(smoothed: np.ndarray) -> dict[str, float]:
        per_tile = tile_rmse(smoothed, target)
        result = {"overall": float(per_tile.mean())}
        for sat, mask in sat_masks.items():
            result[sat] = float(per_tile[mask].mean())
        return result

    rows = []
    for cw, pw in WEIGHT_GRID:
        smoothed = smooth(cw, pw) if pw else pred
        rows.append({"center": cw, "prev": pw, **score(smoothed)})
        print(rows[-1], flush=True)
    baseline = rows[0]
    best = min(rows, key=lambda r: r["overall"])

    with (OUT_DIR / "causal_smoothing_sweep.csv").open("w", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        writer.writeheader()
        writer.writerows(rows)

    lines = [
        "# Causal-only temporal smoothing OOF sweep (g_eda/exp009, exp038 base)", "",
        f"- baseline (no smoothing): {baseline['overall']:.5f}",
        f"- exp046 current (center=0.85, prev=0.15): "
        f"{next(r['overall'] for r in rows if r['center'] == 0.85):.5f}",
        f"- best in grid: center={best['center']} prev={best['prev']} -> {best['overall']:.5f} "
        f"(delta vs baseline {best['overall'] - baseline['overall']:+.5f})", "",
        "Full grid in causal_smoothing_sweep.csv. If best beats exp046's current weights by",
        "more than the E-3 noise band (~0.004-0.005 on this smaller single-model OOF; treat",
        "cautiously), rebuild exp046 with the new (center, prev) and re-submit.",
    ]
    (OUT_DIR / "CAUSAL_SMOOTHING.md").write_text("\n".join(lines), encoding="utf-8")
    print("\n".join(lines), flush=True)


if __name__ == "__main__":
    main()
